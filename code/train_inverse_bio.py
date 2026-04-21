import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.data import Batch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy, policy_load_incompatibilities
from src.inverse.family_index_builder import build_family_index_artifacts
from src.inverse.phase4_il import (
    build_stage_plan,
    compute_phase4_losses,
    ensure_multistep_expert_paths,
    evaluate_multistep_reconstruction,
    filter_paths_by_families,
    load_step_split,
    subset_by_indices,
)
from src.inverse.action_codebook import codebook_tensor, load_action_codebook
from src.inverse.pretrain_links import (
    ensure_links_pretrain_cache,
    load_links_pretrained_weights,
    run_links_pretraining,
)

TRACKED_METRICS = (
    "objective",
    "total",
    "loss_action_u",
    "loss_action_v",
    "loss_action_w",
    "loss_action_code",
    "loss_stop",
    "loss_step_role",
    "loss_step_count",
    "stop_accuracy",
    "step_role_accuracy",
    "step_count_accuracy",
    "action_u_accuracy",
    "action_v_accuracy",
    "action_w_accuracy",
    "action_code_accuracy",
    "step_action_accuracy",
)


def _init_metric_sums():
    return {key: 0.0 for key in TRACKED_METRICS}


def _accumulate_metric_sums(sums, metrics):
    for key in TRACKED_METRICS:
        sums[key] += float(metrics[key].item())


def _average_metric_sums(sums, num_batches):
    denom = max(num_batches, 1)
    return {key: value / denom for key, value in sums.items()}


def _format_metrics(metrics):
    return "  ".join(
        [
            f"obj={metrics['objective']:.6f}",
            f"code_acc={metrics['action_code_accuracy']:.4f}",
            f"stop_acc={metrics['stop_accuracy']:.4f}",
            f"step_acc={metrics['step_action_accuracy']:.4f}",
            f"role_acc={metrics['step_role_accuracy']:.4f}",
        ]
    )


def _to_float_dict(metrics):
    return {key: float(value) for key, value in metrics.items()}


class PreBatchedLoader:
    def __init__(self, dataset, batch_size: int, device, shuffle: bool = True):
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.batches = []

        print(f"  Pre-batching {len(dataset)} multistep IL samples to {device} (batch_size={batch_size}) ...")
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            chunk = [dataset[indices[i]] for i in range(start, end)]
            if not chunk:
                continue
            batch = {
                "base_data": Batch.from_data_list([item["base_data"] for item in chunk]).to(device),
                "action_topo": torch.stack([item["action_topo"] for item in chunk]).to(device),
                "y_foot": torch.stack([item["y_foot"] for item in chunk]).to(device),
                "y_knee": torch.stack([item["y_knee"] for item in chunk]).to(device),
                "y_ankle": torch.stack([item["y_ankle"] for item in chunk]).to(device),
                "family_index": torch.tensor([int(item["family_index"]) for item in chunk], dtype=torch.long, device=device),
                "step_index": torch.tensor([int(item["step_index"]) for item in chunk], dtype=torch.long, device=device),
                "step_count": torch.tensor([int(item["step_count"]) for item in chunk], dtype=torch.long, device=device),
                "step_role_index": torch.tensor(
                    [int(item["step_role_index"]) for item in chunk],
                    dtype=torch.long,
                    device=device,
                ),
                "stop_token": torch.tensor(
                    [float(item["stop_token"]) for item in chunk],
                    dtype=torch.float32,
                    device=device,
                ),
                "action_code_id": torch.tensor(
                    [int(item["action_code_id"]) for item in chunk],
                    dtype=torch.long,
                    device=device,
                ),
            }
            self.batches.append(batch)

        print(f"  Pre-batching done: {len(self.batches)} batches cached on {device}")

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _forward_phase4_batch(policy, curve_encoder, batch, cfg):
    z_c = curve_encoder(batch["y_foot"], batch["y_knee"], batch["y_ankle"])
    return compute_phase4_losses(
        policy,
        batch,
        z_c,
        cfg,
    )


def train_epoch_prebatched(policy, curve_encoder, optimizer, loader, cfg, all_params):
    policy.train()
    curve_encoder.train()
    metric_sums = _init_metric_sums()
    for batch in loader:
        metrics = _forward_phase4_batch(policy, curve_encoder, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        metrics["objective"].backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        _accumulate_metric_sums(metric_sums, metrics)
    return _average_metric_sums(metric_sums, len(loader))


def eval_epoch_prebatched(policy, curve_encoder, loader, cfg):
    policy.eval()
    curve_encoder.eval()
    metric_sums = _init_metric_sums()
    with torch.no_grad():
        for batch in loader:
            metrics = _forward_phase4_batch(policy, curve_encoder, batch, cfg)
            _accumulate_metric_sums(metric_sums, metrics)
    return _average_metric_sums(metric_sums, len(loader))


def _save_plot(history_by_stage, plot_path: str):
    if not history_by_stage:
        return
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for stage_name, stage_history in history_by_stage.items():
        plt.plot(stage_history["train_objective"], label=f"{stage_name} train")
        plt.plot(stage_history["val_objective"], linestyle="--", label=f"{stage_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title("Phase4 IL Objective")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    for stage_name, stage_history in history_by_stage.items():
        plt.plot(stage_history["val_step_action_accuracy"], label=f"{stage_name} step_action")
        plt.plot(stage_history["val_stop_accuracy"], linestyle="--", label=f"{stage_name} stop")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Phase4 IL Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def _select_eval_paths(train_paths, val_paths, test_paths):
    if test_paths:
        return test_paths
    if val_paths:
        return val_paths
    return train_paths


def _stage_summary(stage_cfg, train_paths, val_paths, test_paths):
    return {
        "name": stage_cfg["name"],
        "families": list(stage_cfg["families"]),
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "test_size": len(test_paths),
        "train_trace_count": len({int(item["sample_id"]) for item in train_paths}),
        "val_trace_count": len({int(item["sample_id"]) for item in val_paths}),
        "test_trace_count": len({int(item["sample_id"]) for item in test_paths}),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase4 multistep IL pretraining")
    parser.add_argument("--config", type=str, default="src/train_links4meta_il.yaml")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--run_pretrain", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    paths_cfg = cfg["paths"]
    resolve_mapping_paths(
        paths_cfg,
        (
            "pkl_dataset",
            "precomputed_split_input",
            "links_pretrain_input",
            "links_pretrain_cache_output",
            "links_pretrain_model_output",
            "links_pretrain_report_output",
            "forward_model",
            "config_forward",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_plot_output",
            "il_report_output",
            "il_split_output",
            "family_index_output_dir",
            "rl_model_output",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    for key in (
        "il_model_output",
        "il_plot_output",
        "il_report_output",
        "il_split_output",
        "il_multistep_dataset_output",
        "family_index_output_dir",
        "links_pretrain_cache_output",
        "links_pretrain_model_output",
        "links_pretrain_report_output",
    ):
        if key in paths_cfg:
            ensure_parent_dir(paths_cfg[key])

    il_cfg = cfg["il_training"]
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training phase4 IL on device: {device}")

    multistep_dataset_path = paths_cfg.get("il_multistep_dataset_output", paths_cfg["il_dataset_output"])
    expert_paths = ensure_multistep_expert_paths(
        pkl_path=paths_cfg["pkl_dataset"],
        output_path=multistep_dataset_path,
        use_cached=args.skip_extract,
    )
    action_codebook = load_action_codebook(multistep_dataset_path)
    cfg.setdefault("gnn_policy", {})
    cfg["gnn_policy"]["num_geometry_codes"] = max(1, len(action_codebook.get("entries", [])))
    cfg["gnn_policy"]["action_code_dim"] = int(action_codebook.get("code_dim", 6))
    print(f"[*] Total multistep expert samples: {len(expert_paths)}")

    split = load_step_split(
        expert_paths,
        split_path=paths_cfg["il_split_output"],
        precomputed_split_path=paths_cfg.get("precomputed_split_input"),
        val_ratio=il_cfg.get("val_ratio", 0.1),
        test_ratio=il_cfg.get("test_ratio", 0.1),
        split_seed=il_cfg.get("split_seed", 42),
    )
    base_train_paths = subset_by_indices(expert_paths, split["train_indices"])
    base_val_paths = subset_by_indices(expert_paths, split["val_indices"])
    base_test_paths = subset_by_indices(expert_paths, split["test_indices"])
    print(
        f"[*] Using {split.get('split_source')} from {split.get('source_path')} | "
        f"train={len(base_train_paths)} val={len(base_val_paths)} test={len(base_test_paths)}"
    )
    family_index_cfg = cfg.get("family_index", {})
    family_index_report = None
    if family_index_cfg.get("build_on_train", False) and paths_cfg.get("family_index_output_dir"):
        family_index_report = build_family_index_artifacts(
            expert_paths,
            split=split,
            codebook=action_codebook,
            output_dir=paths_cfg["family_index_output_dir"],
            dataset_path=multistep_dataset_path,
            export_jsonl=bool(family_index_cfg.get("export_jsonl", True)),
        )
        print(f"[*] Family IL index exported to {family_index_report['paths']['index_pt']}")

    curve_encoder = CurveEncoder(
        input_dim=cfg["curve_encoder"]["input_dim"],
        hidden_dims=cfg["curve_encoder"]["hidden_dims"],
        latent_dim=cfg["curve_encoder"]["latent_dim"],
    ).to(device)
    policy = GNNPolicy(cfg).to(device)
    policy.set_action_codebook(
        codebook_tensor(action_codebook).to(device),
        buckets=action_codebook.get("bucket_to_ids", {}),
    )
    pretrain_cfg = cfg.get("links_pretrain", {})
    pretrain_report = None
    pretrain_path = paths_cfg.get("links_pretrain_model_output")
    if pretrain_cfg.get("enabled", False) and not args.skip_pretrain:
        should_run_pretrain = bool(args.run_pretrain)
        if pretrain_path and os.path.exists(pretrain_path) and not args.run_pretrain:
            pretrain_report = load_links_pretrained_weights(policy, curve_encoder, pretrain_path, device)
            print(f"[*] Loaded LINKS pretrain checkpoint: {pretrain_path}")
        else:
            should_run_pretrain = should_run_pretrain or bool(pretrain_cfg.get("auto_run_if_missing", False))
            if should_run_pretrain:
                print("[*] Running LINKS graph/validity pretraining before IL ...")
                pretrain_cache = ensure_links_pretrain_cache(
                    dataset_path=paths_cfg.get("links_pretrain_input", paths_cfg["pkl_dataset"]),
                    cache_path=paths_cfg["links_pretrain_cache_output"],
                    split_path=paths_cfg.get("precomputed_split_input"),
                    max_samples=int(pretrain_cfg.get("max_samples", 0)),
                    seed=int(pretrain_cfg.get("seed", 42)),
                    constraint_cfg=cfg.get("constraints", {}),
                    use_cached=not args.run_pretrain,
                )
                pretrain_report = run_links_pretraining(
                    policy=policy,
                    curve_encoder=curve_encoder,
                    cache=pretrain_cache,
                    cfg=cfg,
                    device=device,
                    output_model_path=paths_cfg["links_pretrain_model_output"],
                    output_report_path=paths_cfg["links_pretrain_report_output"],
                )
                print(f"[*] LINKS pretrain complete: {paths_cfg['links_pretrain_model_output']}")
            else:
                print("[*] LINKS pretrain enabled but checkpoint missing; continuing without pretrained encoder.")
    all_params = list(policy.parameters()) + list(curve_encoder.parameters())
    optimizer = optim.Adam(all_params, lr=il_cfg["learning_rate"])

    stage_plan = build_stage_plan(il_cfg)
    history_by_stage = {}
    stage_reports = []

    best_stage_ckpt = None
    for stage_cfg in stage_plan:
        train_paths = filter_paths_by_families(base_train_paths, stage_cfg["families"])
        val_paths = filter_paths_by_families(base_val_paths, stage_cfg["families"])
        test_paths = filter_paths_by_families(base_test_paths, stage_cfg["families"])
        if not train_paths:
            raise RuntimeError(f"Stage '{stage_cfg['name']}' has no training samples for families={stage_cfg['families']}")

        print(
            f"\n[*] {stage_cfg['name']} families={stage_cfg['families']} | "
            f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}"
        )
        train_loader = PreBatchedLoader(train_paths, il_cfg["batch_size"], device, shuffle=True)
        val_loader = PreBatchedLoader(val_paths, il_cfg["batch_size"], device, shuffle=False)
        test_loader = PreBatchedLoader(test_paths, il_cfg["batch_size"], device, shuffle=False)
        selection_loader = val_loader if len(val_loader) > 0 else (test_loader if len(test_loader) > 0 else train_loader)

        stage_history = {
            "train_objective": [],
            "val_objective": [],
            "val_step_action_accuracy": [],
            "val_stop_accuracy": [],
        }
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(stage_cfg["epochs"])))
        best_val_objective = float("inf")
        patience_counter = 0
        best_local_state = None

        for epoch in range(int(stage_cfg["epochs"])):
            train_metrics = train_epoch_prebatched(policy, curve_encoder, optimizer, train_loader, cfg, all_params)
            val_metrics = eval_epoch_prebatched(policy, curve_encoder, selection_loader, cfg)
            scheduler.step()

            stage_history["train_objective"].append(train_metrics["objective"])
            stage_history["val_objective"].append(val_metrics["objective"])
            stage_history["val_step_action_accuracy"].append(val_metrics["step_action_accuracy"])
            stage_history["val_stop_accuracy"].append(val_metrics["stop_accuracy"])
            print(
                f"Epoch [{epoch + 1:3d}/{int(stage_cfg['epochs'])}]  "
                f"Train {_format_metrics(train_metrics)}  "
                f"Val {_format_metrics(val_metrics)}"
            )

            if val_metrics["objective"] < best_val_objective:
                best_val_objective = val_metrics["objective"]
                patience_counter = 0
                best_local_state = {
                    "policy": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                    "curve_encoder": {key: value.detach().cpu() for key, value in curve_encoder.state_dict().items()},
                    "action_codebook": action_codebook,
                    "links_pretrain": pretrain_report or {},
                }
            else:
                patience_counter += 1
                if patience_counter >= int(stage_cfg.get("patience", il_cfg.get("patience", 20))):
                    print(f"  [!] Early stopping {stage_cfg['name']} after {epoch + 1} epochs.")
                    break

        if best_local_state is None:
            raise RuntimeError(f"Stage '{stage_cfg['name']}' failed to produce a checkpoint")

        policy_state = policy.load_state_dict(best_local_state["policy"], strict=False)
        missing_keys, unexpected_keys = policy_load_incompatibilities(policy_state)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Stage '{stage_cfg['name']}' checkpoint incompatible. Missing={missing_keys}, unexpected={unexpected_keys}"
            )
        curve_encoder.load_state_dict(best_local_state["curve_encoder"], strict=False)
        best_stage_ckpt = best_local_state

        final_train_metrics = eval_epoch_prebatched(policy, curve_encoder, train_loader, cfg)
        final_val_metrics = eval_epoch_prebatched(policy, curve_encoder, val_loader, cfg) if len(val_loader) > 0 else {}
        final_test_metrics = eval_epoch_prebatched(policy, curve_encoder, test_loader, cfg) if len(test_loader) > 0 else {}
        reconstruction = evaluate_multistep_reconstruction(
            policy,
            curve_encoder,
            _select_eval_paths(train_paths, val_paths, test_paths),
            cfg,
            device,
            max_traces=il_cfg.get("max_reconstruction_traces", 256),
        )
        stage_report = {
            **_stage_summary(stage_cfg, train_paths, val_paths, test_paths),
            "best_val_objective": float(best_val_objective),
            "train_metrics": _to_float_dict(final_train_metrics),
            "val_metrics": _to_float_dict(final_val_metrics) if final_val_metrics else {},
            "test_metrics": _to_float_dict(final_test_metrics) if final_test_metrics else {},
            "reconstruction": reconstruction,
            "history": stage_history,
        }
        stage_reports.append(stage_report)
        history_by_stage[stage_cfg["name"]] = stage_history
        print(f"[*] Stage summary {stage_cfg['name']}: {json.dumps(stage_report['reconstruction'], ensure_ascii=False)}")

    if best_stage_ckpt is None:
        raise RuntimeError("No phase4 IL stage checkpoint produced")

    model_path = paths_cfg["il_model_output"]
    torch.save(best_stage_ckpt, model_path)
    _save_plot(history_by_stage, paths_cfg["il_plot_output"])

    report = {
        "phase": "phase4_multistep_il",
        "dataset": {
            "num_multistep_samples": len(expert_paths),
            "num_action_codes": len(action_codebook.get("entries", [])),
            "train_size": len(base_train_paths),
            "val_size": len(base_val_paths),
            "test_size": len(base_test_paths),
            "split_source": split.get("split_source"),
            "split_path": paths_cfg["il_split_output"],
            "multistep_dataset_path": multistep_dataset_path,
        },
        "links_pretrain": pretrain_report or {},
        "family_index": family_index_report or {},
        "stages": stage_reports,
        "final_checkpoint": model_path,
    }
    report_path = paths_cfg["il_report_output"]
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"[OK] Phase4 IL plot saved to '{paths_cfg['il_plot_output']}'")
    print(f"[OK] Phase4 IL checkpoint saved to '{model_path}'")
    print(f"[OK] Phase4 IL report saved to '{report_path}'")


if __name__ == "__main__":
    main()
