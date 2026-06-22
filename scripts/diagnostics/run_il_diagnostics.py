import argparse
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.action_codebook import family_name_from_index, load_action_codebook, resolve_codebook_bucket_for_step
from src.inverse.inference_runtime import demo_root_from_workspace, load_inverse_bundle
from src.inverse.phase4_il import (
    attach_oracle_code_targets,
    evaluate_constrained_decoder_reconstruction,
    evaluate_multistep_reconstruction,
    evaluate_multistep_reconstruction_detailed,
    ensure_multistep_expert_paths,
    filter_paths_by_families,
    group_paths_by_trace,
    load_step_split,
    subset_by_indices,
)
from train_inverse_bio import PreBatchedLoader, eval_epoch_prebatched


FAMILY_ORDER = ("6bar", "7bar", "8bar", "9bar")


def _parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Phase4 multistep IL checkpoint quality.")
    parser.add_argument("--config", type=str, default="src/config_inverse.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--max_traces_per_family", type=int, default=64)
    parser.add_argument("--include_failure_analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_failure_examples_per_family", type=int, default=20)
    parser.add_argument("--constrained_code_topk", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_cfg(args):
    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        (
            "pkl_dataset",
            "precomputed_split_input",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_split_output",
            "rl_model_output",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    return cfg, config_path


def _to_float_dict(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items()}


def _eval_paths(policy, curve_encoder, paths, cfg, device):
    if not paths:
        return {}
    loader = PreBatchedLoader(
        paths,
        int(cfg["il_training"].get("batch_size", 4096)),
        device,
        shuffle=False,
        num_geometry_codes=int(cfg.get("gnn_policy", {}).get("num_geometry_codes", 0) or 0),
    )
    return _to_float_dict(eval_epoch_prebatched(policy, curve_encoder, loader, cfg))


def _metrics_by_group(policy, curve_encoder, test_paths, cfg, device):
    grouped = {
        "overall": {"all": test_paths},
        "by_family": {},
        "by_step_role": {},
        "by_step_index": {},
        "by_family_step_role": {},
        "by_family_step_index": {},
    }
    for family in FAMILY_ORDER:
        grouped["by_family"][family] = [item for item in test_paths if str(item["family_id"]) == family]
    for role in ("aux", "semantic"):
        grouped["by_step_role"][role] = [item for item in test_paths if str(item["step_role"]) == role]
    for step_index in sorted({int(item["step_index"]) for item in test_paths}):
        grouped["by_step_index"][str(step_index)] = [item for item in test_paths if int(item["step_index"]) == step_index]
    for family in FAMILY_ORDER:
        for role in ("aux", "semantic"):
            key = f"{family}/{role}"
            grouped["by_family_step_role"][key] = [
                item for item in test_paths if str(item["family_id"]) == family and str(item["step_role"]) == role
            ]
        for step_index in sorted({int(item["step_index"]) for item in test_paths if str(item["family_id"]) == family}):
            key = f"{family}/step{step_index}"
            grouped["by_family_step_index"][key] = [
                item for item in test_paths if str(item["family_id"]) == family and int(item["step_index"]) == step_index
            ]

    report = {}
    for group_name, group_items in grouped.items():
        report[group_name] = {}
        for key, paths in group_items.items():
            metrics = _eval_paths(policy, curve_encoder, paths, cfg, device)
            report[group_name][key] = {
                "count": int(len(paths)),
                "trace_count": int(len({int(item["trace_id"]) for item in paths})),
                "metrics": metrics,
            }
    return report


def _distribution_report(step_paths, split):
    split_sets = {
        "train": subset_by_indices(step_paths, split["train_indices"]),
        "val": subset_by_indices(step_paths, split["val_indices"]),
        "test": subset_by_indices(step_paths, split["test_indices"]),
    }
    payload = {}
    for split_name, items in split_sets.items():
        payload[split_name] = {
            "num_steps": int(len(items)),
            "num_traces": int(len({int(item["trace_id"]) for item in items})),
            "family_counts": dict(sorted(Counter(str(item["family_id"]) for item in items).items())),
            "step_role_counts": dict(sorted(Counter(str(item["step_role"]) for item in items).items())),
            "step_index_counts": {str(k): int(v) for k, v in sorted(Counter(int(item["step_index"]) for item in items).items())},
            "step_count_counts": {str(k): int(v) for k, v in sorted(Counter(int(item["step_count"]) for item in items).items())},
            "bucket_counts": dict(sorted(Counter(str(item["action_code_bucket"]) for item in items).items())),
        }
    trace_sets = {
        split_name: {int(item["trace_id"]) for item in items}
        for split_name, items in split_sets.items()
    }
    payload["trace_leakage"] = {
        "train_val": int(len(trace_sets["train"] & trace_sets["val"])),
        "train_test": int(len(trace_sets["train"] & trace_sets["test"])),
        "val_test": int(len(trace_sets["val"] & trace_sets["test"])),
    }
    return payload, split_sets


def _baseline_report(test_paths, action_codebook):
    entries = action_codebook.get("entries", [])
    code_ids = {int(entry["id"]) for entry in entries}
    truth_ids = [int(item["action_code_id"]) for item in test_paths]
    truth_covered = all(code_id in code_ids for code_id in truth_ids)
    majority_code_id, majority_count = Counter(truth_ids).most_common(1)[0]

    bucket_to_ids = {str(key): [int(idx) for idx in value] for key, value in action_codebook.get("bucket_to_ids", {}).items()}
    by_bucket = {}
    random_expected = []
    for bucket, bucket_items in sorted(_group_by(test_paths, lambda item: str(item["action_code_bucket"])).items()):
        bucket_truths = [int(item["action_code_id"]) for item in bucket_items]
        majority = Counter(bucket_truths).most_common(1)[0]
        bucket_size = max(1, len(bucket_to_ids.get(bucket, [])))
        by_bucket[bucket] = {
            "count": int(len(bucket_items)),
            "num_codes": int(bucket_size),
            "majority_code_id": int(majority[0]),
            "majority_accuracy": float(majority[1] / max(1, len(bucket_items))),
            "random_valid_expected_accuracy": float(1.0 / bucket_size),
        }
        random_expected.extend([1.0 / bucket_size] * len(bucket_items))

    return {
        "truth_code_coverage": bool(truth_covered),
        "num_codes": int(len(entries)),
        "overall_majority_code_id": int(majority_code_id),
        "overall_majority_accuracy": float(majority_count / max(1, len(test_paths))),
        "bucket_valid_random_expected_accuracy": float(sum(random_expected) / max(1, len(random_expected))),
        "by_bucket": by_bucket,
    }


def _group_by(items, key_fn):
    grouped = defaultdict(list)
    for item in items:
        grouped[key_fn(item)].append(item)
    return grouped


def _balanced_reconstruction(policy, curve_encoder, test_paths, cfg, device, max_traces_per_family):
    report = {}
    for family in FAMILY_ORDER:
        family_paths = filter_paths_by_families(test_paths, [family])
        traces = group_paths_by_trace(family_paths)
        selected_trace_ids = {int(trace[0]["trace_id"]) for trace in traces[: max(0, int(max_traces_per_family))]}
        selected_paths = [item for item in family_paths if int(item["trace_id"]) in selected_trace_ids]
        report[family] = evaluate_multistep_reconstruction(
            policy,
            curve_encoder,
            selected_paths,
            cfg,
            device,
            max_traces=max_traces_per_family,
        )
    return report


def _balanced_failure_analysis(
    policy,
    curve_encoder,
    test_paths,
    cfg,
    device,
    max_traces_per_family,
    max_failure_examples_per_family,
):
    selected_paths = []
    for family in FAMILY_ORDER:
        family_paths = filter_paths_by_families(test_paths, [family])
        traces = group_paths_by_trace(family_paths)
        selected_trace_ids = {int(trace[0]["trace_id"]) for trace in traces[: max(0, int(max_traces_per_family))]}
        selected_paths.extend([item for item in family_paths if int(item["trace_id"]) in selected_trace_ids])
    return evaluate_multistep_reconstruction_detailed(
        policy,
        curve_encoder,
        selected_paths,
        cfg,
        device,
        max_traces=None,
        max_failure_examples_per_family=max_failure_examples_per_family,
    )


def _codebook_bucket_check(test_paths, action_codebook):
    bucket_to_ids = {str(key): set(int(idx) for idx in value) for key, value in action_codebook.get("bucket_to_ids", {}).items()}
    missing = []
    for item in test_paths:
        bucket = resolve_codebook_bucket_for_step(
            bucket_to_ids,
            family_name_from_index(int(item["family_index"])),
            "semantic" if int(item["step_role_index"]) == 1 else "aux",
            step_index=int(item["step_index"]),
            action_topo=item["action_topo"],
            topology_signature=item.get("topology_signature"),
            fine_bucket_policy=action_codebook.get("fine_bucket_policy", {}),
        )
        code_id = int(item["action_code_id"])
        if code_id not in bucket_to_ids.get(bucket, set()):
            missing.append(
                {
                    "trace_id": int(item["trace_id"]),
                    "step_index": int(item["step_index"]),
                    "family": str(item["family_id"]),
                    "step_role": str(item["step_role"]),
                    "expected_bucket": bucket,
                    "stored_bucket": str(item["action_code_bucket"]),
                    "action_code_id": code_id,
                }
            )
            if len(missing) >= 20:
                break
    return {
        "bucket_assignment_valid": not missing,
        "example_mismatches": missing,
    }


def main():
    args = _parse_args()
    cfg, config_path = _resolve_cfg(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = args.checkpoint or cfg["paths"]["il_model_output"]
    output_json = args.output_json
    if output_json is None:
        output_json = str(demo_root_from_workspace(WORKSPACE_ROOT) / "outputs" / "il_v5" / "reports" / "il_diagnostics_current.json")
    output_json = str(Path(output_json))
    ensure_parent_dir(output_json)

    dataset_path = cfg["paths"].get("il_multistep_dataset_output", cfg["paths"]["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=cfg["paths"]["pkl_dataset"],
        output_path=dataset_path,
        use_cached=True,
        action_codebook_cfg=cfg.get("action_codebook", {}),
        constraint_cfg=cfg.get("constraints", {}),
    )
    split = load_step_split(
        step_paths,
        split_path=cfg["paths"]["il_split_output"],
        precomputed_split_path=cfg["paths"].get("precomputed_split_input"),
        val_ratio=cfg["il_training"].get("val_ratio", 0.1),
        test_ratio=cfg["il_training"].get("test_ratio", 0.1),
        split_seed=cfg["il_training"].get("split_seed", 42),
    )
    distributions, split_sets = _distribution_report(step_paths, split)
    test_paths = split_sets["test"]
    action_codebook = load_action_codebook(dataset_path)
    cfg.setdefault("gnn_policy", {})
    cfg["gnn_policy"]["num_geometry_codes"] = max(1, len(action_codebook.get("entries", [])))
    cfg["gnn_policy"]["action_code_dim"] = int(action_codebook.get("code_dim", 6))
    oracle_code_report = None
    constrained_topk = args.constrained_code_topk
    if constrained_topk is None:
        constrained_topk = int(cfg.get("diagnostics", {}).get("constrained_code_topk", 0) or 0)
    if bool(cfg.get("il_training", {}).get("oracle_code_loss", {}).get("enabled", False)) or int(constrained_topk) > 0:
        step_paths, oracle_code_report = attach_oracle_code_targets(step_paths, action_codebook, cfg)
        split_sets = {
            "train": subset_by_indices(step_paths, split["train_indices"]),
            "val": subset_by_indices(step_paths, split["val_indices"]),
            "test": subset_by_indices(step_paths, split["test_indices"]),
        }
        test_paths = split_sets["test"]

    bundle = load_inverse_bundle(cfg, checkpoint, device, allow_fresh_fallback=False, require_geometry_code_ready=True)
    if bundle is None:
        raise RuntimeError(f"failed to load IL checkpoint: {checkpoint}")
    policy = bundle["policy"]
    curve_encoder = bundle["curve_encoder"]

    report = {
        "phase": "phase4_il_diagnostics",
        "config": {
            "config_path": str(config_path),
            "checkpoint": str(checkpoint),
            "device": str(device),
            "max_traces_per_family": int(args.max_traces_per_family),
            "include_failure_analysis": bool(args.include_failure_analysis),
            "max_failure_examples_per_family": int(args.max_failure_examples_per_family),
            "constrained_code_topk": int(constrained_topk),
            "split_source": split.get("split_source"),
            "source_path": split.get("source_path"),
            "dataset_path": dataset_path,
        },
        "checkpoint_status": {
            "loaded": bool(bundle.get("checkpoint_loaded")),
            "warning": bundle.get("checkpoint_warning"),
            "geometry_code_ready": bool(bundle.get("geometry_code_ready")),
            "geometry_code_issue": bundle.get("geometry_code_issue"),
            "geometry_code_status": bundle.get("geometry_code_status"),
        },
        "distribution": distributions,
        "oracle_code_targets": oracle_code_report or {},
        "oracle_sanity": {
            "baselines": _baseline_report(test_paths, action_codebook),
            "bucket_check": _codebook_bucket_check(test_paths, action_codebook),
        },
        "teacher_forced": _metrics_by_group(policy, curve_encoder, test_paths, cfg, device),
        "balanced_reconstruction_by_family": _balanced_reconstruction(
            policy,
            curve_encoder,
            test_paths,
            cfg,
            device,
            max_traces_per_family=args.max_traces_per_family,
        ),
    }
    if args.include_failure_analysis:
        report["autoregressive_failure_analysis"] = _balanced_failure_analysis(
            policy,
            curve_encoder,
            test_paths,
            cfg,
            device,
            max_traces_per_family=args.max_traces_per_family,
            max_failure_examples_per_family=args.max_failure_examples_per_family,
        )
    if int(constrained_topk) > 0:
        report["constrained_decoder"] = evaluate_constrained_decoder_reconstruction(
            policy,
            curve_encoder,
            test_paths,
            cfg,
            device,
            top_k=int(constrained_topk),
            max_traces=args.max_traces_per_family * len(FAMILY_ORDER),
        )

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[OK] IL diagnostics saved to {output_json}")


if __name__ == "__main__":
    main()
