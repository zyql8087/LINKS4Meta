from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths, resolve_path
from src.inverse.experiment_utils import build_target_feature, compute_joint_metrics_batch, stack_target_features
from src.inverse.inference_runtime import (
    demo_root_from_workspace,
    load_inverse_bundle,
    rollout_trace_policy,
    rollout_trace_with_mcts,
)
from src.inverse.phase4_il import ensure_multistep_expert_paths, load_step_split, subset_by_indices
from src.inverse.phase5_rl import build_trace_dataset, reconstruct_expert_final_graph
from src.inverse.rl_env import (
    _prepare_graph_for_surrogate,
    load_frozen_surrogate,
    validate_graph_structure,
)


ALL_METHODS = ["retrieval_baseline", "il_only", "il_rl", "il_rl_mcts"]


def _predict_metrics(graph, trace_record, surrogate, cfg, device):
    penalty_metric = 1e6
    base_metrics = {
        "joint_score": penalty_metric,
        "foot_score": penalty_metric,
        "foot_nrmse": penalty_metric,
        "foot_chamfer_norm": penalty_metric,
        "knee_nrmse": penalty_metric,
        "ankle_nrmse": penalty_metric,
        "smoothness": penalty_metric,
        "success": 0.0,
        "valid": 0.0,
    }
    if graph is None:
        return base_metrics
    is_valid, _ = validate_graph_structure(graph, cfg.get("constraints", {}))
    if not is_valid:
        return base_metrics

    target = trace_record["target"]
    prepared_graph = _prepare_graph_for_surrogate(
        graph,
        family_index=int(trace_record["family_index"]),
        step_index=int(trace_record["expected_j_steps"]),
        expected_j_steps=int(trace_record["expected_j_steps"]),
    )
    batch = Batch.from_data_list([prepared_graph]).to(device)
    with torch.no_grad():
        pred_foot, pred_knee, pred_ankle = surrogate(batch)
    metrics = compute_joint_metrics_batch(
        pred_foot.cpu(),
        pred_knee.cpu(),
        pred_ankle.cpu(),
        target,
        cfg.get("reward", {}),
    )
    return {
        "joint_score": float(metrics["joint_score"][0].item()),
        "foot_score": float(metrics["foot_score"][0].item()),
        "foot_nrmse": float(metrics["foot_nrmse"][0].item()),
        "foot_chamfer_norm": float(metrics["foot_chamfer_norm"][0].item()),
        "knee_nrmse": float(metrics["knee_nrmse"][0].item()),
        "ankle_nrmse": float(metrics["ankle_nrmse"][0].item()),
        "smoothness": float(metrics["smoothness"][0].item()),
        "success": 1.0,
        "valid": 1.0,
    }


def _aggregate_metrics(metric_dicts, elapsed_times):
    if not metric_dicts:
        return {}
    keys = sorted(metric_dicts[0].keys())
    summary = {key: float(np.mean([item[key] for item in metric_dicts])) for key in keys}
    summary["avg_inference_sec"] = float(np.mean(elapsed_times)) if elapsed_times else 0.0
    return summary


def _family_breakdown(metric_rows):
    by_family = {}
    family_names = sorted({row["family_id"] for row in metric_rows})
    for family_name in family_names:
        rows = [row["metrics"] for row in metric_rows if row["family_id"] == family_name]
        if not rows:
            continue
        by_family[family_name] = _aggregate_metrics(rows, [row["elapsed_sec"] for row in metric_rows if row["family_id"] == family_name])
    return by_family


def _build_phase6_analysis(method_reports):
    analysis = {
        "mcts_only_inference": True,
    }
    rl_report = method_reports.get("il_rl", {})
    mcts_report = method_reports.get("il_rl_mcts", {})
    rl_summary = rl_report.get("summary") or {}
    mcts_summary = mcts_report.get("summary") or {}
    if rl_summary and mcts_summary:
        analysis["valid_rate_improvement"] = float(mcts_summary.get("valid", 0.0) - rl_summary.get("valid", 0.0))
        analysis["target_match_improvement"] = float(rl_summary.get("joint_score", 0.0) - mcts_summary.get("joint_score", 0.0))
    rl_9bar = (rl_report.get("family_breakdown") or {}).get("9bar") or {}
    mcts_9bar = (mcts_report.get("family_breakdown") or {}).get("9bar") or {}
    if rl_9bar and mcts_9bar:
        analysis["family_9bar"] = {
            "valid_rate_improvement": float(mcts_9bar.get("valid", 0.0) - rl_9bar.get("valid", 0.0)),
            "target_match_improvement": float(rl_9bar.get("joint_score", 0.0) - mcts_9bar.get("joint_score", 0.0)),
            "mcts_more_stable_than_no_search": bool(
                mcts_9bar.get("valid", 0.0) >= rl_9bar.get("valid", 0.0)
                and mcts_9bar.get("joint_score", float("inf")) <= rl_9bar.get("joint_score", float("inf"))
            ),
        }
    return analysis


def _evaluate_method(method_name, traces, train_traces, train_features, bundles, surrogate, cfg, device):
    metrics = []
    elapsed_times = []
    metric_rows = []
    search_diagnostics = []

    for trace in traces:
        start_t = time.time()
        if method_name == "retrieval_baseline":
            target_feature = build_target_feature(trace["target"], cfg.get("experiment", {}).get("retrieval_weights", {}))
            distances = torch.norm(train_features - target_feature.unsqueeze(0), dim=1)
            nearest_idx = int(torch.argmin(distances).item())
            graph = reconstruct_expert_final_graph(train_traces[nearest_idx])
            search_info = None
        elif method_name == "il_only":
            graph = rollout_trace_policy(bundles["il"], trace, cfg, device)
            search_info = None
        elif method_name == "il_rl":
            graph = rollout_trace_policy(bundles["rl"], trace, cfg, device)
            search_info = None
        elif method_name == "il_rl_mcts":
            graph, search_info = rollout_trace_with_mcts(bundles["rl"], trace, surrogate, cfg, device)
        else:
            raise ValueError(f"Unsupported method: {method_name}")
        elapsed = time.time() - start_t
        metric = _predict_metrics(graph, trace, surrogate, cfg, device)
        elapsed_times.append(elapsed)
        metrics.append(metric)
        metric_rows.append(
            {
                "family_id": str(trace["family_id"]),
                "metrics": metric,
                "elapsed_sec": elapsed,
            }
        )
        if search_info is not None:
            search_diagnostics.append(search_info)

    report = {
        "num_samples": len(traces),
        "summary": _aggregate_metrics(metrics, elapsed_times),
        "family_breakdown": _family_breakdown(metric_rows),
    }
    if search_diagnostics:
        report["search_diagnostics"] = {
            "avg_candidate_count": float(np.mean([item["candidate_count"] for item in search_diagnostics])),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="Run phase6 inference rerank comparison experiment.")
    parser.add_argument("--config", type=str, default="src/config_inverse.yaml")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        (
            "pkl_dataset",
            "precomputed_split_input",
            "forward_model",
            "config_forward",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_split_output",
            "rl_model_output",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    if cfg.get("experiment", {}).get("eval_report_output"):
        cfg["experiment"]["eval_report_output"] = str(
            resolve_path(cfg["experiment"]["eval_report_output"], config_path.parent, WORKSPACE_ROOT)
        )
        ensure_parent_dir(cfg["experiment"]["eval_report_output"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase6 Experiment] device={device}")

    multistep_dataset_path = cfg["paths"].get("il_multistep_dataset_output", cfg["paths"]["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=cfg["paths"]["pkl_dataset"],
        output_path=multistep_dataset_path,
        use_cached=True,
    )
    split = load_step_split(
        step_paths,
        split_path=cfg["paths"]["il_split_output"],
        precomputed_split_path=cfg["paths"].get("precomputed_split_input"),
        val_ratio=cfg["il_training"].get("val_ratio", 0.1),
        test_ratio=cfg["il_training"].get("test_ratio", 0.1),
        split_seed=cfg["il_training"].get("split_seed", 42),
    )

    train_traces = build_trace_dataset(subset_by_indices(step_paths, split["train_indices"]))
    test_traces = build_trace_dataset(subset_by_indices(step_paths, split["test_indices"]))
    sample_cap = args.num_samples or cfg.get("experiment", {}).get("eval_num_samples")
    if sample_cap:
        test_traces = test_traces[:sample_cap]

    surrogate, _ = load_frozen_surrogate(cfg["paths"]["forward_model"], cfg["paths"]["config_forward"], device)
    bundles = {
        "il": load_inverse_bundle(cfg, cfg["paths"]["il_model_output"], device, allow_fresh_fallback=True),
        "rl": load_inverse_bundle(cfg, cfg["paths"]["rl_model_output"], device, allow_fresh_fallback=True),
    }
    for bundle_name, bundle in bundles.items():
        if bundle is not None and bundle.get("checkpoint_warning"):
            print(f"[Phase6 Experiment] warning ({bundle_name}): {bundle['checkpoint_warning']}")

    methods = ALL_METHODS if args.methods == "all" else [item.strip() for item in args.methods.split(",") if item.strip()]
    train_features = stack_target_features([trace["target"] for trace in train_traces], weights=cfg.get("experiment", {}).get("retrieval_weights", {}))

    report = {
        "phase": "phase6_inference_rerank",
        "split": {
            "train_trace_count": len(train_traces),
            "test_trace_count": len(test_traces),
            "split_source": split.get("split_source"),
            "source_path": split.get("source_path"),
            "sample_cap": sample_cap,
        },
        "bundles": {
            name: {
                "checkpoint_path": bundle.get("checkpoint_path"),
                "checkpoint_loaded": bool(bundle.get("checkpoint_loaded")),
                "checkpoint_warning": bundle.get("checkpoint_warning"),
            }
            for name, bundle in bundles.items()
        },
        "methods": {},
    }

    for method in methods:
        print(f"[Phase6 Experiment] evaluating {method} ...")
        report["methods"][method] = _evaluate_method(
            method,
            test_traces,
            train_traces,
            train_features,
            bundles,
            surrogate,
            cfg,
            device,
        )

    report["phase6_analysis"] = _build_phase6_analysis(report["methods"])

    report_path = Path(
        cfg.get("experiment", {}).get("eval_report_output", demo_root_from_workspace(WORKSPACE_ROOT) / "outputs" / "phase6_experiment_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[Phase6 Experiment] report saved to {report_path}")


if __name__ == "__main__":
    main()
