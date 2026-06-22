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
from src.inverse.action_codebook import default_action_codebook_path, load_action_codebook
from src.inverse.phase4_il import ensure_multistep_expert_paths


def _parse_args():
    parser = argparse.ArgumentParser(description="Rebuild Phase4 IL dataset with the v2 validity-preserving action codebook.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def _resolve_cfg(config_path: str):
    cfg, resolved_path = load_yaml_config(config_path, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        (
            "pkl_dataset",
            "precomputed_split_input",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_plot_output",
            "il_report_output",
            "il_split_output",
            "family_index_output_dir",
            "rl_model_output",
        ),
        config_dir=resolved_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    return cfg, resolved_path


def _entry_validity_summary(entries):
    counts = []
    rates = []
    for entry in entries:
        context_count = int(entry.get("validity_context_count", 0))
        pass_count = int(entry.get("validity_pass_count", 0))
        counts.append(context_count)
        if context_count > 0:
            rates.append(pass_count / context_count)
    return {
        "entries_with_validation": int(sum(1 for count in counts if count > 0)),
        "mean_validity_rate": float(sum(rates) / len(rates)) if rates else 0.0,
        "min_validity_rate": float(min(rates)) if rates else 0.0,
        "max_validity_rate": float(max(rates)) if rates else 0.0,
    }


def _bucket_distribution(step_paths):
    by_family_step = defaultdict(Counter)
    by_bucket = Counter()
    ninebar_semantic = Counter()
    for item in step_paths:
        family = str(item.get("family_id", "unknown"))
        step_index = int(item.get("step_index", -1))
        step_role = str(item.get("step_role", "unknown"))
        bucket = str(item.get("action_code_bucket", "unknown"))
        by_bucket[bucket] += 1
        by_family_step[f"{family}/step{step_index}/{step_role}"][bucket] += 1
        if family == "9bar" and step_role == "semantic":
            ninebar_semantic[bucket] += 1
    return {
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_family_step": {key: dict(sorted(value.items())) for key, value in sorted(by_family_step.items())},
        "ninebar_semantic": dict(sorted(ninebar_semantic.items())),
    }


def main():
    args = _parse_args()
    cfg, config_path = _resolve_cfg(args.config)
    output_json = str(Path(args.output_json))
    ensure_parent_dir(output_json)

    dataset_path = cfg["paths"].get("il_multistep_dataset_output", cfg["paths"]["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=cfg["paths"]["pkl_dataset"],
        output_path=dataset_path,
        use_cached=False,
        action_codebook_cfg=cfg.get("action_codebook", {}),
        constraint_cfg=cfg.get("constraints", {}),
    )
    codebook = load_action_codebook(dataset_path)
    entries = list(codebook.get("entries", []))
    bucket_to_ids = dict(codebook.get("bucket_to_ids", {}))
    report = {
        "phase": "phase4_il_codebook_rebuild",
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "codebook_path": default_action_codebook_path(dataset_path),
        "num_steps": int(len(step_paths)),
        "num_traces": int(len({int(item["trace_id"]) for item in step_paths})),
        "codebook": {
            "version": str(codebook.get("version")),
            "strategy": str(codebook.get("strategy")),
            "representative_strategy": str(codebook.get("representative_strategy")),
            "cluster_radius": float(codebook.get("cluster_radius", 0.0)),
            "max_codes_per_bucket": int(codebook.get("max_codes_per_bucket", 0)),
            "max_codes_per_bucket_overrides": dict(codebook.get("max_codes_per_bucket_overrides", {})),
            "entry_count": int(len(entries)),
            "bucket_count": int(len(bucket_to_ids)),
            "fine_bucket_policy": dict(codebook.get("fine_bucket_policy", {})),
            "bucket_sizes": {str(key): int(len(value)) for key, value in sorted(bucket_to_ids.items())},
            "representative_validity": _entry_validity_summary(entries),
        },
        "distribution": _bucket_distribution(step_paths),
        "go_no_go": {
            "has_v2_metadata": str(codebook.get("version")) == "geom_codebook_v2_validity_context_bucket",
            "bucket_count_gt_4": int(len(bucket_to_ids)) > 4,
            "has_semantic_9bar_step1_topo_bucket": any(
                str(bucket).startswith("semantic_9bar_step1_topo_") for bucket in bucket_to_ids
            ),
        },
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[OK] Codebook rebuild report saved to {output_json}")


if __name__ == "__main__":
    main()
