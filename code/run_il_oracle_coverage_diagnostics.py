from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.action_codebook import (
    allowed_code_ids_for_context,
    family_name_from_index,
    load_action_codebook,
    resolve_codebook_bucket_for_step,
)
from src.inverse.phase4_il import (
    _apply_code_vector_for_oracle,
    _normalised_geometry_error,
    ensure_multistep_expert_paths,
)


FAMILY_ORDER = ("6bar", "7bar", "8bar", "9bar")
DEFAULT_THRESHOLDS = (0.025, 0.05, 0.075, 0.10, 0.15)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full/layered oracle coverage diagnostics for Phase4 IL geometry code buckets."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--families", type=str, nargs="*", default=list(FAMILY_ORDER))
    parser.add_argument("--max_steps_per_family", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=5000)
    return parser.parse_args()


def _resolve_cfg(config_arg: str) -> tuple[dict, Path]:
    cfg, config_path = load_yaml_config(config_arg, SCRIPT_DIR, WORKSPACE_ROOT)
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


def _threshold_key(threshold: float) -> str:
    return str(float(threshold))


def _as_topo_list(action_topo) -> list[int]:
    if hasattr(action_topo, "detach"):
        return [int(value) for value in action_topo.detach().cpu().view(-1).tolist()]
    return [int(value) for value in action_topo]


def _family_name(item: dict[str, object]) -> str:
    value = item.get("family_id")
    if value is not None:
        return str(value)
    return family_name_from_index(int(item.get("family_index", -1)))


def _step_role(item: dict[str, object]) -> str:
    value = item.get("step_role")
    if value is not None:
        return str(value)
    return "semantic" if int(item.get("step_role_index", 0)) == 1 else "aux"


def _quantiles(values: Sequence[float], *, include_mean: bool = True) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not numeric:
        out: dict[str, float | int | None] = {"count": 0, "min": None, "max": None, "p50": None, "p90": None, "p95": None}
        if include_mean:
            out["mean"] = None
        return out
    arr = np.asarray(numeric, dtype=np.float64)
    out = {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }
    if include_mean:
        out["mean"] = float(np.mean(arr))
    return out


def _empty_accumulator() -> dict[str, object]:
    return {
        "count": 0,
        "valid_code_available": 0,
        "truth_code_valid": 0,
        "candidate_counts": [],
        "best_valid_errors": [],
        "valid_errors_by_step": [],
        "invalid_reason_counts": Counter(),
    }


def _update_accumulator(group: dict[str, object], record: dict[str, object]) -> None:
    valid_errors = [float(value) for value in record.get("valid_errors", []) or []]
    group["count"] = int(group["count"]) + 1
    group["valid_code_available"] = int(group["valid_code_available"]) + int(len(valid_errors) > 0)
    group["truth_code_valid"] = int(group["truth_code_valid"]) + int(bool(record.get("truth_code_valid", False)))
    group["candidate_counts"].append(int(record.get("candidate_count", 0)))
    if record.get("best_valid_error") is not None:
        group["best_valid_errors"].append(float(record["best_valid_error"]))
    group["valid_errors_by_step"].append(valid_errors)
    group["invalid_reason_counts"].update(Counter(record.get("invalid_reason_counts", {}) or {}))


def _finalize_accumulator(group: dict[str, object], thresholds: Sequence[float]) -> dict[str, object]:
    count = int(group.get("count", 0))
    valid = int(group.get("valid_code_available", 0))
    truth_valid = int(group.get("truth_code_valid", 0))
    valid_errors_by_step = list(group.get("valid_errors_by_step", []) or [])
    coverage_by_threshold = {}
    for threshold in thresholds:
        key = _threshold_key(float(threshold))
        positive = sum(1 for errors in valid_errors_by_step if any(float(error) <= float(threshold) for error in errors))
        coverage_by_threshold[key] = {
            "oracle_positive": int(positive),
            "oracle_positive_coverage_rate": float(positive / count) if count else 0.0,
            "oracle_uncovered": int(count - positive),
            "oracle_uncovered_rate": float((count - positive) / count) if count else 0.0,
        }

    return {
        "count": count,
        "valid_code_available": valid,
        "valid_code_available_rate": float(valid / count) if count else 0.0,
        "truth_code_valid": truth_valid,
        "truth_code_valid_rate": float(truth_valid / count) if count else 0.0,
        "coverage_by_threshold": coverage_by_threshold,
        "candidate_count_stats": _quantiles(group.get("candidate_counts", []) or [], include_mean=True),
        "best_valid_error_quantiles": _quantiles(group.get("best_valid_errors", []) or [], include_mean=True),
        "invalid_reason_counts": dict(sorted(Counter(group.get("invalid_reason_counts", {})).items())),
    }


def _group_records(records: Sequence[dict[str, object]], key_fn, thresholds: Sequence[float]) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = defaultdict(_empty_accumulator)
    for record in records:
        _update_accumulator(groups[str(key_fn(record))], record)
    return {key: _finalize_accumulator(value, thresholds) for key, value in sorted(groups.items())}


def _coverage_rate(group: dict[str, object] | None, threshold: float) -> float:
    if not group:
        return 0.0
    payload = group.get("coverage_by_threshold", {}).get(_threshold_key(threshold), {})
    return float(payload.get("oracle_positive_coverage_rate", 0.0))


def _target_buckets(summary: dict[str, object], threshold: float) -> list[str]:
    targets = []
    for key, group in summary.get("by_family_bucket", {}).items():
        if not str(key).startswith("9bar/"):
            continue
        valid_rate = float(group.get("valid_code_available_rate", 0.0))
        coverage = _coverage_rate(group, threshold)
        if valid_rate < 0.80 or coverage < 0.70:
            targets.append(str(key).split("/", 1)[1])
    return sorted(set(targets))


def choose_oracle_coverage_recommendation(summary: dict[str, object], thresholds: Sequence[float]) -> dict[str, object]:
    thresholds = sorted(float(value) for value in thresholds)
    strict = 0.025 if 0.025 in thresholds else thresholds[0]
    loose = 0.15 if 0.15 in thresholds else thresholds[-1]
    nine = summary.get("by_family", {}).get("9bar")
    if not nine:
        return {
            "action": "insufficient_9bar_data",
            "train_v8_go": False,
            "reason": "no_9bar_records_in_diagnostic_scope",
            "selected_threshold": None,
            "target_buckets": [],
        }

    valid_rate = float(nine.get("valid_code_available_rate", 0.0))
    strict_coverage = _coverage_rate(nine, strict)
    loose_coverage = _coverage_rate(nine, loose)
    if valid_rate < 0.80:
        return {
            "action": "rebuild_or_expand_codebook",
            "train_v8_go": False,
            "reason": "9bar_valid_code_available_below_0.80",
            "selected_threshold": None,
            "9bar_valid_code_available_rate": valid_rate,
            "target_buckets": _target_buckets(summary, loose),
        }

    relaxed_threshold = None
    for threshold in thresholds:
        if threshold <= strict:
            continue
        if _coverage_rate(nine, threshold) >= 0.70:
            relaxed_threshold = float(threshold)
            break
    if strict_coverage < 0.70 and relaxed_threshold is not None:
        return {
            "action": "relax_equivalence_threshold",
            "train_v8_go": False,
            "reason": "9bar_coverage_passes_only_after_threshold_relaxation",
            "selected_threshold": relaxed_threshold,
            "9bar_strict_coverage": strict_coverage,
            "9bar_relaxed_coverage": _coverage_rate(nine, relaxed_threshold),
            "target_buckets": _target_buckets(summary, relaxed_threshold),
        }

    if loose_coverage < 0.70:
        return {
            "action": "increase_per_bucket_codes_or_rebuild",
            "train_v8_go": False,
            "reason": "9bar_coverage_below_0.70_even_at_loose_threshold",
            "selected_threshold": None,
            "9bar_loose_coverage": loose_coverage,
            "target_buckets": _target_buckets(summary, loose),
        }

    return {
        "action": "go_train_v8_oracle_ranking",
        "train_v8_go": True,
        "reason": "9bar_valid_availability_and_oracle_coverage_pass",
        "selected_threshold": strict,
        "9bar_valid_code_available_rate": valid_rate,
        "9bar_strict_coverage": strict_coverage,
        "target_buckets": [],
    }


def summarize_oracle_candidate_records(
    records: Sequence[dict[str, object]],
    thresholds: Sequence[float],
) -> dict[str, object]:
    thresholds = sorted(float(value) for value in thresholds)
    overall_acc = _empty_accumulator()
    invalid_reasons = Counter()
    for record in records:
        _update_accumulator(overall_acc, record)
        invalid_reasons.update(Counter(record.get("invalid_reason_counts", {}) or {}))
    overall = _finalize_accumulator(overall_acc, thresholds)
    summary = {
        "overall": overall,
        "by_family": _group_records(records, lambda item: item.get("family", "unknown"), thresholds),
        "by_family_step_index": _group_records(
            records,
            lambda item: f"{item.get('family', 'unknown')}/step{int(item.get('step_index', 0))}",
            thresholds,
        ),
        "by_family_step_role": _group_records(
            records,
            lambda item: f"{item.get('family', 'unknown')}/{item.get('step_role', 'unknown')}",
            thresholds,
        ),
        "by_family_step_index_role": _group_records(
            records,
            lambda item: (
                f"{item.get('family', 'unknown')}/step{int(item.get('step_index', 0))}/"
                f"{item.get('step_role', 'unknown')}"
            ),
            thresholds,
        ),
        "by_bucket": _group_records(records, lambda item: item.get("bucket", "unknown"), thresholds),
        "by_family_bucket": _group_records(
            records,
            lambda item: f"{item.get('family', 'unknown')}/{item.get('bucket', 'unknown')}",
            thresholds,
        ),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "threshold_sweep": {},
        "best_valid_error_quantiles": {},
        "candidate_count_stats": {},
    }
    for threshold in thresholds:
        key = _threshold_key(threshold)
        summary["threshold_sweep"][key] = {
            "overall": overall["coverage_by_threshold"][key],
            "by_family": {
                family: group["coverage_by_threshold"][key]
                for family, group in summary["by_family"].items()
            },
        }
    summary["best_valid_error_quantiles"]["overall"] = overall["best_valid_error_quantiles"]
    summary["candidate_count_stats"]["overall"] = overall["candidate_count_stats"]
    for bucket, group in summary["by_bucket"].items():
        summary["best_valid_error_quantiles"][bucket] = group["best_valid_error_quantiles"]
        summary["candidate_count_stats"][bucket] = group["candidate_count_stats"]
    summary["recommendation"] = choose_oracle_coverage_recommendation(summary, thresholds)
    return summary


def _select_records_by_family_limit(
    step_paths: Sequence[dict[str, object]],
    families: Iterable[str],
    max_steps_per_family: int,
) -> list[dict[str, object]]:
    family_set = {str(value) for value in families}
    selected = []
    counts = Counter()
    for item in step_paths:
        family = _family_name(item)
        if family not in family_set:
            continue
        if max_steps_per_family > 0 and counts[family] >= max_steps_per_family:
            continue
        selected.append(item)
        counts[family] += 1
    return selected


def _candidate_record_for_step(
    item: dict[str, object],
    codebook: dict[str, object],
    entries: dict[int, np.ndarray],
    constraints: dict,
) -> dict[str, object]:
    family = _family_name(item)
    role = _step_role(item)
    step_index = int(item.get("step_index", 0))
    topo_list = _as_topo_list(item["action_topo"])
    bucket = resolve_codebook_bucket_for_step(
        codebook.get("bucket_to_ids", {}),
        family,
        role,
        step_index=step_index,
        action_topo=topo_list,
        topology_signature=item.get("topology_signature"),
        fine_bucket_policy=codebook.get("fine_bucket_policy", {}),
    )
    allowed_ids = allowed_code_ids_for_context(
        codebook,
        family_name=family,
        step_role=role,
        step_index=step_index,
        action_topo=topo_list,
        topology_signature=item.get("topology_signature"),
    )
    if not allowed_ids:
        allowed_ids = sorted(entries.keys())
        bucket = f"{bucket}|fallback_all_codes"

    valid_errors = []
    invalid_reasons = Counter()
    best_valid_error = None
    best_valid_code_id = None
    valid_code_count = 0
    for code_id in allowed_ids:
        vector = entries.get(int(code_id))
        if vector is None:
            invalid_reasons["missing_codebook_entry"] += 1
            continue
        is_valid, reason, candidate_geo = _apply_code_vector_for_oracle(
            item["base_data"],
            topo_list,
            vector,
            constraints,
        )
        if not is_valid or candidate_geo is None:
            invalid_reasons[str(reason or "invalid")] += 1
            continue
        error = _normalised_geometry_error(item["base_data"], topo_list, candidate_geo, item["action_geo"])
        valid_errors.append(float(error))
        valid_code_count += 1
        if best_valid_error is None or error < best_valid_error:
            best_valid_error = float(error)
            best_valid_code_id = int(code_id)

    truth_code_id = int(item.get("action_code_id", -1))
    truth_code_valid = False
    truth_code_error = None
    truth_code_reason = None
    truth_vec = entries.get(truth_code_id)
    if truth_vec is None:
        truth_code_reason = "missing_codebook_entry"
    else:
        truth_valid, truth_reason, truth_geo = _apply_code_vector_for_oracle(
            item["base_data"],
            topo_list,
            truth_vec,
            constraints,
        )
        truth_code_valid = bool(truth_valid and truth_geo is not None)
        truth_code_reason = None if truth_code_valid else str(truth_reason or "invalid")
        if truth_code_valid:
            truth_code_error = _normalised_geometry_error(item["base_data"], topo_list, truth_geo, item["action_geo"])

    return {
        "trace_id": int(item.get("trace_id", -1)),
        "family": family,
        "step_index": step_index,
        "step_role": role,
        "bucket": str(bucket),
        "candidate_count": int(len(allowed_ids)),
        "valid_code_count": int(valid_code_count),
        "valid_errors": valid_errors,
        "best_valid_error": best_valid_error,
        "best_valid_code_id": best_valid_code_id,
        "truth_code_id": truth_code_id,
        "truth_code_valid": bool(truth_code_valid),
        "truth_code_error": None if truth_code_error is None else float(truth_code_error),
        "truth_code_invalid_reason": truth_code_reason,
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
    }


def build_oracle_candidate_records(
    step_paths: Sequence[dict[str, object]],
    codebook: dict[str, object],
    cfg: dict,
    *,
    progress_every: int = 5000,
) -> list[dict[str, object]]:
    entries = {
        int(entry["id"]): np.asarray(entry["vector"], dtype=np.float32)
        for entry in codebook.get("entries", [])
    }
    constraints = cfg.get("constraints", {})
    records = []
    total = len(step_paths)
    for index, item in enumerate(step_paths, start=1):
        records.append(_candidate_record_for_step(item, codebook, entries, constraints))
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"[oracle-coverage] evaluated {index}/{total} steps")
    return records


def main() -> None:
    args = _parse_args()
    cfg, config_path = _resolve_cfg(args.config)
    dataset_path = cfg["paths"].get("il_multistep_dataset_output", cfg["paths"]["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=cfg["paths"]["pkl_dataset"],
        output_path=dataset_path,
        use_cached=True,
        action_codebook_cfg=cfg.get("action_codebook", {}),
        constraint_cfg=cfg.get("constraints", {}),
    )
    selected_paths = _select_records_by_family_limit(
        step_paths,
        args.families,
        int(args.max_steps_per_family),
    )
    codebook = load_action_codebook(dataset_path)
    records = build_oracle_candidate_records(
        selected_paths,
        codebook,
        cfg,
        progress_every=int(args.progress_every),
    )
    summary = summarize_oracle_candidate_records(records, args.thresholds)
    family_counts = Counter(str(record["family"]) for record in records)
    report = {
        "phase": "phase4_il_oracle_coverage_diagnostics",
        "config": {
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "output_json": str(args.output_json),
            "families": [str(value) for value in args.families],
            "max_steps_per_family": int(args.max_steps_per_family),
            "thresholds": [float(value) for value in sorted(args.thresholds)],
        },
        "codebook": {
            "version": str(codebook.get("version", "unknown")),
            "representative_strategy": str(codebook.get("representative_strategy", "unknown")),
            "fine_bucket_policy": dict(codebook.get("fine_bucket_policy", {}) or {}),
            "num_entries": int(len(codebook.get("entries", []))),
            "num_buckets": int(len(codebook.get("bucket_to_ids", {}))),
            "bucket_sizes": {
                str(key): int(len(value))
                for key, value in sorted(codebook.get("bucket_to_ids", {}).items())
            },
        },
        "selection": {
            "num_steps": int(len(records)),
            "family_counts": dict(sorted(family_counts.items())),
            "scope": "all_cached_steps" if int(args.max_steps_per_family) <= 0 else "family_capped_steps",
        },
        **summary,
    }
    output_path = ensure_parent_dir(args.output_json)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[OK] oracle coverage diagnostics saved to {output_path}")
    print(json.dumps(report["recommendation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
