import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.inverse.readout_assignment import (  # noqa: E402
        AssignmentTarget,
        LearnedChainScorerReadoutAssignment,
        ReadoutAssignmentRecord,
        RuleBasedReadoutAssignment,
        SlotPointerReadoutAssignment,
        enumerate_leg_candidates,
    )
except Exception as exc:  # pragma: no cover - runtime environment hint
    print(f"[ERROR] Failed to import real readout benchmark dependencies: {exc}")
    print("[HINT] Run this benchmark in the GMM environment.")
    print("       Example: run_gmm.cmd GraphMetaMat-LINKS\\code\\run_real_readout_benchmark.py")
    raise


DEFAULT_INPUT_PKL = WORKSPACE_ROOT / "LINKS-main" / "output" / "data_gen_v2_final80k_20260331" / "diverse_dataset_v2.pkl"
DEFAULT_SPLIT_JSON = WORKSPACE_ROOT / "LINKS-main" / "output" / "data_gen_v2_final80k_20260331" / "split_indices_v2.json"
DEFAULT_OUTPUT_JSON = WORKSPACE_ROOT / "demo" / "outputs" / "readout_assignment_fullsplit_modes_summary.json"
DEFAULT_FAILURE_JSON = WORKSPACE_ROOT / "demo" / "outputs" / "readout_assignment_fullsplit_modes_failures.json"

MODE_SPECS = {
    "graph_motion_target": {
        "label": "graph+motion+target",
        "use_motion": True,
        "use_target": True,
    },
    "graph_target": {
        "label": "graph+target",
        "use_motion": False,
        "use_target": True,
    },
    "graph_only": {
        "label": "graph_only",
        "use_motion": False,
        "use_target": False,
    },
}

SCHEME_ORDER = ("scheme_a", "scheme_b", "scheme_c")


@dataclass
class BenchmarkItem:
    record: ReadoutAssignmentRecord
    sample_id: int
    family: str
    num_nodes: int
    target_bar_count: Optional[int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-split readout assignment benchmark on real LINKS4Meta samples.")
    parser.add_argument("--input_pkl", type=Path, default=DEFAULT_INPUT_PKL)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--families", nargs="*", default=["6bar", "7bar", "8bar", "9bar"])
    parser.add_argument("--train_per_family", type=int, default=32)
    parser.add_argument("--test_per_family", type=int, default=24)
    parser.add_argument("--full_test_split", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs_b", type=int, default=60)
    parser.add_argument("--epochs_c", type=int, default=80)
    parser.add_argument("--lr_b", type=float, default=8e-3)
    parser.add_argument("--lr_c", type=float, default=8e-3)
    parser.add_argument("--eval_modes", nargs="*", default=list(MODE_SPECS.keys()))
    parser.add_argument("--schemes", nargs="*", default=list(SCHEME_ORDER))
    parser.add_argument("--max_failures_per_scheme", type=int, default=12)
    parser.add_argument("--max_candidates_cap", type=int, default=256)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--failure_json", type=Path, default=DEFAULT_FAILURE_JSON)
    return parser.parse_args()


def _load_raw_samples(path: Path) -> list[dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list dataset from {path}, got {type(data)!r}")
    return data


def _load_split_indices(path: Path, raw_samples: list[dict]) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    if not isinstance(split, dict):
        raise TypeError(f"Expected dict split file from {path}, got {type(split)!r}")

    id_to_index = {int(sample.get("id", idx)): idx for idx, sample in enumerate(raw_samples)}

    def resolve(values: list[int]) -> list[int]:
        if values and all(isinstance(v, int) and 0 <= int(v) < len(raw_samples) for v in values):
            return [int(v) for v in values]
        resolved = []
        for value in values:
            index = id_to_index.get(int(value))
            if index is not None:
                resolved.append(index)
        return resolved

    out = {}
    for key in ("train", "val", "test"):
        values = split.get(key, [])
        out[key] = resolve(list(values))
    if not out["train"] or not out["test"]:
        raise ValueError(f"Split file {path} must provide non-empty train/test entries.")
    return out


def _family_name(sample: dict) -> str:
    family = sample.get("family") or sample.get("family_id") or "unknown"
    return str(family)


def _strip_graph_from_sample(sample: dict) -> dict[str, np.ndarray]:
    adjacency = np.asarray(sample["A"])
    x0 = np.asarray(sample["x0"], dtype=np.float32)
    types = np.asarray(sample["types"])
    is_fixed = (types == 1).astype(np.float32)
    is_ground = np.zeros_like(is_fixed, dtype=np.float32)
    if is_ground.shape[0] > 0:
        is_ground[0] = 1.0
    x = np.column_stack([x0, is_fixed, is_ground]).astype(np.float32)
    edges = np.array(np.where(adjacency)).astype(np.int64)
    return {
        "x": x,
        "pos": x0.astype(np.float32),
        "edge_index": edges,
    }


def _benchmark_item_from_sample(sample: dict, sample_index: int) -> BenchmarkItem:
    analysis = sample["analysis"]
    truth = {
        "hip": int(analysis["hip"]),
        "knee": int(analysis["knee"]),
        "ankle": int(analysis["ankle"]),
        "foot": int(analysis["foot"]),
    }
    motion = np.asarray(analysis["x_sol"], dtype=np.float32)
    target = AssignmentTarget.from_motion(
        motion,
        hip=truth["hip"],
        knee=truth["knee"],
        ankle=truth["ankle"],
        foot=truth["foot"],
    )
    record = ReadoutAssignmentRecord(
        graph=_strip_graph_from_sample(sample),
        motion=motion,
        target=target,
        truth=truth,
    )
    return BenchmarkItem(
        record=record,
        sample_id=int(sample.get("id", sample_index)),
        family=_family_name(sample),
        num_nodes=int(np.asarray(sample["x0"]).shape[0]),
        target_bar_count=int(sample["target_bar_count"]) if sample.get("target_bar_count") is not None else None,
    )


def _sample_family_subset(
    raw_samples: list[dict],
    split_indices: list[int],
    *,
    families: list[str],
    per_family: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(int(seed))
    grouped: dict[str, list[int]] = defaultdict(list)
    family_set = {str(name) for name in families}
    for sample_index in split_indices:
        family = _family_name(raw_samples[sample_index])
        if family in family_set:
            grouped[family].append(int(sample_index))

    selected: list[int] = []
    for family in families:
        candidates = list(grouped.get(str(family), []))
        rng.shuffle(candidates)
        selected.extend(candidates[: min(int(per_family), len(candidates))])
    return selected


def _select_eval_indices(
    raw_samples: list[dict],
    split_indices: list[int],
    *,
    families: list[str],
    per_family: int,
    seed: int,
    full_split: bool,
) -> list[int]:
    if full_split:
        family_set = {str(name) for name in families}
        return [int(idx) for idx in split_indices if _family_name(raw_samples[idx]) in family_set]
    return _sample_family_subset(
        raw_samples,
        split_indices,
        families=families,
        per_family=per_family,
        seed=seed,
    )


def _record_inputs_for_mode(record: ReadoutAssignmentRecord, mode_key: str) -> Tuple[object, Optional[np.ndarray], Optional[AssignmentTarget]]:
    mode = MODE_SPECS[mode_key]
    motion = record.motion if mode["use_motion"] else None
    target = record.resolved_target() if mode["use_target"] else None
    return record.graph, motion, target


def _mode_train_records(items: list[BenchmarkItem], mode_key: str) -> list[ReadoutAssignmentRecord]:
    records: list[ReadoutAssignmentRecord] = []
    for item in items:
        graph, motion, target = _record_inputs_for_mode(item.record, mode_key)
        records.append(
            ReadoutAssignmentRecord(
                graph=graph,
                motion=motion,
                target=target,
                truth=dict(item.record.truth or {}),
            )
        )
    return records


def _graph_diagnostics(record: ReadoutAssignmentRecord) -> dict[str, object]:
    graph = record.graph
    x = np.asarray(graph["x"], dtype=np.float32)
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    num_nodes = int(x.shape[0])
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edge_index.T.tolist():
        adjacency[int(u)].add(int(v))
    anchors = (x[:, 2] > 0.5) | (x[:, 3] > 0.5)
    truth = record.truth or {}
    truth_foot = int(truth.get("foot", -1))
    ground_adjacent_truth_foot = bool(
        0 <= truth_foot < num_nodes and any(bool(anchors[nbr]) for nbr in adjacency[truth_foot])
    )
    return {
        "num_nodes": num_nodes,
        "num_edges_directed": int(edge_index.shape[1]),
        "ground_adjacent_truth_foot": ground_adjacent_truth_foot,
    }


def _failure_tags(item: BenchmarkItem, result) -> list[str]:
    tags: list[str] = []
    if item.family in {"8bar", "9bar"}:
        tags.append("high_bar_family")
    diagnostics = _graph_diagnostics(item.record)
    if diagnostics["ground_adjacent_truth_foot"]:
        tags.append("ground_adjacent_truth_foot")
    if result is not None and int(result.candidate_count) >= 8:
        tags.append("many_candidates")
    if result is not None and len(result.top_candidates) >= 2:
        margin = float(result.top_candidates[0].score - result.top_candidates[1].score)
        if margin < 0.15:
            tags.append("ambiguous_top2")
    return tags


def _top_candidates_payload(result, *, limit: int = 3) -> list[dict[str, object]]:
    if result is None:
        return []
    payload = []
    for candidate in result.top_candidates[:limit]:
        payload.append(
            {
                "keypoints": candidate.keypoints(),
                "path": list(candidate.path),
                "score": float(candidate.score),
                "score_breakdown": dict(candidate.score_breakdown),
            }
        )
    return payload


def _candidate_count_summary(counts: list[int], *, max_candidates_cap: int) -> dict[str, float]:
    if not counts:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "reaches_cap_rate": 0.0,
        }
    array = np.asarray(counts, dtype=np.float32)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "reaches_cap_rate": float(np.mean(array >= float(max_candidates_cap))),
    }


def _mode_candidate_metrics(
    items: list[BenchmarkItem],
    *,
    mode_key: str,
    max_candidates_cap: int,
) -> dict[str, object]:
    coverage = 0
    candidate_counts: list[int] = []
    by_family: dict[str, dict[str, object]] = defaultdict(lambda: {
        "count": 0,
        "coverage": 0,
        "candidate_counts": [],
    })

    for item in items:
        graph, motion, target = _record_inputs_for_mode(item.record, mode_key)
        candidates = enumerate_leg_candidates(graph, motion=motion, target=target)
        candidate_count = int(len(candidates))
        candidate_counts.append(candidate_count)

        truth = item.record.truth or {}
        covered = int(any(candidate.keypoints() == truth for candidate in candidates))
        coverage += covered

        family_stats = by_family[item.family]
        family_stats["count"] += 1
        family_stats["coverage"] += covered
        family_stats["candidate_counts"].append(candidate_count)

    denom = max(1, len(items))
    by_family_out = {}
    for family, values in sorted(by_family.items()):
        family_denom = max(1, int(values["count"]))
        by_family_out[family] = {
            "count": int(values["count"]),
            "truth_candidate_coverage_rate": float(values["coverage"] / family_denom),
            "candidate_count": _candidate_count_summary(values["candidate_counts"], max_candidates_cap=max_candidates_cap),
        }

    no_assignment_ratio = float(np.mean(np.asarray(candidate_counts, dtype=np.float32) == 0.0)) if candidate_counts else 0.0
    return {
        "truth_candidate_coverage_rate": float(coverage / denom),
        "candidate_count": _candidate_count_summary(candidate_counts, max_candidates_cap=max_candidates_cap),
        "no_assignment_ratio": no_assignment_ratio,
        "by_family": by_family_out,
    }


def _evaluate_detailed(
    module,
    items: list[BenchmarkItem],
    *,
    mode_key: str,
    max_failures: int,
    max_candidates_cap: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    exact = 0
    knee = 0
    ankle = 0
    foot = 0
    assigned = 0
    score_total = 0.0
    by_family_counts: dict[str, dict[str, float]] = defaultdict(lambda: {
        "count": 0.0,
        "exact": 0.0,
        "knee": 0.0,
        "ankle": 0.0,
        "foot": 0.0,
        "candidate_counts": [],
    })
    failures: list[dict[str, object]] = []
    failure_buckets: Counter[str] = Counter()

    for item in items:
        graph, motion, target = _record_inputs_for_mode(item.record, mode_key)
        result = module.assign(graph, motion=motion, target=target)
        family_stats = by_family_counts[item.family]
        family_stats["count"] += 1.0

        if result is None:
            family_stats["candidate_counts"].append(0)
            failure_tags = ["no_assignment"] + _failure_tags(item, None)
            for tag in failure_tags:
                failure_buckets[tag] += 1
            if len(failures) < int(max_failures):
                failure = {
                    "sample_id": item.sample_id,
                    "family": item.family,
                    "num_nodes": item.num_nodes,
                    "target_bar_count": item.target_bar_count,
                    "truth": dict(item.record.truth or {}),
                    "prediction": None,
                    "candidate_count": 0,
                    "tags": failure_tags,
                    "top_candidates": [],
                }
                failures.append(failure)
            continue

        assigned += 1
        family_stats["candidate_counts"].append(int(result.candidate_count))
        truth = item.record.truth or {}
        hit_exact = (
            result.keypoints.get("knee") == truth.get("knee")
            and result.keypoints.get("ankle") == truth.get("ankle")
            and result.keypoints.get("foot") == truth.get("foot")
        )
        hit_knee = result.keypoints.get("knee") == truth.get("knee")
        hit_ankle = result.keypoints.get("ankle") == truth.get("ankle")
        hit_foot = result.keypoints.get("foot") == truth.get("foot")

        exact += int(hit_exact)
        knee += int(hit_knee)
        ankle += int(hit_ankle)
        foot += int(hit_foot)
        score_total += float(result.score)

        family_stats["exact"] += float(hit_exact)
        family_stats["knee"] += float(hit_knee)
        family_stats["ankle"] += float(hit_ankle)
        family_stats["foot"] += float(hit_foot)

        if not hit_exact and len(failures) < int(max_failures):
            failure_tags = _failure_tags(item, result)
            for tag in failure_tags:
                failure_buckets[tag] += 1
            failure = {
                "sample_id": item.sample_id,
                "family": item.family,
                "num_nodes": item.num_nodes,
                "target_bar_count": item.target_bar_count,
                "truth": dict(truth),
                "prediction": dict(result.keypoints),
                "candidate_count": int(result.candidate_count),
                "tags": failure_tags,
                "top_candidates": _top_candidates_payload(result),
            }
            failures.append(failure)
        elif not hit_exact:
            for tag in _failure_tags(item, result):
                failure_buckets[tag] += 1

    denom = max(1, len(items))
    by_family = {}
    for family, values in sorted(by_family_counts.items()):
        family_denom = max(1.0, values["count"])
        by_family[family] = {
            "count": int(values["count"]),
            "exact_chain_accuracy": float(values["exact"] / family_denom),
            "knee_accuracy": float(values["knee"] / family_denom),
            "ankle_accuracy": float(values["ankle"] / family_denom),
            "foot_accuracy": float(values["foot"] / family_denom),
            "candidate_count": _candidate_count_summary(values["candidate_counts"], max_candidates_cap=max_candidates_cap),
        }

    summary = {
        "assignment_rate": float(assigned / denom),
        "exact_chain_accuracy": float(exact / denom),
        "knee_accuracy": float(knee / denom),
        "ankle_accuracy": float(ankle / denom),
        "foot_accuracy": float(foot / denom),
        "mean_assignment_score": float(score_total / denom),
        "no_assignment_ratio": float(1.0 - (assigned / denom)),
        "by_family": by_family,
        "num_failures_saved": int(len(failures)),
        "failure_bucket_counts": dict(sorted(failure_buckets.items())),
    }
    return summary, failures


def _truth_semantic_stats(items: list[BenchmarkItem]) -> dict[str, object]:
    totals = {
        "count": len(items),
        "hip_fixed": 0,
        "hip_ground": 0,
        "foot_adjacent_anchor": 0,
    }
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {
        "count": 0,
        "hip_fixed": 0,
        "hip_ground": 0,
        "foot_adjacent_anchor": 0,
    })

    for item in items:
        graph = item.record.graph
        x = np.asarray(graph["x"], dtype=np.float32)
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
        adjacency = [set() for _ in range(int(x.shape[0]))]
        for u, v in edge_index.T.tolist():
            adjacency[int(u)].add(int(v))
        anchors = (x[:, 2] > 0.5) | (x[:, 3] > 0.5)
        truth = item.record.truth or {}
        hip = int(truth.get("hip", -1))
        foot = int(truth.get("foot", -1))
        hip_fixed = int(0 <= hip < x.shape[0] and x[hip, 2] > 0.5)
        hip_ground = int(hip == 0)
        foot_adjacent = int(0 <= foot < x.shape[0] and any(bool(anchors[nbr]) for nbr in adjacency[foot]))

        totals["hip_fixed"] += hip_fixed
        totals["hip_ground"] += hip_ground
        totals["foot_adjacent_anchor"] += foot_adjacent

        family_stats = by_family[item.family]
        family_stats["count"] += 1
        family_stats["hip_fixed"] += hip_fixed
        family_stats["hip_ground"] += hip_ground
        family_stats["foot_adjacent_anchor"] += foot_adjacent

    def normalize(values: dict[str, int]) -> dict[str, float]:
        denom = max(1, int(values["count"]))
        return {
            "count": int(values["count"]),
            "hip_fixed_rate": float(values["hip_fixed"] / denom),
            "hip_ground_rate": float(values["hip_ground"] / denom),
            "foot_adjacent_anchor_rate": float(values["foot_adjacent_anchor"] / denom),
        }

    return {
        "overall": normalize(totals),
        "by_family": {family: normalize(values) for family, values in sorted(by_family.items())},
    }


def main() -> None:
    args = _parse_args()
    mode_keys = [mode for mode in args.eval_modes if mode in MODE_SPECS]
    if not mode_keys:
        raise ValueError("No valid eval modes were provided.")
    scheme_names = [scheme for scheme in args.schemes if scheme in SCHEME_ORDER]
    if not scheme_names:
        raise ValueError("No valid schemes were provided.")

    raw_samples = _load_raw_samples(args.input_pkl)
    split = _load_split_indices(args.split_json, raw_samples)

    train_indices = _sample_family_subset(
        raw_samples,
        split["train"],
        families=[str(name) for name in args.families],
        per_family=int(args.train_per_family),
        seed=int(args.seed),
    )
    test_indices = _select_eval_indices(
        raw_samples,
        split["test"],
        families=[str(name) for name in args.families],
        per_family=int(args.test_per_family),
        seed=int(args.seed) + 997,
        full_split=bool(args.full_test_split),
    )

    train_items = [_benchmark_item_from_sample(raw_samples[idx], idx) for idx in train_indices]
    test_items = [_benchmark_item_from_sample(raw_samples[idx], idx) for idx in test_indices]

    summary = {
        "config": {
            "input_pkl": str(args.input_pkl),
            "split_json": str(args.split_json),
            "families": [str(name) for name in args.families],
            "train_per_family": int(args.train_per_family),
            "test_per_family": int(args.test_per_family),
            "full_test_split": bool(args.full_test_split),
            "seed": int(args.seed),
            "epochs_b": int(args.epochs_b),
            "epochs_c": int(args.epochs_c),
            "lr_b": float(args.lr_b),
            "lr_c": float(args.lr_c),
            "eval_modes": mode_keys,
            "schemes": scheme_names,
            "max_candidates_cap": int(args.max_candidates_cap),
        },
        "dataset": {
            "raw_samples": len(raw_samples),
            "train_records": len(train_items),
            "test_records": len(test_items),
            "train_family_counts": dict(sorted(Counter(item.family for item in train_items).items())),
            "test_family_counts": dict(sorted(Counter(item.family for item in test_items).items())),
            "test_truth_semantic_stats": _truth_semantic_stats(test_items),
        },
        "train": {},
        "eval": {},
    }
    failures: dict[str, dict[str, list[dict[str, object]]]] = {}

    for mode_key in mode_keys:
        train_records = _mode_train_records(train_items, mode_key)
        scheme_a = RuleBasedReadoutAssignment()
        scheme_b = LearnedChainScorerReadoutAssignment()
        scheme_c = SlotPointerReadoutAssignment()

        train_summary = {}
        if "scheme_b" in scheme_names:
            train_summary["scheme_b"] = scheme_b.fit(
                train_records,
                epochs=int(args.epochs_b),
                lr=float(args.lr_b),
                seed=int(args.seed),
            )
        if "scheme_c" in scheme_names:
            train_summary["scheme_c"] = scheme_c.fit(
                train_records,
                epochs=int(args.epochs_c),
                lr=float(args.lr_c),
                seed=int(args.seed),
            )

        mode_eval = {
            "mode_label": MODE_SPECS[mode_key]["label"],
            "candidate_space": _mode_candidate_metrics(
                test_items,
                mode_key=mode_key,
                max_candidates_cap=int(args.max_candidates_cap),
            ),
        }
        mode_failures = {}

        modules = {
            "scheme_a": scheme_a,
            "scheme_b": scheme_b,
            "scheme_c": scheme_c,
        }
        for scheme_name in scheme_names:
            module = modules[scheme_name]
            metrics, scheme_failures = _evaluate_detailed(
                module,
                test_items,
                mode_key=mode_key,
                max_failures=int(args.max_failures_per_scheme),
                max_candidates_cap=int(args.max_candidates_cap),
            )
            mode_eval[scheme_name] = metrics
            mode_failures[scheme_name] = scheme_failures

        summary["train"][mode_key] = train_summary
        summary["eval"][mode_key] = mode_eval
        failures[mode_key] = mode_failures

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.failure_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.failure_json.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {args.output_json}")
    print(f"Saved failures to: {args.failure_json}")


if __name__ == "__main__":
    main()
