"""Surrogate ranking quality diagnostic (Experiment B1).

For each test sample, enumerate all candidate leg chains, score each with
the frozen forward surrogate, and compare against an oracle (exact match
with ground-truth readout).  Reports Spearman rho, Kendall tau, NDCG@k,
and truth candidate rank per family.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inverse.readout_assignment import (  # noqa: E402
    AssignmentTarget,
    CandidateLegChain,
    _assignment_target_to_tensor_dict,
    _candidate_surrogate_data,
    enumerate_leg_candidates,
)
from src.inverse.experiment_utils import compute_joint_metrics_batch  # noqa: E402
from src.inverse.rl_env import load_frozen_surrogate  # noqa: E402

# ---------- CLI ----------

DEFAULT_INPUT_PKL = WORKSPACE_ROOT / "LINKS-main" / "output" / "data_gen_v2_final80k_20260331" / "diverse_dataset_v2.pkl"
DEFAULT_SPLIT_JSON = WORKSPACE_ROOT / "LINKS-main" / "output" / "data_gen_v2_final80k_20260331" / "split_indices_v2.json"
DEFAULT_INVERSE_CONFIG = PROJECT_ROOT / "src" / "config_inverse.yaml"
DEFAULT_OUTPUT_JSON = WORKSPACE_ROOT / "demo" / "outputs" / "readout_final_v4_B_ranking_quality.json"

FAMILY_NAME_TO_ID = {"6bar": 0, "7bar": 1, "8bar": 2, "9bar": 3}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surrogate ranking quality diagnostic (B1).")
    parser.add_argument("--input_pkl", type=Path, default=DEFAULT_INPUT_PKL)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--families", nargs="*", default=["6bar", "7bar", "8bar", "9bar"])
    parser.add_argument("--full_test_split", action="store_true")
    parser.add_argument("--test_per_family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max_candidates_cap", type=int, default=512)
    parser.add_argument("--surrogate_batch_size", type=int, default=64)
    parser.add_argument("--config_inverse", type=Path, default=DEFAULT_INVERSE_CONFIG)
    parser.add_argument("--forward_model_path", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


# ---------- helpers ----------

def _family_name(sample: dict) -> str:
    return str(sample.get("family") or sample.get("family_id") or "unknown")


def _load_raw_samples(path: Path) -> list[dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list from {path}, got {type(data)!r}")
    return data


def _load_split_indices(path: Path, raw_samples: list[dict]) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    id_to_index = {int(s.get("id", i)): i for i, s in enumerate(raw_samples)}

    def resolve(values):
        out = []
        for v in values:
            v = int(v)
            if 0 <= v < len(raw_samples):
                out.append(v)
            elif v in id_to_index:
                out.append(id_to_index[v])
        return out

    return {k: resolve(split.get(k, [])) for k in ("train", "val", "test")}


def _resolve_path(path_value, *, base_dir):
    p = Path(path_value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _load_surrogate(args):
    with args.config_inverse.open("r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
    paths = dict(cfg.get("paths", {}))
    src_dir = args.config_inverse.parent
    if args.forward_model_path is not None:
        model_path = Path(args.forward_model_path)
    else:
        model_path = _resolve_path(paths["forward_model"], base_dir=src_dir)
    config_forward = _resolve_path(paths["config_forward"], base_dir=src_dir)
    device = torch.device(args.device)
    surrogate, _ = load_frozen_surrogate(str(model_path), str(config_forward), device)
    reward_cfg = dict(cfg.get("reward", {}))
    return surrogate, device, reward_cfg


def _graph_arrays(sample: dict):
    A = np.asarray(sample["A"])
    x0 = np.asarray(sample["x0"], dtype=np.float32)
    types = np.asarray(sample["types"])
    is_fixed = (types == 1).astype(np.float32)
    is_ground = np.zeros_like(is_fixed, dtype=np.float32)
    if is_ground.shape[0] > 0:
        is_ground[0] = 1.0
    x = np.column_stack([x0, is_fixed, is_ground]).astype(np.float32)
    pos = x0.copy()
    edge_index = np.array(np.nonzero(A), dtype=np.int64)
    return x, pos, edge_index


def _truth_keypoints(sample: dict) -> dict[str, int]:
    a = sample["analysis"]
    return {"hip": int(a["hip"]), "knee": int(a["knee"]), "ankle": int(a["ankle"]), "foot": int(a["foot"])}


def _target_from_sample(sample: dict) -> AssignmentTarget:
    a = sample["analysis"]
    truth = _truth_keypoints(sample)
    motion = np.asarray(a["x_sol"], dtype=np.float32)
    return AssignmentTarget.from_motion(
        motion, hip=truth["hip"], knee=truth["knee"], ankle=truth["ankle"], foot=truth["foot"],
    )


# ---------- ranking metrics ----------

def _spearman_rho(values_a: list[float], values_b: list[float]) -> float:
    """Spearman rank correlation (manual, no scipy dependency)."""
    n = len(values_a)
    if n < 3:
        return float("nan")
    rank_a = _rank(values_a)
    rank_b = _rank(values_b)
    d = [ra - rb for ra, rb in zip(rank_a, rank_b)]
    return float(1.0 - 6.0 * sum(di * di for di in d) / (n * (n * n - 1)))


def _kendall_tau(values_a: list[float], values_b: list[float]) -> float:
    """Kendall tau-b (manual)."""
    n = len(values_a)
    if n < 3:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = np.sign(values_a[i] - values_a[j])
            sb = values_b[i] - values_b[j]
            if sa * sb > 0:
                concordant += 1
            elif sa * sb < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)


def _rank(values: list[float]) -> list[float]:
    """Assign average ranks (lower value = rank 1)."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _ndcg_at_k(oracle_scores: list[float], surrogate_scores: list[float], k: int) -> float:
    """NDCG@k: oracle_scores are 1 for exact-match, 0 otherwise."""
    n = len(oracle_scores)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    # DCG: sort by surrogate_scores descending
    order = sorted(range(n), key=lambda i: surrogate_scores[i], reverse=True)
    dcg = sum(oracle_scores[order[i]] / np.log2(i + 2) for i in range(k))
    # ideal DCG: sort by oracle_scores descending
    ideal_order = sorted(range(n), key=lambda i: oracle_scores[i], reverse=True)
    idcg = sum(oracle_scores[ideal_order[i]] / np.log2(i + 2) for i in range(k))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


# ---------- main logic ----------

def _score_candidates_with_surrogate(
    candidates: list[CandidateLegChain],
    graph_dict: dict,
    surrogate,
    reward_cfg: dict,
    device,
    target_tensor_dict: dict[str, torch.Tensor],
    family_index: int,
    batch_size: int,
) -> list[float]:
    """Return joint_score (lower = better) for each candidate."""
    scores: list[float] = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        data_list = [
            _candidate_surrogate_data(
                graph_dict, c, family_index=family_index,
            )
            for c in chunk
        ]
        batch = Batch.from_data_list(data_list).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch)
        metrics = compute_joint_metrics_batch(
            pred_foot.cpu(), pred_knee.cpu(), pred_ankle.cpu(),
            target_tensor_dict, reward_cfg,
        )
        scores.extend(float(score.item()) for score in metrics["joint_score"][:len(chunk)])
    return scores


def _graph_dict_from_sample(sample: dict) -> dict:
    x, pos, edge_index = _graph_arrays(sample)
    return {"x": x, "pos": pos, "edge_index": edge_index}


class _GraphProxy:
    """Minimal object exposing arrays as attributes for enumerate_leg_candidates."""

    def __init__(self, d: dict):
        self.x = d["x"]
        self.pos = d["pos"]
        self.edge_index = d["edge_index"]
        # carry over optional fields
        for k in ("family_id", "step_context"):
            if k in d:
                setattr(self, k, d[k])


def run_one_sample(
    sample: dict,
    surrogate,
    reward_cfg: dict,
    device,
    max_candidates: int,
    batch_size: int,
) -> dict | None:
    truth = _truth_keypoints(sample)
    target = _target_from_sample(sample)
    graph_dict = _graph_dict_from_sample(sample)
    graph_obj = _GraphProxy(graph_dict)

    family_name = _family_name(sample)
    family_index = FAMILY_NAME_TO_ID.get(family_name, -1)

    candidates = enumerate_leg_candidates(
        graph_obj, target=target, max_candidates=max_candidates,
    )
    if not candidates:
        return None

    target_tensor_dict = _assignment_target_to_tensor_dict(target)
    surrogate_scores = _score_candidates_with_surrogate(
        candidates, graph_dict, surrogate, reward_cfg, device,
        target_tensor_dict, family_index, batch_size,
    )

    # oracle: exact match
    oracle_scores = []
    for c in candidates:
        match = (
            c.knee == truth["knee"]
            and c.ankle == truth["ankle"]
            and c.foot == truth["foot"]
        )
        oracle_scores.append(1.0 if match else 0.0)

    truth_in_candidates = any(o > 0.5 for o in oracle_scores)
    truth_rank = None
    if truth_in_candidates:
        # sort by surrogate score ascending (lower joint_score = better = rank 1)
        order = sorted(range(len(candidates)), key=lambda i: surrogate_scores[i])
        for rank_idx, idx in enumerate(order):
            if oracle_scores[idx] > 0.5:
                truth_rank = rank_idx + 1
                break

    # surrogate scores are "lower = better", negate for NDCG/Spearman/Kendall
    # (these metrics expect "higher = better" input)
    neg_surrogate = [-s for s in surrogate_scores]

    rho = _spearman_rho(oracle_scores, neg_surrogate)
    tau = _kendall_tau(oracle_scores, neg_surrogate)
    ndcg1 = _ndcg_at_k(oracle_scores, neg_surrogate, 1)
    ndcg3 = _ndcg_at_k(oracle_scores, neg_surrogate, 3)
    ndcg5 = _ndcg_at_k(oracle_scores, neg_surrogate, 5)

    return {
        "family": family_name,
        "candidate_count": len(candidates),
        "truth_in_candidates": truth_in_candidates,
        "truth_rank": truth_rank,
        "spearman_rho": rho if not np.isnan(rho) else None,
        "kendall_tau": tau if not np.isnan(tau) else None,
        "ndcg_at_1": ndcg1,
        "ndcg_at_3": ndcg3,
        "ndcg_at_5": ndcg5,
    }


def main() -> None:
    args = _parse_args()
    raw_samples = _load_raw_samples(args.input_pkl)
    split = _load_split_indices(args.split_json, raw_samples)

    family_set = set(str(f) for f in args.families)

    # select test indices
    rng = np.random.RandomState(args.seed)
    test_by_family: dict[str, list[int]] = defaultdict(list)
    for idx in split["test"]:
        fam = _family_name(raw_samples[idx])
        if fam in family_set:
            test_by_family[fam].append(idx)

    test_indices = []
    for fam in sorted(test_by_family):
        indices = test_by_family[fam]
        if args.full_test_split:
            test_indices.extend(indices)
        else:
            rng.shuffle(indices)
            test_indices.extend(indices[: args.test_per_family])

    print(f"Loading surrogate ...")
    surrogate, device, reward_cfg = _load_surrogate(args)

    results = []
    family_stats: dict[str, list[dict]] = defaultdict(list)

    print(f"Evaluating {len(test_indices)} samples ...")
    for i, idx in enumerate(test_indices):
        sample = raw_samples[idx]
        res = run_one_sample(
            sample, surrogate, reward_cfg, device,
            max_candidates=int(args.max_candidates_cap),
            batch_size=int(args.surrogate_batch_size),
        )
        if res is not None:
            res["sample_index"] = int(idx)
            results.append(res)
            family_stats[res["family"]].append(res)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_indices)} done")

    # aggregate
    def _agg(items: list[dict]) -> dict:
        n = len(items)
        if n == 0:
            return {"count": 0}
        truth_cov = sum(1 for r in items if r["truth_in_candidates"]) / n
        ranks = [r["truth_rank"] for r in items if r["truth_rank"] is not None]
        rhos = [r["spearman_rho"] for r in items if r["spearman_rho"] is not None]
        taus = [r["kendall_tau"] for r in items if r["kendall_tau"] is not None]
        nd1 = [r["ndcg_at_1"] for r in items]
        nd3 = [r["ndcg_at_3"] for r in items]
        nd5 = [r["ndcg_at_5"] for r in items]
        # top-k: truth rank <= k
        top1 = sum(1 for r in ranks if r <= 1) / len(ranks) if ranks else 0.0
        top3 = sum(1 for r in ranks if r <= 3) / len(ranks) if ranks else 0.0
        top5 = sum(1 for r in ranks if r <= 5) / len(ranks) if ranks else 0.0
        # MRR: mean reciprocal rank
        mrr = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0
        return {
            "count": n,
            "truth_candidate_coverage": truth_cov,
            "mean_truth_rank": float(np.mean(ranks)) if ranks else None,
            "median_truth_rank": float(np.median(ranks)) if ranks else None,
            "top_1": top1,
            "top_3": top3,
            "top_5": top5,
            "mrr": mrr,
            "mean_spearman_rho": float(np.mean(rhos)) if rhos else None,
            "mean_kendall_tau": float(np.mean(taus)) if taus else None,
            "mean_ndcg_at_1": float(np.mean(nd1)),
            "mean_ndcg_at_3": float(np.mean(nd3)),
            "mean_ndcg_at_5": float(np.mean(nd5)),
            "mean_candidate_count": float(np.mean([r["candidate_count"] for r in items])),
        }

    summary = {
        "config": {
            "families": list(args.families),
            "full_test_split": bool(args.full_test_split),
            "max_candidates_cap": int(args.max_candidates_cap),
            "seed": int(args.seed),
            "num_evaluated": len(results),
        },
        "overall": _agg(results),
        "per_family": {fam: _agg(items) for fam, items in sorted(family_stats.items())},
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
