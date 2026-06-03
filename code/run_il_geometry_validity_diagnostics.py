import argparse
import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.action_codebook import (
    decode_local_dyad_code,
    family_name_from_index,
    load_action_codebook,
    resolve_codebook_bucket_for_step,
)
from src.inverse.inference_runtime import load_inverse_bundle
from src.inverse.phase4_il import (
    ensure_multistep_expert_paths,
    filter_paths_by_families,
    group_paths_by_trace,
    load_step_split,
    subset_by_indices,
)
from src.inverse.rl_env import apply_j_operator, validate_graph_structure


FAMILY_ORDER = ("6bar", "7bar", "8bar", "9bar")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose Phase4 IL geometry-code decoding, codebook buckets, and J-operator validity."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--max_traces_per_family", type=int, default=64)
    parser.add_argument("--max_examples_per_family", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
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
            "il_split_output",
            "rl_model_output",
        ),
        config_dir=resolved_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    return cfg, resolved_path


def _select_balanced_test_paths(step_paths, split, max_traces_per_family: int):
    test_paths = subset_by_indices(step_paths, split["test_indices"])
    selected = []
    selected_trace_ids = {}
    for family in FAMILY_ORDER:
        family_paths = filter_paths_by_families(test_paths, [family])
        traces = group_paths_by_trace(family_paths)
        trace_ids = {
            int(trace[0]["trace_id"])
            for trace in traces[: max(0, int(max_traces_per_family))]
        }
        selected_trace_ids[family] = sorted(trace_ids)
        selected.extend(
            item for item in family_paths if int(item["trace_id"]) in trace_ids
        )
    return sorted(selected, key=lambda item: (int(item["trace_id"]), int(item["step_index"]))), selected_trace_ids


def _reason_key(reason) -> str:
    if isinstance(reason, dict):
        return str(reason.get("reason", reason))
    return str(reason)


def _apply_code_vector(graph, topo, code_vec, constraints):
    try:
        u, v, w = [int(value) for value in topo]
        if min(u, v, w) < 0 or max(u, v, w) >= int(graph.pos.size(0)):
            return False, "topology_index_out_of_range", None
        pred_geo = decode_local_dyad_code(
            graph.pos[u].detach().cpu().numpy(),
            graph.pos[v].detach().cpu().numpy(),
            graph.pos[w].detach().cpu().numpy(),
            np.asarray(code_vec, dtype=np.float32),
        )
        next_graph = apply_j_operator(graph, u, v, w, pred_geo[0], pred_geo[1])
        is_valid, reason = validate_graph_structure(next_graph, constraints)
        return bool(is_valid), _reason_key(reason), next_graph if is_valid else None
    except Exception as exc:
        return False, type(exc).__name__, None


def _single_graph_prediction(logits: torch.Tensor, valid_mask: torch.Tensor) -> int:
    masked = logits.view(-1).masked_fill(~valid_mask.view(-1), -1e9)
    return int(torch.argmax(masked).item())


def _allowed_code_ids(policy, step, action_topo):
    family_name = family_name_from_index(int(step["family_index"]))
    step_role = "semantic" if int(step["step_role_index"]) == 1 else "aux"
    bucket = resolve_codebook_bucket_for_step(
        getattr(policy, "action_codebook_buckets", {}),
        family_name,
        step_role,
        step_index=int(step["step_index"]),
        action_topo=action_topo,
    )
    allowed = list(getattr(policy, "action_codebook_buckets", {}).get(bucket, []))
    if not allowed:
        allowed = list(range(int(policy.action_codebook.size(0))))
    return [int(idx) for idx in allowed]


def _predict_step(policy, curve_encoder, graph, step, device, top_k: int):
    batch_graph = Batch.from_data_list([copy.deepcopy(graph)]).to(device)
    target_foot = step["y_foot"].unsqueeze(0).to(device)
    target_knee = step["y_knee"].unsqueeze(0).to(device)
    target_ankle = step["y_ankle"].unsqueeze(0).to(device)
    if curve_encoder is None:
        latent_dim = int(getattr(policy, "curve_latent_dim", 128))
        z_c = torch.zeros((1, latent_dim), dtype=torch.float32, device=device)
    else:
        z_c = curve_encoder(target_foot, target_knee, target_ankle)

    x_enc = policy.encode_graph(batch_graph)
    outputs = policy.phase4_outputs(
        batch_graph,
        x_enc,
        z_c,
        family_ids=torch.tensor([int(step["family_index"])], dtype=torch.long, device=device),
        step_indices=torch.tensor([int(step["step_index"])], dtype=torch.long, device=device),
        step_counts=torch.tensor([int(step["step_count"])], dtype=torch.long, device=device),
    )
    pred_topo = [
        _single_graph_prediction(outputs["u_logits"], batch_graph.x[:, 2] <= 0.5),
        _single_graph_prediction(outputs["v_logits"], batch_graph.x[:, 2] <= 0.5),
        _single_graph_prediction(outputs["w_logits"], batch_graph.x[:, 2] > 0.5),
    ]
    logits = policy.geometry_code_logits(
        batch_graph,
        x_enc,
        outputs["graph_context"],
        torch.tensor([pred_topo], dtype=torch.long, device=device),
    )[0]
    allowed = _allowed_code_ids(policy, step, pred_topo)
    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
    allowed_logits = logits[allowed_tensor]
    order = torch.argsort(allowed_logits, descending=True)
    ordered_ids = allowed_tensor[order].detach().cpu().tolist()
    ordered_scores = allowed_logits[order].detach().cpu().tolist()
    cap = max(1, min(int(top_k), len(ordered_ids)))
    return {
        "topo": pred_topo,
        "ordered_code_ids": [int(idx) for idx in ordered_ids],
        "ordered_scores": [float(score) for score in ordered_scores],
        "topk_code_ids": [int(idx) for idx in ordered_ids[:cap]],
    }


def _empty_group():
    return {
        "count": 0,
        "truth_raw_valid": 0,
        "truth_codebook_valid": 0,
        "policy_topology_match": 0,
        "policy_top1_valid": 0,
        "policy_top3_has_valid": 0,
        "policy_top5_has_valid": 0,
        "policy_topk_has_valid": 0,
        "policy_bucket_has_valid": 0,
        "top1_invalid_but_topk_valid": 0,
        "top1_invalid_but_bucket_valid": 0,
        "top1_invalid_reason_counts": Counter(),
        "truth_codebook_invalid_reason_counts": Counter(),
        "truth_raw_invalid_reason_counts": Counter(),
        "first_valid_rank_counts": Counter(),
    }


def _bump_rate_fields(payload):
    count = max(1, int(payload["count"]))
    out = {}
    for key, value in payload.items():
        if isinstance(value, Counter):
            out[key] = dict(value)
        elif key == "count":
            out[key] = int(value)
        else:
            out[key] = int(value)
            out[f"{key}_rate"] = float(value / count)
    return out


def _update_group(group, key, value=1):
    group[key] += int(value)


def _code_vec(policy, code_id: int):
    return policy.action_codebook[int(code_id)].detach().cpu().numpy()


def _diagnose(policy, curve_encoder, paths, cfg, device, top_k: int, max_examples_per_family: int):
    constraints = cfg.get("constraints", {})
    traces = group_paths_by_trace(paths)
    groups = defaultdict(_empty_group)
    examples = []
    examples_by_family = Counter()

    policy.eval()
    if curve_encoder is not None:
        curve_encoder.eval()

    with torch.no_grad():
        for trace in traces:
            current_graph = copy.deepcopy(trace[0]["base_data"])
            for step in trace:
                family = str(step["family_id"])
                step_key = f"{family}/step{int(step['step_index'])}"
                for key in ("all", family, step_key):
                    groups[key]["count"] += 1

                truth_topo = [int(value) for value in step["action_topo"].tolist()]
                truth_code_id = int(step["action_code_id"])
                truth_raw_valid, truth_raw_reason, _ = _apply_code_vector(
                    step["base_data"], truth_topo, step["action_code_vec"].detach().cpu().numpy(), constraints
                )
                truth_book_valid, truth_book_reason, _ = _apply_code_vector(
                    step["base_data"], truth_topo, step["action_code_target"].detach().cpu().numpy(), constraints
                )

                prediction = _predict_step(policy, curve_encoder, current_graph, step, device, top_k)
                pred_topo = prediction["topo"]
                topo_match = pred_topo == truth_topo
                ordered_ids = prediction["ordered_code_ids"]
                top1_id = int(ordered_ids[0])

                valid_by_rank = []
                reason_by_rank = []
                graph_by_rank = []
                for code_id in ordered_ids:
                    is_valid, reason, next_graph = _apply_code_vector(
                        current_graph, pred_topo, _code_vec(policy, code_id), constraints
                    )
                    valid_by_rank.append(bool(is_valid))
                    reason_by_rank.append(reason)
                    graph_by_rank.append(next_graph)

                first_valid_rank = None
                for rank, is_valid in enumerate(valid_by_rank, start=1):
                    if is_valid:
                        first_valid_rank = rank
                        break

                top1_valid = bool(valid_by_rank[0])
                top3_has_valid = any(valid_by_rank[: min(3, len(valid_by_rank))])
                top5_has_valid = any(valid_by_rank[: min(5, len(valid_by_rank))])
                topk_has_valid = any(valid_by_rank[: min(int(top_k), len(valid_by_rank))])
                bucket_has_valid = any(valid_by_rank)

                for key in ("all", family, step_key):
                    group = groups[key]
                    _update_group(group, "truth_raw_valid", truth_raw_valid)
                    _update_group(group, "truth_codebook_valid", truth_book_valid)
                    _update_group(group, "policy_topology_match", topo_match)
                    _update_group(group, "policy_top1_valid", top1_valid)
                    _update_group(group, "policy_top3_has_valid", top3_has_valid)
                    _update_group(group, "policy_top5_has_valid", top5_has_valid)
                    _update_group(group, "policy_topk_has_valid", topk_has_valid)
                    _update_group(group, "policy_bucket_has_valid", bucket_has_valid)
                    _update_group(group, "top1_invalid_but_topk_valid", (not top1_valid) and topk_has_valid)
                    _update_group(group, "top1_invalid_but_bucket_valid", (not top1_valid) and bucket_has_valid)
                    if not top1_valid:
                        group["top1_invalid_reason_counts"][reason_by_rank[0]] += 1
                    if not truth_book_valid:
                        group["truth_codebook_invalid_reason_counts"][truth_book_reason] += 1
                    if not truth_raw_valid:
                        group["truth_raw_invalid_reason_counts"][truth_raw_reason] += 1
                    if first_valid_rank is not None:
                        group["first_valid_rank_counts"][str(first_valid_rank)] += 1

                if (
                    (not top1_valid or not truth_book_valid)
                    and examples_by_family[family] < int(max_examples_per_family)
                ):
                    examples.append(
                        {
                            "trace_id": int(step["trace_id"]),
                            "family": family,
                            "step_index": int(step["step_index"]),
                            "step_role": str(step["step_role"]),
                            "truth_topo": truth_topo,
                            "pred_topo": pred_topo,
                            "topology_match": bool(topo_match),
                            "truth_code_id": truth_code_id,
                            "top1_code_id": top1_id,
                            "truth_raw_valid": bool(truth_raw_valid),
                            "truth_raw_reason": truth_raw_reason,
                            "truth_codebook_valid": bool(truth_book_valid),
                            "truth_codebook_reason": truth_book_reason,
                            "top1_valid": bool(top1_valid),
                            "top1_reason": reason_by_rank[0],
                            "topk_has_valid": bool(topk_has_valid),
                            "bucket_has_valid": bool(bucket_has_valid),
                            "first_valid_rank": first_valid_rank,
                            "topk_code_ids": prediction["topk_code_ids"],
                        }
                    )
                    examples_by_family[family] += 1

                if top1_valid:
                    current_graph = graph_by_rank[0]
                else:
                    break

    return {
        "by_group": {key: _bump_rate_fields(value) for key, value in sorted(groups.items())},
        "examples": examples,
    }


def _evaluate_validity_reranked_rollout(policy, curve_encoder, paths, cfg, device, top_k: int):
    constraints = cfg.get("constraints", {})
    traces = group_paths_by_trace(paths)
    by_family = defaultdict(lambda: {
        "trace_count": 0,
        "top1_valid_trace": 0,
        "rerank_valid_trace": 0,
        "rerank_exact_trace": 0,
        "rerank_used_non_top1": 0,
        "rerank_no_valid_code": 0,
        "first_valid_rank_counts": Counter(),
    })

    policy.eval()
    if curve_encoder is not None:
        curve_encoder.eval()

    with torch.no_grad():
        for trace in traces:
            family = str(trace[0]["family_id"])
            by_family[family]["trace_count"] += 1
            current_graph = copy.deepcopy(trace[0]["base_data"])
            top1_valid_trace = True
            rerank_valid_trace = True
            rerank_exact_trace = True
            used_non_top1 = False
            no_valid_code = False

            for step in trace:
                prediction = _predict_step(policy, curve_encoder, current_graph, step, device, top_k)
                pred_topo = prediction["topo"]
                truth_topo = [int(value) for value in step["action_topo"].tolist()]
                truth_code_id = int(step["action_code_id"])

                top1_id = int(prediction["ordered_code_ids"][0])
                top1_valid, _, top1_graph = _apply_code_vector(
                    current_graph, pred_topo, _code_vec(policy, top1_id), constraints
                )
                if not top1_valid:
                    top1_valid_trace = False

                selected_rank = None
                selected_code_id = None
                selected_graph = None
                for rank, code_id in enumerate(prediction["ordered_code_ids"], start=1):
                    is_valid, _, next_graph = _apply_code_vector(
                        current_graph, pred_topo, _code_vec(policy, code_id), constraints
                    )
                    if is_valid:
                        selected_rank = rank
                        selected_code_id = int(code_id)
                        selected_graph = next_graph
                        break

                if selected_rank is None:
                    rerank_valid_trace = False
                    rerank_exact_trace = False
                    no_valid_code = True
                    break

                by_family[family]["first_valid_rank_counts"][str(selected_rank)] += 1
                if selected_rank != 1:
                    used_non_top1 = True
                rerank_exact_trace = bool(
                    rerank_exact_trace
                    and pred_topo == truth_topo
                    and int(selected_code_id) == truth_code_id
                )
                current_graph = selected_graph if selected_graph is not None else top1_graph

            by_family[family]["top1_valid_trace"] += int(top1_valid_trace)
            by_family[family]["rerank_valid_trace"] += int(rerank_valid_trace)
            by_family[family]["rerank_exact_trace"] += int(rerank_exact_trace)
            by_family[family]["rerank_used_non_top1"] += int(used_non_top1)
            by_family[family]["rerank_no_valid_code"] += int(no_valid_code)

    out = {}
    for family, payload in sorted(by_family.items()):
        trace_count = max(1, int(payload["trace_count"]))
        out[family] = {
            "trace_count": int(payload["trace_count"]),
            "top1_valid_trace_rate": float(payload["top1_valid_trace"] / trace_count),
            "rerank_valid_trace_rate": float(payload["rerank_valid_trace"] / trace_count),
            "rerank_exact_trace_rate": float(payload["rerank_exact_trace"] / trace_count),
            "rerank_used_non_top1_rate": float(payload["rerank_used_non_top1"] / trace_count),
            "rerank_no_valid_code_rate": float(payload["rerank_no_valid_code"] / trace_count),
            "first_valid_rank_counts": dict(payload["first_valid_rank_counts"]),
        }
    return out


def main():
    args = _parse_args()
    cfg, config_path = _resolve_cfg(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

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
    selected_paths, selected_trace_ids = _select_balanced_test_paths(
        step_paths, split, args.max_traces_per_family
    )

    action_codebook = load_action_codebook(dataset_path)
    bundle = load_inverse_bundle(
        cfg,
        args.checkpoint,
        device,
        allow_fresh_fallback=False,
        require_geometry_code_ready=True,
    )
    if bundle is None:
        raise RuntimeError(f"failed to load IL checkpoint: {args.checkpoint}")

    report = {
        "phase": "phase4_il_geometry_validity_diagnostics",
        "config": {
            "config_path": str(config_path),
            "checkpoint": str(args.checkpoint),
            "dataset_path": str(dataset_path),
            "device": str(device),
            "max_traces_per_family": int(args.max_traces_per_family),
            "max_examples_per_family": int(args.max_examples_per_family),
            "top_k": int(args.top_k),
            "num_codebook_entries": int(len(action_codebook.get("entries", []))),
            "bucket_sizes": {
                str(key): int(len(value))
                for key, value in sorted(action_codebook.get("bucket_to_ids", {}).items())
            },
            "selected_trace_ids": selected_trace_ids,
        },
        "diagnostics": _diagnose(
            bundle["policy"],
            bundle["curve_encoder"],
            selected_paths,
            cfg,
            device,
            int(args.top_k),
            int(args.max_examples_per_family),
        ),
        "validity_reranked_rollout": _evaluate_validity_reranked_rollout(
            bundle["policy"],
            bundle["curve_encoder"],
            selected_paths,
            cfg,
            device,
            int(args.top_k),
        ),
    }

    ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[OK] geometry validity diagnostics saved to {args.output_json}")


if __name__ == "__main__":
    main()
