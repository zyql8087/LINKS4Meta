from __future__ import annotations

import copy
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax
from tqdm import tqdm

from src.inverse.action_codebook import (
    CODEBOOK_VERSION,
    allowed_code_ids_for_context,
    attach_action_codebook,
    build_action_codebook,
    codebook_bucket_for_step,
    decode_local_dyad_code,
    default_action_codebook_path,
    encode_local_dyad_code,
    family_name_from_index,
    load_action_codebook,
    resolve_codebook_bucket_for_step,
    save_action_codebook,
)
from src.inverse.rl_env import apply_j_operator, validate_graph_structure
from src.inverse.train_il import _batch_offsets
from src.kinematics_extract import extract_kinematics


FAMILY_ORDER = ("6bar", "7bar", "8bar", "9bar")
FAMILY_TO_INDEX = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
STEP_ROLE_TO_INDEX = {"aux": 0, "semantic": 1}
INDEX_TO_STEP_ROLE = {idx: name for name, idx in STEP_ROLE_TO_INDEX.items()}
DEFAULT_CURRICULUM_STAGES = (
    {"name": "stage_a_single_step", "families": ["6bar", "7bar"]},
    {"name": "stage_b_two_step", "families": ["8bar", "9bar"]},
    {"name": "stage_c_family_conditioned", "families": list(FAMILY_ORDER)},
)


def family_name_to_index(name: str) -> int:
    return FAMILY_TO_INDEX.get(str(name), len(FAMILY_ORDER))


def _curve_tensor(sample: dict, key: str, fallback_idx: int) -> torch.Tensor:
    value = sample.get(key)
    if value is not None:
        return torch.as_tensor(value, dtype=torch.float32)

    curves = extract_kinematics(sample)
    return torch.as_tensor(curves[fallback_idx], dtype=torch.float32)


def _analysis_knee_idx(sample: dict) -> int | None:
    analysis = sample.get("analysis") or {}
    if "knee" in analysis:
        return int(analysis["knee"])
    semantic = sample.get("gen_info") or analysis.get("gen_info")
    if semantic is not None and "u" in semantic:
        return int(semantic["u"])
    return None


def _base_nodes_before_step(total_nodes: int, trace: Sequence[dict], step_index: int) -> list[int]:
    removed_nodes: set[int] = set()
    for step in trace[step_index:]:
        removed_nodes.add(int(step["n1"]))
        removed_nodes.add(int(step["n2"]))
    return [node_idx for node_idx in range(total_nodes) if node_idx not in removed_nodes]


def _build_step_base_graph(sample: dict, step_index: int):
    A = np.asarray(sample["A"])
    x0 = np.asarray(sample["x0"], dtype=np.float32)
    types = np.asarray(sample["types"])
    trace = list(sample.get("generation_trace") or [])

    base_nodes = _base_nodes_before_step(A.shape[0], trace, step_index)
    node_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(base_nodes)}

    A_base = A[np.ix_(base_nodes, base_nodes)]
    x0_base = x0[base_nodes]
    types_base = types[base_nodes]
    rows, cols = np.where(A_base)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

    is_fixed = (types_base == 1).astype(np.float32)
    is_grounded = np.zeros_like(is_fixed)
    if is_grounded.size > 0:
        is_grounded[0] = 1.0
    x_feat = np.column_stack([x0_base, is_fixed, is_grounded])

    base_data = Data(
        x=torch.tensor(x_feat, dtype=torch.float32),
        pos=torch.tensor(x0_base, dtype=torch.float32),
        edge_index=edge_index,
    )

    knee_idx = _analysis_knee_idx(sample)
    if knee_idx is not None and knee_idx in node_remap:
        base_data.knee_idx = torch.tensor([node_remap[knee_idx]], dtype=torch.long)
    else:
        base_data.knee_idx = torch.tensor([-1], dtype=torch.long)

    return base_data, node_remap


def _default_anchor_pair_mask(base_data: Data) -> torch.Tensor:
    moving_mask = (base_data.x[:, 2] <= 0.5)
    pair_mask = moving_mask[:, None] & moving_mask[None, :]
    pair_mask.fill_diagonal_(False)
    return pair_mask


def _step_semantic_mask(base_data: Data, action_topo: torch.Tensor, step_role: str) -> torch.Tensor:
    mask = torch.zeros(base_data.x.size(0), dtype=torch.bool)
    if str(step_role) == "semantic":
        mask[action_topo.long()] = True
    return mask


def _attach_doc_masks(
    step_paths: Sequence[dict[str, object]],
    codebook: dict[str, object],
) -> list[dict[str, object]]:
    updated = []
    num_codes = max(1, len(codebook.get("entries", [])))
    for item in step_paths:
        new_item = dict(item)
        base_data = item["base_data"]
        anchor_mask = (base_data.x[:, 2] <= 0.5).clone().to(torch.bool)
        pair_mask = _default_anchor_pair_mask(base_data)
        anchor_i = int(item["action_anchor_i"])
        anchor_j = int(item["action_anchor_j"])
        if str(item["step_role"]) == "semantic":
            pair_mask = torch.zeros_like(pair_mask)
            pair_mask[anchor_i, anchor_j] = True
        geom_mask = torch.zeros(num_codes, dtype=torch.bool)
        allowed_ids = allowed_code_ids_for_context(
            codebook,
            family_name=str(item["family_id"]),
            step_role=str(item["step_role"]),
            step_index=int(item["step_index"]),
            action_topo=item["action_topo"],
            topology_signature=item.get("topology_signature"),
        )
        if allowed_ids:
            geom_mask[torch.tensor(allowed_ids, dtype=torch.long)] = True
        else:
            geom_mask[:] = True
        semantic_mask = _step_semantic_mask(base_data, item["action_topo"], str(item["step_role"]))

        new_item["trace_prefix"] = base_data
        new_item["seed_graph"] = item.get("seed_graph", item["trace_seed_graph"])
        new_item["target_curves"] = {
            "foot": item["y_foot"].clone(),
            "knee": item["y_knee"].clone(),
            "ankle": item["y_ankle"].clone(),
        }
        new_item["action_anchor_i"] = anchor_i
        new_item["action_anchor_j"] = anchor_j
        new_item["action_geom_code_id"] = int(item["action_code_id"])
        new_item["valid_anchor_mask"] = anchor_mask
        new_item["valid_pair_mask"] = pair_mask
        new_item["valid_geom_mask"] = geom_mask
        new_item["semantic_mask"] = semantic_mask
        updated.append(new_item)
    return updated


def multistep_paths_have_phase4_format(paths) -> bool:
    if not paths:
        return False
    sample = paths[0]
    required = {
        "raw_index",
        "trace_id",
        "step_index",
        "step_count",
        "stop_token",
        "family_index",
        "step_role_index",
        "action_code_id",
        "action_code_vec",
    }
    return required.issubset(sample.keys())


def _action_codebook_cfg(codebook_cfg: dict | None) -> dict[str, object]:
    cfg = dict(codebook_cfg or {})
    fine_bucket_policy = dict(cfg.get("fine_buckets", {}))
    return {
        "version": str(cfg.get("version", CODEBOOK_VERSION)),
        "cluster_radius": float(cfg.get("cluster_radius", 0.075)),
        "max_codes_per_bucket": int(cfg.get("max_codes_per_bucket", 24)),
        "max_codes_per_bucket_overrides": dict(cfg.get("max_codes_per_bucket_overrides", {})),
        "representative_strategy": str(cfg.get("representative_strategy", "validity_best_medoid")),
        "max_validation_candidates": int(cfg.get("max_validation_candidates", 32)),
        "max_validation_contexts": int(cfg.get("max_validation_contexts", 96)),
        "fine_bucket_policy": fine_bucket_policy,
    }


def codebook_cache_is_stale(
    expert_paths: Sequence[dict[str, object]],
    codebook: dict[str, object],
    *,
    required_version: str = CODEBOOK_VERSION,
) -> bool:
    if not multistep_paths_have_phase4_format(expert_paths):
        return True
    if str(codebook.get("version", "")) != str(required_version):
        return True
    if str(codebook.get("representative_strategy", "")) != "validity_best_medoid":
        return True
    bucket_to_ids = dict(codebook.get("bucket_to_ids", {}))
    if len(bucket_to_ids) <= 4:
        return True
    has_9bar_semantic = any(
        str(item.get("family_id")) == "9bar" and str(item.get("step_role")) == "semantic"
        for item in expert_paths
    )
    if has_9bar_semantic and not any(
        str(bucket).startswith("semantic_9bar_step") and "_topo_" in str(bucket)
        for bucket in bucket_to_ids
    ):
        return True
    return False


def extract_multistep_expert_paths(
    pkl_path: str,
    output_path: str,
    *,
    action_codebook_cfg: dict | None = None,
    constraint_cfg: dict | None = None,
):
    print(f"Loading multistep IL pkl from {pkl_path} ...")
    with open(pkl_path, "rb") as handle:
        raw_data = pickle.load(handle)
    print(f"Loaded {len(raw_data)} raw samples.")
    codebook_cfg = _action_codebook_cfg(action_codebook_cfg)
    fine_bucket_policy = dict(codebook_cfg.get("fine_bucket_policy", {}))
    fine_buckets_enabled = bool(fine_bucket_policy.get("enabled", True))

    expert_paths: list[dict[str, object]] = []
    errors = 0
    error_examples: list[str] = []

    for raw_idx, sample in enumerate(tqdm(raw_data, desc="Extracting multistep IL expert paths")):
        try:
            trace = list(sample.get("generation_trace") or [])
            if not trace:
                errors += 1
                continue

            family_name = str(sample.get("family_id") or sample.get("family") or "unknown")
            family_index = family_name_to_index(family_name)
            y_foot = _curve_tensor(sample, "foot_curve", 0)
            y_knee = _curve_tensor(sample, "knee_curve", 1)
            y_ankle = _curve_tensor(sample, "ankle_curve", 2)
            step_count = int(sample.get("step_count", len(trace)))
            sample_id = int(sample.get("id", sample.get("sample_id", raw_idx)))
            seed_graph, _ = _build_step_base_graph(sample, 0)

            for step_index, step in enumerate(trace):
                base_data, node_remap = _build_step_base_graph(sample, step_index)
                u = int(step["u"])
                v = int(step["v"])
                w = int(step["w"])
                n1 = int(step["n1"])
                n2 = int(step["n2"])
                u_r = node_remap.get(u)
                v_r = node_remap.get(v)
                w_r = node_remap.get(w)
                if any(idx is None for idx in (u_r, v_r, w_r)):
                    raise ValueError("step anchor node missing from base graph")

                step_role = str(step.get("step_role") or ("semantic" if bool(step.get("is_semantic")) else "aux"))
                action_code_vec = encode_local_dyad_code(
                    sample["x0"][u],
                    sample["x0"][v],
                    sample["x0"][w],
                    sample["x0"][n1],
                    sample["x0"][n2],
                )
                expert_paths.append(
                    {
                        "expert_step_id": len(expert_paths),
                        "raw_index": int(raw_idx),
                        "trace_id": sample_id,
                        "sample_id": sample_id,
                        "family_id": family_name,
                        "family_index": family_index,
                        "seed_type": sample.get("seed_type"),
                        "step_index": step_index,
                        "step_index_1based": step_index + 1,
                        "step_count": step_count,
                        "step_role": step_role,
                        "step_role_index": STEP_ROLE_TO_INDEX[step_role],
                        "stop_token": float(step_index == step_count - 1),
                        "topology_signature": sample.get("topology_signature"),
                        "generation_trace": trace,
                        "base_data": base_data,
                        "trace_seed_graph": seed_graph,
                        "action_topo": torch.tensor([u_r, v_r, w_r], dtype=torch.long),
                        "action_geo": torch.tensor(
                            [
                                float(sample["x0"][n1][0]),
                                float(sample["x0"][n1][1]),
                                float(sample["x0"][n2][0]),
                                float(sample["x0"][n2][1]),
                            ],
                            dtype=torch.float32,
                        ),
                        "action_anchor_pair": torch.tensor([u_r, v_r], dtype=torch.long),
                        "action_anchor_i": int(u_r),
                        "action_anchor_j": int(v_r),
                        "action_support": int(w_r),
                        "action_code_vec": torch.tensor(action_code_vec, dtype=torch.float32),
                        "action_code_bucket": codebook_bucket_for_step(
                            family_name,
                            step_role,
                            step_index=step_index,
                            action_topo=[u_r, v_r, w_r],
                            topology_signature=sample.get("topology_signature"),
                            fine_topology_buckets=fine_buckets_enabled,
                            fine_bucket_policy=fine_bucket_policy,
                        ),
                        "y_foot": y_foot,
                        "y_knee": y_knee,
                        "y_ankle": y_ankle,
                    }
                )
        except Exception as exc:
            errors += 1
            if len(error_examples) < 5:
                error_examples.append(f"sample_idx={raw_idx}: {type(exc).__name__}: {exc}")

    print(f"\n[OK] Extracted {len(expert_paths)} multistep expert steps. ({errors} errors skipped)")
    if error_examples:
        print("[!] Example extraction failures:")
        for message in error_examples:
            print(f"    - {message}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    codebook = build_action_codebook(
        expert_paths,
        cluster_radius=float(codebook_cfg["cluster_radius"]),
        max_codes_per_bucket=int(codebook_cfg["max_codes_per_bucket"]),
        max_codes_per_bucket_overrides=dict(codebook_cfg.get("max_codes_per_bucket_overrides", {})),
        representative_strategy=str(codebook_cfg["representative_strategy"]),
        constraint_cfg=constraint_cfg,
        max_validation_candidates=int(codebook_cfg["max_validation_candidates"]),
        max_validation_contexts=int(codebook_cfg["max_validation_contexts"]),
        fine_bucket_policy=fine_bucket_policy,
    )
    expert_paths = attach_action_codebook(expert_paths, codebook)
    expert_paths = _attach_doc_masks(expert_paths, codebook)
    torch.save(expert_paths, output_path)
    save_action_codebook(output_path, codebook, step_paths=expert_paths)
    print(f"[OK] Saved multistep IL cache to {output_path}")
    return expert_paths


def ensure_multistep_expert_paths(
    pkl_path: str,
    output_path: str,
    use_cached: bool = True,
    *,
    action_codebook_cfg: dict | None = None,
    constraint_cfg: dict | None = None,
):
    if use_cached and os.path.exists(output_path):
        print(f"[*] Loading cached multistep IL dataset from {output_path}")
        expert_paths = torch.load(output_path, map_location="cpu", weights_only=False)
        codebook_path = default_action_codebook_path(output_path)
        if multistep_paths_have_phase4_format(expert_paths) and os.path.exists(codebook_path):
            codebook = load_action_codebook(output_path)
            if not codebook_cache_is_stale(expert_paths, codebook):
                return expert_paths
        print("[*] Cached multistep IL dataset is stale; regenerating...")
    return extract_multistep_expert_paths(
        pkl_path=pkl_path,
        output_path=output_path,
        action_codebook_cfg=action_codebook_cfg,
        constraint_cfg=constraint_cfg,
    )


def load_step_split(
    step_paths: Sequence[dict[str, object]],
    *,
    split_path: str,
    precomputed_split_path: str | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
) -> dict[str, object]:
    if precomputed_split_path and os.path.exists(precomputed_split_path):
        raw_split = _load_split_artifact(precomputed_split_path)
        split = _map_precomputed_group_split(step_paths, raw_split, source_path=precomputed_split_path)
    else:
        split = _random_group_split(
            step_paths,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )

    split_dir = os.path.dirname(split_path)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
    torch.save(split, split_path)
    return split


def _load_split_artifact(split_path: str) -> dict[str, object]:
    path = Path(split_path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return torch.load(path, map_location="cpu", weights_only=False)


def _canonical_split_indices(split: dict[str, object]) -> dict[str, list[int]]:
    return {
        "train": [int(idx) for idx in split.get("train_indices", split.get("train", []))],
        "val": [int(idx) for idx in split.get("val_indices", split.get("val", []))],
        "test": [int(idx) for idx in split.get("test_indices", split.get("test", []))],
    }


def _map_precomputed_group_split(
    step_paths: Sequence[dict[str, object]],
    raw_split: dict[str, object],
    *,
    source_path: str,
) -> dict[str, object]:
    split_ids = {name: set(indices) for name, indices in _canonical_split_indices(raw_split).items()}
    mapped = {"train_indices": [], "val_indices": [], "test_indices": []}
    for local_idx, item in enumerate(step_paths):
        raw_index = item.get("raw_index")
        split_key_candidates = []
        if raw_index is not None:
            split_key_candidates.append(int(raw_index))
        split_key_candidates.append(int(item["sample_id"]))
        split_key_candidates = list(dict.fromkeys(split_key_candidates))

        if any(key in split_ids["train"] for key in split_key_candidates):
            mapped["train_indices"].append(local_idx)
        elif any(key in split_ids["val"] for key in split_key_candidates):
            mapped["val_indices"].append(local_idx)
        elif any(key in split_ids["test"] for key in split_key_candidates):
            mapped["test_indices"].append(local_idx)
        else:
            raise ValueError(
                f"Sample id {int(item['sample_id'])} (raw_index={raw_index}) missing from precomputed split '{source_path}'"
            )

    _validate_step_split(step_paths, mapped)
    return {
        **mapped,
        "split_source": "precomputed_group_split_multistep",
        "source_path": source_path,
        "split_seed": raw_split.get("split_seed"),
        "val_ratio": len(mapped["val_indices"]) / max(1, len(step_paths)),
        "test_ratio": len(mapped["test_indices"]) / max(1, len(step_paths)),
        "unique_trace_counts": _trace_counts_for_split(step_paths, mapped),
    }


def _random_group_split(
    step_paths: Sequence[dict[str, object]],
    *,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> dict[str, object]:
    traces: dict[int, list[int]] = defaultdict(list)
    for local_idx, item in enumerate(step_paths):
        traces[int(item["sample_id"])].append(local_idx)

    trace_ids = list(traces.keys())
    rng = random.Random(split_seed)
    rng.shuffle(trace_ids)

    n_traces = len(trace_ids)
    n_test = max(1, int(round(n_traces * test_ratio)))
    n_val = max(1, int(round(n_traces * val_ratio)))
    if n_traces - n_test - n_val <= 0:
        raise ValueError(
            f"Invalid trace split for n_traces={n_traces}, val_ratio={val_ratio}, test_ratio={test_ratio}"
        )

    test_ids = set(trace_ids[:n_test])
    val_ids = set(trace_ids[n_test : n_test + n_val])
    train_ids = set(trace_ids[n_test + n_val :])
    mapped = {
        "train_indices": sorted(idx for trace_id in train_ids for idx in traces[trace_id]),
        "val_indices": sorted(idx for trace_id in val_ids for idx in traces[trace_id]),
        "test_indices": sorted(idx for trace_id in test_ids for idx in traces[trace_id]),
    }
    _validate_step_split(step_paths, mapped)
    return {
        **mapped,
        "split_source": "random_group_split_multistep",
        "source_path": None,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "unique_trace_counts": _trace_counts_for_split(step_paths, mapped),
    }


def _trace_counts_for_split(step_paths: Sequence[dict[str, object]], split: dict[str, list[int]]) -> dict[str, int]:
    counts = {}
    for split_name in ("train_indices", "val_indices", "test_indices"):
        trace_ids = {int(step_paths[idx]["sample_id"]) for idx in split[split_name]}
        counts[split_name.replace("_indices", "")] = len(trace_ids)
    return counts


def _validate_step_split(step_paths: Sequence[dict[str, object]], split: dict[str, list[int]]) -> None:
    seen: set[int] = set()
    trace_to_split: dict[int, str] = {}
    for split_name in ("train_indices", "val_indices", "test_indices"):
        for idx in split[split_name]:
            if idx < 0 or idx >= len(step_paths):
                raise ValueError(f"Split index {idx} out of range for {len(step_paths)} step paths")
            if idx in seen:
                raise ValueError(f"Step index {idx} appears in multiple splits")
            seen.add(idx)
            trace_id = int(step_paths[idx]["sample_id"])
            previous = trace_to_split.get(trace_id)
            if previous is not None and previous != split_name:
                raise ValueError(f"Trace id {trace_id} appears in both {previous} and {split_name}")
            trace_to_split[trace_id] = split_name
    if len(seen) != len(step_paths):
        raise ValueError("Step split does not cover all multistep IL samples")


def subset_by_indices(items: Sequence, indices: Sequence[int]) -> list:
    return [items[int(idx)] for idx in indices]


def filter_paths_by_families(paths: Sequence[dict[str, object]], families: Iterable[str]) -> list[dict[str, object]]:
    family_set = {str(name) for name in families}
    return [item for item in paths if str(item["family_id"]) in family_set]


def build_stage_plan(il_cfg: dict) -> list[dict[str, object]]:
    configured = il_cfg.get("curriculum_stages")
    if configured:
        return [dict(stage) for stage in configured]
    epochs = int(il_cfg.get("epochs", 100))
    patience = int(il_cfg.get("patience", 20))
    return [
        {
            **stage,
            "epochs": int(stage.get("epochs", epochs)),
            "patience": int(stage.get("patience", patience)),
        }
        for stage in DEFAULT_CURRICULUM_STAGES
    ]


def _masked_group_nll_loss(
    logits: torch.Tensor,
    batch_index: torch.Tensor,
    valid_mask: torch.Tensor,
    global_targets: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    probs = softmax(masked_logits, batch_index)
    return (-torch.log(probs[global_targets] + 1e-12)).mean()


def _argmax_per_graph(logits: torch.Tensor, batch_index: torch.Tensor, valid_mask: torch.Tensor, offsets: torch.Tensor):
    predictions = []
    for graph_idx in range(int(offsets.numel())):
        start = int(offsets[graph_idx].item())
        end = int(offsets[graph_idx + 1].item()) if graph_idx + 1 < offsets.numel() else int(logits.size(0))
        graph_logits = logits[start:end].masked_fill(~valid_mask[start:end], -1e9)
        predictions.append(int(torch.argmax(graph_logits).item()))
    return torch.tensor(predictions, dtype=torch.long, device=logits.device)


def _code_bucket_mask(
    policy,
    family_ids: torch.Tensor,
    step_role_ids: torch.Tensor,
    target_ids: torch.Tensor,
    action_topos: torch.Tensor | None = None,
    step_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    num_rows = int(target_ids.numel())
    num_codes = int(policy.action_codebook.size(0)) if hasattr(policy, "action_codebook") else 0
    if num_codes <= 0:
        raise ValueError("policy action_codebook is empty; cannot build geometry-code mask")

    mask = torch.zeros((num_rows, num_codes), dtype=torch.bool, device=target_ids.device)
    bucket_map = getattr(policy, "action_codebook_buckets", {}) or {}
    for row_idx in range(num_rows):
        family_name = family_name_from_index(int(family_ids[row_idx].item()))
        step_role = "semantic" if int(step_role_ids[row_idx].item()) == 1 else "aux"
        action_topo = action_topos[row_idx] if action_topos is not None else None
        bucket = resolve_codebook_bucket_for_step(
            bucket_map,
            family_name,
            step_role,
            step_index=int(step_indices[row_idx].item()) if step_indices is not None else None,
            action_topo=action_topo,
        )
        allowed_ids = list(bucket_map.get(bucket, []))
        if not allowed_ids:
            allowed_ids = list(range(num_codes))
        # Keep the truth class legal even when a stale bucket map is missing an id.
        truth_id = int(target_ids[row_idx].item())
        if 0 <= truth_id < num_codes and truth_id not in allowed_ids:
            allowed_ids.append(truth_id)
        mask[row_idx, torch.tensor(allowed_ids, dtype=torch.long, device=target_ids.device)] = True
    return mask


def _topk_code_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(logits.size(-1)))
    topk = torch.topk(logits, k=k, dim=-1).indices
    return (topk == targets.view(-1, 1)).any(dim=-1).float().mean()


def _topk_oracle_accuracy(logits: torch.Tensor, positive_mask: torch.Tensor | None, k: int) -> torch.Tensor:
    if positive_mask is None or positive_mask.numel() == 0:
        return logits.sum() * 0.0
    k = min(int(k), int(logits.size(-1)))
    topk = torch.topk(logits, k=k, dim=-1).indices
    hits = positive_mask.gather(1, topk).any(dim=-1)
    covered = positive_mask.any(dim=-1)
    if not bool(covered.any().item()):
        return logits.sum() * 0.0
    return hits[covered].float().mean()


def _apply_code_vector_for_oracle(
    graph: Data,
    action_topo: Sequence[int],
    code_vec: Sequence[float],
    constraint_cfg: dict | None,
) -> tuple[bool, str | None, np.ndarray | None]:
    try:
        u, v, w = [int(value) for value in list(action_topo)[:3]]
        node_count = int(graph.pos.size(0))
        if min(u, v, w) < 0 or max(u, v, w) >= node_count:
            return False, "action_topology_out_of_bounds", None
        n1, n2 = decode_local_dyad_code(
            graph.pos[u].detach().cpu().numpy(),
            graph.pos[v].detach().cpu().numpy(),
            graph.pos[w].detach().cpu().numpy(),
            np.asarray(code_vec, dtype=np.float32),
        )
        next_graph = apply_j_operator(graph, u, v, w, n1, n2)
        is_valid, reason = validate_graph_structure(next_graph, constraint_cfg or {})
        if not bool(is_valid):
            return False, str(reason), None
        return True, None, np.concatenate([n1, n2]).astype(np.float32)
    except Exception as exc:
        return False, type(exc).__name__, None


def _normalised_geometry_error(graph: Data, action_topo: Sequence[int], candidate_geo: np.ndarray, truth_geo) -> float:
    u, v, _ = [int(value) for value in list(action_topo)[:3]]
    span = float(torch.linalg.norm(graph.pos[v] - graph.pos[u]).detach().cpu().item())
    span = max(span, 1.0e-8)
    truth = truth_geo.detach().cpu().numpy() if hasattr(truth_geo, "detach") else np.asarray(truth_geo, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate_geo, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean((candidate - truth) ** 2)) / span)


def _oracle_group_template() -> dict[str, object]:
    return {
        "count": 0,
        "oracle_positive": 0,
        "valid_code_available": 0,
        "oracle_uncovered": 0,
    }


def _finalise_oracle_group(group: dict[str, object]) -> dict[str, object]:
    count = int(group.get("count", 0))
    positive = int(group.get("oracle_positive", 0))
    valid = int(group.get("valid_code_available", 0))
    uncovered = int(group.get("oracle_uncovered", 0))
    return {
        "count": count,
        "oracle_positive": positive,
        "oracle_positive_coverage_rate": float(positive / count) if count else 0.0,
        "valid_code_available": valid,
        "valid_code_available_rate": float(valid / count) if count else 0.0,
        "oracle_uncovered": uncovered,
        "oracle_uncovered_rate": float(uncovered / count) if count else 0.0,
    }


def attach_oracle_code_targets(
    paths: Sequence[dict[str, object]],
    codebook: dict[str, object],
    cfg: dict,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    oracle_cfg = dict(cfg.get("il_training", {}).get("oracle_code_loss", {}) or {})
    threshold = float(oracle_cfg.get("positive_error_threshold", 0.025))
    constraints = cfg.get("constraints", {})
    entries = {int(entry["id"]): np.asarray(entry["vector"], dtype=np.float32) for entry in codebook.get("entries", [])}
    all_ids = sorted(entries.keys())
    groups: dict[str, dict[str, object]] = defaultdict(_oracle_group_template)
    enriched: list[dict[str, object]] = []

    for item in paths:
        family_name = str(item.get("family_id", "unknown"))
        step_role = str(item.get("step_role", "semantic"))
        action_topo = item["action_topo"]
        topo_list = [int(value) for value in action_topo.detach().cpu().view(-1).tolist()] if hasattr(action_topo, "detach") else [int(value) for value in action_topo]
        allowed_ids = allowed_code_ids_for_context(
            codebook,
            family_name=family_name,
            step_role=step_role,
            step_index=int(item.get("step_index", 0)),
            action_topo=topo_list,
            topology_signature=item.get("topology_signature"),
        )
        if not allowed_ids:
            allowed_ids = list(all_ids)

        valid_ids: list[int] = []
        valid_errors: list[float] = []
        positive_ids: list[int] = []
        positive_errors: list[float] = []
        invalid_ids: list[int] = []
        invalid_reasons: Counter[str] = Counter()
        best_valid_id = -1
        best_valid_error = float("inf")

        for code_id in allowed_ids:
            code_id = int(code_id)
            if code_id not in entries:
                continue
            is_valid, reason, candidate_geo = _apply_code_vector_for_oracle(
                item["base_data"],
                topo_list,
                entries[code_id],
                constraints,
            )
            if not is_valid or candidate_geo is None:
                invalid_ids.append(code_id)
                invalid_reasons[str(reason or "invalid")] += 1
                continue
            error = _normalised_geometry_error(item["base_data"], topo_list, candidate_geo, item["action_geo"])
            valid_ids.append(code_id)
            valid_errors.append(error)
            if error < best_valid_error:
                best_valid_error = float(error)
                best_valid_id = int(code_id)
            if error <= threshold:
                positive_ids.append(code_id)
                positive_errors.append(float(error))

        new_item = dict(item)
        new_item["oracle_candidate_ids"] = [int(idx) for idx in allowed_ids]
        new_item["valid_code_ids"] = valid_ids
        new_item["valid_code_errors"] = valid_errors
        new_item["oracle_positive_ids"] = positive_ids
        new_item["oracle_positive_errors"] = positive_errors
        new_item["oracle_invalid_code_ids"] = invalid_ids
        new_item["oracle_invalid_reason_counts"] = dict(invalid_reasons)
        new_item["best_valid_code_id"] = int(best_valid_id)
        new_item["best_valid_code_error"] = float(best_valid_error) if np.isfinite(best_valid_error) else None
        new_item["oracle_uncovered"] = len(positive_ids) == 0
        enriched.append(new_item)

        for key in ("overall", family_name, f"{family_name}/step{int(item.get('step_index', 0))}"):
            group = groups[key]
            group["count"] = int(group["count"]) + 1
            group["oracle_positive"] = int(group["oracle_positive"]) + int(len(positive_ids) > 0)
            group["valid_code_available"] = int(group["valid_code_available"]) + int(len(valid_ids) > 0)
            group["oracle_uncovered"] = int(group["oracle_uncovered"]) + int(len(positive_ids) == 0)

    return enriched, {key: _finalise_oracle_group(value) for key, value in sorted(groups.items())}


def _zero_like_loss(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def _oracle_batch_field(batch: dict, key: str, logits: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return default.to(logits.device)
    return value.to(logits.device)


def compute_oracle_code_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor | Data],
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    il_cfg = cfg.get("il_training", {})
    oracle_cfg = dict(il_cfg.get("oracle_code_loss", {}) or {})
    targets = batch["action_code_id"].long().to(logits.device)
    hard_ce = F.cross_entropy(logits, targets)
    if not bool(oracle_cfg.get("enabled", False)):
        zero = _zero_like_loss(logits)
        return hard_ce, {
            "loss_action_code_hard_ce": hard_ce.detach(),
            "loss_oracle_soft_ce": zero.detach(),
            "loss_oracle_rank": zero.detach(),
            "loss_validity_margin": zero.detach(),
            "oracle_positive_coverage": zero.detach(),
            "oracle_uncovered_rate": zero.detach(),
            "oracle_best_valid_available_rate": zero.detach(),
        }

    positive_mask = _oracle_batch_field(
        batch,
        "oracle_positive_mask",
        logits,
        torch.zeros_like(logits, dtype=torch.bool),
    ).bool()
    soft_targets = _oracle_batch_field(
        batch,
        "oracle_soft_targets",
        logits,
        torch.zeros_like(logits, dtype=torch.float32),
    ).float()
    valid_mask = _oracle_batch_field(
        batch,
        "oracle_valid_mask",
        logits,
        torch.zeros_like(logits, dtype=torch.bool),
    ).bool()
    best_valid = _oracle_batch_field(
        batch,
        "oracle_best_valid_code_id",
        logits,
        torch.full((logits.size(0),), -1, dtype=torch.long, device=logits.device),
    ).long()
    uncovered = _oracle_batch_field(
        batch,
        "oracle_uncovered",
        logits,
        torch.ones((logits.size(0),), dtype=torch.bool, device=logits.device),
    ).bool()
    covered = (~uncovered) & positive_mask.any(dim=-1)
    finite_mask = logits > -1.0e8
    zero = _zero_like_loss(logits)

    if bool(covered.any().item()):
        log_probs = F.log_softmax(logits[covered], dim=-1)
        soft_ce = -(soft_targets[covered] * log_probs).sum(dim=-1).mean()
    else:
        soft_ce = zero

    rank_losses = []
    valid_losses = []
    rank_margin = float(oracle_cfg.get("rank_margin", 0.5))
    validity_margin = float(oracle_cfg.get("validity_margin", 0.25))
    for row_idx in range(int(logits.size(0))):
        finite = finite_mask[row_idx]
        positives = positive_mask[row_idx] & finite
        best_id = int(best_valid[row_idx].item())
        if not bool(positives.any().item()) and 0 <= best_id < int(logits.size(1)) and bool(finite[best_id].item()):
            positives = torch.zeros_like(finite)
            positives[best_id] = True
        if bool(positives.any().item()):
            pos_score = logits[row_idx][positives].max()
            negatives = finite & (~positives)
            if bool(negatives.any().item()):
                rank_losses.append(F.relu(rank_margin - pos_score + logits[row_idx][negatives]).mean())
        if 0 <= best_id < int(logits.size(1)):
            invalids = finite & (~valid_mask[row_idx])
            if bool(invalids.any().item()):
                valid_losses.append(F.relu(validity_margin - logits[row_idx, best_id] + logits[row_idx][invalids]).mean())

    rank_loss = torch.stack(rank_losses).mean() if rank_losses else zero
    validity_loss = torch.stack(valid_losses).mean() if valid_losses else zero
    total = (
        float(oracle_cfg.get("w_hard_code_ce", 0.25)) * hard_ce
        + float(oracle_cfg.get("w_soft_oracle_ce", 1.0)) * soft_ce
        + float(oracle_cfg.get("w_pairwise_rank", 1.0)) * rank_loss
        + float(oracle_cfg.get("w_validity_margin", 1.0)) * validity_loss
    )
    return total, {
        "loss_action_code_hard_ce": hard_ce.detach(),
        "loss_oracle_soft_ce": soft_ce.detach(),
        "loss_oracle_rank": rank_loss.detach(),
        "loss_validity_margin": validity_loss.detach(),
        "oracle_positive_coverage": covered.float().mean().detach(),
        "oracle_uncovered_rate": uncovered.float().mean().detach(),
        "oracle_best_valid_available_rate": (best_valid >= 0).float().mean().detach(),
    }


def constrained_code_choice(
    graph: Data,
    action_topo: Sequence[int],
    ordered_code_ids: Sequence[int],
    codebook: torch.Tensor,
    constraint_cfg: dict | None,
) -> dict[str, object]:
    ordered = [int(idx) for idx in ordered_code_ids]
    if not ordered:
        return {"code_id": -1, "valid": False, "used_non_top1": False, "invalid_reason": "empty_candidates"}
    last_reason = None
    for rank, code_id in enumerate(ordered):
        if code_id < 0 or code_id >= int(codebook.size(0)):
            last_reason = "code_id_out_of_bounds"
            continue
        vector = codebook[code_id].detach().cpu().numpy()
        is_valid, reason, _ = _apply_code_vector_for_oracle(graph, action_topo, vector, constraint_cfg)
        if is_valid:
            return {
                "code_id": int(code_id),
                "valid": True,
                "used_non_top1": bool(rank > 0),
                "rank": int(rank + 1),
                "invalid_reason": None,
            }
        last_reason = reason
    return {
        "code_id": int(ordered[0]),
        "valid": False,
        "used_non_top1": False,
        "rank": 1,
        "invalid_reason": last_reason,
    }


def compute_phase4_losses(
    policy,
    batch: dict[str, torch.Tensor | Data],
    z_c: torch.Tensor,
    cfg: dict,
):
    il_cfg = cfg.get("il_training", {})
    base_data = batch["base_data"]
    action_topo = batch["action_topo"]
    offsets = _batch_offsets(base_data)
    batch_index = base_data.batch

    x_enc = policy.encode_graph(base_data)
    phase4_outputs = policy.phase4_outputs(
        base_data,
        x_enc,
        z_c,
        family_ids=batch["family_index"],
        step_indices=batch["step_index"],
        step_counts=batch["step_count"],
    )
    global_action = action_topo.to(offsets.device) + offsets.unsqueeze(1)
    code_logits = policy.geometry_code_logits(
        base_data,
        x_enc,
        phase4_outputs["graph_context"],
        action_topo,
    )
    action_code_targets = batch["action_code_id"].long()
    code_mask = _code_bucket_mask(
        policy,
        batch["family_index"],
        batch["step_role_index"],
        action_code_targets,
        action_topos=action_topo,
        step_indices=batch["step_index"],
    )
    masked_code_logits = code_logits.masked_fill(~code_mask, -1e9)

    is_fixed = base_data.x[:, 2] > 0.5
    moving_mask = ~is_fixed
    fixed_mask = is_fixed

    loss_u = _masked_group_nll_loss(
        phase4_outputs["u_logits"],
        batch_index,
        moving_mask,
        global_action[:, 0],
    )
    loss_v = _masked_group_nll_loss(
        phase4_outputs["v_logits"],
        batch_index,
        moving_mask,
        global_action[:, 1],
    )
    loss_w = _masked_group_nll_loss(
        phase4_outputs["w_logits"],
        batch_index,
        fixed_mask,
        global_action[:, 2],
    )
    loss_stop = F.binary_cross_entropy_with_logits(
        phase4_outputs["stop_logits"].view(-1),
        batch["stop_token"].view(-1),
    )
    loss_step_role = F.cross_entropy(phase4_outputs["step_role_logits"], batch["step_role_index"])
    loss_step_count = F.cross_entropy(
        phase4_outputs["step_count_logits"],
        torch.clamp(batch["step_count"].long() - 1, min=0),
    )
    if bool(il_cfg.get("oracle_code_loss", {}).get("enabled", False)):
        loss_action_code, oracle_loss_metrics = compute_oracle_code_loss(masked_code_logits, batch, cfg)
    elif bool(il_cfg.get("use_bucket_masked_code_loss", False)):
        loss_action_code = F.cross_entropy(masked_code_logits, action_code_targets)
        oracle_loss_metrics = {}
    else:
        loss_action_code = F.cross_entropy(code_logits, action_code_targets)
        oracle_loss_metrics = {}

    total = (
        float(il_cfg.get("w_action", 1.0)) * (loss_u + loss_v + loss_w)
        + float(il_cfg.get("w_geometry_code", 1.0)) * loss_action_code
        + float(il_cfg.get("w_stop", 1.0)) * loss_stop
        + float(il_cfg.get("w_step_role", 0.5)) * loss_step_role
        + float(il_cfg.get("w_step_count", 0.5)) * loss_step_count
    )

    pred_u = _argmax_per_graph(phase4_outputs["u_logits"], batch_index, moving_mask, offsets)
    pred_v = _argmax_per_graph(phase4_outputs["v_logits"], batch_index, moving_mask, offsets)
    pred_w = _argmax_per_graph(phase4_outputs["w_logits"], batch_index, fixed_mask, offsets)
    pred_stop = (torch.sigmoid(phase4_outputs["stop_logits"].view(-1)) >= 0.5).long()
    pred_step_role = torch.argmax(phase4_outputs["step_role_logits"], dim=-1)
    pred_step_count = torch.argmax(phase4_outputs["step_count_logits"], dim=-1) + 1
    pred_action_code_raw = torch.argmax(code_logits, dim=-1)
    pred_action_code = torch.argmax(masked_code_logits, dim=-1)
    topology_exact = (
        (pred_u == action_topo[:, 0])
        & (pred_v == action_topo[:, 1])
        & (pred_w == action_topo[:, 2])
    )
    full_step_exact = topology_exact & (pred_action_code == action_code_targets)

    metrics = {
        "objective": total,
        "total": total,
        "total_prior": torch.zeros_like(total),
        "loss_topo": loss_u + loss_v + loss_w,
        "loss_geo": loss_action_code,
        "loss_recon": loss_action_code,
        "loss_kl": torch.zeros_like(total),
        "loss_geo_prior": torch.zeros_like(total),
        "loss_geo_regularizer": torch.zeros_like(total),
        "loss_action_u": loss_u,
        "loss_action_v": loss_v,
        "loss_action_w": loss_w,
        "loss_action_code": loss_action_code,
        "loss_stop": loss_stop,
        "loss_step_role": loss_step_role,
        "loss_step_count": loss_step_count,
        "stop_accuracy": (pred_stop == batch["stop_token"].long()).float().mean(),
        "step_role_accuracy": (pred_step_role == batch["step_role_index"]).float().mean(),
        "step_count_accuracy": (pred_step_count == batch["step_count"].long()).float().mean(),
        "action_u_accuracy": (pred_u == action_topo[:, 0]).float().mean(),
        "action_v_accuracy": (pred_v == action_topo[:, 1]).float().mean(),
        "action_w_accuracy": (pred_w == action_topo[:, 2]).float().mean(),
        "topology_exact": topology_exact.float().mean(),
        "action_code_accuracy_raw": (pred_action_code_raw == action_code_targets).float().mean(),
        "action_code_accuracy_bucket": (pred_action_code == action_code_targets).float().mean(),
        "action_code_accuracy": (pred_action_code == action_code_targets).float().mean(),
        "action_code_top1": _topk_code_accuracy(masked_code_logits, action_code_targets, 1),
        "action_code_top3": _topk_code_accuracy(masked_code_logits, action_code_targets, 3),
        "action_code_top5": _topk_code_accuracy(masked_code_logits, action_code_targets, 5),
        "full_step_exact": full_step_exact.float().mean(),
    }
    positive_mask = batch.get("oracle_positive_mask")
    if positive_mask is not None:
        positive_mask = positive_mask.to(masked_code_logits.device).bool()
        metrics["oracle_equiv_top1"] = _topk_oracle_accuracy(masked_code_logits, positive_mask, 1)
        metrics["oracle_equiv_top3"] = _topk_oracle_accuracy(masked_code_logits, positive_mask, 3)
        metrics["oracle_equiv_top5"] = _topk_oracle_accuracy(masked_code_logits, positive_mask, 5)
    else:
        zero_metric = _zero_like_loss(masked_code_logits).detach()
        metrics["oracle_equiv_top1"] = zero_metric
        metrics["oracle_equiv_top3"] = zero_metric
        metrics["oracle_equiv_top5"] = zero_metric
    if not oracle_loss_metrics:
        zero_metric = _zero_like_loss(masked_code_logits).detach()
        oracle_loss_metrics = {
            "loss_action_code_hard_ce": loss_action_code.detach(),
            "loss_oracle_soft_ce": zero_metric,
            "loss_oracle_rank": zero_metric,
            "loss_validity_margin": zero_metric,
            "oracle_positive_coverage": zero_metric,
            "oracle_uncovered_rate": zero_metric,
            "oracle_best_valid_available_rate": zero_metric,
        }
    metrics.update(oracle_loss_metrics)
    metrics["step_action_accuracy"] = metrics["full_step_exact"]
    return metrics


def group_paths_by_trace(paths: Sequence[dict[str, object]]) -> list[list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in paths:
        grouped[int(item["trace_id"])].append(item)
    return [sorted(trace_items, key=lambda item: int(item["step_index"])) for _, trace_items in sorted(grouped.items())]


def scheduled_sampling_ratio(stage_cfg: dict, *, epoch: int, total_epochs: int) -> float:
    rollout_cfg = stage_cfg.get("rollout_aware", {}) or {}
    if not bool(rollout_cfg.get("enabled", False)):
        return 0.0
    start = float(rollout_cfg.get("start_ratio", rollout_cfg.get("ratio", 0.0)))
    end = float(rollout_cfg.get("end_ratio", start))
    total_epochs = max(1, int(total_epochs))
    epoch = min(max(int(epoch), 0), total_epochs - 1)
    if total_epochs == 1:
        return max(0.0, min(1.0, end))
    progress = float(epoch) / float(total_epochs - 1)
    ratio = start + (end - start) * progress
    return max(0.0, min(1.0, ratio))


def _target_curves_for_step(step: dict[str, object], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_foot = step["y_foot"].unsqueeze(0).to(device)
    target_knee = step["y_knee"].unsqueeze(0).to(device)
    target_ankle = step["y_ankle"].unsqueeze(0).to(device)
    return target_foot, target_knee, target_ankle


def _encode_step_target(policy, curve_encoder, step: dict[str, object], device) -> torch.Tensor:
    target_foot, target_knee, target_ankle = _target_curves_for_step(step, device)
    if curve_encoder is None:
        latent_dim = int(getattr(policy, "curve_latent_dim", 128))
        return torch.zeros((1, latent_dim), dtype=torch.float32, device=device)
    return curve_encoder(target_foot, target_knee, target_ankle)


def _truth_action_valid_on_graph(graph: Data, step: dict[str, object], cfg: dict) -> tuple[bool, str | None]:
    topo = step["action_topo"].detach().cpu().view(-1).long()
    if topo.numel() != 3:
        return False, "truth_action_topology_malformed"
    node_count = int(graph.pos.size(0))
    if int(topo.min().item()) < 0 or int(topo.max().item()) >= node_count:
        return False, "truth_action_index_out_of_bounds"
    if getattr(graph, "x", None) is not None and graph.x.dim() == 2 and graph.x.size(1) > 2:
        fixed_mask = graph.x[:, 2] > 0.5
        u, v, w = (int(topo[0].item()), int(topo[1].item()), int(topo[2].item()))
        if bool(fixed_mask[u].item()) or bool(fixed_mask[v].item()) or not bool(fixed_mask[w].item()):
            return False, "truth_action_anchor_type_invalid"
    try:
        code_vec = step.get("action_code_target")
        if code_vec is None:
            return False, "truth_action_code_missing"
        code_array = code_vec.detach().cpu().numpy() if hasattr(code_vec, "detach") else np.asarray(code_vec, dtype=np.float32)
        u, v, w = (int(topo[0].item()), int(topo[1].item()), int(topo[2].item()))
        n1, n2 = decode_local_dyad_code(
            graph.pos[u].detach().cpu().numpy(),
            graph.pos[v].detach().cpu().numpy(),
            graph.pos[w].detach().cpu().numpy(),
            code_array,
        )
        next_graph = apply_j_operator(graph, u, v, w, n1, n2)
        is_valid, reason = validate_graph_structure(next_graph, cfg.get("constraints", {}))
        if not bool(is_valid):
            return False, str(reason)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _apply_predicted_step(policy, graph: Data, pred: dict[str, int], cfg: dict) -> tuple[Data | None, str | None]:
    try:
        node_count = int(graph.pos.size(0))
        if min(int(pred["u"]), int(pred["v"]), int(pred["w"])) < 0 or max(int(pred["u"]), int(pred["v"]), int(pred["w"])) >= node_count:
            return None, "predicted_action_index_out_of_bounds"
        pred_geo = decode_local_dyad_code(
            graph.pos[int(pred["u"])].detach().cpu().numpy(),
            graph.pos[int(pred["v"])].detach().cpu().numpy(),
            graph.pos[int(pred["w"])].detach().cpu().numpy(),
            policy.action_codebook[int(pred["code"])].detach().cpu().numpy(),
        )
        next_graph = apply_j_operator(
            graph,
            int(pred["u"]),
            int(pred["v"]),
            int(pred["w"]),
            pred_geo[0],
            pred_geo[1],
        )
        is_valid, reason = validate_graph_structure(next_graph, cfg.get("constraints", {}))
        if not bool(is_valid):
            return None, str(reason)
    except Exception as exc:
        return None, str(exc)
    return next_graph, None


def _candidate_rollout_code_ids(
    policy,
    rollout_prediction: dict[str, object],
    pred: dict[str, int],
    step: dict[str, object],
    device,
    *,
    max_codes: int,
) -> list[int]:
    if max_codes <= 0 or not hasattr(policy, "geometry_code_logits"):
        return [int(pred["code"])]
    batch_graph = rollout_prediction.get("batch_graph")
    x_enc = rollout_prediction.get("x_enc")
    graph_context = rollout_prediction.get("graph_context")
    if batch_graph is None or x_enc is None or graph_context is None:
        return [int(pred["code"])]
    action_topo = torch.tensor([[int(pred["u"]), int(pred["v"]), int(pred["w"])]], dtype=torch.long, device=device)
    try:
        logits = policy.geometry_code_logits(batch_graph, x_enc, graph_context, action_topo).view(-1)
    except Exception:
        return [int(pred["code"])]
    allowed_map = getattr(policy, "action_codebook_buckets", {}) or {}
    if allowed_map:
        family_name = family_name_from_index(int(step["family_index"]))
        step_role = INDEX_TO_STEP_ROLE.get(int(step["step_role_index"]), str(step.get("step_role", "semantic")))
        bucket = resolve_codebook_bucket_for_step(
            allowed_map,
            family_name,
            step_role,
            step_index=int(step["step_index"]),
            action_topo=action_topo[0],
        )
        allowed_ids = allowed_map.get(bucket, [])
        if allowed_ids:
            mask = torch.ones_like(logits, dtype=torch.bool)
            ids_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=logits.device)
            ids_tensor = ids_tensor[(ids_tensor >= 0) & (ids_tensor < logits.numel())]
            if ids_tensor.numel() > 0:
                mask[ids_tensor] = False
                logits = logits.masked_fill(mask, -1.0e9)
    k = min(max(1, int(max_codes)), int(logits.numel()))
    ids = [int(idx) for idx in torch.topk(logits, k=k, dim=-1).indices.detach().cpu().tolist()]
    if int(pred["code"]) not in ids:
        ids.insert(0, int(pred["code"]))
    return list(dict.fromkeys(ids))


def _rollout_aware_cfg(cfg: dict) -> dict:
    return ((cfg.get("il_training", {}) or {}).get("rollout_aware", {}) or {})


def _apply_predicted_step_with_optional_rerank(
    policy,
    current_graph: Data,
    rollout_prediction: dict[str, object],
    step: dict[str, object],
    cfg: dict,
    device,
) -> tuple[Data | None, dict[str, int], str | None, bool, bool]:
    pred = dict(rollout_prediction["pred"])
    next_graph, invalid_reason = _apply_predicted_step(policy, current_graph, pred, cfg)
    if next_graph is not None:
        return next_graph, pred, invalid_reason, False, False

    rollout_cfg = _rollout_aware_cfg(cfg)
    if not bool(rollout_cfg.get("use_validity_rerank_for_generation", False)):
        return None, pred, invalid_reason, False, False

    candidate_codes = _candidate_rollout_code_ids(
        policy,
        rollout_prediction,
        pred,
        step,
        device,
        max_codes=int(rollout_cfg.get("max_rerank_codes", 10)),
    )
    for code_id in candidate_codes:
        if int(code_id) == int(pred["code"]):
            continue
        candidate_pred = dict(pred)
        candidate_pred["code"] = int(code_id)
        candidate_graph, candidate_reason = _apply_predicted_step(policy, current_graph, candidate_pred, cfg)
        if candidate_graph is not None:
            return candidate_graph, candidate_pred, candidate_reason, True, False
        invalid_reason = candidate_reason
    return None, pred, invalid_reason, False, True


def generate_rollout_aware_samples(
    policy,
    curve_encoder,
    paths: Sequence[dict[str, object]],
    cfg: dict,
    device,
    *,
    max_traces: int | None = None,
    max_samples: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    traces = group_paths_by_trace(paths)
    if max_traces is not None:
        traces = traces[: max(0, int(max_traces))]

    policy.eval()
    if curve_encoder is not None:
        curve_encoder.eval()

    samples: list[dict[str, object]] = []
    by_family: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    max_examples = 20
    rerank_used_count = 0
    rerank_no_valid_code_count = 0

    with torch.no_grad():
        for trace in traces:
            if not trace:
                continue
            first = trace[0]
            family_name = str(first["family_id"])
            trace_id = int(first["trace_id"])
            current_graph = copy.deepcopy(first["base_data"])
            z_c = _encode_step_target(policy, curve_encoder, first, device)
            prior_step_failed = False

            for step_pos, step in enumerate(trace):
                if step_pos > 0:
                    truth_valid, invalid_reason = _truth_action_valid_on_graph(current_graph, step, cfg)
                    if truth_valid:
                        augmented = copy.deepcopy(step)
                        augmented["base_data"] = copy.deepcopy(current_graph)
                        augmented["rollout_origin"] = "policy_rollout"
                        augmented["prior_step_failed"] = bool(prior_step_failed)
                        augmented["rollout_truth_action_valid"] = True
                        samples.append(augmented)
                        by_family[family_name] += 1
                        if max_samples is not None and len(samples) >= int(max_samples):
                            return samples, {
                                "trace_count": int(len(traces)),
                                "generated_count": int(len(samples)),
                                "by_family": dict(sorted(by_family.items())),
                                "drop_reasons": dict(sorted(drop_reasons.items())),
                                "rerank_used_count": int(rerank_used_count),
                                "rerank_no_valid_code_count": int(rerank_no_valid_code_count),
                                "examples": examples,
                            }
                    else:
                        drop_reasons["truth_action_invalid_on_rollout_graph"] += 1
                        if len(examples) < max_examples:
                            examples.append(
                                {
                                    "trace_id": trace_id,
                                    "family": family_name,
                                    "step_index": int(step["step_index"]),
                                    "drop_reason": "truth_action_invalid_on_rollout_graph",
                                    "invalid_reason": invalid_reason,
                                }
                            )

                rollout_prediction = _phase4_step_prediction(policy, current_graph, z_c, step, device)
                next_graph, pred, invalid_reason, rerank_used, rerank_no_valid = _apply_predicted_step_with_optional_rerank(
                    policy,
                    current_graph,
                    rollout_prediction,
                    step,
                    cfg,
                    device,
                )
                rerank_used_count += int(rerank_used)
                rerank_no_valid_code_count += int(rerank_no_valid)
                matches = _prediction_matches(pred, _truth_step_payload(step))
                prior_step_failed = bool(prior_step_failed or not _matches_are_clean(matches))
                if next_graph is None:
                    drop_reasons["predicted_rollout_invalid_or_failed"] += 1
                    if len(examples) < max_examples:
                        examples.append(
                            {
                                "trace_id": trace_id,
                                "family": family_name,
                                "step_index": int(step["step_index"]),
                                "drop_reason": "predicted_rollout_invalid_or_failed",
                                "invalid_reason": invalid_reason,
                                "pred": pred,
                            }
                        )
                    break
                current_graph = next_graph

    return samples, {
        "trace_count": int(len(traces)),
        "generated_count": int(len(samples)),
        "by_family": dict(sorted(by_family.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "rerank_used_count": int(rerank_used_count),
        "rerank_no_valid_code_count": int(rerank_no_valid_code_count),
        "examples": examples,
    }


def evaluate_multistep_reconstruction(
    policy,
    curve_encoder,
    paths: Sequence[dict[str, object]],
    cfg: dict,
    device,
    *,
    max_traces: int | None = None,
) -> dict[str, object]:
    traces = group_paths_by_trace(paths)
    if max_traces is not None:
        traces = traces[: max(0, int(max_traces))]
    if not traces:
        return {
            "trace_count": 0,
            "valid_reconstruction_rate": 0.0,
            "reconstruction_success_rate": 0.0,
            "family_success_rate": {},
        }

    policy.eval()
    if curve_encoder is not None:
        curve_encoder.eval()

    valid_flags: list[float] = []
    success_flags: list[float] = []
    family_flags: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for trace in traces:
            first = trace[0]
            family_name = str(first["family_id"])
            current_graph = copy.deepcopy(first["base_data"])
            target_foot = first["y_foot"].unsqueeze(0).to(device)
            target_knee = first["y_knee"].unsqueeze(0).to(device)
            target_ankle = first["y_ankle"].unsqueeze(0).to(device)
            if curve_encoder is None:
                latent_dim = int(getattr(policy, "curve_latent_dim", 128))
                z_c = torch.zeros((1, latent_dim), dtype=torch.float32, device=device)
            else:
                z_c = curve_encoder(target_foot, target_knee, target_ankle)

            trace_valid = True
            trace_success = True
            for step in trace:
                batch_graph = Batch.from_data_list([current_graph]).to(device)
                x_enc = policy.encode_graph(batch_graph)
                outputs = policy.phase4_outputs(
                    batch_graph,
                    x_enc,
                    z_c,
                    family_ids=torch.tensor([int(step["family_index"])], dtype=torch.long, device=device),
                    step_indices=torch.tensor([int(step["step_index"])], dtype=torch.long, device=device),
                    step_counts=torch.tensor([int(step["step_count"])], dtype=torch.long, device=device),
                )

                pred_u = _single_graph_prediction(outputs["u_logits"], batch_graph.x[:, 2] <= 0.5)
                pred_v = _single_graph_prediction(outputs["v_logits"], batch_graph.x[:, 2] <= 0.5)
                pred_w = _single_graph_prediction(outputs["w_logits"], batch_graph.x[:, 2] > 0.5)
                pred_stop = int(torch.sigmoid(outputs["stop_logits"].view(-1)[0]).item() >= 0.5)
                pred_role = int(torch.argmax(outputs["step_role_logits"], dim=-1)[0].item())

                trace_success &= pred_u == int(step["action_topo"][0].item())
                trace_success &= pred_v == int(step["action_topo"][1].item())
                trace_success &= pred_w == int(step["action_topo"][2].item())
                trace_success &= pred_stop == int(step["stop_token"])
                trace_success &= pred_role == int(step["step_role_index"])

                pred_code = int(
                    policy.predict_geometry_code(
                        batch_graph,
                        x_enc,
                        outputs["graph_context"],
                        torch.tensor([[pred_u, pred_v, pred_w]], dtype=torch.long, device=device),
                        family_ids=torch.tensor([int(step["family_index"])], dtype=torch.long, device=device),
                        step_roles=torch.tensor([int(step["step_role_index"])], dtype=torch.long, device=device),
                        step_indices=torch.tensor([int(step["step_index"])], dtype=torch.long, device=device),
                    )[0].item()
                )
                trace_success &= pred_code == int(step["action_code_id"])
                try:
                    pred_geo = decode_local_dyad_code(
                        current_graph.pos[pred_u].detach().cpu().numpy(),
                        current_graph.pos[pred_v].detach().cpu().numpy(),
                        current_graph.pos[pred_w].detach().cpu().numpy(),
                        policy.action_codebook[pred_code].detach().cpu().numpy(),
                    )
                    next_graph = apply_j_operator(
                        current_graph,
                        pred_u,
                        pred_v,
                        pred_w,
                        pred_geo[0],
                        pred_geo[1],
                    )
                except Exception:
                    trace_valid = False
                    trace_success = False
                    break
                is_valid, _ = validate_graph_structure(next_graph, cfg.get("constraints", {}))
                trace_valid &= bool(is_valid)
                if not trace_valid:
                    trace_success = False
                    break
                current_graph = next_graph

            valid_flags.append(float(trace_valid))
            success_flags.append(float(trace_success))
            family_flags[family_name].append(float(trace_success))

    family_success = {
        family_name: float(np.mean(flags))
        for family_name, flags in sorted(family_flags.items())
    }
    return {
        "trace_count": len(traces),
        "valid_reconstruction_rate": float(np.mean(valid_flags)),
        "reconstruction_success_rate": float(np.mean(success_flags)),
        "family_success_rate": family_success,
    }


def _phase4_step_prediction(policy, graph: Data, z_c: torch.Tensor, step: dict[str, object], device) -> dict[str, object]:
    batch_graph = Batch.from_data_list([copy.deepcopy(graph)]).to(device)
    x_enc = policy.encode_graph(batch_graph)
    outputs = policy.phase4_outputs(
        batch_graph,
        x_enc,
        z_c,
        family_ids=torch.tensor([int(step["family_index"])], dtype=torch.long, device=device),
        step_indices=torch.tensor([int(step["step_index"])], dtype=torch.long, device=device),
        step_counts=torch.tensor([int(step["step_count"])], dtype=torch.long, device=device),
    )

    pred_u = _single_graph_prediction(outputs["u_logits"], batch_graph.x[:, 2] <= 0.5)
    pred_v = _single_graph_prediction(outputs["v_logits"], batch_graph.x[:, 2] <= 0.5)
    pred_w = _single_graph_prediction(outputs["w_logits"], batch_graph.x[:, 2] > 0.5)
    pred_stop = int(torch.sigmoid(outputs["stop_logits"].view(-1)[0]).item() >= 0.5)
    pred_role = int(torch.argmax(outputs["step_role_logits"], dim=-1)[0].item())
    pred_code = int(
        policy.predict_geometry_code(
            batch_graph,
            x_enc,
            outputs["graph_context"],
        torch.tensor([[pred_u, pred_v, pred_w]], dtype=torch.long, device=device),
        family_ids=torch.tensor([int(step["family_index"])], dtype=torch.long, device=device),
        step_roles=torch.tensor([int(step["step_role_index"])], dtype=torch.long, device=device),
        step_indices=torch.tensor([int(step["step_index"])], dtype=torch.long, device=device),
    )[0].item()
    )
    return {
        "batch_graph": batch_graph,
        "x_enc": x_enc,
        "graph_context": outputs["graph_context"],
        "pred": {
            "u": int(pred_u),
            "v": int(pred_v),
            "w": int(pred_w),
            "code": int(pred_code),
            "stop": int(pred_stop),
            "role": int(pred_role),
        },
    }


def _truth_step_payload(step: dict[str, object]) -> dict[str, int]:
    return {
        "u": int(step["action_topo"][0].item()),
        "v": int(step["action_topo"][1].item()),
        "w": int(step["action_topo"][2].item()),
        "code": int(step["action_code_id"]),
        "stop": int(step["stop_token"]),
        "role": int(step["step_role_index"]),
    }


def _prediction_matches(pred: dict[str, int], truth: dict[str, int]) -> dict[str, bool]:
    topology_match = (
        int(pred["u"]) == int(truth["u"])
        and int(pred["v"]) == int(truth["v"])
        and int(pred["w"]) == int(truth["w"])
    )
    return {
        "topology_match": bool(topology_match),
        "code_match": bool(int(pred["code"]) == int(truth["code"])),
        "stop_match": bool(int(pred["stop"]) == int(truth["stop"])),
        "role_match": bool(int(pred["role"]) == int(truth["role"])),
    }


def _matches_are_clean(matches: dict[str, bool]) -> bool:
    return bool(
        matches["topology_match"]
        and matches["code_match"]
        and matches["stop_match"]
        and matches["role_match"]
    )


def _classify_step_failure(
    matches: dict[str, bool],
    *,
    prior_step_failed: bool,
    truth_graph_matches: dict[str, bool],
    invalid_graph: bool,
) -> str:
    if prior_step_failed and _matches_are_clean(truth_graph_matches) and not _matches_are_clean(matches):
        return "state_drift_after_prior_error"
    if not matches["topology_match"]:
        return "topology_error"
    if not matches["code_match"]:
        return "geometry_code_error"
    if not matches["stop_match"] or not matches["role_match"]:
        return "stop_role_error"
    if invalid_graph:
        return "invalid_geometry_or_graph"
    return "success"


def _empty_family_failure_summary() -> dict[str, object]:
    return {
        "trace_count": 0,
        "valid_reconstruction_rate": 0.0,
        "reconstruction_success_rate": 0.0,
        "first_failure_type_counts": {},
        "step_failure_type_counts": {},
    }


def evaluate_multistep_reconstruction_detailed(
    policy,
    curve_encoder,
    paths: Sequence[dict[str, object]],
    cfg: dict,
    device,
    *,
    max_traces: int | None = None,
    max_failure_examples_per_family: int = 20,
) -> dict[str, object]:
    traces = group_paths_by_trace(paths)
    if max_traces is not None:
        traces = traces[: max(0, int(max_traces))]
    if not traces:
        return {
            "trace_count": 0,
            "valid_reconstruction_rate": 0.0,
            "reconstruction_success_rate": 0.0,
            "by_family": {},
            "by_family_step": {},
            "first_failure_type_counts": {},
            "failure_examples": [],
        }

    policy.eval()
    if curve_encoder is not None:
        curve_encoder.eval()

    valid_flags: list[float] = []
    success_flags: list[float] = []
    first_failure_type_counts: Counter[str] = Counter()
    by_family_flags: dict[str, dict[str, list[float] | Counter[str]]] = defaultdict(
        lambda: {
            "valid": [],
            "success": [],
            "first_failure_type_counts": Counter(),
            "step_failure_type_counts": Counter(),
        }
    )
    by_family_step_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_family_step_total: Counter[str] = Counter()
    failure_examples: list[dict[str, object]] = []
    examples_by_family: Counter[str] = Counter()

    with torch.no_grad():
        for trace in traces:
            first = trace[0]
            family_name = str(first["family_id"])
            trace_id = int(first["trace_id"])
            current_graph = copy.deepcopy(first["base_data"])
            target_foot = first["y_foot"].unsqueeze(0).to(device)
            target_knee = first["y_knee"].unsqueeze(0).to(device)
            target_ankle = first["y_ankle"].unsqueeze(0).to(device)
            if curve_encoder is None:
                latent_dim = int(getattr(policy, "curve_latent_dim", 128))
                z_c = torch.zeros((1, latent_dim), dtype=torch.float32, device=device)
            else:
                z_c = curve_encoder(target_foot, target_knee, target_ankle)

            trace_valid = True
            trace_success = True
            prior_step_failed = False
            first_failure_type: str | None = None

            for step in trace:
                prior_step_failed_before_step = bool(prior_step_failed)
                truth = _truth_step_payload(step)
                step_index = int(step["step_index"])
                step_role = str(step["step_role"])
                rollout_prediction = _phase4_step_prediction(policy, current_graph, z_c, step, device)
                truth_prediction = _phase4_step_prediction(policy, step["base_data"], z_c, step, device)
                pred = rollout_prediction["pred"]
                truth_graph_pred = truth_prediction["pred"]
                matches = _prediction_matches(pred, truth)
                truth_graph_matches = _prediction_matches(truth_graph_pred, truth)

                invalid_graph = False
                invalid_reason = None
                next_graph = None
                try:
                    pred_geo = decode_local_dyad_code(
                        current_graph.pos[pred["u"]].detach().cpu().numpy(),
                        current_graph.pos[pred["v"]].detach().cpu().numpy(),
                        current_graph.pos[pred["w"]].detach().cpu().numpy(),
                        policy.action_codebook[pred["code"]].detach().cpu().numpy(),
                    )
                    next_graph = apply_j_operator(
                        current_graph,
                        pred["u"],
                        pred["v"],
                        pred["w"],
                        pred_geo[0],
                        pred_geo[1],
                    )
                    is_valid, reason = validate_graph_structure(next_graph, cfg.get("constraints", {}))
                    invalid_graph = not bool(is_valid)
                    invalid_reason = None if is_valid else str(reason)
                except Exception as exc:
                    invalid_graph = True
                    invalid_reason = str(exc)

                failure_type = _classify_step_failure(
                    matches,
                    prior_step_failed=prior_step_failed,
                    truth_graph_matches=truth_graph_matches,
                    invalid_graph=invalid_graph,
                )
                step_failed = failure_type != "success"
                if step_failed:
                    trace_success = False
                    prior_step_failed = True
                    if first_failure_type is None:
                        first_failure_type = failure_type
                    if examples_by_family[family_name] < int(max_failure_examples_per_family):
                        failure_examples.append(
                            {
                                "trace_id": trace_id,
                                "family": family_name,
                                "step_index": step_index,
                                "step_role": step_role,
                                "truth": truth,
                                "pred": pred,
                                "truth_graph_pred": truth_graph_pred,
                                "topology_match": matches["topology_match"],
                                "code_match": matches["code_match"],
                                "stop_match": matches["stop_match"],
                                "role_match": matches["role_match"],
                                "prior_step_failed": prior_step_failed_before_step,
                                "truth_graph_topology_match": truth_graph_matches["topology_match"],
                                "truth_graph_code_match": truth_graph_matches["code_match"],
                                "truth_graph_stop_match": truth_graph_matches["stop_match"],
                                "truth_graph_role_match": truth_graph_matches["role_match"],
                                "invalid_reason": invalid_reason,
                                "failure_type": failure_type,
                            }
                        )
                        examples_by_family[family_name] += 1

                step_key = f"{family_name}/step{step_index}"
                by_family_step_total[step_key] += 1
                by_family_step_counts[step_key][failure_type] += 1
                by_family_flags[family_name]["step_failure_type_counts"][failure_type] += 1

                if invalid_graph:
                    trace_valid = False
                    trace_success = False
                    if first_failure_type is None:
                        first_failure_type = failure_type
                    break
                if next_graph is not None:
                    current_graph = next_graph

            first_failure_type = first_failure_type or "success"
            first_failure_type_counts[first_failure_type] += 1
            valid_flags.append(float(trace_valid))
            success_flags.append(float(trace_success))
            by_family_flags[family_name]["valid"].append(float(trace_valid))
            by_family_flags[family_name]["success"].append(float(trace_success))
            by_family_flags[family_name]["first_failure_type_counts"][first_failure_type] += 1

    by_family = {}
    for family_name, payload in sorted(by_family_flags.items()):
        valid = list(payload["valid"])
        success = list(payload["success"])
        by_family[family_name] = {
            "trace_count": int(len(success)),
            "valid_reconstruction_rate": float(np.mean(valid)) if valid else 0.0,
            "reconstruction_success_rate": float(np.mean(success)) if success else 0.0,
            "first_failure_type_counts": dict(payload["first_failure_type_counts"]),
            "step_failure_type_counts": dict(payload["step_failure_type_counts"]),
        }

    by_family_step = {}
    for step_key in sorted(by_family_step_total):
        by_family_step[step_key] = {
            "count": int(by_family_step_total[step_key]),
            "failure_type_counts": dict(by_family_step_counts[step_key]),
        }

    return {
        "trace_count": int(len(traces)),
        "valid_reconstruction_rate": float(np.mean(valid_flags)),
        "reconstruction_success_rate": float(np.mean(success_flags)),
        "by_family": by_family,
        "by_family_step": by_family_step,
        "first_failure_type_counts": dict(first_failure_type_counts),
        "failure_examples": failure_examples,
    }


def evaluate_constrained_decoder_reconstruction(
    policy,
    curve_encoder,
    paths: Sequence[dict[str, object]],
    cfg: dict,
    device,
    *,
    top_k: int = 10,
    max_traces: int | None = None,
) -> dict[str, object]:
    traces = group_paths_by_trace(paths)
    if max_traces is not None:
        traces = traces[: max(0, int(max_traces))]
    if not traces:
        return {"trace_count": 0, "by_family": {}}

    policy.eval()
    if curve_encoder is not None:
        curve_encoder.eval()

    by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "trace_count": 0,
            "top1_valid_trace": 0,
            "constrained_valid_trace": 0,
            "constrained_oracle_equiv_trace": 0,
            "constrained_success_trace": 0,
            "constrained_used_non_top1_trace": 0,
        }
    )

    with torch.no_grad():
        for trace in traces:
            first = trace[0]
            family_name = str(first["family_id"])
            current_graph = copy.deepcopy(first["base_data"])
            z_c = _encode_step_target(policy, curve_encoder, first, device)
            top1_valid_trace = True
            constrained_valid_trace = True
            constrained_oracle_equiv_trace = True
            constrained_success_trace = True
            used_non_top1_trace = False

            for step in trace:
                rollout_prediction = _phase4_step_prediction(policy, current_graph, z_c, step, device)
                pred = dict(rollout_prediction["pred"])
                pred_topo = [int(pred["u"]), int(pred["v"]), int(pred["w"])]
                truth = _truth_step_payload(step)
                topology_match = (
                    pred_topo[0] == int(truth["u"])
                    and pred_topo[1] == int(truth["v"])
                    and pred_topo[2] == int(truth["w"])
                )

                top1_choice = constrained_code_choice(
                    current_graph,
                    pred_topo,
                    [int(pred["code"])],
                    policy.action_codebook,
                    cfg.get("constraints", {}),
                )
                top1_valid_trace = bool(top1_valid_trace and top1_choice["valid"])

                candidate_codes = _candidate_rollout_code_ids(
                    policy,
                    rollout_prediction,
                    pred,
                    step,
                    device,
                    max_codes=int(top_k),
                )
                constrained_choice = constrained_code_choice(
                    current_graph,
                    pred_topo,
                    candidate_codes,
                    policy.action_codebook,
                    cfg.get("constraints", {}),
                )
                used_non_top1_trace = bool(used_non_top1_trace or constrained_choice["used_non_top1"])
                constrained_valid_trace = bool(constrained_valid_trace and constrained_choice["valid"])

                positive_ids = {int(code_id) for code_id in (step.get("oracle_positive_ids", []) or [])}
                oracle_equiv = bool(topology_match and int(constrained_choice["code_id"]) in positive_ids)
                constrained_oracle_equiv_trace = bool(constrained_oracle_equiv_trace and oracle_equiv)
                constrained_success_trace = bool(
                    constrained_success_trace
                    and oracle_equiv
                    and int(pred["stop"]) == int(truth["stop"])
                    and int(pred["role"]) == int(truth["role"])
                )
                if not bool(constrained_choice["valid"]):
                    break

                chosen_pred = dict(pred)
                chosen_pred["code"] = int(constrained_choice["code_id"])
                next_graph, _ = _apply_predicted_step(policy, current_graph, chosen_pred, cfg)
                if next_graph is None:
                    constrained_valid_trace = False
                    constrained_success_trace = False
                    break
                current_graph = next_graph

            payload = by_family[family_name]
            payload["trace_count"] += 1
            payload["top1_valid_trace"] += int(top1_valid_trace)
            payload["constrained_valid_trace"] += int(constrained_valid_trace)
            payload["constrained_oracle_equiv_trace"] += int(constrained_oracle_equiv_trace)
            payload["constrained_success_trace"] += int(constrained_success_trace)
            payload["constrained_used_non_top1_trace"] += int(used_non_top1_trace)

    family_report = {}
    for family_name, payload in sorted(by_family.items()):
        trace_count = max(1, int(payload["trace_count"]))
        family_report[family_name] = {
            "trace_count": int(payload["trace_count"]),
            "top1_valid_trace_rate": float(payload["top1_valid_trace"] / trace_count),
            "constrained_valid_trace_rate": float(payload["constrained_valid_trace"] / trace_count),
            "constrained_oracle_equiv_trace_rate": float(payload["constrained_oracle_equiv_trace"] / trace_count),
            "constrained_reconstruction_success_rate": float(payload["constrained_success_trace"] / trace_count),
            "constrained_used_non_top1_trace_rate": float(payload["constrained_used_non_top1_trace"] / trace_count),
        }
    return {
        "trace_count": int(len(traces)),
        "top_k": int(top_k),
        "by_family": family_report,
    }


def _single_graph_prediction(logits: torch.Tensor, valid_mask: torch.Tensor) -> int:
    masked = logits.view(-1).masked_fill(~valid_mask.view(-1), -1e9)
    return int(torch.argmax(masked).item())
