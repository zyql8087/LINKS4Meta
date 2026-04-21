from __future__ import annotations

import copy
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax
from tqdm import tqdm

from src.inverse.action_codebook import (
    allowed_code_ids_for_context,
    attach_action_codebook,
    build_action_codebook,
    codebook_bucket_for_step,
    decode_local_dyad_code,
    default_action_codebook_path,
    encode_local_dyad_code,
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


def extract_multistep_expert_paths(pkl_path: str, output_path: str):
    print(f"Loading multistep IL pkl from {pkl_path} ...")
    with open(pkl_path, "rb") as handle:
        raw_data = pickle.load(handle)
    print(f"Loaded {len(raw_data)} raw samples.")

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
                        "action_code_bucket": codebook_bucket_for_step(family_name, step_role),
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
    codebook = build_action_codebook(expert_paths)
    expert_paths = attach_action_codebook(expert_paths, codebook)
    expert_paths = _attach_doc_masks(expert_paths, codebook)
    torch.save(expert_paths, output_path)
    save_action_codebook(output_path, codebook)
    print(f"[OK] Saved multistep IL cache to {output_path}")
    return expert_paths


def ensure_multistep_expert_paths(pkl_path: str, output_path: str, use_cached: bool = True):
    if use_cached and os.path.exists(output_path):
        print(f"[*] Loading cached multistep IL dataset from {output_path}")
        expert_paths = torch.load(output_path, map_location="cpu", weights_only=False)
        if multistep_paths_have_phase4_format(expert_paths) and os.path.exists(default_action_codebook_path(output_path)):
            return expert_paths
        print("[*] Cached multistep IL dataset is stale; regenerating...")
    return extract_multistep_expert_paths(pkl_path=pkl_path, output_path=output_path)


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
        sample_id = int(item["sample_id"])
        if sample_id in split_ids["train"]:
            mapped["train_indices"].append(local_idx)
        elif sample_id in split_ids["val"]:
            mapped["val_indices"].append(local_idx)
        elif sample_id in split_ids["test"]:
            mapped["test_indices"].append(local_idx)
        else:
            raise ValueError(f"Sample id {sample_id} missing from precomputed split '{source_path}'")

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
    loss_action_code = F.cross_entropy(code_logits, batch["action_code_id"].long())

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
    pred_action_code = torch.argmax(code_logits, dim=-1)

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
        "action_code_accuracy": (pred_action_code == batch["action_code_id"].long()).float().mean(),
    }
    metrics["step_action_accuracy"] = (
        (pred_u == action_topo[:, 0])
        & (pred_v == action_topo[:, 1])
        & (pred_w == action_topo[:, 2])
        & (pred_action_code == batch["action_code_id"].long())
    ).float().mean()
    return metrics


def group_paths_by_trace(paths: Sequence[dict[str, object]]) -> list[list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in paths:
        grouped[int(item["trace_id"])].append(item)
    return [sorted(trace_items, key=lambda item: int(item["step_index"])) for _, trace_items in sorted(grouped.items())]


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
                    )[0].item()
                )
                trace_success &= pred_code == int(step["action_code_id"])
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


def _single_graph_prediction(logits: torch.Tensor, valid_mask: torch.Tensor) -> int:
    masked = logits.view(-1).masked_fill(~valid_mask.view(-1), -1e9)
    return int(torch.argmax(masked).item())
