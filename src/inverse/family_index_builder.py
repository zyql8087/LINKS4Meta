from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from src.config_utils import ensure_parent_dir
from src.inverse.action_codebook import export_action_codebook_v1_json


def _tensor_to_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _graph_summary(graph: Data) -> dict[str, object]:
    edge_index = graph.edge_index.detach().cpu()
    num_edges = int(edge_index.size(1) // 2) if edge_index.numel() > 0 else 0
    fixed_nodes = []
    if graph.x.size(1) >= 3:
        fixed_nodes = [int(idx) for idx in torch.where(graph.x[:, 2] > 0.5)[0].detach().cpu().tolist()]
    return {
        "num_nodes": int(graph.x.size(0)),
        "num_edges": num_edges,
        "fixed_nodes": fixed_nodes,
        "node_coords": graph.pos.detach().cpu().tolist(),
    }


def _motion_cluster(item: dict[str, object]) -> str:
    foot = np.asarray(item["y_foot"], dtype=np.float32)
    knee = np.asarray(item["y_knee"], dtype=np.float32).reshape(-1)
    ankle = np.asarray(item["y_ankle"], dtype=np.float32).reshape(-1)
    descriptor = np.asarray(
        [
            float(foot[:, 0].mean()),
            float(foot[:, 1].mean()),
            float(foot[:, 0].std()),
            float(foot[:, 1].std()),
            float(knee.mean()),
            float(knee.std()),
            float(ankle.mean()),
            float(ankle.std()),
        ],
        dtype=np.float32,
    )
    quantized = [f"{value:.2f}" for value in descriptor]
    return "motion_" + "_".join(quantized)


def _trace_group_meta(trace_steps: Sequence[dict[str, object]]) -> dict[str, object]:
    first = trace_steps[0]
    geometry_ids = [int(step["action_code_id"]) for step in trace_steps]
    return {
        "family_id": str(first["family_id"]),
        "topology_cluster": str(first.get("topology_signature") or "unknown"),
        "geometry_cluster": "|".join(str(code_id) for code_id in geometry_ids),
        "motion_cluster": _motion_cluster(first),
        "step_count": int(first["step_count"]),
        "step_roles": [str(step["step_role"]) for step in trace_steps],
        "seed_type": first.get("seed_type"),
    }


def _trace_id_to_split(split: dict[str, object], step_paths: Sequence[dict[str, object]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for split_name in ("train_indices", "val_indices", "test_indices"):
        label = split_name.replace("_indices", "")
        for idx in split.get(split_name, []):
            mapping[int(step_paths[int(idx)]["sample_id"])] = label
    return mapping


def _group_step_paths(step_paths: Sequence[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in step_paths:
        grouped[int(item["sample_id"])].append(item)
    return {
        trace_id: sorted(items, key=lambda item: int(item["step_index"]))
        for trace_id, items in grouped.items()
    }


def _record_payload(item: dict[str, object], split_name: str) -> dict[str, object]:
    return {
        "sample_id": int(item["sample_id"]),
        "trace_id": int(item["trace_id"]),
        "expert_step_id": int(item["expert_step_id"]),
        "family_id": str(item["family_id"]),
        "seed_type": item.get("seed_type"),
        "step_index": int(item["step_index"]),
        "step_index_1based": int(item["step_index_1based"]),
        "step_count": int(item["step_count"]),
        "step_role": str(item["step_role"]),
        "split": split_name,
        "topology_signature": item.get("topology_signature"),
        "labels": {
            "anchor_i": int(item["action_anchor_i"]),
            "anchor_j": int(item["action_anchor_j"]),
            "support_w": int(item["action_support"]),
            "geom_code": int(item["action_geom_code_id"]),
            "stop": float(item["stop_token"]),
        },
        "masks": {
            "valid_anchor_mask": _tensor_to_list(item.get("valid_anchor_mask")),
            "valid_pair_mask": _tensor_to_list(item.get("valid_pair_mask")),
            "valid_geom_mask": _tensor_to_list(item.get("valid_geom_mask")),
            "semantic_mask": _tensor_to_list(item.get("semantic_mask")),
        },
        "seed_graph": _graph_summary(item["seed_graph"]),
        "trace_prefix": _graph_summary(item["trace_prefix"]),
        "target_curves": {
            "foot": _tensor_to_list(item["target_curves"]["foot"]),
            "knee": _tensor_to_list(item["target_curves"]["knee"]),
            "ankle": _tensor_to_list(item["target_curves"]["ankle"]),
        },
        "generation_trace": item.get("generation_trace"),
    }


def build_family_index_artifacts(
    step_paths: Sequence[dict[str, object]],
    *,
    split: dict[str, object],
    codebook: dict[str, object],
    output_dir: str | Path,
    dataset_path: str | Path,
    export_jsonl: bool = True,
) -> dict[str, object]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    index_pt_path = output_root / "family_step_index_v1.pt"
    index_jsonl_path = output_root / "family_step_index_v1.jsonl"
    split_json_path = output_root / "family_group_split_v1.json"
    manifest_path = output_root / "family_index_manifest_v1.json"
    codebook_json_path = output_root / "geom_codebook_v1.json"

    trace_groups = _group_step_paths(step_paths)
    trace_split = _trace_id_to_split(split, step_paths)
    trace_assignments = []
    group_meta = {}
    split_members = {"train": [], "val": [], "test": []}
    family_counts = defaultdict(int)

    for trace_id, items in sorted(trace_groups.items()):
        split_name = trace_split[int(trace_id)]
        meta = _trace_group_meta(items)
        split_members[split_name].append(int(trace_id))
        family_counts[meta["family_id"]] += 1
        trace_assignments.append(
            {
                "sample_id": int(trace_id),
                "split": split_name,
                **meta,
            }
        )
        group_meta[str(trace_id)] = meta

    serialized_records = []
    step_split_lookup = _trace_id_to_split(split, step_paths)
    for item in step_paths:
        serialized_records.append(_record_payload(item, step_split_lookup[int(item["sample_id"])]))

    torch.save(
        {
            "version": "family_step_index_v1",
            "records": list(step_paths),
            "split": split,
            "codebook": codebook,
            "serialized_records": serialized_records,
        },
        index_pt_path,
    )

    if export_jsonl:
        ensure_parent_dir(index_jsonl_path)
        with index_jsonl_path.open("w", encoding="utf-8") as handle:
            for record in serialized_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_payload = {
        "version": "family_group_split_v1",
        "split_source": split.get("split_source"),
        "source_path": split.get("source_path"),
        "split_seed": split.get("split_seed"),
        "train": split_members["train"],
        "val": split_members["val"],
        "test": split_members["test"],
        "group_meta": group_meta,
        "trace_assignments": trace_assignments,
    }
    split_json_path.write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    export_action_codebook_v1_json(dataset_path, codebook, step_paths=step_paths, output_path=codebook_json_path)

    manifest = {
        "version": "family_index_manifest_v1",
        "num_step_records": len(step_paths),
        "num_trace_records": len(trace_groups),
        "family_trace_counts": dict(sorted(family_counts.items())),
        "paths": {
            "index_pt": str(index_pt_path),
            "index_jsonl": str(index_jsonl_path),
            "split_json": str(split_json_path),
            "codebook_json": str(codebook_json_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        **manifest,
        "paths": manifest["paths"],
        "split_payload": split_payload,
    }
