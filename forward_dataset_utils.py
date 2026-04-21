from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch_geometric.data import Data

from src.kinematics_extract import extract_kinematics

FAMILY_TO_ID = {
    "6bar": 0,
    "7bar": 1,
    "8bar": 2,
    "9bar": 3,
}
ID_TO_FAMILY = {value: key for key, value in FAMILY_TO_ID.items()}
MAX_GEOMETRY_DESCRIPTOR_DIM = 16


def family_name_to_id(family_name: str | None) -> int:
    if family_name is None:
        return -1
    return int(FAMILY_TO_ID.get(str(family_name), -1))


def family_id_to_name(family_id: int) -> str:
    return ID_TO_FAMILY.get(int(family_id), "unknown")


def extract_family_name(sample: Mapping[str, object]) -> str | None:
    family_name = sample.get("family_id") or sample.get("family")
    if family_name is None:
        return None
    return str(family_name)


def _semantic_role_matrix(num_nodes: int, analysis: Mapping[str, object]) -> np.ndarray:
    role_matrix = np.zeros((num_nodes, 4), dtype=np.float32)
    for col, key in enumerate(("hip", "knee", "ankle", "foot")):
        idx = analysis.get(key)
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < num_nodes:
            role_matrix[idx, col] = 1.0
    return role_matrix


def build_step_context(sample: Mapping[str, object]) -> np.ndarray:
    step_roles = list(sample.get("step_roles", []))
    step_count = int(sample.get("step_count", len(step_roles) or 0))
    aux_steps = sum(role == "aux" for role in step_roles)
    semantic_steps = sum(role == "semantic" for role in step_roles)
    return np.array([step_count, aux_steps, semantic_steps], dtype=np.float32)


def _padded_geometry_descriptor(sample: Mapping[str, object]) -> np.ndarray:
    geometry = np.asarray(
        sample.get("descriptors", {}).get("geometry", []),
        dtype=np.float32,
    )
    out = np.zeros((MAX_GEOMETRY_DESCRIPTOR_DIM,), dtype=np.float32)
    dim = min(len(geometry), MAX_GEOMETRY_DESCRIPTOR_DIM)
    if dim > 0:
        out[:dim] = geometry[:dim]
    return out


def build_retrieval_feature(sample: Mapping[str, object]) -> np.ndarray:
    family_id = family_name_to_id(extract_family_name(sample))
    family_one_hot = np.zeros((len(FAMILY_TO_ID),), dtype=np.float32)
    if family_id >= 0:
        family_one_hot[family_id] = 1.0
    step_context = build_step_context(sample)
    if step_context[0] > 0:
        step_context = step_context / max(step_context[0], 1.0)
    geometry = _padded_geometry_descriptor(sample)
    return np.concatenate([family_one_hot, step_context, geometry], axis=0).astype(np.float32)


def _curve_targets(sample: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(key in sample for key in ("foot_curve", "knee_curve", "ankle_curve")):
        return (
            np.asarray(sample["foot_curve"], dtype=np.float32),
            np.asarray(sample["knee_curve"], dtype=np.float32),
            np.asarray(sample["ankle_curve"], dtype=np.float32),
        )
    foot_traj, knee_angle, ankle_angle = extract_kinematics(sample)
    return (
        np.asarray(foot_traj, dtype=np.float32),
        np.asarray(knee_angle, dtype=np.float32),
        np.asarray(ankle_angle, dtype=np.float32),
    )


def sample_to_pyg_data(sample: Mapping[str, object], sample_id: int) -> Data:
    A = np.asarray(sample["A"])
    x0 = np.asarray(sample["x0"], dtype=np.float32)
    types = np.asarray(sample["types"])
    analysis = sample["analysis"]

    is_fixed = (types == 1).astype(np.float32)
    is_grounded = np.zeros_like(is_fixed, dtype=np.float32)
    is_grounded[0] = 1.0
    semantic_roles = _semantic_role_matrix(x0.shape[0], analysis)
    x_features = np.column_stack([x0, is_fixed, is_grounded, semantic_roles])

    edges = np.array(np.where(A)).T
    edge_index = edges.T
    keypoints = np.array([analysis["foot"], analysis["knee"], analysis["ankle"]], dtype=np.int64)
    foot_traj, knee_angle, ankle_angle = _curve_targets(sample)

    family_name = extract_family_name(sample)
    family_id = family_name_to_id(family_name)
    step_context = build_step_context(sample)
    retrieval_feature = build_retrieval_feature(sample)

    mask_hip = semantic_roles[:, 0] > 0.5
    mask_knee = semantic_roles[:, 1] > 0.5
    mask_ankle = semantic_roles[:, 2] > 0.5
    mask_foot = semantic_roles[:, 3] > 0.5

    return Data(
        x=torch.tensor(x_features, dtype=torch.float32),
        pos=torch.tensor(x0, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        keypoints=torch.tensor(keypoints, dtype=torch.long),
        sample_id=torch.tensor([int(sample.get("id", sample_id))], dtype=torch.long),
        family_id=torch.tensor([family_id], dtype=torch.long),
        step_context=torch.tensor(step_context[None, :], dtype=torch.float32),
        retrieval_feature=torch.tensor(retrieval_feature[None, :], dtype=torch.float32),
        mask_hip=torch.tensor(mask_hip, dtype=torch.bool),
        mask_knee=torch.tensor(mask_knee, dtype=torch.bool),
        mask_ankle=torch.tensor(mask_ankle, dtype=torch.bool),
        mask_foot=torch.tensor(mask_foot, dtype=torch.bool),
        y_foot=torch.tensor(foot_traj, dtype=torch.float32),
        y_knee=torch.tensor(knee_angle, dtype=torch.float32),
        y_ankle=torch.tensor(ankle_angle, dtype=torch.float32),
    )
