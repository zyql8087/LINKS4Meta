from __future__ import annotations

# src/inverse/rl_env.py
# RL MDP 环境：以冻结的前向代理模型为评估器，实现自回归图构建的 MDP
#
# State  : 当前机构图 (PyG Data) + 目标曲线条件向量 z_c + Residual 差值
# Action : J-Operator 动作 (u,v,w) 选择 + C-VAE 几何坐标 (n1, n2)
# Reward : R_sim (Chamfer + MSE) + R_physics (残差惩罚) + R_valid (卡死惩罚)

import copy
import numpy as np
import torch
import yaml
from typing import Optional
from torch_geometric.data import Data, Batch

from src.inverse.experiment_utils import compute_joint_metrics_batch, compute_reward_batch
from src.inverse.readout_assignment import (
    RuleBasedReadoutAssignment,
    SurrogateTargetReadoutAssignment,
    assignment_to_masks,
)


_DEFAULT_READOUT_ASSIGNER = RuleBasedReadoutAssignment(top_k=3)


# ──────────────────────────────────────────────────────────────────────────────
# 前向代理模型加载（冻结）
# ──────────────────────────────────────────────────────────────────────────────
def _extract_model_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported forward checkpoint type: {type(checkpoint)!r}")


def inspect_forward_checkpoint_compatibility(checkpoint_state: dict, cfg: dict) -> dict:
    encoder_cfg = dict(cfg.get("encoder", {}))
    decoder_cfg = dict(cfg.get("decoder", {}))

    expected_node_input_dim = int(encoder_cfg.get("node_input_dim", 4))
    expected_hidden_dim = int(encoder_cfg.get("hidden_dim", decoder_cfg.get("hidden_dim", 128)))
    expected_family_embedding_dim = int(decoder_cfg.get("family_embedding_dim", 0))
    expected_step_context_hidden_dim = int(decoder_cfg.get("step_context_hidden_dim", 0))
    expected_decoder_input_dim = (
        expected_hidden_dim + expected_family_embedding_dim + expected_step_context_hidden_dim
    )

    node_weight = checkpoint_state.get("encoder.pre_mlp_nodes.weight")
    decoder_weight = checkpoint_state.get("decoder_foot.linear_li.0.weight")
    actual_node_input_dim = (
        int(node_weight.size(1))
        if isinstance(node_weight, torch.Tensor) and node_weight.dim() == 2
        else None
    )
    actual_decoder_input_dim = (
        int(decoder_weight.size(1))
        if isinstance(decoder_weight, torch.Tensor) and decoder_weight.dim() == 2
        else None
    )
    has_family_embedding = "family_embedding.weight" in checkpoint_state
    has_step_context_encoder = any(
        str(key).startswith("step_context_encoder.") for key in checkpoint_state.keys()
    )

    issues = []
    if actual_node_input_dim != expected_node_input_dim:
        issues.append(
            f"node_input_dim mismatch: checkpoint={actual_node_input_dim}, config={expected_node_input_dim}"
        )
    if expected_family_embedding_dim > 0 and not has_family_embedding:
        issues.append("checkpoint is missing decoder family_embedding weights")
    if expected_step_context_hidden_dim > 0 and not has_step_context_encoder:
        issues.append("checkpoint is missing decoder step_context_encoder weights")
    if actual_decoder_input_dim != expected_decoder_input_dim:
        issues.append(
            f"decoder input mismatch: checkpoint={actual_decoder_input_dim}, config={expected_decoder_input_dim}"
        )

    if actual_node_input_dim == 4 and not has_family_embedding and not has_step_context_encoder:
        checkpoint_variant = "legacy_4d_no_context"
    elif (
        actual_node_input_dim == expected_node_input_dim
        and has_family_embedding == (expected_family_embedding_dim > 0)
        and has_step_context_encoder == (expected_step_context_hidden_dim > 0)
    ):
        checkpoint_variant = "semantic_context"
    else:
        checkpoint_variant = "mixed_or_unknown"

    return {
        "checkpoint_variant": checkpoint_variant,
        "expected_node_input_dim": expected_node_input_dim,
        "actual_node_input_dim": actual_node_input_dim,
        "expected_decoder_input_dim": expected_decoder_input_dim,
        "actual_decoder_input_dim": actual_decoder_input_dim,
        "has_family_embedding": bool(has_family_embedding),
        "has_step_context_encoder": bool(has_step_context_encoder),
        "issues": issues,
    }


def load_frozen_surrogate(model_path: str, config_path: str, device):
    """加载并冻结前向 BioKinematicsGNN 作为 Reward 计算器"""
    from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = BioKinematicsGNN(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = _extract_model_state_dict(ckpt)
    compatibility = inspect_forward_checkpoint_compatibility(state_dict, cfg)
    if compatibility["issues"]:
        issue_text = "; ".join(str(item) for item in compatibility["issues"])
        raise RuntimeError(
            f"Forward checkpoint/config mismatch for '{model_path}'. "
            f"variant={compatibility['checkpoint_variant']}, "
            f"issues={issue_text}. "
            f"Use a checkpoint exported with the current 8-dim semantic/context forward config "
            f"or retrain the forward model for '{config_path}'."
        )
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load forward checkpoint '{model_path}' even though the high-level shape signature looked compatible: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[Surrogate] Loaded frozen forward model from '{model_path}'")
    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Reward 计算
# ──────────────────────────────────────────────────────────────────────────────
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flag_is_true(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().view(-1)[0].item())
    return bool(value)


def _build_phase3_step_context(step_index: int, expected_j_steps: int) -> torch.Tensor:
    semantic_steps = 1 if int(step_index) >= int(expected_j_steps) else 0
    aux_steps = max(0, int(step_index) - semantic_steps)
    return torch.tensor(
        [[float(step_index), float(aux_steps), float(semantic_steps)]],
        dtype=torch.float32,
    )


def _build_surrogate_readout_assigner(
    surrogate,
    reward_cfg: Optional[dict],
    device,
    *,
    family_index: int,
    step_index: int,
    expected_j_steps: int,
):
    if surrogate is None:
        return None
    return SurrogateTargetReadoutAssignment(
        surrogate,
        top_k=3,
        batch_size=64,
        metric_cfg=reward_cfg or {},
        device=device,
        family_index=int(family_index),
        step_index=int(step_index),
        expected_j_steps=int(expected_j_steps),
    )


def _infer_semantic_masks(
    graph_data: Data,
    *,
    target: Optional[dict] = None,
    motion=None,
    allow_assignment_fallback: bool = True,
    readout_assigner=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(graph_data.x.size(0))
    device = graph_data.x.device
    semantic_dirty = _flag_is_true(getattr(graph_data, "semantic_dirty", None))
    mask_hip = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_knee = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_ankle = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_foot = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    if not semantic_dirty:
        cached_masks = {
            "hip": getattr(graph_data, "mask_hip", None),
            "knee": getattr(graph_data, "mask_knee", None),
            "ankle": getattr(graph_data, "mask_ankle", None),
            "foot": getattr(graph_data, "mask_foot", None),
        }
        if any(mask is not None for mask in cached_masks.values()):
            if cached_masks["hip"] is not None:
                mask_hip = cached_masks["hip"].to(device=device, dtype=torch.bool).view(-1)[:num_nodes]
            if cached_masks["knee"] is not None:
                mask_knee = cached_masks["knee"].to(device=device, dtype=torch.bool).view(-1)[:num_nodes]
            if cached_masks["ankle"] is not None:
                mask_ankle = cached_masks["ankle"].to(device=device, dtype=torch.bool).view(-1)[:num_nodes]
            if cached_masks["foot"] is not None:
                mask_foot = cached_masks["foot"].to(device=device, dtype=torch.bool).view(-1)[:num_nodes]
            if mask_knee.any() and mask_ankle.any() and mask_foot.any():
                return mask_hip, mask_knee, mask_ankle, mask_foot

        if _flag_is_true(getattr(graph_data, "semantic_feature_layout", None)) and graph_data.x.size(-1) >= 8:
            semantic = graph_data.x[:, -4:]
            return (
                semantic[:, 0] > 0.5,
                semantic[:, 1] > 0.5,
                semantic[:, 2] > 0.5,
                semantic[:, 3] > 0.5,
            )

        keypoints = getattr(graph_data, "keypoints", None)
        if keypoints is not None:
            keypoints = keypoints.view(-1).long()
            if keypoints.numel() >= 3:
                foot_idx = int(keypoints[0].item())
                knee_idx = int(keypoints[1].item())
                ankle_idx = int(keypoints[2].item())
                if 0 <= knee_idx < num_nodes:
                    mask_knee[knee_idx] = True
                if 0 <= ankle_idx < num_nodes:
                    mask_ankle[ankle_idx] = True
                if 0 <= foot_idx < num_nodes:
                    mask_foot[foot_idx] = True

        if not mask_knee.any() and hasattr(graph_data, "knee_idx") and graph_data.knee_idx is not None:
            knee_idx = int(graph_data.knee_idx.view(-1)[0].item())
            if 0 <= knee_idx < num_nodes:
                mask_knee[knee_idx] = True

    if allow_assignment_fallback and (not mask_knee.any() or not mask_ankle.any() or not mask_foot.any()):
        assigner = readout_assigner or _DEFAULT_READOUT_ASSIGNER
        assignment = assigner.assign(graph_data, target=target, motion=motion)
        if assignment is not None:
            inferred = assignment_to_masks(assignment, num_nodes, device=device)
            if not mask_hip.any():
                mask_hip |= inferred["hip"]
            if not mask_knee.any():
                mask_knee |= inferred["knee"]
            if not mask_ankle.any():
                mask_ankle |= inferred["ankle"]
            if not mask_foot.any():
                mask_foot |= inferred["foot"]

    return mask_hip, mask_knee, mask_ankle, mask_foot


def _prepare_graph_for_surrogate(
    graph_data: Data,
    *,
    family_index: int,
    step_index: int,
    expected_j_steps: int,
    target: Optional[dict] = None,
    motion=None,
    readout_assigner=None,
) -> Data:
    prepared = copy.deepcopy(graph_data)
    x = prepared.x.float()
    if x.size(-1) < 4:
        raise ValueError(f"Expected at least 4 node features, got {x.size(-1)}")
    mask_hip, mask_knee, mask_ankle, mask_foot = _infer_semantic_masks(
        prepared,
        target=target,
        motion=motion,
        readout_assigner=readout_assigner,
    )
    semantic = torch.stack(
        [
            mask_hip.float(),
            mask_knee.float(),
            mask_ankle.float(),
            mask_foot.float(),
        ],
        dim=-1,
    )
    x_aug = torch.cat([x[:, :4], semantic], dim=-1)
    prepared.semantic_feature_layout = torch.tensor([1], dtype=torch.long)
    prepared.semantic_dirty = torch.tensor([False], dtype=torch.bool)

    prepared.x = x_aug
    prepared.family_id = torch.tensor([int(family_index)], dtype=torch.long)
    prepared.step_context = _build_phase3_step_context(step_index, expected_j_steps)
    prepared.mask_hip = mask_hip
    prepared.mask_knee = mask_knee
    prepared.mask_ankle = mask_ankle
    prepared.mask_foot = mask_foot
    return prepared


def _sorted_undirected_edges(edge_index: torch.Tensor):
    edges = set()
    for u, v in edge_index.detach().cpu().numpy().T.tolist():
        if u == v:
            continue
        a, b = sorted((int(u), int(v)))
        edges.add((a, b))
    return sorted(edges)


def _orientation(a, b, c, eps: float):
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a, b, c, eps: float):
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(p1, p2, p3, p4, eps: float):
    o1 = _orientation(p1, p2, p3, eps)
    o2 = _orientation(p1, p2, p4, eps)
    o3 = _orientation(p3, p4, p1, eps)
    o4 = _orientation(p3, p4, p2, eps)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3, eps):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4, eps):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1, eps):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2, eps):
        return True
    return False


def validate_graph_structure(graph_data: Data, constraint_cfg: Optional[dict] = None):
    constraint_cfg = constraint_cfg or {}
    min_link_length = float(constraint_cfg.get('min_link_length', 0.05))
    min_node_distance = float(constraint_cfg.get('min_node_distance', 1e-3))
    eps = float(constraint_cfg.get('intersection_eps', 1e-8))

    pos = _to_numpy(graph_data.pos if getattr(graph_data, 'pos', None) is not None else graph_data.x[:, :2])
    x = _to_numpy(graph_data.x)
    edge_index = graph_data.edge_index
    if pos.ndim != 2 or pos.shape[1] != 2:
        return False, {'reason': 'invalid_position_shape'}
    if x.shape[0] != pos.shape[0]:
        return False, {'reason': 'x_pos_size_mismatch'}
    if edge_index.numel() == 0:
        return False, {'reason': 'empty_edge_set'}

    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if np.linalg.norm(pos[i] - pos[j]) < min_node_distance:
                return False, {'reason': 'duplicate_nodes', 'nodes': (i, j)}

    undirected_edges = _sorted_undirected_edges(edge_index)
    if not undirected_edges:
        return False, {'reason': 'no_undirected_edges'}

    for u, v in undirected_edges:
        if np.linalg.norm(pos[u] - pos[v]) < min_link_length:
            return False, {'reason': 'short_edge', 'edge': (u, v)}

    for idx_a, (u1, v1) in enumerate(undirected_edges):
        p1, p2 = pos[u1], pos[v1]
        for u2, v2 in undirected_edges[idx_a + 1:]:
            if len({u1, v1, u2, v2}) < 4:
                continue
            p3, p4 = pos[u2], pos[v2]
            if _segments_intersect(p1, p2, p3, p4, eps):
                return False, {'reason': 'edge_intersection', 'edges': ((u1, v1), (u2, v2))}

    if hasattr(graph_data, 'keypoints') and graph_data.keypoints is not None:
        keypoints = _to_numpy(graph_data.keypoints).reshape(-1).astype(int).tolist()
        if any(idx < 0 or idx >= pos.shape[0] for idx in keypoints):
            return False, {'reason': 'invalid_keypoints', 'keypoints': keypoints}
        if len(set(keypoints)) != len(keypoints):
            return False, {'reason': 'duplicate_keypoints', 'keypoints': keypoints}

    return True, {'reason': 'ok'}


def compute_reward(surrogate, graph_data: Data, target: dict,
                   reward_cfg: dict, device, constraint_cfg: Optional[dict] = None, readout_assigner=None) -> tuple:
    """
    Single-graph reward computation (fallback path).
    """
    is_valid, valid_info = validate_graph_structure(graph_data, constraint_cfg)
    if not is_valid:
        return reward_cfg.get('penalty_locking', -100.0), False, valid_info

    try:
        assigner = readout_assigner or _build_surrogate_readout_assigner(
            surrogate,
            reward_cfg,
            device,
            family_index=-1,
            step_index=0,
            expected_j_steps=1,
        )
        prepared_graph = _prepare_graph_for_surrogate(
            graph_data,
            family_index=-1,
            step_index=0,
            expected_j_steps=1,
            target=target,
            readout_assigner=assigner,
        )
        batch = Batch.from_data_list([prepared_graph]).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch)
        pred_foot  = pred_foot.squeeze(0).cpu()
        pred_knee  = pred_knee.squeeze(0).cpu()
        pred_ankle = pred_ankle.squeeze(0).cpu()
    except Exception:
        return reward_cfg.get('penalty_locking', -100.0), False, {}

    reward_t, metrics = compute_reward_batch(
        pred_foot.unsqueeze(0),
        pred_knee.unsqueeze(0),
        pred_ankle.unsqueeze(0),
        target,
        reward_cfg,
    )
    total_reward = float(reward_t.squeeze(0).item())
    info = {
        'pred_foot': pred_foot,
        'pred_knee': pred_knee,
        'pred_ankle': pred_ankle,
        'joint_score': float(metrics['joint_score'].item()),
        'foot_nrmse': float(metrics['foot_nrmse'].item()),
        'knee_nrmse': float(metrics['knee_nrmse'].item()),
        'ankle_nrmse': float(metrics['ankle_nrmse'].item()),
        'smoothness': float(metrics['smoothness'].item()),
        'reward': total_reward,
    }
    return total_reward, True, info


def batch_compute_rewards(surrogate, graphs: list, target: dict,
                          reward_cfg: dict, device, constraint_cfg: Optional[dict] = None, readout_assigner=None) -> list:
    """
    Batch-compute rewards for a list of graphs in ONE GPU call.
    Returns list of (reward, valid) tuples.
    """
    if not graphs:
        return []

    penalty = reward_cfg.get('penalty_locking', -100.0)
    valid_graphs = []
    valid_indices = []
    results = [(float(penalty), False)] * len(graphs)

    for idx, graph in enumerate(graphs):
        is_valid, _ = validate_graph_structure(graph, constraint_cfg)
        if is_valid:
            valid_indices.append(idx)
            valid_graphs.append(graph)

    if not valid_graphs:
        return results

    prepared_graphs = [
        _prepare_graph_for_surrogate(
            graph,
            family_index=-1,
            step_index=0,
            expected_j_steps=1,
            target=target,
            readout_assigner=readout_assigner or _build_surrogate_readout_assigner(
                surrogate,
                reward_cfg,
                device,
                family_index=-1,
                step_index=0,
                expected_j_steps=1,
            ),
        )
        for graph in valid_graphs
    ]

    try:
        batch = Batch.from_data_list(prepared_graphs).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch)
        reward_t, _ = compute_reward_batch(
            pred_foot.cpu(), pred_knee.cpu(), pred_ankle.cpu(), target, reward_cfg
        )
        for batch_idx, graph_idx in enumerate(valid_indices):
            results[graph_idx] = (float(reward_t[batch_idx].item()), True)
        return results
    except Exception:
        for idx, prepared_graph in zip(valid_indices, prepared_graphs):
            reward, valid, _ = compute_reward(
                surrogate,
                prepared_graph,
                target,
                reward_cfg,
                device,
                constraint_cfg=constraint_cfg,
                readout_assigner=readout_assigner,
            )
            results[idx] = (reward if valid else penalty, valid)
        return results


def batch_compute_phase5_rewards(
    surrogate,
    graphs: list,
    target: dict,
    reward_cfg: dict,
    device,
    *,
    step_indices: list[int],
    stop_flags: list[bool],
    expected_j_steps: int,
    family_index: int = -1,
    constraint_cfg: Optional[dict] = None,
    readout_assigner=None,
) -> tuple[list[tuple[float, bool]], list[dict[str, float]]]:
    """
    Strategy-A phase5 reward:
      - illegal / invalid graph: immediate negative terminal reward
      - intermediate valid step: small alive bonus or 0
      - terminal valid graph: foot/knee/ankle match - uncertainty - step penalty
    """
    if not graphs:
        return [], []

    r_invalid_penalty = float(reward_cfg.get('r_invalid_penalty', -1.0))
    alive_bonus = float(reward_cfg.get('alive_bonus', 0.0))
    uncertainty_weight = float(reward_cfg.get('lambda_uncertainty', 0.0))
    step_penalty_weight = float(reward_cfg.get('lambda_step', 0.05))
    w_foot = float(reward_cfg.get('w_foot', 0.5))
    w_knee = float(reward_cfg.get('w_knee', 0.25))
    w_ankle = float(reward_cfg.get('w_ankle', 0.25))

    valid_flags = []
    valid_graphs = []
    valid_graph_indices = []
    for graph_idx, graph in enumerate(graphs):
        is_valid, _ = validate_graph_structure(graph, constraint_cfg)
        valid_flags.append(bool(is_valid))
        if is_valid:
            valid_graphs.append(graph)
            valid_graph_indices.append(graph_idx)

    base_rewards = [r_invalid_penalty] * len(graphs)
    metric_payloads = [
        {
            'r_foot': 0.0,
            'r_knee': 0.0,
            'r_ankle': 0.0,
            'r_uncertainty': 0.0,
            'r_step_penalty': 0.0,
            'reward_total': r_invalid_penalty,
            'valid': 0.0,
            'terminal': 0.0,
        }
        for _ in graphs
    ]

    if valid_graphs:
        prepared_graphs = []
        for graph, graph_idx in zip(valid_graphs, valid_graph_indices):
            step_index = int(step_indices[graph_idx])
            assigner = readout_assigner or _build_surrogate_readout_assigner(
                surrogate,
                reward_cfg,
                device,
                family_index=int(family_index),
                step_index=step_index,
                expected_j_steps=int(expected_j_steps),
            )
            prepared_graphs.append(
                _prepare_graph_for_surrogate(
                    graph,
                    family_index=family_index,
                    step_index=step_index,
                    expected_j_steps=int(expected_j_steps),
                    target=target,
                    readout_assigner=assigner,
                )
            )
        batch = Batch.from_data_list(prepared_graphs).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch)
        pred_foot_cpu = pred_foot.cpu()
        pred_knee_cpu = pred_knee.cpu()
        pred_ankle_cpu = pred_ankle.cpu()
        metrics = compute_joint_metrics_batch(
            pred_foot_cpu,
            pred_knee_cpu,
            pred_ankle_cpu,
            target,
            reward_cfg,
        )
        for local_idx, graph_idx in enumerate(valid_graph_indices):
            step_index = int(step_indices[graph_idx])
            stop_selected = bool(stop_flags[graph_idx])
            foot_reward = max(0.0, 1.0 - float(metrics['foot_chamfer_norm'][local_idx].item()))
            knee_reward = max(0.0, 1.0 - float(metrics['knee_nrmse'][local_idx].item()))
            ankle_reward = max(0.0, 1.0 - float(metrics['ankle_nrmse'][local_idx].item()))
            uncertainty = 0.0
            step_penalty = step_penalty_weight * float(abs(int(expected_j_steps) - step_index))
            if stop_selected or step_index >= int(expected_j_steps):
                total_reward = (
                    w_foot * foot_reward
                    + w_knee * knee_reward
                    + w_ankle * ankle_reward
                    - uncertainty_weight * uncertainty
                    - step_penalty
                )
            else:
                total_reward = alive_bonus
            base_rewards[graph_idx] = total_reward
            metric_payloads[graph_idx] = {
                'r_foot': foot_reward,
                'r_knee': knee_reward,
                'r_ankle': ankle_reward,
                'r_uncertainty': -uncertainty_weight * uncertainty,
                'r_step_penalty': -step_penalty,
                'reward_total': total_reward,
                'valid': 1.0,
                'terminal': 1.0 if (stop_selected or step_index >= int(expected_j_steps)) else 0.0,
                'joint_score': float(metrics['joint_score'][local_idx].item()),
                'foot_score': float(metrics['foot_score'][local_idx].item()),
                'knee_nrmse': float(metrics['knee_nrmse'][local_idx].item()),
                'ankle_nrmse': float(metrics['ankle_nrmse'][local_idx].item()),
            }

    return list(zip(base_rewards, valid_flags)), metric_payloads


# ──────────────────────────────────────────────────────────────────────────────
# J-Operator：将新 Dyad 嵌入当前图
# ──────────────────────────────────────────────────────────────────────────────
FIXED_4BAR_ADJACENCY = [
    (0, 1), (1, 0),   # 地铰-曲柄
    (1, 2), (2, 1),   # 曲柄-连杆
    (2, 3), (3, 2),   # 连杆-摇杆
    (3, 0), (0, 3),   # 摇杆-地铰
]

def apply_j_operator(graph_data: Data, u: int, v: int, w: int,
                     n1_pos: np.ndarray, n2_pos: np.ndarray) -> Data:
    """
    将 J-Operator 应用到当前图:
      - 在节点 u, v（动节点）中间插入 n1
      - n2 连接 n1 和固定节点 w
      返回包含 n1, n2 的新图 Data
    """
    old_x   = graph_data.x.detach().cpu().numpy()
    old_pos = graph_data.pos.detach().cpu().numpy()
    N = old_x.shape[0]
    feat_dim = old_x.shape[1]

    n1_feat = np.zeros((1, feat_dim), dtype=np.float32)
    n2_feat = np.zeros((1, feat_dim), dtype=np.float32)
    n1_feat[0, :2] = n1_pos[:2]
    n2_feat[0, :2] = n2_pos[:2]

    new_x   = np.vstack([old_x, n1_feat, n2_feat])
    new_pos = np.vstack([old_pos, n1_pos.reshape(1,2), n2_pos.reshape(1,2)])

    n1_idx, n2_idx = N, N + 1

    old_ei = graph_data.edge_index.detach().cpu().numpy().T.tolist()
    new_edges = old_ei + [
        [u, n1_idx], [n1_idx, u],
        [v, n1_idx], [n1_idx, v],
        [n1_idx, n2_idx], [n2_idx, n1_idx],
        [w, n2_idx], [n2_idx, w],
    ]

    keypoints = None
    semantic_dirty = False
    if hasattr(graph_data, 'keypoints') and graph_data.keypoints is not None:
        semantic_dirty = True
    elif hasattr(graph_data, 'knee_idx') and graph_data.knee_idx is not None:
        knee_idx = int(graph_data.knee_idx.reshape(-1)[0].item())
        if knee_idx >= 0:
            keypoints = torch.tensor([n2_idx, knee_idx, n1_idx], dtype=torch.long)

    out_kwargs = dict(
        x=torch.tensor(new_x, dtype=torch.float32),
        pos=torch.tensor(new_pos, dtype=torch.float32),
        edge_index=torch.tensor(new_edges, dtype=torch.long).T,
    )
    if keypoints is not None:
        out_kwargs["keypoints"] = keypoints
    out = Data(**out_kwargs)
    if hasattr(graph_data, 'knee_idx') and graph_data.knee_idx is not None:
        out.knee_idx = graph_data.knee_idx.clone().detach()
    if semantic_dirty:
        out.semantic_dirty = torch.tensor([True], dtype=torch.bool)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MDP 环境类（延迟 reward 计算版本）
# ──────────────────────────────────────────────────────────────────────────────
class MechanismEnv:
    """
    自回归机构图构建 MDP 环境

    Episode 流程：
      reset(target, base_graph) → 初始 state (4杆)
      step(action) → next_state, reward, done, info
      - action = {'u':int, 'v':int, 'w':int,
                   'n1': np.ndarray(2,), 'n2': np.ndarray(2,)}
      达到 max_steps 或 reward 超阈值后 done=True

    Deferred reward mode:
      step() 只做 J-Operator（纯 CPU），不调 surrogate → reward 占位为 0
      episode 结束后调 compute_episode_rewards() 一次性 batch 计算
    """

    def __init__(self, surrogate, reward_cfg: dict, max_steps: int = 5, device='cpu', constraint_cfg: Optional[dict] = None):
        self.surrogate  = surrogate
        self.reward_cfg = reward_cfg
        self.max_steps  = max_steps
        self.device     = device
        self.constraint_cfg = constraint_cfg or {}
        self._reset_state()

    def _reset_state(self):
        self.current_graph = None
        self.target        = None
        self.z_c           = None
        self.step_count    = 0
        self.decision_count = 0
        self.family_id = None
        self.family_index = None
        self.expected_j_steps = 1
        self.fixed_stop_by_family = True
        self.done          = False
        self.episode_reward = 0.0
        self._reward_events = []

    def reset(
        self,
        target: dict,
        base_graph: Data,
        z_c: torch.Tensor = None,
        *,
        family_id: str | None = None,
        family_index: int | None = None,
        expected_j_steps: int | None = None,
        fixed_stop_by_family: bool = True,
    ):
        self._reset_state()
        self.current_graph = copy.deepcopy(base_graph)
        self.target        = target
        self.z_c           = z_c
        self.step_count    = 0
        self.decision_count = 0
        self.family_id = family_id
        self.family_index = family_index
        self.expected_j_steps = int(expected_j_steps or max(1, self.max_steps - 1))
        self.fixed_stop_by_family = bool(fixed_stop_by_family)
        self.done          = False
        return self._get_obs()

    def _get_obs(self):
        return {
            'graph': self.current_graph,
            'z_c': self.z_c,
            'step': self.step_count,
            'semantic_state': float(self.step_count >= max(0, self.expected_j_steps - 1)),
            'family_id': self.family_id,
            'family_index': self.family_index,
            'expected_j_steps': self.expected_j_steps,
            'can_stop': bool((not self.fixed_stop_by_family) and self.step_count > 0),
        }

    def step(self, action: dict):
        """
        Execute one phase5 RL action. Reward is deferred and evaluated on the
        reached graph plus terminal stop bonus.
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")
        self.decision_count += 1
        reward_info = {'stop': bool(action.get('stop', False))}

        if reward_info['stop']:
            if self.fixed_stop_by_family:
                self.done = True
                self._reward_events.append(
                    {
                        'graph': copy.deepcopy(self.current_graph),
                        'step_index': int(self.step_count),
                        'stop': True,
                    }
                )
                return self._get_obs(), float(self.reward_cfg.get('r_invalid_penalty', -1.0)), True, reward_info
            self.done = True
            self._reward_events.append(
                {
                    'graph': copy.deepcopy(self.current_graph),
                    'step_index': int(self.step_count),
                    'stop': True,
                }
            )
            return self._get_obs(), 0.0, True, reward_info

        u, v, w = action['u'], action['v'], action['w']
        n1, n2  = action['n1'], action['n2']

        new_graph = apply_j_operator(self.current_graph, u, v, w, n1, n2)
        self.current_graph = new_graph
        self.step_count += 1
        self._reward_events.append(
            {
                'graph': copy.deepcopy(new_graph),
                'step_index': int(self.step_count),
                'stop': False,
            }
        )

        if self.step_count >= self.expected_j_steps or self.decision_count >= self.max_steps:
            self.done = True

        return self._get_obs(), 0.0, self.done, reward_info

    def compute_episode_rewards(self) -> list:
        """
        Batch-compute phase5 rewards for all executed steps.
        """
        if not self._reward_events:
            return [], []
        results, payloads = batch_compute_phase5_rewards(
            self.surrogate,
            [item['graph'] for item in self._reward_events],
            self.target,
            self.reward_cfg,
            self.device,
            step_indices=[int(item['step_index']) for item in self._reward_events],
            stop_flags=[bool(item['stop']) for item in self._reward_events],
            expected_j_steps=self.expected_j_steps,
            family_index=-1 if self.family_index is None else int(self.family_index),
            constraint_cfg=self.constraint_cfg,
        )
        return results, payloads

    @property
    def num_nodes(self):
        return self.current_graph.x.size(0) if self.current_graph else 0

    def get_valid_j_operator_actions(self):
        x = self.current_graph.x.numpy()
        is_fixed = x[:, 2]
        moving_nodes = [i for i in range(len(is_fixed)) if is_fixed[i] == 0]
        fixed_nodes  = [i for i in range(len(is_fixed)) if is_fixed[i] == 1]

        actions = []
        for i_u, u in enumerate(moving_nodes):
            for v in moving_nodes[i_u + 1:]:
                for w in fixed_nodes:
                    actions.append((u, v, w))
        return actions
