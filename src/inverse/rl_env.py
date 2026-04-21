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


# ──────────────────────────────────────────────────────────────────────────────
# 前向代理模型加载（冻结）
# ──────────────────────────────────────────────────────────────────────────────
def load_frozen_surrogate(model_path: str, config_path: str, device):
    """加载并冻结前向 BioKinematicsGNN 作为 Reward 计算器"""
    from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = BioKinematicsGNN(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
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


def _build_phase3_step_context(step_index: int, expected_j_steps: int) -> torch.Tensor:
    semantic_steps = 1 if int(step_index) >= int(expected_j_steps) else 0
    aux_steps = max(0, int(step_index) - semantic_steps)
    return torch.tensor(
        [[float(step_index), float(aux_steps), float(semantic_steps)]],
        dtype=torch.float32,
    )


def _infer_semantic_masks(graph_data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(graph_data.x.size(0))
    device = graph_data.x.device
    mask_hip = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_knee = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_ankle = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_foot = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    if graph_data.x.size(-1) >= 8:
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

    if not mask_ankle.any() and num_nodes >= 2:
        mask_ankle[num_nodes - 2] = True
    if not mask_foot.any() and num_nodes >= 1:
        mask_foot[num_nodes - 1] = True
    return mask_hip, mask_knee, mask_ankle, mask_foot


def _prepare_graph_for_surrogate(
    graph_data: Data,
    *,
    family_index: int,
    step_index: int,
    expected_j_steps: int,
) -> Data:
    prepared = copy.deepcopy(graph_data)
    x = prepared.x.float()
    if x.size(-1) < 4:
        raise ValueError(f"Expected at least 4 node features, got {x.size(-1)}")
    if x.size(-1) >= 8:
        x_aug = x
    else:
        mask_hip, mask_knee, mask_ankle, mask_foot = _infer_semantic_masks(prepared)
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

    prepared.x = x_aug
    prepared.family_id = torch.tensor([int(family_index)], dtype=torch.long)
    prepared.step_context = _build_phase3_step_context(step_index, expected_j_steps)
    prepared.mask_hip, prepared.mask_knee, prepared.mask_ankle, prepared.mask_foot = _infer_semantic_masks(prepared)
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
                   reward_cfg: dict, device, constraint_cfg: Optional[dict] = None) -> tuple:
    """
    Single-graph reward computation (fallback path).
    """
    is_valid, valid_info = validate_graph_structure(graph_data, constraint_cfg)
    if not is_valid:
        return reward_cfg.get('penalty_locking', -100.0), False, valid_info

    try:
        batch = Batch.from_data_list([graph_data]).to(device)
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
                          reward_cfg: dict, device, constraint_cfg: Optional[dict] = None) -> list:
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
        prepared_graphs = [
            _prepare_graph_for_surrogate(
                graph,
                family_index=family_index,
                step_index=int(step_indices[graph_idx]),
                expected_j_steps=int(expected_j_steps),
            )
            for graph, graph_idx in zip(valid_graphs, valid_graph_indices)
        ]
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

    try:
        batch = Batch.from_data_list(valid_graphs).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch)
        reward_t, _ = compute_reward_batch(
            pred_foot.cpu(), pred_knee.cpu(), pred_ankle.cpu(), target, reward_cfg
        )
        for batch_idx, graph_idx in enumerate(valid_indices):
            results[graph_idx] = (float(reward_t[batch_idx].item()), True)
        return results
    except Exception:
        # Fallback: per-graph evaluation
        for idx, g in zip(valid_indices, valid_graphs):
            reward, valid, _ = compute_reward(
                surrogate, g, target, reward_cfg, device, constraint_cfg=constraint_cfg
            )
            results[idx] = (reward if valid else penalty, valid)
        return results


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
    old_x   = graph_data.x.numpy()
    old_pos = graph_data.pos.numpy()
    N = old_x.shape[0]

    n1_feat = np.array([[n1_pos[0], n1_pos[1], 0.0, 0.0]], dtype=np.float32)
    n2_feat = np.array([[n2_pos[0], n2_pos[1], 0.0, 0.0]], dtype=np.float32)

    new_x   = np.vstack([old_x, n1_feat, n2_feat])
    new_pos = np.vstack([old_pos, n1_pos.reshape(1,2), n2_pos.reshape(1,2)])

    n1_idx, n2_idx = N, N + 1

    old_ei = graph_data.edge_index.numpy().T.tolist()
    new_edges = old_ei + [
        [u, n1_idx], [n1_idx, u],
        [v, n1_idx], [n1_idx, v],
        [n1_idx, n2_idx], [n2_idx, n1_idx],
        [w, n2_idx], [n2_idx, w],
    ]

    keypoints = None
    if hasattr(graph_data, 'keypoints') and graph_data.keypoints is not None:
        keypoints = graph_data.keypoints.clone().detach()
    elif hasattr(graph_data, 'knee_idx') and graph_data.knee_idx is not None:
        knee_idx = int(graph_data.knee_idx.reshape(-1)[0].item())
        if knee_idx >= 0:
            keypoints = torch.tensor([n2_idx, knee_idx, n1_idx], dtype=torch.long)

    out = Data(
        x=torch.tensor(new_x, dtype=torch.float32),
        pos=torch.tensor(new_pos, dtype=torch.float32),
        edge_index=torch.tensor(new_edges, dtype=torch.long).T,
        keypoints=keypoints,
    )
    if hasattr(graph_data, 'knee_idx') and graph_data.knee_idx is not None:
        out.knee_idx = graph_data.knee_idx.clone().detach()
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
