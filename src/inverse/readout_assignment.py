"""
Readout 关节链分配模块。
将机构图中的节点映射为腿部关节链（hip-knee-ankle-foot），支持规则、代理模型、MLP 和槽位指针四种方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from src.inverse.experiment_utils import compute_joint_metrics_batch

# 候选链特征的固定顺序
_CANDIDATE_FEATURE_ORDER = (
    "rule_score",
    "hip_anchor",
    "foot_depth_norm",
    "path_length_norm",
    "branch_penalty",
    "foot_ground_penalty",
    "foot_anchor_bonus",
    "distal_margin_norm",
    "foot_rom_x",
    "foot_rom_y",
    "foot_curvature",
    "knee_amp",
    "ankle_amp",
    "circle_penalty",
    "has_motion",
    "has_target",
    "target_error_available",
    "foot_target_error",
    "knee_target_error",
    "ankle_target_error",
)

# 节点级特征的固定顺序
_NODE_FEATURE_ORDER = (
    "x",
    "y",
    "is_fixed",
    "is_ground",
    "degree_norm",
    "depth_norm",
    "ground_adjacent",
    "motion_rom_x",
    "motion_rom_y",
    "motion_curvature",
    "motion_loop_closure",
    "motion_isotropy",
    "has_motion",
    "has_target",
    "knee_target_error",
    "ankle_target_error",
    "foot_target_error",
)


@dataclass
class AssignmentTarget:
    """关节链分配目标：足部轨迹、膝关节角度、踝关节角度。"""
    y_foot: np.ndarray | None = None
    y_knee: np.ndarray | None = None
    y_ankle: np.ndarray | None = None

    @classmethod
    def from_mapping(cls, target: Mapping[str, object] | None) -> "AssignmentTarget | None":
        """从字典创建实例，target 为 None 时返回 None。"""
        if target is None:
            return None
        return cls(
            y_foot=_maybe_array(target.get("y_foot"), min_ndim=2),
            y_knee=_maybe_array(target.get("y_knee"), min_ndim=1),
            y_ankle=_maybe_array(target.get("y_ankle"), min_ndim=1),
        )

    @classmethod
    def from_motion(
        cls,
        motion: np.ndarray,
        *,
        hip: int,
        knee: int,
        ankle: int,
        foot: int,
    ) -> "AssignmentTarget":
        """从运动数据和关节索引创建目标。"""
        curves = _candidate_curves(motion, hip=hip, knee=knee, ankle=ankle, foot=foot)
        return cls(
            y_foot=curves["foot"],
            y_knee=curves["knee"],
            y_ankle=curves["ankle"],
        )


@dataclass
class CandidateLegChain:
    """候选腿部关节链：hip-knee-ankle-foot 四个节点的索引、路径、特征和评分。"""
    hip: int
    knee: int
    ankle: int
    foot: int
    path: tuple[int, ...]
    features: dict[str, float]
    score_breakdown: dict[str, float] = field(default_factory=dict)
    score: float = float("-inf")

    def keypoints(self) -> dict[str, int]:
        """返回关键节点字典 {'hip', 'knee', 'ankle', 'foot'}。"""
        return {
            "hip": int(self.hip),
            "knee": int(self.knee),
            "ankle": int(self.ankle),
            "foot": int(self.foot),
        }


@dataclass
class AssignmentResult:
    """关节链分配结果：方法名、关键节点、路径、评分、候选列表。"""
    method: str
    keypoints: dict[str, int]
    path: tuple[int, ...]
    score: float
    score_breakdown: dict[str, float]
    candidate_count: int
    top_candidates: list[CandidateLegChain]


@dataclass
class ReadoutAssignmentRecord:
    """训练/评估用的单条记录：图结构、运动数据、分配目标、真实标签。"""
    graph: object
    motion: np.ndarray | None = None
    target: AssignmentTarget | Mapping[str, object] | None = None
    truth: dict[str, int] | None = None

    def resolved_target(self) -> AssignmentTarget | None:
        """解析并返回 AssignmentTarget 实例。"""
        if isinstance(self.target, AssignmentTarget):
            return self.target
        return AssignmentTarget.from_mapping(self.target)


# ==============================================================================
# 工具函数
# ==============================================================================

def _as_numpy(value, *, dtype=np.float32) -> np.ndarray:
    """将输入转换为 NumPy 数组（支持 ndarray/Tensor/其他）。"""
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _maybe_array(value, *, min_ndim: int) -> np.ndarray | None:
    """尝试转换为至少指定维度的数组，维度不足返回 None。"""
    if value is None:
        return None
    array = _as_numpy(value)
    if array.ndim < min_ndim:
        return None
    return array


def _target_has_any_curve(target: AssignmentTarget | None) -> bool:
    """目标是否包含至少一条有效曲线。"""
    return bool(
        target is not None
        and (target.y_foot is not None or target.y_knee is not None or target.y_ankle is not None)
    )


def _target_has_all_curves(target: AssignmentTarget | None) -> bool:
    """目标是否包含所有三条有效曲线。"""
    return bool(
        target is not None
        and target.y_foot is not None
        and target.y_knee is not None
        and target.y_ankle is not None
    )


def _graph_value(graph: object, name: str, default=None):
    """从图对象（字典或对象）中获取属性值。"""
    if isinstance(graph, Mapping):
        return graph.get(name, default)
    return getattr(graph, name, default)


def _graph_arrays(graph: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从图对象提取 (x, pos, edge_index) 数组。"""
    if hasattr(graph, "x") and hasattr(graph, "edge_index"):
        x = _as_numpy(graph.x)
        pos_value = graph.pos if hasattr(graph, "pos") and graph.pos is not None else graph.x[:, :2]
        pos = _as_numpy(pos_value)
        edge_index = _as_numpy(graph.edge_index, dtype=np.int64)
        return x, pos, edge_index
    if isinstance(graph, Mapping):
        x = _as_numpy(graph["x"])
        pos = _as_numpy(graph.get("pos", x[:, :2]))
        edge_index = _as_numpy(graph["edge_index"], dtype=np.int64)
        return x, pos, edge_index
    raise TypeError(f"Unsupported graph container: {type(graph)!r}")


def _resolve_motion(graph: object, motion: np.ndarray | Mapping[str, object] | None) -> np.ndarray | None:
    """按优先级解析运动数据数组。"""
    if motion is not None:
        if isinstance(motion, Mapping):
            if "x_sol" in motion:
                return _as_numpy(motion["x_sol"])
            analysis = motion.get("analysis")
            if isinstance(analysis, Mapping) and "x_sol" in analysis:
                return _as_numpy(analysis["x_sol"])
        return _as_numpy(motion)
    if isinstance(graph, Mapping):
        if "x_sol" in graph:
            return _as_numpy(graph["x_sol"])
        analysis = graph.get("analysis")
        if isinstance(analysis, Mapping) and "x_sol" in analysis:
            return _as_numpy(analysis["x_sol"])
    if hasattr(graph, "x_sol") and getattr(graph, "x_sol") is not None:
        return _as_numpy(graph.x_sol)
    if hasattr(graph, "analysis"):
        analysis = getattr(graph, "analysis")
        if isinstance(analysis, Mapping) and "x_sol" in analysis:
            return _as_numpy(analysis["x_sol"])
    return None


def _normalize_trajectory(traj: np.ndarray) -> np.ndarray:
    """将轨迹 min-max 归一化到 [0, 1]。"""
    traj = _as_numpy(traj)
    lo = traj.min(axis=0)
    hi = traj.max(axis=0)
    scale = hi - lo
    scale[scale < 1e-6] = 1.0
    return (traj - lo) / scale


def _unsigned_angle_series(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """计算三个点序列形成的无符号角度序列，归一化到 [0, 1]。"""
    v1 = a - b
    v2 = c - b
    dot = np.sum(v1 * v2, axis=-1)
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    cosine = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(cosine) / np.pi


def _trajectory_curvature(traj: np.ndarray) -> float:
    """轨迹平均曲率（二阶差分范数均值）。"""
    if traj.shape[0] < 3:
        return 0.0
    second = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
    return float(np.linalg.norm(second, axis=-1).mean())


def _trajectory_span(traj: np.ndarray) -> tuple[float, float]:
    """轨迹 X/Y 方向跨度。"""
    span = traj.max(axis=0) - traj.min(axis=0)
    return float(span[0]), float(span[1])


def _loop_closure_ratio(traj: np.ndarray) -> float:
    """轨迹闭环比：首尾距离 / 跨度。"""
    span = max(1e-6, float(np.linalg.norm(traj.max(axis=0) - traj.min(axis=0))))
    return float(np.linalg.norm(traj[0] - traj[-1]) / span)


def _trajectory_isotropy(traj: np.ndarray) -> float:
    """轨迹各向同性指标（协方差矩阵最小/最大特征值比），值 1=圆形, 0=线性。"""
    centered = traj - traj.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0], 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    return float(eigvals.min() / eigvals.max())


def _normalized_curve_error(pred: np.ndarray | None, target: np.ndarray | None) -> float:
    """计算预测与目标曲线的归一化 RMSE。"""
    if pred is None or target is None:
        return 0.0
    pred = _as_numpy(pred)
    target = _as_numpy(target)
    if pred.shape != target.shape:
        steps = min(pred.shape[0], target.shape[0])
        pred = pred[:steps]
        target = target[:steps]
    scale = np.max(target) - np.min(target) if target.ndim == 1 else float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    scale = max(scale, 1e-6)
    return float(np.sqrt(np.mean((pred - target) ** 2)) / scale)


def _candidate_curves(
    motion: np.ndarray,
    *,
    hip: int,
    knee: int,
    ankle: int,
    foot: int,
) -> dict[str, np.ndarray]:
    """从运动数据提取候选关节链的三条曲线（足部归一化轨迹、膝/踝角度序列）。"""
    hip_traj = motion[hip]
    knee_traj = motion[knee]
    ankle_traj = motion[ankle]
    foot_traj = motion[foot]
    return {
        "foot": _normalize_trajectory(foot_traj),
        "knee": _unsigned_angle_series(hip_traj, knee_traj, ankle_traj),
        "ankle": _unsigned_angle_series(knee_traj, ankle_traj, foot_traj),
    }


def _build_adjacency(num_nodes: int, edge_index: np.ndarray) -> list[set[int]]:
    """从边索引构建邻接表。"""
    adjacency = [set() for _ in range(num_nodes)]
    if edge_index.size == 0:
        return adjacency
    for u, v in edge_index.T.tolist():
        u = int(u)
        v = int(v)
        if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _ground_and_fixed_masks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从节点特征提取地面/固定节点掩码。默认第一个节点为地面。"""
    is_fixed = x[:, 2] > 0.5 if x.shape[1] >= 3 else np.zeros(x.shape[0], dtype=bool)
    is_ground = x[:, 3] > 0.5 if x.shape[1] >= 4 else np.zeros(x.shape[0], dtype=bool)
    if not is_ground.any() and x.shape[0] > 0:
        is_ground[0] = True
    return is_ground, is_fixed


def _depth_from_anchors(adjacency: Sequence[set[int]], anchors: np.ndarray) -> np.ndarray:
    """BFS 计算每个节点到最近锚点的深度。"""
    depth = np.full(len(adjacency), fill_value=max(1, len(adjacency)), dtype=np.int64)
    frontier = [int(idx) for idx in np.where(anchors)[0]]
    for idx in frontier:
        depth[idx] = 0
    queue = list(frontier)
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for nxt in adjacency[node]:
            if depth[nxt] > depth[node] + 1:
                depth[nxt] = depth[node] + 1
                queue.append(int(nxt))
    return depth


def _candidate_feature_vector(candidate: CandidateLegChain) -> np.ndarray:
    """将候选链特征字典转为固定顺序的向量。"""
    return np.array([float(candidate.features.get(name, 0.0)) for name in _CANDIDATE_FEATURE_ORDER], dtype=np.float32)


def _clip01(value: float) -> float:
    """裁剪到 [0, 1]。"""
    return float(max(0.0, min(1.0, value)))


# ==============================================================================
# 候选枚举
# ==============================================================================

def enumerate_leg_candidates(
    graph: object,
    *,
    motion: np.ndarray | Mapping[str, object] | None = None,
    target: AssignmentTarget | Mapping[str, object] | None = None,
    require_consecutive_semantic_chain: bool = False,
    max_path_nodes: int = 10,
    max_candidates: int = 256,
) -> list[CandidateLegChain]:
    """
    枚举图中所有可能的腿部关节链候选。
    DFS 从髋关节出发沿边遍历，生成路径后选取 knee/ankle 组合。
    """
    x, pos, edge_index = _graph_arrays(graph)
    num_nodes = int(x.shape[0])
    if num_nodes < 4:
        return []
    adjacency = _build_adjacency(num_nodes, edge_index)
    is_ground, is_fixed = _ground_and_fixed_masks(x)
    anchors = np.logical_or(is_ground, is_fixed)
    moving = ~anchors
    if not moving.any() and not anchors.any():
        return []

    depth = _depth_from_anchors(adjacency, anchors)
    adjacent_to_anchor = np.array([any(anchors[nbr] for nbr in adjacency[idx]) for idx in range(num_nodes)], dtype=bool)
    hip_anchor_candidates = np.where(anchors)[0].tolist()
    hip_moving_candidates = np.where(np.logical_and(moving, adjacent_to_anchor))[0].tolist()
    hip_candidates = hip_anchor_candidates + [idx for idx in hip_moving_candidates if idx not in hip_anchor_candidates]
    if not hip_candidates:
        hip_candidates = np.where(np.logical_or(anchors, moving))[0].tolist()

    target_obj = target if isinstance(target, AssignmentTarget) else AssignmentTarget.from_mapping(target)
    motion_array = _resolve_motion(graph, motion)
    candidates: list[CandidateLegChain] = []
    seen: set[tuple[int, int, int, int, tuple[int, ...]]] = set()

    def add_candidate(path: tuple[int, ...]) -> None:
        """为给定路径添加候选链。"""
        if len(path) < 4:
            return
        foot = int(path[-1])
        if anchors[foot]:
            return
        interior = list(path[1:])
        if require_consecutive_semantic_chain:
            semantic_pairs = [(int(path[-3]), int(path[-2]))]
        else:
            semantic_pairs = [(int(knee), int(ankle)) for knee, ankle in combinations(interior[:-1], 2)]
        for knee, ankle in semantic_pairs:
            key = (int(path[0]), knee, ankle, foot, tuple(int(node) for node in path))
            if key in seen:
                continue
            seen.add(key)
            candidate = _build_candidate(
                path=tuple(int(node) for node in path),
                hip=int(path[0]), knee=knee, ankle=ankle, foot=foot,
                adjacency=adjacency, anchors=anchors, depth=depth,
                pos=pos, motion=motion_array, target=target_obj,
            )
            candidates.append(candidate)

    def dfs(path: list[int]) -> None:
        """深度优先搜索扩展候选。"""
        if len(candidates) >= max_candidates:
            return
        add_candidate(tuple(path))
        if len(path) >= max_path_nodes:
            return
        current = path[-1]
        for nxt in sorted(adjacency[current]):
            if nxt in path:
                continue
            if anchors[nxt] and nxt != path[0]:
                continue
            if depth[nxt] + 1 < depth[current]:
                continue
            dfs(path + [int(nxt)])

    for hip in sorted(hip_candidates, key=lambda idx: (depth[idx], idx)):
        dfs([int(hip)])
        if len(candidates) >= max_candidates:
            break

    return candidates


def _build_candidate(
    *,
    path: tuple[int, ...],
    hip: int,
    knee: int,
    ankle: int,
    foot: int,
    adjacency: Sequence[set[int]],
    anchors: np.ndarray,
    depth: np.ndarray,
    pos: np.ndarray,
    motion: np.ndarray | None,
    target: AssignmentTarget | None,
) -> CandidateLegChain:
    """构建单个候选链，计算结构/运动/目标特征并加权评分。"""
    max_depth = max(1, int(depth.max()))
    has_motion = bool(motion is not None and motion.shape[0] > foot)
    has_target = _target_has_any_curve(target)
    foot_anchor_bonus = 1.0 if any(anchors[nbr] for nbr in adjacency[foot]) else 0.0
    branch_penalty = _clip01((len(adjacency[knee]) + len(adjacency[ankle]) - 4) / 4.0)
    knee_pos = int(path.index(knee))
    ankle_pos = int(path.index(ankle))
    foot_pos = int(path.index(foot))
    distal_margin = max(0.0, float(foot_pos - ankle_pos))

    features = {
        "hip_anchor": 1.0 if anchors[hip] or any(anchors[nbr] for nbr in adjacency[hip]) else 0.0,
        "foot_depth_norm": float(depth[foot] / max_depth),
        "path_length_norm": float((len(path) - 1) / max(1, max_depth)),
        "branch_penalty": branch_penalty,
        "foot_ground_penalty": 0.0,
        "foot_anchor_bonus": foot_anchor_bonus,
        "distal_margin_norm": float(distal_margin / max(1, len(path) - 1)),
        "foot_rom_x": 0.0, "foot_rom_y": 0.0, "foot_curvature": 0.0,
        "knee_amp": 0.0, "ankle_amp": 0.0, "circle_penalty": 0.0,
        "has_motion": 1.0 if has_motion else 0.0,
        "has_target": 1.0 if has_target else 0.0,
        "target_error_available": 0.0,
        "foot_target_error": 1.0 if target is not None and target.y_foot is not None else 0.0,
        "knee_target_error": 1.0 if target is not None and target.y_knee is not None else 0.0,
        "ankle_target_error": 1.0 if target is not None and target.y_ankle is not None else 0.0,
    }

    if has_motion:
        curves = _candidate_curves(motion, hip=hip, knee=knee, ankle=ankle, foot=foot)
        foot_traj = curves["foot"]
        rom_x, rom_y = _trajectory_span(foot_traj)
        loop_closure = _loop_closure_ratio(foot_traj)
        isotropy = _trajectory_isotropy(foot_traj)
        features["foot_rom_x"] = _clip01(rom_x)
        features["foot_rom_y"] = _clip01(rom_y)
        features["foot_curvature"] = _clip01(_trajectory_curvature(foot_traj) * 3.0)
        features["knee_amp"] = _clip01(float(curves["knee"].max() - curves["knee"].min()))
        features["ankle_amp"] = _clip01(float(curves["ankle"].max() - curves["ankle"].min()))
        features["circle_penalty"] = _clip01(isotropy * np.exp(-2.0 * loop_closure))
        if target is not None:
            features["foot_target_error"] = _normalized_curve_error(curves["foot"], target.y_foot)
            features["knee_target_error"] = _normalized_curve_error(curves["knee"], target.y_knee)
            features["ankle_target_error"] = _normalized_curve_error(curves["ankle"], target.y_ankle)
            features["target_error_available"] = 1.0 if has_target else 0.0

    motion_score = (
        0.55 * features["foot_rom_x"]
        + 0.30 * features["foot_rom_y"]
        + 0.25 * features["foot_curvature"]
        + 0.35 * features["knee_amp"]
        + 0.35 * features["ankle_amp"]
        - 0.40 * features["circle_penalty"]
    )
    structural_score = (
        1.45 * features["hip_anchor"]
        + 0.60 * features["path_length_norm"]
        + 0.35 * features["distal_margin_norm"]
        + 0.90 * features["foot_anchor_bonus"]
        - 0.65 * features["branch_penalty"]
    )
    target_score = -(
        1.40 * features["foot_target_error"]
        + 0.90 * features["knee_target_error"]
        + 0.90 * features["ankle_target_error"]
    )
    rule_score = structural_score + motion_score + target_score
    features["rule_score"] = float(rule_score)

    breakdown = {
        "structural": float(structural_score),
        "motion": float(motion_score),
        "target": float(target_score),
        "total": float(rule_score),
    }
    return CandidateLegChain(
        hip=hip, knee=knee, ankle=ankle, foot=foot,
        path=path, features=features,
        score_breakdown=breakdown, score=float(rule_score),
    )


def assignment_to_masks(
    assignment: AssignmentResult | CandidateLegChain | None,
    num_nodes: int,
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """将分配结果转为节点级布尔掩码张量。"""
    resolved_device = device or "cpu"
    masks = {
        "hip": torch.zeros(num_nodes, dtype=torch.bool, device=resolved_device),
        "knee": torch.zeros(num_nodes, dtype=torch.bool, device=resolved_device),
        "ankle": torch.zeros(num_nodes, dtype=torch.bool, device=resolved_device),
        "foot": torch.zeros(num_nodes, dtype=torch.bool, device=resolved_device),
    }
    if assignment is None:
        return masks
    keypoints = assignment.keypoints if isinstance(assignment, AssignmentResult) else assignment.keypoints()
    for name in ("hip", "knee", "ankle", "foot"):
        idx = int(keypoints[name])
        if 0 <= idx < num_nodes:
            masks[name][idx] = True
    return masks


# ==============================================================================
# 分配方法实现
# ==============================================================================

class RuleBasedReadoutAssignment:
    """基于规则的关节链分配：手工特征 + 加权评分，最快但不依赖学习模型。"""

    def __init__(self, *, top_k: int = 5, require_consecutive_semantic_chain: bool = False):
        self.top_k = int(top_k)
        self.require_consecutive_semantic_chain = bool(require_consecutive_semantic_chain)

    def assign(
        self,
        graph: object,
        *,
        target: AssignmentTarget | Mapping[str, object] | None = None,
        motion: np.ndarray | Mapping[str, object] | None = None,
    ) -> AssignmentResult | None:
        """执行规则分配，返回最优候选。无有效候选时返回 None。"""
        candidates = enumerate_leg_candidates(
            graph, motion=motion, target=target,
            require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
        )
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        return AssignmentResult(
            method="A_rule_based_chain_assignment",
            keypoints=best.keypoints(),
            path=best.path,
            score=float(best.score),
            score_breakdown=dict(best.score_breakdown),
            candidate_count=len(ranked),
            top_candidates=ranked[: self.top_k],
        )


def _step_context_from_indices(step_index: int | None, expected_j_steps: int | None) -> torch.Tensor | None:
    """从步数索引构建 (step_index, aux_steps, semantic_steps) 上下文张量。"""
    if step_index is None or expected_j_steps is None:
        return None
    semantic_steps = 1 if int(step_index) >= int(expected_j_steps) else 0
    aux_steps = max(0, int(step_index) - semantic_steps)
    return torch.tensor([[float(step_index), float(aux_steps), float(semantic_steps)]], dtype=torch.float32)


def _candidate_surrogate_data(
    graph: object,
    candidate: CandidateLegChain,
    *,
    family_index: int | None = None,
    step_index: int | None = None,
    expected_j_steps: int | None = None,
) -> Data:
    """将候选链转为代理模型可接受的 PyG Data（拼接语义 one-hot 到节点特征）。"""
    x, pos, edge_index = _graph_arrays(graph)
    num_nodes = int(x.shape[0])
    base = np.zeros((num_nodes, 4), dtype=np.float32)
    base[:, : min(4, x.shape[1])] = x[:, : min(4, x.shape[1])]
    semantic = np.zeros((num_nodes, 4), dtype=np.float32)
    for col, idx in enumerate((candidate.hip, candidate.knee, candidate.ankle, candidate.foot)):
        if 0 <= int(idx) < num_nodes:
            semantic[int(idx), col] = 1.0
    data = Data(
        x=torch.tensor(np.concatenate([base, semantic], axis=1), dtype=torch.float32),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        mask_hip=torch.tensor(semantic[:, 0] > 0.5, dtype=torch.bool),
        mask_knee=torch.tensor(semantic[:, 1] > 0.5, dtype=torch.bool),
        mask_ankle=torch.tensor(semantic[:, 2] > 0.5, dtype=torch.bool),
        mask_foot=torch.tensor(semantic[:, 3] > 0.5, dtype=torch.bool),
        semantic_feature_layout=torch.tensor([1], dtype=torch.long),
        semantic_dirty=torch.tensor([False], dtype=torch.bool),
    )

    resolved_family = family_index
    if resolved_family is None:
        raw_family = _graph_value(graph, "family_id")
        if raw_family is not None:
            if isinstance(raw_family, str):
                resolved_family = {"6bar": 0, "7bar": 1, "8bar": 2, "9bar": 3}.get(raw_family)
            else:
                resolved_family = int(torch.as_tensor(raw_family).view(-1)[0].item())
    if resolved_family is not None:
        data.family_id = torch.tensor([int(resolved_family)], dtype=torch.long)

    resolved_step_context = _step_context_from_indices(step_index, expected_j_steps)
    if resolved_step_context is None:
        raw_step_context = _graph_value(graph, "step_context")
        if raw_step_context is not None:
            resolved_step_context = torch.as_tensor(raw_step_context, dtype=torch.float32).view(1, -1)
    if resolved_step_context is not None:
        data.step_context = resolved_step_context.float()
    return data


def _assignment_target_to_tensor_dict(target: AssignmentTarget) -> dict[str, torch.Tensor]:
    """将 AssignmentTarget 转为张量字典。"""
    if not _target_has_all_curves(target):
        raise ValueError("Surrogate target readout requires y_foot, y_knee, and y_ankle.")
    return {
        "y_foot": torch.tensor(target.y_foot, dtype=torch.float32),
        "y_knee": torch.tensor(target.y_knee, dtype=torch.float32),
        "y_ankle": torch.tensor(target.y_ankle, dtype=torch.float32),
    }


class SurrogateTargetReadoutAssignment:
    """
    基于代理模型的关节链分配。
    规则方法枚举候选后用代理模型评分，按 -joint_score + 结构先验排序。
    """

    def __init__(
        self,
        surrogate_model,
        *,
        top_k: int = 5,
        batch_size: int = 64,
        metric_cfg: Mapping[str, float] | None = None,
        device: torch.device | str | None = None,
        structural_prior_weight: float = 0.05,
        require_consecutive_semantic_chain: bool = False,
        family_index: int | None = None,
        step_index: int | None = None,
        expected_j_steps: int | None = None,
        max_surrogate_candidates: int | None = None,
    ):
        self.surrogate_model = surrogate_model
        self.top_k = int(top_k)
        self.batch_size = int(batch_size)
        # 仅对 rule-prior 排名前 N 的候选跑 surrogate（None=全跑，保持原行为）。
        # rule_score 已含 foot/knee/ankle 目标误差先验，是有效筛子；剪枝只会让最终
        # 选中的 readout 略保守（joint_score 不会更优），不会高估命中。
        self.max_surrogate_candidates = (
            int(max_surrogate_candidates) if max_surrogate_candidates else None
        )
        self.metric_cfg = dict(metric_cfg or {})
        self.structural_prior_weight = float(structural_prior_weight)
        self.require_consecutive_semantic_chain = bool(require_consecutive_semantic_chain)
        self.family_index = family_index
        self.step_index = step_index
        self.expected_j_steps = expected_j_steps
        self.teacher = RuleBasedReadoutAssignment(
            top_k=top_k,
            require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
        )
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(surrogate_model.parameters()).device
            except Exception:
                self.device = torch.device("cpu")

    def assign(
        self,
        graph: object,
        *,
        target: AssignmentTarget | Mapping[str, object] | None = None,
        motion: np.ndarray | Mapping[str, object] | None = None,
        family_index: int | None = None,
        step_index: int | None = None,
        expected_j_steps: int | None = None,
    ) -> AssignmentResult | None:
        """执行代理模型分配。代理模型不可用时回退到规则方法。"""
        target_obj = target if isinstance(target, AssignmentTarget) else AssignmentTarget.from_mapping(target)
        if self.surrogate_model is None or not _target_has_all_curves(target_obj):
            return self.teacher.assign(graph, target=target_obj, motion=motion)

        candidates = enumerate_leg_candidates(
            graph, motion=None, target=target_obj,
            require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
        )
        if not candidates:
            return None

        # 用无 surrogate 的 rule_score 先粗排，只对 top-N 跑 surrogate（GPU 大头在此）。
        if self.max_surrogate_candidates is not None and len(candidates) > self.max_surrogate_candidates:
            candidates = sorted(candidates, key=lambda c: c.score, reverse=True)[: self.max_surrogate_candidates]

        target_dict = _assignment_target_to_tensor_dict(target_obj)
        resolved_family = self.family_index if family_index is None else family_index
        resolved_step = self.step_index if step_index is None else step_index
        resolved_expected = self.expected_j_steps if expected_j_steps is None else expected_j_steps
        ranked_candidates: list[CandidateLegChain] = []
        was_training = bool(getattr(self.surrogate_model, "training", False))
        self.surrogate_model.eval()

        with torch.no_grad():
            for start in range(0, len(candidates), self.batch_size):
                chunk = candidates[start : start + self.batch_size]
                data_list = [
                    _candidate_surrogate_data(
                        graph, candidate,
                        family_index=resolved_family,
                        step_index=resolved_step,
                        expected_j_steps=resolved_expected,
                    )
                    for candidate in chunk
                ]
                batch = Batch.from_data_list(data_list).to(self.device)
                pred_foot, pred_knee, pred_ankle = self.surrogate_model(batch)
                metrics = compute_joint_metrics_batch(
                    pred_foot, pred_knee, pred_ankle, target_dict, self.metric_cfg,
                )
                joint_scores = metrics["joint_score"].detach().cpu().numpy()
                foot_scores = metrics["foot_score"].detach().cpu().numpy()
                knee_scores = metrics["knee_nrmse"].detach().cpu().numpy()
                ankle_scores = metrics["ankle_nrmse"].detach().cpu().numpy()
                smoothness = metrics["smoothness"].detach().cpu().numpy()

                for local_idx, candidate in enumerate(chunk):
                    structural_prior = float(candidate.score_breakdown.get("structural", 0.0))
                    prior_score = self.structural_prior_weight * structural_prior
                    joint_score = float(joint_scores[local_idx])
                    total = -joint_score + prior_score
                    features = dict(candidate.features)
                    features.update(
                        {
                            "has_target": 1.0,
                            "target_error_available": 1.0,
                            "foot_target_error": float(foot_scores[local_idx]),
                            "knee_target_error": float(knee_scores[local_idx]),
                            "ankle_target_error": float(ankle_scores[local_idx]),
                            "surrogate_joint_score": joint_score,
                            "surrogate_smoothness": float(smoothness[local_idx]),
                        }
                    )
                    ranked_candidates.append(
                        CandidateLegChain(
                            hip=candidate.hip, knee=candidate.knee,
                            ankle=candidate.ankle, foot=candidate.foot,
                            path=candidate.path, features=features,
                            score_breakdown={
                                "surrogate_target": -joint_score,
                                "structural_prior": prior_score,
                                "joint_score": joint_score,
                                "foot_score": float(foot_scores[local_idx]),
                                "knee_nrmse": float(knee_scores[local_idx]),
                                "ankle_nrmse": float(ankle_scores[local_idx]),
                                "smoothness": float(smoothness[local_idx]),
                                "total": total,
                            },
                            score=float(total),
                        )
                    )

        if was_training:
            self.surrogate_model.train()
        ranked = sorted(ranked_candidates, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        return AssignmentResult(
            method="D_surrogate_target_chain_assignment",
            keypoints=best.keypoints(),
            path=best.path,
            score=float(best.score),
            score_breakdown=dict(best.score_breakdown),
            candidate_count=len(ranked),
            top_candidates=ranked[: self.top_k],
        )


class _ChainScorerMLP(nn.Module):
    """3 层链评分 MLP：特征向量 -> 标量评分。"""

    def __init__(self, input_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LearnedChainScorerReadoutAssignment:
    """
    基于学习的链评分器分配。
    训练时以规则方法输出为伪标签，推理时结合 MLP 评分和规则先验排序。
    """

    def __init__(
        self,
        *,
        top_k: int = 5,
        hidden_dim: int = 48,
        require_consecutive_semantic_chain: bool = False,
    ):
        self.top_k = int(top_k)
        self.hidden_dim = int(hidden_dim)
        self.require_consecutive_semantic_chain = bool(require_consecutive_semantic_chain)
        self.teacher = RuleBasedReadoutAssignment(
            top_k=top_k,
            require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
        )
        self.model: _ChainScorerMLP | None = None
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def fit(
        self,
        records: Sequence[ReadoutAssignmentRecord],
        *,
        teacher: RuleBasedReadoutAssignment | None = None,
        epochs: int = 80,
        lr: float = 1e-2,
        seed: int = 0,
    ) -> dict[str, float]:
        """训练链评分 MLP。用规则方法伪标签 + 交叉熵损失。返回 {'loss', 'num_records'}。"""
        if not records:
            return {"loss": 0.0, "num_records": 0.0}
        rng = torch.Generator().manual_seed(int(seed))
        torch.manual_seed(int(seed))
        teacher = teacher or self.teacher
        dataset: list[tuple[torch.Tensor, int]] = []
        all_features: list[torch.Tensor] = []

        for record in records:
            target = record.resolved_target()
            candidates = enumerate_leg_candidates(
                record.graph, motion=record.motion, target=target,
                require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
            )
            if len(candidates) < 2:
                continue
            teacher_result = teacher.assign(record.graph, motion=record.motion, target=target)
            if teacher_result is None:
                continue
            label_idx = next(
                (
                    idx for idx, candidate in enumerate(candidates)
                    if candidate.keypoints() == teacher_result.keypoints
                ),
                None,
            )
            if label_idx is None:
                continue
            feature_tensor = torch.tensor(
                np.stack([_candidate_feature_vector(candidate) for candidate in candidates], axis=0),
                dtype=torch.float32,
            )
            dataset.append((feature_tensor, int(label_idx)))
            all_features.append(feature_tensor)

        if not dataset:
            return {"loss": 0.0, "num_records": 0.0}

        stacked = torch.cat(all_features, dim=0)
        self.feature_mean = stacked.mean(dim=0)
        self.feature_std = stacked.std(dim=0).clamp_min(1e-6)
        self.model = _ChainScorerMLP(stacked.size(1), hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        last_loss = 0.0

        for _ in range(int(epochs)):
            total_loss = 0.0
            order = torch.randperm(len(dataset), generator=rng).tolist()
            for dataset_idx in order:
                features, label_idx = dataset[dataset_idx]
                normalized = (features - self.feature_mean) / self.feature_std
                logits = self.model(normalized)
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label_idx], dtype=torch.long))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            last_loss = total_loss / max(1, len(dataset))

        return {"loss": float(last_loss), "num_records": float(len(dataset))}

    def assign(
        self,
        graph: object,
        *,
        target: AssignmentTarget | Mapping[str, object] | None = None,
        motion: np.ndarray | Mapping[str, object] | None = None,
    ) -> AssignmentResult | None:
        """使用训练好的 MLP 执行分配。模型未训练时回退到规则方法。"""
        target_obj = target if isinstance(target, AssignmentTarget) else AssignmentTarget.from_mapping(target)
        candidates = enumerate_leg_candidates(
            graph, motion=motion, target=target_obj,
            require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
        )
        if not candidates:
            return None
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            return self.teacher.assign(graph, target=target_obj, motion=motion)

        features = torch.tensor(
            np.stack([_candidate_feature_vector(candidate) for candidate in candidates], axis=0),
            dtype=torch.float32,
        )
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(normalized).cpu().numpy()
        ranked_candidates: list[CandidateLegChain] = []
        for candidate, model_score in zip(candidates, logits.tolist()):
            combined = float(model_score + 0.20 * candidate.features.get("rule_score", 0.0))
            ranked_candidates.append(
                CandidateLegChain(
                    hip=candidate.hip, knee=candidate.knee,
                    ankle=candidate.ankle, foot=candidate.foot,
                    path=candidate.path, features=dict(candidate.features),
                    score_breakdown={
                        "learned": float(model_score),
                        "rule_prior": float(0.20 * candidate.features.get("rule_score", 0.0)),
                        "total": combined,
                    },
                    score=combined,
                )
            )

        ranked = sorted(ranked_candidates, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        return AssignmentResult(
            method="B_pseudo_label_chain_scorer",
            keypoints=best.keypoints(),
            path=best.path,
            score=float(best.score),
            score_breakdown=dict(best.score_breakdown),
            candidate_count=len(ranked),
            top_candidates=ranked[: self.top_k],
        )


class _SlotPointerMLP(nn.Module):
    """槽位指针 MLP：为每个节点输出 3 个槽位分数（knee/ankle/foot）。"""

    def __init__(self, input_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.slot_head = nn.Linear(hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        return self.slot_head(hidden)


def _node_feature_matrix(
    graph: object,
    *,
    motion: np.ndarray | None,
    target: AssignmentTarget | None,
    candidates: Sequence[CandidateLegChain] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """构建节点级特征矩阵和有效掩码，用于槽位指针方法。"""
    x, pos, edge_index = _graph_arrays(graph)
    num_nodes = int(x.shape[0])
    has_motion = bool(motion is not None)
    has_target = _target_has_any_curve(target)
    adjacency = _build_adjacency(num_nodes, edge_index)
    is_ground, is_fixed = _ground_and_fixed_masks(x)
    anchors = np.logical_or(is_ground, is_fixed)
    depth = _depth_from_anchors(adjacency, anchors)
    max_depth = max(1, int(depth.max()))
    pos_center = pos.mean(axis=0, keepdims=True)
    pos_scale = np.abs(pos - pos_center).max(axis=0, keepdims=True)
    pos_scale[pos_scale < 1e-6] = 1.0
    pos_norm = (pos - pos_center) / pos_scale

    slot_error_defaults = {
        "knee": np.ones(num_nodes, dtype=np.float32),
        "ankle": np.ones(num_nodes, dtype=np.float32),
        "foot": np.ones(num_nodes, dtype=np.float32),
    }
    for candidate in candidates or []:
        slot_error_defaults["knee"][candidate.knee] = min(
            float(slot_error_defaults["knee"][candidate.knee]),
            float(candidate.features.get("knee_target_error", 1.0)),
        )
        slot_error_defaults["ankle"][candidate.ankle] = min(
            float(slot_error_defaults["ankle"][candidate.ankle]),
            float(candidate.features.get("ankle_target_error", 1.0)),
        )
        slot_error_defaults["foot"][candidate.foot] = min(
            float(slot_error_defaults["foot"][candidate.foot]),
            float(candidate.features.get("foot_target_error", 1.0)),
        )

    features = np.zeros((num_nodes, len(_NODE_FEATURE_ORDER)), dtype=np.float32)
    for node_idx in range(num_nodes):
        values = {
            "x": float(pos_norm[node_idx, 0]),
            "y": float(pos_norm[node_idx, 1]),
            "is_fixed": 1.0 if is_fixed[node_idx] else 0.0,
            "is_ground": 1.0 if is_ground[node_idx] else 0.0,
            "degree_norm": float(len(adjacency[node_idx]) / max(1, num_nodes - 1)),
            "depth_norm": float(depth[node_idx] / max_depth),
            "ground_adjacent": 1.0 if any(anchors[nbr] for nbr in adjacency[node_idx]) else 0.0,
            "motion_rom_x": 0.0, "motion_rom_y": 0.0,
            "motion_curvature": 0.0, "motion_loop_closure": 0.0, "motion_isotropy": 0.0,
            "has_motion": 1.0 if has_motion else 0.0,
            "has_target": 1.0 if has_target else 0.0,
            "knee_target_error": float(slot_error_defaults["knee"][node_idx]),
            "ankle_target_error": float(slot_error_defaults["ankle"][node_idx]),
            "foot_target_error": float(slot_error_defaults["foot"][node_idx]),
        }
        if motion is not None and motion.shape[0] > node_idx:
            traj = _normalize_trajectory(motion[node_idx])
            rom_x, rom_y = _trajectory_span(traj)
            values["motion_rom_x"] = _clip01(rom_x)
            values["motion_rom_y"] = _clip01(rom_y)
            values["motion_curvature"] = _clip01(_trajectory_curvature(traj) * 3.0)
            values["motion_loop_closure"] = _clip01(_loop_closure_ratio(traj))
            values["motion_isotropy"] = _clip01(_trajectory_isotropy(traj))
        features[node_idx] = np.array([values[name] for name in _NODE_FEATURE_ORDER], dtype=np.float32)
    valid_mask = ~(anchors)
    return features, valid_mask


class SlotPointerReadoutAssignment:
    """
    基于槽位指针的关节链分配。
    训练 MLP 为每个节点输出 3 个槽位分数，推理时结合规则先验排序。
    """

    def __init__(
        self,
        *,
        top_k: int = 5,
        hidden_dim: int = 48,
        require_consecutive_semantic_chain: bool = False,
    ):
        self.top_k = int(top_k)
        self.hidden_dim = int(hidden_dim)
        self.require_consecutive_semantic_chain = bool(require_consecutive_semantic_chain)
        self.teacher = RuleBasedReadoutAssignment(
            top_k=top_k,
            require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
        )
        self.model: _SlotPointerMLP | None = None
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def fit(
        self,
        records: Sequence[ReadoutAssignmentRecord],
        *,
        epochs: int = 120,
        lr: float = 1e-2,
        seed: int = 0,
    ) -> dict[str, float]:
        """训练槽位指针 MLP。损失 = 三槽位交叉熵 + 唯一性正则。"""
        if not records:
            return {"loss": 0.0, "num_records": 0.0}
        torch.manual_seed(int(seed))
        rng = torch.Generator().manual_seed(int(seed))

        dataset: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        all_features: list[torch.Tensor] = []
        for record in records:
            target = record.resolved_target()
            candidates = enumerate_leg_candidates(
                record.graph, motion=record.motion, target=target,
                require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
            )
            if not candidates:
                continue
            node_features, valid_mask = _node_feature_matrix(
                record.graph, motion=record.motion, target=target, candidates=candidates,
            )
            if not valid_mask.any():
                continue
            label = None
            if record.truth is not None:
                label = torch.tensor(
                    [int(record.truth["knee"]), int(record.truth["ankle"]), int(record.truth["foot"])],
                    dtype=torch.long,
                )
            else:
                teacher_result = self.teacher.assign(record.graph, motion=record.motion, target=target)
                if teacher_result is not None:
                    label = torch.tensor(
                        [
                            int(teacher_result.keypoints["knee"]),
                            int(teacher_result.keypoints["ankle"]),
                            int(teacher_result.keypoints["foot"]),
                        ],
                        dtype=torch.long,
                    )
            if label is None:
                continue
            feature_tensor = torch.tensor(node_features, dtype=torch.float32)
            valid_tensor = torch.tensor(valid_mask, dtype=torch.bool)
            dataset.append((feature_tensor, valid_tensor, label))
            all_features.append(feature_tensor[valid_tensor])

        if not dataset:
            return {"loss": 0.0, "num_records": 0.0}

        stacked = torch.cat(all_features, dim=0)
        self.feature_mean = stacked.mean(dim=0)
        self.feature_std = stacked.std(dim=0).clamp_min(1e-6)
        self.model = _SlotPointerMLP(stacked.size(1), hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        last_loss = 0.0

        for _ in range(int(epochs)):
            total_loss = 0.0
            order = torch.randperm(len(dataset), generator=rng).tolist()
            for dataset_idx in order:
                features, valid_mask, label = dataset[dataset_idx]
                normalized = (features - self.feature_mean) / self.feature_std
                logits = self.model(normalized)
                masked_logits = logits.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
                loss = 0.0
                for slot_idx in range(3):
                    loss = loss + F.cross_entropy(masked_logits[:, slot_idx].unsqueeze(0), label[slot_idx].view(1))
                probs = torch.softmax(masked_logits, dim=0)
                uniqueness = sum(torch.sum(probs[:, i] * probs[:, j]) for i, j in ((0, 1), (0, 2), (1, 2)))
                total = loss + 0.10 * uniqueness
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                total_loss += float(total.item())
            last_loss = total_loss / max(1, len(dataset))

        return {"loss": float(last_loss), "num_records": float(len(dataset))}

    def assign(
        self,
        graph: object,
        *,
        target: AssignmentTarget | Mapping[str, object] | None = None,
        motion: np.ndarray | Mapping[str, object] | None = None,
    ) -> AssignmentResult | None:
        """使用槽位指针 MLP 执行分配。模型未训练时回退到规则方法。"""
        target_obj = target if isinstance(target, AssignmentTarget) else AssignmentTarget.from_mapping(target)
        motion_array = _resolve_motion(graph, motion)
        candidates = enumerate_leg_candidates(
            graph, motion=motion_array, target=target_obj,
            require_consecutive_semantic_chain=self.require_consecutive_semantic_chain,
        )
        if not candidates:
            return None
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            return self.teacher.assign(graph, target=target_obj, motion=motion_array)

        node_features, valid_mask = _node_feature_matrix(
            graph, motion=motion_array, target=target_obj, candidates=candidates,
        )
        normalized = (torch.tensor(node_features, dtype=torch.float32) - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(normalized)
        masked_logits = logits.masked_fill(~torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(-1), float("-inf"))
        slot_scores = masked_logits.cpu().numpy()

        ranked_candidates: list[CandidateLegChain] = []
        for candidate in candidates:
            pointer_score = float(
                slot_scores[candidate.knee, 0]
                + slot_scores[candidate.ankle, 1]
                + slot_scores[candidate.foot, 2]
            )
            combined = pointer_score + 0.10 * candidate.features.get("rule_score", 0.0)
            ranked_candidates.append(
                CandidateLegChain(
                    hip=candidate.hip, knee=candidate.knee,
                    ankle=candidate.ankle, foot=candidate.foot,
                    path=candidate.path, features=dict(candidate.features),
                    score_breakdown={
                        "slot_pointer": pointer_score,
                        "rule_prior": float(0.10 * candidate.features.get("rule_score", 0.0)),
                        "total": combined,
                    },
                    score=float(combined),
                )
            )

        ranked = sorted(ranked_candidates, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        return AssignmentResult(
            method="C_slot_pointer_soft_matching",
            keypoints=best.keypoints(),
            path=best.path,
            score=float(best.score),
            score_breakdown=dict(best.score_breakdown),
            candidate_count=len(ranked),
            top_candidates=ranked[: self.top_k],
        )


# ==============================================================================
# 合成数据和评估工具
# ==============================================================================

def build_synthetic_readout_records(
    *,
    num_records: int,
    seed: int = 0,
    steps: int = 64,
) -> list[ReadoutAssignmentRecord]:
    """构建合成的读出分配训练记录（9 节点图，upper/lower 两分支交替）。"""
    rng = np.random.default_rng(int(seed))
    t = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float32)
    base_edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 8]],
        dtype=np.int64,
    )
    edge_index = np.concatenate([base_edges.T, base_edges[:, ::-1].T], axis=1)
    records: list[ReadoutAssignmentRecord] = []

    for record_idx in range(int(num_records)):
        truth_branch = "upper" if record_idx % 2 == 0 else "lower"
        pos = np.array(
            [
                [0.0, 0.0], [0.0, 1.0], [0.7, 1.6], [1.4, 1.2], [2.0, 0.4],
                [0.7, 0.5], [1.4, 0.2], [2.0, -0.1], [1.1, 2.2],
            ],
            dtype=np.float32,
        )
        pos += rng.normal(scale=0.035, size=pos.shape).astype(np.float32)

        x = np.column_stack(
            [
                pos,
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ]
        ).astype(np.float32)

        hip = np.stack(
            [
                0.05 * np.cos(t + rng.uniform(-0.4, 0.4)),
                1.0 + 0.04 * np.sin(t + rng.uniform(-0.4, 0.4)),
            ],
            axis=-1,
        ).astype(np.float32)

        rich_phase = float(rng.uniform(-0.6, 0.6))
        rich_scale = float(rng.uniform(0.9, 1.1))
        rich_knee = np.stack(
            [0.55 + 0.18 * np.cos(t + 0.4 + rich_phase), 1.45 + 0.12 * np.sin(t + rich_phase)],
            axis=-1,
        ).astype(np.float32)
        rich_ankle = np.stack(
            [1.20 + rich_scale * 0.28 * np.cos(t + 0.1 + rich_phase), 0.95 + 0.18 * np.sin(t + 0.5 + rich_phase)],
            axis=-1,
        ).astype(np.float32)
        rich_foot = np.stack(
            [
                1.85 + rich_scale * 0.45 * np.cos(t + rich_phase),
                0.30 + rich_scale * (0.22 * np.sin(t + rich_phase) + 0.08 * np.sin(2.0 * t + rich_phase)),
            ],
            axis=-1,
        ).astype(np.float32)

        distractor_phase = float(rng.uniform(-0.6, 0.6))
        distractor_scale = float(rng.uniform(0.75, 1.0))
        distractor_knee = np.stack(
            [0.50 + 0.10 * np.cos(t + distractor_phase), 0.55 + 0.08 * np.sin(t + distractor_phase)],
            axis=-1,
        ).astype(np.float32)
        distractor_ankle = np.stack(
            [1.15 + 0.10 * np.cos(t + 0.5 + distractor_phase), 0.18 + 0.08 * np.sin(t + distractor_phase)],
            axis=-1,
        ).astype(np.float32)
        distractor_foot = np.stack(
            [1.80 + distractor_scale * 0.17 * np.cos(t + distractor_phase), -0.02 + distractor_scale * 0.16 * np.sin(t + distractor_phase)],
            axis=-1,
        ).astype(np.float32)

        near_static = np.stack(
            [1.05 + 0.03 * np.cos(t + rng.uniform(-0.3, 0.3)), 2.15 + 0.03 * np.sin(t + rng.uniform(-0.3, 0.3))],
            axis=-1,
        ).astype(np.float32)

        motion = np.zeros((9, steps, 2), dtype=np.float32)
        motion[0] = np.repeat(np.array([[0.0, 0.0]], dtype=np.float32), steps, axis=0)
        motion[1] = hip
        if truth_branch == "upper":
            motion[2], motion[3], motion[4] = rich_knee, rich_ankle, rich_foot
            motion[5], motion[6], motion[7] = distractor_knee, distractor_ankle, distractor_foot
            truth = {"hip": 1, "knee": 2, "ankle": 3, "foot": 4}
        else:
            lower_offset = np.array([0.0, -0.72], dtype=np.float32)
            motion[2], motion[3], motion[4] = distractor_knee + np.array([0.0, 0.92], dtype=np.float32), distractor_ankle + np.array([0.0, 0.92], dtype=np.float32), distractor_foot + np.array([0.0, 0.72], dtype=np.float32)
            motion[5], motion[6], motion[7] = rich_knee + lower_offset, rich_ankle + lower_offset, rich_foot + lower_offset
            truth = {"hip": 1, "knee": 5, "ankle": 6, "foot": 7}
        motion[8] = near_static

        target = AssignmentTarget.from_motion(
            motion, hip=truth["hip"], knee=truth["knee"],
            ankle=truth["ankle"], foot=truth["foot"],
        )
        graph = {"x": x, "pos": pos, "edge_index": edge_index}
        records.append(
            ReadoutAssignmentRecord(graph=graph, motion=motion, target=target, truth=truth)
        )
    return records


def evaluate_assignment_module(
    module,
    records: Sequence[ReadoutAssignmentRecord],
) -> dict[str, float]:
    """评估分配模块准确率：完全匹配率和各关节单独匹配率。"""
    exact = 0
    knee = 0
    ankle = 0
    foot = 0
    score_total = 0.0
    for record in records:
        result = module.assign(record.graph, motion=record.motion, target=record.resolved_target())
        if result is None:
            continue
        truth = record.truth or {}
        exact += int(
            result.keypoints.get("knee") == truth.get("knee")
            and result.keypoints.get("ankle") == truth.get("ankle")
            and result.keypoints.get("foot") == truth.get("foot")
        )
        knee += int(result.keypoints.get("knee") == truth.get("knee"))
        ankle += int(result.keypoints.get("ankle") == truth.get("ankle"))
        foot += int(result.keypoints.get("foot") == truth.get("foot"))
        score_total += float(result.score)
    denom = max(1, len(records))
    return {
        "exact_chain_accuracy": float(exact / denom),
        "knee_accuracy": float(knee / denom),
        "ankle_accuracy": float(ankle / denom),
        "foot_accuracy": float(foot / denom),
        "mean_assignment_score": float(score_total / denom),
    }
