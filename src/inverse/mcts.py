"""
MCTS 推理重排序模块。
推理时枚举 top-k 策略轨迹，使用冻结的代理模型评分并重排序选择最优结果。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Batch, Data

from src.inverse.experiment_utils import compute_joint_metrics_batch, compute_reward_batch
from src.inverse.rl_env import _prepare_graph_for_surrogate, validate_graph_structure


def compute_batched_rewards(
    pred_foot: torch.Tensor,
    pred_knee: torch.Tensor,
    pred_ankle: torch.Tensor,
    target: Dict[str, torch.Tensor],
    reward_cfg: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """计算批量奖励值。封装 compute_reward_batch，仅返回奖励张量。"""
    reward, _ = compute_reward_batch(pred_foot, pred_knee, pred_ankle, target, reward_cfg)
    return reward


@dataclass
class RolloutCandidate:
    """一次 rollout 的候选结果：当前图、动作序列、累积对数概率、是否停止、步数。"""
    graph: Data
    actions: List[Dict]
    log_prob: float
    stopped: bool
    step_count: int


class MCTS:
    """
    Phase6 推理专用 beam search 重排序器。
    逐步展开候选轨迹，保留得分最高的 beam_width 条，最后用代理模型评分选最优。
    """

    def __init__(self, agent, surrogate, cfg: dict, device):
        self.agent = agent
        self.surrogate = surrogate
        self.cfg = cfg
        self.device = device
        mcts_cfg = cfg.get("mcts", {})
        self.branch_top_k = int(mcts_cfg.get("top_k_rollouts", mcts_cfg.get("num_rollouts", 8)))
        self.beam_width = int(mcts_cfg.get("beam_width", self.branch_top_k))

    def _expand_candidates(
        self,
        candidate: RolloutCandidate,
        z_c: torch.Tensor,
        *,
        family_index: int,
        expected_j_steps: int,
    ) -> List[RolloutCandidate]:
        """展开单个候选，调用策略网络获取 top-k 动作并生成后继候选。"""
        if candidate.stopped:
            return [candidate]

        context = {
            "family_index": family_index,
            "step_index": candidate.step_count,
            "expected_j_steps": expected_j_steps,
            "can_stop": candidate.step_count > 0,
            "stop_threshold": self.cfg.get("reward", {}).get("stop_threshold", 0.5),
        }
        ranked_actions = self.agent.rank_action_candidates(
            candidate.graph, z_c, context=context, top_k=self.branch_top_k,
        )
        expanded: List[RolloutCandidate] = []
        for item in ranked_actions:
            action = item["action"]
            if bool(action.get("stop", False)):
                expanded.append(
                    RolloutCandidate(
                        graph=copy.deepcopy(candidate.graph),
                        actions=candidate.actions + [dict(action)],
                        log_prob=float(candidate.log_prob + item["log_prob"]),
                        stopped=True,
                        step_count=int(candidate.step_count),
                    )
                )
                continue

            next_graph = copy.deepcopy(item.get("graph"))
            if next_graph is None:
                continue
            expanded.append(
                RolloutCandidate(
                    graph=next_graph,
                    actions=candidate.actions + [dict(action)],
                    log_prob=float(candidate.log_prob + item["log_prob"]),
                    stopped=False,
                    step_count=int(candidate.step_count + 1),
                )
            )
        return expanded or [candidate]

    def _score_candidates(
        self,
        candidates: List[RolloutCandidate],
        target: dict,
        *,
        family_index: int,
        expected_j_steps: int,
    ) -> List[Dict]:
        """
        对候选列表评分：验证图结构有效性，有效候选送入代理模型计算奖励。
        按有效性 > 代理奖励 > 对数概率降序排序。
        """
        if not candidates:
            return []
        constraint_cfg = self.cfg.get("constraints", {})
        scored = []
        valid_candidates = []
        valid_scored_indices = []

        for candidate in candidates:
            is_valid, valid_info = validate_graph_structure(candidate.graph, constraint_cfg)
            scored.append(
                {
                    "graph": candidate.graph,
                    "actions": candidate.actions,
                    "stopped": candidate.stopped,
                    "step_count": candidate.step_count,
                    "policy_log_prob": float(candidate.log_prob),
                    "surrogate_reward": float("-inf"),
                    "joint_score": float("inf"),
                    "valid": 1.0 if is_valid else 0.0,
                    "invalid_reason": None if is_valid else valid_info.get("reason", "invalid_structure"),
                }
            )
            if is_valid:
                valid_candidates.append(candidate)
                valid_scored_indices.append(len(scored) - 1)

        if valid_candidates:
            prepared_graphs = [
                _prepare_graph_for_surrogate(
                    candidate.graph,
                    family_index=family_index,
                    step_index=candidate.step_count,
                    expected_j_steps=expected_j_steps,
                    target=target,
                )
                for candidate in valid_candidates
            ]
            batch = Batch.from_data_list(prepared_graphs).to(self.device)
            with torch.no_grad():
                pred_foot, pred_knee, pred_ankle = self.surrogate(batch)
            reward_t, _ = compute_reward_batch(
                pred_foot.cpu(), pred_knee.cpu(), pred_ankle.cpu(),
                target, self.cfg.get("reward", {}),
            )
            metrics = compute_joint_metrics_batch(
                pred_foot.cpu(), pred_knee.cpu(), pred_ankle.cpu(),
                target, self.cfg.get("reward", {}),
            )
            for local_idx, scored_idx in enumerate(valid_scored_indices):
                scored[scored_idx]["surrogate_reward"] = float(reward_t[local_idx].item())
                scored[scored_idx]["joint_score"] = float(metrics["joint_score"][local_idx].item())

        scored.sort(
            key=lambda item: (item["valid"], item["surrogate_reward"], item["policy_log_prob"]),
            reverse=True,
        )
        return scored

    def rerank_rollouts(
        self,
        base_graph: Data,
        z_c: torch.Tensor,
        target: dict,
        *,
        family_index: int,
        expected_j_steps: int,
    ) -> Dict:
        """
        执行 beam search 风格的 rollout 重排序。
        返回 {'best': 最优候选, 'candidates': 所有候选}。
        """
        beam = [
            RolloutCandidate(
                graph=copy.deepcopy(base_graph),
                actions=[],
                log_prob=0.0,
                stopped=False,
                step_count=0,
            )
        ]
        max_decisions = max(1, int(expected_j_steps) + 1)
        for _ in range(max_decisions):
            expanded: List[RolloutCandidate] = []
            for candidate in beam:
                expanded.extend(
                    self._expand_candidates(
                        candidate, z_c,
                        family_index=family_index,
                        expected_j_steps=expected_j_steps,
                    )
                )
            expanded.sort(key=lambda item: item.log_prob, reverse=True)
            beam = expanded[: max(1, self.beam_width)]
            if all(candidate.stopped for candidate in beam):
                break

        scored = self._score_candidates(
            beam, target,
            family_index=family_index,
            expected_j_steps=expected_j_steps,
        )
        if scored:
            return {"best": scored[0], "candidates": scored}
        return {"best": None, "candidates": []}

    def search(
        self,
        root_graph: Data,
        z_c: torch.Tensor,
        target: dict,
        *,
        family_index: int = 0,
        expected_j_steps: int = 1,
    ):
        """执行 MCTS 搜索，返回推荐的第一个动作和附加信息。"""
        result = self.rerank_rollouts(
            root_graph, z_c, target,
            family_index=family_index,
            expected_j_steps=expected_j_steps,
        )
        best = result["best"]
        if best is None or not best["actions"]:
            return None, None
        first_action = best["actions"][0]
        if bool(first_action.get("stop", False)):
            return None, None
        return first_action, None
