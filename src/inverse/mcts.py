from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Batch, Data

from src.inverse.experiment_utils import compute_joint_metrics_batch, compute_reward_batch
from src.inverse.rl_env import _prepare_graph_for_surrogate


@dataclass
class RolloutCandidate:
    graph: Data
    actions: List[Dict]
    log_prob: float
    stopped: bool
    step_count: int


class MCTS:
    """
    Phase6 inference-only reranker.

    Training stays on IL/RL. At inference time we enumerate top-k policy rollouts,
    score the resulting final graphs with the frozen surrogate, and rerank them.
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
            candidate.graph,
            z_c,
            context=context,
            top_k=self.branch_top_k,
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
        if not candidates:
            return []
        prepared_graphs = [
            _prepare_graph_for_surrogate(
                candidate.graph,
                family_index=family_index,
                step_index=candidate.step_count,
                expected_j_steps=expected_j_steps,
            )
            for candidate in candidates
        ]
        batch = Batch.from_data_list(prepared_graphs).to(self.device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = self.surrogate(batch)
        reward_t, _ = compute_reward_batch(
            pred_foot.cpu(),
            pred_knee.cpu(),
            pred_ankle.cpu(),
            target,
            self.cfg.get("reward", {}),
        )
        metrics = compute_joint_metrics_batch(
            pred_foot.cpu(),
            pred_knee.cpu(),
            pred_ankle.cpu(),
            target,
            self.cfg.get("reward", {}),
        )

        scored = []
        for idx, candidate in enumerate(candidates):
            scored.append(
                {
                    "graph": candidate.graph,
                    "actions": candidate.actions,
                    "stopped": candidate.stopped,
                    "step_count": candidate.step_count,
                    "policy_log_prob": float(candidate.log_prob),
                    "surrogate_reward": float(reward_t[idx].item()),
                    "joint_score": float(metrics["joint_score"][idx].item()),
                    "valid": 1.0,
                }
            )
        scored.sort(key=lambda item: (item["surrogate_reward"], item["policy_log_prob"]), reverse=True)
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
                        candidate,
                        z_c,
                        family_index=family_index,
                        expected_j_steps=expected_j_steps,
                    )
                )
            expanded.sort(key=lambda item: item.log_prob, reverse=True)
            beam = expanded[: max(1, self.beam_width)]
            if all(candidate.stopped for candidate in beam):
                break

        scored = self._score_candidates(
            beam,
            target,
            family_index=family_index,
            expected_j_steps=expected_j_steps,
        )
        if scored:
            return {
                "best": scored[0],
                "candidates": scored,
            }
        return {
            "best": None,
            "candidates": [],
        }

    def search(
        self,
        root_graph: Data,
        z_c: torch.Tensor,
        target: dict,
        *,
        family_index: int = 0,
        expected_j_steps: int = 1,
    ):
        result = self.rerank_rollouts(
            root_graph,
            z_c,
            target,
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
