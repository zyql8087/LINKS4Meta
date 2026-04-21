from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (GMM_ROOT, GMM_ROOT / "code"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from run_experiment_bio import _build_phase6_analysis
from src.inverse.mcts import MCTS


def _graph_with_terminal_x(x_coord: float) -> Data:
    pos = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [x_coord, 0.0],
        ],
        dtype=np.float32,
    )
    types = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    grounded = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    x_feat = np.column_stack([pos, types, grounded])
    edges = np.array(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 0], [0, 3]],
        dtype=np.int64,
    )
    return Data(
        x=torch.tensor(x_feat, dtype=torch.float32),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).T,
        keypoints=torch.tensor([3, 1, 2], dtype=torch.long),
    )


class _DummyAgent:
    def __init__(self):
        self.calls = 0

    def rank_action_candidates(self, graph, z_c, *, context=None, top_k=None):
        self.calls += 1
        if self.calls > 1:
            return [{"action": {"stop": True}, "log_prob": -0.1, "policy_score": 0.9, "stop": True}]
        return [
            {
                "action": {"u": 1, "v": 2, "w": 0, "n1": np.array([0.2, 0.7], dtype=np.float32), "n2": np.array([0.2, 0.3], dtype=np.float32), "stop": False},
                "graph": _graph_with_terminal_x(0.1),
                "log_prob": -2.0,
                "policy_score": 0.1,
                "stop": False,
            },
            {
                "action": {"u": 1, "v": 2, "w": 0, "n1": np.array([0.8, 0.7], dtype=np.float32), "n2": np.array([0.8, 0.3], dtype=np.float32), "stop": False},
                "graph": _graph_with_terminal_x(1.2),
                "log_prob": -0.1,
                "policy_score": 0.9,
                "stop": False,
            },
        ][: max(1, int(top_k or 1))]


class _DummySurrogate(torch.nn.Module):
    def forward(self, batch):
        batch_size = int(batch.ptr.numel() - 1)
        steps = 12
        foot = torch.zeros((batch_size, steps, 2), dtype=torch.float32, device=batch.x.device)
        knee = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.x.device)
        ankle = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.x.device)
        for idx in range(batch_size):
            start = int(batch.ptr[idx].item())
            end = int(batch.ptr[idx + 1].item())
            terminal_x = float(batch.pos[end - 1, 0].item())
            foot[idx, :, 0] = terminal_x
        return foot, knee, ankle


class TestPhase6MCTS(unittest.TestCase):
    def test_rerank_rollouts_prefers_surrogate_best_over_policy_best(self):
        reranker = MCTS(
            _DummyAgent(),
            _DummySurrogate(),
            {
                "mcts": {"top_k_rollouts": 2, "beam_width": 2},
                "reward": {"w_foot": 1.0, "w_knee": 0.0, "w_ankle": 0.0, "foot_mix_chamfer": 0.0, "foot_mix_nrmse": 1.0, "w_smooth": 0.0},
            },
            torch.device("cpu"),
        )
        target = {
            "y_foot": torch.zeros((12, 2), dtype=torch.float32),
            "y_knee": torch.zeros(12, dtype=torch.float32),
            "y_ankle": torch.zeros(12, dtype=torch.float32),
        }
        result = reranker.rerank_rollouts(
            _graph_with_terminal_x(0.5),
            torch.zeros((1, 8), dtype=torch.float32),
            target,
            family_index=0,
            expected_j_steps=0,
        )
        self.assertIsNotNone(result["best"])
        self.assertAlmostEqual(result["best"]["graph"].pos[-1, 0].item(), 0.1, places=5)
        self.assertGreater(len(result["candidates"]), 1)

    def test_phase6_analysis_reports_mcts_gain_and_9bar_stability(self):
        analysis = _build_phase6_analysis(
            {
                "il_rl": {
                    "summary": {"valid": 0.70, "joint_score": 0.50},
                    "family_breakdown": {"9bar": {"valid": 0.40, "joint_score": 0.90}},
                },
                "il_rl_mcts": {
                    "summary": {"valid": 0.80, "joint_score": 0.30},
                    "family_breakdown": {"9bar": {"valid": 0.60, "joint_score": 0.50}},
                },
            }
        )
        self.assertTrue(analysis["mcts_only_inference"])
        self.assertGreater(analysis["valid_rate_improvement"], 0.0)
        self.assertGreater(analysis["target_match_improvement"], 0.0)
        self.assertTrue(analysis["family_9bar"]["mcts_more_stable_than_no_search"])


if __name__ == "__main__":
    unittest.main()
