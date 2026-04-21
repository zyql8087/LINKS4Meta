from __future__ import annotations

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

from src.inverse.phase5_rl import build_family_curriculum, build_trace_dataset
from src.inverse.rl_env import MechanismEnv, apply_j_operator, batch_compute_phase5_rewards


def _base_4bar_graph():
    x0 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    types = np.array([1, 0, 0, 1], dtype=np.float32)
    grounded = np.array([1, 0, 0, 0], dtype=np.float32)
    x_feat = np.column_stack([x0, types, grounded])
    edges = np.array(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 0], [0, 3]],
        dtype=np.int64,
    )
    return Data(
        x=torch.tensor(x_feat, dtype=torch.float32),
        pos=torch.tensor(x0, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).T,
        knee_idx=torch.tensor([1], dtype=torch.long),
    )


class _DummySurrogate(torch.nn.Module):
    def forward(self, batch):
        batch_size = int(batch.ptr.numel() - 1)
        steps = 16
        foot = torch.zeros((batch_size, steps, 2), dtype=torch.float32, device=batch.x.device)
        knee = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.x.device)
        ankle = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.x.device)
        return foot, knee, ankle


class TestPhase5RL(unittest.TestCase):
    def test_family_curriculum_order_matches_phase5(self):
        curriculum = build_family_curriculum({"episodes_per_family": 10})
        self.assertEqual([stage["family"] for stage in curriculum], ["6bar", "7bar", "8bar", "9bar"])

    def test_build_trace_dataset_collapses_multistep_samples(self):
        trace_dataset = build_trace_dataset(
            [
                {
                    "trace_id": 3,
                    "sample_id": 3,
                    "family_id": "8bar",
                    "family_index": 2,
                    "step_count": 2,
                    "base_data": _base_4bar_graph(),
                    "y_foot": torch.zeros((16, 2), dtype=torch.float32),
                    "y_knee": torch.zeros(16, dtype=torch.float32),
                    "y_ankle": torch.zeros(16, dtype=torch.float32),
                    "action_topo": torch.tensor([1, 2, 0], dtype=torch.long),
                    "action_geo": torch.tensor([0.2, 0.7, 0.2, 0.3], dtype=torch.float32),
                    "step_index": 0,
                },
                {
                    "trace_id": 3,
                    "sample_id": 3,
                    "family_id": "8bar",
                    "family_index": 2,
                    "step_count": 2,
                    "base_data": _base_4bar_graph(),
                    "y_foot": torch.zeros((16, 2), dtype=torch.float32),
                    "y_knee": torch.zeros(16, dtype=torch.float32),
                    "y_ankle": torch.zeros(16, dtype=torch.float32),
                    "action_topo": torch.tensor([1, 2, 0], dtype=torch.long),
                    "action_geo": torch.tensor([0.2, 0.7, 0.2, 0.3], dtype=torch.float32),
                    "step_index": 1,
                },
            ]
        )
        self.assertEqual(len(trace_dataset), 1)
        self.assertEqual(trace_dataset[0]["expected_j_steps"], 2)
        self.assertEqual(trace_dataset[0]["family_id"], "8bar")

    def test_phase5_reward_prefers_expected_terminal_stop(self):
        graph = apply_j_operator(_base_4bar_graph(), 1, 2, 0, np.array([0.2, 0.7], dtype=np.float32), np.array([0.2, 0.3], dtype=np.float32))
        target = {
            "y_foot": torch.zeros((16, 2), dtype=torch.float32),
            "y_knee": torch.zeros(16, dtype=torch.float32),
            "y_ankle": torch.zeros(16, dtype=torch.float32),
        }
        rewards, payloads = batch_compute_phase5_rewards(
            _DummySurrogate(),
            [graph, graph],
            target,
            {
                "w_foot": 0.5,
                "w_knee": 0.25,
                "w_ankle": 0.25,
                "foot_mix_chamfer": 0.5,
                "foot_mix_nrmse": 0.5,
                "w_smooth": 0.0,
                "alive_bonus": 0.0,
                "lambda_step": 0.05,
            },
            torch.device("cpu"),
            step_indices=[1, 0],
            stop_flags=[True, True],
            expected_j_steps=1,
            constraint_cfg={"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )
        self.assertGreater(rewards[0][0], rewards[1][0])
        self.assertGreater(payloads[0]["terminal"], 0.0)
        self.assertLessEqual(payloads[0]["r_step_penalty"], 0.0)

    def test_mechanism_env_auto_stops_after_expected_family_steps(self):
        env = MechanismEnv(
            _DummySurrogate(),
            {
                "w_foot": 0.5,
                "w_knee": 0.25,
                "w_ankle": 0.25,
                "foot_mix_chamfer": 0.5,
                "foot_mix_nrmse": 0.5,
                "w_smooth": 0.0,
            },
            max_steps=2,
            device=torch.device("cpu"),
            constraint_cfg={"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )
        obs = env.reset(
            {
                "y_foot": torch.zeros((16, 2), dtype=torch.float32),
                "y_knee": torch.zeros(16, dtype=torch.float32),
                "y_ankle": torch.zeros(16, dtype=torch.float32),
            },
            _base_4bar_graph(),
            torch.zeros((1, 8), dtype=torch.float32),
            family_id="6bar",
            family_index=0,
            expected_j_steps=1,
        )
        self.assertFalse(obs["can_stop"])

        obs, _, done, _ = env.step(
            {
                "u": 1,
                "v": 2,
                "w": 0,
                "n1": np.array([0.2, 0.7], dtype=np.float32),
                "n2": np.array([0.2, 0.3], dtype=np.float32),
                "stop": False,
            }
        )
        self.assertTrue(done)
        rewards, payloads = env.compute_episode_rewards()
        self.assertEqual(len(rewards), 1)
        self.assertEqual(len(payloads), 1)
        self.assertGreaterEqual(payloads[0]["terminal"], 1.0)


if __name__ == "__main__":
    unittest.main()
