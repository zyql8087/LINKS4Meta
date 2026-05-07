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
from src.inverse.inference_runtime import inspect_inverse_checkpoint_geometry_code_state
from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy
from src.inverse.mcts import MCTS, RolloutCandidate
from src.inverse.rl_agent import PPOAgent
from src.inverse.rl_env import MechanismEnv, apply_j_operator, batch_compute_phase5_rewards, batch_compute_rewards


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
    def test_inverse_checkpoint_guard_marks_legacy_geo_head_without_codebook_unready(self):
        status = inspect_inverse_checkpoint_geometry_code_state(
            {
                "policy": {
                    "geo_head.encoder.0.weight": torch.zeros((8, 8), dtype=torch.float32),
                }
            },
            checkpoint_loaded=True,
            action_codebook=None,
            action_codebook_source=None,
            dataset_path="F:/LINKS4Meta/demo/missing_multistep.pt",
        )

        self.assertFalse(status["ready"])
        self.assertEqual(status["checkpoint_variant"], "legacy_geo_head")
        self.assertIn("legacy geo_head", status["issue"])
        self.assertIn(".codebook.pt", status["issue"])

    def test_batch_select_actions_reports_geometry_code_unavailable_when_bundle_not_ready(self):
        cfg = {
            "gnn_policy": {
                "node_input_dim": 4,
                "edge_input_dim": 1,
                "hidden_dim": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "num_families": 4,
                "family_embedding_dim": 4,
                "step_embedding_dim": 4,
                "context_hidden_dim": 16,
                "max_step_count": 2,
                "num_geometry_codes": 4,
                "action_code_dim": 6,
            },
            "curve_encoder": {"input_dim": 8, "hidden_dims": [16], "latent_dim": 8},
            "rl_training": {"learning_rate": 1.0e-4},
            "constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        }
        policy = GNNPolicy(cfg)
        policy.geometry_code_ready = False
        policy.geometry_code_issue = "checkpoint predates geometry_code_head"
        curve_encoder = CurveEncoder(input_dim=8, hidden_dims=[16], latent_dim=8)
        agent = PPOAgent(policy, curve_encoder, cfg, torch.device("cpu"))

        actions, _, _, diagnostics = agent.batch_select_actions(
            [_base_4bar_graph()],
            torch.zeros((1, 8), dtype=torch.float32),
            deterministic=True,
            return_diagnostics=True,
            contexts=[{"family_index": 0, "step_index": 0, "expected_j_steps": 1, "can_stop": False}],
        )

        self.assertIsNone(actions[0])
        self.assertEqual(diagnostics[0]["failure_reason"], "geometry_code_unavailable")
        self.assertIn("geometry_code_head", diagnostics[0]["geometry_code_issue"])

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

    def test_batch_compute_rewards_returns_dense_results_for_valid_graphs(self):
        graph = apply_j_operator(
            _base_4bar_graph(),
            1,
            2,
            0,
            np.array([0.2, 0.7], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
        )
        target = {
            "y_foot": torch.zeros((16, 2), dtype=torch.float32),
            "y_knee": torch.zeros(16, dtype=torch.float32),
            "y_ankle": torch.zeros(16, dtype=torch.float32),
        }
        results = batch_compute_rewards(
            _DummySurrogate(),
            [graph],
            target,
            {
                "w_foot": 0.5,
                "w_knee": 0.25,
                "w_ankle": 0.25,
                "foot_mix_chamfer": 0.5,
                "foot_mix_nrmse": 0.5,
                "w_smooth": 0.0,
            },
            torch.device("cpu"),
            constraint_cfg={"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0][0], float)
        self.assertTrue(bool(results[0][1]))

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

    def test_mcts_rerank_filters_invalid_graphs_before_surrogate(self):
        valid_graph = _base_4bar_graph()
        invalid_graph = Data(
            x=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            pos=torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            edge_index=torch.tensor(
                [
                    [0, 1, 2, 3],
                    [1, 0, 3, 2],
                ],
                dtype=torch.long,
            ),
        )
        mcts = MCTS(
            agent=None,
            surrogate=_DummySurrogate(),
            cfg={
                "reward": {
                    "w_foot": 0.5,
                    "w_knee": 0.25,
                    "w_ankle": 0.25,
                    "foot_mix_chamfer": 0.5,
                    "foot_mix_nrmse": 0.5,
                    "w_smooth": 0.0,
                },
                "constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
            },
            device=torch.device("cpu"),
        )
        target = {
            "y_foot": torch.zeros((16, 2), dtype=torch.float32),
            "y_knee": torch.zeros(16, dtype=torch.float32),
            "y_ankle": torch.zeros(16, dtype=torch.float32),
        }
        scored = mcts._score_candidates(
            [
                RolloutCandidate(graph=invalid_graph, actions=[], log_prob=1.0, stopped=True, step_count=0),
                RolloutCandidate(graph=valid_graph, actions=[], log_prob=0.0, stopped=True, step_count=0),
            ],
            target,
            family_index=0,
            expected_j_steps=1,
        )

        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0]["valid"], 1.0)
        self.assertEqual(scored[1]["valid"], 0.0)
        self.assertEqual(scored[1]["invalid_reason"], "edge_intersection")


if __name__ == "__main__":
    unittest.main()
