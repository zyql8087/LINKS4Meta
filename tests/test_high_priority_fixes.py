import unittest
from unittest import mock

import numpy as np
import torch
from torch_geometric.data import Data

from train_inverse_bio import PreBatchedLoader
from src.inverse.gnn_policy import GNNPolicy
from src.inverse.mcts import compute_batched_rewards
from src.inverse.rl_agent import PPOAgent


class TestHighPriorityFixes(unittest.TestCase):
    def test_prebatched_loader_keeps_tail_batch(self):
        dataset = []
        for idx in range(5):
            base_data = Data(
                x=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
                pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
            )
            dataset.append({
                'base_data': base_data,
                'action_topo': torch.tensor([0, 0, 0], dtype=torch.long),
                'action_geo': torch.zeros(4, dtype=torch.float32),
                'y_foot': torch.zeros(200, 2, dtype=torch.float32),
                'y_knee': torch.zeros(200, dtype=torch.float32),
                'y_ankle': torch.zeros(200, dtype=torch.float32),
            })

        loader = PreBatchedLoader(dataset, batch_size=4, device='cpu', shuffle=False)
        self.assertEqual(len(loader), 2)
        batch_sizes = [batch['action_topo'].shape[0] for batch in loader]
        self.assertEqual(batch_sizes, [4, 1])

    def test_policy_uses_full_config_hidden_dim(self):
        cfg = {
            'gnn_policy': {
                'node_input_dim': 4,
                'edge_input_dim': 1,
                'hidden_dim': 32,
                'num_layers': 2,
                'dropout': 0.0,
            },
            'curve_encoder': {'latent_dim': 16},
            'cvae': {'latent_dim': 8},
        }
        policy = GNNPolicy(cfg)
        self.assertEqual(policy.hidden_dim, 32)
        self.assertEqual(policy.curve_latent_dim, 16)
        self.assertEqual(policy.geo_head.encoder[0].in_features, 4 + 32 + 16)

    def test_batched_rewards_returns_per_graph_scores(self):
        pred_foot = torch.ones(3, 200, 2)
        pred_knee = torch.ones(3, 200)
        pred_ankle = torch.ones(3, 200)
        target = {
            'y_foot': torch.zeros(200, 2),
            'y_knee': torch.zeros(200),
            'y_ankle': torch.zeros(200),
        }
        rewards = compute_batched_rewards(pred_foot, pred_knee, pred_ankle, target, {'w_physics': 0.3})
        self.assertEqual(rewards.shape, (3,))
        self.assertTrue(np.isfinite(rewards).all())

    def test_select_action_evaluates_executed_action_probability(self):
        cfg = {
            'gnn_policy': {
                'node_input_dim': 4,
                'edge_input_dim': 1,
                'hidden_dim': 16,
                'num_layers': 1,
                'dropout': 0.0,
            },
            'curve_encoder': {'latent_dim': 8},
            'cvae': {'latent_dim': 4},
            'rl_training': {'learning_rate': 1e-4},
        }
        policy = GNNPolicy(cfg)
        curve_encoder = torch.nn.Linear(1, 1)
        agent = PPOAgent(policy, curve_encoder, cfg, device='cpu')

        graph = Data(
            x=torch.tensor([
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 1.0, 0.0],
            ], dtype=torch.float32),
            pos=torch.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ], dtype=torch.float32),
            edge_index=torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 0],
                [1, 0, 2, 1, 3, 2, 0, 3],
            ], dtype=torch.long),
        )
        obs = {'graph': graph, 'z_c': torch.zeros(1, 8), 'step': 0}
        action = {'u': 1, 'v': 2, 'w': 0, 'n1': np.zeros(2), 'n2': np.zeros(2)}

        with mock.patch.object(agent, '_evaluate_single_action', wraps=agent._evaluate_single_action) as spy:
            returned_action, log_prob, value = agent.select_action(obs, action=action)
        self.assertEqual(returned_action, action)
        self.assertEqual(spy.call_args[0][1], action)
        self.assertTrue(np.isfinite(log_prob))
        self.assertTrue(np.isfinite(value))


if __name__ == '__main__':
    unittest.main()
