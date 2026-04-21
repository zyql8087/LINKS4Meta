import unittest
import os
import sys

import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inverse.experiment_utils import compute_joint_metrics_batch, select_hard_test_indices
from src.inverse.rl_env import apply_j_operator, validate_graph_structure


class TestExperimentPipeline(unittest.TestCase):
    def test_apply_j_operator_restores_semantic_keypoints(self):
        base_graph = Data(
            x=torch.tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ], dtype=torch.float32),
            pos=torch.tensor([
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ], dtype=torch.float32),
            edge_index=torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 0],
                [1, 0, 2, 1, 3, 2, 0, 3],
            ], dtype=torch.long),
            knee_idx=torch.tensor([2], dtype=torch.long),
        )

        out = apply_j_operator(
            base_graph,
            u=1,
            v=2,
            w=0,
            n1_pos=torch.tensor([0.4, 0.8]).numpy(),
            n2_pos=torch.tensor([0.5, 0.3]).numpy(),
        )

        self.assertTrue(hasattr(out, 'keypoints'))
        self.assertEqual(out.keypoints.tolist(), [5, 2, 4])

    def test_validate_graph_structure_rejects_crossing_edges(self):
        graph = Data(
            x=torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
            pos=torch.tensor([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ], dtype=torch.float32),
            edge_index=torch.tensor([
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ], dtype=torch.long),
        )
        is_valid, info = validate_graph_structure(graph, {'min_link_length': 0.01, 'min_node_distance': 0.01})
        self.assertFalse(is_valid)
        self.assertEqual(info['reason'], 'edge_intersection')

    def test_joint_metrics_zero_for_matching_linear_curves(self):
        foot = torch.stack([torch.linspace(0, 1, 8), torch.linspace(1, 2, 8)], dim=-1).unsqueeze(0)
        knee = torch.linspace(0, 1, 8).unsqueeze(0)
        ankle = torch.linspace(1, 0, 8).unsqueeze(0)
        metrics = compute_joint_metrics_batch(
            foot,
            knee,
            ankle,
            {'y_foot': foot[0], 'y_knee': knee[0], 'y_ankle': ankle[0]},
            {'w_foot': 0.5, 'w_knee': 0.25, 'w_ankle': 0.25},
        )
        self.assertAlmostEqual(metrics['joint_score'][0].item(), 0.0, places=6)
        self.assertAlmostEqual(metrics['smoothness'][0].item(), 0.0, places=6)

    def test_hard_test_selection_prefers_large_amplitude_samples(self):
        samples = []
        for scale in [0.1, 0.2, 1.0, 2.0]:
            foot = torch.stack([torch.linspace(0, scale, 16), torch.linspace(0, scale, 16)], dim=-1)
            samples.append({
                'y_foot': foot,
                'y_knee': torch.linspace(0, scale, 16),
                'y_ankle': torch.linspace(0, scale * 1.5, 16),
            })
        split = {'test_indices': [0, 1, 2, 3]}
        hard = select_hard_test_indices(samples, split, hard_fraction=0.25, min_hard_samples=1)
        self.assertEqual(hard, [3])


if __name__ == '__main__':
    unittest.main()
