import os
import os
import sys
import unittest

import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inverse.readout_assignment import (
    LearnedChainScorerReadoutAssignment,
    RuleBasedReadoutAssignment,
    SlotPointerReadoutAssignment,
    build_synthetic_readout_records,
    enumerate_leg_candidates,
    evaluate_assignment_module,
)
from src.inverse.rl_env import _infer_semantic_masks, _prepare_graph_for_surrogate


class TestReadoutAssignment(unittest.TestCase):
    def test_rule_based_assignment_recovers_truth_on_branching_graph(self):
        record = build_synthetic_readout_records(num_records=1, seed=7)[0]
        module = RuleBasedReadoutAssignment()
        result = module.assign(record.graph, motion=record.motion, target=record.target)

        self.assertIsNotNone(result)
        self.assertEqual(result.keypoints["knee"], record.truth["knee"])
        self.assertEqual(result.keypoints["ankle"], record.truth["ankle"])
        self.assertEqual(result.keypoints["foot"], record.truth["foot"])

    def test_rl_env_fallback_uses_chain_assignment_when_keypoints_missing(self):
        record = build_synthetic_readout_records(num_records=1, seed=11)[0]
        graph = Data(
            x=torch.tensor(record.graph["x"], dtype=torch.float32),
            pos=torch.tensor(record.graph["pos"], dtype=torch.float32),
            edge_index=torch.tensor(record.graph["edge_index"], dtype=torch.long),
        )

        mask_hip, mask_knee, mask_ankle, mask_foot = _infer_semantic_masks(
            graph,
            target={
                "y_foot": torch.tensor(record.target.y_foot, dtype=torch.float32),
                "y_knee": torch.tensor(record.target.y_knee, dtype=torch.float32),
                "y_ankle": torch.tensor(record.target.y_ankle, dtype=torch.float32),
            },
            motion=torch.tensor(record.motion, dtype=torch.float32),
        )

        self.assertEqual(int(mask_hip.sum().item()), 1)
        self.assertEqual(int(mask_knee.sum().item()), 1)
        self.assertEqual(int(mask_ankle.sum().item()), 1)
        self.assertEqual(int(mask_foot.sum().item()), 1)
        self.assertTrue(bool(mask_knee[record.truth["knee"]].item()))
        self.assertTrue(bool(mask_ankle[record.truth["ankle"]].item()))
        self.assertTrue(bool(mask_foot[record.truth["foot"]].item()))

    def test_enumeration_uses_ordered_semantic_chain_by_default(self):
        graph = {
            "x": torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.6, 1.3, 0.0, 0.0],
                    [1.2, 1.2, 0.0, 0.0],
                    [1.8, 0.8, 0.0, 0.0],
                    [2.4, 0.3, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "pos": torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.6, 1.3],
                    [1.2, 1.2],
                    [1.8, 0.8],
                    [2.4, 0.3],
                ],
                dtype=torch.float32,
            ),
            "edge_index": torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
                ],
                dtype=torch.long,
            ),
        }
        candidates = enumerate_leg_candidates(graph)
        self.assertTrue(candidates)
        self.assertTrue(any(candidate.keypoints() == {"hip": 0, "knee": 2, "ankle": 4, "foot": 5} for candidate in candidates))
        for candidate in candidates:
            ordered_positions = [candidate.path.index(node) for node in (candidate.hip, candidate.knee, candidate.ankle, candidate.foot)]
            self.assertEqual(ordered_positions, sorted(ordered_positions))

    def test_enumeration_can_still_enforce_consecutive_semantic_chain(self):
        graph = {
            "x": torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.6, 1.3, 0.0, 0.0],
                    [1.2, 1.2, 0.0, 0.0],
                    [1.8, 0.8, 0.0, 0.0],
                    [2.4, 0.3, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "pos": torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.6, 1.3],
                    [1.2, 1.2],
                    [1.8, 0.8],
                    [2.4, 0.3],
                ],
                dtype=torch.float32,
            ),
            "edge_index": torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
                ],
                dtype=torch.long,
            ),
        }
        candidates = enumerate_leg_candidates(graph, require_consecutive_semantic_chain=True)
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.path[-3:] == (candidate.knee, candidate.ankle, candidate.foot) for candidate in candidates))

    def test_enumeration_supports_anchor_hip_and_nonmonotonic_anchor_depth(self):
        graph = {
            "x": torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 0.8, 0.0, 0.0],
                    [1.5, 1.4, 0.0, 0.0],
                    [2.2, 0.9, 0.0, 0.0],
                    [1.4, -0.1, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "pos": torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.8],
                    [1.5, 1.4],
                    [2.2, 0.9],
                    [1.4, -0.1],
                ],
                dtype=torch.float32,
            ),
            "edge_index": torch.tensor(
                [
                    [0, 0, 1, 2, 2, 3, 4, 5, 2, 3, 3, 4, 4, 5, 5, 0],
                    [1, 2, 0, 0, 3, 2, 5, 4, 4, 1, 4, 2, 3, 0, 4, 5],
                ],
                dtype=torch.long,
            ),
        }
        truth = {"hip": 0, "knee": 2, "ankle": 4, "foot": 5}
        candidates = enumerate_leg_candidates(graph)
        self.assertTrue(any(candidate.keypoints() == truth for candidate in candidates))

    def test_prepare_graph_rewrites_semantic_tail_even_for_rich_features(self):
        graph = Data(
            x=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0],
                    [0.0, 1.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0],
                    [1.0, 1.0, 0.0, 0.0, 7.0, 7.0, 7.0, 7.0],
                    [1.0, 0.0, 1.0, 0.0, 6.0, 6.0, 6.0, 6.0],
                ],
                dtype=torch.float32,
            ),
            pos=torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 3, 0],
                    [1, 0, 2, 1, 3, 2, 0, 3],
                ],
                dtype=torch.long,
            ),
            mask_hip=torch.tensor([1, 0, 0, 0], dtype=torch.bool),
            mask_knee=torch.tensor([0, 1, 0, 0], dtype=torch.bool),
            mask_ankle=torch.tensor([0, 0, 1, 0], dtype=torch.bool),
            mask_foot=torch.tensor([0, 0, 0, 1], dtype=torch.bool),
        )

        prepared = _prepare_graph_for_surrogate(
            graph,
            family_index=0,
            step_index=0,
            expected_j_steps=1,
        )

        self.assertEqual(tuple(prepared.x.shape), (4, 8))
        self.assertTrue(torch.allclose(prepared.x[:, :4], graph.x[:, :4]))
        self.assertTrue(torch.equal(prepared.x[:, 4:], torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )))
        self.assertTrue(bool(prepared.semantic_feature_layout.view(-1)[0].item()))

    def test_semantic_dirty_forces_recompute_instead_of_cached_masks(self):
        record = build_synthetic_readout_records(num_records=1, seed=17)[0]
        graph = Data(
            x=torch.tensor(record.graph["x"], dtype=torch.float32),
            pos=torch.tensor(record.graph["pos"], dtype=torch.float32),
            edge_index=torch.tensor(record.graph["edge_index"], dtype=torch.long),
            mask_knee=torch.tensor([1] + [0] * (len(record.graph["x"]) - 1), dtype=torch.bool),
            mask_ankle=torch.tensor([0, 1] + [0] * (len(record.graph["x"]) - 2), dtype=torch.bool),
            mask_foot=torch.tensor([0, 0, 1] + [0] * (len(record.graph["x"]) - 3), dtype=torch.bool),
            keypoints=torch.tensor([2, 1, 0], dtype=torch.long),
            semantic_dirty=torch.tensor([True], dtype=torch.bool),
        )

        _, mask_knee, mask_ankle, mask_foot = _infer_semantic_masks(
            graph,
            target={
                "y_foot": torch.tensor(record.target.y_foot, dtype=torch.float32),
                "y_knee": torch.tensor(record.target.y_knee, dtype=torch.float32),
                "y_ankle": torch.tensor(record.target.y_ankle, dtype=torch.float32),
            },
            motion=torch.tensor(record.motion, dtype=torch.float32),
        )

        self.assertTrue(bool(mask_knee[record.truth["knee"]].item()))
        self.assertTrue(bool(mask_ankle[record.truth["ankle"]].item()))
        self.assertTrue(bool(mask_foot[record.truth["foot"]].item()))
        if record.truth["knee"] != 0:
            self.assertFalse(bool(mask_knee[0].item()))
        if record.truth["ankle"] != 1 and graph.x.size(0) > 1:
            self.assertFalse(bool(mask_ankle[1].item()))
        if record.truth["foot"] != 2 and graph.x.size(0) > 2:
            self.assertFalse(bool(mask_foot[2].item()))

    def test_learned_demos_train_and_recover_semantics_on_synthetic_split(self):
        train_records = build_synthetic_readout_records(num_records=24, seed=3)
        test_records = build_synthetic_readout_records(num_records=8, seed=103)

        scheme_b = LearnedChainScorerReadoutAssignment()
        scheme_c = SlotPointerReadoutAssignment()
        scheme_b.fit(train_records, epochs=35, lr=8e-3, seed=5)
        scheme_c.fit(train_records, epochs=45, lr=8e-3, seed=5)

        metrics_b = evaluate_assignment_module(scheme_b, test_records)
        metrics_c = evaluate_assignment_module(scheme_c, test_records)

        self.assertGreaterEqual(metrics_b["exact_chain_accuracy"], 0.75)
        self.assertGreaterEqual(metrics_b["foot_accuracy"], 0.75)
        self.assertGreaterEqual(metrics_c["exact_chain_accuracy"], 0.75)
        self.assertGreaterEqual(metrics_c["foot_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
