import os
import os
import sys
import unittest

import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inverse.readout_assignment import (
    LearnedChainScorerReadoutAssignment,
    RuleBasedReadoutAssignment,
    SlotPointerReadoutAssignment,
    SurrogateTargetReadoutAssignment,
    build_synthetic_readout_records,
    enumerate_leg_candidates,
    evaluate_assignment_module,
)
from src.inverse.rl_env import _infer_semantic_masks, _prepare_graph_for_surrogate


class FakeTargetAwareSurrogate(torch.nn.Module):
    def __init__(self, target, truth):
        super().__init__()
        self.good_foot = torch.tensor(target.y_foot, dtype=torch.float32)
        self.good_knee = torch.tensor(target.y_knee, dtype=torch.float32)
        self.good_ankle = torch.tensor(target.y_ankle, dtype=torch.float32)
        self.truth = {name: int(value) for name, value in truth.items()}

    def _local_mask_index(self, mask, graph_nodes):
        selected = torch.nonzero(mask[graph_nodes], as_tuple=False).view(-1)
        if selected.numel() == 0:
            return -1
        return int(selected[0].item())

    def forward(self, data):
        num_graphs = int(data.batch.max().item()) + 1 if data.batch.numel() else 1
        foot_rows = []
        knee_rows = []
        ankle_rows = []
        for graph_idx in range(num_graphs):
            graph_nodes = torch.nonzero(data.batch == graph_idx, as_tuple=False).view(-1)
            keypoints = {
                "knee": self._local_mask_index(data.mask_knee, graph_nodes),
                "ankle": self._local_mask_index(data.mask_ankle, graph_nodes),
                "foot": self._local_mask_index(data.mask_foot, graph_nodes),
            }
            matches = all(keypoints[name] == self.truth[name] for name in ("knee", "ankle", "foot"))
            if matches:
                foot_rows.append(self.good_foot)
                knee_rows.append(self.good_knee)
                ankle_rows.append(self.good_ankle)
            else:
                foot_rows.append(torch.ones_like(self.good_foot) * 3.0)
                knee_rows.append(torch.ones_like(self.good_knee) * 3.0)
                ankle_rows.append(torch.ones_like(self.good_ankle) * 3.0)
        return torch.stack(foot_rows), torch.stack(knee_rows), torch.stack(ankle_rows)


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

    def test_graph_target_without_motion_does_not_default_to_perfect_target_error(self):
        record = build_synthetic_readout_records(num_records=1, seed=7)[0]
        candidates = enumerate_leg_candidates(record.graph, target=record.target, motion=None)

        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.features["has_motion"], 0.0)
            self.assertEqual(candidate.features["has_target"], 1.0)
            self.assertEqual(candidate.features["target_error_available"], 0.0)
            self.assertEqual(candidate.features["foot_target_error"], 1.0)
            self.assertEqual(candidate.features["knee_target_error"], 1.0)
            self.assertEqual(candidate.features["ankle_target_error"], 1.0)

    def test_surrogate_target_assignment_uses_target_without_motion(self):
        record = build_synthetic_readout_records(num_records=1, seed=7)[0]
        module = SurrogateTargetReadoutAssignment(
            FakeTargetAwareSurrogate(record.target, record.truth),
            top_k=3,
            batch_size=16,
            metric_cfg={"w_foot": 0.5, "w_knee": 0.25, "w_ankle": 0.25},
        )

        result = module.assign(record.graph, target=record.target, motion=None)

        self.assertIsNotNone(result)
        self.assertEqual(result.method, "D_surrogate_target_chain_assignment")
        self.assertEqual(result.keypoints["knee"], record.truth["knee"])
        self.assertEqual(result.keypoints["ankle"], record.truth["ankle"])
        self.assertEqual(result.keypoints["foot"], record.truth["foot"])
        self.assertIn("joint_score", result.score_breakdown)

    def test_infer_semantic_masks_accepts_surrogate_target_assigner(self):
        record = build_synthetic_readout_records(num_records=1, seed=7)[0]
        graph = Data(
            x=torch.tensor(record.graph["x"], dtype=torch.float32),
            pos=torch.tensor(record.graph["pos"], dtype=torch.float32),
            edge_index=torch.tensor(record.graph["edge_index"], dtype=torch.long),
        )
        assigner = SurrogateTargetReadoutAssignment(
            FakeTargetAwareSurrogate(record.target, record.truth),
            top_k=3,
            batch_size=16,
        )

        mask_hip, mask_knee, mask_ankle, mask_foot = _infer_semantic_masks(
            graph,
            target={
                "y_foot": torch.tensor(record.target.y_foot, dtype=torch.float32),
                "y_knee": torch.tensor(record.target.y_knee, dtype=torch.float32),
                "y_ankle": torch.tensor(record.target.y_ankle, dtype=torch.float32),
            },
            motion=None,
            readout_assigner=assigner,
        )

        self.assertEqual(int(mask_hip.sum().item()), 1)
        self.assertTrue(bool(mask_knee[record.truth["knee"]].item()))
        self.assertTrue(bool(mask_ankle[record.truth["ankle"]].item()))
        self.assertTrue(bool(mask_foot[record.truth["foot"]].item()))

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
