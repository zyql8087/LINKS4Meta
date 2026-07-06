from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (GMM_ROOT, GMM_ROOT / "code"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dataset_tool import sample_to_pyg
from src.data_load import DataLoaderFactory
from src.forward_metrics import compute_forward_metrics_batch
from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN
from src.inverse.rl_env import inspect_forward_checkpoint_compatibility


def _mock_sample(family: str = "8bar", sample_id: int = 3) -> dict[str, object]:
    A = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    x0 = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    steps = 200
    return {
        "id": sample_id,
        "family": family,
        "step_count": 2,
        "step_roles": ["aux", "semantic"],
        "A": A,
        "x0": x0,
        "types": np.array([1, 0, 0, 1], dtype=np.int64),
        "foot_curve": np.stack([np.linspace(0.0, 1.0, steps), np.linspace(1.0, 0.0, steps)], axis=1),
        "knee_curve": np.linspace(0.0, 1.0, steps),
        "ankle_curve": np.linspace(1.0, 0.0, steps),
        "descriptors": {"geometry": np.array([0.9, 1.0, 1.1], dtype=np.float32)},
        "analysis": {
            "hip": 0,
            "knee": 1,
            "ankle": 2,
            "foot": 3,
            "x_sol": np.zeros((4, steps, 2), dtype=np.float32),
        },
    }


class TestPhase3Forward(unittest.TestCase):
    def test_sample_to_pyg_includes_phase3_inputs(self):
        data = sample_to_pyg(_mock_sample(), 0)

        self.assertEqual(tuple(data.x.shape), (4, 8))
        self.assertEqual(data.family_id.view(-1).tolist(), [2])
        self.assertEqual(tuple(data.step_context.shape), (1, 3))
        self.assertEqual(tuple(data.retrieval_feature.shape), (1, 23))
        self.assertTrue(bool(data.mask_foot[-1].item()))
        self.assertTrue(bool(data.mask_knee[1].item()))

    def test_forward_model_accepts_family_and_step_context(self):
        sample_a = sample_to_pyg(_mock_sample("6bar", 10), 0)
        sample_b = sample_to_pyg(_mock_sample("9bar", 11), 1)
        batch = Batch.from_data_list([sample_a, sample_b])
        cfg = {
            "encoder": {"hidden_dim": 16, "node_input_dim": 8, "num_layers": 2, "dropout": 0.0},
            "decoder": {
                "hidden_dim": 16,
                "num_layers": 3,
                "num_families": 4,
                "family_embedding_dim": 8,
                "step_context_input_dim": 3,
                "step_context_hidden_dim": 8,
            },
            "training": {"curve_steps": 200},
        }

        model = BioKinematicsGNN(cfg)
        pred_foot, pred_knee, pred_ankle = model(batch)
        self.assertEqual(tuple(pred_foot.shape), (2, 200, 2))
        self.assertEqual(tuple(pred_knee.shape), (2, 200))
        self.assertEqual(tuple(pred_ankle.shape), (2, 200))

    def test_config_encoder_type_matches_actual_forward_encoder(self):
        cfg_path = GMM_ROOT / "src" / "config_model_bio.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        model = BioKinematicsGNN(cfg)

        self.assertEqual(cfg["encoder"]["type"], "MPNN")
        self.assertEqual(model.encoder.conv_type, cfg["encoder"]["type"])

    def test_forward_model_rejects_non_surrogate_node_feature_width(self):
        data = sample_to_pyg(_mock_sample("6bar", 20), 0)
        data.x = data.x[:, :4]
        batch = Batch.from_data_list([data])
        cfg = {
            "encoder": {"hidden_dim": 16, "node_input_dim": 8, "num_layers": 2, "dropout": 0.0},
            "decoder": {
                "hidden_dim": 16,
                "num_layers": 3,
                "num_families": 4,
                "family_embedding_dim": 8,
                "step_context_input_dim": 3,
                "step_context_hidden_dim": 8,
            },
            "training": {"curve_steps": 200},
        }

        model = BioKinematicsGNN(cfg)

        with self.assertRaisesRegex(ValueError, "node feature width"):
            model(batch)

    def test_forward_model_rejects_incomplete_semantic_roles_per_graph(self):
        data = sample_to_pyg(_mock_sample("6bar", 21), 0)
        data.mask_ankle = torch.zeros_like(data.mask_ankle)
        data.x[:, 6] = 0.0
        batch = Batch.from_data_list([data])
        cfg = {
            "encoder": {"hidden_dim": 16, "node_input_dim": 8, "num_layers": 2, "dropout": 0.0},
            "decoder": {
                "hidden_dim": 16,
                "num_layers": 3,
                "num_families": 4,
                "family_embedding_dim": 8,
                "step_context_input_dim": 3,
                "step_context_hidden_dim": 8,
            },
            "training": {"curve_steps": 200},
        }

        model = BioKinematicsGNN(cfg)

        with self.assertRaisesRegex(ValueError, "semantic role 'ankle'"):
            model(batch)

    def test_forward_model_rejects_unknown_family_id(self):
        data = sample_to_pyg(_mock_sample("unknown", 22), 0)
        batch = Batch.from_data_list([data])
        cfg = {
            "encoder": {"hidden_dim": 16, "node_input_dim": 8, "num_layers": 2, "dropout": 0.0},
            "decoder": {
                "hidden_dim": 16,
                "num_layers": 3,
                "num_families": 4,
                "family_embedding_dim": 8,
                "step_context_input_dim": 3,
                "step_context_hidden_dim": 8,
            },
            "training": {"curve_steps": 200},
        }

        model = BioKinematicsGNN(cfg)

        with self.assertRaisesRegex(ValueError, "Unknown forward family_id"):
            model(batch)

    def test_forward_metrics_zero_for_identical_targets(self):
        foot = torch.stack([torch.linspace(0, 1, 8), torch.linspace(1, 0, 8)], dim=-1).unsqueeze(0)
        knee = torch.linspace(0, 1, 8).unsqueeze(0)
        ankle = torch.linspace(1, 0, 8).unsqueeze(0)
        metrics = compute_forward_metrics_batch(foot, knee, ankle, foot, knee, ankle)

        self.assertAlmostEqual(metrics["foot_path_error"][0].item(), 0.0, places=6)
        self.assertAlmostEqual(metrics["foot_chamfer"][0].item(), 0.0, places=6)
        self.assertAlmostEqual(metrics["knee_nmae"][0].item(), 0.0, places=6)
        self.assertAlmostEqual(metrics["ankle_nmae"][0].item(), 0.0, places=6)

    def test_forward_checkpoint_guard_detects_legacy_no_context_variant(self):
        cfg = {
            "encoder": {"hidden_dim": 128, "node_input_dim": 8},
            "decoder": {"family_embedding_dim": 16, "step_context_hidden_dim": 16},
        }
        legacy_state = {
            "encoder.pre_mlp_nodes.weight": torch.zeros((128, 4), dtype=torch.float32),
            "decoder_foot.linear_li.0.weight": torch.zeros((128, 128), dtype=torch.float32),
        }

        result = inspect_forward_checkpoint_compatibility(legacy_state, cfg)

        self.assertEqual(result["checkpoint_variant"], "legacy_4d_no_context")
        self.assertIn("node_input_dim mismatch", "; ".join(result["issues"]))
        self.assertIn("family_embedding", "; ".join(result["issues"]))
        self.assertIn("step_context_encoder", "; ".join(result["issues"]))

    def test_family_filter_remaps_precomputed_split_for_6bar_baseline(self):
        tmp_dir = WORKSPACE_ROOT / "demo" / "outputs" / "phase3_unit_family_filter"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        dataset = [
            sample_to_pyg(_mock_sample("6bar", 10), 0),
            sample_to_pyg(_mock_sample("8bar", 11), 1),
            sample_to_pyg(_mock_sample("6bar", 12), 2),
        ]
        dataset_path = tmp_dir / "dataset.pt"
        torch.save(dataset, dataset_path)
        split_path = tmp_dir / "split.json"
        split_path.write_text(
            json.dumps({"train": [10, 11], "val": [12], "test": []}),
            encoding="utf-8",
        )

        factory = DataLoaderFactory(
            {
                "dataset_path": str(dataset_path),
                "split_indices_path": str(split_path),
                "allowed_families": ["6bar"],
                "__config_dir__": str(tmp_dir),
            }
        )

        self.assertEqual(factory.train_indices, [0])
        self.assertEqual(factory.val_indices, [1])
        self.assertEqual(factory.test_indices, [])


if __name__ == "__main__":
    unittest.main()
