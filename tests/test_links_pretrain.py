from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (GMM_ROOT, GMM_ROOT / "code"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy
from src.inverse.pretrain_links import (
    build_links_pretrain_records,
    load_links_pretrained_weights,
    run_links_pretraining,
)
from src.inverse.rl_env import validate_graph_structure


def _add_edge(A: np.ndarray, u: int, v: int) -> None:
    A[u, v] = 1
    A[v, u] = 1


def _curve_bundle(steps: int = 16):
    return {
        "foot_curve": np.stack([np.linspace(0.0, 1.0, steps), np.linspace(1.0, 0.0, steps)], axis=1),
        "knee_curve": np.linspace(0.0, 1.0, steps).astype(np.float32),
        "ankle_curve": np.linspace(1.0, 0.0, steps).astype(np.float32),
    }


def _sample(sample_id: int, family: str) -> dict[str, object]:
    A = np.zeros((6, 6), dtype=np.int64)
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 4), (4, 5), (0, 5)):
        _add_edge(A, u, v)
    x0 = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.2 + 0.1 * sample_id, 0.7],
            [0.2 + 0.1 * sample_id, 0.3],
        ],
        dtype=np.float32,
    )
    return {
        "id": sample_id,
        "family": family,
        "A": A,
        "x0": x0,
        "types": np.array([1, 0, 0, 1, 0, 0], dtype=np.int64),
        **_curve_bundle(),
    }


class TestLinksPretrain(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = WORKSPACE_ROOT / "demo" / "outputs" / "links_pretrain_unit"
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)

    def test_build_records_generates_invalid_negatives(self):
        samples = [_sample(1, "6bar"), _sample(2, "7bar")]
        split_path = self.tmp_dir / "split.json"
        split_path.write_text(json.dumps({"train": [1], "val": [], "test": [2]}), encoding="utf-8")

        cache = build_links_pretrain_records(
            samples,
            split_path=str(split_path),
            max_samples=0,
            seed=7,
            constraint_cfg={"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )
        self.assertEqual(len(cache["records"]), 2)
        self.assertEqual(cache["split"]["train_indices"], [0])
        self.assertEqual(cache["split"]["test_indices"], [1])
        is_valid, info = validate_graph_structure(
            cache["records"][0]["invalid_graph"],
            {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )
        self.assertFalse(is_valid)
        self.assertEqual(info["reason"], "duplicate_nodes")

    def test_run_pretrain_and_reload_checkpoint(self):
        samples = [_sample(1, "6bar"), _sample(2, "7bar"), _sample(3, "8bar")]
        cache = build_links_pretrain_records(
            samples,
            split_path=None,
            max_samples=0,
            seed=3,
            constraint_cfg={"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )
        cfg = {
            "curve_encoder": {"input_dim": 64, "hidden_dims": [32], "latent_dim": 16},
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
            },
            "links_pretrain": {
                "batch_size": 2,
                "learning_rate": 1.0e-3,
                "epochs": 1,
                "patience": 1,
                "temperature": 0.1,
                "contrastive_weight": 1.0,
                "validity_weight": 0.5,
                "projection_dim": 8,
            },
        }
        policy = GNNPolicy(cfg)
        curve_encoder = CurveEncoder(input_dim=64, hidden_dims=[32], latent_dim=16, dropout=0.0)
        ckpt_path = self.tmp_dir / "links_pretrain.pt"
        report_path = self.tmp_dir / "links_pretrain_report.json"

        report = run_links_pretraining(
            policy=policy,
            curve_encoder=curve_encoder,
            cache=cache,
            cfg=cfg,
            device=torch.device("cpu"),
            output_model_path=str(ckpt_path),
            output_report_path=str(report_path),
        )
        self.assertTrue(ckpt_path.exists())
        self.assertTrue(report_path.exists())
        self.assertEqual(report["phase"], "links_pretrain_encoder_validity")
        self.assertIn("forward_pretrain", report["tasks"])
        self.assertTrue((self.tmp_dir / "graph_encoder.pt").exists())
        self.assertTrue((self.tmp_dir / "forward_backbone.pt").exists())
        self.assertTrue((self.tmp_dir / "validity_head.pt").exists())

        policy_reload = GNNPolicy(cfg)
        curve_reload = CurveEncoder(input_dim=64, hidden_dims=[32], latent_dim=16, dropout=0.0)
        load_report = load_links_pretrained_weights(policy_reload, curve_reload, str(ckpt_path), torch.device("cpu"))
        self.assertEqual(load_report["phase"], "links_pretrain_encoder_validity")


if __name__ == "__main__":
    unittest.main()
