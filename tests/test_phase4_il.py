from __future__ import annotations

import json
import pickle
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (GMM_ROOT, GMM_ROOT / "code"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.inverse.gnn_policy import GNNPolicy
from src.inverse.phase4_il import (
    ensure_multistep_expert_paths,
    evaluate_multistep_reconstruction,
    load_step_split,
)


def _add_edge(A: np.ndarray, u: int, v: int) -> None:
    A[u, v] = 1
    A[v, u] = 1


def _curve_bundle(steps: int = 16):
    return {
        "foot_curve": np.stack([np.linspace(0.0, 1.0, steps), np.linspace(1.0, 0.0, steps)], axis=1),
        "knee_curve": np.linspace(0.0, 1.0, steps),
        "ankle_curve": np.linspace(1.0, 0.0, steps),
    }


def _sample_6bar(sample_id: int = 100) -> dict[str, object]:
    A = np.zeros((6, 6), dtype=np.int64)
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 4), (4, 5), (0, 5)):
        _add_edge(A, u, v)
    x0 = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.20, 0.70],
            [0.20, 0.30],
        ],
        dtype=np.float32,
    )
    return {
        "id": sample_id,
        "family_id": "6bar",
        "family": "6bar",
        "step_count": 1,
        "step_roles": ["semantic"],
        "A": A,
        "x0": x0,
        "types": np.array([1, 0, 0, 1, 0, 0], dtype=np.int64),
        "generation_trace": [
            {
                "step_id": 1,
                "is_semantic": True,
                "step_role": "semantic",
                "u": 1,
                "v": 2,
                "w": 0,
                "n1": 4,
                "n2": 5,
            }
        ],
        "analysis": {"knee": 1},
        **_curve_bundle(),
    }


def _sample_8bar(sample_id: int = 200) -> dict[str, object]:
    A = np.zeros((8, 8), dtype=np.int64)
    for u, v in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (1, 4),
        (2, 4),
        (4, 5),
        (0, 5),
        (4, 6),
        (2, 6),
        (6, 7),
        (3, 7),
    ):
        _add_edge(A, u, v)
    x0 = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.55, 1.20],
            [0.20, 0.50],
            [1.25, 1.20],
            [1.35, 0.45],
        ],
        dtype=np.float32,
    )
    return {
        "id": sample_id,
        "family_id": "8bar",
        "family": "8bar",
        "step_count": 2,
        "step_roles": ["aux", "semantic"],
        "A": A,
        "x0": x0,
        "types": np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64),
        "generation_trace": [
            {
                "step_id": 1,
                "is_semantic": False,
                "step_role": "aux",
                "u": 1,
                "v": 2,
                "w": 0,
                "n1": 4,
                "n2": 5,
            },
            {
                "step_id": 2,
                "is_semantic": True,
                "step_role": "semantic",
                "u": 4,
                "v": 2,
                "w": 3,
                "n1": 6,
                "n2": 7,
            },
        ],
        "analysis": {"knee": 4},
        **_curve_bundle(),
    }


class _ZeroCurveEncoder(torch.nn.Module):
    def forward(self, y_foot, y_knee, y_ankle):
        return torch.zeros((y_foot.size(0), 8), dtype=torch.float32, device=y_foot.device)


class _DummyPolicy(torch.nn.Module):
    def __init__(self, action_topo, action_code_id, action_codebook):
        super().__init__()
        self.curve_latent_dim = 8
        self.action_topo = action_topo
        codebook_tensor = torch.zeros((max(int(action_code_id) + 1, 1), 6), dtype=torch.float32)
        codebook_tensor[int(action_code_id)] = torch.tensor(action_codebook[0], dtype=torch.float32)
        self.action_codebook = codebook_tensor
        self.action_code_id = int(action_code_id)

    def encode_graph(self, data):
        return torch.zeros((data.x.size(0), 8), dtype=torch.float32, device=data.x.device)

    def phase4_outputs(self, data, x_enc, z_c, family_ids=None, step_indices=None, step_counts=None):
        num_nodes = x_enc.size(0)
        u_logits = torch.full((num_nodes,), -10.0, dtype=torch.float32, device=x_enc.device)
        v_logits = torch.full((num_nodes,), -10.0, dtype=torch.float32, device=x_enc.device)
        w_logits = torch.full((num_nodes,), -10.0, dtype=torch.float32, device=x_enc.device)
        u_logits[self.action_topo[0]] = 10.0
        v_logits[self.action_topo[1]] = 10.0
        w_logits[self.action_topo[2]] = 10.0
        return {
            "graph_context": torch.zeros((1, 8), dtype=torch.float32, device=x_enc.device),
            "u_logits": u_logits,
            "v_logits": v_logits,
            "w_logits": w_logits,
            "stop_logits": torch.tensor([10.0], dtype=torch.float32, device=x_enc.device),
            "step_role_logits": torch.tensor([[0.0, 10.0]], dtype=torch.float32, device=x_enc.device),
            "step_count_logits": torch.tensor([[10.0, 0.0]], dtype=torch.float32, device=x_enc.device),
        }

    def predict_geometry_code(self, *args, **kwargs):
        device = kwargs["family_ids"].device
        return torch.tensor([self.action_code_id], dtype=torch.long, device=device)


class TestPhase4IL(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = WORKSPACE_ROOT / "demo" / "outputs" / "phase4_unit"
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)
        self.pkl_path = self.tmp_dir / "phase4_raw.pkl"
        with self.pkl_path.open("wb") as handle:
            pickle.dump([_sample_6bar(), _sample_8bar()], handle)
        self.cache_path = self.tmp_dir / "phase4_steps.pt"

    def test_extracts_multistep_expert_paths(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)

        self.assertEqual(len(paths), 3)
        self.assertEqual(paths[0]["family_id"], "6bar")
        self.assertEqual(paths[0]["step_count"], 1)
        self.assertEqual(paths[0]["stop_token"], 1.0)
        self.assertEqual(paths[1]["family_id"], "8bar")
        self.assertEqual(paths[1]["step_role"], "aux")
        self.assertEqual(paths[1]["stop_token"], 0.0)
        self.assertEqual(paths[1]["base_data"].x.size(0), 4)
        self.assertEqual(paths[2]["base_data"].x.size(0), 6)
        self.assertIn("action_code_id", paths[0])
        self.assertEqual(tuple(paths[0]["action_code_vec"].shape), (6,))
        self.assertIn("valid_anchor_mask", paths[0])
        self.assertIn("valid_pair_mask", paths[0])
        self.assertIn("valid_geom_mask", paths[0])
        self.assertIn("semantic_mask", paths[0])
        self.assertIn("trace_prefix", paths[0])
        self.assertIn("seed_graph", paths[0])

    def test_step_split_keeps_same_trace_in_one_group(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        split_json = self.tmp_dir / "split_indices_v2.json"
        split_json.write_text(json.dumps({"train": [100], "val": [], "test": [200]}), encoding="utf-8")

        split = load_step_split(
            paths,
            split_path=str(self.tmp_dir / "phase4_split.pt"),
            precomputed_split_path=str(split_json),
        )

        self.assertEqual(split["train_indices"], [0])
        self.assertEqual(split["test_indices"], [1, 2])

    def test_phase4_policy_outputs_expected_shapes(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        batch = Batch.from_data_list([paths[0]["base_data"], paths[1]["base_data"]])
        cfg = {
            "curve_encoder": {"latent_dim": 8},
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
                "num_geometry_codes": 8,
                "action_code_dim": 6,
            },
            "cvae": {"latent_dim": 4, "prior_bias_init": 0.0, "prior_bias_max": 0.0},
        }

        policy = GNNPolicy(cfg)
        policy.set_action_codebook(torch.zeros((8, 6), dtype=torch.float32))
        x_enc = policy.encode_graph(batch)
        outputs = policy.phase4_outputs(
            batch,
            x_enc,
            torch.zeros((2, 8), dtype=torch.float32),
            family_ids=torch.tensor([0, 2], dtype=torch.long),
            step_indices=torch.tensor([0, 0], dtype=torch.long),
            step_counts=torch.tensor([1, 2], dtype=torch.long),
        )

        self.assertEqual(tuple(outputs["u_logits"].shape), (8,))
        self.assertEqual(tuple(outputs["v_logits"].shape), (8,))
        self.assertEqual(tuple(outputs["w_logits"].shape), (8,))
        self.assertEqual(tuple(outputs["stop_logits"].shape), (2,))
        self.assertEqual(tuple(outputs["step_role_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["step_count_logits"].shape), (2, 2))
        code_logits = policy.geometry_code_logits(batch, x_enc, outputs["graph_context"], torch.stack([paths[0]["action_topo"], paths[1]["action_topo"]]))
        self.assertEqual(tuple(code_logits.shape), (2, 8))

    def test_reconstruction_metric_reports_success_for_exact_trace(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        single_trace = [paths[0]]
        dummy_policy = _DummyPolicy(
            paths[0]["action_topo"].tolist(),
            int(paths[0]["action_code_id"]),
            [paths[0]["action_code_target"].tolist()],
        )
        metrics = evaluate_multistep_reconstruction(
            dummy_policy,
            _ZeroCurveEncoder(),
            single_trace,
            {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
            torch.device("cpu"),
            max_traces=8,
        )

        self.assertEqual(metrics["trace_count"], 1)
        self.assertEqual(metrics["valid_reconstruction_rate"], 1.0)
        self.assertEqual(metrics["reconstruction_success_rate"], 1.0)
        self.assertEqual(metrics["family_success_rate"], {"6bar": 1.0})


if __name__ == "__main__":
    unittest.main()
