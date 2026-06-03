from __future__ import annotations

import copy
import json
import pickle
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (GMM_ROOT, GMM_ROOT / "code"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.inverse.gnn_policy import GNNPolicy
from src.inverse.action_codebook import (
    build_action_codebook,
    codebook_bucket_for_step,
    export_action_codebook_v1_json,
    resolve_codebook_bucket_for_step,
)
from src.inverse.phase4_il import (
    attach_oracle_code_targets,
    codebook_cache_is_stale,
    compute_phase4_losses,
    compute_oracle_code_loss,
    constrained_code_choice,
    ensure_multistep_expert_paths,
    evaluate_multistep_reconstruction,
    evaluate_multistep_reconstruction_detailed,
    generate_rollout_aware_samples,
    load_step_split,
    scheduled_sampling_ratio,
)
from train_inverse_bio import _load_initial_il_checkpoint_if_configured


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


class _FailingGeometryPolicy(_DummyPolicy):
    def __init__(self, action_topo):
        super().__init__(action_topo, 0, [[10.0, 10.0, 10.0, 10.0, 1.0, 1.0]])


class _WrongCodePolicy(_DummyPolicy):
    def __init__(self, action_topo, wrong_code_id):
        super().__init__(action_topo, wrong_code_id, [[10.0, 10.0, 10.0, 10.0, 1.0, 1.0]])


class _StepwiseDriftPolicy(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()
        self.curve_latent_dim = 8
        exact_code0 = paths[0]["action_code_target"].float()
        exact_code1 = paths[1]["action_code_target"].float()
        self.code_id0 = int(paths[0]["action_code_id"])
        self.code_id1 = int(paths[1]["action_code_id"])
        self.action_codebook = torch.zeros((max(self.code_id0, self.code_id1) + 1, 6), dtype=torch.float32)
        self.action_codebook[self.code_id0] = exact_code0
        self.action_codebook[self.code_id1] = exact_code1
        self.truth_step1_pos = paths[1]["base_data"].pos.clone()
        self.truth_topos = {
            0: paths[0]["action_topo"].tolist(),
            1: paths[1]["action_topo"].tolist(),
        }

    def encode_graph(self, data):
        return torch.zeros((data.x.size(0), 8), dtype=torch.float32, device=data.x.device)

    def _is_truth_step1_graph(self, data):
        if data.pos.size(0) != self.truth_step1_pos.size(0):
            return False
        truth_pos = self.truth_step1_pos.to(data.pos.device)
        return bool(torch.allclose(data.pos, truth_pos, atol=1e-5))

    def phase4_outputs(self, data, x_enc, z_c, family_ids=None, step_indices=None, step_counts=None):
        step_index = int(step_indices.view(-1)[0].item())
        topo = list(self.truth_topos[step_index])
        if step_index == 0:
            topo = [2, 1, 3]
        if step_index == 1 and not self._is_truth_step1_graph(data):
            topo = [2, 1, 0]
        num_nodes = x_enc.size(0)
        u_logits = torch.full((num_nodes,), -10.0, dtype=torch.float32, device=x_enc.device)
        v_logits = torch.full((num_nodes,), -10.0, dtype=torch.float32, device=x_enc.device)
        w_logits = torch.full((num_nodes,), -10.0, dtype=torch.float32, device=x_enc.device)
        u_logits[topo[0]] = 10.0
        v_logits[topo[1]] = 10.0
        w_logits[topo[2]] = 10.0
        stop_logit = 10.0 if step_index == 1 else -10.0
        role_logits = [10.0, 0.0] if step_index == 0 else [0.0, 10.0]
        return {
            "graph_context": torch.zeros((1, 8), dtype=torch.float32, device=x_enc.device),
            "u_logits": u_logits,
            "v_logits": v_logits,
            "w_logits": w_logits,
            "stop_logits": torch.tensor([stop_logit], dtype=torch.float32, device=x_enc.device),
            "step_role_logits": torch.tensor([role_logits], dtype=torch.float32, device=x_enc.device),
            "step_count_logits": torch.tensor([[0.0, 10.0]], dtype=torch.float32, device=x_enc.device),
        }

    def predict_geometry_code(self, batch_graph, x_enc, graph_context, action_topos, family_ids=None, step_roles=None, step_indices=None):
        step_role = int(step_roles.view(-1)[0].item())
        if step_role == 0:
            return torch.tensor([self.code_id0], dtype=torch.long, device=x_enc.device)
        return torch.tensor([self.code_id1], dtype=torch.long, device=x_enc.device)


class TestPhase4IL(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = GMM_ROOT / "demo" / "outputs" / "phase4_unit"
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

    def test_action_codebook_builder_uses_representative_vector_and_assignments(self):
        step_paths = [
            {"action_code_bucket": "semantic_67", "action_code_vec": np.array([0.10, 0.10, 0.10, 0.10, 1.0, 1.0], dtype=np.float32)},
            {"action_code_bucket": "semantic_67", "action_code_vec": np.array([0.12, 0.08, 0.11, 0.09, 1.0, 1.0], dtype=np.float32)},
            {"action_code_bucket": "semantic_67", "action_code_vec": np.array([0.90, 0.90, 0.90, 0.90, -1.0, -1.0], dtype=np.float32)},
        ]

        codebook = build_action_codebook(step_paths, cluster_radius=0.10, max_codes_per_bucket=4)

        self.assertEqual(len(codebook["entries"]), 2)
        self.assertEqual(sorted(int(entry["count"]) for entry in codebook["entries"]), [1, 2])
        self.assertEqual(len(codebook["item_assignments"]), 3)
        self.assertEqual(len(codebook["bucket_to_ids"]["semantic_67"]), 2)
        vectors = [np.array(entry["vector"], dtype=np.float32) for entry in codebook["entries"]]
        self.assertTrue(any(np.allclose(vector, step_paths[0]["action_code_vec"]) for vector in vectors))
        self.assertFalse(any(np.allclose(vector[:4], np.array([0.11, 0.09, 0.105, 0.095], dtype=np.float32)) for vector in vectors))

    def test_action_codebook_uses_targeted_bucket_code_limits(self):
        step_paths = []
        for bucket in ("aux_shared", "semantic_67"):
            for idx in range(4):
                step_paths.append(
                    {
                        "action_code_bucket": bucket,
                        "action_code_vec": np.array(
                            [float(idx), float(idx), float(idx), float(idx), 1.0, 1.0],
                            dtype=np.float32,
                        ),
                    }
                )

        codebook = build_action_codebook(
            step_paths,
            cluster_radius=0.01,
            max_codes_per_bucket=1,
            max_codes_per_bucket_overrides={"aux_shared": 3},
        )

        self.assertEqual(len(codebook["bucket_to_ids"]["aux_shared"]), 3)
        self.assertEqual(len(codebook["bucket_to_ids"]["semantic_67"]), 1)
        self.assertEqual(codebook["max_codes_per_bucket"], 1)
        self.assertEqual(codebook["max_codes_per_bucket_overrides"], {"aux_shared": 3})

    def test_action_codebook_uses_wildcard_bucket_code_limits(self):
        step_paths = []
        for bucket in ("semantic_9bar_step1_topo_5_4_1", "semantic_8bar_step1_topo_4_2_3"):
            for idx in range(4):
                step_paths.append(
                    {
                        "action_code_bucket": bucket,
                        "action_code_vec": np.array(
                            [float(idx), float(idx), float(idx), float(idx), 1.0, 1.0],
                            dtype=np.float32,
                        ),
                    }
                )

        codebook = build_action_codebook(
            step_paths,
            cluster_radius=0.01,
            max_codes_per_bucket=1,
            max_codes_per_bucket_overrides={"semantic_9bar_step1_topo_*": 2},
        )

        self.assertEqual(len(codebook["bucket_to_ids"]["semantic_9bar_step1_topo_5_4_1"]), 2)
        self.assertEqual(len(codebook["bucket_to_ids"]["semantic_8bar_step1_topo_4_2_3"]), 1)

    def test_action_codebook_v2_selects_validity_best_medoid_and_exports_metadata(self):
        step_paths = [
            {
                "action_code_bucket": "semantic_9bar_step1_topo_5_4_1",
                "action_code_vec": np.array([0.10, 0.10, 0.10, 0.10, 1.0, 1.0], dtype=np.float32),
                "base_data": torch.zeros(1),
                "action_topo": torch.tensor([0, 0, 0]),
            },
            {
                "action_code_bucket": "semantic_9bar_step1_topo_5_4_1",
                "action_code_vec": np.array([0.12, 0.08, 0.11, 0.09, 1.0, 1.0], dtype=np.float32),
                "base_data": torch.zeros(1),
                "action_topo": torch.tensor([0, 0, 0]),
            },
            {
                "action_code_bucket": "semantic_9bar_step1_topo_5_4_1",
                "action_code_vec": np.array([0.14, 0.06, 0.12, 0.08, 1.0, 1.0], dtype=np.float32),
                "base_data": torch.zeros(1),
                "action_topo": torch.tensor([0, 0, 0]),
            },
        ]

        def fake_valid(vector, _step, _constraint_cfg):
            return bool(np.allclose(vector, step_paths[1]["action_code_vec"]))

        with patch("src.inverse.action_codebook._vector_valid_for_step", side_effect=fake_valid):
            codebook = build_action_codebook(
                step_paths,
                cluster_radius=0.20,
                max_codes_per_bucket=4,
                representative_strategy="validity_best_medoid",
                fine_bucket_policy={"enabled": True, "include_step_index": True, "include_action_topo": True},
            )

        self.assertEqual(codebook["version"], "geom_codebook_v2_validity_context_bucket")
        self.assertEqual(codebook["representative_strategy"], "validity_best_medoid")
        self.assertEqual(codebook["fine_bucket_policy"]["include_step_index"], True)
        entry = codebook["entries"][0]
        self.assertTrue(np.allclose(np.array(entry["vector"], dtype=np.float32), step_paths[1]["action_code_vec"]))
        self.assertEqual(entry["validity_pass_count"], 3)
        self.assertEqual(entry["validity_context_count"], 3)
        self.assertEqual(entry["representative_item_idx"], 1)

        json_path = self.tmp_dir / "geom_codebook_v2.json"
        export_action_codebook_v1_json(self.cache_path, codebook, step_paths=step_paths, output_path=json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], "geom_codebook_v2_validity_context_bucket")
        self.assertEqual(payload["codes"][0]["validity_pass_count"], 3)
        self.assertEqual(payload["codes"][0]["representative_item_idx"], 1)

    def test_codebook_bucket_can_split_complex_semantic_by_topology(self):
        self.assertEqual(
            codebook_bucket_for_step("9bar", "semantic", step_index=1, action_topo=[5, 4, 1], fine_topology_buckets=True),
            "semantic_9bar_step1_topo_5_4_1",
        )
        self.assertEqual(
            codebook_bucket_for_step("8bar", "semantic", step_index=1, action_topo=torch.tensor([4, 5, 1]), fine_topology_buckets=True),
            "semantic_8bar_step1_topo_4_5_1",
        )
        self.assertEqual(
            codebook_bucket_for_step("9bar", "semantic", fine_topology_buckets=True),
            "semantic_9bar",
        )
        self.assertEqual(
            resolve_codebook_bucket_for_step(
                {"semantic_9bar": [3]},
                "9bar",
                "semantic",
                step_index=1,
                action_topo=[5, 4, 1],
            ),
            "semantic_9bar",
        )

    def test_codebook_bucket_can_split_complex_aux_by_topology_when_enabled(self):
        policy = {
            "enabled": True,
            "complex_families": ["8bar", "9bar"],
            "semantic_only": False,
            "include_step_index": True,
            "include_action_topo": True,
        }

        self.assertEqual(
            codebook_bucket_for_step(
                "9bar",
                "aux",
                step_index=0,
                action_topo=[5, 2, 1],
                fine_topology_buckets=True,
                fine_bucket_policy=policy,
            ),
            "aux_9bar_step0_topo_5_2_1",
        )
        self.assertEqual(
            resolve_codebook_bucket_for_step(
                {"aux_shared": [0]},
                "9bar",
                "aux",
                step_index=0,
                action_topo=[5, 2, 1],
                fine_bucket_policy=policy,
            ),
            "aux_shared",
        )
        self.assertEqual(
            resolve_codebook_bucket_for_step(
                {"aux_9bar_step0_topo_5_2_1": [1], "aux_shared": [0]},
                "9bar",
                "aux",
                step_index=0,
                action_topo=[5, 2, 1],
                fine_bucket_policy=policy,
            ),
            "aux_9bar_step0_topo_5_2_1",
        )

    def test_legacy_four_bucket_codebook_is_stale_for_v2_policy(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        legacy_codebook = {
            "representative_strategy": None,
            "bucket_to_ids": {
                "aux_shared": [0],
                "semantic_67": [1],
                "semantic_8bar": [2],
                "semantic_9bar": [3],
            },
            "entries": [{"id": idx, "bucket": bucket, "vector": [0, 0, 0, 0, 1, 1], "count": 1} for idx, bucket in enumerate(["aux_shared", "semantic_67", "semantic_8bar", "semantic_9bar"])],
        }

        self.assertTrue(codebook_cache_is_stale(paths, legacy_codebook))

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

    def test_step_split_accepts_precomputed_raw_index_split(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        split_json = self.tmp_dir / "split_indices_raw.json"
        split_json.write_text(json.dumps({"train": [0], "val": [], "test": [1]}), encoding="utf-8")

        split = load_step_split(
            paths,
            split_path=str(self.tmp_dir / "phase4_split_raw.pt"),
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

    def test_detailed_reconstruction_reports_success_for_exact_trace(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        single_trace = [paths[0]]
        dummy_policy = _DummyPolicy(
            paths[0]["action_topo"].tolist(),
            int(paths[0]["action_code_id"]),
            [paths[0]["action_code_target"].tolist()],
        )

        report = evaluate_multistep_reconstruction_detailed(
            dummy_policy,
            _ZeroCurveEncoder(),
            single_trace,
            {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
            torch.device("cpu"),
            max_traces=8,
        )

        self.assertEqual(report["trace_count"], 1)
        self.assertEqual(report["first_failure_type_counts"], {"success": 1})
        self.assertEqual(report["failure_examples"], [])

    def test_detailed_reconstruction_reports_topology_error(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        single_trace = [paths[0]]
        wrong_topology_policy = _DummyPolicy(
            [2, 1, 0],
            int(paths[0]["action_code_id"]),
            [paths[0]["action_code_target"].tolist()],
        )

        report = evaluate_multistep_reconstruction_detailed(
            wrong_topology_policy,
            _ZeroCurveEncoder(),
            single_trace,
            {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
            torch.device("cpu"),
            max_traces=8,
        )

        self.assertEqual(report["first_failure_type_counts"], {"topology_error": 1})
        self.assertEqual(report["failure_examples"][0]["failure_type"], "topology_error")
        self.assertEqual(report["failure_examples"][0]["truth"]["u"], int(paths[0]["action_topo"][0]))
        self.assertEqual(report["failure_examples"][0]["pred"]["u"], 2)

    def test_detailed_reconstruction_reports_geometry_code_error(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        single_trace = [paths[0]]
        wrong_code_policy = _WrongCodePolicy(paths[0]["action_topo"].tolist(), int(paths[0]["action_code_id"]) + 1)

        report = evaluate_multistep_reconstruction_detailed(
            wrong_code_policy,
            _ZeroCurveEncoder(),
            single_trace,
            {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
            torch.device("cpu"),
            max_traces=8,
        )

        self.assertEqual(report["first_failure_type_counts"], {"geometry_code_error": 1})
        self.assertEqual(report["failure_examples"][0]["failure_type"], "geometry_code_error")
        self.assertTrue(report["failure_examples"][0]["topology_match"])
        self.assertFalse(report["failure_examples"][0]["code_match"])

    def test_detailed_reconstruction_reports_state_drift_after_prior_error(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        trace = [item for item in paths if str(item["family_id"]) == "8bar"]
        drift_policy = _StepwiseDriftPolicy(trace)

        report = evaluate_multistep_reconstruction_detailed(
            drift_policy,
            _ZeroCurveEncoder(),
            trace,
            {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
            torch.device("cpu"),
            max_traces=8,
        )

        self.assertEqual(report["first_failure_type_counts"], {"topology_error": 1})
        drift_examples = [
            item for item in report["failure_examples"]
            if item["failure_type"] == "state_drift_after_prior_error"
        ]
        self.assertEqual(len(drift_examples), 1)
        self.assertEqual(drift_examples[0]["step_index"], 1)
        self.assertTrue(drift_examples[0]["prior_step_failed"])
        self.assertTrue(drift_examples[0]["truth_graph_topology_match"])
        self.assertTrue(drift_examples[0]["truth_graph_code_match"])

    def test_reconstruction_metric_marks_invalid_geometry_as_failed_trace(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        single_trace = [paths[0]]
        failing_policy = _FailingGeometryPolicy(paths[0]["action_topo"].tolist())

        metrics = evaluate_multistep_reconstruction(
            failing_policy,
            _ZeroCurveEncoder(),
            single_trace,
            {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
            torch.device("cpu"),
            max_traces=8,
        )

        self.assertEqual(metrics["trace_count"], 1)
        self.assertEqual(metrics["valid_reconstruction_rate"], 0.0)
        self.assertEqual(metrics["reconstruction_success_rate"], 0.0)
        self.assertEqual(metrics["family_success_rate"], {"6bar": 0.0})

    def test_rollout_aware_generation_adds_next_step_from_valid_wrong_rollout(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        trace = [item for item in paths if str(item["family_id"]) == "8bar"]
        drift_policy = _StepwiseDriftPolicy(trace)
        rolled_graph = copy.deepcopy(trace[1]["base_data"])
        rolled_graph.pos = rolled_graph.pos + 0.01

        with patch("src.inverse.phase4_il._apply_predicted_step", return_value=(rolled_graph, None)), patch(
            "src.inverse.phase4_il._truth_action_valid_on_graph",
            return_value=(True, None),
        ):
            samples, report = generate_rollout_aware_samples(
                drift_policy,
                _ZeroCurveEncoder(),
                trace,
                {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
                torch.device("cpu"),
                max_traces=1,
            )

        self.assertEqual(report["generated_count"], 1)
        self.assertEqual(report["by_family"], {"8bar": 1})
        self.assertEqual(samples[0]["step_index"], 1)
        self.assertEqual(samples[0]["rollout_origin"], "policy_rollout")
        self.assertTrue(samples[0]["prior_step_failed"])
        self.assertTrue(samples[0]["rollout_truth_action_valid"])
        self.assertFalse(torch.allclose(samples[0]["base_data"].pos, trace[1]["base_data"].pos))

    def test_rollout_aware_generation_filters_invalid_truth_action_on_rollout_graph(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        trace = [item for item in paths if str(item["family_id"]) == "8bar"]
        drift_policy = _StepwiseDriftPolicy(trace)
        rolled_graph = copy.deepcopy(trace[1]["base_data"])
        rolled_graph.pos = rolled_graph.pos + 0.01

        with patch("src.inverse.phase4_il._apply_predicted_step", return_value=(rolled_graph, None)), patch(
            "src.inverse.phase4_il._truth_action_valid_on_graph",
            return_value=(False, "forced-invalid"),
        ):
            samples, report = generate_rollout_aware_samples(
                drift_policy,
                _ZeroCurveEncoder(),
                trace,
                {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
                torch.device("cpu"),
                max_traces=1,
            )

        self.assertEqual(samples, [])
        self.assertEqual(report["generated_count"], 0)
        self.assertEqual(report["drop_reasons"], {"truth_action_invalid_on_rollout_graph": 1})
        self.assertEqual(report["examples"][0]["invalid_reason"], "forced-invalid")

    def test_rollout_aware_generation_can_rerank_code_to_valid_rollout_state(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        trace = [item for item in paths if str(item["family_id"]) == "8bar"]
        drift_policy = _StepwiseDriftPolicy(trace)
        rolled_graph = copy.deepcopy(trace[1]["base_data"])
        rolled_graph.pos = rolled_graph.pos + 0.01
        top1_code = int(trace[0]["action_code_id"])
        fallback_code = int(trace[1]["action_code_id"])

        def fake_apply(_policy, _graph, pred, _cfg):
            if int(pred["code"]) == top1_code:
                return None, "top1-invalid"
            if int(pred["code"]) == fallback_code:
                return rolled_graph, None
            return None, "unexpected-code"

        cfg = {
            "constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
            "il_training": {"rollout_aware": {"use_validity_rerank_for_generation": True, "max_rerank_codes": 2}},
        }
        with patch("src.inverse.phase4_il._candidate_rollout_code_ids", return_value=[top1_code, fallback_code]), patch(
            "src.inverse.phase4_il._apply_predicted_step",
            side_effect=fake_apply,
        ), patch("src.inverse.phase4_il._truth_action_valid_on_graph", return_value=(True, None)):
            samples, report = generate_rollout_aware_samples(
                drift_policy,
                _ZeroCurveEncoder(),
                trace,
                cfg,
                torch.device("cpu"),
                max_traces=1,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(report["generated_count"], 1)
        self.assertEqual(report["rerank_used_count"], 1)
        self.assertEqual(report["rerank_no_valid_code_count"], 0)

    def test_scheduled_sampling_ratio_interpolates_by_epoch(self):
        self.assertEqual(
            scheduled_sampling_ratio({"rollout_aware": {"enabled": False}}, epoch=4, total_epochs=10),
            0.0,
        )
        stage_cfg = {"rollout_aware": {"enabled": True, "start_ratio": 0.15, "end_ratio": 0.50}}
        self.assertAlmostEqual(scheduled_sampling_ratio(stage_cfg, epoch=0, total_epochs=5), 0.15)
        self.assertAlmostEqual(scheduled_sampling_ratio(stage_cfg, epoch=2, total_epochs=5), 0.325)
        self.assertAlmostEqual(scheduled_sampling_ratio(stage_cfg, epoch=4, total_epochs=5), 0.50)

    def test_initial_il_checkpoint_loader_restores_policy_and_curve_encoder(self):
        policy = torch.nn.Linear(2, 2)
        curve_encoder = torch.nn.Linear(2, 2)
        with torch.no_grad():
            policy.weight.fill_(1.5)
            curve_encoder.weight.fill_(2.5)
        ckpt_path = self.tmp_dir / "initial_il.pt"
        torch.save(
            {
                "policy": {key: value.detach().clone() for key, value in policy.state_dict().items()},
                "curve_encoder": {key: value.detach().clone() for key, value in curve_encoder.state_dict().items()},
            },
            ckpt_path,
        )
        with torch.no_grad():
            policy.weight.zero_()
            curve_encoder.weight.zero_()

        report = _load_initial_il_checkpoint_if_configured(
            policy,
            curve_encoder,
            {"initial_checkpoint": str(ckpt_path)},
            torch.device("cpu"),
        )

        self.assertTrue(report["loaded"])
        self.assertTrue(torch.allclose(policy.weight, torch.full_like(policy.weight, 1.5)))
        self.assertTrue(torch.allclose(curve_encoder.weight, torch.full_like(curve_encoder.weight, 2.5)))

    def test_rollout_aware_sample_runs_existing_phase4_loss(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        trace = [item for item in paths if str(item["family_id"]) == "8bar"]
        rolled_graph = copy.deepcopy(trace[1]["base_data"])
        rolled_graph.pos = rolled_graph.pos + 0.01
        with patch("src.inverse.phase4_il._apply_predicted_step", return_value=(rolled_graph, None)), patch(
            "src.inverse.phase4_il._truth_action_valid_on_graph",
            return_value=(True, None),
        ):
            samples, _ = generate_rollout_aware_samples(
                _StepwiseDriftPolicy(trace),
                _ZeroCurveEncoder(),
                trace,
                {"constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8}},
                torch.device("cpu"),
                max_traces=1,
            )
        self.assertEqual(len(samples), 1)

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
            "il_training": {"use_bucket_masked_code_loss": True},
        }
        policy = GNNPolicy(cfg)
        policy.set_action_codebook(torch.zeros((8, 6), dtype=torch.float32))
        sample = samples[0]
        batch = {
            "base_data": Batch.from_data_list([sample["base_data"]]),
            "action_topo": sample["action_topo"].view(1, -1),
            "y_foot": sample["y_foot"].view(1, -1),
            "y_knee": sample["y_knee"].view(1, -1),
            "y_ankle": sample["y_ankle"].view(1, -1),
            "family_index": torch.tensor([int(sample["family_index"])], dtype=torch.long),
            "step_index": torch.tensor([int(sample["step_index"])], dtype=torch.long),
            "step_count": torch.tensor([int(sample["step_count"])], dtype=torch.long),
            "step_role_index": torch.tensor([int(sample["step_role_index"])], dtype=torch.long),
            "stop_token": torch.tensor([float(sample["stop_token"])], dtype=torch.float32),
            "action_code_id": torch.tensor([int(sample["action_code_id"])], dtype=torch.long),
        }

        metrics = compute_phase4_losses(policy, batch, torch.zeros((1, 8), dtype=torch.float32), cfg)

        self.assertTrue(torch.isfinite(metrics["objective"]))

    def test_oracle_code_targets_mark_valid_expert_equivalent_code(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        step = dict(paths[0])
        exact_vec = step["action_code_vec"].detach().cpu().numpy()
        invalid_vec = np.asarray([10.0, 10.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        bucket = str(step["action_code_bucket"])
        codebook = {
            "entries": [
                {"id": 0, "vector": invalid_vec.tolist(), "bucket": bucket},
                {"id": 1, "vector": exact_vec.tolist(), "bucket": bucket},
            ],
            "bucket_to_ids": {bucket: [0, 1]},
            "fine_bucket_policy": {},
            "code_dim": 6,
        }

        enriched, report = attach_oracle_code_targets(
            [step],
            codebook,
            {
                "constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
                "il_training": {"oracle_code_loss": {"positive_error_threshold": 0.025}},
            },
        )

        self.assertEqual(report["overall"]["count"], 1)
        self.assertEqual(report["overall"]["oracle_positive_coverage_rate"], 1.0)
        self.assertEqual(report["overall"]["valid_code_available_rate"], 1.0)
        self.assertEqual(enriched[0]["oracle_positive_ids"], [1])
        self.assertEqual(enriched[0]["best_valid_code_id"], 1)
        self.assertEqual(enriched[0]["valid_code_ids"], [1])
        self.assertFalse(enriched[0]["oracle_uncovered"])

    def test_oracle_code_loss_prefers_equivalent_valid_code_over_invalid_truth_id(self):
        logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
        batch = {
            "action_code_id": torch.tensor([0], dtype=torch.long),
            "oracle_positive_mask": torch.tensor([[False, True]]),
            "oracle_soft_targets": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "oracle_valid_mask": torch.tensor([[False, True]]),
            "oracle_best_valid_code_id": torch.tensor([1], dtype=torch.long),
            "oracle_uncovered": torch.tensor([False]),
        }
        loss, metrics = compute_oracle_code_loss(
            logits,
            batch,
            {
                "il_training": {
                    "oracle_code_loss": {
                        "enabled": True,
                        "w_hard_code_ce": 0.25,
                        "w_soft_oracle_ce": 1.0,
                        "w_pairwise_rank": 1.0,
                        "w_validity_margin": 1.0,
                        "rank_margin": 0.5,
                        "validity_margin": 0.25,
                    }
                }
            },
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(metrics["loss_oracle_soft_ce"]), 0.0)
        self.assertGreater(float(metrics["loss_oracle_rank"]), 0.0)
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)

    def test_oracle_code_loss_skips_soft_target_when_uncovered(self):
        logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
        batch = {
            "action_code_id": torch.tensor([0], dtype=torch.long),
            "oracle_positive_mask": torch.tensor([[False, False]]),
            "oracle_soft_targets": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "oracle_valid_mask": torch.tensor([[False, True]]),
            "oracle_best_valid_code_id": torch.tensor([1], dtype=torch.long),
            "oracle_uncovered": torch.tensor([True]),
        }
        loss, metrics = compute_oracle_code_loss(
            logits,
            batch,
            {
                "il_training": {
                    "oracle_code_loss": {
                        "enabled": True,
                        "w_hard_code_ce": 0.25,
                        "w_pairwise_rank": 1.0,
                        "w_validity_margin": 1.0,
                    }
                }
            },
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(metrics["loss_oracle_soft_ce"]), 0.0)
        self.assertEqual(float(metrics["oracle_positive_coverage"]), 0.0)
        self.assertGreater(float(metrics["loss_oracle_rank"]), 0.0)
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)

    def test_constrained_code_choice_prefers_first_valid_topk_and_falls_back_to_top1(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        step = dict(paths[0])
        exact_vec = step["action_code_vec"].detach().cpu().numpy()
        invalid_vec = np.asarray([10.0, 10.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        codebook_tensor = torch.tensor([invalid_vec, exact_vec], dtype=torch.float32)

        selected = constrained_code_choice(
            step["base_data"],
            [int(v) for v in step["action_topo"].tolist()],
            [0, 1],
            codebook_tensor,
            {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )

        self.assertEqual(selected["code_id"], 1)
        self.assertTrue(selected["used_non_top1"])
        self.assertTrue(selected["valid"])

        fallback = constrained_code_choice(
            step["base_data"],
            [int(v) for v in step["action_topo"].tolist()],
            [0],
            codebook_tensor,
            {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
        )

        self.assertEqual(fallback["code_id"], 0)
        self.assertFalse(fallback["valid"])


class OracleCoverageDiagnosticsTest(unittest.TestCase):
    def test_threshold_sweep_changes_coverage_and_reports_groups(self):
        from run_il_oracle_coverage_diagnostics import summarize_oracle_candidate_records

        records = [
            {
                "family": "9bar",
                "step_index": 1,
                "step_role": "semantic",
                "bucket": "semantic_9bar_step1_topo_4_2_3",
                "candidate_count": 3,
                "valid_errors": [0.06],
                "best_valid_error": 0.06,
                "truth_code_valid": True,
                "invalid_reason_counts": {"edge_intersection": 2},
            },
            {
                "family": "9bar",
                "step_index": 1,
                "step_role": "semantic",
                "bucket": "semantic_9bar_step1_topo_4_2_3",
                "candidate_count": 2,
                "valid_errors": [],
                "best_valid_error": None,
                "truth_code_valid": False,
                "invalid_reason_counts": {"min_link_length": 2},
            },
        ]

        report = summarize_oracle_candidate_records(records, [0.025, 0.10])

        self.assertEqual(report["overall"]["count"], 2)
        self.assertEqual(report["overall"]["valid_code_available_rate"], 0.5)
        self.assertEqual(report["overall"]["truth_code_valid_rate"], 0.5)
        self.assertEqual(report["overall"]["coverage_by_threshold"]["0.025"]["oracle_positive_coverage_rate"], 0.0)
        self.assertEqual(report["overall"]["coverage_by_threshold"]["0.1"]["oracle_positive_coverage_rate"], 0.5)
        self.assertIn("9bar", report["by_family"])
        self.assertIn("9bar/step1", report["by_family_step_index"])
        self.assertIn("9bar/semantic", report["by_family_step_role"])
        self.assertIn("9bar/step1/semantic", report["by_family_step_index_role"])
        self.assertIn("semantic_9bar_step1_topo_4_2_3", report["by_bucket"])
        self.assertIn("9bar/semantic_9bar_step1_topo_4_2_3", report["by_family_bucket"])
        self.assertEqual(report["invalid_reason_counts"]["edge_intersection"], 2)
        self.assertEqual(report["invalid_reason_counts"]["min_link_length"], 2)
        self.assertEqual(report["candidate_count_stats"]["overall"]["p50"], 2.5)
        self.assertEqual(report["best_valid_error_quantiles"]["overall"]["p50"], 0.06)

    def test_recommendation_relaxes_threshold_when_9bar_is_valid_but_strict_threshold_fails(self):
        from run_il_oracle_coverage_diagnostics import summarize_oracle_candidate_records

        records = [
            {
                "family": "9bar",
                "step_index": 1,
                "step_role": "semantic",
                "bucket": "semantic_9bar_step1_topo_4_2_3",
                "candidate_count": 4,
                "valid_errors": [0.06],
                "best_valid_error": 0.06,
                "truth_code_valid": True,
                "invalid_reason_counts": {},
            }
            for _ in range(10)
        ]

        report = summarize_oracle_candidate_records(records, [0.025, 0.075, 0.10, 0.15])

        self.assertEqual(report["recommendation"]["action"], "relax_equivalence_threshold")
        self.assertEqual(report["recommendation"]["selected_threshold"], 0.075)
        self.assertFalse(report["recommendation"]["train_v8_go"])

    def test_recommendation_rebuilds_when_9bar_valid_candidates_are_missing(self):
        from run_il_oracle_coverage_diagnostics import summarize_oracle_candidate_records

        records = [
            {
                "family": "9bar",
                "step_index": 1,
                "step_role": "semantic",
                "bucket": "semantic_9bar_step1_topo_4_2_3",
                "candidate_count": 4,
                "valid_errors": [],
                "best_valid_error": None,
                "truth_code_valid": False,
                "invalid_reason_counts": {"edge_intersection": 4},
            }
            for _ in range(10)
        ]

        report = summarize_oracle_candidate_records(records, [0.025, 0.075, 0.10, 0.15])

        self.assertEqual(report["recommendation"]["action"], "rebuild_or_expand_codebook")
        self.assertIn("semantic_9bar_step1_topo_4_2_3", report["recommendation"]["target_buckets"])
        self.assertFalse(report["recommendation"]["train_v8_go"])

    def test_recommendation_expands_bucket_when_even_loose_threshold_has_low_coverage(self):
        from run_il_oracle_coverage_diagnostics import summarize_oracle_candidate_records

        records = []
        for idx in range(10):
            records.append(
                {
                    "family": "9bar",
                    "step_index": 1,
                    "step_role": "semantic",
                    "bucket": "semantic_9bar_step1_topo_4_2_3",
                    "candidate_count": 4,
                    "valid_errors": [0.20 if idx < 8 else 0.05],
                    "best_valid_error": 0.20 if idx < 8 else 0.05,
                    "truth_code_valid": True,
                    "invalid_reason_counts": {},
                }
            )

        report = summarize_oracle_candidate_records(records, [0.025, 0.075, 0.10, 0.15])

        self.assertEqual(report["recommendation"]["action"], "increase_per_bucket_codes_or_rebuild")
        self.assertIn("semantic_9bar_step1_topo_4_2_3", report["recommendation"]["target_buckets"])
        self.assertFalse(report["recommendation"]["train_v8_go"])


if __name__ == "__main__":
    unittest.main()
