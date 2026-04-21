from __future__ import annotations

import json
import pickle
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (GMM_ROOT, GMM_ROOT / "code"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.inverse.action_codebook import load_action_codebook
from src.inverse.family_index_builder import build_family_index_artifacts
from src.inverse.phase4_il import ensure_multistep_expert_paths, load_step_split


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
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.20, 0.70], [0.20, 0.30]],
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
            {"step_id": 1, "is_semantic": True, "step_role": "semantic", "u": 1, "v": 2, "w": 0, "n1": 4, "n2": 5}
        ],
        "analysis": {"knee": 1},
        "topology_signature": "sig_6bar",
        **_curve_bundle(),
    }


def _sample_8bar(sample_id: int = 200) -> dict[str, object]:
    A = np.zeros((8, 8), dtype=np.int64)
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 4), (4, 5), (0, 5), (4, 6), (2, 6), (6, 7), (3, 7)):
        _add_edge(A, u, v)
    x0 = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.55, 1.20], [0.20, 0.50], [1.25, 1.20], [1.35, 0.45]],
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
            {"step_id": 1, "is_semantic": False, "step_role": "aux", "u": 1, "v": 2, "w": 0, "n1": 4, "n2": 5},
            {"step_id": 2, "is_semantic": True, "step_role": "semantic", "u": 4, "v": 2, "w": 3, "n1": 6, "n2": 7},
        ],
        "analysis": {"knee": 4},
        "topology_signature": "sig_8bar",
        **_curve_bundle(),
    }


class TestFamilyIndexBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = WORKSPACE_ROOT / "demo" / "outputs" / "family_index_unit"
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)
        self.pkl_path = self.tmp_dir / "phase4_raw.pkl"
        with self.pkl_path.open("wb") as handle:
            pickle.dump([_sample_6bar(), _sample_8bar()], handle)
        self.cache_path = self.tmp_dir / "phase4_steps.pt"

    def test_exports_family_index_split_and_codebook_json(self):
        paths = ensure_multistep_expert_paths(str(self.pkl_path), str(self.cache_path), use_cached=False)
        split_json = self.tmp_dir / "split_indices_v2.json"
        split_json.write_text(json.dumps({"train": [100], "val": [], "test": [200]}), encoding="utf-8")
        split = load_step_split(paths, split_path=str(self.tmp_dir / "phase4_split.pt"), precomputed_split_path=str(split_json))
        codebook = load_action_codebook(self.cache_path)

        manifest = build_family_index_artifacts(
            paths,
            split=split,
            codebook=codebook,
            output_dir=self.tmp_dir / "artifacts",
            dataset_path=self.cache_path,
            export_jsonl=True,
        )

        split_payload = json.loads((self.tmp_dir / "artifacts" / "family_group_split_v1.json").read_text(encoding="utf-8"))
        codebook_payload = json.loads((self.tmp_dir / "artifacts" / "geom_codebook_v1.json").read_text(encoding="utf-8"))
        first_record = json.loads((self.tmp_dir / "artifacts" / "family_step_index_v1.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(manifest["num_step_records"], 3)
        self.assertEqual(split_payload["version"], "family_group_split_v1")
        self.assertIn("100", split_payload["group_meta"])
        self.assertEqual(codebook_payload["version"], "geom_codebook_v1")
        self.assertGreaterEqual(len(codebook_payload["codes"]), 1)
        self.assertIsInstance(first_record["labels"]["geom_code"], int)
        self.assertIn("valid_geom_mask", first_record["masks"])
        self.assertIn("trace_prefix", first_record)


if __name__ == "__main__":
    unittest.main()
