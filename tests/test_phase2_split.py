from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LINKS_ROOT = WORKSPACE_ROOT / "LINKS-main"
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (LINKS_ROOT, GMM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from data_gen_v2.diversity_sampler import cluster_leakage_check, group_split
from src.data_load import DataLoaderFactory
from src.inverse.experiment_utils import load_or_create_fixed_split


def _sample(family: str, topology: str, geometry: list[float], motion: list[float]) -> dict[str, object]:
    return {
        "family": family,
        "descriptors": {
            "topology_cluster_key": topology,
            "geometry": np.asarray(geometry, dtype=np.float32),
            "motion": np.asarray(motion, dtype=np.float32),
        },
    }


class TestPhase2Split(unittest.TestCase):
    def test_group_split_keeps_family_components_unsplit(self):
        samples = [
            _sample("6bar", "topo_a", [0.0, 0.0], [0.0, 0.0]),
            _sample("6bar", "topo_a", [1.0, 1.0], [1.0, 1.0]),
            _sample("6bar", "topo_b", [0.02, 0.0], [2.0, 2.0]),
            _sample("6bar", "topo_c", [2.0, 2.0], [1.02, 1.0]),
            _sample("6bar", "topo_d", [5.0, 5.0], [5.0, 5.0]),
        ]

        splits = group_split(
            samples,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=7,
            geometry_radius=0.05,
            motion_radius=0.05,
        )
        split_by_index = {
            idx: split_name
            for split_name, indices in splits.items()
            for idx in indices
        }

        self.assertEqual(len({split_by_index[idx] for idx in (0, 1, 2, 3)}), 1)
        self.assertEqual(sorted(split_by_index), [0, 1, 2, 3, 4])

    def test_cluster_leakage_check_reports_topology_geometry_and_motion_overlap(self):
        samples = [
            _sample("6bar", "topo_a", [0.0, 0.0], [0.0, 0.0]),
            _sample("6bar", "topo_a", [1.0, 1.0], [1.0, 1.0]),
            _sample("6bar", "topo_b", [0.01, 0.0], [0.01, 0.0]),
        ]
        audit = cluster_leakage_check(
            samples,
            {"train": [0], "val": [], "test": [1, 2]},
            geometry_radius=0.05,
            motion_radius=0.05,
        )
        metrics = audit["pairwise_audit"]["train_test"]

        self.assertEqual(metrics["target_count"], 2)
        self.assertEqual(metrics["topology_repeat_rate"], 0.5)
        self.assertEqual(metrics["geometry_neighbor_rate"], 0.5)
        self.assertEqual(metrics["motion_neighbor_rate"], 0.5)

    def test_data_loader_factory_uses_precomputed_split_indices_json(self):
        tmp_path = WORKSPACE_ROOT / "demo" / "outputs" / "phase2_unit_loader"
        if tmp_path.exists():
            import shutil

            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True)
        dataset_path = tmp_path / "dataset.pt"
        torch.save(
            [
                Data(x=torch.tensor([[0.0]], dtype=torch.float32)),
                Data(x=torch.tensor([[1.0]], dtype=torch.float32)),
                Data(x=torch.tensor([[2.0]], dtype=torch.float32)),
                Data(x=torch.tensor([[3.0]], dtype=torch.float32)),
            ],
            dataset_path,
        )
        split_path = tmp_path / "split.json"
        split_path.write_text(
            json.dumps({"train": [2, 0], "val": [1], "test": [3]}),
            encoding="utf-8",
        )

        factory = DataLoaderFactory(
            {
                "dataset_path": str(dataset_path),
                "split_indices_path": str(split_path),
                "__config_dir__": str(tmp_path),
            }
        )

        self.assertEqual(factory.train_indices, [2, 0])
        self.assertEqual(factory.val_indices, [1])
        self.assertEqual(factory.test_indices, [3])

    def test_inverse_split_uses_precomputed_sample_ids(self):
        tmp_path = WORKSPACE_ROOT / "demo" / "outputs" / "phase2_unit_inverse"
        if tmp_path.exists():
            import shutil

            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True)
        precomputed_split = tmp_path / "split_indices_v2.json"
        precomputed_split.write_text(
            json.dumps({"train": [11], "val": [12], "test": [10]}),
            encoding="utf-8",
        )
        cached_split = tmp_path / "il_split.pt"

        split = load_or_create_fixed_split(
            num_samples=3,
            val_ratio=0.1,
            test_ratio=0.1,
            split_seed=42,
            split_path=str(cached_split),
            precomputed_split_path=str(precomputed_split),
            sample_ids=[10, 12, 11],
        )

        self.assertEqual(split["split_source"], "precomputed_group_split")
        self.assertEqual(split["train_indices"], [2])
        self.assertEqual(split["val_indices"], [1])
        self.assertEqual(split["test_indices"], [0])
        self.assertTrue(cached_split.exists())


if __name__ == "__main__":
    unittest.main()
