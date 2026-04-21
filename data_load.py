import os
import pickle
import json
import torch
import csv
import numpy as np
from pathlib import Path
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from src.forward_dataset_utils import FAMILY_TO_ID, family_id_to_name, sample_to_pyg_data

class GraphCurveDataset(Dataset):
    """
    PyG Dataset for loading the biological 6-bar mechanism data from a .pt file.
    Each sample has: x (node coords), edge_index, edge_attr (lengths),
                     mask_foot, mask_knee, mask_ankle,
                     y_foot, y_knee, y_ankle
    """
    def __init__(self, data_list, transform=None, pre_transform=None):
        super(GraphCurveDataset, self).__init__(None, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class DataLoaderFactory:
    """
    Factory class to create train/val/test DataLoaders from a .pt dataset file.
    """
    def __init__(self, config):
        self.config = config
        self.dataset_path = self._resolve_dataset_path(config.get('dataset_path', ''))
        self.split_indices_path = self._resolve_optional_path(config.get('split_indices_path'))
        self.allowed_family_ids = self._normalize_allowed_family_ids(config.get('allowed_families'))
        self.train_ratio = config.get('train_ratio', 0.8)
        self.val_ratio = config.get('val_ratio', 0.1)
        self.test_ratio = config.get('test_ratio', 0.1)
        self.seed = config.get('seed', 42)
        
        # Load all data
        self._load_data()

    def _resolve_dataset_path(self, dataset_path):
        path = Path(dataset_path)
        if path.is_absolute():
            return str(path)
        candidates = self._path_candidates(path)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _path_candidates(self, path: Path):
        config_dir = Path(self.config.get('__config_dir__', Path.cwd()))
        return [
            (config_dir / path).resolve(),
            (Path.cwd() / path).resolve(),
        ]

    def _resolve_optional_path(self, path_value):
        if not path_value:
            return None
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        candidates = self._path_candidates(path)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _normalize_allowed_family_ids(self, family_values):
        if not family_values:
            return None
        if isinstance(family_values, str):
            family_values = [family_values]
        normalized = set()
        for value in family_values:
            if isinstance(value, str):
                normalized.add(int(FAMILY_TO_ID.get(value, -1)))
            else:
                normalized.add(int(value))
        normalized.discard(-1)
        return normalized if normalized else None

    @staticmethod
    def _scalar_attr(data, attr_name, default=-1):
        value = getattr(data, attr_name, None)
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return int(value.view(-1)[0].item())
        return int(value)

    def _sample_ids(self):
        return [self._scalar_attr(data, 'sample_id', idx) for idx, data in enumerate(self.all_data)]

    def _apply_family_filter(self):
        if not self.allowed_family_ids:
            return
        filtered = []
        for data in self.all_data:
            family_id = self._scalar_attr(data, 'family_id', -1)
            if family_id in self.allowed_family_ids:
                filtered.append(data)
        self.all_data = filtered
        family_names = [family_id_to_name(family_id) for family_id in sorted(self.allowed_family_ids)]
        print(f"Applied family filter: {family_names} -> {len(self.all_data)} samples")
        
    def _load_data(self):
        """Load and split the dataset"""
        print(f"Loading dataset from: {self.dataset_path}")
        
        if self.dataset_path.endswith('.pt'):
            self.all_data = torch.load(self.dataset_path, weights_only=False)
        elif self.dataset_path.endswith('.pkl'):
            with open(self.dataset_path, 'rb') as f:
                raw_data = pickle.load(f)
            # Convert raw data to PyG Data objects
            self.all_data = self._convert_to_pyg(raw_data)
        else:
            raise ValueError(f"Unsupported file format: {self.dataset_path}")

        self._apply_family_filter()
        self._warn_if_phase3_features_missing()
            
        print(f"Loaded {len(self.all_data)} samples")
        
        if self.split_indices_path and Path(self.split_indices_path).exists():
            self._load_precomputed_split()
        else:
            self._random_split()
        
        print(f"Split: train={len(self.train_indices)}, val={len(self.val_indices)}, test={len(self.test_indices)}")

    def _random_split(self):
        torch.manual_seed(self.seed)
        indices = torch.randperm(len(self.all_data)).tolist()

        n_total = len(self.all_data)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train + n_val]
        self.test_indices = indices[n_train + n_val:]

    def _load_precomputed_split(self):
        split_path = Path(self.split_indices_path)
        if split_path.suffix.lower() == '.json':
            with split_path.open('r', encoding='utf-8') as f:
                split = json.load(f)
        else:
            split = torch.load(split_path, map_location='cpu', weights_only=False)

        train_indices = list(split.get('train_indices', split.get('train', [])))
        val_indices = list(split.get('val_indices', split.get('val', [])))
        test_indices = list(split.get('test_indices', split.get('test', [])))

        if self.allowed_family_ids:
            id_to_local = {sample_id: idx for idx, sample_id in enumerate(self._sample_ids())}
            self.train_indices = [id_to_local[idx] for idx in train_indices if idx in id_to_local]
            self.val_indices = [id_to_local[idx] for idx in val_indices if idx in id_to_local]
            self.test_indices = [id_to_local[idx] for idx in test_indices if idx in id_to_local]
        else:
            self.train_indices = train_indices
            self.val_indices = val_indices
            self.test_indices = test_indices

        if not (self.train_indices or self.val_indices or self.test_indices):
            raise ValueError(f'No valid split indices found in {split_path}')
        self._validate_split_indices(split_path)

    def _validate_split_indices(self, split_path):
        seen = set()
        for split_name, indices in (
            ('train', self.train_indices),
            ('val', self.val_indices),
            ('test', self.test_indices),
        ):
            if len(indices) != len(set(indices)):
                raise ValueError(f'Duplicate indices detected in {split_name} split from {split_path}')
            for idx in indices:
                if idx < 0 or idx >= len(self.all_data):
                    raise ValueError(
                        f'Index {idx} in {split_name} split is out of range for dataset of size {len(self.all_data)}'
                    )
            overlap = seen.intersection(indices)
            if overlap:
                raise ValueError(f'Overlap detected across splits in {split_path}: {sorted(overlap)}')
            seen.update(indices)

        if len(seen) != len(self.all_data):
            missing = sorted(set(range(len(self.all_data))) - seen)
            raise ValueError(f'Split file {split_path} does not cover all samples; missing={missing[:10]}')

    def _warn_if_phase3_features_missing(self):
        if not self.all_data:
            return
        sample = self.all_data[0]
        missing = []
        x = getattr(sample, 'x', None)
        if x is None or x.size(-1) < 8:
            missing.append('semantic node channels')
        if not hasattr(sample, 'family_id'):
            missing.append('family_id')
        if not hasattr(sample, 'step_context'):
            missing.append('step_context')
        if not hasattr(sample, 'retrieval_feature'):
            missing.append('retrieval_feature')
        if missing:
            print(
                "[WARN] Dataset is missing phase3 forward features: "
                + ", ".join(missing)
                + ". Regenerate the PT from the latest pkl via dataset_tool convert, "
                + "or point config_dataset.yaml to the pkl source."
            )
    
    def _convert_to_pyg(self, raw_data):
        """Convert raw pickle data to PyG Data objects with curve generation"""
        print("Converting raw samples to PyG Data objects...")
        data_list = []
        errors = 0
        
        for idx, sample in enumerate(tqdm(raw_data, desc="Converting")):
            try:
                data = sample_to_pyg_data(sample, idx)
                data_list.append(data)
                
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"Error converting sample {idx}: {e}")
        
        if errors > 0:
            print(f"Encountered {errors} errors during conversion.")
            
        return data_list
        
    def create_train_loader(self, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False):
        train_data = [self.all_data[i] for i in self.train_indices]
        dataset = GraphCurveDataset(train_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    def create_val_loader(self, batch_size=32, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False):
        val_data = [self.all_data[i] for i in self.val_indices]
        dataset = GraphCurveDataset(val_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    def create_test_loader(self, batch_size=32, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False):
        test_data = [self.all_data[i] for i in self.test_indices]
        dataset = GraphCurveDataset(test_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    def get_split_data(self, split_name):
        split_map = {
            'train': self.train_indices,
            'val': self.val_indices,
            'test': self.test_indices,
        }
        indices = split_map[split_name]
        return [self.all_data[i] for i in indices]
