import os
import pickle
import torch
import csv
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from src.kinematics_extract import extract_kinematics

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
        self.dataset_path = config.get('dataset_path', '')
        self.train_ratio = config.get('train_ratio', 0.8)
        self.val_ratio = config.get('val_ratio', 0.1)
        self.test_ratio = config.get('test_ratio', 0.1)
        self.seed = config.get('seed', 42)
        
        # Load all data
        self._load_data()
        
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
            
        print(f"Loaded {len(self.all_data)} samples")
        
        # Shuffle and split
        torch.manual_seed(self.seed)
        indices = torch.randperm(len(self.all_data)).tolist()
        
        n_total = len(self.all_data)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train + n_val]
        self.test_indices = indices[n_train + n_val:]
        
        print(f"Split: train={len(self.train_indices)}, val={len(self.val_indices)}, test={len(self.test_indices)}")
    
    def _convert_to_pyg(self, raw_data):
        """Convert raw pickle data to PyG Data objects with curve generation"""
        print("Converting raw samples to PyG Data objects...")
        data_list = []
        errors = 0
        
        for idx, sample in enumerate(tqdm(raw_data, desc="Converting")):
            try:
                # 1. Standard features
                A = sample['A']
                x0 = sample['x0']
                types = sample['types']
                analysis = sample['analysis']
                
                # Node features: [x, y, is_fixed, is_grounded]
                is_fixed = (types == 1).astype(np.float32)
                is_grounded = np.zeros_like(is_fixed)
                is_grounded[0] = 1  # Node 0 is always on ground
                
                x_features = np.column_stack([x0, is_fixed, is_grounded])
                
                # Edge index
                edges = np.array(np.where(A)).T
                edge_index = edges.T
                
                # Keypoints
                keypoints = np.array([analysis['foot'], analysis['knee'], analysis['ankle']])
                
                # 2. Kinematics Curves (Targets)
                foot_traj, knee_angle, ankle_angle = extract_kinematics(sample)
                
                # Create PyG Data
                data = Data(
                    x=torch.tensor(x_features, dtype=torch.float32),
                    pos=torch.tensor(x0, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    keypoints=torch.tensor(keypoints, dtype=torch.long),
                    sample_id=idx,
                    y_foot=torch.tensor(foot_traj, dtype=torch.float32),
                    y_knee=torch.tensor(knee_angle, dtype=torch.float32),
                    y_ankle=torch.tensor(ankle_angle, dtype=torch.float32),
                )
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
