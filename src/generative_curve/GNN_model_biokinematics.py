# src/generative_curve/model_biokinematics.py

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

# 引用现有组件
from src.layers_encoder import GNNEncoder
from src.utils import MLP 

class BioKinematicsGNN(nn.Module):
    def __init__(self, config):
        super(BioKinematicsGNN, self).__init__()
        self.config = config
        
        # --- 1. 共享图编码器 ---
        # 必须处理 edge_attr (杆长)
        encoder_cfg = config.get('encoder', {})
        self.hidden_dim = encoder_cfg.get('hidden_dim', 128)
        
        # GNNEncoder expects: dim_input_nodes, dim_input_edges, n_layers, dim_hidden, ...
        # Data has 4D node features and no edge attributes (we'll compute edge lengths)
        self.encoder = GNNEncoder(
            dim_input_nodes=4,  # actual data has 4 features per node
            dim_input_edges=1,  # edge length (computed from positions)
            n_layers=encoder_cfg.get('num_layers', 4),
            dim_hidden=self.hidden_dim,
            dropout=encoder_cfg.get('dropout', 0.1)
        )
        
        # --- 2. 多头解码器 ---
        training_cfg = config.get('training', {})
        self.curve_steps = training_cfg.get('curve_steps', 100)
        
        # Helper to create MLP dims list
        def make_mlp_dims(input_dim, hidden_dim, output_dim, num_layers):
            return [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        decoder_cfg = config.get('decoder', {})
        decoder_hidden = decoder_cfg.get('hidden_dim', 128)
        decoder_layers = decoder_cfg.get('num_layers', 4)

        # Head A: 足端轨迹 (输出 X, Y)
        self.decoder_foot = MLP(
            dims=make_mlp_dims(
                input_dim=self.hidden_dim, 
                hidden_dim=decoder_hidden, 
                output_dim=self.curve_steps * 2, 
                num_layers=decoder_layers
            )
        )
        
        # Head B: 膝关节角度
        self.decoder_knee = MLP(
            dims=make_mlp_dims(
                input_dim=self.hidden_dim, 
                hidden_dim=decoder_hidden, 
                output_dim=self.curve_steps,
                num_layers=decoder_layers - 1
            )
        )
        
        # Head C: 踝关节角度
        self.decoder_ankle = MLP(
            dims=make_mlp_dims(
                input_dim=self.hidden_dim, 
                hidden_dim=decoder_hidden, 
                output_dim=self.curve_steps,
                num_layers=decoder_layers - 1
            )
        )

    def semantic_pool(self, x, mask, batch_idx):
        """利用 Mask 提取特定节点的特征"""
        if mask.sum() == 0:
            num_graphs = batch_idx.max().item() + 1
            return x.new_zeros(num_graphs, x.size(-1))
            
        target_nodes_x = x * mask.unsqueeze(-1).float()
        return scatter(target_nodes_x, batch_idx, dim=0, reduce='add')

    def forward(self, data):
        # Compute edge attributes if not present
        edge_attr = data.edge_attr
        if edge_attr is None:
            # Compute edge lengths from node positions
            pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :2]
            row, col = data.edge_index
            edge_vec = pos[col] - pos[row]
            edge_attr = torch.norm(edge_vec, dim=-1, keepdim=True)
        
        # 1. 编码
        x_encoded, _ = self.encoder(
            emb_nodes=data.x, 
            emb_edges=edge_attr, 
            edge_index=data.edge_index,
            graph_node_index=data.batch,
            graph_edge_index=getattr(data, 'edge_batch', None)
        )
        
        # 2. Create semantic masks from keypoints
        # keypoints format: [foot_idx, knee_idx, ankle_idx] per sample
        num_nodes = data.x.size(0)
        
        # Check if we have pre-computed masks
        if hasattr(data, 'mask_foot') and data.mask_foot is not None:
            mask_foot = data.mask_foot
            mask_knee = data.mask_knee
            mask_ankle = data.mask_ankle
        elif hasattr(data, 'keypoints') and data.keypoints is not None:
            # Create masks from keypoints
            # keypoints is batched: shape (batch_size, 3) or flattened
            mask_foot = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
            mask_knee = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
            mask_ankle = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
            
            # Get batch info
            batch = data.batch
            
            # --- Vectorized Mask Creation (Zero CPU-GPU loop syncs) ---
            num_nodes_per_graph = torch.bincount(batch)
            num_graphs = len(num_nodes_per_graph)
            
            # keypoints shape handling
            kp = data.keypoints
            if kp.dim() == 1:
                # Single sample or needs reshaping
                if kp.numel() == 3 * num_graphs:
                    kp = kp.view(-1, 3)
                else:
                    kp = kp.unsqueeze(0)
            
            offsets = torch.zeros(num_graphs, dtype=torch.long, device=data.x.device)
            if num_graphs > 1:
                offsets[1:] = torch.cumsum(num_nodes_per_graph[:-1], dim=0)
                
            num_samples = min(kp.size(0), num_graphs)
            kp = kp[:num_samples]
            offsets = offsets[:num_samples]
            
            global_kp = kp + offsets.unsqueeze(1)
            valid_mask = global_kp < num_nodes
            
            mask_foot[global_kp[:, 0][valid_mask[:, 0]]] = True
            mask_knee[global_kp[:, 1][valid_mask[:, 1]]] = True
            mask_ankle[global_kp[:, 2][valid_mask[:, 2]]] = True
        else:
            # Fallback: use global pooling (all nodes)
            mask_foot = torch.ones(num_nodes, dtype=torch.bool, device=data.x.device)
            mask_knee = mask_foot
            mask_ankle = mask_foot
        
        # 3. 语义池化
        feat_foot = self.semantic_pool(x_encoded, mask_foot, data.batch)
        feat_knee = self.semantic_pool(x_encoded, mask_knee, data.batch)
        feat_ankle = self.semantic_pool(x_encoded, mask_ankle, data.batch)
        
        # 4. 解码
        pred_foot = self.decoder_foot(feat_foot).view(-1, self.curve_steps, 2)
        pred_knee = self.decoder_knee(feat_knee).view(-1, self.curve_steps)
        pred_ankle = self.decoder_ankle(feat_ankle).view(-1, self.curve_steps)
        
        return pred_foot, pred_knee, pred_ankle
