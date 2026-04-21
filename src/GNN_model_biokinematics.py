# src/generative_curve/GNN_model_biokinematics.py

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from src.layers_encoder import GNNEncoder
from src.utils import MLP


class BioKinematicsGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        encoder_cfg = config.get("encoder", {})
        decoder_cfg = config.get("decoder", {})
        training_cfg = config.get("training", {})

        self.hidden_dim = int(encoder_cfg.get("hidden_dim", 128))
        self.node_input_dim = int(encoder_cfg.get("node_input_dim", 8))
        self.curve_steps = int(training_cfg.get("curve_steps", 200))

        self.encoder = GNNEncoder(
            dim_input_nodes=self.node_input_dim,
            dim_input_edges=1,
            n_layers=encoder_cfg.get("num_layers", 4),
            dim_hidden=self.hidden_dim,
            dropout=encoder_cfg.get("dropout", 0.1),
        )

        self.num_families = int(decoder_cfg.get("num_families", 4))
        self.family_embedding_dim = int(decoder_cfg.get("family_embedding_dim", 16))
        self.step_context_input_dim = int(decoder_cfg.get("step_context_input_dim", 3))
        self.step_context_hidden_dim = int(decoder_cfg.get("step_context_hidden_dim", 16))

        self.family_embedding = nn.Embedding(max(self.num_families, 1), max(self.family_embedding_dim, 1))
        self.step_context_encoder = nn.Sequential(
            nn.Linear(self.step_context_input_dim, self.step_context_hidden_dim),
            nn.ELU(),
            nn.Linear(self.step_context_hidden_dim, self.step_context_hidden_dim),
        )

        decoder_hidden = int(decoder_cfg.get("hidden_dim", 128))
        decoder_layers = int(decoder_cfg.get("num_layers", 4))
        decoder_input_dim = self.hidden_dim + self.family_embedding_dim + self.step_context_hidden_dim

        def make_mlp_dims(input_dim, output_dim, num_layers):
            return [input_dim] + [decoder_hidden] * max(num_layers - 1, 1) + [output_dim]

        self.decoder_foot = MLP(
            dims=make_mlp_dims(decoder_input_dim, self.curve_steps * 2, decoder_layers)
        )
        self.decoder_knee = MLP(
            dims=make_mlp_dims(decoder_input_dim, self.curve_steps, max(decoder_layers - 1, 2))
        )
        self.decoder_ankle = MLP(
            dims=make_mlp_dims(decoder_input_dim, self.curve_steps, max(decoder_layers - 1, 2))
        )

    @staticmethod
    def semantic_pool(x, mask, batch_idx):
        if mask is None or int(mask.sum().item()) == 0:
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
            return x.new_zeros((num_graphs, x.size(-1)))
        target_nodes_x = x * mask.unsqueeze(-1).float()
        return scatter(target_nodes_x, batch_idx, dim=0, reduce="add")

    def _graph_context(self, data):
        num_graphs = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1

        family_id = getattr(data, "family_id", None)
        if family_id is None:
            family_id = torch.zeros(num_graphs, dtype=torch.long, device=data.x.device)
        else:
            family_id = family_id.view(-1).long()
            if family_id.numel() < num_graphs:
                padded = torch.zeros(num_graphs, dtype=torch.long, device=data.x.device)
                padded[: family_id.numel()] = family_id
                family_id = padded
            else:
                family_id = family_id[:num_graphs]
            family_id = family_id.clamp(min=0, max=max(self.num_families - 1, 0))

        step_context = getattr(data, "step_context", None)
        if step_context is None:
            step_context = data.x.new_zeros((num_graphs, self.step_context_input_dim))
        else:
            if step_context.dim() == 1:
                step_context = step_context.view(num_graphs, -1)
            step_context = step_context[:num_graphs].float()

        family_context = self.family_embedding(family_id)
        step_role_context = self.step_context_encoder(step_context)
        return torch.cat([family_context, step_role_context], dim=-1)

    @staticmethod
    def _semantic_masks(data):
        num_nodes = data.x.size(0)

        if hasattr(data, "mask_foot") and data.mask_foot is not None:
            return data.mask_foot, getattr(data, "mask_knee", None), getattr(data, "mask_ankle", None)

        if data.x.size(-1) >= 8:
            semantic_roles = data.x[:, -4:]
            return (
                semantic_roles[:, 3] > 0.5,
                semantic_roles[:, 1] > 0.5,
                semantic_roles[:, 2] > 0.5,
            )

        if hasattr(data, "keypoints") and data.keypoints is not None:
            mask_foot = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
            mask_knee = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
            mask_ankle = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)

            batch = data.batch
            num_nodes_per_graph = torch.bincount(batch)
            num_graphs = len(num_nodes_per_graph)

            kp = data.keypoints
            if kp.dim() == 1:
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
            return mask_foot, mask_knee, mask_ankle

        mask_all = torch.ones(num_nodes, dtype=torch.bool, device=data.x.device)
        return mask_all, mask_all, mask_all

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            pos = data.pos if hasattr(data, "pos") and data.pos is not None else data.x[:, :2]
            row, col = data.edge_index
            edge_vec = pos[col] - pos[row]
            edge_attr = torch.norm(edge_vec, dim=-1, keepdim=True)

        x_encoded, _ = self.encoder(
            emb_nodes=data.x,
            emb_edges=edge_attr,
            edge_index=data.edge_index,
            graph_node_index=data.batch,
            graph_edge_index=getattr(data, "edge_batch", None),
        )

        mask_foot, mask_knee, mask_ankle = self._semantic_masks(data)
        feat_foot = self.semantic_pool(x_encoded, mask_foot, data.batch)
        feat_knee = self.semantic_pool(x_encoded, mask_knee, data.batch)
        feat_ankle = self.semantic_pool(x_encoded, mask_ankle, data.batch)
        graph_context = self._graph_context(data)

        feat_foot = torch.cat([feat_foot, graph_context], dim=-1)
        feat_knee = torch.cat([feat_knee, graph_context], dim=-1)
        feat_ankle = torch.cat([feat_ankle, graph_context], dim=-1)

        pred_foot = self.decoder_foot(feat_foot).view(-1, self.curve_steps, 2)
        pred_knee = self.decoder_knee(feat_knee).view(-1, self.curve_steps)
        pred_ankle = self.decoder_ankle(feat_ankle).view(-1, self.curve_steps)
        return pred_foot, pred_knee, pred_ankle
