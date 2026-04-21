# src/utils.py
# Minimal utilities for BioKinematics GNN

import torch
import torch.nn as nn
import torch_geometric.nn.norm as norm

class MLP(nn.Module):
    def __init__(self, dims, dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=False):
        super(MLP, self).__init__()
        self.gnn_batch = gnn_batch
        self.linear_li = nn.ModuleList([
            nn.Linear(dim1, dim2)
            for (dim1, dim2) in zip(dims, dims[1:])
        ])
        self.norm_li = nn.ModuleList([
            (norm.BatchNorm(dim2) if self.gnn_batch else nn.BatchNorm1d(dim2))
            if use_norm else None for (dim1, dim2) in zip(dims, dims[1:])
        ])
        self.use_skip = use_skip
        self.use_norm = use_norm
        self.dropout = None if dropout is None else nn.Dropout(p=dropout)
        self.act = getattr(nn, act)()

    def forward(self, X, gnn_batch=None):
        for i, (linear, norm_layer) in enumerate(zip(self.linear_li, self.norm_li)):
            X_out = linear(X)
            if self.use_skip and X_out.shape[-1] == X.shape[-1]:
                X = X_out + X
            else:
                X = X_out
            if i != len(self.linear_li) - 1:
                X = self.act(X)
                if self.use_norm and norm_layer is not None:
                    X_shape = X.shape
                    if self.gnn_batch:
                        assert gnn_batch is not None
                        X = norm_layer(X.view(-1, X.shape[-1]), batch=gnn_batch).view(*X_shape)
                    else:
                        X = norm_layer(X.view(-1, X.shape[-1])).view(*X_shape)
                if self.dropout is not None:
                    X = self.dropout(X)
        return X
