# src/layers_pooler.py
# 图池化与合并层模块，将变长节点/边嵌入聚合为固定大小的图级表示。

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax, scatter_max
from src.utils import MLP


class Pooler(nn.Module):
    """图池化层，分别对节点和边嵌入按图索引聚合，支持 SUM/ATT/MAX 策略。"""

    def __init__(self, dim, pool_name):
        super().__init__()
        self.pool_nodes = STR2POOL[pool_name](dim)
        self.pool_edges = STR2POOL[pool_name](dim)

    def forward(self, emb_nodes, emb_edges, graph):
        """返回 (emb_nodes_pooled, emb_edges_pooled)，形状均为 (num_graphs, dim)。"""
        emb_nodes_pooled = \
            self.pool_nodes(
                src=emb_nodes, index=graph.graph_node_index,
                dim=0, dim_size=len(graph.g_li)
            )
        emb_edges_pooled = \
            self.pool_edges(
                src=emb_edges, index=graph.graph_edge_index,
                dim=0, dim_size=len(graph.g_li)
            )

        return emb_nodes_pooled, emb_edges_pooled


class Merger(nn.Module):
    """嵌入合并层，将多个嵌入合并为单一向量并做批归一化。"""

    def __init__(self, dim_in, dim_out, merge_name, num_inputs=2):
        super().__init__()
        self.merger = STR2MERGER[merge_name](dim_in, dim_out, num_inputs)
        self.bn = nn.BatchNorm1d(dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, *emb_li):
        """返回合并后的向量，形状 (batch_size, dim_out)。"""
        emb_all = self.merger(*emb_li)
        emb_all = self.bn(emb_all)
        return emb_all


class CatMerger(nn.Module):
    """拼接合并器，将多个嵌入在特征维度拼接后线性映射到目标维度。"""

    def __init__(self, dim_in, dim_hidden, num_inputs=2):
        super().__init__()
        self.mlp = nn.Linear(num_inputs * dim_in, dim_hidden)

    def forward(self, *emb_li):
        return self.mlp(torch.cat(emb_li, dim=-1))


class GatedAttentionPooling(nn.Module):
    """门控注意力池化，通过 MLP 注意力分数对嵌入加权聚合。"""

    def __init__(self, dim_in):
        super().__init__()
        self.mlp_att = MLP([dim_in, dim_in//2, dim_in//4, 1])

    def forward(self, src, index, dim, dim_size):
        """返回注意力加权池化后的图级表示，形状 (dim_size, dim_in)。"""
        att = scatter_softmax(self.mlp_att(src).view(-1), index=index).unsqueeze(-1)
        Z = scatter_add(src=att*src, index=index, dim=dim, dim_size=dim_size)
        return Z


# 池化策略映射：SUM 求和、ATT 注意力、MAX 最大值
STR2POOL = {
    'SUM': (lambda x: scatter_add),
    'ATT': GatedAttentionPooling,
    'MAX': (lambda x: scatter_max),
}

# 合并策略映射：CAT 拼接合并
STR2MERGER = {
    'CAT': CatMerger,
}
