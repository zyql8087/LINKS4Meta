# src/layers_encoder.py
# GNN 编码器层模块，定义多种卷积实现（MPNN、EGNN、TIGNN、CrystalGNN 等）。
# 所有层遵循统一接口：forward(X, E, emb_nodes, emb_edges, edge_index, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, MessagePassing
import torch_geometric.nn.norm as norm
from torch_geometric.data import Data
from src.utils import MLP


def scatter_add(src, index, dim=0, dim_size=None):
    """按 index 将 src 累加求和。"""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='add')


class GNNEncoder(nn.Module):
    """通用 GNN 编码器，由多层卷积堆叠而成，支持多种卷积/归一化类型和跳跃连接。"""

    def __init__(
            self, dim_input_nodes, dim_input_edges, n_layers=5, dim_hidden=64,
            conv_type='MPNN', norm_type='BatchNorm', act='ELU', skip=False, dropout=0.1
    ):
        """
        conv_type: 可选 'MPNN','EGNN','IGNN','TIGNN','GCN','GCN2','GAT','crystal'。
        norm_type: 可选 'BatchNorm','InstanceNorm','LayerNorm','GraphNorm','GraphSizeNorm','PairNorm','MeanSubtractionNorm'。
        """
        super().__init__()
        self.n_layers = n_layers
        self.dim_input_nodes = dim_input_nodes
        self.dim_input_edges = dim_input_edges
        self.dim_hidden = dim_hidden
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.skip = skip

        self.pre_mlp_nodes = nn.Linear(dim_input_nodes, dim_hidden)
        self.pre_mlp_edges = nn.Linear(dim_input_edges, dim_hidden)
        self.act = getattr(nn, act)()

        conv_li = []
        norm_li_nodes = []
        norm_li_edges = []
        dropout_li_nodes = []
        dropout_li_edges = []
        self.dim_hidden = dim_hidden

        for _ in range(self.n_layers):
            conv_li.append(STR2CONV[conv_type](dim_hidden, dim_hidden))

            if norm_type in ['GraphSizeNorm', 'PairNorm', 'MeanSubtractionNorm']:
                norm_li_nodes.append(getattr(norm, norm_type)())
                norm_li_edges.append(getattr(norm, norm_type)())
            elif norm_type in ['BatchNorm', 'InstanceNorm', 'LayerNorm', 'GraphNorm']:
                norm_li_nodes.append(getattr(norm, norm_type)(dim_hidden))
                norm_li_edges.append(getattr(norm, norm_type)(dim_hidden))
            else:
                assert False, f'unrecognized norm {norm_type}'

            dropout_li_nodes.append(nn.Dropout(p=dropout))
            dropout_li_edges.append(nn.Dropout(p=dropout))

        self.conv_li = nn.ModuleList(conv_li)
        self.norm_li_nodes = nn.ModuleList(norm_li_nodes)
        self.norm_li_edges = nn.ModuleList(norm_li_edges)
        self.dropout_li_nodes = nn.ModuleList(dropout_li_nodes)
        self.dropout_li_edges = nn.ModuleList(dropout_li_edges)

        # 全局跳跃连接的可学习权重
        if self.skip:
            self.w = nn.Parameter(torch.ones((2, self.n_layers + 1), dtype=torch.float))

    def add_adapters(self):
        """预留适配器接口（当前为空实现）。"""
        pass

    def forward(self, emb_nodes, emb_edges, edge_index, graph_node_index=None, graph_edge_index=None):
        """返回 (V, E): 编码后的节点特征和边特征。"""
        V = self.pre_mlp_nodes(emb_nodes)
        E = None if emb_edges is None else self.pre_mlp_edges(emb_edges)

        for conv, norm_nodes, norm_edges, dropout_nodes, dropout_edges in \
                zip(
                    self.conv_li, self.norm_li_nodes, self.norm_li_edges,
                    self.dropout_li_nodes, self.dropout_li_edges
                ):
            dV, dE, emb_nodes, emb_edges = \
                conv(
                    V, E, emb_nodes, emb_edges, edge_index,
                    graph_node_index=graph_node_index,
                    graph_edge_index=graph_edge_index
                )

            if self.skip:
                V = V + dV
                E = None if E is None else E + dE
            else:
                V, E = dV, dE

            # BatchNorm 不需要 batch 索引，其他归一化方式需要
            if self.norm_type == 'BatchNorm':
                V = dropout_nodes(norm_nodes(self.act(V)))
                E = None if E is None else \
                    dropout_edges(norm_edges(self.act(E)))
            else:
                V = dropout_nodes(norm_nodes(self.act(V), batch=graph_node_index))
                E = None if E is None else \
                    dropout_edges(norm_edges(self.act(E), batch=graph_edge_index))

        return V, E


class TIGNN(torch.nn.Module):
    """基于 Transformer 的不变图神经网络，利用相对位移和边特征学习消息。"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # 消息函数：拼接两端节点特征 + 相对位移(3维) + 边特征(5维)
        self.phi_e = nn.Sequential(
            nn.Linear(2 * dim_in + 3 + 5, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_in),
            nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * dim_in, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        """返回 (X, E, emb_nodes, emb_edges)，边特征和坐标嵌入保持不变。"""
        mij = \
            self.phi_e(torch.cat((
                X[edge_index[0]], X[edge_index[1]],
                emb_nodes[edge_index[1]] - emb_nodes[edge_index[0]],
                emb_edges
            ), dim=-1))

        mi = scatter_add(mij, edge_index[1], dim=0, dim_size=X.shape[0])
        X = self.phi_h(torch.cat((X, mi), dim=-1))

        return X, E, emb_nodes, emb_edges


class IGNN(torch.nn.Module):
    """不变图神经网络，内部使用 EGNN 但禁用等变更新，仅更新节点特征。"""

    def __init__(self, dim_in, dim_out, equivariant=True):
        super().__init__()
        self.gnn = EGNN(dim_in, dim_out, equivariant=False)

    def forward(self, *args, **kwargs):
        return self.gnn(*args, **kwargs)


class EGNN(torch.nn.Module):
    """等变图神经网络，消息传递中同时更新节点特征和坐标，保证等变性。"""

    def __init__(self, dim_in, dim_out, equivariant=True):
        """
        equivariant: False 时仅更新特征，不更新坐标。
        """
        super().__init__()
        self.equivariant = equivariant

        # 消息函数：拼接两端节点特征 + 距离(1维) + 边特征(2维)
        self.phi_e = nn.Sequential(
            nn.Linear(2 * dim_in + 1 + 2, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_in),
            nn.SiLU(),
        )

        if equivariant:
            self.phi_x = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.SiLU(),
                nn.Linear(dim_in, 3),
            )

        self.phi_h = nn.Sequential(
            nn.Linear(2 * dim_in, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, graph_node_index=None, **kwargs):
        """返回 (X, E, emb_nodes, emb_edges)，含更新后的节点特征、消息作为边特征、更新后坐标。"""
        # 计算每个节点所在图的节点数 M，用于坐标更新归一化
        if graph_node_index is None:
            M = X.shape[0]
        else:
            ones = torch.ones_like(graph_node_index)
            M = torch.zeros(ones.size(0), dtype=ones.dtype, device=X.device)
            M = M.scatter_add(0, graph_node_index, ones)[graph_node_index].unsqueeze(-1)

        mij = \
            self.phi_e(torch.cat((
                X[edge_index[0]], X[edge_index[1]],
                torch.norm(emb_nodes[edge_index[1]] - emb_nodes[edge_index[0]], dim=-1).unsqueeze(-1),
                emb_edges
            ), dim=-1))

        # 等变坐标更新
        if self.equivariant:
            emb_nodes = \
                emb_nodes + 1 / (M - 1) * scatter_add(
                    (emb_nodes[edge_index[1]] - emb_nodes[edge_index[0]]) * self.phi_x(mij),
                    edge_index[1], dim=0, dim_size=emb_nodes.shape[0])

        mi = scatter_add(mij, edge_index[1], dim=0, dim_size=X.shape[0])
        X = self.phi_h(torch.cat((X, mi), dim=-1))
        E = mij

        return X, E, emb_nodes, emb_edges


class CrystalGNN(torch.nn.Module):
    """晶体图神经网络，使用注意力门控的消息传递，适用于材料属性预测。"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_in *= 3  # 拼接源节点、目标节点和边特征
        self.mlp_att = MLP([dim_in, max(dim_in // 4, 16), max(dim_in // 8, 4), 1])
        self.mlp_msg = MLP([dim_in, (dim_in + dim_out) // 2, dim_out])

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        """返回 (X, E, None, None)，边特征不变。"""
        H = torch.cat((X[edge_index[0]], X[edge_index[1]], E), dim=-1)
        # 注意力门控消息
        M = torch.sigmoid(self.mlp_att(H)) * self.mlp_msg(H)
        X = X + scatter_add(src=M, index=edge_index[1], dim=0, dim_size=X.shape[0])

        return X, E, None, None


class GraphNetBlock(MessagePassing):
    """图网络消息传递块，实现边更新 + 节点更新 + 残差连接。"""

    def __init__(self, hidden_size, dim_edge_in, dim_node_in):
        super().__init__(aggr='add')
        self.edge_net = MLP([dim_edge_in, hidden_size],
                            dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=True)
        self.node_net = MLP([dim_node_in, hidden_size],
                            dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=True)

    def forward(self, X, E, edge_index):
        """返回 (X_new, E_new, edge_index)。"""
        X_new = self.propagate(edge_index, x=X, edge_attr=E)

        row, col = edge_index
        E_new = self.edge_net(torch.cat([X[row], X[col], E], dim=-1))

        # 残差连接
        X_new = X_new + X
        E_new = E_new + E

        return X_new, E_new, edge_index

    def message(self, x_i, x_j, edge_attr):
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_net(features)

    def update(self, aggr_out, x):
        tmp = torch.cat([aggr_out, x], dim=-1)
        return self.node_net(tmp)


class MPNN(torch.nn.Module):
    """消息传递神经网络，使用 GraphNetBlock 实现单层消息传递。"""

    def __init__(self, latent_size, name='EncodeProcessDecode'):
        super().__init__()
        self.message_pass = GraphNetBlock(latent_size, latent_size * 3, latent_size * 2)

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        """返回 (X, E, None, None)。"""
        X, E, _ = self.message_pass(X, E, edge_index)
        return X, E, None, None


class GNNWrapper(nn.Module):
    """将 PyG 标准卷积（GCNConv、GATConv 等）封装为统一接口。"""

    def __init__(self, *args, conv=None, **kwargs):
        super().__init__()
        assert conv is not None
        dim_in, dim_out, *_ = args
        self.conv = conv(*args)
        self.e_mlp = nn.Linear(dim_in, dim_out)

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        """返回 (X_new, E, X, None)。"""
        # GCN2Conv 需要初始节点特征 x_0
        if type(self.conv) == GCN2Conv:
            if emb_nodes.shape[-1] == X.shape[-1]:
                x_0 = emb_nodes
            else:
                x_0 = X
            X_new = self.conv(X, x_0, edge_index)
        else:
            X_new = self.conv(X, edge_index)

        if E is not None:
            E = self.e_mlp(E)

        return X_new, E, X, None


def get_gnn_wrapper(conv):
    """工厂函数：为 PyG 卷积类创建 GNNWrapper 构造函数。"""
    def f(*args, **kwargs):
        return GNNWrapper(*args, conv=conv, **kwargs)
    return f


# 卷积类型名称到实现的映射
STR2CONV = {
    'TIGNN': TIGNN,
    'IGNN': IGNN,
    'EGNN': EGNN,
    'GCN': get_gnn_wrapper(GCNConv),
    'GCN2': get_gnn_wrapper(GCN2Conv),
    'GAT': get_gnn_wrapper(GATConv),
    'crystal': CrystalGNN,
    'MPNN': MPNN,
}
