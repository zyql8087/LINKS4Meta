# GNN 策略网络：基于图神经网络的机构综合逆向设计策略，支持 IL/RL/推理。
# 包含 GNN 编码器、拓扑评分、动作预测 (u/v/w)、几何码预测和 CVAE 几何生成头。

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from src.layers_encoder import GNNEncoder

# 模型加载时可选缺失的键集合（旧版模型兼容）
OPTIONAL_POLICY_MISSING_KEYS = {
    "geo_head.prior_bias_scale",
    "action_codebook",
}

OPTIONAL_POLICY_MISSING_PREFIXES = (
    "geo_head.prior_bias.",
    "family_embedding.",
    "step_index_embedding.",
    "step_count_embedding.",
    "context_mlp.",
    "stop_head.",
    "step_role_head.",
    "step_count_head.",
    "geometry_code_head.",
    "action_codebook.",
    "action_u_head.",
    "action_v_head.",
    "action_w_head.",
    # 条件融合改造新增层（旧 checkpoint 可缺省，零初始化等价于旧行为）
    "graph_feat_norm.",
    "cond_norm.",
    "film.",
)


def filter_optional_policy_missing_keys(missing_keys):
    """从缺失键列表中移除可选键，仅返回真正缺失的必需键。"""
    return [
        key for key in missing_keys
        if key not in OPTIONAL_POLICY_MISSING_KEYS
        and not any(key.startswith(prefix) for prefix in OPTIONAL_POLICY_MISSING_PREFIXES)
    ]


def policy_load_incompatibilities(load_result):
    """分析加载结果，返回 (过滤后的缺失键, 意外键)。"""
    missing = filter_optional_policy_missing_keys(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    return missing, unexpected


class GeometryHead(nn.Module):
    """CVAE 几何生成头：编码器将 (x_true, condition) 映射为潜在分布 (mu, logvar)，
    解码器将 (z, condition) 解码为几何参数，先验偏置提供基线估计。

    输出维度为 4（dyad 的 4 个几何参数），潜在空间维度默认 64。
    """

    def __init__(
        self,
        condition_dim: int = 128,
        latent_dim: int = 64,
        output_dim: int = 4,
        prior_bias_init: float = 0.10,
        prior_bias_max: float = 0.50,
    ):
        """初始化 CVAE 几何头。condition_dim 为条件向量维度，latent_dim 为潜在空间维度。"""
        super().__init__()
        self.latent_dim = latent_dim
        self.prior_bias_max = float(prior_bias_max)

        # 编码器: (x_true, condition) -> (mu, logvar)
        enc_in = output_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # 解码器: (z, condition) -> 几何参数
        dec_in = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, output_dim),
        )
        # 先验偏置网络: 从条件向量估计基线几何参数
        self.prior_bias = nn.Sequential(
            nn.Linear(condition_dim, 64), nn.ELU(),
            nn.Linear(64, output_dim), nn.Tanh(),
        )
        self.prior_bias_scale = nn.Parameter(torch.tensor(float(prior_bias_init)))

    def encode(self, x_true: torch.Tensor, condition: torch.Tensor):
        """将 (x_true, condition) 编码为潜在分布 (mu, logvar)。"""
        h = self.encoder(torch.cat([x_true, condition], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """训练时重参数化采样 z = mu + sigma * epsilon，推理时直接返回 mu。"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """解码：decoder(z, condition) + prior_bias(condition) * clamp(scale)。"""
        base = self.decoder(torch.cat([z, condition], dim=-1))
        bias_scale = torch.clamp(self.prior_bias_scale, min=0.0, max=self.prior_bias_max)
        bias = self.prior_bias(condition) * bias_scale
        return base + bias

    def forward(self, x_true, condition):
        """训练前向传播：encode -> reparameterize -> decode，返回 (x_pred, mu, logvar)。"""
        mu, logvar = self.encode(x_true, condition)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z, condition)
        return x_pred, mu, logvar

    def sample(self, condition: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """从先验 N(0,I) 采样潜在向量并解码，返回 (B, n_samples, output_dim)。"""
        batch_size = condition.size(0)
        z = torch.randn(batch_size * n_samples, self.latent_dim, device=condition.device)
        cond_rep = condition.repeat_interleave(n_samples, dim=0)
        return self.decode(z, cond_rep).view(batch_size, n_samples, -1)

    def prior_mean(self, condition: torch.Tensor) -> torch.Tensor:
        """将潜在向量设为零（先验均值），仅通过条件和偏置得到基线几何参数。"""
        z = torch.zeros(condition.size(0), self.latent_dim, device=condition.device)
        return self.decode(z, condition)


def _resolve_policy_cfg(cfg: dict):
    """解析配置字典，支持嵌套格式 {"gnn_policy":...} 和扁平格式，返回 (gnn_cfg, cvae_cfg, curve_cfg)。"""
    if "gnn_policy" in cfg:
        gnn_cfg = dict(cfg.get("gnn_policy", {}))
        cvae_cfg = dict(cfg.get("cvae", {}))
        curve_cfg = dict(cfg.get("curve_encoder", {}))
    else:
        gnn_cfg = dict(cfg)
        cvae_cfg = dict(cfg.get("cvae", {}))
        curve_cfg = dict(cfg.get("curve_encoder", {}))
    return gnn_cfg, cvae_cfg, curve_cfg


class GNNPolicy(nn.Module):
    """GNN 策略网络：整合 GNN 编码器、拓扑评分、动作预测 (u/v/w)、几何码预测和 CVAE 几何生成头。
    支持 family/step_index/step_count 嵌入和动作码本的动态设置。
    """

    def __init__(self, cfg: dict):
        """初始化策略网络。cfg 支持嵌套和扁平两种格式。"""
        super().__init__()
        gnn_cfg, cvae_cfg, curve_cfg = _resolve_policy_cfg(cfg)

        node_dim = gnn_cfg.get("node_input_dim", 4)
        edge_dim = gnn_cfg.get("edge_input_dim", 1)
        hidden_dim = gnn_cfg.get("hidden_dim", 128)
        n_layers = gnn_cfg.get("num_layers", 4)
        dropout = gnn_cfg.get("dropout", 0.1)

        self.hidden_dim = hidden_dim
        self.curve_latent_dim = curve_cfg.get("latent_dim", cfg.get("condition_latent_dim", 128))
        self.max_step_count = int(gnn_cfg.get("max_step_count", 2))
        self.num_families = int(gnn_cfg.get("num_families", 4))
        self.num_geometry_codes = int(gnn_cfg.get("num_geometry_codes", 32))
        self.action_code_dim = int(gnn_cfg.get("action_code_dim", 6))

        # GNN 编码器
        self.gnn = GNNEncoder(
            dim_input_nodes=node_dim, dim_input_edges=edge_dim,
            n_layers=n_layers, dim_hidden=hidden_dim, dropout=dropout,
        )

        # 拓扑评分头
        self.topo_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ELU(), nn.Linear(64, 1))

        # CVAE 几何头
        latent_dim = cvae_cfg.get("latent_dim", 64)
        condition_dim = hidden_dim + self.curve_latent_dim
        self.geo_head = GeometryHead(
            condition_dim=condition_dim, latent_dim=latent_dim, output_dim=4,
            prior_bias_init=cvae_cfg.get("prior_bias_init", 0.10),
            prior_bias_max=cvae_cfg.get("prior_bias_max", 0.50),
        )

        # 嵌入层
        family_embedding_dim = int(gnn_cfg.get("family_embedding_dim", 8))
        step_embedding_dim = int(gnn_cfg.get("step_embedding_dim", 8))
        context_hidden_dim = int(gnn_cfg.get("context_hidden_dim", hidden_dim))

        self.family_embedding = nn.Embedding(self.num_families + 1, family_embedding_dim)
        self.step_index_embedding = nn.Embedding(self.max_step_count + 1, step_embedding_dim)
        self.step_count_embedding = nn.Embedding(self.max_step_count + 1, step_embedding_dim)

        # 融合前对齐：对图级特征和曲线条件各自做 LayerNorm，平衡两者幅值，
        # 避免范数较大的图特征淹没曲线条件（batch-size 无关，兼容 RL batch=1）。
        self.graph_feat_norm = nn.LayerNorm(hidden_dim)
        self.cond_norm = nn.LayerNorm(self.curve_latent_dim)

        # 上下文 MLP: 融合图特征、曲线编码和各种嵌入
        context_input_dim = hidden_dim + self.curve_latent_dim + family_embedding_dim + step_embedding_dim * 2
        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, context_hidden_dim), nn.ELU(),
            nn.Linear(context_hidden_dim, context_hidden_dim), nn.ELU(),
        )

        # FiLM 节点级条件调制：由曲线条件 z_c 生成逐通道 (gamma, beta)，
        # 对每个节点的图嵌入做仿射调制，使节点级动作选择(u/v/w)依赖于目标曲线。
        # 零初始化 -> 初始 gamma=beta=0 -> 等价于不调制，便于在旧 checkpoint 上热启动。
        self.film = nn.Linear(self.curve_latent_dim, 2 * hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        # 动作预测头: 为每个节点预测被选为 u/v/w 的分数
        node_head_input_dim = hidden_dim + context_hidden_dim
        self.action_u_head = nn.Sequential(nn.Linear(node_head_input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.action_v_head = nn.Sequential(nn.Linear(node_head_input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.action_w_head = nn.Sequential(nn.Linear(node_head_input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))

        # 其他预测头
        self.stop_head = nn.Sequential(nn.Linear(context_hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.step_role_head = nn.Sequential(nn.Linear(context_hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2))
        self.step_count_head = nn.Sequential(nn.Linear(context_hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, self.max_step_count))

        # 几何码预测头: 上下文 + u/v/w 节点特征 -> 码本分布
        geometry_head_input_dim = context_hidden_dim + hidden_dim * 3
        self.geometry_code_head = nn.Sequential(
            nn.Linear(geometry_head_input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, self.num_geometry_codes),
        )

        # 动作码本缓冲区
        self.register_buffer("action_codebook", torch.zeros((self.num_geometry_codes, self.action_code_dim), dtype=torch.float32))
        self.action_codebook_buckets: dict[str, list[int]] = {}

    def encode_graph(self, data):
        """用 GNN 编码图为节点特征。若无 edge_attr 则根据节点位置计算欧氏距离作为边特征。"""
        edge_attr = data.edge_attr
        if edge_attr is None:
            pos = data.pos if hasattr(data, "pos") and data.pos is not None else data.x[:, :2]
            row, col = data.edge_index
            edge_attr = torch.norm(pos[col] - pos[row], dim=-1, keepdim=True)

        x_enc, _ = self.gnn(
            emb_nodes=data.x, emb_edges=edge_attr,
            edge_index=data.edge_index, graph_node_index=getattr(data, "batch", None),
        )
        return x_enc

    def topology_scores(self, x_enc: torch.Tensor) -> torch.Tensor:
        """计算节点拓扑重要性分数，形状 (N,)。"""
        return self.topo_head(x_enc)

    def _batch_index(self, data, x_enc: torch.Tensor) -> torch.Tensor:
        """获取节点批次索引，单图模式下全为 0。"""
        batch_index = getattr(data, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(x_enc.size(0), dtype=torch.long, device=x_enc.device)
        return batch_index

    def _prepare_condition(self, z_c: torch.Tensor | None, num_graphs: int, ref: torch.Tensor) -> torch.Tensor:
        """规范化曲线条件向量：处理 None/一维输入，并做 LayerNorm 对齐。返回 (num_graphs, curve_latent_dim)。"""
        if z_c is None:
            z_c = torch.zeros((num_graphs, self.curve_latent_dim), dtype=ref.dtype, device=ref.device)
        elif z_c.dim() == 1:
            z_c = z_c.unsqueeze(0)
        return self.cond_norm(z_c)

    def build_il_context(
        self, data, x_enc: torch.Tensor, z_c: torch.Tensor | None,
        family_ids: torch.Tensor | None = None, step_indices: torch.Tensor | None = None,
        step_counts: torch.Tensor | None = None,
    ):
        """构建 IL 上下文：聚合节点特征为图级特征，融合曲线编码和嵌入，返回 (context, batch_index)。"""
        batch_index = self._batch_index(data, x_enc)
        graph_feat = scatter(x_enc, batch_index, dim=0, reduce="mean")
        num_graphs = graph_feat.size(0)

        # 融合前对齐：图特征与曲线条件各自归一化
        graph_feat = self.graph_feat_norm(graph_feat)
        z_c = self._prepare_condition(z_c, num_graphs, x_enc)

        if family_ids is None:
            family_ids = torch.full((num_graphs,), self.num_families, dtype=torch.long, device=x_enc.device)
        if step_indices is None:
            step_indices = torch.zeros(num_graphs, dtype=torch.long, device=x_enc.device)
        if step_counts is None:
            step_counts = torch.ones(num_graphs, dtype=torch.long, device=x_enc.device)

        family_ids = family_ids.long().clamp(min=0, max=self.num_families)
        step_indices = step_indices.long().clamp(min=0, max=self.max_step_count)
        step_counts = step_counts.long().clamp(min=1, max=self.max_step_count)

        family_emb = self.family_embedding(family_ids)
        step_index_emb = self.step_index_embedding(step_indices)
        step_count_emb = self.step_count_embedding(step_counts)

        context = self.context_mlp(
            torch.cat([graph_feat, z_c, family_emb, step_index_emb, step_count_emb], dim=-1)
        )
        return context, batch_index

    def phase4_outputs(
        self, data, x_enc: torch.Tensor, z_c: torch.Tensor | None,
        family_ids: torch.Tensor | None = None, step_indices: torch.Tensor | None = None,
        step_counts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Phase 4 主接口：一次性输出 u/v/w logits、stop/step_role/step_count logits 和 graph_context。"""
        context, batch_index = self.build_il_context(
            data, x_enc, z_c, family_ids=family_ids,
            step_indices=step_indices, step_counts=step_counts,
        )
        # FiLM 节点级条件调制：用曲线条件生成 (gamma, beta)，逐节点仿射调制图嵌入，
        # 使 u/v/w 节点打分依赖于目标曲线。零初始化时等价于 x_enc 不变。
        num_graphs = context.size(0)
        z_c_norm = self._prepare_condition(z_c, num_graphs, x_enc)
        gamma, beta = self.film(z_c_norm).chunk(2, dim=-1)
        x_mod = x_enc * (1.0 + gamma[batch_index]) + beta[batch_index]

        node_context = context[batch_index]
        node_inputs = torch.cat([x_mod, node_context], dim=-1)

        return {
            "graph_context": context,
            "u_logits": self.action_u_head(node_inputs).squeeze(-1),
            "v_logits": self.action_v_head(node_inputs).squeeze(-1),
            "w_logits": self.action_w_head(node_inputs).squeeze(-1),
            "stop_logits": self.stop_head(context).squeeze(-1),
            "step_role_logits": self.step_role_head(context),
            "step_count_logits": self.step_count_head(context),
        }

    def resize_geometry_code_head(self, num_geometry_codes: int):
        """动态调整几何码预测头输出维度和码本缓冲区大小。"""
        num_geometry_codes = max(1, int(num_geometry_codes))
        if num_geometry_codes == self.num_geometry_codes:
            return

        hidden_dim = self.hidden_dim
        input_dim = self.geometry_code_head[0].in_features
        self.geometry_code_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, num_geometry_codes),
        ).to(self.action_codebook.device)
        self.num_geometry_codes = num_geometry_codes
        self._buffers["action_codebook"] = torch.zeros(
            (num_geometry_codes, self.action_code_dim),
            dtype=torch.float32, device=self.action_codebook.device,
        )

    def set_action_codebook(self, codebook: torch.Tensor, buckets: dict[str, list[int]] | None = None):
        """设置动作码本及桶映射。自动调整几何码头输出维度以匹配码本大小。"""
        codebook = codebook.detach().float()
        if codebook.dim() != 2:
            raise ValueError(f"expected 2D codebook tensor, got {tuple(codebook.shape)}")
        self.action_code_dim = int(codebook.size(1))
        self.resize_geometry_code_head(int(codebook.size(0)))
        self._buffers["action_codebook"] = codebook.to(self.action_codebook.device)
        self.action_codebook_buckets = {str(key): [int(idx) for idx in value] for key, value in (buckets or {}).items()}

    def geometry_code_logits(
        self, data, x_enc: torch.Tensor, graph_context: torch.Tensor, action_topo: torch.Tensor,
    ) -> torch.Tensor:
        """根据选定的 u/v/w 节点特征和图上下文，输出几何码 logits (B, num_geometry_codes)。"""
        batch_index = self._batch_index(data, x_enc)
        offsets = data.ptr[:-1].to(x_enc.device) if hasattr(data, "ptr") and data.ptr is not None else torch.tensor([0], dtype=torch.long, device=x_enc.device)
        global_action = action_topo.long().to(x_enc.device) + offsets.unsqueeze(1)
        u_feat = x_enc[global_action[:, 0]]
        v_feat = x_enc[global_action[:, 1]]
        w_feat = x_enc[global_action[:, 2]]
        head_in = torch.cat([graph_context, u_feat, v_feat, w_feat], dim=-1)
        return self.geometry_code_head(head_in)

    def predict_geometry_code(
        self, data, x_enc: torch.Tensor, graph_context: torch.Tensor, action_topo: torch.Tensor,
        *, family_ids: torch.Tensor | None = None, step_roles: torch.Tensor | None = None,
        step_indices: torch.Tensor | None = None, bucket_map: dict[str, list[int]] | None = None,
    ) -> torch.Tensor:
        """带桶约束的几何码预测：根据 family/step_role 确定允许的码本桶，在桶内选最高分码。"""
        from src.inverse.action_codebook import family_name_from_index, resolve_codebook_bucket_for_step

        logits = self.geometry_code_logits(data, x_enc, graph_context, action_topo)
        if family_ids is None or step_roles is None:
            return torch.argmax(logits, dim=-1)

        allowed_map = bucket_map if bucket_map is not None else self.action_codebook_buckets
        masked = logits.clone()
        for row_idx in range(masked.size(0)):
            family_name = family_name_from_index(int(family_ids[row_idx].item()))
            step_role = "semantic" if int(step_roles[row_idx].item()) == 1 else "aux"
            bucket = resolve_codebook_bucket_for_step(
                allowed_map, family_name, step_role,
                step_index=int(step_indices[row_idx].item()) if step_indices is not None else None,
                action_topo=action_topo[row_idx],
            )
            allowed_ids = allowed_map.get(bucket, [])
            if allowed_ids:
                invalid = torch.ones(masked.size(1), dtype=torch.bool, device=masked.device)
                invalid[torch.tensor(allowed_ids, dtype=torch.long, device=masked.device)] = False
                masked[row_idx] = masked[row_idx].masked_fill(invalid, -1e9)
        return torch.argmax(masked, dim=-1)

    def geometry_condition(
        self, x_enc: torch.Tensor, u_idx: int, v_idx: int, w_idx: int, z_c: torch.Tensor,
    ) -> torch.Tensor:
        """将 u/v/w 三节点特征取均值后与曲线条件 z_c 拼接，作为 CVAE 几何头的输入条件。"""
        feat_uvw = (x_enc[u_idx] + x_enc[v_idx] + x_enc[w_idx]) / 3.0
        feat_uvw = feat_uvw.unsqueeze(0)
        z_c_flat = z_c.view(1, -1)
        return torch.cat([feat_uvw, z_c_flat], dim=-1)

    def forward(self, data, z_c: torch.Tensor, true_coords=None, u_idx=None, v_idx=None, w_idx=None):
        """前向传播：GNN 编码 -> 拓扑评分 -> (可选) CVAE 生成几何参数。返回 (topo_scores, geo_out)。"""
        x_enc = self.encode_graph(data)
        topo_scores = self.topology_scores(x_enc).squeeze(-1)

        geo_out = None
        if u_idx is not None and v_idx is not None and w_idx is not None:
            cond = self.geometry_condition(x_enc, u_idx, v_idx, w_idx, z_c)
            if true_coords is not None:
                geo_out = self.geo_head(true_coords, cond)
            else:
                geo_out = self.geo_head.sample(cond, n_samples=1).squeeze(1)

        return topo_scores, geo_out
