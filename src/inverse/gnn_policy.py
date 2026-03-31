import torch
import torch.nn as nn

from src.layers_encoder import GNNEncoder

OPTIONAL_POLICY_MISSING_KEYS = {
    'geo_head.prior_bias_scale',
}
OPTIONAL_POLICY_MISSING_PREFIXES = (
    'geo_head.prior_bias.',
)


def filter_optional_policy_missing_keys(missing_keys):
    return [
        key for key in missing_keys
        if key not in OPTIONAL_POLICY_MISSING_KEYS
        and not any(key.startswith(prefix) for prefix in OPTIONAL_POLICY_MISSING_PREFIXES)
    ]


def policy_load_incompatibilities(load_result):
    missing = filter_optional_policy_missing_keys(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    return missing, unexpected


class GeometryHead(nn.Module):
    """Conditional VAE head for dyad geometry generation."""

    def __init__(
        self,
        condition_dim: int = 128,
        latent_dim: int = 64,
        output_dim: int = 4,
        prior_bias_init: float = 0.10,
        prior_bias_max: float = 0.50,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior_bias_max = float(prior_bias_max)

        enc_in = output_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        dec_in = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, output_dim),
        )
        self.prior_bias = nn.Sequential(
            nn.Linear(condition_dim, 64), nn.ELU(),
            nn.Linear(64, output_dim),
            nn.Tanh(),
        )
        self.prior_bias_scale = nn.Parameter(torch.tensor(float(prior_bias_init)))

    def encode(self, x_true: torch.Tensor, condition: torch.Tensor):
        h = self.encoder(torch.cat([x_true, condition], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        base = self.decoder(torch.cat([z, condition], dim=-1))
        bias_scale = torch.clamp(self.prior_bias_scale, min=0.0, max=self.prior_bias_max)
        bias = self.prior_bias(condition) * bias_scale
        return base + bias

    def forward(self, x_true, condition):
        mu, logvar = self.encode(x_true, condition)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z, condition)
        return x_pred, mu, logvar

    def sample(self, condition: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        batch_size = condition.size(0)
        z = torch.randn(batch_size * n_samples, self.latent_dim, device=condition.device)
        cond_rep = condition.repeat_interleave(n_samples, dim=0)
        return self.decode(z, cond_rep).view(batch_size, n_samples, -1)

    def prior_mean(self, condition: torch.Tensor) -> torch.Tensor:
        """Deterministic prior prediction used for stable validation."""
        z = torch.zeros(condition.size(0), self.latent_dim, device=condition.device)
        return self.decode(z, condition)


def _resolve_policy_cfg(cfg: dict):
    if 'gnn_policy' in cfg:
        gnn_cfg = dict(cfg.get('gnn_policy', {}))
        cvae_cfg = dict(cfg.get('cvae', {}))
        curve_cfg = dict(cfg.get('curve_encoder', {}))
    else:
        gnn_cfg = dict(cfg)
        cvae_cfg = dict(cfg.get('cvae', {}))
        curve_cfg = dict(cfg.get('curve_encoder', {}))
    return gnn_cfg, cvae_cfg, curve_cfg


class GNNPolicy(nn.Module):
    """Graph policy used by IL, RL and inference."""

    def __init__(self, cfg: dict):
        super().__init__()
        gnn_cfg, cvae_cfg, curve_cfg = _resolve_policy_cfg(cfg)

        node_dim = gnn_cfg.get('node_input_dim', 4)
        edge_dim = gnn_cfg.get('edge_input_dim', 1)
        hidden_dim = gnn_cfg.get('hidden_dim', 128)
        n_layers = gnn_cfg.get('num_layers', 4)
        dropout = gnn_cfg.get('dropout', 0.1)

        self.hidden_dim = hidden_dim
        self.curve_latent_dim = curve_cfg.get('latent_dim', cfg.get('condition_latent_dim', 128))

        self.gnn = GNNEncoder(
            dim_input_nodes=node_dim,
            dim_input_edges=edge_dim,
            n_layers=n_layers,
            dim_hidden=hidden_dim,
            dropout=dropout,
        )

        self.topo_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

        latent_dim = cvae_cfg.get('latent_dim', 64)
        condition_dim = hidden_dim + self.curve_latent_dim
        self.geo_head = GeometryHead(
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            output_dim=4,
            prior_bias_init=cvae_cfg.get('prior_bias_init', 0.10),
            prior_bias_max=cvae_cfg.get('prior_bias_max', 0.50),
        )

    def encode_graph(self, data):
        edge_attr = data.edge_attr
        if edge_attr is None:
            pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :2]
            row, col = data.edge_index
            edge_attr = torch.norm(pos[col] - pos[row], dim=-1, keepdim=True)

        x_enc, _ = self.gnn(
            emb_nodes=data.x,
            emb_edges=edge_attr,
            edge_index=data.edge_index,
            graph_node_index=getattr(data, 'batch', None),
        )
        return x_enc

    def topology_scores(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.topo_head(x_enc)

    def geometry_condition(self, x_enc: torch.Tensor, u_idx: int, v_idx: int, w_idx: int,
                           z_c: torch.Tensor) -> torch.Tensor:
        feat_uvw = (x_enc[u_idx] + x_enc[v_idx] + x_enc[w_idx]) / 3.0
        feat_uvw = feat_uvw.unsqueeze(0)
        z_c_flat = z_c.view(1, -1)
        return torch.cat([feat_uvw, z_c_flat], dim=-1)

    def forward(self, data, z_c: torch.Tensor, true_coords=None,
                u_idx=None, v_idx=None, w_idx=None):
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
