from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, Data
from typing import Dict, List, Optional, Tuple


class GraphCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        from src.layers_encoder import GNNEncoder

        self.gnn = GNNEncoder(
            dim_input_nodes=4,
            dim_input_edges=1,
            n_layers=3,
            dim_hidden=hidden_dim,
            dropout=0.1,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, data: Data, z_c: torch.Tensor) -> torch.Tensor:
        edge_attr = data.edge_attr
        if edge_attr is None:
            pos = data.pos if data.pos is not None else data.x[:, :2]
            row, col = data.edge_index
            edge_attr = torch.norm(pos[col] - pos[row], dim=-1, keepdim=True)

        x_enc, _ = self.gnn(
            emb_nodes=data.x,
            emb_edges=edge_attr,
            edge_index=data.edge_index,
            graph_node_index=data.batch,
        )
        from torch_geometric.utils import scatter

        graph_feat = scatter(x_enc, data.batch, dim=0, reduce='mean')
        inp = torch.cat([graph_feat, z_c], dim=-1)
        return self.value_head(inp).squeeze(-1)


class PPOBuffer:
    def __init__(self):
        self.states: List[Data] = []
        self.z_cs: List[torch.Tensor] = []
        self.actions: List[Dict] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(self, state, z_c, action, reward, log_prob, value, done):
        self.states.append(state)
        self.z_cs.append(z_c)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma: float = 0.99, last_value: float = 0.0):
        rewards = self.rewards + [last_value]
        returns = []
        running = last_value
        for t in reversed(range(len(self.rewards))):
            running = rewards[t] + gamma * running * (1 - self.dones[t])
            returns.insert(0, running)
        values = torch.tensor(self.values, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages_t = returns_t - values
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        return returns_t, advantages_t


class PPOAgent:
    def __init__(self, policy, curve_encoder, cfg: dict, device):
        self.policy = policy
        self.curve_encoder = curve_encoder
        self.device = device
        rl_cfg = cfg.get('rl_training', {})

        self.gamma = rl_cfg.get('gamma', 0.99)
        self.clip_eps = rl_cfg.get('clip_eps', 0.2)
        self.entropy_c = rl_cfg.get('entropy_coef', 0.01)
        self.lr = rl_cfg.get('learning_rate', 1e-4)
        self.geometry_samples_per_topology = int(rl_cfg.get('geometry_samples_per_topology', 4))
        self.geometry_prior_clearance_scale = float(rl_cfg.get('geometry_prior_clearance_scale', 1.0))
        self.constraint_cfg = cfg.get('constraints', {})

        hidden_dim = cfg.get('gnn_policy', {}).get('hidden_dim', 128)
        latent_dim = cfg.get('curve_encoder', {}).get('latent_dim', 128)

        self.critic = GraphCritic(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        
        # Unfreeze geo_head.decoder so RL can optimize geometry coordinates.
        # Keep GNN and geo_head.encoder frozen to prevent feature drift.
        # The geometry prior bias is trainable so RL can refine the learned legal-geometry bias.
        self.optimizer = optim.Adam([
            {'params': self.policy.topo_head.parameters(), 'lr': self.lr},
            {'params': self.policy.geo_head.decoder.parameters(), 'lr': self.lr * 0.5},
            {'params': self.policy.geo_head.prior_bias.parameters(), 'lr': self.lr * 0.5},
            {'params': [self.policy.geo_head.prior_bias_scale], 'lr': self.lr * 0.25},
            {'params': self.critic.parameters(), 'lr': self.lr},
        ], lr=self.lr)
        self._trainable_params = (
            list(self.policy.topo_head.parameters())
            + list(self.policy.geo_head.decoder.parameters())
            + list(self.policy.geo_head.prior_bias.parameters())
            + [self.policy.geo_head.prior_bias_scale]
            + list(self.critic.parameters())
        )
        self.buffer = PPOBuffer()

    def _enumerate_topologies(self, graph: Data):
        x_np = graph.x.detach().cpu().numpy()
        is_fixed = x_np[:, 2]
        moving = [idx for idx in range(len(is_fixed)) if is_fixed[idx] == 0]
        fixed = [idx for idx in range(len(is_fixed)) if is_fixed[idx] == 1]
        topologies = []
        for i, u in enumerate(moving):
            for v in moving[i + 1:]:
                for w in fixed:
                    topologies.append((u, v, w))
        return topologies

    @staticmethod
    def _sorted_undirected_edges(edge_index: torch.Tensor):
        edges = set()
        for u, v in edge_index.detach().cpu().numpy().T.tolist():
            if u == v:
                continue
            a, b = sorted((int(u), int(v)))
            edges.add((a, b))
        return sorted(edges)

    @staticmethod
    def _orientation(a, b, c, eps: float):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(value) <= eps:
            return 0
        return 1 if value > 0 else -1

    @staticmethod
    def _on_segment(a, b, c, eps: float):
        return (
            min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        )

    @classmethod
    def _segments_intersect(cls, p1, p2, p3, p4, eps: float):
        o1 = cls._orientation(p1, p2, p3, eps)
        o2 = cls._orientation(p1, p2, p4, eps)
        o3 = cls._orientation(p3, p4, p1, eps)
        o4 = cls._orientation(p3, p4, p2, eps)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and cls._on_segment(p1, p2, p3, eps):
            return True
        if o2 == 0 and cls._on_segment(p1, p2, p4, eps):
            return True
        if o3 == 0 and cls._on_segment(p3, p4, p1, eps):
            return True
        if o4 == 0 and cls._on_segment(p3, p4, p2, eps):
            return True
        return False

    def _passes_geometry_prior(self, graph: Data, u: int, v: int, w: int, n1, n2):
        pos = graph.pos.detach().cpu().numpy()
        min_link_length = float(self.constraint_cfg.get('min_link_length', 0.05))
        min_node_distance = float(self.constraint_cfg.get('min_node_distance', 0.01))
        intersection_eps = float(self.constraint_cfg.get('intersection_eps', 1.0e-8))
        node_clearance = min_node_distance * self.geometry_prior_clearance_scale

        anchor_nodes = {int(u), int(v), int(w)}
        for idx in range(pos.shape[0]):
            if idx in anchor_nodes:
                continue
            if np.linalg.norm(n1 - pos[idx]) < node_clearance:
                return False, 'prior_n1_close_to_node'
            if np.linalg.norm(n2 - pos[idx]) < node_clearance:
                return False, 'prior_n2_close_to_node'

        if np.linalg.norm(n1 - n2) < max(min_link_length, node_clearance):
            return False, 'prior_short_n1_n2'

        candidate_segments = [
            ((int(u), 'n1'), pos[u], n1),
            ((int(v), 'n1'), pos[v], n1),
            (('n1', 'n2'), n1, n2),
            ((int(w), 'n2'), pos[w], n2),
        ]
        for seg_id, p_a, p_b in candidate_segments:
            if np.linalg.norm(p_a - p_b) < min_link_length:
                return False, 'prior_short_edge'

        existing_edges = self._sorted_undirected_edges(graph.edge_index)
        for seg_id, p_a, p_b in candidate_segments:
            seg_nodes = {node for node in seg_id if isinstance(node, int)}
            for e_u, e_v in existing_edges:
                if len({e_u, e_v} & seg_nodes) > 0:
                    continue
                p_c, p_d = pos[e_u], pos[e_v]
                if self._segments_intersect(p_a, p_b, p_c, p_d, intersection_eps):
                    return False, 'prior_edge_intersection'

        for idx_a, (seg_id_a, p_a1, p_a2) in enumerate(candidate_segments):
            seg_nodes_a = set(seg_id_a)
            for seg_id_b, p_b1, p_b2 in candidate_segments[idx_a + 1:]:
                if len(seg_nodes_a & set(seg_id_b)) > 0:
                    continue
                if self._segments_intersect(p_a1, p_a2, p_b1, p_b2, intersection_eps):
                    return False, 'prior_new_edge_intersection'

        return True, 'ok'

    @staticmethod
    def _finalize_diagnostics(diag: Dict) -> Dict:
        out = dict(diag)
        for key, value in list(out.items()):
            if isinstance(value, Counter):
                out[key] = dict(value)
        return out

    def _select_valid_action(self, graph: Data, x_enc: torch.Tensor, probs_i: np.ndarray,
                             z_c_i: torch.Tensor, deterministic: bool):
        from src.inverse.rl_env import apply_j_operator, validate_graph_structure

        topologies = self._enumerate_topologies(graph)
        diagnostics = {
            'num_topologies': len(topologies),
            'sampled_geometries': 0,
            'geometry_prior_rejects': 0,
            'structure_rejects': 0,
            'geometry_prior_reasons': Counter(),
            'structure_reasons': Counter(),
            'valid_action': False,
        }
        if not topologies:
            diagnostics['failure_reason'] = 'no_topology'
            return None, 0.0, self._finalize_diagnostics(diagnostics)

        topo_scores = np.array(
            [probs_i[u] * probs_i[v] * probs_i[w] for (u, v, w) in topologies],
            dtype=np.float64,
        )
        topo_scores = np.maximum(topo_scores, 1e-12)
        topo_probs = topo_scores / topo_scores.sum()

        if deterministic:
            candidate_order = list(np.argsort(topo_probs)[::-1])
        else:
            candidate_order = list(np.random.choice(len(topologies), size=len(topologies), replace=False, p=topo_probs))

        for topo_idx in candidate_order:
            u, v, w = topologies[topo_idx]
            u_feat = x_enc[u].unsqueeze(0)
            v_feat = x_enc[v].unsqueeze(0)
            w_feat = x_enc[w].unsqueeze(0)
            cond = torch.cat([(u_feat + v_feat + w_feat) / 3.0, z_c_i], dim=-1)

            if deterministic:
                coord_candidates = [
                    self.policy.geo_head.prior_mean(cond).squeeze(0).detach().cpu().numpy()
                ]
            else:
                n_samples = max(1, self.geometry_samples_per_topology)
                sampled = self.policy.geo_head.sample(cond, n_samples=n_samples).squeeze(0).detach().cpu().numpy()
                if sampled.ndim == 1:
                    coord_candidates = [sampled]
                else:
                    coord_candidates = [sampled[i] for i in range(sampled.shape[0])]

            for coords in coord_candidates:
                diagnostics['sampled_geometries'] += 1
                is_prior_valid, prior_reason = self._passes_geometry_prior(graph, u, v, w, coords[:2], coords[2:])
                if not is_prior_valid:
                    diagnostics['geometry_prior_rejects'] += 1
                    diagnostics['geometry_prior_reasons'][prior_reason] += 1
                    continue

                candidate_graph = apply_j_operator(graph, u, v, w, coords[:2], coords[2:])
                is_valid, valid_info = validate_graph_structure(candidate_graph, self.constraint_cfg)
                if not is_valid:
                    diagnostics['structure_rejects'] += 1
                    diagnostics['structure_reasons'][valid_info.get('reason', 'invalid_structure')] += 1
                    continue

                log_prob = float(np.log(topo_probs[topo_idx] + 1e-9))
                action = {
                    'u': u,
                    'v': v,
                    'w': w,
                    'n1': coords[:2],
                    'n2': coords[2:],
                }
                diagnostics['valid_action'] = True
                diagnostics['chosen_topology'] = (u, v, w)
                return action, log_prob, self._finalize_diagnostics(diagnostics)

        diagnostics['failure_reason'] = 'no_valid_geometry'
        return None, 0.0, self._finalize_diagnostics(diagnostics)

    def _topology_distribution(self, graph: Data, probs_i: np.ndarray):
        topologies = self._enumerate_topologies(graph)
        if not topologies:
            return [], np.empty(0, dtype=np.float64)
        topo_scores = np.array(
            [probs_i[u] * probs_i[v] * probs_i[w] for (u, v, w) in topologies],
            dtype=np.float64,
        )
        topo_scores = np.maximum(topo_scores, 1e-12)
        topo_probs = topo_scores / topo_scores.sum()
        return topologies, topo_probs

    # ──────────────────────────────────────────────────────────────────────
    # Batched action selection: process N graphs in ONE GPU call
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def batch_select_actions(self, graphs: List[Data], z_cs: torch.Tensor,
                             deterministic: bool = False,
                             return_diagnostics: bool = False,
                             ) -> Tuple[List[Optional[Dict]], List[float], List[float]]:
        """
        Select actions for N graphs simultaneously using the policy network.
        One GPU call for GNN encoding + topology + geometry + critic.

        Args:
            graphs: list of N PyG Data objects (current state graphs)
            z_cs: (N, latent_dim) curve condition vectors, already on device

        Returns:
            actions: list of N action dicts (or None if no valid action)
            log_probs: list of N log-probability floats
            values: list of N value estimates
        """
        from torch_geometric.utils import softmax as pyg_softmax
        from torch_geometric.utils import scatter

        N = len(graphs)
        if N == 0:
            return [], [], []

        # ── Single batched GPU forward ─────────────────────────────────────
        batch = Batch.from_data_list(graphs).to(self.device)
        x_enc = self.policy.encode_graph(batch)

        # Topology scores → per-graph softmax probabilities
        topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)
        node_probs = pyg_softmax(topo_scores, batch.batch)

        # Critic values (batched)
        graph_feat = scatter(x_enc, batch.batch, dim=0, reduce='mean')  # (N, hidden)
        z_cs_dev = z_cs.to(self.device) if not z_cs.is_cuda else z_cs
        critic_inp = torch.cat([graph_feat, z_cs_dev.view(N, -1)], dim=-1)
        values_t = self.critic.value_head(critic_inp).squeeze(-1)  # (N,)
        values_list = values_t.cpu().tolist()

        # ── Per-graph action sampling (CPU, fast since it's just indexing) ──
        ptr = batch.ptr  # node boundaries per graph
        actions = []
        log_probs = []
        diagnostics = []

        for i in range(N):
            start, end = int(ptr[i]), int(ptr[i + 1])
            probs_i = node_probs[start:end].cpu().numpy().astype(np.float64)
            z_c_i = z_cs_dev[i:i+1]
            action, lp, diag = self._select_valid_action(
                graphs[i],
                x_enc[start:end],
                probs_i,
                z_c_i,
                deterministic=deterministic,
            )
            actions.append(action)
            log_probs.append(lp)
            diagnostics.append(diag)

        if return_diagnostics:
            return actions, log_probs, values_list, diagnostics
        return actions, log_probs, values_list

    @torch.no_grad()
    def select_action(self, obs: dict, action: Optional[Dict] = None) -> tuple:
        """Legacy single-graph interface."""
        graph = obs['graph']
        z_c = obs['z_c'].to(self.device) if obs['z_c'] is not None else torch.zeros(
            1, self.policy.curve_latent_dim, device=self.device
        )
        if action is None:
            batch = Batch.from_data_list([graph]).to(self.device)
            value = self.critic(batch, z_c.view(1, -1)).item()
            return None, 0.0, value

        log_prob, value = self._evaluate_single_action(obs, action)
        return action, log_prob, value

    @torch.no_grad()
    def _evaluate_single_action(self, obs: dict, action: Dict) -> tuple:
        """Evaluate the log-probability and value for one explicit action."""
        graph = obs['graph']
        z_c = obs['z_c'].to(self.device) if obs['z_c'] is not None else torch.zeros(
            1, self.policy.curve_latent_dim, device=self.device
        )
        batch = Batch.from_data_list([graph]).to(self.device)
        x_enc = self.policy.encode_graph(batch)
        topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)
        node_probs = torch.softmax(topo_scores, dim=0).detach().cpu().numpy()
        value = self.critic(batch, z_c.view(1, -1)).item()
        topologies, topo_probs = self._topology_distribution(graph, node_probs)
        try:
            topo_idx = topologies.index((action['u'], action['v'], action['w']))
            log_prob = float(np.log(topo_probs[topo_idx] + 1e-9))
        except ValueError:
            log_prob = float(np.log(1e-9))
        return log_prob, value

    def update(self, buffer: PPOBuffer, n_epochs: int = 4):
        valid_idx = [i for i, action in enumerate(buffer.actions) if action is not None]
        if not valid_idx:
            return

        states = [buffer.states[i] for i in valid_idx]
        z_cs = torch.stack([buffer.z_cs[i] for i in valid_idx]).to(self.device).view(len(valid_idx), -1)
        actions = [buffer.actions[i] for i in valid_idx]
        u_s = torch.tensor([a['u'] for a in actions], dtype=torch.long, device=self.device)
        v_s = torch.tensor([a['v'] for a in actions], dtype=torch.long, device=self.device)
        w_s = torch.tensor([a['w'] for a in actions], dtype=torch.long, device=self.device)

        returns, advantages = buffer.compute_returns(self.gamma)
        returns = returns[valid_idx].to(self.device)
        adv_t = advantages[valid_idx].to(self.device)
        old_log_probs = torch.tensor(
            [buffer.log_probs[i] for i in valid_idx], dtype=torch.float32, device=self.device
        )

        batch = Batch.from_data_list(states).to(self.device)
        ptr = batch.ptr[:-1]

        for _ in range(n_epochs):
            x_enc = self.policy.encode_graph(batch)
            topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)

            from torch_geometric.utils import softmax as pyg_softmax
            probs = pyg_softmax(topo_scores, batch.batch)
            new_lp_values = []
            for local_idx, state in enumerate(states):
                start = int(ptr[local_idx].item())
                end = int(batch.ptr[local_idx + 1].item())
                probs_i = probs[start:end].detach().cpu().numpy().astype(np.float64)
                topologies, topo_probs = self._topology_distribution(state, probs_i)
                action_tuple = (int(u_s[local_idx].item()), int(v_s[local_idx].item()), int(w_s[local_idx].item()))
                try:
                    topo_idx = topologies.index(action_tuple)
                    new_lp_values.append(np.log(topo_probs[topo_idx] + 1e-9))
                except ValueError:
                    new_lp_values.append(np.log(1e-9))
            new_lp = torch.tensor(new_lp_values, dtype=torch.float32, device=self.device)
            ratio = torch.exp(new_lp - old_log_probs)

            actor_loss = -torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t,
            ).mean()

            value_pred = self.critic(batch, z_cs)
            critic_loss = nn.functional.mse_loss(value_pred, returns)

            from torch_geometric.utils import scatter
            node_entropies = -(probs * torch.log(probs + 1e-9))
            graph_entropies = scatter(node_entropies, batch.batch, dim=0, reduce='sum')
            entropy = graph_entropies.mean()

            loss = actor_loss + 0.5 * critic_loss - self.entropy_c * entropy
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self._trainable_params, 0.5)
            self.optimizer.step()
