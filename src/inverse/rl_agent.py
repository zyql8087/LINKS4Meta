from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from src.inverse.action_codebook import (
    codebook_bucket_for_step,
    decode_local_dyad_code,
    family_name_from_index,
    step_role_for_index,
)


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
            nn.Linear(hidden_dim + latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
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
        graph_feat = scatter(x_enc, data.batch, dim=0, reduce='mean')
        return self.value_head(torch.cat([graph_feat, z_c], dim=-1)).squeeze(-1)


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
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.dones.append(bool(done))

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

        self.gamma = float(rl_cfg.get('gamma', 0.99))
        self.clip_eps = float(rl_cfg.get('clip_eps', 0.2))
        self.entropy_c = float(rl_cfg.get('entropy_coef', 0.01))
        self.lr = float(rl_cfg.get('learning_rate', 1e-4))
        self.geometry_samples_per_topology = int(rl_cfg.get('geometry_samples_per_topology', 4))
        self.geometry_prior_clearance_scale = float(rl_cfg.get('geometry_prior_clearance_scale', 1.0))
        self.constraint_cfg = cfg.get('constraints', {})

        hidden_dim = int(cfg.get('gnn_policy', {}).get('hidden_dim', 128))
        latent_dim = int(cfg.get('curve_encoder', {}).get('latent_dim', 128))
        self.critic = GraphCritic(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

        self.optimizer = optim.Adam(
            [
                {'params': self.policy.topo_head.parameters(), 'lr': self.lr},
                {'params': self.policy.geometry_code_head.parameters(), 'lr': self.lr},
                {'params': self.critic.parameters(), 'lr': self.lr},
            ],
            lr=self.lr,
        )
        self._trainable_params = (
            list(self.policy.topo_head.parameters())
            + list(self.policy.geometry_code_head.parameters())
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
        for _, p_a, p_b in candidate_segments:
            if np.linalg.norm(p_a - p_b) < min_link_length:
                return False, 'prior_short_edge'

        existing_edges = self._sorted_undirected_edges(graph.edge_index)
        for seg_id, p_a, p_b in candidate_segments:
            seg_nodes = {node for node in seg_id if isinstance(node, int)}
            for e_u, e_v in existing_edges:
                if len({e_u, e_v} & seg_nodes) > 0:
                    continue
                if self._segments_intersect(p_a, p_b, pos[e_u], pos[e_v], intersection_eps):
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

    def _masked_code_distribution(
        self,
        graph: Data,
        x_enc_local: torch.Tensor,
        graph_context: torch.Tensor,
        action_topo: torch.Tensor,
        *,
        family_index: int,
        step_index: int,
        expected_j_steps: int,
    ) -> tuple[list[int], np.ndarray]:
        family_name = family_name_from_index(family_index)
        step_role = step_role_for_index(step_index, expected_j_steps)
        bucket = codebook_bucket_for_step(family_name, step_role)
        allowed_ids = list(self.policy.action_codebook_buckets.get(bucket, []))
        if not allowed_ids:
            allowed_ids = list(range(int(self.policy.action_codebook.size(0))))
        batch = Batch.from_data_list([graph]).to(self.device)
        if x_enc_local.size(0) != graph.x.size(0):
            x_enc_local = self.policy.encode_graph(batch)
        logits = self.policy.geometry_code_logits(
            batch,
            x_enc_local,
            graph_context,
            action_topo.view(1, -1).to(self.device),
        )[0]
        allowed_logits = logits[torch.tensor(allowed_ids, dtype=torch.long, device=logits.device)]
        probs = torch.softmax(allowed_logits, dim=0).detach().cpu().numpy().astype(np.float64)
        return allowed_ids, probs

    def _select_valid_action(self, graph: Data, x_enc: torch.Tensor, probs_i: np.ndarray,
                             z_c_i: torch.Tensor, deterministic: bool, context: Optional[Dict] = None):
        from src.inverse.rl_env import apply_j_operator, validate_graph_structure

        topologies = self._enumerate_topologies(graph)
        diagnostics = {
            'num_topologies': len(topologies),
            'sampled_codes': 0,
            'code_decode_rejects': 0,
            'structure_rejects': 0,
            'code_decode_reasons': Counter(),
            'structure_reasons': Counter(),
            'valid_action': False,
        }
        if not topologies:
            diagnostics['failure_reason'] = 'no_topology'
            return None, 0.0, self._finalize_diagnostics(diagnostics)

        topo_scores = np.array([probs_i[u] * probs_i[v] * probs_i[w] for (u, v, w) in topologies], dtype=np.float64)
        topo_scores = np.maximum(topo_scores, 1e-12)
        topo_probs = topo_scores / topo_scores.sum()

        if deterministic:
            candidate_order = list(np.argsort(topo_probs)[::-1])
        else:
            candidate_order = list(np.random.choice(len(topologies), size=len(topologies), replace=False, p=topo_probs))

        family_index = int((context or {}).get('family_index', self.policy.num_families))
        step_index = int((context or {}).get('step_index', 0))
        expected_j_steps = int((context or {}).get('expected_j_steps', 1))
        graph_context = self.policy.build_il_context(
            Batch.from_data_list([graph]).to(self.device),
            x_enc,
            z_c_i,
            family_ids=torch.tensor([family_index], dtype=torch.long, device=self.device),
            step_indices=torch.tensor([step_index], dtype=torch.long, device=self.device),
            step_counts=torch.tensor([expected_j_steps], dtype=torch.long, device=self.device),
        )[0]

        for topo_idx in candidate_order:
            u, v, w = topologies[topo_idx]
            action_topo = torch.tensor([u, v, w], dtype=torch.long, device=self.device)
            allowed_code_ids, code_probs = self._masked_code_distribution(
                graph,
                x_enc,
                graph_context,
                action_topo,
                family_index=family_index,
                step_index=step_index,
                expected_j_steps=expected_j_steps,
            )
            if deterministic:
                code_order = list(np.argsort(code_probs)[::-1])
            else:
                code_order = list(np.random.choice(len(allowed_code_ids), size=len(allowed_code_ids), replace=False, p=code_probs))

            for code_local_idx in code_order:
                diagnostics['sampled_codes'] += 1
                code_id = int(allowed_code_ids[code_local_idx])
                try:
                    n1, n2 = decode_local_dyad_code(
                        graph.pos[u].detach().cpu().numpy(),
                        graph.pos[v].detach().cpu().numpy(),
                        graph.pos[w].detach().cpu().numpy(),
                        self.policy.action_codebook[code_id].detach().cpu().numpy(),
                    )
                except Exception as exc:
                    diagnostics['code_decode_rejects'] += 1
                    diagnostics['code_decode_reasons'][type(exc).__name__] += 1
                    continue

                is_prior_valid, prior_reason = self._passes_geometry_prior(graph, u, v, w, n1, n2)
                if not is_prior_valid:
                    diagnostics['code_decode_rejects'] += 1
                    diagnostics['code_decode_reasons'][prior_reason] += 1
                    continue

                candidate_graph = apply_j_operator(graph, u, v, w, n1, n2)
                is_valid, valid_info = validate_graph_structure(candidate_graph, self.constraint_cfg)
                if not is_valid:
                    diagnostics['structure_rejects'] += 1
                    diagnostics['structure_reasons'][valid_info.get('reason', 'invalid_structure')] += 1
                    continue

                diagnostics['valid_action'] = True
                diagnostics['chosen_topology'] = (u, v, w)
                diagnostics['chosen_code_id'] = code_id
                return {
                    'u': u,
                    'v': v,
                    'w': w,
                    'code_id': code_id,
                    'n1': n1,
                    'n2': n2,
                    'family_index': family_index,
                    'step_index': step_index,
                    'expected_j_steps': expected_j_steps,
                }, float(np.log(topo_probs[topo_idx] + 1e-9) + np.log(code_probs[code_local_idx] + 1e-9)), self._finalize_diagnostics(diagnostics)

        diagnostics['failure_reason'] = 'no_valid_geometry_code'
        return None, 0.0, self._finalize_diagnostics(diagnostics)

    def _topology_distribution(self, graph: Data, probs_i: np.ndarray):
        topologies = self._enumerate_topologies(graph)
        if not topologies:
            return [], np.empty(0, dtype=np.float64)
        topo_scores = np.array([probs_i[u] * probs_i[v] * probs_i[w] for (u, v, w) in topologies], dtype=np.float64)
        topo_scores = np.maximum(topo_scores, 1e-12)
        topo_probs = topo_scores / topo_scores.sum()
        return topologies, topo_probs

    @torch.no_grad()
    def rank_action_candidates(
        self,
        graph: Data,
        z_c: torch.Tensor,
        *,
        context: Optional[Dict] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        batch = Batch.from_data_list([graph]).to(self.device)
        x_enc = self.policy.encode_graph(batch)
        from torch_geometric.utils import softmax as pyg_softmax

        topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)
        node_probs = pyg_softmax(topo_scores, batch.batch).detach().cpu().numpy().astype(np.float64)
        probs_i = node_probs[: graph.x.size(0)]
        z_c_i = z_c.to(self.device).view(1, -1)

        stop_prob = 0.0
        allow_stop = False
        if context is not None:
            family_ids = torch.tensor(
                [int(context.get('family_index', self.policy.num_families))],
                dtype=torch.long,
                device=self.device,
            )
            step_indices = torch.tensor(
                [int(context.get('step_index', 0))],
                dtype=torch.long,
                device=self.device,
            )
            step_counts = torch.tensor(
                [int(context.get('expected_j_steps', 1))],
                dtype=torch.long,
                device=self.device,
            )
            phase4_outputs = self.policy.phase4_outputs(
                batch,
                x_enc,
                z_c_i,
                family_ids=family_ids,
                step_indices=step_indices,
                step_counts=step_counts,
            )
            stop_prob = float(torch.sigmoid(phase4_outputs['stop_logits'][0]).item())
            allow_stop = bool(context.get('can_stop', False))

        topologies, topo_probs = self._topology_distribution(graph, probs_i)
        candidates = []
        if allow_stop:
            candidates.append(
                {
                    'action': {'stop': True},
                    'log_prob': float(np.log(stop_prob + 1e-9)),
                    'policy_score': stop_prob,
                    'stop_probability': stop_prob,
                    'stop': True,
                }
            )

        topo_order = list(np.argsort(topo_probs)[::-1]) if len(topo_probs) > 0 else []
        keep_count = max(int(top_k or 1), 1)
        for topo_idx in topo_order:
            if len([item for item in candidates if not item['stop']]) >= keep_count:
                break
            u, v, w = topologies[topo_idx]
            action_topo = torch.tensor([u, v, w], dtype=torch.long, device=self.device)
            allowed_code_ids, code_probs = self._masked_code_distribution(
                graph,
                x_enc[: graph.x.size(0)],
                phase4_outputs['graph_context'],
                action_topo,
                family_index=int(context.get('family_index', self.policy.num_families)) if context is not None else self.policy.num_families,
                step_index=int(context.get('step_index', 0)) if context is not None else 0,
                expected_j_steps=int(context.get('expected_j_steps', 1)) if context is not None else 1,
            )
            if not allowed_code_ids:
                continue
            code_order = list(np.argsort(code_probs)[::-1])
            candidate_graph = None
            action = None
            chosen_code_prob = 0.0
            for code_local_idx in code_order:
                code_id = int(allowed_code_ids[code_local_idx])
                try:
                    n1, n2 = decode_local_dyad_code(
                        graph.pos[u].detach().cpu().numpy(),
                        graph.pos[v].detach().cpu().numpy(),
                        graph.pos[w].detach().cpu().numpy(),
                        self.policy.action_codebook[code_id].detach().cpu().numpy(),
                    )
                except Exception:
                    continue
                is_prior_valid, _ = self._passes_geometry_prior(graph, u, v, w, n1, n2)
                if not is_prior_valid:
                    continue
                try:
                    from src.inverse.rl_env import apply_j_operator, validate_graph_structure

                    candidate_graph = apply_j_operator(graph, u, v, w, n1, n2)
                    is_valid, _ = validate_graph_structure(candidate_graph, self.constraint_cfg)
                    if not is_valid:
                        candidate_graph = None
                        continue
                except Exception:
                    candidate_graph = None
                    continue
                chosen_code_prob = float(code_probs[code_local_idx])
                action = {
                    'u': int(u),
                    'v': int(v),
                    'w': int(w),
                    'code_id': code_id,
                    'n1': n1,
                    'n2': n2,
                    'family_index': int(context.get('family_index', self.policy.num_families)) if context is not None else self.policy.num_families,
                    'step_index': int(context.get('step_index', 0)) if context is not None else 0,
                    'expected_j_steps': int(context.get('expected_j_steps', 1)) if context is not None else 1,
                    'stop': False,
                }
                break
            if action is None or candidate_graph is None:
                continue
            non_stop_prob = max(1e-9, 1.0 - stop_prob) if allow_stop else 1.0
            policy_score = float(topo_probs[topo_idx] * chosen_code_prob * non_stop_prob)
            candidates.append(
                {
                    'action': action,
                    'graph': candidate_graph,
                    'log_prob': float(np.log(policy_score + 1e-9)),
                    'policy_score': policy_score,
                    'stop_probability': stop_prob,
                    'stop': False,
                }
            )

        candidates.sort(key=lambda item: item['log_prob'], reverse=True)
        return candidates[:keep_count]

    @torch.no_grad()
    def batch_select_actions(self, graphs: List[Data], z_cs: torch.Tensor,
                             deterministic: bool = False,
                             return_diagnostics: bool = False,
                             contexts: Optional[List[Dict]] = None,
                             ) -> Tuple[List[Optional[Dict]], List[float], List[float]]:
        if not graphs:
            return [], [], []

        batch = Batch.from_data_list(graphs).to(self.device)
        x_enc = self.policy.encode_graph(batch)
        from torch_geometric.utils import softmax as pyg_softmax

        topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)
        node_probs = pyg_softmax(topo_scores, batch.batch)

        graph_feat = scatter(x_enc, batch.batch, dim=0, reduce='mean')
        z_cs_dev = z_cs.to(self.device) if not z_cs.is_cuda else z_cs
        values_t = self.critic.value_head(torch.cat([graph_feat, z_cs_dev.view(len(graphs), -1)], dim=-1)).squeeze(-1)
        values_list = values_t.detach().cpu().tolist()

        stop_probs = None
        if contexts is not None:
            family_ids = torch.tensor(
                [int(ctx.get('family_index', self.policy.num_families)) for ctx in contexts],
                dtype=torch.long,
                device=self.device,
            )
            step_indices = torch.tensor(
                [int(ctx.get('step_index', 0)) for ctx in contexts],
                dtype=torch.long,
                device=self.device,
            )
            step_counts = torch.tensor(
                [int(ctx.get('expected_j_steps', 1)) for ctx in contexts],
                dtype=torch.long,
                device=self.device,
            )
            phase4_outputs = self.policy.phase4_outputs(
                batch,
                x_enc,
                z_cs_dev.view(len(graphs), -1),
                family_ids=family_ids,
                step_indices=step_indices,
                step_counts=step_counts,
            )
            stop_probs = torch.sigmoid(phase4_outputs['stop_logits']).detach().cpu().numpy().astype(np.float64)

        ptr = batch.ptr
        actions = []
        log_probs = []
        diagnostics = []
        for i, graph in enumerate(graphs):
            start, end = int(ptr[i]), int(ptr[i + 1])
            probs_i = node_probs[start:end].detach().cpu().numpy().astype(np.float64)
            z_c_i = z_cs_dev[i:i + 1]
            stop_prob = float(stop_probs[i]) if stop_probs is not None else 0.0
            allow_stop = bool(contexts is not None and contexts[i].get('can_stop', False))
            stop_threshold = float(contexts[i].get('stop_threshold', 0.5)) if contexts is not None else 0.5
            if allow_stop:
                stop_selected = bool(stop_prob >= stop_threshold) if deterministic else bool(np.random.rand() < stop_prob)
                if stop_selected:
                    actions.append({'stop': True})
                    log_probs.append(float(np.log(stop_prob + 1e-9)))
                    diagnostics.append(
                        {
                            'valid_action': True,
                            'stop_selected': True,
                            'stop_probability': stop_prob,
                            'num_topologies': 0,
                            'sampled_geometries': 0,
                            'geometry_prior_rejects': 0,
                            'structure_rejects': 0,
                            'geometry_prior_reasons': {},
                            'structure_reasons': {},
                        }
                    )
                    continue

            action, lp, diag = self._select_valid_action(
                graph,
                x_enc[start:end],
                probs_i,
                z_c_i,
                deterministic=deterministic,
                context=contexts[i] if contexts is not None else None,
            )
            if action is not None:
                action['stop'] = False
                action['stop_probability'] = stop_prob
                action['allow_stop'] = allow_stop
            if allow_stop:
                lp += float(np.log(max(1e-9, 1.0 - stop_prob)))
            diag['stop_selected'] = False
            diag['stop_probability'] = stop_prob
            actions.append(action)
            log_probs.append(lp)
            diagnostics.append(diag)

        if return_diagnostics:
            return actions, log_probs, values_list, diagnostics
        return actions, log_probs, values_list

    @torch.no_grad()
    def select_action(self, obs: dict, action: Optional[Dict] = None) -> tuple:
        graph = obs['graph']
        z_c = obs['z_c'].to(self.device) if obs['z_c'] is not None else torch.zeros(1, self.policy.curve_latent_dim, device=self.device)
        if action is None:
            batch = Batch.from_data_list([graph]).to(self.device)
            value = self.critic(batch, z_c.view(1, -1)).item()
            return None, 0.0, value
        log_prob, value = self._evaluate_single_action(obs, action)
        return action, log_prob, value

    @torch.no_grad()
    def _evaluate_single_action(self, obs: dict, action: Dict) -> tuple:
        graph = obs['graph']
        z_c = obs['z_c'].to(self.device) if obs['z_c'] is not None else torch.zeros(1, self.policy.curve_latent_dim, device=self.device)
        batch = Batch.from_data_list([graph]).to(self.device)
        x_enc = self.policy.encode_graph(batch)
        topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)
        node_probs = torch.softmax(topo_scores, dim=0).detach().cpu().numpy()
        value = self.critic(batch, z_c.view(1, -1)).item()
        topologies, topo_probs = self._topology_distribution(graph, node_probs)
        try:
            topo_idx = topologies.index((action['u'], action['v'], action['w']))
            action_topo = torch.tensor([[action['u'], action['v'], action['w']]], dtype=torch.long, device=self.device)
            context, _ = self.policy.build_il_context(
                batch,
                x_enc,
                z_c.view(1, -1),
                family_ids=torch.tensor([int(action.get('family_index', obs.get('family_index', self.policy.num_families)))], dtype=torch.long, device=self.device),
                step_indices=torch.tensor([int(action.get('step_index', obs.get('step', 0)))], dtype=torch.long, device=self.device),
                step_counts=torch.tensor([int(action.get('expected_j_steps', obs.get('expected_j_steps', 1)))], dtype=torch.long, device=self.device),
            )
            allowed_code_ids, code_probs = self._masked_code_distribution(
                graph,
                x_enc,
                context,
                action_topo.view(-1),
                family_index=int(action.get('family_index', obs.get('family_index', self.policy.num_families))),
                step_index=int(action.get('step_index', obs.get('step', 0))),
                expected_j_steps=int(action.get('expected_j_steps', obs.get('expected_j_steps', 1))),
            )
            code_prob = 1.0e-9
            if int(action.get('code_id', -1)) in allowed_code_ids:
                code_prob = float(code_probs[allowed_code_ids.index(int(action['code_id']))])
            non_stop_prob = max(1e-9, 1.0 - float(action.get('stop_probability', 0.0))) if bool(action.get('allow_stop', False)) else 1.0
            log_prob = float(np.log(topo_probs[topo_idx] + 1e-9) + np.log(code_prob + 1e-9) + np.log(non_stop_prob + 1e-9))
        except ValueError:
            log_prob = float(np.log(1e-9))
        return log_prob, value

    def update(self, buffer: PPOBuffer, n_epochs: int = 4):
        valid_idx = [
            idx for idx, action in enumerate(buffer.actions)
            if action is not None and not bool(action.get('stop', False))
        ]
        if not valid_idx:
            return

        states = [buffer.states[idx] for idx in valid_idx]
        z_cs = torch.stack([buffer.z_cs[idx] for idx in valid_idx]).to(self.device).view(len(valid_idx), -1)
        actions = [buffer.actions[idx] for idx in valid_idx]
        u_s = torch.tensor([action['u'] for action in actions], dtype=torch.long, device=self.device)
        v_s = torch.tensor([action['v'] for action in actions], dtype=torch.long, device=self.device)
        w_s = torch.tensor([action['w'] for action in actions], dtype=torch.long, device=self.device)
        code_s = torch.tensor([action['code_id'] for action in actions], dtype=torch.long, device=self.device)
        family_s = torch.tensor([int(action.get('family_index', self.policy.num_families)) for action in actions], dtype=torch.long, device=self.device)
        step_s = torch.tensor([int(action.get('step_index', 0)) for action in actions], dtype=torch.long, device=self.device)
        expected_s = torch.tensor([int(action.get('expected_j_steps', 1)) for action in actions], dtype=torch.long, device=self.device)
        stop_prob_s = torch.tensor([float(action.get('stop_probability', 0.0)) for action in actions], dtype=torch.float32, device=self.device)
        allow_stop_s = torch.tensor([1.0 if bool(action.get('allow_stop', False)) else 0.0 for action in actions], dtype=torch.float32, device=self.device)

        returns, advantages = buffer.compute_returns(self.gamma)
        returns = returns[valid_idx].to(self.device)
        adv_t = advantages[valid_idx].to(self.device)
        old_log_probs = torch.tensor([buffer.log_probs[idx] for idx in valid_idx], dtype=torch.float32, device=self.device)

        batch = Batch.from_data_list(states).to(self.device)
        ptr = batch.ptr[:-1]

        for _ in range(int(n_epochs)):
            from torch_geometric.utils import softmax as pyg_softmax

            x_enc = self.policy.encode_graph(batch)
            topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)
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
                    state_batch = Batch.from_data_list([state]).to(self.device)
                    local_x_enc = x_enc[start:end]
                    context, _ = self.policy.build_il_context(
                        state_batch,
                        local_x_enc,
                        z_cs[local_idx : local_idx + 1],
                    )
                    allowed_code_ids, code_probs = self._masked_code_distribution(
                        state,
                        local_x_enc,
                        context,
                        torch.tensor(action_tuple, dtype=torch.long, device=self.device),
                        family_index=int(family_s[local_idx].item()),
                        step_index=int(step_s[local_idx].item()),
                        expected_j_steps=int(expected_s[local_idx].item()),
                    )
                    code_prob = 1.0e-9
                    if int(code_s[local_idx].item()) in allowed_code_ids:
                        code_prob = float(code_probs[allowed_code_ids.index(int(code_s[local_idx].item()))])
                    non_stop_prob = max(1e-9, 1.0 - float(stop_prob_s[local_idx].item())) if float(allow_stop_s[local_idx].item()) > 0.5 else 1.0
                    new_lp_values.append(np.log(topo_probs[topo_idx] + 1e-9) + np.log(code_prob + 1e-9) + np.log(non_stop_prob + 1e-9))
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
            entropy = (-(probs * torch.log(probs + 1e-9))).mean()

            loss = actor_loss + 0.5 * critic_loss - self.entropy_c * entropy
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self._trainable_params, 0.5)
            self.optimizer.step()
