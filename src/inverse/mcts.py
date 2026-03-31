import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from src.inverse.experiment_utils import compute_reward_batch


class MCTSNode:
    def __init__(self, state_graph, action=None, parent=None, prior: float = 0.0):
        self.state_graph = state_graph
        self.action = action
        self.parent = parent
        self.prior = prior
        self.children: List['MCTSNode'] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def uct_score(self, c_puct: float = 1.4) -> float:
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct: float = 1.4) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.uct_score(c_puct))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def compute_batched_rewards(pred_foot, pred_knee, pred_ankle, target: dict, reward_cfg: dict) -> np.ndarray:
    reward_t, _ = compute_reward_batch(pred_foot, pred_knee, pred_ankle, target, reward_cfg)
    return reward_t.detach().cpu().numpy().astype(np.float32)


class MCTS:
    def __init__(self, policy, surrogate, env, cfg: dict, device):
        self.policy = policy
        self.surrogate = surrogate
        self.env = env
        self.cfg = cfg
        self.device = device
        self.c_puct = cfg.get('mcts', {}).get('c_puct', 1.4)
        self.n_rollouts = cfg.get('mcts', {}).get('num_rollouts', 32)
        self.constraint_cfg = cfg.get('constraints', {})

    @torch.no_grad()
    def _expand_unified(self, node: MCTSNode, z_c: torch.Tensor):
        """Expand a node with one encoding pass and constraint-aware children."""
        from torch_geometric.data import Batch
        from src.inverse.rl_env import apply_j_operator, validate_graph_structure

        x = node.state_graph.x.numpy()
        is_fixed = x[:, 2]
        moving = [i for i in range(len(is_fixed)) if is_fixed[i] == 0]
        fixed = [i for i in range(len(is_fixed)) if is_fixed[i] == 1]

        valid_actions = []
        for i, u in enumerate(moving):
            for v in moving[i + 1:]:
                for w in fixed:
                    valid_actions.append((u, v, w))
        if not valid_actions:
            return None

        was_training = self.policy.training
        self.policy.eval()

        # ── Single GPU call: encode graph ONCE ──────────────────────────────
        batch = Batch.from_data_list([node.state_graph]).to(self.device)
        x_enc = self.policy.encode_graph(batch)

        # ── Topo priors (reuse x_enc) ──────────────────────────────────────
        topo_scores = self.policy.topology_scores(x_enc).squeeze(-1)
        node_probs = torch.softmax(topo_scores, dim=0).cpu().numpy()

        priors = {}
        for (u, v, w) in valid_actions:
            priors[(u, v, w)] = float(node_probs[u]) * float(node_probs[v]) * float(node_probs[w])
        total = sum(priors.values()) + 1e-9
        priors = {k: v / total for k, v in priors.items()}

        # ── Geometry sampling (reuse x_enc) ────────────────────────────────
        action_topo = torch.tensor(valid_actions, dtype=torch.long, device=self.device)

        u_f = x_enc[action_topo[:, 0]]
        v_f = x_enc[action_topo[:, 1]]
        w_f = x_enc[action_topo[:, 2]]
        uvw_feat = (u_f + v_f + w_f) / 3.0

        z_c_d = z_c.to(self.device) if z_c is not None else torch.zeros(
            1, self.policy.curve_latent_dim, device=self.device
        )
        z_c_expanded = z_c_d.expand(len(valid_actions), -1)
        cond = torch.cat([uvw_feat, z_c_expanded], dim=-1)
        batched_coords = self.policy.geo_head.sample(cond, n_samples=1).squeeze(1).cpu().numpy()

        if was_training:
            self.policy.train()

        # ── Build children ─────────────────────────────────────────────────
        for idx, (u, v, w) in enumerate(valid_actions):
            n1 = batched_coords[idx, :2]
            n2 = batched_coords[idx, 2:]
            candidate_graph = apply_j_operator(node.state_graph, u, v, w, n1, n2)
            is_valid, _ = validate_graph_structure(candidate_graph, self.constraint_cfg)
            if not is_valid:
                continue
            child = MCTSNode(
                state_graph=node.state_graph,
                action={'u': u, 'v': v, 'w': w, 'n1': n1, 'n2': n2},
                parent=node,
                prior=priors.get((u, v, w), 0.0),
            )
            node.children.append(child)
        node.is_expanded = True
        return x_enc

    def _backprop(self, node: MCTSNode, value: float):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def search(self, root_graph, z_c: torch.Tensor, target: dict) -> Tuple[Optional[dict], Optional[torch.Tensor]]:
        """
        Returns:
            action: dict or None
            x_enc: the cached GNN encoding of the root graph (reusable by agent)
        """
        from src.inverse.rl_env import apply_j_operator, batch_compute_rewards

        root = MCTSNode(state_graph=root_graph)
        sim_nodes = []
        cached_x_enc = None

        for _ in range(self.n_rollouts):
            node = root
            while not node.is_leaf() and node.is_expanded:
                node = node.best_child(self.c_puct)

            if not node.is_expanded:
                x_enc = self._expand_unified(node, z_c)
                if cached_x_enc is None and node is root:
                    cached_x_enc = x_enc

            sim_nodes.append(np.random.choice(node.children) if node.children else node)

        graphs_to_eval = []
        for s_node in sim_nodes:
            action = s_node.action
            if action is None:
                graphs_to_eval.append(s_node.state_graph)
                continue
            graphs_to_eval.append(
                apply_j_operator(
                    s_node.parent.state_graph,
                    action['u'], action['v'], action['w'],
                    action['n1'], action['n2'],
                )
            )

        batch_rewards = batch_compute_rewards(
            self.surrogate,
            graphs_to_eval,
            target,
            self.cfg.get('reward', {}),
            self.device,
            constraint_cfg=self.constraint_cfg,
        )

        for idx, s_node in enumerate(sim_nodes):
            reward, _ = batch_rewards[idx]
            self._backprop(s_node, float(reward))

        if not root.children:
            return None, cached_x_enc
        best = max(root.children, key=lambda n: n.visit_count)
        return best.action, cached_x_enc
