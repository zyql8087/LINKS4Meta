# src/inverse/train_il.py
# IL 琛屼负鍏嬮殕璁粌寰幆锛堝叏鍚戦噺鍖栨壒閲忕増鏈級
# 浠?pkl 鐨?gen_info 涓彁鍙栦笓瀹惰矾寰勶紝浣跨敤瀹屽叏鎵归噺鍖栫殑 PyG 鍓嶅悜浼犳挱璁粌 GNN Policy

import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from src.kinematics_extract import extract_kinematics


def _resolve_semantic_action(sample: dict):
    analysis = sample['analysis']
    gen_info = sample.get('gen_info') or analysis.get('gen_info')
    if gen_info is not None:
        return {
            'u': int(gen_info['u']),
            'v': int(gen_info['v']),
            'w': int(gen_info['w']),
            'n1': int(gen_info['n1']),
            'n2': int(gen_info['n2']),
        }

    trace = sample.get('generation_trace') or []
    if not trace:
        return None

    semantic_step_id = sample.get('semantic_step_id')
    semantic_step = None
    for step in trace:
        if bool(step.get('is_semantic', False)):
            semantic_step = step
            break
    if semantic_step is None and semantic_step_id is not None:
        semantic_step = next(
            (step for step in trace if int(step.get('step_id', -1)) == int(semantic_step_id)),
            None,
        )
    if semantic_step is None:
        semantic_step = trace[-1]

    return {
        'u': int(semantic_step['u']),
        'v': int(semantic_step['v']),
        'w': int(semantic_step['w']),
        'n1': int(semantic_step['n1']),
        'n2': int(semantic_step['n2']),
    }


def _batch_offsets(base_data):
    ptr = getattr(base_data, 'ptr', None)
    if ptr is not None:
        return ptr[:-1].to(dtype=torch.long, device=base_data.x.device)
    return torch.tensor([0], dtype=torch.long, device=base_data.x.device)


def _batch_graph_slices(base_data):
    ptr = getattr(base_data, 'ptr', None)
    if ptr is not None:
        return [(int(ptr[i].item()), int(ptr[i + 1].item())) for i in range(ptr.numel() - 1)]
    return [(0, int(base_data.x.size(0)))]


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 涓撳璺緞鎻愬彇锛氫粠 6 鏉?pkl 鏍锋湰閲嶅缓 4 鏉?鈫?6 鏉?鐨勪笓瀹跺姩浣?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def extract_expert_paths(pkl_path: str, output_path: str):
    """
    璇诲彇 pkl锛屼负姣忎釜鏍锋湰鎻愬彇 IL 鐘舵€?鍔ㄤ綔瀵?
      state        = 4 鏉嗗浘鐨?PyG Data (鍘绘帀 n1, n2 鍙婂叾杩炶竟)
      action_topo  = (u, v, w) 鎸傝浇鐐圭储寮曪紙鐩稿浜?4 鏉嗚妭鐐归泦鍚堬級
      action_geo   = (n1.x, n1.y, n2.x, n2.y) 鏂拌妭鐐瑰綊涓€鍖栧潗鏍?
      condition    = (y_foot, y_knee, y_ankle) 鐩爣鏇茬嚎
    """
    print(f"Loading pkl from {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Loaded {len(raw_data)} samples.")

    expert_paths = []
    errors = 0
    error_examples = []

    for sample_idx, sample in enumerate(tqdm(raw_data, desc="Extracting IL expert paths")):
        try:
            A = sample['A']
            x0 = sample['x0']
            types = sample['types']
            analysis = sample['analysis']
            semantic_action = _resolve_semantic_action(sample)

            if semantic_action is None:
                errors += 1
                continue

            u = semantic_action['u']
            v = semantic_action['v']
            w = semantic_action['w']
            n1 = semantic_action['n1']
            n2 = semantic_action['n2']

            # Remove the semantic dyad to recover the graph state immediately before the semantic step.
            base_nodes = [i for i in range(A.shape[0]) if i not in (n1, n2)]
            node_remap = {old: new for new, old in enumerate(base_nodes)}

            A_base = A[np.ix_(base_nodes, base_nodes)]
            x0_base = x0[base_nodes]
            types_base = types[base_nodes]

            rows, cols = np.where(A_base)
            edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

            is_fixed = (types_base == 1).astype(np.float32)
            is_grounded = np.zeros_like(is_fixed)
            is_grounded[0] = 1
            x_feat = np.column_stack([x0_base, is_fixed, is_grounded])

            base_data = Data(
                x=torch.tensor(x_feat, dtype=torch.float32),
                pos=torch.tensor(x0_base, dtype=torch.float32),
                edge_index=edge_index,
            )
            knee_idx = analysis.get('knee')
            if knee_idx in node_remap:
                base_data.knee_idx = torch.tensor([node_remap[knee_idx]], dtype=torch.long)

            # 鈹€鈹€ 鍔ㄤ綔 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            u_r = node_remap.get(u)
            v_r = node_remap.get(v)
            w_r = node_remap.get(w)
            if any(x is None for x in [u_r, v_r, w_r]):
                errors += 1
                continue

            action_topo = torch.tensor([u_r, v_r, w_r], dtype=torch.long)
            action_geo = torch.tensor(
                [x0[n1, 0], x0[n1, 1], x0[n2, 0], x0[n2, 1]],
                dtype=torch.float32
            )

            # 鈹€鈹€ 鐩爣鏇茬嚎锛堟潯浠讹級鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            foot_traj, knee_angle, ankle_angle = extract_kinematics(sample)
            y_foot = torch.tensor(foot_traj, dtype=torch.float32)    # (200, 2)
            y_knee = torch.tensor(knee_angle, dtype=torch.float32)   # (200,)
            y_ankle = torch.tensor(ankle_angle, dtype=torch.float32) # (200,)

            expert_paths.append({
                'sample_id': sample_idx,
                'base_data': base_data,
                'action_topo': action_topo,
                'action_geo': action_geo,
                'y_foot': y_foot,
                'y_knee': y_knee,
                'y_ankle': y_ankle,
            })

        except Exception as e:
            errors += 1
            if len(error_examples) < 5:
                error_examples.append(f"sample_idx={sample_idx}: {type(e).__name__}: {e}")

    print(f"\n[OK] Extracted {len(expert_paths)} expert paths. ({errors} errors skipped)")
    if error_examples:
        print("[!] Example extraction failures:")
        for msg in error_examples:
            print(f"    - {msg}")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(expert_paths, output_path)
    print(f"[OK] Saved to {output_path}")
    return expert_paths


def expert_paths_have_semantics(expert_paths) -> bool:
    if not expert_paths:
        return False
    base_data = expert_paths[0].get('base_data')
    return base_data is not None and hasattr(base_data, 'knee_idx')


def ensure_expert_paths(pkl_path: str, output_path: str, use_cached: bool = True):
    if use_cached and os.path.exists(output_path):
        print(f"[*] Loading cached IL dataset from {output_path}")
        expert_paths = torch.load(output_path, map_location='cpu', weights_only=False)
        if expert_paths_have_semantics(expert_paths):
            return expert_paths
        print("[*] Cached IL dataset is missing knee semantics; regenerating...")

    return extract_expert_paths(
        pkl_path=pkl_path,
        output_path=output_path,
    )


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# IL Dataset
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class ILDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鎵归噺璁＄畻 IL 鎹熷け锛堝叏鍚戦噺鍖栵紝鍋囪鎵€鏈夊浘 4 鑺傜偣锛?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def compute_il_metrics_batched(topo_scores_all, action_topo,
                               geo_pred, geo_mu, geo_logvar, true_geo, cfg,
                               nodes_per_graph=4, geo_prior_pred=None,
                               base_data=None,
                               geo_prior_regularizer_post=None,
                               geo_prior_regularizer_prior=None):
    """Return decomposed IL losses for both posterior and prior geometry paths."""
    il_cfg = cfg.get('il_training', cfg)
    w_topo = il_cfg.get('w_topology', 1.0)
    w_geo  = il_cfg.get('w_geometry', 1.0)
    beta   = cfg.get('cvae', {}).get('beta', 1.0)

    scores = topo_scores_all.squeeze(-1)
    if base_data is None:
        B = action_topo.size(0)
        expected_num_scores = B * nodes_per_graph
        if scores.numel() != expected_num_scores:
            raise ValueError(
                f"Topology score shape mismatch: got {scores.numel()} scores for batch={B}, nodes_per_graph={nodes_per_graph}"
            )
        scores_2d = scores.view(B, nodes_per_graph)
        if action_topo.min().item() < 0 or action_topo.max().item() >= nodes_per_graph:
            raise ValueError(
                f"Topology action index out of range for nodes_per_graph={nodes_per_graph}: "
                f"min={action_topo.min().item()}, max={action_topo.max().item()}"
            )
        topo_targets = torch.zeros_like(scores_2d)
        topo_targets.scatter_(1, action_topo, 1.0)
        loss_topo = nn.functional.binary_cross_entropy_with_logits(scores_2d, topo_targets)
    else:
        offsets = _batch_offsets(base_data)
        if offsets.numel() != action_topo.size(0):
            raise ValueError(
                f"Batch graph count mismatch: offsets={offsets.numel()} action_topo={action_topo.size(0)}"
            )
        global_action = action_topo.to(offsets.device) + offsets.unsqueeze(1)
        if global_action.min().item() < 0 or global_action.max().item() >= scores.numel():
            raise ValueError(
                f"Topology action index out of range for batched graph: "
                f"min={global_action.min().item()}, max={global_action.max().item()}, scores={scores.numel()}"
            )
        topo_targets = torch.zeros_like(scores)
        topo_targets[global_action.reshape(-1)] = 1.0
        loss_topo = nn.functional.binary_cross_entropy_with_logits(scores, topo_targets)

    loss_recon = nn.functional.mse_loss(geo_pred, true_geo)
    loss_kl = -0.5 * torch.mean(1 + geo_logvar - geo_mu.pow(2) - geo_logvar.exp())
    loss_geo = loss_recon + beta * loss_kl

    loss_geo_prior = None
    if geo_prior_pred is not None:
        loss_geo_prior = nn.functional.mse_loss(geo_prior_pred, true_geo)

    prior_weight = il_cfg.get('w_geometry_prior', 0.0)
    reg_weight = il_cfg.get('w_geometry_prior_regularizer', 0.0)
    reg_post = geo_prior_regularizer_post if geo_prior_regularizer_post is not None else torch.zeros_like(loss_recon)
    reg_prior = geo_prior_regularizer_prior if geo_prior_regularizer_prior is not None else torch.zeros_like(loss_recon)
    loss_geo_regularizer = 0.5 * reg_post + reg_prior
    total_posterior = w_topo * loss_topo + w_geo * loss_geo
    total_prior = (
        w_topo * loss_topo
        + w_geo * (loss_geo_prior if loss_geo_prior is not None else loss_recon)
        + reg_weight * reg_prior
    )
    total = total_posterior
    if loss_geo_prior is not None and prior_weight > 0:
        total = total + w_geo * prior_weight * loss_geo_prior
    if reg_weight > 0:
        total = total + reg_weight * loss_geo_regularizer

    if loss_geo_prior is None:
        loss_geo_prior = torch.zeros_like(loss_recon)

    return {
        'total': total,
        'total_posterior': total_posterior,
        'total_prior': total_prior,
        'loss_topo': loss_topo,
        'loss_geo': loss_geo,
        'loss_recon': loss_recon,
        'loss_kl': loss_kl,
        'loss_geo_prior': loss_geo_prior,
        'loss_geo_regularizer': loss_geo_regularizer,
    }


def compute_il_loss_batched(topo_scores_all, action_topo,
                            geo_pred, geo_mu, geo_logvar, true_geo, cfg,
                            nodes_per_graph=4, base_data=None):
    metrics = compute_il_metrics_batched(
        topo_scores_all, action_topo,
        geo_pred, geo_mu, geo_logvar, true_geo, cfg,
        nodes_per_graph=nodes_per_graph,
        base_data=base_data,
    )
    return metrics['total'], metrics['loss_topo'].item(), metrics['loss_geo'].item()



def _build_geo_conditions(x_enc, action_topo, z_c, base_data=None, nodes_per_graph=4):
    """
    鏋勯€?C-VAE 鏉′欢鍚戦噺锛堟壒閲忥級
    x_enc       : (N_total, hidden_dim)
    action_topo : (B, 3)
    z_c         : (B, latent_dim)
    """
    B = action_topo.size(0)
    if base_data is None:
        expected_num_nodes = B * nodes_per_graph
        if x_enc.size(0) != expected_num_nodes:
            raise ValueError(
                f"Encoded node shape mismatch: got {x_enc.size(0)} nodes for batch={B}, nodes_per_graph={nodes_per_graph}"
            )
        x_2d = x_enc.view(B, nodes_per_graph, -1)
        u_f = x_2d[torch.arange(B), action_topo[:, 0]]
        v_f = x_2d[torch.arange(B), action_topo[:, 1]]
        w_f = x_2d[torch.arange(B), action_topo[:, 2]]
    else:
        offsets = _batch_offsets(base_data).to(x_enc.device)
        if offsets.numel() != B:
            raise ValueError(
                f"Batch graph count mismatch: offsets={offsets.numel()} action_topo={B}"
            )
        global_action = action_topo.to(x_enc.device) + offsets.unsqueeze(1)
        u_f = x_enc[global_action[:, 0]]
        v_f = x_enc[global_action[:, 1]]
        w_f = x_enc[global_action[:, 2]]
    uvw_feat = (u_f + v_f + w_f) / 3.0                           # (B, hidden)
    return torch.cat([uvw_feat, z_c], dim=-1)                     # (B, cond_dim)


def _local_undirected_edges(edge_index):
    edges = set()
    for u, v in edge_index.detach().cpu().numpy().T.tolist():
        lu = int(u)
        lv = int(v)
        if lu == lv:
            continue
        a, b = sorted((lu, lv))
        edges.add((a, b))
    return sorted(edges)


def _point_to_segment_distance(points, seg_start, seg_end):
    seg_vec = seg_end - seg_start
    seg_len_sq = torch.sum(seg_vec * seg_vec, dim=-1, keepdim=True).clamp_min(1e-8)
    t = torch.sum((points - seg_start) * seg_vec, dim=-1, keepdim=True) / seg_len_sq
    t = t.clamp(0.0, 1.0)
    projection = seg_start + t * seg_vec
    return torch.norm(points - projection, dim=-1)


def _segment_clearance_distance(seg_a0, seg_a1, seg_b0, seg_b1):
    dists = torch.stack([
        _point_to_segment_distance(seg_a0, seg_b0, seg_b1),
        _point_to_segment_distance(seg_a1, seg_b0, seg_b1),
        _point_to_segment_distance(seg_b0, seg_a0, seg_a1),
        _point_to_segment_distance(seg_b1, seg_a0, seg_a1),
    ], dim=-1)
    return dists.min(dim=-1).values


def compute_geometry_prior_regularizer(pred_geo, base_data, action_topo, cfg, nodes_per_graph=4):
    constraint_cfg = cfg.get('constraints', {})
    min_link_length = float(constraint_cfg.get('min_link_length', 0.05))
    min_node_distance = float(constraint_cfg.get('min_node_distance', 0.01))
    margin_scale = float(cfg.get('il_training', {}).get('geometry_prior_margin_scale', 1.25))
    edge_margin = min_link_length * margin_scale
    node_margin = min_node_distance * margin_scale

    graph_slices = _batch_graph_slices(base_data)
    edge_index = base_data.edge_index
    penalties = []

    for graph_idx, (start, end) in enumerate(graph_slices):
        local_pos = base_data.pos[start:end]
        edge_mask = (
            (edge_index[0] >= start)
            & (edge_index[0] < end)
            & (edge_index[1] >= start)
            & (edge_index[1] < end)
        )
        local_edge_index = edge_index[:, edge_mask] - start

        u_idx = int(action_topo[graph_idx, 0].item())
        v_idx = int(action_topo[graph_idx, 1].item())
        w_idx = int(action_topo[graph_idx, 2].item())
        n1 = pred_geo[graph_idx, :2]
        n2 = pred_geo[graph_idx, 2:]
        u_pos = local_pos[u_idx]
        v_pos = local_pos[v_idx]
        w_pos = local_pos[w_idx]

        link_lengths = torch.stack([
            torch.norm(n1 - u_pos, dim=-1),
            torch.norm(n1 - v_pos, dim=-1),
            torch.norm(n2 - w_pos, dim=-1),
            torch.norm(n2 - n1, dim=-1),
        ], dim=-1)
        short_edge_penalty = F.softplus(edge_margin - link_lengths).mean()

        node_mask = torch.ones(local_pos.size(0), dtype=torch.bool, device=pred_geo.device)
        node_mask[u_idx] = False
        node_mask[v_idx] = False
        node_mask[w_idx] = False
        dist_n1 = torch.norm(local_pos - n1.unsqueeze(0), dim=-1)
        dist_n2 = torch.norm(local_pos - n2.unsqueeze(0), dim=-1)
        node_penalty = pred_geo.new_tensor(0.0)
        if node_mask.any():
            node_penalty = (
                F.softplus(node_margin - dist_n1[node_mask]).mean()
                + F.softplus(node_margin - dist_n2[node_mask]).mean()
            )

        edge_penalties = []
        local_edges = _local_undirected_edges(local_edge_index)
        candidate_segments = [
            ((u_idx, "n1"), u_pos, n1),
            ((v_idx, "n1"), v_pos, n1),
            (("n1", "n2"), n1, n2),
            ((w_idx, "n2"), w_pos, n2),
        ]
        for seg_nodes, seg_a, seg_b in candidate_segments:
            seg_anchor_ids = [node for node in seg_nodes if isinstance(node, int)]
            for e_u, e_v in local_edges:
                if any(node in (e_u, e_v) for node in seg_anchor_ids):
                    continue
                edge_a = local_pos[e_u]
                edge_b = local_pos[e_v]
                clearance = _segment_clearance_distance(
                    seg_a.unsqueeze(0),
                    seg_b.unsqueeze(0),
                    edge_a.unsqueeze(0),
                    edge_b.unsqueeze(0),
                )
                edge_penalties.append(F.softplus(node_margin - clearance).mean())

        for idx_a, idx_b in ((0, 3), (1, 3)):
            _, seg_a0, seg_a1 = candidate_segments[idx_a]
            _, seg_b0, seg_b1 = candidate_segments[idx_b]
            clearance = _segment_clearance_distance(
                seg_a0.unsqueeze(0),
                seg_a1.unsqueeze(0),
                seg_b0.unsqueeze(0),
                seg_b1.unsqueeze(0),
            )
            edge_penalties.append(F.softplus(node_margin - clearance).mean())

        edge_penalty = torch.stack(edge_penalties).mean() if edge_penalties else pred_geo.new_tensor(0.0)
        penalties.append(short_edge_penalty + node_penalty + edge_penalty)

    return torch.stack(penalties).mean() if penalties else pred_geo.new_tensor(0.0)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鎵归噺楠岃瘉锛堟棤姊害锛?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def eval_il_epoch(policy, curve_encoder, dataloader, device, cfg):
    policy.eval()
    curve_encoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            base_data   = batch['base_data'].to(device)
            action_topo = batch['action_topo'].to(device)
            action_geo  = batch['action_geo'].to(device)
            y_foot  = batch['y_foot'].to(device)
            y_knee  = batch['y_knee'].to(device)
            y_ankle = batch['y_ankle'].to(device)

            z_c = curve_encoder(y_foot, y_knee, y_ankle)
            x_enc = policy.encode_graph(base_data)
            topo_scores = policy.topology_scores(x_enc)

            cond = _build_geo_conditions(x_enc, action_topo, z_c, base_data=base_data)
            geo_post_pred, geo_mu, geo_logvar = policy.geo_head(action_geo, cond)
            geo_prior_pred = policy.geo_head.prior_mean(cond)
            geo_reg_post = compute_geometry_prior_regularizer(
                geo_post_pred, base_data, action_topo, cfg,
            )
            geo_reg_prior = compute_geometry_prior_regularizer(
                geo_prior_pred, base_data, action_topo, cfg,
            )

            metrics = compute_il_metrics_batched(
                topo_scores, action_topo,
                geo_post_pred, geo_mu, geo_logvar, action_geo, cfg,
                base_data=base_data,
                geo_prior_pred=geo_prior_pred,
                geo_prior_regularizer_post=geo_reg_post,
                geo_prior_regularizer_prior=geo_reg_prior,
            )
            total_loss += metrics['total_prior'].item()

    return total_loss / max(len(dataloader), 1)



# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鎵归噺璁粌锛堝叏鍚戦噺鍖栵紝鏃?Python sample loop锛?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def train_il_epoch(policy, curve_encoder, optimizer, dataloader, device, cfg,
                   all_params=None):
    """Train one IL epoch using both posterior reconstruction and prior regularization."""
    policy.train()
    curve_encoder.train()
    total_loss = 0.0

    if all_params is None:
        all_params = list(policy.parameters()) + list(curve_encoder.parameters())

    for batch in dataloader:
        base_data   = batch['base_data'].to(device, non_blocking=True)
        action_topo = batch['action_topo'].to(device, non_blocking=True)
        action_geo  = batch['action_geo'].to(device, non_blocking=True)
        y_foot  = batch['y_foot'].to(device, non_blocking=True)
        y_knee  = batch['y_knee'].to(device, non_blocking=True)
        y_ankle = batch['y_ankle'].to(device, non_blocking=True)

        z_c = curve_encoder(y_foot, y_knee, y_ankle)
        x_enc = policy.encode_graph(base_data)
        topo_scores = policy.topology_scores(x_enc)

        cond = _build_geo_conditions(x_enc, action_topo, z_c, base_data=base_data)
        geo_post_pred, geo_mu, geo_logvar = policy.geo_head(action_geo, cond)
        geo_prior_pred = policy.geo_head.prior_mean(cond)
        geo_reg_post = compute_geometry_prior_regularizer(
            geo_post_pred, base_data, action_topo, cfg,
        )
        geo_reg_prior = compute_geometry_prior_regularizer(
            geo_prior_pred, base_data, action_topo, cfg,
        )

        metrics = compute_il_metrics_batched(
            topo_scores, action_topo,
            geo_post_pred, geo_mu, geo_logvar, action_geo, cfg,
            base_data=base_data,
            geo_prior_pred=geo_prior_pred,
            geo_prior_regularizer_post=geo_reg_post,
            geo_prior_regularizer_prior=geo_reg_prior,
        )

        optimizer.zero_grad(set_to_none=True)
        metrics['total'].backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        total_loss += metrics['total'].item()

    return total_loss / max(len(dataloader), 1)

