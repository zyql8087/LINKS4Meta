import argparse
import copy
import json
import os
import time
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch_geometric.data import Batch, Data

from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy, policy_load_incompatibilities
from src.inverse.rl_agent import PPOAgent
from src.inverse.train_il import ensure_expert_paths
from src.inverse.rl_env import (
    MechanismEnv, apply_j_operator, load_frozen_surrogate, batch_compute_rewards,
)


def gaussian_smooth_1d(x, kernel_size=11, sigma=2.0):
    """
    Differentiable 1D Gaussian smoothing for removing MLP output jitter.
    Works on tensors of shape (T,) or (T, C).
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
        squeeze_back = True
    else:
        squeeze_back = False
    
    T, C = x.shape
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)  # (1, 1, K)
    
    # Process each channel independently
    results = []
    for c in range(C):
        ch = x[:, c].unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        ch_pad = torch.nn.functional.pad(ch, (half, half), mode='reflect')
        smoothed = torch.nn.functional.conv1d(ch_pad, kernel)  # (1, 1, T)
        results.append(smoothed.squeeze(0).squeeze(0))  # (T,)
    
    out = torch.stack(results, dim=1)  # (T, C)
    return out.squeeze(1) if squeeze_back else out


def refine_coordinates(graph, target, surrogate, device,
                       n_iters=500, lr=0.005):
    """
    Directly optimize ALL moving-node coordinates by backpropagating
    through the differentiable Surrogate model.
    
    Applies Gaussian smoothing to Surrogate output to remove MLP jitter.
    """
    graph = graph.to(device)
    n_nodes = graph.x.size(0)
    if n_nodes <= 4:
        return graph

    # Identify moving vs fixed nodes
    is_fixed = graph.x[:, 2]
    moving_mask = (is_fixed == 0)
    fixed_mask = (is_fixed == 1)
    
    if moving_mask.sum() == 0:
        return graph
    
    moving_coords = graph.x[moving_mask, :2].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([moving_coords], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=lr * 0.01)
    
    edge_index = graph.edge_index
    keypoints = graph.keypoints if hasattr(graph, 'keypoints') else None
    other_feats = graph.x[:, 2:].detach()
    fixed_coords = graph.x[fixed_mask, :2].detach()
    
    y_foot = target['y_foot'].to(device)
    y_knee = target['y_knee'].to(device)
    y_ankle = target['y_ankle'].to(device)
    
    best_loss = float('inf')
    best_moving = moving_coords.clone().detach()
    
    for i in range(n_iters):
        optimizer.zero_grad()
        
        all_coords = torch.zeros(n_nodes, 2, device=device)
        all_coords[moving_mask] = moving_coords
        all_coords[fixed_mask] = fixed_coords
        
        x_new = torch.cat([all_coords, other_feats], dim=1)
        
        from torch_geometric.data import Data as TGData
        diff_graph = TGData(
            x=x_new, pos=all_coords,
            edge_index=edge_index,
            keypoints=keypoints,
        )
        diff_graph.batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
        
        row, col = edge_index
        edge_vec = all_coords[col] - all_coords[row]
        diff_graph.edge_attr = torch.norm(edge_vec, dim=-1, keepdim=True)
        
        pred_foot, pred_knee, pred_ankle = surrogate(diff_graph)
        
        # Smooth surrogate outputs to remove MLP jitter
        sf = gaussian_smooth_1d(pred_foot.squeeze(0))    # (200, 2)
        sk = gaussian_smooth_1d(pred_knee.squeeze(0))     # (200,)
        sa = gaussian_smooth_1d(pred_ankle.squeeze(0))     # (200,)
        
        loss_foot = torch.nn.functional.mse_loss(sf, y_foot)
        loss_knee = torch.nn.functional.mse_loss(sk, y_knee)
        loss_ankle = torch.nn.functional.mse_loss(sa, y_ankle)
        loss = loss_foot + 0.5 * loss_knee + 0.1 * loss_ankle
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_moving = moving_coords.clone().detach()
    
    # Build final graph
    final_x = graph.x.clone()
    final_x[moving_mask, :2] = best_moving
    final_pos = graph.pos.clone()
    final_pos[moving_mask] = best_moving
    
    return Data(
        x=final_x.cpu(), pos=final_pos.cpu(),
        edge_index=graph.edge_index.cpu(),
        keypoints=keypoints.cpu() if keypoints is not None else None,
    )


def sample_episode_from_dataset(il_paths, idx=None):
    if idx is None:
        idx = np.random.randint(len(il_paths))
    sample = il_paths[idx]
    target = {
        'y_foot': sample['y_foot'],
        'y_knee': sample['y_knee'],
        'y_ankle': sample['y_ankle'],
    }
    return target, sample['base_data']


def build_base_4bar() -> Data:
    x0 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    is_fixed = np.array([1, 0, 0, 1], dtype=np.float32)
    is_grounded = np.array([1, 0, 0, 0], dtype=np.float32)
    x_feat = np.column_stack([x0, is_fixed, is_grounded])
    edges = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 0], [0, 3]]
    return Data(
        x=torch.tensor(x_feat, dtype=torch.float32),
        pos=torch.tensor(x0, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).T,
        knee_idx=torch.tensor([2], dtype=torch.long),
    )


def load_inverse_checkpoint(policy, curve_encoder, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_state = policy.load_state_dict(ckpt['policy'], strict=False)
    missing_keys, unexpected_keys = policy_load_incompatibilities(policy_state)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Policy checkpoint is incompatible with current config. "
            f"Missing={missing_keys}, unexpected={unexpected_keys}"
        )
    encoder_state = curve_encoder.load_state_dict(ckpt['curve_encoder'], strict=False)
    if encoder_state.missing_keys or encoder_state.unexpected_keys:
        print(
            f"[RL] Warning: curve encoder checkpoint has key drift. "
            f"Missing={encoder_state.missing_keys}, unexpected={encoder_state.unexpected_keys}"
        )


def plot_kinematics_result(y_foot, y_knee, y_ankle, 
                           pred_foot, pred_knee, pred_ankle, 
                           ep, reward, save_dir='demo/outputs/rl'):
    """Plot target vs generated kinematics for periodic RL evaluation."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    axes[0].plot(y_foot[:, 0], y_foot[:, 1], label='Target', color='blue', linestyle='dashed', alpha=0.7)
    axes[0].plot(pred_foot[:, 0], pred_foot[:, 1], label='Generated', color='red', alpha=0.7)
    axes[0].set_title(f'Foot Trajectory (Ep {ep}, Rw {reward:.2f})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True)
    
    steps = range(len(y_knee))
    axes[1].plot(steps, y_foot[:, 0], label='Target X', color='lightblue', linestyle='dashed')
    axes[1].plot(steps, y_foot[:, 1], label='Target Y', color='blue', linestyle='dashed')
    axes[1].plot(steps, pred_foot[:, 0], label='Generated X', color='lightcoral')
    axes[1].plot(steps, pred_foot[:, 1], label='Generated Y', color='red')
    axes[1].set_title('Foot Position vs Steps')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(steps, y_knee, label='Target', color='blue', linestyle='dashed', alpha=0.7)
    axes[2].plot(steps, pred_knee, label='Generated', color='red', alpha=0.7)
    axes[2].set_title('Knee Angle')
    axes[2].legend()
    axes[2].grid(True)
    
    axes[3].plot(steps, y_ankle, label='Target', color='blue', linestyle='dashed', alpha=0.7)
    axes[3].plot(steps, pred_ankle, label='Generated', color='red', alpha=0.7)
    axes[3].set_title('Ankle Angle')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'eval_ep_{ep:05d}_rw_{reward:.2f}.png'), dpi=150)
    plt.close()


def save_reward_curve(episode_rewards, reward_curve_path, window=50):
    os.makedirs(os.path.dirname(reward_curve_path), exist_ok=True)
    reward_history_path = os.path.splitext(reward_curve_path)[0] + '.json'

    with open(reward_history_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'num_episodes': len(episode_rewards),
                'rewards': [float(r) for r in episode_rewards],
                'moving_average_window': int(window),
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(12, 5))
    num_points = len(episode_rewards)

    if num_points == 0:
        ax.text(
            0.5, 0.5,
            'No reward data recorded',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title('Inverse Model RL Training Reward Curve')
    elif num_points == 1:
        ax.plot([0], episode_rewards, marker='o', markersize=8, linewidth=1.5, label='Episode Reward')
        ax.annotate(
            f'{episode_rewards[0]:.4f}',
            (0, episode_rewards[0]),
            textcoords='offset points',
            xytext=(8, 8),
        )
        ax.set_xlim(-0.5, 0.5)
        ax.set_title('Inverse Model RL Training Reward Curve (single-episode run)')
    else:
        x = np.arange(num_points)
        ax.plot(x, episode_rewards, alpha=0.45, linewidth=1.5, marker='o', markersize=3, label='Episode Reward')
        if num_points >= window:
            ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
            ax.plot(
                np.arange(window - 1, num_points),
                ma,
                'r-',
                linewidth=2,
                label=f'MA-{window}',
            )
        ax.set_title(f'Inverse Model RL Training Reward Curve ({num_points} episodes)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.tight_layout()
    fig.savefig(reward_curve_path, dpi=150)
    plt.close(fig)
    return reward_history_path


def save_rl_summary(
    summary_path,
    episode_rewards,
    none_action_total,
    action_total,
    geometry_prior_reject_total,
    structure_reject_total,
    geometry_prior_reason_total,
    structure_reason_total,
):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    rewards = [float(r) for r in episode_rewards]
    payload = {
        'num_episodes': len(rewards),
        'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
        'min_reward': float(np.min(rewards)) if rewards else 0.0,
        'max_reward': float(np.max(rewards)) if rewards else 0.0,
        'none_action_total': int(none_action_total),
        'action_total': int(action_total),
        'none_action_rate': float(none_action_total / action_total) if action_total > 0 else 0.0,
        'geometry_prior_reject_total': int(geometry_prior_reject_total),
        'structure_reject_total': int(structure_reject_total),
        'geometry_prior_reasons': dict(geometry_prior_reason_total),
        'structure_reasons': dict(structure_reason_total),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return summary_path


@torch.no_grad()
def evaluate_and_visualize(agent, surrogate, test_sample, ep, reward, device, max_steps):
    """Run evaluation on a fixed test sample, refine coordinates, and save kinematics plot."""
    target = {
        'y_foot': test_sample['y_foot'],
        'y_knee': test_sample['y_knee'],
        'y_ankle': test_sample['y_ankle']
    }
    base_graph = copy.deepcopy(test_sample['base_data'])
    
    foot_t = target['y_foot'].unsqueeze(0).to(device)
    knee_t = target['y_knee'].unsqueeze(0).to(device)
    ankle_t = target['y_ankle'].unsqueeze(0).to(device)
    z_c = agent.curve_encoder(foot_t, knee_t, ankle_t)
    
    current_graph = base_graph
    for step in range(max_steps):
        actions, _, _ = agent.batch_select_actions([current_graph], z_c, deterministic=True)
        action = actions[0]
        if action is None:
            break
        current_graph = apply_j_operator(
            current_graph, action['u'], action['v'], action['w'], action['n1'], action['n2']
        )
    
    # === Direct coordinate optimization through the differentiable Surrogate ===
    torch.set_grad_enabled(True)
    current_graph = refine_coordinates(current_graph, target, surrogate, device)
    torch.set_grad_enabled(False)
    
    batch_eval = Batch.from_data_list([current_graph]).to(device)
    pred_foot, pred_knee, pred_ankle = surrogate(batch_eval)
    
    # Smooth the final output for clean visualization
    sf = gaussian_smooth_1d(pred_foot.squeeze(0)).cpu().numpy()
    sk = gaussian_smooth_1d(pred_knee.squeeze(0)).cpu().numpy()
    sa = gaussian_smooth_1d(pred_ankle.squeeze(0)).cpu().numpy()
    
    plot_kinematics_result(
        target['y_foot'].numpy(), target['y_knee'].numpy(), target['y_ankle'].numpy(),
        sf, sk, sa, ep, reward
    )


def main():
    parser = argparse.ArgumentParser(description='Inverse-model RL refinement (GPU-optimized)')
    parser.add_argument('--config', type=str, default='src/config_inverse.yaml')
    parser.add_argument('--il_model', type=str, default='model_inverse_il.pt')
    parser.add_argument('--n_episodes', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    rl_cfg = cfg['rl_training']
    paths_cfg = cfg['paths']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_episodes = args.n_episodes or rl_cfg.get('episodes', 2000)
    max_steps = rl_cfg.get('steps_per_episode', 5)
    print(f"[RL] Device: {device} | Episodes: {n_episodes} | Max steps/ep: {max_steps}")
    if n_episodes <= 1:
        print('[RL] Warning: only one episode configured; reward curve will contain a single point.')

    enc_cfg = cfg['curve_encoder']
    curve_encoder = CurveEncoder(
        input_dim=enc_cfg['input_dim'],
        hidden_dims=enc_cfg['hidden_dims'],
        latent_dim=enc_cfg['latent_dim'],
    ).to(device)
    policy = GNNPolicy(cfg).to(device)

    if os.path.exists(args.il_model):
        load_inverse_checkpoint(policy, curve_encoder, args.il_model, device)
        print(f"[RL] IL weights loaded from '{args.il_model}'")
    else:
        print(f"[RL] WARNING: IL model '{args.il_model}' not found. Using random init.")

    surrogate, _ = load_frozen_surrogate(
        model_path=paths_cfg['forward_model'],
        config_path=paths_cfg['config_forward'],
        device=device,
    )

    agent = PPOAgent(policy, curve_encoder, cfg, device)

    il_pt_path = paths_cfg.get('il_dataset_output', '')
    if il_pt_path:
        print(f"[RL] Loading IL expert paths from '{il_pt_path}'...")
        il_paths = ensure_expert_paths(
            pkl_path=paths_cfg['pkl_dataset'],
            output_path=il_pt_path,
            use_cached=True,
        )
        print(f"[RL] Loaded {len(il_paths)} expert paths.")
        test_sample = il_paths[-1]  # Fixed evaluation sample from test holdout
        il_paths = il_paths[:-50]   # Exclude test end from training
    else:
        print('[RL] No IL dataset found. Falling back to fixed 4-bar base graph.')
        il_paths = None
        test_sample = None

    episode_rewards = []
    best_reward = float('-inf')
    start_time = time.time()
    reward_cfg = cfg['reward']
    no_action_penalty = float(reward_cfg.get('penalty_locking', -100.0))
    none_action_total = 0
    action_total = 0
    geometry_prior_reject_total = 0
    structure_reject_total = 0
    geometry_prior_reason_total = Counter()
    structure_reason_total = Counter()

    N_ROLLOUTS_PER_UPDATE = 32

    print(f"\n[RL] Starting PPO training (batched, no MCTS): {n_episodes} episodes | "
          f"{N_ROLLOUTS_PER_UPDATE} rollouts/update | {max_steps} steps/ep ...")

    ep = 0
    policy.eval()
    curve_encoder.eval()

    while ep < n_episodes:
        n_this_batch = min(N_ROLLOUTS_PER_UPDATE, n_episodes - ep)

        # ══════════════════════════════════════════════════════════════════
        # Phase 0: Sample targets & pre-batch curve encoding (1 GPU call)
        # ══════════════════════════════════════════════════════════════════
        targets = []
        base_graphs = []
        for _ in range(n_this_batch):
            if il_paths is not None:
                target, base_graph = sample_episode_from_dataset(il_paths)
            else:
                base_graph = build_base_4bar()
                target = {
                    'y_foot': torch.zeros(200, 2),
                    'y_knee': torch.zeros(200),
                    'y_ankle': torch.zeros(200),
                }
            targets.append(target)
            base_graphs.append(base_graph)

        with torch.no_grad():
            foot_b = torch.stack([t['y_foot'] for t in targets]).to(device)
            knee_b = torch.stack([t['y_knee'] for t in targets]).to(device)
            ankle_b = torch.stack([t['y_ankle'] for t in targets]).to(device)
            z_c_batch = curve_encoder(foot_b, knee_b, ankle_b)  # (N, 128)

        # ══════════════════════════════════════════════════════════════════
        # Phase 1: Parallel vectorized rollout collection
        #   All N episodes step simultaneously, 1 GPU call per step
        # ══════════════════════════════════════════════════════════════════
        agent.buffer.clear()

        # Initialize all environments
        current_graphs = [copy.deepcopy(g) for g in base_graphs]
        alive = [True] * n_this_batch  # track which episodes aren't done
        ep_rewards = [0.0] * n_this_batch
        # Track step graphs per episode for batch reward
        ep_step_graphs = [[] for _ in range(n_this_batch)]

        for step in range(max_steps):
            # Collect graphs of alive episodes
            alive_idx = [i for i in range(n_this_batch) if alive[i]]
            if not alive_idx:
                break

            alive_graphs = [current_graphs[i] for i in alive_idx]
            alive_z_cs = z_c_batch[alive_idx]  # (K, 128)

            # ── ONE GPU call: batch action selection ────────────────────
            actions, log_probs, values, action_diags = agent.batch_select_actions(
                alive_graphs, alive_z_cs, return_diagnostics=True
            )

            batch_none_actions = sum(action is None for action in actions)
            none_action_total += batch_none_actions
            action_total += len(actions)
            for diag in action_diags:
                geometry_prior_reject_total += int(diag.get('geometry_prior_rejects', 0))
                structure_reject_total += int(diag.get('structure_rejects', 0))
                geometry_prior_reason_total.update(diag.get('geometry_prior_reasons', {}))
                structure_reason_total.update(diag.get('structure_reasons', {}))

            # ── Apply actions (CPU only, fast) ─────────────────────────
            for j, gi in enumerate(alive_idx):
                action = actions[j]
                if action is None:
                    alive[gi] = False
                    continue

                new_graph = apply_j_operator(
                    current_graphs[gi],
                    action['u'], action['v'], action['w'],
                    action['n1'], action['n2'],
                )
                current_graphs[gi] = new_graph
                ep_step_graphs[gi].append(copy.deepcopy(new_graph))

                done = (step == max_steps - 1)

                # Store with placeholder reward (to be filled later)
                agent.buffer.store(
                    alive_graphs[j],  # state before action
                    z_c_batch[gi:gi+1],
                    action,
                    0.0,  # reward placeholder
                    log_probs[j],
                    values[j],
                    done,
                )

        # ══════════════════════════════════════════════════════════════════
        # Phase 1b: Batch compute ALL rewards in one GPU call
        # ══════════════════════════════════════════════════════════════════
        all_step_graphs = []
        graph_to_ep_step = []  # (episode_idx, step_idx) mapping
        for gi in range(n_this_batch):
            for si, g in enumerate(ep_step_graphs[gi]):
                all_step_graphs.append(g)
                graph_to_ep_step.append((gi, si))

        if all_step_graphs:
            # ONE big batch surrogate call for ALL graphs across ALL episodes
            all_rewards = batch_compute_rewards(
                surrogate, all_step_graphs, targets[0], reward_cfg, device,
                constraint_cfg=cfg.get('constraints', {}),
            )

            # But each episode has its own target — need per-target rewards
            # Recompute properly: group by episode target
            reward_idx = 0
            buf_idx = 0
            for gi in range(n_this_batch):
                n_steps_this_ep = len(ep_step_graphs[gi])
                if n_steps_this_ep == 0:
                    ep_rewards[gi] = no_action_penalty
                    continue

                # Batch reward for this episode's graphs with its own target
                ep_rewards_list = batch_compute_rewards(
                    surrogate, ep_step_graphs[gi], targets[gi], reward_cfg, device,
                    constraint_cfg=cfg.get('constraints', {}),
                )

                ep_reward = 0.0
                for si in range(n_steps_this_ep):
                    r, valid = ep_rewards_list[si]
                    if not valid:
                        r = reward_cfg.get('penalty_locking', -100.0)
                    agent.buffer.rewards[buf_idx] = r
                    ep_reward += r
                    buf_idx += 1

                ep_rewards[gi] = ep_reward

        ep += n_this_batch
        episode_rewards.extend(ep_rewards)

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: PPO update (single large batch on GPU)
        # ══════════════════════════════════════════════════════════════════
        policy.train()
        curve_encoder.train()
        agent.update(agent.buffer, n_epochs=4)
        policy.eval()
        curve_encoder.eval()

        if ep % 50 == 0 or ep <= N_ROLLOUTS_PER_UPDATE:
            mean_rw = (np.mean(episode_rewards[-50:])
                       if len(episode_rewards) >= 50
                       else np.mean(episode_rewards))
            elapsed = (time.time() - start_time) / 60
            print(
                f"Episode [{ep:5d}/{n_episodes}]  Mean Reward (last 50): {mean_rw:.4f}  "
                f"Time: {elapsed:.1f} min  Buffer: {len(agent.buffer.states)} samples"
            )
            none_rate = (none_action_total / action_total) if action_total > 0 else 0.0
            top_prior = geometry_prior_reason_total.most_common(2)
            top_structure = structure_reason_total.most_common(2)
            print(
                f"  [action] None rate: {none_rate:.2%} ({none_action_total}/{action_total})  "
                f"Prior rejects: {geometry_prior_reject_total}  Structure rejects: {structure_reject_total}"
            )
            if top_prior:
                print(f"  [action] Top geometry-prior rejects: {top_prior}")
            if top_structure:
                print(f"  [action] Top structure rejects: {top_structure}")
            if mean_rw > best_reward:
                best_reward = mean_rw
                torch.save(
                    {
                        'policy': policy.state_dict(),
                        'curve_encoder': curve_encoder.state_dict(),
                        'critic': agent.critic.state_dict(),
                    },
                    paths_cfg.get('rl_model_output', 'model_inverse_rl.pt'),
                )
                print(f"  [save] Best RL model saved (mean_reward={best_reward:.4f})")
                
                # Evaluation hooks (visualize the newly found best model)
                if test_sample is not None and ep >= 0:
                    evaluate_and_visualize(agent, surrogate, test_sample, ep, best_reward, device, max_steps)
                    print(f"  [eval] Saved kinematics visualization for best ep {ep}.")

    reward_curve_path = os.path.join('demo', 'outputs', 'rl', 'rl_reward_curve.png')
    reward_history_path = save_reward_curve(episode_rewards, reward_curve_path, window=50)
    rl_summary_path = os.path.join('demo', 'outputs', 'rl', 'rl_summary.json')
    save_rl_summary(
        rl_summary_path,
        episode_rewards,
        none_action_total,
        action_total,
        geometry_prior_reject_total,
        structure_reject_total,
        geometry_prior_reason_total,
        structure_reason_total,
    )
    print(f"\n[RL] Training complete. Reward curve saved to '{reward_curve_path}'")
    print(f"[RL] Reward history saved to '{reward_history_path}'")
    print(f"[RL] RL summary saved to '{rl_summary_path}'")
    print(f"[RL] Best model path: {paths_cfg.get('rl_model_output', 'model_inverse_rl.pt')}")


if __name__ == '__main__':
    main()
