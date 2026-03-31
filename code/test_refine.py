"""Standalone test: before/after with Gaussian-smoothed surrogate output."""
import os
import torch
import yaml
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN
from src.inverse.gnn_policy import GNNPolicy
from src.inverse.curve_encoder import CurveEncoder
from src.inverse.rl_env import apply_j_operator
from src.inverse.rl_agent import PPOAgent
from rl_refine_bio import refine_coordinates, gaussian_smooth_1d

# Load surrogate
with open('src/config_model_bio.yaml', 'r', encoding='utf-8') as f:
    cfg_bio = yaml.safe_load(f)
device = torch.device('cuda')
surrogate = BioKinematicsGNN(cfg_bio).to(device)
surrogate.load_state_dict(torch.load('model_bio_best.pt', map_location=device, weights_only=True))
surrogate.eval()
for p in surrogate.parameters():
    p.requires_grad_(False)

# Load policy
with open('src/config_inverse.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
enc_cfg = cfg['curve_encoder']
curve_encoder = CurveEncoder(input_dim=enc_cfg['input_dim'], hidden_dims=enc_cfg['hidden_dims'], latent_dim=enc_cfg['latent_dim']).to(device)
policy = GNNPolicy(cfg).to(device)
ckpt = torch.load('model_inverse_il.pt', map_location=device, weights_only=False)
policy.load_state_dict(ckpt['policy'], strict=False)
curve_encoder.load_state_dict(ckpt['curve_encoder'], strict=False)
policy.eval(); curve_encoder.eval()

# Test sample
il_paths = torch.load('f:/LINKS4Meta/LINKS-main/il_expert_paths_80k.pt', weights_only=False)
test_sample = il_paths[-1]
target = {k: test_sample[k] for k in ['y_foot', 'y_knee', 'y_ankle']}
base_graph = test_sample['base_data']

# Generate initial 6-bar
with torch.no_grad():
    z_c = curve_encoder(target['y_foot'].unsqueeze(0).to(device), target['y_knee'].unsqueeze(0).to(device), target['y_ankle'].unsqueeze(0).to(device))
    agent = PPOAgent(policy, curve_encoder, cfg, device)
    actions, _, _ = agent.batch_select_actions([base_graph], z_c, deterministic=True)
    action = actions[0]
    initial_graph = apply_j_operator(base_graph, action['u'], action['v'], action['w'], action['n1'], action['n2'])
    
    # BEFORE (raw + smoothed)
    batch_before = Batch.from_data_list([initial_graph]).to(device)
    pf_b, pk_b, pa_b = surrogate(batch_before)
    pf_bs = gaussian_smooth_1d(pf_b.squeeze(0)).cpu().numpy()
    pk_bs = gaussian_smooth_1d(pk_b.squeeze(0)).cpu().numpy()
    pa_bs = gaussian_smooth_1d(pa_b.squeeze(0)).cpu().numpy()

# AFTER: optimize + smooth
print("Optimizing ALL moving-node coordinates (500 iters) + Gaussian smoothing...")
torch.set_grad_enabled(True)
refined_graph = refine_coordinates(initial_graph, target, surrogate, device, n_iters=500, lr=0.005)
torch.set_grad_enabled(False)

with torch.no_grad():
    batch_after = Batch.from_data_list([refined_graph]).to(device)
    pf_a, pk_a, pa_a = surrogate(batch_after)
    pf_as = gaussian_smooth_1d(pf_a.squeeze(0)).cpu().numpy()
    pk_as = gaussian_smooth_1d(pk_a.squeeze(0)).cpu().numpy()
    pa_as = gaussian_smooth_1d(pa_a.squeeze(0)).cpu().numpy()

# MSEs (against smooth predictions)
y_f, y_k, y_a = target['y_foot'].numpy(), target['y_knee'].numpy(), target['y_ankle'].numpy()
mse_f_b = np.mean((pf_bs - y_f)**2)
mse_k_b = np.mean((pk_bs - y_k)**2)
mse_a_b = np.mean((pa_bs - y_a)**2)
mse_f_a = np.mean((pf_as - y_f)**2)
mse_k_a = np.mean((pk_as - y_k)**2)
mse_a_a = np.mean((pa_as - y_a)**2)

print(f"BEFORE (smooth) - Foot: {mse_f_b:.6f}, Knee: {mse_k_b:.6f}, Ankle: {mse_a_b:.6f}")
print(f"AFTER  (smooth) - Foot: {mse_f_a:.6f}, Knee: {mse_k_a:.6f}, Ankle: {mse_a_a:.6f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0,0].plot(y_f[:,0], y_f[:,1], 'b--', lw=2, label='Target')
axes[0,0].plot(pf_bs[:,0], pf_bs[:,1], 'r-', lw=1.5, label='Before (smooth)')
axes[0,0].set_title(f'BEFORE Foot (MSE={mse_f_b:.4f})'); axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(y_k, 'b--', lw=2, label='Target')
axes[0,1].plot(pk_bs, 'r-', lw=1.5, label='Before (smooth)')
axes[0,1].set_title(f'BEFORE Knee (MSE={mse_k_b:.4f})'); axes[0,1].legend(); axes[0,1].grid(True)

axes[0,2].plot(y_a, 'b--', lw=2, label='Target')
axes[0,2].plot(pa_bs, 'r-', lw=1.5, label='Before (smooth)')
axes[0,2].set_title(f'BEFORE Ankle (MSE={mse_a_b:.4f})'); axes[0,2].legend(); axes[0,2].grid(True)

axes[1,0].plot(y_f[:,0], y_f[:,1], 'b--', lw=2, label='Target')
axes[1,0].plot(pf_as[:,0], pf_as[:,1], 'r-', lw=1.5, label='After (smooth)')
axes[1,0].set_title(f'AFTER Foot (MSE={mse_f_a:.4f})'); axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].plot(y_k, 'b--', lw=2, label='Target')
axes[1,1].plot(pk_as, 'r-', lw=1.5, label='After (smooth)')
axes[1,1].set_title(f'AFTER Knee (MSE={mse_k_a:.4f})'); axes[1,1].legend(); axes[1,1].grid(True)

axes[1,2].plot(y_a, 'b--', lw=2, label='Target')
axes[1,2].plot(pa_as, 'r-', lw=1.5, label='After (smooth)')
axes[1,2].set_title(f'AFTER Ankle (MSE={mse_a_a:.4f})'); axes[1,2].legend(); axes[1,2].grid(True)

plt.tight_layout()
output_path = 'demo/outputs/rl/before_after_refinement.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150)
print(f"Saved to {output_path}")
