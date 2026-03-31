# train_inverse_bio.py  (PreBatchedLoader GPU-optimized version)
# 閫嗗悜妯″瀷 IL 棰勮缁冧富鍏ュ彛
#
# GPU 鍔犻€熺瓥鐣ワ細
#   - PreBatchedLoader: 璁粌寮€濮嬪墠锛屽皢鍏ㄩ儴 80k 鏍锋湰涓€娆℃€ф壒娆″寲骞跺瓨鍏ユ樉瀛?
#   - num_workers=0 (Windows PyG 鍏煎锛岄€氳繃棰勬壒娆″寲寮ヨˉ鏁版嵁鍔犺浇鐡堕)
#   - batch_size=4096: 鎵撴弧 GPU 璁＄畻鍗曞厓锛堝皬鍥惧繀椤荤敤澶?batch锛?
#   - non_blocking=True: 寮傛 CPU鈫扜PU 浼犺緭
#   - set_to_none=True: 鏇村揩鐨勬搴︽竻闆?

import argparse
import json
import os
import random
import yaml
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy, policy_load_incompatibilities
from src.inverse.experiment_utils import load_or_create_fixed_split, subset_by_indices
from src.inverse.train_il import (
    ensure_expert_paths,
    _build_geo_conditions,
    compute_il_metrics_batched,
    compute_geometry_prior_regularizer,
)

TRACKED_IL_METRICS = (
    'total',
    'total_posterior',
    'total_prior',
    'loss_topo',
    'loss_geo',
    'loss_recon',
    'loss_kl',
    'loss_geo_prior',
    'loss_geo_regularizer',
)


def _init_metric_sums():
    return {key: 0.0 for key in TRACKED_IL_METRICS}


def _accumulate_metric_sums(sums, metrics):
    for key in TRACKED_IL_METRICS:
        sums[key] += metrics[key].item()


def _average_metric_sums(sums, num_batches):
    denom = max(num_batches, 1)
    return {key: value / denom for key, value in sums.items()}
def _format_metrics(metrics):
    return "  ".join([
        f"total={metrics['total']:.6f}",
        f"prior={metrics['total_prior']:.6f}",
        f"topo={metrics['loss_topo']:.6f}",
        f"geo_prior={metrics['loss_geo_prior']:.6f}",
    ])


def _to_float_dict(metrics):
    return {key: float(value) for key, value in metrics.items()}



# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# PreBatchedLoader锛氬皢鍏ㄩ儴鏁版嵁涓€娆℃€ф壒娆″寲鍚庢斁鍏?GPU
# 姣?Epoch 鍙亶鍘?list锛屾秷闄?DataLoader collate 寮€閿€
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class PreBatchedLoader:
    """
    灏?ILDataset 閲屽叏閮ㄦ牱鏈鍏堟墦鍖呬负 batched GPU tensors 鍒楄〃銆?
    浼樼偣锛歟poch 鍐呮棤 CPU 鎿嶄綔锛孏PU 鍚炲悙閲?鈮?100%
    缂虹偣锛氭樉瀛樻秷鑰楃害 = batch_size 脳 num_batches 脳 tensor_size
    """
    def __init__(self, dataset, batch_size: int, device, shuffle: bool = True):
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.batches = []

        print(f"  Pre-batching {len(dataset)} samples to {device} "
              f"(batch_size={batch_size}) ...")

        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            chunk = [dataset[indices[i]] for i in range(start, end)]
            if not chunk:
                continue
            base_data_list = [x['base_data'] for x in chunk]
            batch = {
                'base_data':   Batch.from_data_list(base_data_list).to(device),
                'action_topo': torch.stack([x['action_topo'] for x in chunk]).to(device),
                'action_geo':  torch.stack([x['action_geo']  for x in chunk]).to(device),
                'y_foot':      torch.stack([x['y_foot']      for x in chunk]).to(device),
                'y_knee':      torch.stack([x['y_knee']      for x in chunk]).to(device),
                'y_ankle':     torch.stack([x['y_ankle']     for x in chunk]).to(device),
            }
            self.batches.append(batch)

        print(f"  Pre-batching done: {len(self.batches)} batches cached on {device}")

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def train_epoch_prebatched(policy, curve_encoder, optimizer, loader, cfg, all_params):
    policy.train()
    curve_encoder.train()
    metric_sums = _init_metric_sums()
    for batch in loader:
        # Keep optional conditioning noise local to the curve encoder input only.
        noise_std = 0.00
        y_foot_aug  = batch['y_foot']  + torch.randn_like(batch['y_foot'])  * noise_std
        y_knee_aug  = batch['y_knee']  + torch.randn_like(batch['y_knee'])  * noise_std
        y_ankle_aug = batch['y_ankle'] + torch.randn_like(batch['y_ankle']) * noise_std

        z_c = curve_encoder(y_foot_aug, y_knee_aug, y_ankle_aug)
        x_enc = policy.encode_graph(batch['base_data'])
        topo_scores = policy.topology_scores(x_enc)
        cond = _build_geo_conditions(x_enc, batch['action_topo'], z_c)
        geo_post_pred, geo_mu, geo_logvar = policy.geo_head(batch['action_geo'], cond)
        geo_prior_pred = policy.geo_head.prior_mean(cond)
        geo_reg_post = compute_geometry_prior_regularizer(
            geo_post_pred, batch['base_data'], batch['action_topo'], cfg,
        )
        geo_reg_prior = compute_geometry_prior_regularizer(
            geo_prior_pred, batch['base_data'], batch['action_topo'], cfg,
        )
        metrics = compute_il_metrics_batched(
            topo_scores, batch['action_topo'],
            geo_post_pred, geo_mu, geo_logvar, batch['action_geo'], cfg,
            geo_prior_pred=geo_prior_pred,
            geo_prior_regularizer_post=geo_reg_post,
            geo_prior_regularizer_prior=geo_reg_prior,
        )
        optimizer.zero_grad(set_to_none=True)
        metrics['total'].backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        _accumulate_metric_sums(metric_sums, metrics)
    return _average_metric_sums(metric_sums, len(loader))

def eval_epoch_prebatched(policy, curve_encoder, loader, cfg):
    policy.eval()
    curve_encoder.eval()
    metric_sums = _init_metric_sums()
    with torch.no_grad():
        for batch in loader:
            z_c = curve_encoder(batch['y_foot'], batch['y_knee'], batch['y_ankle'])
            x_enc = policy.encode_graph(batch['base_data'])
            topo_scores = policy.topology_scores(x_enc)
            cond = _build_geo_conditions(x_enc, batch['action_topo'], z_c)
            geo_post_pred, geo_mu, geo_logvar = policy.geo_head(batch['action_geo'], cond)
            geo_prior_pred = policy.geo_head.prior_mean(cond)
            geo_reg_post = compute_geometry_prior_regularizer(
                geo_post_pred, batch['base_data'], batch['action_topo'], cfg,
            )
            geo_reg_prior = compute_geometry_prior_regularizer(
                geo_prior_pred, batch['base_data'], batch['action_topo'], cfg,
            )
            metrics = compute_il_metrics_batched(
                topo_scores, batch['action_topo'],
                geo_post_pred, geo_mu, geo_logvar, batch['action_geo'], cfg,
                geo_prior_pred=geo_prior_pred,
                geo_prior_regularizer_post=geo_reg_post,
                geo_prior_regularizer_prior=geo_reg_prior,
            )
            _accumulate_metric_sums(metric_sums, metrics)
    return _average_metric_sums(metric_sums, len(loader))


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 涓诲嚱鏁?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def main():
    parser = argparse.ArgumentParser(description="閫嗗悜妯″瀷 IL 棰勮缁冿紙PreBatched GPU 鐗堬級")
    parser.add_argument('--config', type=str, default='src/config_inverse.yaml')
    parser.add_argument('--skip_extract', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg['paths']
    il_cfg    = cfg['il_training']
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = il_cfg['batch_size']
    print(f"Training IL on device: {device}")

    # 鈹€鈹€ 涓撳璺緞鎻愬彇 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    il_dataset_path = paths_cfg['il_dataset_output']
    expert_paths = ensure_expert_paths(
        pkl_path=paths_cfg['pkl_dataset'],
        output_path=il_dataset_path,
        use_cached=args.skip_extract,
    )
    print(f"[*] Total expert paths: {len(expert_paths)}")

    # 鈹€鈹€ Train / Val 鍒嗗壊 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    split = load_or_create_fixed_split(
        num_samples=len(expert_paths),
        val_ratio=il_cfg.get('val_ratio', 0.1),
        test_ratio=il_cfg.get('test_ratio', 0.1),
        split_seed=il_cfg.get('split_seed', 42),
        split_path=paths_cfg.get('il_split_output', 'artifacts/il_split_indices.pt'),
    )
    train_paths = subset_by_indices(expert_paths, split['train_indices'])
    val_paths = subset_by_indices(expert_paths, split['val_indices'])
    test_paths = subset_by_indices(expert_paths, split['test_indices'])
    print(
        f"[*] Using fixed split from {paths_cfg.get('il_split_output', 'artifacts/il_split_indices.pt')} | "
        f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} seed={split['split_seed']}"
    )

    print("\n[*] Pre-batching data to GPU (one-time cost) ...")
    train_loader = PreBatchedLoader(train_paths, BATCH_SIZE, device, shuffle=True)
    val_loader   = PreBatchedLoader(val_paths,   BATCH_SIZE, device, shuffle=False)
    test_loader  = PreBatchedLoader(test_paths,  BATCH_SIZE, device, shuffle=False)

    # 鈹€鈹€ 妯″瀷鍒濆鍖?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    enc_cfg = cfg['curve_encoder']
    curve_encoder = CurveEncoder(
        input_dim=enc_cfg['input_dim'],
        hidden_dims=enc_cfg['hidden_dims'],
        latent_dim=enc_cfg['latent_dim'],
    ).to(device)

    policy = GNNPolicy(cfg).to(device)

    all_params = list(policy.parameters()) + list(curve_encoder.parameters())
    optimizer  = optim.Adam(all_params, lr=il_cfg['learning_rate']) 
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=il_cfg['epochs'])

    n_params = sum(p.numel() for p in all_params)
    print(f"[*] Model params: {n_params/1e6:.2f}M | "
          f"Batch size: {BATCH_SIZE} | "
          f"Train batches/epoch: {len(train_loader)}")

    # 鈹€鈹€ 璁粌寰幆 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    train_total_losses = []
    train_prior_losses = []
    val_prior_losses = []
    train_topology_losses = []
    val_topology_losses = []
    train_geo_prior_losses = []
    val_geo_prior_losses = []
    best_val_loss = float('inf')
    best_model_path = paths_cfg.get('il_model_output', 'model_inverse_il.pt')

    patience = il_cfg.get('patience', 20)
    patience_counter = 0

    print(f"\n[*] Starting IL pre-training for {il_cfg['epochs']} epochs ...")
    for epoch in range(il_cfg['epochs']):
        train_metrics = train_epoch_prebatched(policy, curve_encoder, optimizer,
                                               train_loader, cfg, all_params)
        val_metrics = eval_epoch_prebatched(policy, curve_encoder, val_loader, cfg)
        scheduler.step()

        train_total_losses.append(train_metrics['total'])
        train_prior_losses.append(train_metrics['total_prior'])
        val_prior_losses.append(val_metrics['total_prior'])
        train_topology_losses.append(train_metrics['loss_topo'])
        val_topology_losses.append(val_metrics['loss_topo'])
        train_geo_prior_losses.append(train_metrics['loss_geo_prior'])
        val_geo_prior_losses.append(val_metrics['loss_geo_prior'])
        print(
            f"Epoch [{epoch+1:3d}/{il_cfg['epochs']}]  "
            f"Train Opt: {train_metrics['total']:.6f}  "
            f"Train Prior: {train_metrics['total_prior']:.6f}  "
            f"Val Prior: {val_metrics['total_prior']:.6f}  "
            f"Train Topo: {train_metrics['loss_topo']:.6f}  "
            f"Val Topo: {val_metrics['loss_topo']:.6f}  "
            f"Train GeoPrior: {train_metrics['loss_geo_prior']:.6f}  "
            f"Val GeoPrior: {val_metrics['loss_geo_prior']:.6f}"
        )

        if val_metrics['total_prior'] < best_val_loss:
            best_val_loss = val_metrics['total_prior']
            patience_counter = 0
            torch.save({
                'policy':        policy.state_dict(),
                'curve_encoder': curve_encoder.state_dict(),
            }, best_model_path)
            print(f"  [OK] Best model saved (val_prior={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [!] Early stopping triggered. No validation improvement for {patience} epochs.")
                break

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_total_losses, label='Train Opt Total')
        plt.plot(train_prior_losses, label='Train Prior Total')
        plt.plot(val_prior_losses, label='Val Prior Total')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('IL Total Losses')
        plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_topology_losses, label='Train Topology')
        plt.plot(val_topology_losses, label='Val Topology')
        plt.plot(train_geo_prior_losses, label='Train Geometry Prior')
        plt.plot(val_geo_prior_losses, label='Val Geometry Prior')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('IL Loss Breakdown')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plot_path = paths_cfg.get('il_plot_output', 'il_loss_curve.png')
        plot_dir = os.path.dirname(plot_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    policy_state = policy.load_state_dict(best_ckpt['policy'], strict=False)
    missing_keys, unexpected_keys = policy_load_incompatibilities(policy_state)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Best IL checkpoint is incompatible with current policy. Missing={missing_keys}, unexpected={unexpected_keys}"
        )
    curve_encoder.load_state_dict(best_ckpt['curve_encoder'])
    test_metrics = eval_epoch_prebatched(policy, curve_encoder, test_loader, cfg)
    print(f"[*] Final test metrics: {_format_metrics(test_metrics)}")

    report = {
        'best_val_prior': float(best_val_loss),
        'train_metrics_last': _to_float_dict(train_metrics),
        'val_metrics_last': _to_float_dict(val_metrics),
        'test_metrics': _to_float_dict(test_metrics),
        'split': {
            'train_size': len(train_paths),
            'val_size': len(val_paths),
            'test_size': len(test_paths),
            'split_seed': split['split_seed'],
            'val_ratio': split['val_ratio'],
            'test_ratio': split['test_ratio'],
            'split_path': paths_cfg.get('il_split_output', 'artifacts/il_split_indices.pt'),
        },
        'history': {
            'train_total': train_total_losses,
            'train_prior': train_prior_losses,
            'val_prior': val_prior_losses,
            'train_topology': train_topology_losses,
            'val_topology': val_topology_losses,
            'train_geo_prior': train_geo_prior_losses,
            'val_geo_prior': val_geo_prior_losses,
        },
    }
    report_path = paths_cfg.get('il_report_output', 'artifacts/il_training_report.json')
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Done. Loss curve saved to '{paths_cfg.get('il_plot_output', 'il_loss_curve.png')}'")
    print(f"[OK] Best weights saved to '{best_model_path}'")
    print(f"[OK] Report saved to '{report_path}'")

if __name__ == '__main__':
    main()

