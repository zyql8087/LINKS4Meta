# src/generative_curve/GNN_train_biokinematics.py

from __future__ import annotations

import torch
from tqdm import tqdm

from src.forward_metrics import compute_forward_metrics_batch, compute_loss


def train_epoch(model, loader, optimizer, config, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()

        pred_foot, pred_knee, pred_ankle = model(data)
        loss, _, _, _ = compute_loss(pred_foot, pred_knee, pred_ankle, data, config)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def eval_epoch(model, loader, config, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    metric_sums = {
        "foot_path_error": 0.0,
        "foot_chamfer": 0.0,
        "knee_nmae": 0.0,
        "ankle_nmae": 0.0,
        "knee_std_ratio": 0.0,
        "ankle_std_ratio": 0.0,
    }
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)
            pred_foot, pred_knee, pred_ankle = model(data)

            loss, _, _, _ = compute_loss(pred_foot, pred_knee, pred_ankle, data, config)
            metrics = compute_forward_metrics_batch(
                pred_foot,
                pred_knee,
                pred_ankle,
                data.y_foot.view_as(pred_foot),
                data.y_knee.view_as(pred_knee),
                data.y_ankle.view_as(pred_ankle),
            )

            batch_size = pred_foot.size(0)
            total_loss += float(loss.item())
            total_batches += 1
            total_samples += batch_size
            for key in metric_sums:
                metric_sums[key] += float(metrics[key].sum().item())

    avg_loss = total_loss / max(total_batches, 1)
    averaged_metrics = {
        key: value / max(total_samples, 1)
        for key, value in metric_sums.items()
    }
    return avg_loss, averaged_metrics
