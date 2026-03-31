import math
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _as_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def subset_by_indices(items: Sequence, indices: Sequence[int]) -> List:
    return [items[i] for i in indices]


def load_or_create_fixed_split(
    num_samples: int,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    split_path: str,
) -> Dict[str, object]:
    if os.path.exists(split_path):
        split = torch.load(split_path, map_location='cpu', weights_only=False)
        if (
            split.get('num_samples') == num_samples
            and split.get('split_seed') == split_seed
            and abs(split.get('val_ratio', -1.0) - val_ratio) < 1e-12
            and abs(split.get('test_ratio', -1.0) - test_ratio) < 1e-12
        ):
            return split

    n_test = max(1, int(num_samples * test_ratio))
    n_val = max(1, int(num_samples * val_ratio))
    n_train = num_samples - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Invalid split sizes for num_samples={num_samples}, "
            f"val_ratio={val_ratio}, test_ratio={test_ratio}"
        )

    rng = random.Random(split_seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)

    split = {
        'num_samples': num_samples,
        'split_seed': split_seed,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'test_indices': indices[:n_test],
        'val_indices': indices[n_test:n_test + n_val],
        'train_indices': indices[n_test + n_val:],
    }
    split_dir = os.path.dirname(split_path)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
    torch.save(split, split_path)
    return split


def build_target_feature(
    sample: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    weights = weights or {}
    w_foot = math.sqrt(float(weights.get('w_foot', 1.0)))
    w_knee = math.sqrt(float(weights.get('w_knee', 1.0)))
    w_ankle = math.sqrt(float(weights.get('w_ankle', 1.0)))

    y_foot = _as_tensor(sample['y_foot']).reshape(-1) * w_foot
    y_knee = _as_tensor(sample['y_knee']).reshape(-1) * w_knee
    y_ankle = _as_tensor(sample['y_ankle']).reshape(-1) * w_ankle
    return torch.cat([y_foot, y_knee, y_ankle], dim=0)


def stack_target_features(
    samples: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    return torch.stack([build_target_feature(sample, weights=weights) for sample in samples], dim=0)


def compute_sample_difficulty(sample: Dict[str, torch.Tensor]) -> float:
    y_foot = _as_tensor(sample['y_foot'])
    y_knee = _as_tensor(sample['y_knee'])
    y_ankle = _as_tensor(sample['y_ankle'])

    if y_foot.size(0) > 2:
        foot_second = y_foot[2:] - 2.0 * y_foot[1:-1] + y_foot[:-2]
        foot_curvature = torch.norm(foot_second, dim=-1).mean().item()
    else:
        foot_curvature = 0.0

    foot_span = torch.norm(y_foot.max(dim=0).values - y_foot.min(dim=0).values).item()
    knee_amp = (y_knee.max() - y_knee.min()).item()
    ankle_amp = (y_ankle.max() - y_ankle.min()).item()
    return 0.45 * foot_curvature + 0.2 * foot_span + 0.2 * knee_amp + 0.15 * ankle_amp


def select_hard_test_indices(
    samples: Sequence[Dict[str, torch.Tensor]],
    split: Dict[str, object],
    hard_fraction: float = 0.25,
    min_hard_samples: int = 128,
) -> List[int]:
    test_indices = list(split['test_indices'])
    scored = [(compute_sample_difficulty(samples[idx]), idx) for idx in test_indices]
    scored.sort(key=lambda item: item[0], reverse=True)
    num_hard = min(len(scored), max(int(len(scored) * hard_fraction), min_hard_samples))
    hard_indices = [idx for _, idx in scored[:num_hard]]
    hard_indices.sort()
    return hard_indices


def _ensure_batch_target(target_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if target_tensor.dim() == 2:
        return target_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    if target_tensor.dim() == 1:
        return target_tensor.unsqueeze(0).expand(batch_size, -1)
    return target_tensor


def _normalized_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.dim()))
    rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=dims))
    target_flat = target.reshape(target.size(0), -1)
    scale = target_flat.max(dim=1).values - target_flat.min(dim=1).values
    scale = torch.clamp(scale, min=1e-6)
    return rmse / scale


def _foot_scale(target_foot: torch.Tensor) -> torch.Tensor:
    span = target_foot.max(dim=1).values - target_foot.min(dim=1).values
    return torch.clamp(torch.norm(span, dim=-1), min=1e-6)


def chamfer_distance_foot_batch(pred_foot: torch.Tensor, target_foot: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred_foot.float(), target_foot.float())
    d1 = dist.min(dim=2).values.mean(dim=1)
    d2 = dist.min(dim=1).values.mean(dim=1)
    return (d1 + d2) / 2.0


def smoothness_penalty_batch(
    pred_foot: torch.Tensor,
    pred_knee: torch.Tensor,
    pred_ankle: torch.Tensor,
) -> torch.Tensor:
    if pred_foot.size(1) < 3:
        return torch.zeros(pred_foot.size(0), dtype=pred_foot.dtype, device=pred_foot.device)

    foot_second = pred_foot[:, 2:] - 2.0 * pred_foot[:, 1:-1] + pred_foot[:, :-2]
    knee_second = pred_knee[:, 2:] - 2.0 * pred_knee[:, 1:-1] + pred_knee[:, :-2]
    ankle_second = pred_ankle[:, 2:] - 2.0 * pred_ankle[:, 1:-1] + pred_ankle[:, :-2]
    foot_penalty = torch.norm(foot_second, dim=-1).mean(dim=1)
    knee_penalty = torch.abs(knee_second).mean(dim=1)
    ankle_penalty = torch.abs(ankle_second).mean(dim=1)
    return foot_penalty + 0.5 * knee_penalty + 0.5 * ankle_penalty


def compute_joint_metrics_batch(
    pred_foot: torch.Tensor,
    pred_knee: torch.Tensor,
    pred_ankle: torch.Tensor,
    target: Dict[str, torch.Tensor],
    metric_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    metric_cfg = metric_cfg or {}
    batch_size = pred_foot.size(0)
    target_foot = _ensure_batch_target(_as_tensor(target['y_foot'], dtype=pred_foot.dtype), batch_size).to(pred_foot.device)
    target_knee = _ensure_batch_target(_as_tensor(target['y_knee'], dtype=pred_knee.dtype), batch_size).to(pred_knee.device)
    target_ankle = _ensure_batch_target(_as_tensor(target['y_ankle'], dtype=pred_ankle.dtype), batch_size).to(pred_ankle.device)

    foot_chamfer = chamfer_distance_foot_batch(pred_foot, target_foot)
    foot_chamfer_norm = foot_chamfer / _foot_scale(target_foot)
    foot_nrmse = _normalized_rmse(pred_foot, target_foot)
    knee_nrmse = _normalized_rmse(pred_knee, target_knee)
    ankle_nrmse = _normalized_rmse(pred_ankle, target_ankle)
    smoothness = smoothness_penalty_batch(pred_foot, pred_knee, pred_ankle)

    foot_mix_chamfer = float(metric_cfg.get('foot_mix_chamfer', 0.5))
    foot_mix_nrmse = float(metric_cfg.get('foot_mix_nrmse', 0.5))
    foot_score = foot_mix_chamfer * foot_chamfer_norm + foot_mix_nrmse * foot_nrmse

    w_foot = float(metric_cfg.get('w_foot', 0.5))
    w_knee = float(metric_cfg.get('w_knee', 0.25))
    w_ankle = float(metric_cfg.get('w_ankle', 0.25))
    joint_score = w_foot * foot_score + w_knee * knee_nrmse + w_ankle * ankle_nrmse

    return {
        'foot_chamfer': foot_chamfer,
        'foot_chamfer_norm': foot_chamfer_norm,
        'foot_nrmse': foot_nrmse,
        'knee_nrmse': knee_nrmse,
        'ankle_nrmse': ankle_nrmse,
        'foot_score': foot_score,
        'smoothness': smoothness,
        'joint_score': joint_score,
    }


def compute_reward_batch(
    pred_foot: torch.Tensor,
    pred_knee: torch.Tensor,
    pred_ankle: torch.Tensor,
    target: Dict[str, torch.Tensor],
    reward_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    reward_cfg = reward_cfg or {}
    metrics = compute_joint_metrics_batch(pred_foot, pred_knee, pred_ankle, target, metric_cfg=reward_cfg)
    w_smooth = float(reward_cfg.get('w_smooth', 0.05))
    reward = -(metrics['joint_score'] + w_smooth * metrics['smoothness'])
    metrics['reward'] = reward
    return reward, metrics


def metrics_to_numpy(metrics: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value)
    return out


def summarize_metric_dicts(metric_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metric_dicts = list(metric_dicts)
    if not metric_dicts:
        return {}
    keys = sorted(metric_dicts[0].keys())
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts]))
        for key in keys
    }
