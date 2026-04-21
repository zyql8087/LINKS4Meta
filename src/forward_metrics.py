from __future__ import annotations

from collections import defaultdict

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.forward_dataset_utils import family_id_to_name


def _target_curve_range(target: torch.Tensor) -> torch.Tensor:
    flat = target.reshape(target.size(0), -1)
    return torch.clamp(flat.max(dim=1).values - flat.min(dim=1).values, min=1e-6)


def foot_path_error(pred_foot: torch.Tensor, target_foot: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred_foot - target_foot, dim=-1).mean(dim=1)


def foot_chamfer_distance(pred_foot: torch.Tensor, target_foot: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(pred_foot.float(), target_foot.float())
    d1 = distances.min(dim=2).values.mean(dim=1)
    d2 = distances.min(dim=1).values.mean(dim=1)
    return 0.5 * (d1 + d2)


def curve_nmae(pred_curve: torch.Tensor, target_curve: torch.Tensor) -> torch.Tensor:
    mae = torch.mean(torch.abs(pred_curve - target_curve), dim=1)
    return mae / _target_curve_range(target_curve)


def semantic_curve_std_ratio(pred_curve: torch.Tensor, target_curve: torch.Tensor) -> torch.Tensor:
    pred_std = torch.std(pred_curve, dim=1, unbiased=False)
    target_std = torch.clamp(torch.std(target_curve, dim=1, unbiased=False), min=1e-6)
    return pred_std / target_std


def compute_forward_metrics_batch(
    pred_foot: torch.Tensor,
    pred_knee: torch.Tensor,
    pred_ankle: torch.Tensor,
    target_foot: torch.Tensor,
    target_knee: torch.Tensor,
    target_ankle: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "foot_path_error": foot_path_error(pred_foot, target_foot),
        "foot_chamfer": foot_chamfer_distance(pred_foot, target_foot),
        "knee_nmae": curve_nmae(pred_knee, target_knee),
        "ankle_nmae": curve_nmae(pred_ankle, target_ankle),
        "knee_std_ratio": semantic_curve_std_ratio(pred_knee, target_knee),
        "ankle_std_ratio": semantic_curve_std_ratio(pred_ankle, target_ankle),
    }


def _mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted(metric_rows[0].keys())
    return {
        key: float(sum(row[key] for row in metric_rows) / len(metric_rows))
        for key in keys
    }


def _sample_family_id(data, sample_idx: int) -> int:
    family_id = getattr(data, "family_id", None)
    if family_id is None:
        return -1
    if isinstance(family_id, torch.Tensor):
        return int(family_id.view(-1)[sample_idx].item())
    return int(family_id)


def compute_loss(pred_foot, pred_knee, pred_ankle, data, config):
    import torch.nn.functional as F

    w_foot = config.get("w_foot", 1.0)
    w_knee = config.get("w_knee", 0.5)
    w_ankle = config.get("w_ankle", 0.5)

    loss_foot = F.mse_loss(pred_foot, data.y_foot.view_as(pred_foot))
    loss_knee = F.mse_loss(pred_knee, data.y_knee.view_as(pred_knee))
    loss_ankle = F.mse_loss(pred_ankle, data.y_ankle.view_as(pred_ankle))
    total_loss = w_foot * loss_foot + w_knee * loss_knee + w_ankle * loss_ankle
    return total_loss, loss_foot, loss_knee, loss_ankle


def evaluate_forward_model(model, data_items, config, device, batch_size=256) -> dict[str, object]:
    if not data_items:
        return {"sample_count": 0, "overall": {}, "per_family": {}}

    loader = DataLoader(data_items, batch_size=batch_size, shuffle=False, num_workers=0)
    metric_rows = []
    family_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    total_loss = 0.0
    total_batches = 0

    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred_foot, pred_knee, pred_ankle = model(data)
            loss, _, _, _ = compute_loss(pred_foot, pred_knee, pred_ankle, data, config.get("training", {}))
            metrics = compute_forward_metrics_batch(
                pred_foot,
                pred_knee,
                pred_ankle,
                data.y_foot.view_as(pred_foot),
                data.y_knee.view_as(pred_knee),
                data.y_ankle.view_as(pred_ankle),
            )
            total_loss += float(loss.item())
            total_batches += 1

            batch_size_actual = pred_foot.size(0)
            for sample_idx in range(batch_size_actual):
                row = {name: float(values[sample_idx].item()) for name, values in metrics.items()}
                metric_rows.append(row)
                family_name = family_id_to_name(_sample_family_id(data, sample_idx))
                family_rows[family_name].append(row)

    overall = _mean_metrics(metric_rows)
    if overall:
        overall["loss_total"] = float(total_loss / max(total_batches, 1))
    per_family = {
        family_name: {
            **_mean_metrics(rows),
            "sample_count": len(rows),
        }
        for family_name, rows in sorted(family_rows.items())
    }
    return {
        "sample_count": len(metric_rows),
        "overall": overall,
        "per_family": per_family,
    }


def evaluate_retrieval_baseline(train_items, eval_items) -> dict[str, object]:
    if not train_items or not eval_items:
        return {"sample_count": 0, "overall": {}, "per_family": {}}

    train_feature = torch.cat([item.retrieval_feature for item in train_items], dim=0)
    metric_rows = []
    family_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    for eval_item in eval_items:
        query = eval_item.retrieval_feature.view(1, -1)
        distances = torch.norm(train_feature - query, dim=1)
        nearest_idx = int(torch.argmin(distances).item())
        source = train_items[nearest_idx]

        pred_foot = source.y_foot.unsqueeze(0)
        pred_knee = source.y_knee.unsqueeze(0)
        pred_ankle = source.y_ankle.unsqueeze(0)
        target_foot = eval_item.y_foot.unsqueeze(0)
        target_knee = eval_item.y_knee.unsqueeze(0)
        target_ankle = eval_item.y_ankle.unsqueeze(0)

        metrics = compute_forward_metrics_batch(
            pred_foot,
            pred_knee,
            pred_ankle,
            target_foot,
            target_knee,
            target_ankle,
        )
        row = {name: float(values[0].item()) for name, values in metrics.items()}
        metric_rows.append(row)
        family_name = family_id_to_name(_sample_family_id(eval_item, 0))
        family_rows[family_name].append(row)

    return {
        "sample_count": len(metric_rows),
        "overall": _mean_metrics(metric_rows),
        "per_family": {
            family_name: {
                **_mean_metrics(rows),
                "sample_count": len(rows),
            }
            for family_name, rows in sorted(family_rows.items())
        },
    }


def evaluate_semantic_ablation(model, data_items, config, device, batch_size=256) -> dict[str, float]:
    if not data_items:
        return {}

    degraded_items = []
    for item in data_items:
        clone = item.clone()
        clone.x = clone.x.clone()
        if clone.x.size(-1) >= 8:
            clone.x[:, -4:] = 0.0
        if hasattr(clone, "mask_foot"):
            clone.mask_foot = torch.zeros_like(clone.mask_foot)
        if hasattr(clone, "mask_knee"):
            clone.mask_knee = torch.zeros_like(clone.mask_knee)
        if hasattr(clone, "mask_ankle"):
            clone.mask_ankle = torch.zeros_like(clone.mask_ankle)
        degraded_items.append(clone)

    base_report = evaluate_forward_model(model, data_items, config, device, batch_size=batch_size)
    ablated_report = evaluate_forward_model(model, degraded_items, config, device, batch_size=batch_size)
    output = {}
    for metric_name in ("foot_path_error", "foot_chamfer", "knee_nmae", "ankle_nmae"):
        base_value = base_report["overall"].get(metric_name, 0.0)
        ablated_value = ablated_report["overall"].get(metric_name, 0.0)
        output[f"{metric_name}_degradation_ratio"] = float(ablated_value / max(base_value, 1e-6))
    return output


def phase3_gate(report: dict[str, object], gate_cfg: dict[str, float] | None = None) -> dict[str, object]:
    gate_cfg = gate_cfg or {}
    primary_foot_metric = gate_cfg.get("primary_foot_metric", "foot_path_error")
    family_ratio_limit = float(gate_cfg.get("max_family_metric_ratio", 1.8))
    bar89_multiplier = float(gate_cfg.get("max_89_multiplier", 1.8))
    semantic_ablation_limit = float(gate_cfg.get("max_semantic_ablation_ratio", 3.0))

    current = report.get("current_model", {})
    baselines = report.get("baselines", {})
    overall = current.get("test", {}).get("overall", {})
    per_family = current.get("test", {}).get("per_family", {})
    semantic_ablation = report.get("semantic_ablation", {})

    stronger_than = {}
    for baseline_name, baseline_report in baselines.items():
        baseline_overall = baseline_report.get("test", {}).get("overall", {})
        if not baseline_overall:
            stronger_than[baseline_name] = False
            continue
        stronger_than[baseline_name] = all(
            overall.get(metric_name, float("inf")) <= baseline_overall.get(metric_name, float("inf"))
            for metric_name in (primary_foot_metric, "knee_nmae", "ankle_nmae")
        )

    family_metric_values = [
        metrics.get(primary_foot_metric, 0.0)
        for family_name, metrics in per_family.items()
        if family_name != "unknown"
    ]
    family_imbalance = False
    if family_metric_values:
        family_imbalance = max(family_metric_values) / max(min(family_metric_values), 1e-6) > family_ratio_limit

    out_of_control_families = []
    for family_name in ("8bar", "9bar"):
        metrics = per_family.get(family_name, {})
        if not metrics:
            continue
        if any(
            metrics.get(metric_name, 0.0) > overall.get(metric_name, 0.0) * bar89_multiplier
            for metric_name in (primary_foot_metric, "knee_nmae", "ankle_nmae")
            if overall.get(metric_name) is not None
        ):
            out_of_control_families.append(family_name)

    semantic_collapse = any(
        value > semantic_ablation_limit
        for key, value in semantic_ablation.items()
        if key.endswith("_degradation_ratio")
    )

    return {
        "primary_foot_metric": primary_foot_metric,
        "stronger_than_baselines": stronger_than,
        "bar89_out_of_control": bool(out_of_control_families),
        "bar89_out_of_control_families": out_of_control_families,
        "family_imbalance": family_imbalance,
        "semantic_collapse": semantic_collapse,
        "ready_for_rl": bool(
            stronger_than
            and all(stronger_than.values())
            and not out_of_control_families
            and not family_imbalance
            and not semantic_collapse
        ),
    }
