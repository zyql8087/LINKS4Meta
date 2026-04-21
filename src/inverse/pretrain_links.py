from __future__ import annotations

import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from src.config_utils import ensure_parent_dir
from src.inverse.experiment_utils import _canonical_split_indices
from src.inverse.rl_env import validate_graph_structure
from src.kinematics_extract import extract_kinematics


@dataclass
class LinksPretrainBatch:
    valid_graphs: Batch
    invalid_graphs: Batch
    y_foot: torch.Tensor
    y_knee: torch.Tensor
    y_ankle: torch.Tensor
    curve_targets: torch.Tensor


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ValidityHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ForwardCurveHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _curve_targets(sample: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(key in sample for key in ("foot_curve", "knee_curve", "ankle_curve")):
        return (
            np.asarray(sample["foot_curve"], dtype=np.float32),
            np.asarray(sample["knee_curve"], dtype=np.float32),
            np.asarray(sample["ankle_curve"], dtype=np.float32),
        )
    return extract_kinematics(sample)


def sample_to_pretrain_graph(sample: dict[str, object]) -> Data:
    A = np.asarray(sample["A"])
    x0 = np.asarray(sample["x0"], dtype=np.float32)
    types = np.asarray(sample["types"])

    is_fixed = (types == 1).astype(np.float32)
    is_grounded = np.zeros_like(is_fixed, dtype=np.float32)
    if is_grounded.size > 0:
        is_grounded[0] = 1.0
    x_features = np.column_stack([x0, is_fixed, is_grounded])

    rows, cols = np.where(A)
    graph = Data(
        x=torch.tensor(x_features, dtype=torch.float32),
        pos=torch.tensor(x0, dtype=torch.float32),
        edge_index=torch.tensor(np.array([rows, cols]), dtype=torch.long),
    )
    graph.sample_id = torch.tensor([int(sample.get("id", 0))], dtype=torch.long)
    return graph


def _synthesize_invalid_graph(graph: Data, rng: random.Random) -> Data:
    invalid = Data(
        x=graph.x.clone(),
        pos=graph.pos.clone(),
        edge_index=graph.edge_index.clone(),
    )
    num_nodes = int(invalid.x.size(0))
    if num_nodes < 2:
        return invalid

    moving_nodes = [idx for idx in range(num_nodes) if float(invalid.x[idx, 2].item()) < 0.5]
    source_candidates = moving_nodes if moving_nodes else list(range(1, num_nodes))
    source_idx = source_candidates[rng.randrange(len(source_candidates))]
    target_candidates = [idx for idx in range(num_nodes) if idx != source_idx]
    target_idx = target_candidates[rng.randrange(len(target_candidates))]
    target_pos = invalid.pos[target_idx].clone()

    invalid.pos[source_idx] = target_pos
    invalid.x[source_idx, :2] = target_pos
    return invalid


def _load_split_artifact(split_path: str) -> dict[str, object]:
    path = Path(split_path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return torch.load(path, map_location="cpu", weights_only=False)


def _map_split_indices(samples: Sequence[dict[str, object]], split_path: str | None) -> dict[str, list[int]]:
    if split_path and os.path.exists(split_path):
        raw_split = _canonical_split_indices(_load_split_artifact(split_path))
        sample_id_to_local = {
            int(sample.get("id", idx)): idx
            for idx, sample in enumerate(samples)
        }
        mapped = {}
        for split_name in ("train_indices", "val_indices", "test_indices"):
            mapped[split_name] = [
                sample_id_to_local[idx]
                for idx in raw_split[split_name]
                if idx in sample_id_to_local
            ]
        return mapped

    indices = list(range(len(samples)))
    rng = random.Random(42)
    rng.shuffle(indices)
    n_test = max(1, int(round(len(indices) * 0.1)))
    n_val = max(1, int(round(len(indices) * 0.1)))
    return {
        "test_indices": indices[:n_test],
        "val_indices": indices[n_test:n_test + n_val],
        "train_indices": indices[n_test + n_val:],
    }


def _family_subset_indices(samples: Sequence[dict[str, object]], indices: Sequence[int], max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)

    family_to_indices: dict[str, list[int]] = {}
    for idx in indices:
        family_name = str(samples[idx].get("family_id") or samples[idx].get("family") or "unknown")
        family_to_indices.setdefault(family_name, []).append(int(idx))

    rng = random.Random(seed)
    budgets = {family: 0 for family in family_to_indices}
    ordered_families = sorted(family_to_indices.keys())
    remaining = max_samples
    if remaining >= len(ordered_families):
        for family in ordered_families:
            budgets[family] = 1
            remaining -= 1

    total = max(1, len(indices))
    remainders: list[tuple[float, str]] = []
    for family in ordered_families:
        exact = remaining * len(family_to_indices[family]) / total
        extra = min(len(family_to_indices[family]) - budgets[family], int(exact))
        budgets[family] += extra
        remainders.append((exact - extra, family))

    assigned = sum(budgets.values())
    for _, family in sorted(remainders, reverse=True):
        if assigned >= max_samples:
            break
        if budgets[family] < len(family_to_indices[family]):
            budgets[family] += 1
            assigned += 1

    chosen: list[int] = []
    for family in ordered_families:
        family_indices = list(family_to_indices[family])
        if budgets[family] <= 0:
            continue
        chosen.extend(rng.sample(family_indices, budgets[family]))
    return sorted(chosen)


def build_links_pretrain_records(
    samples: Sequence[dict[str, object]],
    *,
    split_path: str | None = None,
    max_samples: int = 0,
    seed: int = 42,
    constraint_cfg: dict | None = None,
) -> dict[str, object]:
    split = _map_split_indices(samples, split_path)
    selected = {
        split_name: _family_subset_indices(samples, indices, max_samples, seed)
        for split_name, indices in split.items()
    }

    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    local_split = {"train_indices": [], "val_indices": [], "test_indices": []}

    for split_name in ("train_indices", "val_indices", "test_indices"):
        for global_idx in selected[split_name]:
            sample = samples[global_idx]
            valid_graph = sample_to_pretrain_graph(sample)
            invalid_graph = _synthesize_invalid_graph(valid_graph, rng)
            is_invalid_valid, _ = validate_graph_structure(invalid_graph, constraint_cfg)
            if is_invalid_valid:
                continue
            y_foot, y_knee, y_ankle = _curve_targets(sample)
            record = {
                "sample_id": int(sample.get("id", global_idx)),
                "family_id": str(sample.get("family_id") or sample.get("family") or "unknown"),
                "valid_graph": valid_graph,
                "invalid_graph": invalid_graph,
                "y_foot": torch.tensor(y_foot, dtype=torch.float32),
                "y_knee": torch.tensor(y_knee, dtype=torch.float32),
                "y_ankle": torch.tensor(y_ankle, dtype=torch.float32),
            }
            local_split[split_name].append(len(records))
            records.append(record)

    return {
        "records": records,
        "split": local_split,
        "split_source": "precomputed_group_split" if split_path and os.path.exists(split_path) else "random_split",
        "max_samples": int(max_samples),
    }


def ensure_links_pretrain_cache(
    *,
    dataset_path: str,
    cache_path: str,
    split_path: str | None = None,
    max_samples: int = 0,
    seed: int = 42,
    constraint_cfg: dict | None = None,
    use_cached: bool = True,
) -> dict[str, object]:
    if use_cached and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    with open(dataset_path, "rb") as handle:
        samples = pickle.load(handle)
    cache = build_links_pretrain_records(
        samples,
        split_path=split_path,
        max_samples=max_samples,
        seed=seed,
        constraint_cfg=constraint_cfg,
    )
    ensure_parent_dir(cache_path)
    torch.save(cache, cache_path)
    return cache


def subset_by_indices(items: Sequence, indices: Sequence[int]) -> list:
    return [items[int(idx)] for idx in indices]


def make_links_pretrain_batches(
    records: Sequence[dict[str, object]],
    *,
    batch_size: int,
    device,
    shuffle: bool,
) -> list[LinksPretrainBatch]:
    indices = list(range(len(records)))
    if shuffle:
        random.shuffle(indices)

    batches: list[LinksPretrainBatch] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_records = [records[idx] for idx in batch_indices]
        if not batch_records:
            continue
        batches.append(
            LinksPretrainBatch(
                valid_graphs=Batch.from_data_list([item["valid_graph"] for item in batch_records]).to(device),
                invalid_graphs=Batch.from_data_list([item["invalid_graph"] for item in batch_records]).to(device),
                y_foot=torch.stack([item["y_foot"] for item in batch_records]).to(device),
                y_knee=torch.stack([item["y_knee"] for item in batch_records]).to(device),
                y_ankle=torch.stack([item["y_ankle"] for item in batch_records]).to(device),
                curve_targets=torch.stack(
                    [
                        torch.cat(
                            [
                                item["y_foot"].reshape(-1),
                                item["y_knee"].reshape(-1),
                                item["y_ankle"].reshape(-1),
                            ],
                            dim=0,
                        )
                        for item in batch_records
                    ]
                ).to(device),
            )
        )
    return batches


def contrastive_loss(graph_proj: torch.Tensor, curve_proj: torch.Tensor, temperature: float) -> tuple[torch.Tensor, float]:
    logits = graph_proj @ curve_proj.t() / max(float(temperature), 1.0e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_g2c = F.cross_entropy(logits, targets)
    loss_c2g = F.cross_entropy(logits.t(), targets)
    retrieval = float((torch.argmax(logits, dim=1) == targets).float().mean().item())
    return 0.5 * (loss_g2c + loss_c2g), retrieval


def _average_dict(metric_sums: dict[str, float], denom: int) -> dict[str, float]:
    denom = max(1, int(denom))
    return {key: value / denom for key, value in metric_sums.items()}


def run_links_pretraining(
    *,
    policy,
    curve_encoder,
    cache: dict[str, object],
    cfg: dict,
    device,
    output_model_path: str,
    output_report_path: str,
) -> dict[str, object]:
    pretrain_cfg = cfg.get("links_pretrain", {})
    train_records = subset_by_indices(cache["records"], cache["split"]["train_indices"])
    val_records = subset_by_indices(cache["records"], cache["split"]["val_indices"])
    test_records = subset_by_indices(cache["records"], cache["split"]["test_indices"])

    train_batches = make_links_pretrain_batches(
        train_records,
        batch_size=int(pretrain_cfg.get("batch_size", 256)),
        device=device,
        shuffle=True,
    )
    val_batches = make_links_pretrain_batches(
        val_records,
        batch_size=int(pretrain_cfg.get("batch_size", 256)),
        device=device,
        shuffle=False,
    )
    test_batches = make_links_pretrain_batches(
        test_records,
        batch_size=int(pretrain_cfg.get("batch_size", 256)),
        device=device,
        shuffle=False,
    )

    hidden_dim = int(cfg.get("gnn_policy", {}).get("hidden_dim", 128))
    latent_dim = int(cfg.get("curve_encoder", {}).get("latent_dim", 128))
    projection_dim = int(pretrain_cfg.get("projection_dim", latent_dim))
    sample_batch = train_batches[0] if train_batches else (val_batches[0] if val_batches else test_batches[0])
    output_dim = int(sample_batch.curve_targets.size(1))
    graph_projection = ProjectionHead(hidden_dim, projection_dim).to(device)
    curve_projection = ProjectionHead(latent_dim, projection_dim).to(device)
    validity_head = ValidityHead(hidden_dim).to(device)
    forward_head = ForwardCurveHead(hidden_dim, output_dim).to(device)

    optimizer = torch.optim.Adam(
        list(policy.gnn.parameters())
        + list(curve_encoder.parameters())
        + list(graph_projection.parameters())
        + list(curve_projection.parameters())
        + list(validity_head.parameters())
        + list(forward_head.parameters()),
        lr=float(pretrain_cfg.get("learning_rate", 3e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(pretrain_cfg.get("epochs", 20))),
    )

    contrastive_weight = float(pretrain_cfg.get("contrastive_weight", 1.0))
    validity_weight = float(pretrain_cfg.get("validity_weight", 0.5))
    forward_weight = float(pretrain_cfg.get("forward_weight", 0.5))
    temperature = float(pretrain_cfg.get("temperature", 0.07))
    patience = int(pretrain_cfg.get("patience", 5))

    def eval_batches(batches: Sequence[LinksPretrainBatch]) -> dict[str, float]:
        if not batches:
            return {
                "loss_total": 0.0,
                "loss_contrastive": 0.0,
                "loss_validity": 0.0,
                "loss_forward": 0.0,
                "retrieval_top1": 0.0,
                "validity_accuracy": 0.0,
            }
        policy.eval()
        curve_encoder.eval()
        graph_projection.eval()
        curve_projection.eval()
        validity_head.eval()
        forward_head.eval()
        metric_sums = {
            "loss_total": 0.0,
            "loss_contrastive": 0.0,
            "loss_validity": 0.0,
            "loss_forward": 0.0,
            "retrieval_top1": 0.0,
            "validity_accuracy": 0.0,
        }
        with torch.no_grad():
            for batch in batches:
                x_valid = policy.encode_graph(batch.valid_graphs)
                graph_feat = scatter(x_valid, batch.valid_graphs.batch, dim=0, reduce="mean")
                curve_feat = curve_encoder(batch.y_foot, batch.y_knee, batch.y_ankle)
                graph_proj = graph_projection(graph_feat)
                curve_proj = curve_projection(curve_feat)
                loss_contrastive, retrieval_top1 = contrastive_loss(graph_proj, curve_proj, temperature)

                x_invalid = policy.encode_graph(batch.invalid_graphs)
                invalid_feat = scatter(x_invalid, batch.invalid_graphs.batch, dim=0, reduce="mean")
                valid_logits = validity_head(graph_feat)
                invalid_logits = validity_head(invalid_feat)
                labels = torch.cat(
                    [
                        torch.ones_like(valid_logits),
                        torch.zeros_like(invalid_logits),
                    ],
                    dim=0,
                )
                logits = torch.cat([valid_logits, invalid_logits], dim=0)
                loss_validity = F.binary_cross_entropy_with_logits(logits, labels)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                validity_acc = float((preds == labels).float().mean().item())
                pred_curves = forward_head(graph_feat)
                loss_forward = F.smooth_l1_loss(pred_curves, batch.curve_targets)

                loss_total = (
                    contrastive_weight * loss_contrastive
                    + validity_weight * loss_validity
                    + forward_weight * loss_forward
                )
                metric_sums["loss_total"] += float(loss_total.item())
                metric_sums["loss_contrastive"] += float(loss_contrastive.item())
                metric_sums["loss_validity"] += float(loss_validity.item())
                metric_sums["loss_forward"] += float(loss_forward.item())
                metric_sums["retrieval_top1"] += retrieval_top1
                metric_sums["validity_accuracy"] += validity_acc
        return _average_dict(metric_sums, len(batches))

    history = {
        "train_loss_total": [],
        "val_loss_total": [],
        "val_retrieval_top1": [],
        "val_validity_accuracy": [],
        "val_loss_forward": [],
    }
    best_val = math.inf
    best_state = None
    patience_counter = 0

    for epoch in range(int(pretrain_cfg.get("epochs", 20))):
        policy.train()
        curve_encoder.train()
        graph_projection.train()
        curve_projection.train()
        validity_head.train()
        forward_head.train()
        metric_sums = {
            "loss_total": 0.0,
            "loss_contrastive": 0.0,
            "loss_validity": 0.0,
            "loss_forward": 0.0,
            "retrieval_top1": 0.0,
            "validity_accuracy": 0.0,
        }
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            x_valid = policy.encode_graph(batch.valid_graphs)
            graph_feat = scatter(x_valid, batch.valid_graphs.batch, dim=0, reduce="mean")
            curve_feat = curve_encoder(batch.y_foot, batch.y_knee, batch.y_ankle)
            graph_proj = graph_projection(graph_feat)
            curve_proj = curve_projection(curve_feat)
            loss_contrastive, retrieval_top1 = contrastive_loss(graph_proj, curve_proj, temperature)

            x_invalid = policy.encode_graph(batch.invalid_graphs)
            invalid_feat = scatter(x_invalid, batch.invalid_graphs.batch, dim=0, reduce="mean")
            valid_logits = validity_head(graph_feat)
            invalid_logits = validity_head(invalid_feat)
            labels = torch.cat(
                [torch.ones_like(valid_logits), torch.zeros_like(invalid_logits)],
                dim=0,
            )
            logits = torch.cat([valid_logits, invalid_logits], dim=0)
            loss_validity = F.binary_cross_entropy_with_logits(logits, labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            validity_acc = float((preds == labels).float().mean().item())
            pred_curves = forward_head(graph_feat)
            loss_forward = F.smooth_l1_loss(pred_curves, batch.curve_targets)

            loss_total = (
                contrastive_weight * loss_contrastive
                + validity_weight * loss_validity
                + forward_weight * loss_forward
            )
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.gnn.parameters())
                + list(curve_encoder.parameters())
                + list(graph_projection.parameters())
                + list(curve_projection.parameters())
                + list(validity_head.parameters())
                + list(forward_head.parameters()),
                1.0,
            )
            optimizer.step()

            metric_sums["loss_total"] += float(loss_total.item())
            metric_sums["loss_contrastive"] += float(loss_contrastive.item())
            metric_sums["loss_validity"] += float(loss_validity.item())
            metric_sums["loss_forward"] += float(loss_forward.item())
            metric_sums["retrieval_top1"] += retrieval_top1
            metric_sums["validity_accuracy"] += validity_acc

        scheduler.step()
        train_metrics = _average_dict(metric_sums, len(train_batches))
        val_metrics = eval_batches(val_batches if val_batches else train_batches)
        history["train_loss_total"].append(train_metrics["loss_total"])
        history["val_loss_total"].append(val_metrics["loss_total"])
        history["val_retrieval_top1"].append(val_metrics["retrieval_top1"])
        history["val_validity_accuracy"].append(val_metrics["validity_accuracy"])
        history["val_loss_forward"].append(val_metrics["loss_forward"])

        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            patience_counter = 0
            best_state = {
                "policy": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                "curve_encoder": {key: value.detach().cpu() for key, value in curve_encoder.state_dict().items()},
                "graph_projection": graph_projection.state_dict(),
                "curve_projection": curve_projection.state_dict(),
                "validity_head": validity_head.state_dict(),
                "forward_head": forward_head.state_dict(),
                "history": history,
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        raise RuntimeError("LINKS pretraining did not produce a checkpoint")

    _load_matching_state(policy, best_state["policy"])
    _load_matching_state(curve_encoder, best_state["curve_encoder"])

    test_metrics = eval_batches(test_batches if test_batches else val_batches if val_batches else train_batches)
    payload = {
        "policy_encoder": {
            key: value
            for key, value in best_state["policy"].items()
            if key.startswith("gnn.")
        },
        "curve_encoder": best_state["curve_encoder"],
        "graph_projection": best_state["graph_projection"],
        "curve_projection": best_state["curve_projection"],
        "validity_head": best_state["validity_head"],
        "forward_backbone": best_state["forward_head"],
        "history": history,
        "report": {
            "phase": "links_pretrain_encoder_validity",
            "dataset_size": len(cache["records"]),
            "train_size": len(train_records),
            "val_size": len(val_records),
            "test_size": len(test_records),
            "split_source": cache.get("split_source"),
            "best_val_loss": float(best_val),
            "tasks": ["graph_encoder_pretrain", "forward_pretrain", "validity_pretrain"],
            "test_metrics": test_metrics,
        },
    }

    ensure_parent_dir(output_model_path)
    torch.save(payload, output_model_path)
    bundle_path = Path(output_model_path)
    graph_encoder_path = bundle_path.with_name("graph_encoder.pt")
    forward_backbone_path = bundle_path.with_name("forward_backbone.pt")
    validity_head_path = bundle_path.with_name("validity_head.pt")
    torch.save(payload["policy_encoder"], graph_encoder_path)
    torch.save(payload["forward_backbone"], forward_backbone_path)
    torch.save(payload["validity_head"], validity_head_path)
    ensure_parent_dir(output_report_path)
    payload["report"]["artifacts"] = {
        "graph_encoder": str(graph_encoder_path),
        "forward_backbone": str(forward_backbone_path),
        "validity_head": str(validity_head_path),
        "bundle": str(bundle_path),
    }
    with open(output_report_path, "w", encoding="utf-8") as handle:
        json.dump(payload["report"], handle, indent=2, ensure_ascii=False)
    return payload["report"]


def _load_matching_state(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    current_state = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
    }
    module.load_state_dict(compatible, strict=False)


def load_links_pretrained_weights(policy, curve_encoder, ckpt_path: str, device) -> dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _load_matching_state(policy, ckpt.get("policy_encoder", ckpt.get("policy", {})))
    _load_matching_state(curve_encoder, ckpt["curve_encoder"])
    return ckpt.get("report", {})
