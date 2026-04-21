# train_forward_bio.py

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_path
from src.data_load import DataLoaderFactory
from src.forward_metrics import evaluate_forward_model
from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN
from src.generative_curve.GNN_train_biokinematics import eval_epoch, train_epoch


class PreCachedLoader:
    def __init__(self, dataloader, device: str, desc: str, shuffle: bool) -> None:
        self.batches = []
        self.shuffle = shuffle
        for batch in tqdm(dataloader, desc=desc):
            self.batches.append(batch.to(device))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def run_training(
    config_model: dict,
    config_data: dict,
    *,
    output_dir: Path,
    device: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running on device: {device}")
    print("Loading Dataset 80k...")
    factory = DataLoaderFactory(config_data)
    train_loader_orig = factory.create_train_loader(
        batch_size=config_model["training"]["batch_size"],
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader_orig = factory.create_val_loader(
        batch_size=config_model["training"]["batch_size"],
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    print("\nPre-caching batched graphs to Device memory to bypass PyG collation bottleneck...")
    train_loader = PreCachedLoader(train_loader_orig, device, desc="Caching Train", shuffle=True)
    val_loader = PreCachedLoader(val_loader_orig, device, desc="Caching Val", shuffle=False)

    model = BioKinematicsGNN(config_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_model["training"]["learning_rate"])

    print(f"Model Initialized. Params: {sum(p.numel() for p in model.parameters())}")

    train_losses = []
    val_losses = []
    model_path = ensure_parent_dir(output_dir / "model_bio_best.pt")
    train_report_path = ensure_parent_dir(output_dir / "forward_train_report.json")

    best_val_loss = float("inf")
    best_val_metrics = {}
    for epoch in range(config_model["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, config_model["training"], device)
        val_loss, val_metrics = eval_epoch(model, val_loader, config_model["training"], device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Foot Path: {val_metrics.get('foot_path_error', 0.0):.6f} | "
            f"Knee NMAE: {val_metrics.get('knee_nmae', 0.0):.6f} | "
            f"Ankle NMAE: {val_metrics.get('ankle_nmae', 0.0):.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = dict(val_metrics)
            torch.save(model.state_dict(), model_path)
            print("  --> Model Saved!")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    loss_curve_path = output_dir / "loss_curve.png"
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to '{loss_curve_path}'.")

    best_model = BioKinematicsGNN(config_model).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    best_model.eval()
    eval_batch_size = config_model.get("evaluation", {}).get("batch_size", config_model["training"]["batch_size"])
    evaluation_report = {
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "train": evaluate_forward_model(best_model, factory.get_split_data("train"), config_model, device, batch_size=eval_batch_size),
        "val": evaluate_forward_model(best_model, factory.get_split_data("val"), config_model, device, batch_size=eval_batch_size),
        "test": evaluate_forward_model(best_model, factory.get_split_data("test"), config_model, device, batch_size=eval_batch_size),
    }
    train_report_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
    print(f"Forward report saved to '{train_report_path}'.")
    return {
        "model_path": str(model_path),
        "loss_curve_path": str(loss_curve_path),
        "report_path": str(train_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_model", type=str, default="src/config_model_bio.yaml")
    parser.add_argument("--config_dataset", type=str, default="src/config_dataset.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "demo" / "outputs" / "training"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    config_model, config_model_path = load_yaml_config(args.config_model, SCRIPT_DIR, WORKSPACE_ROOT)
    config_data, config_data_path = load_yaml_config(args.config_dataset, SCRIPT_DIR, WORKSPACE_ROOT)
    config_data["__config_dir__"] = str(config_data_path.parent)
    config_data["dataset_path"] = str(
        resolve_path(config_data["dataset_path"], config_data_path.parent, WORKSPACE_ROOT)
    )
    if config_data.get("split_indices_path"):
        config_data["split_indices_path"] = str(
            resolve_path(config_data["split_indices_path"], config_data_path.parent, WORKSPACE_ROOT)
        )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (WORKSPACE_ROOT / output_dir).resolve()
    run_training(config_model, config_data, output_dir=output_dir, device=args.device)


if __name__ == "__main__":
    main()
