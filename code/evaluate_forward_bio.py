from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import load_yaml_config, resolve_path
from src.data_load import DataLoaderFactory
from src.forward_metrics import (
    evaluate_forward_model,
    evaluate_retrieval_baseline,
    evaluate_semantic_ablation,
    phase3_gate,
)
from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN


def _load_model(config_model: dict, model_path: Path, device: str):
    model = BioKinematicsGNN(config_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase3 forward surrogate benchmark.")
    parser.add_argument("--config_model", type=str, default="src/config_model_bio.yaml")
    parser.add_argument("--config_dataset", type=str, default="src/config_dataset.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--legacy_model_path", type=str, default=None)
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(WORKSPACE_ROOT.parent / "demo" / "outputs" / "forward_benchmark.json"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_model, config_model_path = load_yaml_config(args.config_model, SCRIPT_DIR, WORKSPACE_ROOT)
    config_data, config_data_path = load_yaml_config(args.config_dataset, SCRIPT_DIR, WORKSPACE_ROOT)
    config_data["__config_dir__"] = str(config_data_path.parent)
    config_data["dataset_path"] = str(resolve_path(config_data["dataset_path"], config_data_path.parent, WORKSPACE_ROOT))
    if config_data.get("split_indices_path"):
        config_data["split_indices_path"] = str(
            resolve_path(config_data["split_indices_path"], config_data_path.parent, WORKSPACE_ROOT)
        )

    factory = DataLoaderFactory(config_data)
    train_data = factory.get_split_data("train")
    val_data = factory.get_split_data("val")
    test_data = factory.get_split_data("test")
    eval_batch_size = config_model.get("evaluation", {}).get("batch_size", config_model["training"]["batch_size"])

    model_path = resolve_path(args.model_path, SCRIPT_DIR, WORKSPACE_ROOT)
    current_model = _load_model(config_model, model_path, args.device)

    report = {
        "current_model": {
            "model_path": str(model_path),
            "train": evaluate_forward_model(current_model, train_data, config_model, args.device, batch_size=eval_batch_size),
            "val": evaluate_forward_model(current_model, val_data, config_model, args.device, batch_size=eval_batch_size),
            "test": evaluate_forward_model(current_model, test_data, config_model, args.device, batch_size=eval_batch_size),
        },
        "baselines": {
            "retrieval": {
                "train": {},
                "val": evaluate_retrieval_baseline(train_data, val_data),
                "test": evaluate_retrieval_baseline(train_data, test_data),
            }
        },
    }

    if args.legacy_model_path:
        legacy_model_path = resolve_path(args.legacy_model_path, SCRIPT_DIR, WORKSPACE_ROOT)
        if legacy_model_path.exists():
            legacy_model = _load_model(config_model, legacy_model_path, args.device)
            report["baselines"]["legacy_6bar_model"] = {
                "model_path": str(legacy_model_path),
                "train": evaluate_forward_model(legacy_model, train_data, config_model, args.device, batch_size=eval_batch_size),
                "val": evaluate_forward_model(legacy_model, val_data, config_model, args.device, batch_size=eval_batch_size),
                "test": evaluate_forward_model(legacy_model, test_data, config_model, args.device, batch_size=eval_batch_size),
            }

    report["semantic_ablation"] = evaluate_semantic_ablation(
        current_model,
        test_data,
        config_model,
        args.device,
        batch_size=eval_batch_size,
    )
    report["gate"] = phase3_gate(report, config_model.get("evaluation", {}))

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = (WORKSPACE_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Forward benchmark saved to {output_path}")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
