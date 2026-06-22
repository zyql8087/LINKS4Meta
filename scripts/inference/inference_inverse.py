from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import load_yaml_config, resolve_mapping_paths
from src.inverse.inference_runtime import (
    demo_root_from_workspace,
    encode_target,
    load_rollout_bundle_with_fallback,
    rollout_trace_with_mcts,
)
from src.inverse.phase4_il import ensure_multistep_expert_paths
from src.inverse.phase5_rl import build_trace_dataset
from src.inverse.rl_env import _prepare_graph_for_surrogate, load_frozen_surrogate


def plot_kinematics_result(y_foot, y_knee, y_ankle, pred_foot, pred_knee, pred_ankle, sample_id, save_dir):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes[0].plot(y_foot[:, 0], y_foot[:, 1], label="Target", color="blue", linestyle="dashed", alpha=0.7)
    axes[0].plot(pred_foot[:, 0], pred_foot[:, 1], label="Generated", color="red", alpha=0.7)
    axes[0].set_title(f"Foot Trajectory ({sample_id})")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True)

    steps = range(y_knee.shape[0])
    axes[1].plot(steps, y_foot[:, 0], label="Target X", color="lightblue", linestyle="dashed")
    axes[1].plot(steps, y_foot[:, 1], label="Target Y", color="blue", linestyle="dashed")
    axes[1].plot(steps, pred_foot[:, 0], label="Generated X", color="lightcoral")
    axes[1].plot(steps, pred_foot[:, 1], label="Generated Y", color="red")
    axes[1].set_title("Foot Position")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(steps, y_knee, label="Target", color="blue", linestyle="dashed", alpha=0.7)
    axes[2].plot(steps, pred_knee, label="Generated", color="red", alpha=0.7)
    axes[2].set_title("Knee Angle")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(steps, y_ankle, label="Target", color="blue", linestyle="dashed", alpha=0.7)
    axes[3].plot(steps, pred_ankle, label="Generated", color="red", alpha=0.7)
    axes[3].set_title("Ankle Angle")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"inference_result_sample_{sample_id}.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config_inverse.yaml")
    parser.add_argument("--model_type", type=str, choices=["il", "rl"], default="rl")
    parser.add_argument("--output_dir", type=str, default=str(demo_root_from_workspace(WORKSPACE_ROOT) / "outputs" / "inference"))
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        (
            "pkl_dataset",
            "forward_model",
            "config_forward",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "rl_model_output",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (WORKSPACE_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_rollout_bundle_with_fallback(
        cfg,
        device,
        preferred_model_type=args.model_type,
        allow_fresh_fallback=False,
        require_geometry_code_ready=False,
    )
    if bundle is None:
        print(f"[Error] Failed to load inverse rollout bundle for preferred='{args.model_type}'")
        return
    if not bool(bundle.get("geometry_code_ready", True)):
        print(f"[Error] Inverse bundle is not rollout-ready: {bundle.get('geometry_code_issue')}")
        return
    if bool(bundle.get("fallback_used", False)):
        print(
            f"[Inference] requested `{args.model_type}` weights but fell back to "
            f"`{bundle.get('selected_model_type')}` from '{bundle.get('checkpoint_path')}'"
        )
    else:
        print(f"[Inference] Loaded `{bundle.get('selected_model_type')}` weights from '{bundle.get('checkpoint_path')}'")

    surrogate, _ = load_frozen_surrogate(cfg["paths"]["forward_model"], cfg["paths"]["config_forward"], device)
    step_paths = ensure_multistep_expert_paths(
        pkl_path=cfg["paths"]["pkl_dataset"],
        output_path=cfg["paths"].get("il_multistep_dataset_output", cfg["paths"]["il_dataset_output"]),
        use_cached=True,
    )
    traces = build_trace_dataset(step_paths)
    test_traces = traces[-args.num_samples :]

    for idx, trace in enumerate(test_traces):
        print(f"\n--- Inferring Sample {idx + 1}/{len(test_traces)} ---")
        target = trace["target"]
        z_c = encode_target(bundle["curve_encoder"], target, device)

        start_t = time.time()
        graph, search_info = rollout_trace_with_mcts(bundle, trace, surrogate, cfg, device)
        elapsed = time.time() - start_t
        if graph is None:
            print("[Inference] No valid rollout returned by inference-time MCTS reranker; skipping sample.")
            continue
        best = search_info["best"]
        print(
            f"[Inference] rerank finished in {elapsed:.2f}s, "
            f"candidates={search_info['candidate_count']}, steps={best['step_count']}"
        )

        eval_graph = _prepare_graph_for_surrogate(
            graph,
            family_index=int(trace["family_index"]),
            step_index=int(best["step_count"]),
            expected_j_steps=int(trace["expected_j_steps"]),
        )
        batch_eval = Batch.from_data_list([eval_graph]).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch_eval)

        plot_kinematics_result(
            target["y_foot"].cpu().numpy(),
            target["y_knee"].cpu().numpy(),
            target["y_ankle"].cpu().numpy(),
            pred_foot.squeeze(0).cpu().numpy(),
            pred_knee.squeeze(0).cpu().numpy(),
            pred_ankle.squeeze(0).cpu().numpy(),
            idx,
            str(output_dir),
        )
        print(f"[Inference] Saved visualization to {output_dir / f'inference_result_sample_{idx}.png'}")


if __name__ == "__main__":
    main()
