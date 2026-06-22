import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data_load import DataLoaderFactory
from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN


def plot_kinematics(pred_foot, y_foot, pred_knee, y_knee, pred_ankle, y_ankle, sample_id, save_dir):
    """Plot predicted curves against ground-truth curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(y_foot[:, 0], y_foot[:, 1], label="Ground Truth", color="blue", linestyle="dashed", alpha=0.7)
    axes[0].plot(pred_foot[:, 0], pred_foot[:, 1], label="Prediction", color="red", alpha=0.7)
    axes[0].set_title(f"Foot Trajectory (Sample {sample_id})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend()
    axes[0].grid(True)

    steps = range(y_knee.shape[0])
    axes[1].plot(steps, y_knee, label="Ground Truth", color="blue", linestyle="dashed", alpha=0.7)
    axes[1].plot(steps, pred_knee, label="Prediction", color="red", alpha=0.7)
    axes[1].set_title("Knee Angle")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Angle (Norm)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(steps, y_ankle, label="Ground Truth", color="blue", linestyle="dashed", alpha=0.7)
    axes[2].plot(steps, pred_ankle, label="Prediction", color="red", alpha=0.7)
    axes[2].set_title("Ankle Angle")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Angle (Norm)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"test_sample_{sample_id}.png"))
    plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parents[1]
    demo_root = workspace_root.parent / "demo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_model", type=str, default=str(workspace_root / "src" / "config_model_bio.yaml"))
    parser.add_argument("--config_dataset", type=str, default=str(workspace_root / "src" / "config_dataset.yaml"))
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(demo_root / "outputs" / "checkpoints" / "graphmetamat_links" / "model_bio_best.pt"),
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(demo_root / "outputs" / "test" / "latest"),
        help="Directory to save test visualizations",
    )
    parser.add_argument("--num_vis", type=int, default=5, help="Number of test samples to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    vis_dir = os.path.abspath(args.output_dir)
    os.makedirs(vis_dir, exist_ok=True)

    with open(args.config_model, "r", encoding="utf-8") as f:
        config_model = yaml.safe_load(f)
    with open(args.config_dataset, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    print(f"Running on device: {args.device}")

    print("Loading test dataset...")
    factory = DataLoaderFactory(config_data)
    test_loader = factory.create_test_loader(
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    model = BioKinematicsGNN(config_model).to(args.device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))
        print(f"Successfully loaded weights from {args.model_path}")
    else:
        print(f"Warning: Model weights {args.model_path} not found! Testing with randomly initialized weights.")

    model.eval()

    total_samples = 0
    total_foot_loss = 0.0
    total_knee_loss = 0.0
    total_ankle_loss = 0.0

    print("\nStarting evaluation on test set...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(args.device)

            pred_foot, pred_knee, pred_ankle = model(data)

            y_foot = data.y_foot.view_as(pred_foot)
            y_knee = data.y_knee.view_as(pred_knee)
            y_ankle = data.y_ankle.view_as(pred_ankle)

            foot_loss = torch.nn.functional.mse_loss(pred_foot, y_foot).item()
            knee_loss = torch.nn.functional.mse_loss(pred_knee, y_knee).item()
            ankle_loss = torch.nn.functional.mse_loss(pred_ankle, y_ankle).item()

            total_foot_loss += foot_loss
            total_knee_loss += knee_loss
            total_ankle_loss += ankle_loss
            total_samples += 1

            if i < args.num_vis:
                p_foot = pred_foot[0].cpu().numpy()
                g_foot = y_foot[0].cpu().numpy()
                p_knee = pred_knee[0].cpu().numpy()
                g_knee = y_knee[0].cpu().numpy()
                p_ankle = pred_ankle[0].cpu().numpy()
                g_ankle = y_ankle[0].cpu().numpy()

                sample_id = getattr(data, "sample_id", [i])[0].item() if hasattr(data, "sample_id") else i
                plot_kinematics(p_foot, g_foot, p_knee, g_knee, p_ankle, g_ankle, sample_id, vis_dir)
                output_path = os.path.join(vis_dir, f"test_sample_{sample_id}.png")
                print(f"Saved visualization for Sample {sample_id} to '{output_path}'")

    avg_foot_loss = total_foot_loss / total_samples
    avg_knee_loss = total_knee_loss / total_samples
    avg_ankle_loss = total_ankle_loss / total_samples

    print("\n--- Testing Results Summary ---")
    print(f"Total Test Samples Evaluated: {total_samples}")
    print(f"Average Foot Trajectory MSE:  {avg_foot_loss:.6f}")
    print(f"Average Knee Angle MSE:       {avg_knee_loss:.6f}")
    print(f"Average Ankle Angle MSE:      {avg_ankle_loss:.6f}")
    print("-------------------------------")
    print(f"Test visualizations are saved in: {vis_dir}")


if __name__ == "__main__":
    main()
