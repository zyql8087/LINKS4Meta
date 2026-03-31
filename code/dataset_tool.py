import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import torch
    from torch_geometric.data import Data
    TORCH_IMPORT_ERROR = None
except Exception as e:
    torch = None
    Data = None
    TORCH_IMPORT_ERROR = e

from src.kinematics_extract import extract_kinematics

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_PKL = Path(
    os.environ.get(
        'LINKS_PKL_PATH',
        WORKSPACE_ROOT / 'LINKS-main' / 'biological_6bar_dataset_80k_diverse.pkl',
    )
)
DEFAULT_OUTPUT_PT = Path(
    os.environ.get(
        'LINKS_CURVE_PT_PATH',
        WORKSPACE_ROOT / 'LINKS-main' / 'biological_6bar_dataset_80k_with_curves_NEW.pt',
    )
)
DEFAULT_VIS_DIR = Path(
    os.environ.get('LINKS_KIN_OUT_DIR', SCRIPT_DIR / 'demo' / 'outputs' / 'kinematics' / 'example')
)


def _require_torch_for_convert() -> None:
    if torch is None or Data is None:
        print(f'[ERROR] PyTorch/PyG import failed: {TORCH_IMPORT_ERROR}')
        print('[HINT] Use GMM environment, e.g. from workspace root:')
        print('       run_gmm.cmd GraphMetaMat-LINKS\\dataset_tool.py convert')
        sys.exit(1)


def sample_to_pyg(sample: dict, idx: int):
    _require_torch_for_convert()

    A = sample['A']
    x0 = sample['x0']
    types = sample['types']
    analysis = sample['analysis']

    is_fixed = (types == 1).astype(np.float32)
    is_grounded = np.zeros_like(is_fixed)
    is_grounded[0] = 1

    x_features = np.column_stack([x0, is_fixed, is_grounded])
    edges = np.array(np.where(A)).T
    edge_index = edges.T
    keypoints = np.array([analysis['foot'], analysis['knee'], analysis['ankle']])

    foot_traj, knee_angle, ankle_angle = extract_kinematics(sample)

    return Data(
        x=torch.tensor(x_features, dtype=torch.float32),
        pos=torch.tensor(x0, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        keypoints=torch.tensor(keypoints, dtype=torch.long),
        sample_id=torch.tensor([idx], dtype=torch.long),
        y_foot=torch.tensor(foot_traj, dtype=torch.float32),
        y_knee=torch.tensor(knee_angle, dtype=torch.float32),
        y_ankle=torch.tensor(ankle_angle, dtype=torch.float32),
    )


def cmd_convert(args, raw_data):
    _require_torch_for_convert()

    print(f'Converting {len(raw_data)} samples to PyG PT format...')
    pyg_data = []
    errors = 0

    for idx, sample in enumerate(tqdm(raw_data, desc='Converting')):
        try:
            pyg_data.append(sample_to_pyg(sample, idx))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'[WARN] sample {idx} skipped: {e}')

    print(f'[OK] Converted: {len(pyg_data)}  Skipped: {errors}')
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pyg_data, str(args.output_pt))
    print(f'[OK] Saved PT file: {args.output_pt}')
    print(f'     Size: {args.output_pt.stat().st_size / 1024 / 1024:.2f} MB')


def visualize_kinematics(foot_traj, knee_angle, ankle_angle, sample_idx: int, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.plot(foot_traj[:, 0], foot_traj[:, 1], 'b-', linewidth=2)
    ax1.scatter(foot_traj[0, 0], foot_traj[0, 1], c='green', s=100, zorder=5, label='start')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Foot Trajectory (Sample {sample_idx})')
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    t = np.linspace(0, 360, len(foot_traj))
    ax2.plot(t, foot_traj[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(t, foot_traj[:, 1], 'b-', label='Y', linewidth=2)
    ax2.set_xlabel('Crank angle (deg)')
    ax2.set_ylabel('Position')
    ax2.set_title('Foot Position vs Crank Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(t, knee_angle * 180.0, 'g-', linewidth=2)
    ax3.set_xlabel('Crank angle (deg)')
    ax3.set_ylabel('Knee angle (deg)')
    ax3.set_title('Knee Angle')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(t, ankle_angle * 180.0, 'm-', linewidth=2)
    ax4.set_xlabel('Crank angle (deg)')
    ax4.set_ylabel('Ankle angle (deg)')
    ax4.set_title('Ankle Angle')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def cmd_visualize(args, raw_data):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = min(args.num_samples, len(raw_data))

    print(f'Generating visualizations for {count} samples...')
    for idx in range(count):
        sample = raw_data[idx]
        foot_traj, knee_angle, ankle_angle = extract_kinematics(sample)
        save_path = args.output_dir / f'kinematics_sample_{idx}.png'
        visualize_kinematics(foot_traj, knee_angle, ankle_angle, idx, save_path)
        print(f'[OK] {save_path}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='GraphMetaMat-LINKS dataset utility (convert + visualize).',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--input_pkl',
        type=Path,
        default=DEFAULT_INPUT_PKL,
        help='Input dataset pkl path.',
    )

    subparsers = parser.add_subparsers(dest='command', help='subcommands')
    parser_conv = subparsers.add_parser('convert', help='Convert pkl to PT with curve targets.')
    parser_conv.add_argument(
        '--output_pt',
        type=Path,
        default=DEFAULT_OUTPUT_PT,
        help='Output PT file path.',
    )

    parser_vis = subparsers.add_parser('visualize', help='Generate visualization images.')
    parser_vis.add_argument(
        '--output_dir',
        type=Path,
        default=DEFAULT_VIS_DIR,
        help='Directory for PNG outputs.',
    )
    parser_vis.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    if not args.input_pkl.exists():
        print(f'[ERROR] input pkl not found: {args.input_pkl}')
        sys.exit(1)

    print('Loading pkl dataset...')
    with args.input_pkl.open('rb') as f:
        raw_data = pickle.load(f)
    print(f'Loaded samples: {len(raw_data)}')

    if args.command == 'convert':
        cmd_convert(args, raw_data)
    elif args.command == 'visualize':
        cmd_visualize(args, raw_data)


if __name__ == '__main__':
    main()
