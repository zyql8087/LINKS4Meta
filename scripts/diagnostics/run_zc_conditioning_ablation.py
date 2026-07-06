"""z_c 条件信号消融诊断。

目的：量化逆向策略是否真正利用了目标曲线条件 z_c。
做法：在同一测试集上，分别用三种 z_c 评估 teacher-forced 指标并对比：
  - normal  : 真实曲线编码 z_c
  - zero    : 把 z_c 全部置零
  - shuffle : 在 batch 内随机打乱 z_c（图与曲线错配）

判读：
  若 zero / shuffle 相对 normal 指标"几乎不掉"，说明条件信号没被用上
  （拼接式后融合把 z_c 淹没了）；改造后应观察到明显下降，
  即拓扑/几何码准确率对目标曲线敏感。
"""

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
INVERSE_IL_DIR = WORKSPACE_ROOT / "scripts" / "inverse_il"
for _p in (str(WORKSPACE_ROOT), str(INVERSE_IL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.action_codebook import load_action_codebook
from src.inverse.inference_runtime import demo_root_from_workspace, load_inverse_bundle
from src.inverse.phase4_il import (
    attach_oracle_code_targets,
    ensure_multistep_expert_paths,
    load_step_split,
    subset_by_indices,
)
from train_inverse_bio import PreBatchedLoader, eval_epoch_prebatched

# 关注的关键条件敏感指标
TRACKED_KEYS = (
    "action_u_accuracy",
    "action_v_accuracy",
    "action_w_accuracy",
    "topology_exact",
    "action_code_accuracy",
    "action_code_top1",
    "action_code_top3",
    "full_step_exact",
    "stop_accuracy",
    "step_role_accuracy",
)


class _ZeroCurveWrapper(torch.nn.Module):
    """把曲线编码输出整体置零，模拟"无目标条件"。"""

    def __init__(self, base, latent_dim: int):
        super().__init__()
        self.base = base
        self.latent_dim = int(latent_dim)

    def forward(self, y_foot, y_knee, y_ankle):
        z = self.base(y_foot, y_knee, y_ankle)
        return torch.zeros_like(z)


class _ShuffleCurveWrapper(torch.nn.Module):
    """在 batch 内随机打乱 z_c，使图与目标曲线错配。"""

    def __init__(self, base, seed: int = 0):
        super().__init__()
        self.base = base
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def forward(self, y_foot, y_knee, y_ankle):
        z = self.base(y_foot, y_knee, y_ankle)
        if z.size(0) <= 1:
            return z
        perm = torch.randperm(z.size(0), generator=self.generator).to(z.device)
        # 避免恒等置换（小 batch 时多试几次）
        for _ in range(4):
            if not torch.equal(perm, torch.arange(z.size(0), device=z.device)):
                break
            perm = torch.randperm(z.size(0), generator=self.generator).to(z.device)
        return z[perm]


def _parse_args():
    parser = argparse.ArgumentParser(description="Ablate z_c conditioning of the inverse IL policy.")
    parser.add_argument("--config", type=str, default="src/config_inverse.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_cfg(args):
    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        (
            "pkl_dataset",
            "precomputed_split_input",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_split_output",
            "rl_model_output",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    return cfg, config_path


def _eval_mode(policy, curve_encoder, test_paths, cfg, device):
    loader = PreBatchedLoader(
        test_paths,
        int(cfg["il_training"].get("batch_size", 4096)),
        device,
        shuffle=False,
        num_geometry_codes=int(cfg.get("gnn_policy", {}).get("num_geometry_codes", 0) or 0),
    )
    metrics = eval_epoch_prebatched(policy, curve_encoder, loader, cfg)
    return {key: float(metrics[key]) for key in TRACKED_KEYS if key in metrics}


def _deltas(normal: dict, other: dict) -> dict:
    """计算 other 相对 normal 的指标下降（normal - other）。下降越大说明越依赖 z_c。"""
    return {key: float(normal[key] - other.get(key, 0.0)) for key in normal}


def main():
    args = _parse_args()
    cfg, config_path = _resolve_cfg(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = args.checkpoint or cfg["paths"]["il_model_output"]
    output_json = args.output_json
    if output_json is None:
        output_json = str(
            demo_root_from_workspace(WORKSPACE_ROOT) / "outputs" / "il_v5" / "reports" / "zc_conditioning_ablation.json"
        )
    output_json = str(Path(output_json))
    ensure_parent_dir(output_json)

    dataset_path = cfg["paths"].get("il_multistep_dataset_output", cfg["paths"]["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=cfg["paths"]["pkl_dataset"],
        output_path=dataset_path,
        use_cached=True,
        action_codebook_cfg=cfg.get("action_codebook", {}),
        constraint_cfg=cfg.get("constraints", {}),
    )
    split = load_step_split(
        step_paths,
        split_path=cfg["paths"]["il_split_output"],
        precomputed_split_path=cfg["paths"].get("precomputed_split_input"),
        val_ratio=cfg["il_training"].get("val_ratio", 0.1),
        test_ratio=cfg["il_training"].get("test_ratio", 0.1),
        split_seed=cfg["il_training"].get("split_seed", 42),
    )

    action_codebook = load_action_codebook(dataset_path)
    cfg.setdefault("gnn_policy", {})
    cfg["gnn_policy"]["num_geometry_codes"] = max(1, len(action_codebook.get("entries", [])))
    cfg["gnn_policy"]["action_code_dim"] = int(action_codebook.get("code_dim", 6))

    # 若启用 oracle 码目标，需要为 paths 附加相应字段
    if bool(cfg.get("il_training", {}).get("oracle_code_loss", {}).get("enabled", False)):
        step_paths, _ = attach_oracle_code_targets(step_paths, action_codebook, cfg)

    test_paths = subset_by_indices(step_paths, split["test_indices"])

    bundle = load_inverse_bundle(cfg, checkpoint, device, allow_fresh_fallback=False, require_geometry_code_ready=True)
    if bundle is None:
        raise RuntimeError(f"failed to load IL checkpoint: {checkpoint}")
    policy = bundle["policy"]
    curve_encoder = bundle["curve_encoder"]
    latent_dim = int(getattr(policy, "curve_latent_dim", 128))

    normal = _eval_mode(policy, curve_encoder, test_paths, cfg, device)
    zero = _eval_mode(policy, _ZeroCurveWrapper(curve_encoder, latent_dim).to(device), test_paths, cfg, device)
    shuffle = _eval_mode(
        policy, _ShuffleCurveWrapper(curve_encoder, seed=args.shuffle_seed).to(device), test_paths, cfg, device
    )

    report = {
        "phase": "zc_conditioning_ablation",
        "config": {
            "config_path": str(config_path),
            "checkpoint": str(checkpoint),
            "device": str(device),
            "dataset_path": dataset_path,
            "num_test_steps": int(len(test_paths)),
            "shuffle_seed": int(args.shuffle_seed),
        },
        "metrics": {"normal": normal, "zero_zc": zero, "shuffle_zc": shuffle},
        "drop_vs_normal": {
            "zero_zc": _deltas(normal, zero),
            "shuffle_zc": _deltas(normal, shuffle),
        },
        "interpretation": (
            "drop 越大表示策略越依赖目标曲线 z_c。"
            "若 topology_exact / action_code_accuracy 的 drop 接近 0，说明条件信号被忽略，"
            "需加强融合（已引入 LayerNorm 对齐 + FiLM 节点级调制 + 序列化曲线编码器）。"
        ),
    }

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"[OK] z_c conditioning ablation saved to {output_json}")
    print("\n=== 关键指标 (normal -> zero / shuffle) ===")
    for key in TRACKED_KEYS:
        if key in normal:
            print(
                f"  {key:24s}: {normal[key]:.4f}"
                f"  | zero {zero.get(key, 0.0):.4f} (drop {normal[key] - zero.get(key, 0.0):+.4f})"
                f"  | shuffle {shuffle.get(key, 0.0):.4f} (drop {normal[key] - shuffle.get(key, 0.0):+.4f})"
            )


if __name__ == "__main__":
    main()
