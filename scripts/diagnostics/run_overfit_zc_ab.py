"""小集 overfit + z_c 消融实验（架构无关）。

用途：在同一份小子集上现场构建码本、把一个全新策略过拟合，然后做 z_c 消融，
对比旧/新架构对目标曲线条件的依赖程度。

由于同一进程的代码只能是一种架构（旧 flatten-MLP/无FiLM vs 新序列编码器/FiLM），
本脚本本身架构无关：用旧代码跑一次 (--arch_label old)，用新代码跑一次 (--arch_label new)，
两次都现场重建码本并从零训练，因此完全自洽、无 checkpoint 不匹配问题。

判读：过拟合到训练集后，把 z_c 置零/打乱若使 topology_exact / action_code_accuracy 明显下降，
说明该架构确实利用了目标曲线条件；新架构的 drop 应显著大于旧架构。
"""

import argparse
import collections
import json
import pickle
import sys
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
INVERSE_IL_DIR = WORKSPACE_ROOT / "scripts" / "inverse_il"
for _p in (str(WORKSPACE_ROOT), str(INVERSE_IL_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.inverse.action_codebook import load_action_codebook
from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy
from src.inverse.phase4_il import ensure_multistep_expert_paths, group_paths_by_trace
from train_inverse_bio import PreBatchedLoader, _forward_phase4_batch
from run_zc_conditioning_ablation import TRACKED_KEYS, _ZeroCurveWrapper, _ShuffleCurveWrapper, _deltas

FAMILIES = ("6bar", "7bar", "8bar", "9bar")


def _parse_args():
    parser = argparse.ArgumentParser(description="Overfit a fresh policy on a tiny subset and ablate z_c.")
    parser.add_argument("--source_pkl", type=str,
                        default="../../demo/workspace_archive/links_output_runs/data_gen_v2_pilot2k_20260331/diverse_dataset_v2.pkl")
    parser.add_argument("--arch_label", type=str, required=True, choices=["old", "new"])
    parser.add_argument("--per_family", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="../../demo/outputs/zc_overfit_ab")
    return parser.parse_args()


def _build_subset_pkl(source_pkl: Path, per_family: int, workdir: Path) -> Path:
    """从源 pkl 中每族抽取 per_family 条有 trace 的样本，写入临时小 pkl。"""
    with open(source_pkl, "rb") as handle:
        data = pickle.load(handle)
    by_family = collections.defaultdict(list)
    for sample in data:
        fam = str(sample.get("family_id") or sample.get("family") or "?")
        if not (sample.get("generation_trace") or []):
            continue
        if fam in FAMILIES and len(by_family[fam]) < per_family:
            by_family[fam].append(sample)
    subset = [s for fam in FAMILIES for s in by_family[fam]]
    workdir.mkdir(parents=True, exist_ok=True)
    subset_path = workdir / "subset_raw.pkl"
    with open(subset_path, "wb") as handle:
        pickle.dump(subset, handle)
    print(f"[*] subset: {dict((f, len(by_family[f])) for f in FAMILIES)} -> {len(subset)} samples")
    return subset_path


def _make_cfg(args, num_geometry_codes: int, action_code_dim: int) -> dict:
    return {
        "curve_encoder": {"input_dim": 800, "hidden_dims": [256, 128], "latent_dim": args.latent_dim},
        "gnn_policy": {
            "node_input_dim": 4, "edge_input_dim": 1, "hidden_dim": args.hidden_dim,
            "num_layers": 4, "dropout": 0.0, "num_families": 4,
            "family_embedding_dim": 8, "step_embedding_dim": 8, "context_hidden_dim": args.hidden_dim,
            "max_step_count": 2, "num_geometry_codes": num_geometry_codes, "action_code_dim": action_code_dim,
        },
        "cvae": {"latent_dim": 64},
        "il_training": {"batch_size": 4096, "w_action": 1.0, "w_geometry_code": 1.0,
                        "w_stop": 1.0, "w_step_role": 0.5, "w_step_count": 0.5},
        "constraints": {"min_link_length": 0.05, "min_node_distance": 0.01, "intersection_eps": 1.0e-8},
    }


def _train_overfit(policy, curve_encoder, loader, cfg, epochs, lr, device):
    policy.train()
    curve_encoder.train()
    params = list(policy.parameters()) + list(curve_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    last = {}
    for epoch in range(int(epochs)):
        epoch_metrics = collections.defaultdict(float)
        n = 0
        for batch in loader:
            metrics = _forward_phase4_batch(policy, curve_encoder, batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            metrics["objective"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            for key in TRACKED_KEYS + ("objective",):
                if key in metrics:
                    epoch_metrics[key] += float(metrics[key])
            n += 1
        last = {key: epoch_metrics[key] / max(1, n) for key in epoch_metrics}
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}  obj={last.get('objective', 0):.4f}  "
                  f"topo_exact={last.get('topology_exact', 0):.3f}  "
                  f"code_acc={last.get('action_code_accuracy', 0):.3f}  "
                  f"full_step={last.get('full_step_exact', 0):.3f}")
    return last


def _eval_with_encoder(policy, curve_encoder, paths, cfg, device):
    from train_inverse_bio import eval_epoch_prebatched
    loader = PreBatchedLoader(
        paths, int(cfg["il_training"]["batch_size"]), device, shuffle=False,
        num_geometry_codes=int(cfg["gnn_policy"]["num_geometry_codes"]),
    )
    m = eval_epoch_prebatched(policy, curve_encoder, loader, cfg)
    return {k: float(m[k]) for k in TRACKED_KEYS if k in m}


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workdir = (SCRIPT_DIR / args.workdir).resolve()
    source_pkl = (SCRIPT_DIR / args.source_pkl).resolve()
    output_json = (SCRIPT_DIR / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    subset_pkl = _build_subset_pkl(source_pkl, args.per_family, workdir)
    steps_out = workdir / f"subset_steps_{args.arch_label}.pt"
    step_paths = ensure_multistep_expert_paths(
        pkl_path=str(subset_pkl), output_path=str(steps_out), use_cached=False,
        action_codebook_cfg={}, constraint_cfg={"min_link_length": 0.05, "min_node_distance": 0.01},
    )
    action_codebook = load_action_codebook(str(steps_out))
    entries = sorted(action_codebook.get("entries", []), key=lambda e: int(e["id"]))
    num_codes = max(1, len(entries))
    code_dim = int(action_codebook.get("code_dim", len(entries[0]["vector"]) if entries else 6))

    cfg = _make_cfg(args, num_codes, code_dim)
    policy = GNNPolicy(cfg).to(device)
    curve_encoder = CurveEncoder(
        input_dim=cfg["curve_encoder"]["input_dim"],
        hidden_dims=cfg["curve_encoder"]["hidden_dims"],
        latent_dim=cfg["curve_encoder"]["latent_dim"],
    ).to(device)
    cb = torch.tensor([e["vector"] for e in entries], dtype=torch.float32)
    policy.set_action_codebook(cb.to(device), buckets=action_codebook.get("bucket_to_ids", {}))

    print(f"[*] arch={args.arch_label}  num_codes={num_codes}  train_steps={len(step_paths)}  device={device}")
    train_loader = PreBatchedLoader(
        step_paths, int(cfg["il_training"]["batch_size"]), device, shuffle=True,
        num_geometry_codes=num_codes,
    )
    final_train = _train_overfit(policy, curve_encoder, train_loader, cfg, args.epochs, args.lr, device)

    normal = _eval_with_encoder(policy, curve_encoder, step_paths, cfg, device)
    zero = _eval_with_encoder(policy, _ZeroCurveWrapper(curve_encoder, args.latent_dim).to(device), step_paths, cfg, device)
    shuffle = _eval_with_encoder(policy, _ShuffleCurveWrapper(curve_encoder, seed=args.seed).to(device), step_paths, cfg, device)

    report = {
        "phase": "overfit_zc_ablation",
        "arch_label": args.arch_label,
        "config": {"per_family": args.per_family, "epochs": args.epochs, "lr": args.lr,
                   "num_codes": num_codes, "num_train_steps": int(len(step_paths)),
                   "num_traces": int(len(group_paths_by_trace(step_paths))), "device": str(device)},
        "final_train_metrics": {k: float(v) for k, v in final_train.items()},
        "ablation": {"normal": normal, "zero_zc": zero, "shuffle_zc": shuffle},
        "drop_vs_normal": {"zero_zc": _deltas(normal, zero), "shuffle_zc": _deltas(normal, shuffle)},
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\n[OK] saved {output_json}")
    print(f"=== arch={args.arch_label}  z_c 消融 (normal -> zero / shuffle) ===")
    for k in TRACKED_KEYS:
        if k in normal:
            print(f"  {k:24s}: {normal[k]:.3f} | zero {zero.get(k,0):.3f} (drop {normal[k]-zero.get(k,0):+.3f})"
                  f" | shuffle {shuffle.get(k,0):.3f} (drop {normal[k]-shuffle.get(k,0):+.3f})")


if __name__ == "__main__":
    main()
