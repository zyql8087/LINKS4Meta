"""
P0 诊断: surrogate best-of-K 目标命中（target-matching），替代误导性的"精确克隆专家" success。

背景
----
现有 headline `reconstruction_success_rate`（`phase4_il.py::evaluate_multistep_reconstruction`）
判定成功的条件是 `pred_u/v/w == 专家` **且** `pred_code == action_code_id`（1907 码里精确撞中专家那一个），
全程不调 forward surrogate、不算"产出运动与 target 的误差"。因此它测的是"逐字克隆专家"，
不是项目真正的目标"逆向设计：产出运动命中 target 的机构"。对 8/9bar 这个数在算术上注定 ~0.016。

本脚本用对的口径：对每条 held-out target，
  (1) greedy   —— 现有确定性单条 rollout（今天的路径）；
  (2) best-of-K —— 已有的 MCTS/beam + 冻结 surrogate 重排（P0 方法，src/inverse/mcts.py）；
  (3) expert    —— 专家自己的最终机构（reconstruct_expert_final_graph）作自校准参考。
三者都过同一 `_prepare_graph_for_surrogate`（keypoint-based readout，确定性，三条件一致）
+ 同一冻结 surrogate，算 `joint_score`（= w_foot*foot + w_knee*knee + w_ankle*ankle 的目标误差）。

success 用**自校准**口径，避免拍脑袋的绝对阈值：
  success_vs_expert = 生成机构的 joint_score <= 专家机构的 joint_score * (1 + tol)
即"生成的机构在同一 scorer 下，把 target 匹配得和专家自己的机构一样好（或更好）"。
这天然兼容逆向设计的一对多：找到不同但同样命中 target 的机构也算成功。
同时附带几个绝对阈值 success 作交叉参考。

注意口径
--------
- 这里的 readout 是 keypoint-based 确定性分配（_infer_semantic_masks 优先级），
  不是部署态 SurrogateTargetReadoutAssignment；但 greedy/best-of-K/expert 三条件用同一套，
  故 A/B 公平。要上部署态 readout 是另一条正交改动。
- joint_score 由冻结 surrogate 打分，surrogate 在 9bar/ankle 较弱（见 handoff §3），
  故 9bar 的命中数字上限受 surrogate 保真度限制——这正是 P2 要补的，不在本脚本范围。

用法
----
& F:/Anaconda/envs/GMM/python.exe scripts/diagnostics/run_il_surrogate_target_matching.py \
    --config src/_config_newarch_oracle_merge.yaml \
    --checkpoint demo/outputs/il_v8/merge_newarch/checkpoints/model_inverse_il_merge_newarch_oracle.pt \
    --output_json demo/outputs/il_v8/merge_newarch/reports/surrogate_target_matching.json \
    --max_traces_per_family 64 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.experiment_utils import compute_joint_metrics_batch
from src.inverse.inference_runtime import encode_target, load_inverse_bundle
from src.inverse.mcts import MCTS
from src.inverse.phase4_il import (
    ensure_multistep_expert_paths,
    filter_paths_by_families,
    group_paths_by_trace,
    load_step_split,
    subset_by_indices,
)
from src.inverse.phase5_rl import (
    build_trace_dataset,
    reconstruct_expert_final_graph,
)
from src.inverse.rl_env import (
    _build_surrogate_readout_assigner,
    _prepare_graph_for_surrogate,
    apply_j_operator,
    load_frozen_surrogate,
    validate_graph_structure,
)

FAMILY_ORDER = ("6bar", "7bar", "8bar", "9bar")

# 自校准 success 容差：生成 joint <= 专家 joint *(1+tol) 记为命中。
VS_EXPERT_TOLERANCES = (0.0, 0.10, 0.25)
# 交叉参考用的绝对 joint_score 阈值。
ABS_THRESHOLDS = (0.30, 0.50)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="P0: surrogate best-of-K target-matching diagnostics (replaces exact-clone success)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="src/config_inverse.yaml",
                        help="提供 reward/mcts 默认段（merge overlay 未定义这两段）。")
    parser.add_argument("--forward_model", type=str, default=None,
                        help="覆盖冻结 surrogate checkpoint 路径（merge config 的绝对路径已失效，"
                             "canonical 路径见 config_inverse.yaml 的 ../../demo/.../phase3_full/）。")
    parser.add_argument("--config_forward", type=str, default=None,
                        help="覆盖 forward 模型配置路径（默认用 cfg.paths.config_forward）。")
    parser.add_argument("--max_traces_per_family", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=None,
                        help="覆盖 mcts.top_k_rollouts/beam_width（best-of-K 的 K）。默认用 base config。")
    parser.add_argument("--readout_batch_size", type=int, default=1024,
                        help="surrogate-target readout 每次 GPU 批大小（增大以减少 kernel launch）。")
    parser.add_argument("--readout_max_candidates", type=int, default=32,
                        help="readout 只对 rule-prior top-N 候选跑 surrogate（GPU 主要开销）；0=全量。")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=6,
                        help="并行进程数（瓶颈是 CPU 端 MCTS 动作枚举，多进程才能喂满 GPU）。1=单进程。")
    parser.add_argument("--limit_smoke", type=int, default=None,
                        help="只跑前 N 条 trace 做冒烟测试（不分 family 平衡）。")
    return parser.parse_args()


def _resolve_cfg(config_path: str, base_config_path: str):
    cfg, resolved_path = load_yaml_config(config_path, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        (
            "pkl_dataset",
            "precomputed_split_input",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_split_output",
            "forward_model",
            "config_forward",
        ),
        config_dir=resolved_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    # merge overlay 不含 reward/mcts；从 base config_inverse.yaml 注入，保持与项目打分口径一致。
    base_cfg, _ = load_yaml_config(base_config_path, SCRIPT_DIR, WORKSPACE_ROOT)
    for section in ("reward", "mcts"):
        if section not in cfg and section in base_cfg:
            cfg[section] = base_cfg[section]
    return cfg, resolved_path


def _select_balanced_test_traces(step_paths, split, max_traces_per_family, limit_smoke):
    """从 held-out test split 取每族最多 N 条 trace，重建为 trace_dataset。"""
    test_paths = subset_by_indices(step_paths, split["test_indices"])
    selected_step_paths = []
    selected_trace_ids = {}
    for family in FAMILY_ORDER:
        family_paths = filter_paths_by_families(test_paths, [family])
        traces = group_paths_by_trace(family_paths)
        trace_ids = {int(trace[0]["trace_id"]) for trace in traces[: max(0, int(max_traces_per_family))]}
        selected_trace_ids[family] = sorted(trace_ids)
        selected_step_paths.extend(item for item in family_paths if int(item["trace_id"]) in trace_ids)
    trace_dataset = build_trace_dataset(selected_step_paths)
    if limit_smoke is not None:
        trace_dataset = trace_dataset[: max(1, int(limit_smoke))]
    return trace_dataset, selected_trace_ids


def _score_graph(graph, *, family_index, step_index, expected_j_steps, target, surrogate, device,
                 reward_cfg, constraint_cfg, readout_assigner):
    """对单个机构图打 joint_score（目标命中误差）。无效或异常返回 None。"""
    if graph is None:
        return None, "no_graph"
    is_valid, info = validate_graph_structure(graph, constraint_cfg)
    if not is_valid:
        return None, str(info.get("reason", "invalid_structure")) if isinstance(info, dict) else "invalid_structure"
    try:
        prepared = _prepare_graph_for_surrogate(
            graph,
            family_index=int(family_index),
            step_index=int(step_index),
            expected_j_steps=int(expected_j_steps),
            target=target,
            readout_assigner=readout_assigner,
        )
        batch = Batch.from_data_list([prepared]).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch)
        metrics = compute_joint_metrics_batch(
            pred_foot.cpu(), pred_knee.cpu(), pred_ankle.cpu(), target, reward_cfg
        )
        return float(metrics["joint_score"][0].item()), None
    except Exception as exc:  # noqa: BLE001 - 诊断脚本，记录原因即可
        return None, f"{type(exc).__name__}"


def _greedy_rollout_graph(bundle, trace, cfg, device):
    """确定性单条 rollout，返回最终图（异常返回 None）。"""
    z_c = encode_target(bundle["curve_encoder"], trace["target"], device)
    current_graph = copy.deepcopy(trace["base_data"])
    expected_j_steps = int(trace["expected_j_steps"])
    steps_done = 0
    for step_idx in range(expected_j_steps + 1):
        context = {
            "family_index": trace["family_index"],
            "step_index": step_idx,
            "expected_j_steps": expected_j_steps,
            "can_stop": step_idx > 0,
            "stop_threshold": cfg.get("reward", {}).get("stop_threshold", 0.5),
        }
        try:
            actions, _, _ = bundle["agent"].batch_select_actions(
                [current_graph], z_c, deterministic=True, contexts=[context],
            )
        except Exception:  # noqa: BLE001
            return None, steps_done
        action = actions[0]
        if action is None or bool(action.get("stop", False)):
            break
        try:
            current_graph = apply_j_operator(
                current_graph, action["u"], action["v"], action["w"], action["n1"], action["n2"],
            )
        except Exception:  # noqa: BLE001
            return None, steps_done
        steps_done += 1
    return current_graph, steps_done


def _percentile(values, q):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _mean(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def _summarize_family(records):
    """对一族的 per-trace 记录聚合：valid 率、joint 分布、各口径 success 率。"""
    n = max(1, len(records))
    expert = [r["expert_joint"] for r in records]
    greedy = [r["greedy_joint"] for r in records]
    bestk = [r["bestk_joint"] for r in records]

    def _valid_rate(vals):
        return float(sum(1 for v in vals if v is not None) / n)

    def _success_vs_expert(gen_vals, tol):
        hit = 0
        for r, g in zip(records, gen_vals):
            e = r["expert_joint"]
            if g is None:
                continue
            # 专家自身无效（surrogate 打不出分）时，以绝对阈值兜底判定。
            ref = e if (e is not None and np.isfinite(e)) else None
            if ref is None:
                hit += int(g <= ABS_THRESHOLDS[-1])
            else:
                hit += int(g <= ref * (1.0 + tol))
        return float(hit / n)

    def _success_abs(gen_vals, thr):
        return float(sum(1 for v in gen_vals if v is not None and v <= thr) / n)

    out = {
        "trace_count": int(len(records)),
        "valid_rate": {
            "expert": _valid_rate(expert),
            "greedy": _valid_rate(greedy),
            "best_of_k": _valid_rate(bestk),
        },
        "joint_score": {
            "expert": {"mean": _mean(expert), "p50": _percentile(expert, 50), "p90": _percentile(expert, 90)},
            "greedy": {"mean": _mean(greedy), "p50": _percentile(greedy, 50), "p90": _percentile(greedy, 90)},
            "best_of_k": {"mean": _mean(bestk), "p50": _percentile(bestk, 50), "p90": _percentile(bestk, 90)},
        },
        "success_vs_expert": {
            f"tol_{tol:g}": {
                "greedy": _success_vs_expert(greedy, tol),
                "best_of_k": _success_vs_expert(bestk, tol),
            }
            for tol in VS_EXPERT_TOLERANCES
        },
        "success_abs": {
            f"joint_le_{thr:g}": {
                "greedy": _success_abs(greedy, thr),
                "best_of_k": _success_abs(bestk, thr),
            }
            for thr in ABS_THRESHOLDS
        },
    }
    # best-of-K 相对 greedy 的命中增益（P0 的直接价值）。
    paired = [(r["greedy_joint"], r["bestk_joint"]) for r in records
              if r["greedy_joint"] is not None and r["bestk_joint"] is not None]
    if paired:
        improved = sum(1 for g, b in paired if b < g - 1e-9)
        out["bestk_vs_greedy"] = {
            "paired_count": len(paired),
            "bestk_strictly_better_rate": float(improved / len(paired)),
            "mean_joint_reduction": float(np.mean([g - b for g, b in paired])),
        }
    return out


def _score_one_trace(trace, *, bundle, surrogate, cfg, reward_cfg, constraint_cfg,
                     device, readout_batch, readout_max):
    """对单条 trace 算 greedy / best-of-K / expert 三个目标命中 joint_score，返回 (family, record)。"""
    family = str(trace["family_id"])
    family_index = int(trace["family_index"])
    expected_j_steps = int(trace["expected_j_steps"])
    target = trace["target"]

    # 部署态 surrogate-target readout（GPU 加速：大批 + rule-prior top-N 剪枝）。
    # 同一 assigner 注入 MCTS 并用于 greedy/expert，三条件语义角色赋值一致、A/B 公平。
    readout_assigner = _build_surrogate_readout_assigner(
        surrogate, reward_cfg, device,
        family_index=family_index, step_index=expected_j_steps, expected_j_steps=expected_j_steps,
        batch_size=int(readout_batch), max_surrogate_candidates=readout_max,
    )
    reranker = MCTS(bundle["agent"], surrogate, cfg, device, readout_assigner=readout_assigner)

    # (3) expert 参考机构
    try:
        expert_graph = reconstruct_expert_final_graph(trace)
    except Exception:  # noqa: BLE001
        expert_graph = None
    expert_joint, expert_reason = _score_graph(
        expert_graph, family_index=family_index, step_index=expected_j_steps,
        expected_j_steps=expected_j_steps, target=target,
        surrogate=surrogate, device=device, reward_cfg=reward_cfg,
        constraint_cfg=constraint_cfg, readout_assigner=readout_assigner,
    )

    # (1) greedy 单条 rollout
    greedy_graph, greedy_steps = _greedy_rollout_graph(bundle, trace, cfg, device)
    greedy_joint, greedy_reason = _score_graph(
        greedy_graph, family_index=family_index, step_index=greedy_steps,
        expected_j_steps=expected_j_steps, target=target,
        surrogate=surrogate, device=device, reward_cfg=reward_cfg,
        constraint_cfg=constraint_cfg, readout_assigner=readout_assigner,
    )

    # (2) best-of-K surrogate 重排（直接取 MCTS 已算的 joint_score）
    z_c = encode_target(bundle["curve_encoder"], target, device)
    result = reranker.rerank_rollouts(
        trace["base_data"], z_c, target,
        family_index=family_index, expected_j_steps=expected_j_steps,
    )
    best = result.get("best")
    if best is not None and best.get("valid", 0.0) >= 0.5 and np.isfinite(best.get("joint_score", float("inf"))):
        bestk_joint = float(best["joint_score"])
        bestk_reason = None
    else:
        bestk_joint = None
        bestk_reason = (best or {}).get("invalid_reason", "no_valid_candidate")

    return family, {
        "trace_id": int(trace["trace_id"]),
        "expert_joint": expert_joint, "expert_reason": expert_reason,
        "greedy_joint": greedy_joint, "greedy_reason": greedy_reason,
        "bestk_joint": bestk_joint, "bestk_reason": bestk_reason,
        "bestk_candidates": int(len(result.get("candidates", []))),
    }


# 每个 worker 进程加载一次模型，存进模块全局，避免逐 trace 重载。
_WORKER: dict = {}


def _worker_init(cfg, checkpoint, forward_model, config_forward, device_str, readout_batch, readout_max):
    device = torch.device(device_str)
    bundle = load_inverse_bundle(
        cfg, checkpoint, device, allow_fresh_fallback=False, require_geometry_code_ready=True,
    )
    if bundle is None:
        raise RuntimeError(f"worker failed to load IL checkpoint: {checkpoint}")
    surrogate, _ = load_frozen_surrogate(forward_model, config_forward, device)
    _WORKER.update(
        bundle=bundle, surrogate=surrogate, cfg=cfg, device=device,
        reward_cfg=cfg.get("reward", {}), constraint_cfg=cfg.get("constraints", {}),
        readout_batch=readout_batch, readout_max=readout_max,
    )


def _worker_score(trace):
    w = _WORKER
    return _score_one_trace(
        trace, bundle=w["bundle"], surrogate=w["surrogate"], cfg=w["cfg"],
        reward_cfg=w["reward_cfg"], constraint_cfg=w["constraint_cfg"],
        device=w["device"], readout_batch=w["readout_batch"], readout_max=w["readout_max"],
    )


def main():
    args = _parse_args()
    cfg, config_path = _resolve_cfg(args.config, args.base_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.top_k is not None:
        cfg.setdefault("mcts", {})
        cfg["mcts"]["top_k_rollouts"] = int(args.top_k)
        cfg["mcts"]["beam_width"] = int(args.top_k)
    reward_cfg = cfg.get("reward", {})

    # 数据 + held-out split
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
    trace_dataset, selected_trace_ids = _select_balanced_test_traces(
        step_paths, split, args.max_traces_per_family, args.limit_smoke
    )

    forward_model_path = args.forward_model or cfg["paths"]["forward_model"]
    config_forward_path = args.config_forward or cfg["paths"]["config_forward"]
    if not Path(forward_model_path).exists():
        raise FileNotFoundError(
            f"forward surrogate checkpoint not found: {forward_model_path}. "
            f"Pass --forward_model with the canonical path "
            f"(config_inverse.yaml resolves to GraphMetaMat-LINKS/demo/.../graphmetamat_links_phase3_full/model_bio_best.pt)."
        )
    readout_max = int(args.readout_max_candidates) or None
    num_workers = max(1, int(args.num_workers))

    # 大数据集只在主进程加载用于构建 trace_dataset；worker 只拿 trace 记录(小)+各自加载模型。
    # 这里释放 step_paths/split 以给 worker 进程腾 RAM。
    del step_paths, split
    gc.collect()

    by_family = defaultdict(list)
    t_loop = time.perf_counter()
    print(f"[run] scoring {len(trace_dataset)} held-out traces "
          f"(K={cfg.get('mcts', {}).get('beam_width')}, device={device}, workers={num_workers}, "
          f"readout_batch={args.readout_batch_size}, readout_top_n={readout_max}) ...")

    init_args = (cfg, args.checkpoint, forward_model_path, config_forward_path,
                 str(device), int(args.readout_batch_size), readout_max)

    done = 0
    if num_workers == 1:
        # 单进程：主进程加载模型，顺序跑（便于调试）。
        _worker_init(*init_args)
        for trace in trace_dataset:
            family, record = _worker_score(trace)
            by_family[family].append(record)
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(trace_dataset)} done ({(time.perf_counter()-t_loop)/done:.1f}s/trace)")
    else:
        # 多进程：CPU 端 MCTS 枚举是瓶颈，多进程并行并让 GPU 持续有活干。
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=init_args) as pool:
            for family, record in pool.imap_unordered(_worker_score, trace_dataset, chunksize=1):
                by_family[family].append(record)
                done += 1
                if done % 25 == 0:
                    print(f"  ... {done}/{len(trace_dataset)} done ({(time.perf_counter()-t_loop)/done:.1f}s/trace)")

    print(f"[timing] loop_total={time.perf_counter()-t_loop:.1f}s for {len(trace_dataset)} traces "
          f"({(time.perf_counter()-t_loop)/max(1,len(trace_dataset)):.2f}s/trace, workers={num_workers})")

    # 聚合
    per_family = {fam: _summarize_family(by_family[fam]) for fam in sorted(by_family)}
    overall_records = [r for fam in by_family for r in by_family[fam]]
    report = {
        "phase": "phase4_il_surrogate_target_matching_p0",
        "note": (
            "Honest target-matching metric. success_vs_expert = generated joint_score <= "
            "expert's own mechanism joint_score *(1+tol) under the SAME frozen surrogate. "
            "Replaces the exact-clone reconstruction_success_rate (phase4_il.py:1376) which "
            "demands pred_code==action_code_id over 1907 codes and never scores target matching."
        ),
        "config": {
            "config_path": str(config_path),
            "checkpoint": str(args.checkpoint),
            "forward_model": str(forward_model_path),
            "device": str(device),
            "K_top_k_rollouts": int(cfg.get("mcts", {}).get("top_k_rollouts", 8)),
            "K_beam_width": int(cfg.get("mcts", {}).get("beam_width", 8)),
            "reward_cfg": {k: reward_cfg[k] for k in sorted(reward_cfg)} if isinstance(reward_cfg, dict) else {},
            "max_traces_per_family": int(args.max_traces_per_family),
            "vs_expert_tolerances": list(VS_EXPERT_TOLERANCES),
            "abs_thresholds": list(ABS_THRESHOLDS),
            "readout_batch_size": int(args.readout_batch_size),
            "readout_max_candidates": readout_max,
            "readout_note": ("deployment SurrogateTargetReadoutAssignment, identical assigner for "
                             "greedy/best-of-K/expert; GPU-accelerated via large batch + rule-prior "
                             "top-N surrogate pruning (top-N can only make joint_score conservative)."),
            "selected_trace_ids": selected_trace_ids,
        },
        "overall": _summarize_family(overall_records),
        "per_family": per_family,
        "per_trace": {fam: by_family[fam] for fam in sorted(by_family)},
    }

    ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[OK] surrogate target-matching diagnostics saved to {args.output_json}")

    # 控制台速览
    for fam in FAMILY_ORDER:
        if fam not in per_family:
            continue
        s = per_family[fam]
        sv = s["success_vs_expert"]["tol_0.25"]
        js = s["joint_score"]
        print(f"  [{fam}] n={s['trace_count']:>3} | "
              f"success_vs_expert@0.25 greedy={sv['greedy']:.3f} bestK={sv['best_of_k']:.3f} | "
              f"joint p50 expert={_fmt(js['expert']['p50'])} greedy={_fmt(js['greedy']['p50'])} bestK={_fmt(js['best_of_k']['p50'])}")


def _fmt(v):
    return "  na " if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
