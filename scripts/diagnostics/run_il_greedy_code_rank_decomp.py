"""
greedy 选码弱点的归因诊断:目标最优码在策略 logit 排序里排第几?

动机
----
P0(见 surrogate_target_matching.json）显示 best-of-K 远胜 greedy、且 best-of-K 只在策略
自己的 top-k 候选里搜就能 ≈ 专家水平 —— 强烈暗示"好码本来就在候选分布里,只是 greedy 第一名
没选中",即**排序问题**而非码本覆盖问题。本脚本直接证实/证伪:

对每条 held-out trace 的**最终(semantic)决策步**:
  1. 用 greedy 把前缀步建好(8/9bar 的 step0 用 greedy 的选择),得到 prefix 图;
  2. 在 prefix 上跑策略,拿到 greedy 预测的 topology + 该 topology 桶内**全部码的 logit 排序**;
  3. 对桶内 top-N 码逐个 decode→apply J-operator→冻结 surrogate 评 joint_score(对目标);
  4. 找出**目标最优码**(valid 里 joint 最小),记录它在 logit 排序里的名次;
     同时记录 greedy 实际会选的码(logit 序里第一个 valid)的名次与 joint。

判读
----
- oracle-best 多在 top-8 内、但很少 rank-1,且 greedy_joint - oracle_joint 的 gap 明显
  ⇒ 排序问题坐实:好码可达,greedy 没排对 → 改 proposal 的选码训练/打分(oracle_code_loss / rollout_aware)。
- oracle-best 经常掉到 top-8 之外 ⇒ 更偏码本覆盖/打分双重问题。

口径与 P0 一致:同一冻结 surrogate + 部署态 readout(top-N 剪枝),use_sigma_flip 跟随配置。
topology 固定为 greedy 预测值(选码问题的范围;topology 选择是另一个正交问题)。

用法
----
& F:/Anaconda/envs/GMM/python.exe scripts/diagnostics/run_il_greedy_code_rank_decomp.py \
    --config src/_config_newarch_oracle_merge.yaml \
    --checkpoint demo/outputs/il_v8/merge_newarch/checkpoints/model_inverse_il_merge_newarch_oracle.pt \
    --forward_model GraphMetaMat-LINKS/demo/outputs/checkpoints/graphmetamat_links_phase3_full/model_bio_best.pt \
    --output_json demo/outputs/il_v8/merge_newarch/reports/greedy_code_rank_decomp.json \
    --max_traces_per_family 64 --num_workers 6 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.action_codebook import (
    decode_local_dyad_code_candidates,
    family_name_from_index,
    resolve_codebook_bucket_for_step,
)
from src.inverse.experiment_utils import compute_joint_metrics_batch
from src.inverse.inference_runtime import encode_target, load_inverse_bundle
from src.inverse.phase4_il import (
    ensure_multistep_expert_paths,
    filter_paths_by_families,
    group_paths_by_trace,
    load_step_split,
    subset_by_indices,
)
from src.inverse.phase5_rl import build_trace_dataset
from src.inverse.rl_env import (
    _build_surrogate_readout_assigner,
    _prepare_graph_for_surrogate,
    apply_j_operator,
    load_frozen_surrogate,
    validate_graph_structure,
)

FAMILY_ORDER = ("6bar", "7bar", "8bar", "9bar")


def _parse_args():
    p = argparse.ArgumentParser(description="Attribute greedy weakness: rank of target-optimal code in policy ordering.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--base_config", type=str, default="src/config_inverse.yaml")
    p.add_argument("--forward_model", type=str, default=None)
    p.add_argument("--config_forward", type=str, default=None)
    p.add_argument("--max_traces_per_family", type=int, default=64)
    p.add_argument("--max_bucket_codes", type=int, default=256,
                   help="最终步最多枚举桶内多少码(按 logit 序);先廉价 validity 预筛,只对 valid 码跑 surrogate。"
                        "默认覆盖整桶(9bar 最大 128),避免漏掉排名靠后的合法码。")
    p.add_argument("--readout_batch_size", type=int, default=1024)
    p.add_argument("--readout_max_candidates", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit_smoke", type=int, default=None)
    return p.parse_args()


def _resolve_cfg(config_path, base_config_path):
    cfg, rp = load_yaml_config(config_path, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        ("pkl_dataset", "precomputed_split_input", "il_dataset_output",
         "il_multistep_dataset_output", "il_model_output", "il_split_output",
         "forward_model", "config_forward"),
        config_dir=rp.parent, workspace_root=WORKSPACE_ROOT,
    )
    base_cfg, _ = load_yaml_config(base_config_path, SCRIPT_DIR, WORKSPACE_ROOT)
    for section in ("reward", "mcts"):
        if section not in cfg and section in base_cfg:
            cfg[section] = base_cfg[section]
    return cfg, rp


def _select_balanced_test_traces(step_paths, split, max_traces_per_family, limit_smoke):
    test_paths = subset_by_indices(step_paths, split["test_indices"])
    selected = []
    for family in FAMILY_ORDER:
        fam_paths = filter_paths_by_families(test_paths, [family])
        traces = group_paths_by_trace(fam_paths)
        ids = {int(t[0]["trace_id"]) for t in traces[: max(0, int(max_traces_per_family))]}
        selected.extend(item for item in fam_paths if int(item["trace_id"]) in ids)
    trace_dataset = build_trace_dataset(selected)
    if limit_smoke is not None:
        trace_dataset = trace_dataset[: max(1, int(limit_smoke))]
    return trace_dataset


def _single_graph_prediction(logits, valid_mask):
    masked = logits.view(-1).masked_fill(~valid_mask.view(-1), -1e9)
    return int(torch.argmax(masked).item())


def _allowed_code_ids(policy, family_index, step_role_index, step_index, action_topo):
    family_name = family_name_from_index(int(family_index))
    step_role = "semantic" if int(step_role_index) == 1 else "aux"
    bucket = resolve_codebook_bucket_for_step(
        getattr(policy, "action_codebook_buckets", {}),
        family_name, step_role, step_index=int(step_index), action_topo=action_topo,
    )
    allowed = list(getattr(policy, "action_codebook_buckets", {}).get(bucket, []))
    if not allowed:
        allowed = list(range(int(policy.action_codebook.size(0))))
    return [int(idx) for idx in allowed]


def _apply_code_vector(graph, topo, code_vec, constraints, use_sigma_flip):
    """解码码并应用 J-operator,返回第一个合法分支 (is_valid, reason, next_graph)。"""
    try:
        u, v, w = [int(x) for x in topo]
        if min(u, v, w) < 0 or max(u, v, w) >= int(graph.pos.size(0)):
            return False, "topo_oob", None
        branches = decode_local_dyad_code_candidates(
            graph.pos[u].detach().cpu().numpy(),
            graph.pos[v].detach().cpu().numpy(),
            graph.pos[w].detach().cpu().numpy(),
            np.asarray(code_vec, dtype=np.float32),
            include_sigma_flips=bool(use_sigma_flip),
        )
        if not branches:
            return False, "decode_no_branch", None
        last = None
        for n1, n2, _v in branches:
            ng = apply_j_operator(graph, u, v, w, n1, n2)
            ok, reason = validate_graph_structure(ng, constraints)
            if ok:
                return True, "ok", ng
            last = reason
        return False, str(last), None
    except Exception as exc:  # noqa: BLE001
        return False, type(exc).__name__, None


def _joint_of_graph(graph, *, family_index, step_index, expected_j_steps, target,
                    surrogate, device, reward_cfg, readout_assigner):
    """对完整机构图打目标命中 joint_score;失败返回 None。"""
    try:
        prepared = _prepare_graph_for_surrogate(
            graph, family_index=int(family_index), step_index=int(step_index),
            expected_j_steps=int(expected_j_steps), target=target, readout_assigner=readout_assigner,
        )
        batch = Batch.from_data_list([prepared]).to(device)
        with torch.no_grad():
            pf, pk, pa = surrogate(batch)
        metrics = compute_joint_metrics_batch(pf.cpu(), pk.cpu(), pa.cpu(), target, reward_cfg)
        return float(metrics["joint_score"][0].item())
    except Exception:  # noqa: BLE001
        return None


def _greedy_prefix_graph(bundle, trace, cfg, device, num_prefix_steps):
    """用确定性 greedy 建前缀(num_prefix_steps 步);greedy 提前停止/失败返回 None。"""
    if num_prefix_steps <= 0:
        return copy.deepcopy(trace["base_data"])
    z_c = encode_target(bundle["curve_encoder"], trace["target"], device)
    current = copy.deepcopy(trace["base_data"])
    expected = int(trace["expected_j_steps"])
    for step_idx in range(num_prefix_steps):
        context = {
            "family_index": trace["family_index"], "step_index": step_idx,
            "expected_j_steps": expected, "can_stop": step_idx > 0,
            "stop_threshold": cfg.get("reward", {}).get("stop_threshold", 0.5),
        }
        try:
            actions, _, _ = bundle["agent"].batch_select_actions(
                [current], z_c, deterministic=True, contexts=[context])
        except Exception:  # noqa: BLE001
            return None
        a = actions[0]
        if a is None or bool(a.get("stop", False)):
            return None
        try:
            current = apply_j_operator(current, a["u"], a["v"], a["w"], a["n1"], a["n2"])
        except Exception:  # noqa: BLE001
            return None
    return current


def _decompose_one_trace(trace, *, bundle, surrogate, cfg, reward_cfg, constraint_cfg,
                         device, readout_batch, readout_max, max_bucket_codes):
    """对一条 trace 的最终步做选码排序归因,返回 (family, record)。"""
    family = str(trace["family_id"])
    family_index = int(trace["family_index"])
    expected = int(trace["expected_j_steps"])
    target = trace["target"]
    final_step = trace["step_paths"][-1]
    use_sigma_flip = bool(cfg.get("geometry_code_selection", {}).get("use_sigma_flip", True))

    base = {"trace_id": int(trace["trace_id"]), "decomposed": False, "reason": None}

    prefix = _greedy_prefix_graph(bundle, trace, cfg, device, expected - 1)
    if prefix is None:
        base["reason"] = "greedy_no_prefix"
        return family, base

    policy = bundle["policy"]
    curve_encoder = bundle["curve_encoder"]
    readout_assigner = _build_surrogate_readout_assigner(
        surrogate, reward_cfg, device, family_index=family_index,
        step_index=expected, expected_j_steps=expected,
        batch_size=int(readout_batch), max_surrogate_candidates=readout_max,
    )

    z_c = encode_target(curve_encoder, target, device)

    # 关键:greedy topology 要用 agent 实际会选的那个(它跳过"无合法码"的 topology),
    # 而不是 u/v/w logit 的独立 argmax —— 后者对 9bar 常是死 topology(桶里没合法码),
    # 会假性 dec=0。先问 agent 拿真实 greedy 的 (topology, code)。
    ctx = {
        "family_index": family_index, "step_index": int(final_step["step_index"]),
        "expected_j_steps": expected, "can_stop": False,
        "stop_threshold": cfg.get("reward", {}).get("stop_threshold", 0.5),
    }
    ranked = bundle["agent"].rank_action_candidates(prefix, z_c, context=ctx, top_k=1)
    build = [c for c in ranked if not bool(c.get("action", {}).get("stop", False))]
    if not build or "code_id" not in build[0].get("action", {}):
        base["reason"] = "greedy_no_build_action"
        return family, base
    ga = build[0]["action"]
    pred_topo = [int(ga["u"]), int(ga["v"]), int(ga["w"])]
    greedy_code_id = int(ga["code_id"])

    # 该(真实 greedy)topology 桶内的全码 logit 排序
    with torch.no_grad():
        bg = Batch.from_data_list([copy.deepcopy(prefix)]).to(device)
        x_enc = policy.encode_graph(bg)
        outputs = policy.phase4_outputs(
            bg, x_enc, z_c,
            family_ids=torch.tensor([family_index], dtype=torch.long, device=device),
            step_indices=torch.tensor([int(final_step["step_index"])], dtype=torch.long, device=device),
            step_counts=torch.tensor([int(final_step["step_count"])], dtype=torch.long, device=device),
        )
        logits = policy.geometry_code_logits(
            bg, x_enc, outputs["graph_context"],
            torch.tensor([pred_topo], dtype=torch.long, device=device),
        )[0]
        allowed = _allowed_code_ids(policy, family_index, final_step["step_role_index"],
                                    final_step["step_index"], pred_topo)
        allowed_t = torch.tensor(allowed, dtype=torch.long, device=device)
        order = torch.argsort(logits[allowed_t], descending=True)
        ordered_ids = allowed_t[order].detach().cpu().tolist()

    n_bucket = len(ordered_ids)
    enumerated = ordered_ids[: max(1, int(max_bucket_codes))]

    # 全桶廉价 validity 预筛(decode+J+validate,CPU),只对合法码跑 surrogate;
    # 不设评分 cap,避免漏掉 logit 排名靠后但合法的码(9bar 桶达 128,合法码常排得很深)。
    valid_prefilter = []  # (logit_rank(1-based), code_id, graph)
    for rank, code_id in enumerate(enumerated, start=1):
        code_vec = policy.action_codebook[int(code_id)].detach().cpu().numpy()
        ok, _reason, ng = _apply_code_vector(prefix, pred_topo, code_vec, constraint_cfg, use_sigma_flip)
        if ok:
            valid_prefilter.append((rank, int(code_id), ng))

    valid_scored = []  # (logit_rank, code_id, joint)
    for rank, code_id, ng in valid_prefilter:
        j = _joint_of_graph(
            ng, family_index=family_index, step_index=expected, expected_j_steps=expected,
            target=target, surrogate=surrogate, device=device,
            reward_cfg=reward_cfg, readout_assigner=readout_assigner,
        )
        if j is not None:
            valid_scored.append((rank, code_id, j))

    if not valid_scored:
        base["reason"] = "no_valid_scored_code"
        base["n_bucket"] = n_bucket
        base["n_valid_prefilter"] = len(valid_prefilter)
        return family, base

    # greedy = agent 实际选的码(部署 greedy);oracle = valid 里 joint 最小的码
    by_code = {cid: (r, j) for (r, cid, j) in valid_scored}
    if greedy_code_id in by_code:
        greedy_rank, greedy_joint = by_code[greedy_code_id]
    else:
        # agent 的码在我枚举里没评上分(极少见,如 readout 评分失败):退化为 logit 序第一个 valid
        greedy_rank, _gc, greedy_joint = valid_scored[0]
        greedy_code_id = _gc
    oracle_rank, oracle_code, oracle_joint = min(valid_scored, key=lambda t: t[2])

    return family, {
        "trace_id": int(trace["trace_id"]),
        "decomposed": True,
        "reason": None,
        "n_bucket": n_bucket,
        "n_enumerated": len(enumerated),
        "n_valid_scored": len(valid_scored),
        "pred_topo": [int(x) for x in pred_topo],
        "greedy_code_id": int(greedy_code_id),
        "greedy_logit_rank": int(greedy_rank),
        "greedy_joint": float(greedy_joint),
        "oracle_code_id": int(oracle_code),
        "oracle_logit_rank": int(oracle_rank),
        "oracle_joint": float(oracle_joint),
        "joint_gap": float(greedy_joint - oracle_joint),
        "greedy_is_oracle": bool(greedy_code_id == oracle_code),
    }


def _median(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _summarize(records):
    dec = [r for r in records if r.get("decomposed")]
    n_all = max(1, len(records))
    n_dec = max(1, len(dec))
    ranks = [r["oracle_logit_rank"] for r in dec]
    out = {
        "trace_count": int(len(records)),
        "decomposed_count": int(len(dec)),
        "undecomposed_reasons": dict(Counter(r.get("reason") for r in records if not r.get("decomposed"))),
        # oracle-best 落在 logit 排序的哪一档(占 decomposed 的比例)
        "oracle_rank_eq_1": float(sum(1 for x in ranks if x == 1) / n_dec),
        "oracle_rank_le_3": float(sum(1 for x in ranks if x <= 3) / n_dec),
        "oracle_rank_le_8": float(sum(1 for x in ranks if x <= 8) / n_dec),
        "oracle_rank_gt_8": float(sum(1 for x in ranks if x > 8) / n_dec),
        "greedy_is_oracle_rate": float(sum(1 for r in dec if r["greedy_is_oracle"]) / n_dec),
        "median_oracle_logit_rank": _median(ranks),
        "median_greedy_joint": _median([r["greedy_joint"] for r in dec]),
        "median_oracle_joint": _median([r["oracle_joint"] for r in dec]),
        "median_joint_gap": _median([r["joint_gap"] for r in dec]),
        "median_n_bucket": _median([r.get("n_bucket") for r in dec]),
        "median_n_valid_scored": _median([r.get("n_valid_scored") for r in dec]),
        "decompose_rate": float(len(dec) / n_all),
    }
    return out


_WORKER: dict = {}


def _worker_init(cfg, checkpoint, forward_model, config_forward, device_str,
                 readout_batch, readout_max, max_bucket_codes):
    device = torch.device(device_str)
    bundle = load_inverse_bundle(cfg, checkpoint, device, allow_fresh_fallback=False,
                                 require_geometry_code_ready=True)
    if bundle is None:
        raise RuntimeError(f"worker failed to load checkpoint: {checkpoint}")
    surrogate, _ = load_frozen_surrogate(forward_model, config_forward, device)
    _WORKER.update(bundle=bundle, surrogate=surrogate, cfg=cfg, device=device,
                   reward_cfg=cfg.get("reward", {}), constraint_cfg=cfg.get("constraints", {}),
                   readout_batch=readout_batch, readout_max=readout_max, max_bucket_codes=max_bucket_codes)


def _worker_score(trace):
    w = _WORKER
    return _decompose_one_trace(
        trace, bundle=w["bundle"], surrogate=w["surrogate"], cfg=w["cfg"],
        reward_cfg=w["reward_cfg"], constraint_cfg=w["constraint_cfg"], device=w["device"],
        readout_batch=w["readout_batch"], readout_max=w["readout_max"], max_bucket_codes=w["max_bucket_codes"],
    )


def main():
    args = _parse_args()
    cfg, config_path = _resolve_cfg(args.config, args.base_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reward_cfg = cfg.get("reward", {})

    dataset_path = cfg["paths"].get("il_multistep_dataset_output", cfg["paths"]["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=cfg["paths"]["pkl_dataset"], output_path=dataset_path, use_cached=True,
        action_codebook_cfg=cfg.get("action_codebook", {}), constraint_cfg=cfg.get("constraints", {}),
    )
    split = load_step_split(
        step_paths, split_path=cfg["paths"]["il_split_output"],
        precomputed_split_path=cfg["paths"].get("precomputed_split_input"),
        val_ratio=cfg["il_training"].get("val_ratio", 0.1),
        test_ratio=cfg["il_training"].get("test_ratio", 0.1),
        split_seed=cfg["il_training"].get("split_seed", 42),
    )
    trace_dataset = _select_balanced_test_traces(step_paths, split, args.max_traces_per_family, args.limit_smoke)

    forward_model_path = args.forward_model or cfg["paths"]["forward_model"]
    config_forward_path = args.config_forward or cfg["paths"]["config_forward"]
    if not Path(forward_model_path).exists():
        raise FileNotFoundError(f"forward surrogate checkpoint not found: {forward_model_path}")
    readout_max = int(args.readout_max_candidates) or None
    num_workers = max(1, int(args.num_workers))

    del step_paths, split
    gc.collect()

    by_family = defaultdict(list)
    t0 = time.perf_counter()
    print(f"[run] greedy code-rank decomp on {len(trace_dataset)} traces "
          f"(device={device}, workers={num_workers}, max_bucket_codes={args.max_bucket_codes}) ...")
    init_args = (cfg, args.checkpoint, forward_model_path, config_forward_path, str(device),
                 int(args.readout_batch_size), readout_max, int(args.max_bucket_codes))

    done = 0
    if num_workers == 1:
        _worker_init(*init_args)
        for trace in trace_dataset:
            fam, rec = _worker_score(trace)
            by_family[fam].append(rec)
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(trace_dataset)} ({(time.perf_counter()-t0)/done:.1f}s/trace)")
    else:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=init_args) as pool:
            for fam, rec in pool.imap_unordered(_worker_score, trace_dataset, chunksize=1):
                by_family[fam].append(rec)
                done += 1
                if done % 25 == 0:
                    print(f"  ... {done}/{len(trace_dataset)} ({(time.perf_counter()-t0)/done:.1f}s/trace)")

    print(f"[timing] {time.perf_counter()-t0:.1f}s for {len(trace_dataset)} traces "
          f"({(time.perf_counter()-t0)/max(1,len(trace_dataset)):.2f}s/trace)")

    per_family = {fam: _summarize(by_family[fam]) for fam in sorted(by_family)}
    overall = _summarize([r for fam in by_family for r in by_family[fam]])
    report = {
        "phase": "phase4_il_greedy_code_rank_decomp",
        "note": ("Per final greedy step: rank of the target-optimal code within the policy's logit "
                 "ordering of its bucket (topology fixed to greedy). oracle in top-8 but rank>1 + "
                 "large greedy-vs-oracle joint gap => ranking problem, not codebook coverage."),
        "config": {
            "config_path": str(config_path), "checkpoint": str(args.checkpoint),
            "forward_model": str(forward_model_path), "device": str(device),
            "max_bucket_codes": int(args.max_bucket_codes), "max_traces_per_family": int(args.max_traces_per_family),
            "readout_batch_size": int(args.readout_batch_size), "readout_max_candidates": readout_max,
        },
        "overall": overall,
        "per_family": per_family,
        "per_trace": {fam: by_family[fam] for fam in sorted(by_family)},
    }
    ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"[OK] saved to {args.output_json}")

    for fam in FAMILY_ORDER:
        if fam not in per_family:
            continue
        s = per_family[fam]
        print(f"  [{fam}] dec={s['decomposed_count']}/{s['trace_count']} | "
              f"oracle rank==1={s['oracle_rank_eq_1']:.2f} <=3={s['oracle_rank_le_3']:.2f} "
              f"<=8={s['oracle_rank_le_8']:.2f} >8={s['oracle_rank_gt_8']:.2f} | "
              f"greedy=oracle={s['greedy_is_oracle_rate']:.2f} | "
              f"joint greedy={_f(s['median_greedy_joint'])} oracle={_f(s['median_oracle_joint'])} "
              f"gap={_f(s['median_joint_gap'])}")


def _f(v):
    return "na" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
