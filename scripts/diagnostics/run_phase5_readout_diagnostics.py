import argparse
import copy
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.inference_runtime import encode_target, load_rollout_bundle_with_fallback
from src.inverse.phase4_il import _build_step_base_graph, _curve_tensor, family_name_to_index
from src.inverse.rl_env import apply_j_operator
from src.inverse.readout_assignment import RuleBasedReadoutAssignment, SurrogateTargetReadoutAssignment
from src.inverse.rl_env import (
    MechanismEnv,
    _infer_semantic_masks,
    batch_compute_phase5_rewards,
    load_frozen_surrogate,
    validate_graph_structure,
)


DEFAULT_OUTPUT_JSON = WORKSPACE_ROOT.parent / "demo" / "outputs" / "rl" / "phase5_readout_diagnostics.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small-sample phase5 readout diagnostics on real LINKS4Meta traces.")
    parser.add_argument("--config", type=str, default="src/config_inverse.yaml")
    parser.add_argument("--per_family", type=int, default=2)
    parser.add_argument("--families", nargs="*", default=["6bar", "7bar", "8bar", "9bar"])
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


def _load_raw_samples(path: str) -> List[dict]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected list dataset from '{path}', got {type(data)!r}")
    return data


def _raw_sample_lookup(raw_samples: Sequence[dict]) -> Dict[int, dict]:
    lookup: Dict[int, dict] = {}
    for idx, sample in enumerate(raw_samples):
        sample_id = int(sample.get("id", sample.get("sample_id", idx)))
        lookup[sample_id] = sample
    return lookup


def _load_split_indices(path: str, raw_samples: Sequence[dict]) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict):
        raise TypeError(f"Expected dict split file from '{path}', got {type(split)!r}")

    id_to_index = {int(sample.get("id", idx)): idx for idx, sample in enumerate(raw_samples)}

    def _resolve(values: Sequence[int]) -> List[int]:
        if values and all(isinstance(v, int) and 0 <= int(v) < len(raw_samples) for v in values):
            return [int(v) for v in values]
        resolved: List[int] = []
        for value in values:
            index = id_to_index.get(int(value))
            if index is not None:
                resolved.append(index)
        return resolved

    return {
        "train": _resolve(split.get("train", [])),
        "val": _resolve(split.get("val", [])),
        "test": _resolve(split.get("test", [])),
    }


def _family_name(sample: dict) -> str:
    family = sample.get("family") or sample.get("family_id") or "unknown"
    return str(family)


def _select_sample_indices(
    raw_samples: Sequence[dict],
    split_indices: Sequence[int],
    *,
    families: Sequence[str],
    per_family: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(int(seed))
    grouped: Dict[str, List[int]] = defaultdict(list)
    family_set = {str(name) for name in families}
    for sample_index in split_indices:
        family_name = _family_name(raw_samples[int(sample_index)])
        if family_name in family_set:
            grouped[family_name].append(int(sample_index))

    selected: List[int] = []
    for family_name in families:
        candidates = list(grouped.get(str(family_name), []))
        if not candidates:
            continue
        order = rng.permutation(len(candidates)).tolist()
        for idx in order[: min(len(candidates), int(per_family))]:
            selected.append(candidates[idx])
    return selected


def _truth_from_raw_sample(sample: dict) -> dict:
    analysis = sample.get("analysis") or {}
    return {
        "hip": int(analysis["hip"]),
        "knee": int(analysis["knee"]),
        "ankle": int(analysis["ankle"]),
        "foot": int(analysis["foot"]),
    }


def _motion_from_raw_sample(sample: dict) -> np.ndarray:
    analysis = sample.get("analysis") or {}
    return np.asarray(analysis["x_sol"], dtype=np.float32)


def _trace_record_from_raw_sample(sample: dict, sample_index: int) -> dict:
    family_name = _family_name(sample)
    family_index = family_name_to_index(family_name)
    generation_trace = list(sample.get("generation_trace") or [])
    expected_j_steps = int(sample.get("step_count", len(generation_trace)))
    base_data, _ = _build_step_base_graph(sample, 0)

    step_paths: List[dict] = []
    x0 = np.asarray(sample["x0"], dtype=np.float32)
    for step_index, step in enumerate(generation_trace):
        step_base, node_remap = _build_step_base_graph(sample, step_index)
        u = int(step["u"])
        v = int(step["v"])
        w = int(step["w"])
        n1 = int(step["n1"])
        n2 = int(step["n2"])
        u_r = node_remap[u]
        v_r = node_remap[v]
        w_r = node_remap[w]
        step_paths.append(
            {
                "base_data": step_base,
                "action_topo": torch.tensor([u_r, v_r, w_r], dtype=torch.long),
                "action_geo": torch.tensor(
                    [
                        float(x0[n1][0]),
                        float(x0[n1][1]),
                        float(x0[n2][0]),
                        float(x0[n2][1]),
                    ],
                    dtype=torch.float32,
                ),
            }
        )

    sample_id = int(sample.get("id", sample.get("sample_id", sample_index)))
    return {
        "trace_id": sample_id,
        "sample_id": sample_id,
        "family_id": family_name,
        "family_index": family_index,
        "expected_j_steps": expected_j_steps,
        "base_data": base_data,
        "target": {
            "y_foot": _curve_tensor(sample, "foot_curve", 0),
            "y_knee": _curve_tensor(sample, "knee_curve", 1),
            "y_ankle": _curve_tensor(sample, "ankle_curve", 2),
        },
        "step_paths": step_paths,
    }


def _reconstruct_expert_final_graph(trace_record: dict) -> object:
    graph = copy.deepcopy(trace_record["base_data"])
    for step in trace_record["step_paths"]:
        graph = apply_j_operator(
            graph,
            int(step["action_topo"][0].item()),
            int(step["action_topo"][1].item()),
            int(step["action_topo"][2].item()),
            step["action_geo"][:2].detach().cpu().numpy(),
            step["action_geo"][2:].detach().cpu().numpy(),
        )
    return graph


def _mask_indices(mask: torch.Tensor) -> List[int]:
    return [int(idx) for idx in torch.nonzero(mask.view(-1), as_tuple=False).view(-1).tolist()]


def _strip_cached_semantics(graph):
    stripped = copy.deepcopy(graph)
    for attr_name in (
        "mask_hip",
        "mask_knee",
        "mask_ankle",
        "mask_foot",
        "keypoints",
        "knee_idx",
        "semantic_feature_layout",
    ):
        if hasattr(stripped, attr_name):
            try:
                delattr(stripped, attr_name)
            except AttributeError:
                pass
    if hasattr(stripped, "semantic_dirty"):
        stripped.semantic_dirty = torch.tensor([False], dtype=torch.bool)
    return stripped


def _masks_payload(
    graph,
    *,
    target: Optional[dict],
    motion,
    strip_cached_semantics: bool = False,
    readout_assigner=None,
) -> dict:
    graph_input = _strip_cached_semantics(graph) if strip_cached_semantics else graph
    mask_hip, mask_knee, mask_ankle, mask_foot = _infer_semantic_masks(
        graph_input,
        target=target,
        motion=motion,
        readout_assigner=readout_assigner,
    )
    return {
        "hip": _mask_indices(mask_hip),
        "knee": _mask_indices(mask_knee),
        "ankle": _mask_indices(mask_ankle),
        "foot": _mask_indices(mask_foot),
    }


def _assignment_payload(result) -> Optional[dict]:
    if result is None:
        return None
    top_candidates = []
    for candidate in result.top_candidates[:3]:
        top_candidates.append(
            {
                "keypoints": candidate.keypoints(),
                "path": list(candidate.path),
                "score": float(candidate.score),
                "score_breakdown": dict(candidate.score_breakdown),
            }
        )
    return {
        "method": str(result.method),
        "keypoints": dict(result.keypoints),
        "path": list(result.path),
        "score": float(result.score),
        "score_breakdown": dict(result.score_breakdown),
        "candidate_count": int(result.candidate_count),
        "top_candidates": top_candidates,
    }


def _structure_payload(graph, *, constraint_cfg: dict) -> dict:
    valid, info = validate_graph_structure(graph, constraint_cfg)
    return {
        "valid": bool(valid),
        "reason": None if bool(valid) else str((info or {}).get("reason", "invalid_structure")),
        "num_nodes": int(graph.x.size(0)),
        "num_edges_directed": int(graph.edge_index.size(1)),
    }


def _terminal_reward_payload(
    surrogate,
    graph,
    *,
    target: dict,
    reward_cfg: dict,
    device,
    family_index: int,
    expected_j_steps: int,
    constraint_cfg: dict,
) -> dict:
    rewards, payloads = batch_compute_phase5_rewards(
        surrogate,
        [graph],
        target,
        reward_cfg,
        device,
        step_indices=[int(expected_j_steps)],
        stop_flags=[True],
        expected_j_steps=int(expected_j_steps),
        family_index=int(family_index),
        constraint_cfg=constraint_cfg,
    )
    reward_value, valid_flag = rewards[0]
    payload = dict(payloads[0])
    payload["reward"] = float(reward_value)
    payload["valid_flag"] = bool(valid_flag)
    return payload


def _policy_rollout_trace(bundle, trace_record: dict, env) -> tuple[list[dict], list[dict], list[tuple[float, bool]], list[dict]]:
    target = trace_record["target"]
    z_c = encode_target(bundle["curve_encoder"], target, env.device)
    env.reset(
        target,
        trace_record["base_data"],
        z_c,
        family_id=str(trace_record["family_id"]),
        family_index=int(trace_record["family_index"]),
        expected_j_steps=int(trace_record["expected_j_steps"]),
        fixed_stop_by_family=True,
    )

    action_log: List[dict] = []
    graph_snapshots: List[dict] = []
    max_decisions = max(1, int(trace_record["expected_j_steps"]) + 1)
    for step_idx in range(max_decisions):
        context = {
            "family_index": int(trace_record["family_index"]),
            "step_index": int(step_idx),
            "expected_j_steps": int(trace_record["expected_j_steps"]),
            "can_stop": bool(step_idx > 0),
            "stop_threshold": env.reward_cfg.get("stop_threshold", 0.5),
        }
        actions, log_probs, _, diagnostics = bundle["agent"].batch_select_actions(
            [env.current_graph],
            z_c,
            deterministic=True,
            return_diagnostics=True,
            contexts=[context],
        )
        action = actions[0] if actions else None
        if action is None:
            action = {"stop": True}
        log_prob = float(log_probs[0]) if log_probs else 0.0
        diagnostic = diagnostics[0] if diagnostics else {}
        _, _, done, info = env.step(action)
        graph_event = env._reward_events[-1] if env._reward_events else None

        action_payload = {
            "step_index": int(step_idx),
            "stop": bool(action.get("stop", False)),
            "log_prob": log_prob,
            "allow_stop": bool(action.get("allow_stop", False)),
            "stop_probability": float(action.get("stop_probability", diagnostic.get("stop_probability", 0.0))),
            "u": None if action.get("u") is None else int(action["u"]),
            "v": None if action.get("v") is None else int(action["v"]),
            "w": None if action.get("w") is None else int(action["w"]),
            "n1": None if action.get("n1") is None else [float(v) for v in np.asarray(action["n1"], dtype=np.float32).reshape(-1).tolist()],
            "n2": None if action.get("n2") is None else [float(v) for v in np.asarray(action["n2"], dtype=np.float32).reshape(-1).tolist()],
            "info_stop": bool(info.get("stop", False)),
            "diagnostics": diagnostic,
        }
        action_log.append(action_payload)
        if graph_event is not None:
            graph_snapshots.append(
                {
                    "step_index": int(graph_event["step_index"]),
                    "stop": bool(graph_event["stop"]),
                    "graph": graph_event["graph"],
                }
            )
        if done:
            break

    if env.surrogate is None:
        return action_log, graph_snapshots, [], []
    rewards, payloads = env.compute_episode_rewards()
    return action_log, graph_snapshots, rewards, payloads


def main() -> None:
    args = _parse_args()
    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    resolve_mapping_paths(
        cfg["paths"],
        (
            "pkl_dataset",
            "precomputed_split_input",
            "forward_model",
            "config_forward",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_split_output",
            "rl_model_output",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = (WORKSPACE_ROOT / output_path).resolve()
    ensure_parent_dir(str(output_path))

    raw_samples = _load_raw_samples(cfg["paths"]["pkl_dataset"])
    sample_lookup = _raw_sample_lookup(raw_samples)

    split = _load_split_indices(cfg["paths"]["precomputed_split_input"], raw_samples)
    selected_indices = _select_sample_indices(
        raw_samples,
        split["test"],
        families=args.families,
        per_family=int(args.per_family),
        seed=int(args.seed),
    )
    if not selected_indices:
        raise RuntimeError("No test traces selected for phase5 diagnostics.")
    selected_traces = [_trace_record_from_raw_sample(raw_samples[idx], idx) for idx in selected_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = None
    surrogate_load_error = None
    try:
        surrogate, _ = load_frozen_surrogate(cfg["paths"]["forward_model"], cfg["paths"]["config_forward"], device)
    except Exception as exc:
        surrogate = None
        surrogate_load_error = f"{type(exc).__name__}: {exc}"
    bundle = load_rollout_bundle_with_fallback(
        cfg,
        device,
        preferred_model_type="rl",
        allow_fresh_fallback=False,
        require_geometry_code_ready=False,
    )
    if bundle is None or not bool(bundle.get("checkpoint_loaded")):
        raise RuntimeError("Failed to load inverse rollout bundle for preferred='rl'")

    rule_assigner = RuleBasedReadoutAssignment(top_k=3)
    reward_cfg = dict(cfg.get("reward", {}))
    constraint_cfg = dict(cfg.get("constraints", {}))
    env_max_steps = int(cfg.get("rl_training", {}).get("steps_per_episode", 3))

    samples_payload: List[dict] = []
    summary_counters = {
        "count": 0,
        "expert_deploy_exact": 0,
        "expert_deploy_matches_canonical": 0,
        "rollout_valid": 0,
        "rollout_reward_beats_expert": 0,
        "rollout_final_assignment_exists": 0,
    }
    by_family: Dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "expert_deploy_exact": 0,
            "expert_deploy_matches_canonical": 0,
            "rollout_valid": 0,
            "rollout_reward_beats_expert": 0,
        }
    )

    for trace in selected_traces:
        sample_id = int(trace["sample_id"])
        raw_sample = sample_lookup.get(sample_id)
        if raw_sample is None:
            raise KeyError(f"Sample id {sample_id} missing from raw dataset lookup.")
        family_name = str(trace["family_id"])
        truth = _truth_from_raw_sample(raw_sample)
        motion = _motion_from_raw_sample(raw_sample)
        target = trace["target"]
        expected_j_steps = int(trace["expected_j_steps"])
        family_index = int(trace["family_index"])

        expert_graph = _reconstruct_expert_final_graph(trace)
        surrogate_assigner = (
            SurrogateTargetReadoutAssignment(
                surrogate,
                top_k=3,
                batch_size=64,
                metric_cfg=reward_cfg,
                device=device,
                family_index=family_index,
                step_index=expected_j_steps,
                expected_j_steps=expected_j_steps,
            )
            if surrogate is not None
            else None
        )
        expert_canonical = rule_assigner.assign(expert_graph, target=target, motion=motion)
        expert_deploy_rule = rule_assigner.assign(expert_graph, target=target, motion=None)
        expert_deploy = (
            surrogate_assigner.assign(expert_graph, target=target, motion=None)
            if surrogate_assigner is not None
            else expert_deploy_rule
        )
        expert_reward = None
        if surrogate is not None:
            expert_reward = _terminal_reward_payload(
                surrogate,
                expert_graph,
                target=target,
                reward_cfg=reward_cfg,
                device=device,
                family_index=family_index,
                expected_j_steps=expected_j_steps,
                constraint_cfg=constraint_cfg,
            )
        expert_canonical_matches_truth = bool(expert_canonical is not None and expert_canonical.keypoints == truth)
        expert_deploy_matches_truth = bool(expert_deploy is not None and expert_deploy.keypoints == truth)
        expert_deploy_matches_canonical = bool(
            expert_canonical is not None and expert_deploy is not None and expert_deploy.keypoints == expert_canonical.keypoints
        )

        env = MechanismEnv(
            surrogate,
            reward_cfg,
            max_steps=env_max_steps,
            device=device,
            constraint_cfg=constraint_cfg,
        )
        action_log, graph_snapshots, rollout_rewards, rollout_payloads = _policy_rollout_trace(bundle, trace, env)
        rollout_final_graph = graph_snapshots[-1]["graph"] if graph_snapshots else trace["base_data"]
        rollout_structure = _structure_payload(rollout_final_graph, constraint_cfg=constraint_cfg)
        rollout_rule_assignment = rule_assigner.assign(rollout_final_graph, target=target, motion=None)
        rollout_assignment = (
            surrogate_assigner.assign(rollout_final_graph, target=target, motion=None)
            if surrogate_assigner is not None
            else rollout_rule_assignment
        )
        rollout_reward = None
        rollout_reward_margin = None
        if rollout_payloads and rollout_rewards:
            rollout_reward = dict(rollout_payloads[-1])
            rollout_reward["reward"] = float(rollout_rewards[-1][0])
            rollout_reward["valid_flag"] = bool(rollout_rewards[-1][1])
        elif surrogate is not None:
            rollout_reward = {
                "reward_total": float(reward_cfg.get("r_invalid_penalty", -1.0)),
                "reward": float(reward_cfg.get("r_invalid_penalty", -1.0)),
                "valid_flag": False,
            }
        if rollout_reward is not None and expert_reward is not None:
            rollout_reward_margin = float(rollout_reward["reward"] - expert_reward["reward"])

        summary_counters["count"] += 1
        summary_counters["expert_deploy_exact"] += int(expert_deploy_matches_truth)
        summary_counters["expert_deploy_matches_canonical"] += int(expert_deploy_matches_canonical)
        summary_counters["rollout_valid"] += int(rollout_structure["valid"])
        summary_counters["rollout_final_assignment_exists"] += int(rollout_assignment is not None)
        if rollout_reward_margin is not None:
            summary_counters["rollout_reward_beats_expert"] += int(rollout_reward_margin > 1e-6)

        family_stats = by_family[family_name]
        family_stats["count"] += 1
        family_stats["expert_deploy_exact"] += int(expert_deploy_matches_truth)
        family_stats["expert_deploy_matches_canonical"] += int(expert_deploy_matches_canonical)
        family_stats["rollout_valid"] += int(rollout_structure["valid"])
        if rollout_reward_margin is not None:
            family_stats["rollout_reward_beats_expert"] += int(rollout_reward_margin > 1e-6)

        samples_payload.append(
            {
                "sample_id": sample_id,
                "trace_id": int(trace["trace_id"]),
                "family": family_name,
                "family_index": family_index,
                "expected_j_steps": expected_j_steps,
                "truth": truth,
                "expert": {
                    "structure": _structure_payload(expert_graph, constraint_cfg=constraint_cfg),
                    "canonical_assignment": _assignment_payload(expert_canonical),
                    "deploy_rule_assignment": _assignment_payload(expert_deploy_rule),
                    "deploy_surrogate_target_assignment": _assignment_payload(expert_deploy),
                    "deploy_assignment": _assignment_payload(expert_deploy),
                    "canonical_masks": _masks_payload(
                        expert_graph,
                        target=target,
                        motion=motion,
                        strip_cached_semantics=True,
                    ),
                    "deploy_masks": _masks_payload(
                        expert_graph,
                        target=target,
                        motion=None,
                        strip_cached_semantics=True,
                        readout_assigner=surrogate_assigner,
                    ),
                    "canonical_matches_truth": expert_canonical_matches_truth,
                    "deploy_matches_truth": expert_deploy_matches_truth,
                    "deploy_matches_canonical": expert_deploy_matches_canonical,
                    "terminal_reward": expert_reward,
                },
                "rollout": {
                    "actions": action_log,
                    "num_events": len(graph_snapshots),
                    "structure": rollout_structure,
                    "rule_final_assignment": _assignment_payload(rollout_rule_assignment),
                    "final_assignment": _assignment_payload(rollout_assignment),
                    "final_masks": _masks_payload(
                        rollout_final_graph,
                        target=target,
                        motion=None,
                        strip_cached_semantics=True,
                        readout_assigner=surrogate_assigner,
                    ),
                    "reward_events": [
                        {
                            "reward": float(reward_value),
                            "valid": bool(valid_flag),
                            "payload": dict(payload),
                        }
                        for (reward_value, valid_flag), payload in zip(rollout_rewards, rollout_payloads)
                    ],
                    "terminal_reward": rollout_reward,
                    "reward_margin_vs_expert": rollout_reward_margin,
                },
            }
        )

    total = max(1, int(summary_counters["count"]))
    reward_comparable_count = sum(
        1
        for sample in samples_payload
        if sample["expert"]["terminal_reward"] is not None and sample["rollout"]["terminal_reward"] is not None
    )
    summary = {
        "num_samples": int(summary_counters["count"]),
        "expert_deploy_exact_rate": float(summary_counters["expert_deploy_exact"] / total),
        "expert_deploy_matches_canonical_rate": float(summary_counters["expert_deploy_matches_canonical"] / total),
        "rollout_valid_rate": float(summary_counters["rollout_valid"] / total),
        "rollout_final_assignment_rate": float(summary_counters["rollout_final_assignment_exists"] / total),
        "reward_comparable_count": int(reward_comparable_count),
        "rollout_reward_beats_expert_rate": (
            float(summary_counters["rollout_reward_beats_expert"] / max(1, reward_comparable_count))
            if reward_comparable_count > 0
            else None
        ),
        "by_family": {
            family_name: {
                "count": int(values["count"]),
                "expert_deploy_exact_rate": float(values["expert_deploy_exact"] / max(1, int(values["count"]))),
                "expert_deploy_matches_canonical_rate": float(values["expert_deploy_matches_canonical"] / max(1, int(values["count"]))),
                "rollout_valid_rate": float(values["rollout_valid"] / max(1, int(values["count"]))),
                "rollout_reward_beats_expert_rate": (
                    float(values["rollout_reward_beats_expert"] / max(1, int(values["count"])))
                    if reward_comparable_count > 0
                    else None
                ),
            }
            for family_name, values in sorted(by_family.items())
        },
    }

    report = {
        "config": {
            "config_path": str(config_path),
            "per_family": int(args.per_family),
            "families": [str(item) for item in args.families],
            "seed": int(args.seed),
            "device": str(device),
            "requested_model_type": str(bundle.get("requested_model_type")),
            "selected_model_type": str(bundle.get("selected_model_type")),
            "fallback_used": bool(bundle.get("fallback_used", False)),
            "rl_checkpoint": str(bundle.get("checkpoint_path")),
            "rl_checkpoint_loaded": bool(bundle.get("checkpoint_loaded")),
            "rl_checkpoint_warning": bundle.get("checkpoint_warning"),
            "rl_geometry_code_ready": bool(bundle.get("geometry_code_ready", True)),
            "rl_geometry_code_issue": bundle.get("geometry_code_issue"),
            "rl_geometry_code_status": bundle.get("geometry_code_status"),
            "rollout_bundle_candidates": bundle.get("bundle_candidates"),
            "surrogate_checkpoint": str(cfg["paths"]["forward_model"]),
            "surrogate_load_error": surrogate_load_error,
            "split_source": "official_precomputed_split",
            "split_path": str(cfg["paths"]["precomputed_split_input"]),
        },
        "summary": summary,
        "samples": samples_payload,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved phase5 readout diagnostics to: {output_path}")


if __name__ == "__main__":
    main()
