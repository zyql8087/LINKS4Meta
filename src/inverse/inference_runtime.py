from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from src.inverse.curve_encoder import CurveEncoder
from src.inverse.action_codebook import codebook_tensor, load_action_codebook
from src.inverse.gnn_policy import GNNPolicy, policy_load_incompatibilities
from src.inverse.mcts import MCTS
from src.inverse.rl_agent import PPOAgent
from src.inverse.rl_env import apply_j_operator


def demo_root_from_workspace(workspace_root: Path) -> Path:
    return (workspace_root.parent / "demo").resolve()


def inspect_inverse_checkpoint_geometry_code_state(
    checkpoint: Optional[dict],
    *,
    checkpoint_loaded: bool,
    action_codebook: Optional[dict],
    action_codebook_source: Optional[str],
    dataset_path: Optional[str],
) -> dict:
    policy_state = {}
    if isinstance(checkpoint, dict):
        maybe_policy = checkpoint.get("policy")
        if isinstance(maybe_policy, dict):
            policy_state = maybe_policy

    policy_keys = list(policy_state.keys())
    has_geometry_code_head = any(str(key).startswith("geometry_code_head.") for key in policy_keys)
    has_legacy_geo_head = any(str(key).startswith("geo_head.") for key in policy_keys)
    has_action_heads = any(
        str(key).startswith(prefix)
        for key in policy_keys
        for prefix in ("action_u_head.", "action_v_head.", "action_w_head.")
    )
    if has_geometry_code_head:
        checkpoint_variant = "geometry_code_head"
    elif has_legacy_geo_head:
        checkpoint_variant = "legacy_geo_head"
    elif has_action_heads:
        checkpoint_variant = "action_topology_only"
    elif checkpoint is None:
        checkpoint_variant = "missing_checkpoint"
    else:
        checkpoint_variant = "unknown"

    codebook_entry_count = int(len((action_codebook or {}).get("entries", [])))
    codebook_bucket_count = int(len((action_codebook or {}).get("bucket_to_ids", {})))
    codebook_path = None
    if dataset_path:
        codebook_path = str(Path(dataset_path).with_suffix(".codebook.pt"))

    issues = []
    warnings = []
    if not checkpoint_loaded:
        issues.append("inverse checkpoint weights were not loaded")
    if not has_geometry_code_head:
        if has_legacy_geo_head:
            issues.append("checkpoint predates geometry_code_head and only contains legacy geo_head")
        else:
            issues.append("checkpoint does not contain geometry_code_head parameters")
    if action_codebook is None:
        if codebook_path:
            issues.append(
                f"no action codebook loaded; expected checkpoint-embedded codebook or external codebook at '{codebook_path}'"
            )
        else:
            issues.append("no action codebook loaded and no dataset path was configured for fallback")
    elif codebook_entry_count <= 0:
        issues.append("loaded action codebook has zero entries")

    if action_codebook is not None and codebook_bucket_count <= 0:
        warnings.append("loaded action codebook has no bucket_to_ids mapping; rollout will fall back to the full code range")

    ready = bool(checkpoint_loaded and has_geometry_code_head and codebook_entry_count > 0)
    issue = "; ".join(issues) if issues else None
    return {
        "ready": ready,
        "issue": issue,
        "warnings": warnings,
        "checkpoint_variant": checkpoint_variant,
        "checkpoint_has_geometry_code_head": bool(has_geometry_code_head),
        "checkpoint_has_legacy_geo_head": bool(has_legacy_geo_head),
        "action_codebook_source": action_codebook_source,
        "action_codebook_entries": codebook_entry_count,
        "action_codebook_bucket_count": codebook_bucket_count,
        "dataset_path": dataset_path,
        "codebook_path": codebook_path,
    }


def load_inverse_bundle(
    cfg: dict,
    ckpt_path: str,
    device,
    *,
    allow_fresh_fallback: bool,
    require_geometry_code_ready: bool = False,
) -> Optional[dict]:
    ckpt = None
    action_codebook = None
    action_codebook_source = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        action_codebook = ckpt.get("action_codebook")
        if action_codebook is not None:
            action_codebook_source = "checkpoint"
    paths_cfg = cfg.get("paths", {})
    dataset_path = paths_cfg.get("il_multistep_dataset_output") or paths_cfg.get("il_dataset_output")
    if action_codebook is None:
        if dataset_path and os.path.exists(str(Path(dataset_path).with_suffix(".codebook.pt"))):
            action_codebook = load_action_codebook(dataset_path)
            action_codebook_source = "external"
    if action_codebook is not None:
        cfg.setdefault("gnn_policy", {})
        cfg["gnn_policy"]["num_geometry_codes"] = max(1, len(action_codebook.get("entries", [])))
        cfg["gnn_policy"]["action_code_dim"] = int(action_codebook.get("code_dim", 6))

    curve_encoder = CurveEncoder(
        input_dim=cfg["curve_encoder"]["input_dim"],
        hidden_dims=cfg["curve_encoder"]["hidden_dims"],
        latent_dim=cfg["curve_encoder"]["latent_dim"],
    ).to(device)
    policy = GNNPolicy(cfg).to(device)
    if action_codebook is not None:
        policy.set_action_codebook(
            codebook_tensor(action_codebook).to(device),
            buckets=action_codebook.get("bucket_to_ids", {}),
        )

    checkpoint_loaded = False
    warning = None
    if os.path.exists(ckpt_path):
        try:
            if ckpt is None:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            policy_state = policy.load_state_dict(ckpt["policy"], strict=False)
            missing_keys, unexpected_keys = policy_load_incompatibilities(policy_state)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint '{ckpt_path}' incompatible with policy. "
                    f"Missing={missing_keys}, unexpected={unexpected_keys}"
                )
            encoder_state = curve_encoder.load_state_dict(ckpt["curve_encoder"], strict=False)
            if encoder_state.missing_keys or encoder_state.unexpected_keys:
                warning = (
                    f"curve encoder checkpoint has key drift. "
                    f"Missing={encoder_state.missing_keys}, unexpected={encoder_state.unexpected_keys}"
                )
            checkpoint_loaded = True
        except Exception as exc:
            if not allow_fresh_fallback:
                raise
            warning = f"failed to load checkpoint '{ckpt_path}', using fresh weights: {type(exc).__name__}: {exc}"
    else:
        if not allow_fresh_fallback:
            return None
        warning = f"checkpoint missing: {ckpt_path}; using fresh weights"

    geometry_code_status = inspect_inverse_checkpoint_geometry_code_state(
        ckpt,
        checkpoint_loaded=checkpoint_loaded,
        action_codebook=action_codebook,
        action_codebook_source=action_codebook_source,
        dataset_path=str(dataset_path) if dataset_path else None,
    )
    if warning and geometry_code_status["warnings"]:
        warning = f"{warning}; {'; '.join(geometry_code_status['warnings'])}"
    elif geometry_code_status["warnings"]:
        warning = "; ".join(geometry_code_status["warnings"])

    policy.geometry_code_ready = bool(geometry_code_status["ready"])
    policy.geometry_code_issue = geometry_code_status["issue"]
    policy.geometry_code_status = dict(geometry_code_status)
    if require_geometry_code_ready and not policy.geometry_code_ready:
        raise RuntimeError(
            f"Inverse bundle '{ckpt_path}' is not rollout-ready: {policy.geometry_code_issue}"
        )

    policy.eval()
    curve_encoder.eval()
    return {
        "policy": policy,
        "curve_encoder": curve_encoder,
        "agent": PPOAgent(policy, curve_encoder, cfg, device),
        "checkpoint_path": ckpt_path,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_warning": warning,
        "action_codebook": action_codebook,
        "action_codebook_source": action_codebook_source,
        "geometry_code_ready": bool(geometry_code_status["ready"]),
        "geometry_code_issue": geometry_code_status["issue"],
        "geometry_code_status": geometry_code_status,
    }


def encode_target(curve_encoder, target: dict, device):
    with torch.no_grad():
        return curve_encoder(
            target["y_foot"].unsqueeze(0).to(device),
            target["y_knee"].unsqueeze(0).to(device),
            target["y_ankle"].unsqueeze(0).to(device),
        )


def rollout_trace_policy(bundle, trace_record, cfg, device):
    z_c = encode_target(bundle["curve_encoder"], trace_record["target"], device)
    current_graph = trace_record["base_data"]
    expected_j_steps = int(trace_record["expected_j_steps"])

    for step_idx in range(expected_j_steps + 1):
        context = {
            "family_index": trace_record["family_index"],
            "step_index": step_idx,
            "expected_j_steps": expected_j_steps,
            "can_stop": step_idx > 0,
            "stop_threshold": cfg.get("reward", {}).get("stop_threshold", 0.5),
        }
        actions, _, _ = bundle["agent"].batch_select_actions(
            [current_graph],
            z_c,
            deterministic=True,
            contexts=[context],
        )
        action = actions[0]
        if action is None or bool(action.get("stop", False)):
            break

        current_graph = apply_j_operator(
            current_graph,
            action["u"],
            action["v"],
            action["w"],
            action["n1"],
            action["n2"],
        )
    return current_graph


def rollout_trace_with_mcts(bundle, trace_record, surrogate, cfg, device):
    z_c = encode_target(bundle["curve_encoder"], trace_record["target"], device)
    reranker = MCTS(bundle["agent"], surrogate, cfg, device)
    result = reranker.rerank_rollouts(
        trace_record["base_data"],
        z_c,
        trace_record["target"],
        family_index=int(trace_record["family_index"]),
        expected_j_steps=int(trace_record["expected_j_steps"]),
    )
    best = result["best"]
    if best is None:
        return None, {"best": None, "candidate_count": 0}
    return best["graph"], {
        "best": {
            "surrogate_reward": float(best["surrogate_reward"]),
            "policy_log_prob": float(best["policy_log_prob"]),
            "step_count": int(best["step_count"]),
        },
        "candidate_count": len(result["candidates"]),
    }
