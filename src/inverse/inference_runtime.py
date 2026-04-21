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


def load_inverse_bundle(
    cfg: dict,
    ckpt_path: str,
    device,
    *,
    allow_fresh_fallback: bool,
) -> Optional[dict]:
    ckpt = None
    action_codebook = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        action_codebook = ckpt.get("action_codebook")
    if action_codebook is None:
        paths_cfg = cfg.get("paths", {})
        dataset_path = paths_cfg.get("il_multistep_dataset_output") or paths_cfg.get("il_dataset_output")
        if dataset_path and os.path.exists(str(Path(dataset_path).with_suffix(".codebook.pt"))):
            action_codebook = load_action_codebook(dataset_path)
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
