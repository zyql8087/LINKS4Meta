from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths, resolve_path
from src.inverse.action_codebook import codebook_tensor, load_action_codebook
from src.inverse.gnn_policy import GNNPolicy, policy_load_incompatibilities
from src.inverse.inference_runtime import demo_root_from_workspace, encode_target
from src.inverse.curve_encoder import CurveEncoder
from src.inverse.phase4_il import ensure_multistep_expert_paths, load_step_split, subset_by_indices
from src.inverse.phase5_rl import (
    build_family_curriculum,
    build_trace_dataset,
    filter_trace_dataset,
    sample_trace_batch,
    summarize_family_trace_counts,
)
from src.inverse.rl_agent import PPOAgent
from src.inverse.rl_env import MechanismEnv, load_frozen_surrogate


def _load_inverse_checkpoint(policy, curve_encoder, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("action_codebook") is not None:
        policy.set_action_codebook(
            codebook_tensor(ckpt["action_codebook"]).to(device),
            buckets=ckpt["action_codebook"].get("bucket_to_ids", {}),
        )
    policy_state = policy.load_state_dict(ckpt["policy"], strict=False)
    missing_keys, unexpected_keys = policy_load_incompatibilities(policy_state)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Policy checkpoint is incompatible with current config. Missing={missing_keys}, unexpected={unexpected_keys}"
        )
    curve_encoder.load_state_dict(ckpt["curve_encoder"], strict=False)

def _coalesce_transition_rewards(env, rewards):
    merged = []
    for (reward, _), event in zip(rewards, env._reward_events):
        if event["stop"]:
            if merged:
                merged[-1] += float(reward)
            else:
                merged.append(float(reward))
        else:
            merged.append(float(reward))
    return merged


def _save_reward_plot(stage_reports, reward_plot_path: str):
    if not stage_reports:
        return
    plt.figure(figsize=(12, 6))
    for report in stage_reports:
        rewards = report["episode_rewards"]
        plt.plot(rewards, label=report["family"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Phase5 RL Reward By Family Stage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ensure_parent_dir(reward_plot_path)
    plt.savefig(reward_plot_path)
    plt.close()


def _checkpoint_payload(policy, curve_encoder, agent):
    return {
        "policy": policy.state_dict(),
        "curve_encoder": curve_encoder.state_dict(),
        "critic": agent.critic.state_dict(),
        "action_codebook": {
            "entries": [
                {"id": int(idx), "vector": code.detach().cpu().tolist()}
                for idx, code in enumerate(policy.action_codebook)
            ],
            "bucket_to_ids": dict(policy.action_codebook_buckets),
            "code_dim": int(policy.action_codebook.size(-1)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phase5 RL refinement")
    parser.add_argument("--config", type=str, default="src/config_inverse.yaml")
    parser.add_argument("--il_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--family_limit", type=str, default=None, help="Optional comma separated family subset.")
    args = parser.parse_args()

    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    paths_cfg = cfg["paths"]
    resolve_mapping_paths(
        paths_cfg,
        (
            "pkl_dataset",
            "precomputed_split_input",
            "forward_model",
            "config_forward",
            "il_dataset_output",
            "il_multistep_dataset_output",
            "il_model_output",
            "il_plot_output",
            "il_report_output",
            "il_split_output",
            "rl_model_output",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    ensure_parent_dir(paths_cfg["rl_model_output"])

    rl_cfg = cfg["rl_training"]
    reward_cfg = cfg["reward"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Phase5 RL] device={device}")

    multistep_dataset_path = paths_cfg.get("il_multistep_dataset_output", paths_cfg["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=paths_cfg["pkl_dataset"],
        output_path=multistep_dataset_path,
        use_cached=True,
    )
    action_codebook = load_action_codebook(multistep_dataset_path)
    cfg.setdefault("gnn_policy", {})
    cfg["gnn_policy"]["num_geometry_codes"] = max(1, len(action_codebook.get("entries", [])))
    cfg["gnn_policy"]["action_code_dim"] = int(action_codebook.get("code_dim", 6))

    curve_encoder = CurveEncoder(
        input_dim=cfg["curve_encoder"]["input_dim"],
        hidden_dims=cfg["curve_encoder"]["hidden_dims"],
        latent_dim=cfg["curve_encoder"]["latent_dim"],
    ).to(device)
    policy = GNNPolicy(cfg).to(device)
    policy.set_action_codebook(
        codebook_tensor(action_codebook).to(device),
        buckets=action_codebook.get("bucket_to_ids", {}),
    )

    il_model_path = (
        paths_cfg["il_model_output"]
        if args.il_model is None
        else str(resolve_path(args.il_model, config_path.parent, WORKSPACE_ROOT))
    )
    if os.path.exists(il_model_path):
        _load_inverse_checkpoint(policy, curve_encoder, il_model_path, device)
        print(f"[Phase5 RL] loaded IL checkpoint: {il_model_path}")
    else:
        print(f"[Phase5 RL] warning: IL checkpoint missing, using current weights: {il_model_path}")

    surrogate, _ = load_frozen_surrogate(paths_cfg["forward_model"], paths_cfg["config_forward"], device)
    agent = PPOAgent(policy, curve_encoder, cfg, device)

    split = load_step_split(
        step_paths,
        split_path=paths_cfg["il_split_output"],
        precomputed_split_path=paths_cfg.get("precomputed_split_input"),
        val_ratio=cfg["il_training"].get("val_ratio", 0.1),
        test_ratio=cfg["il_training"].get("test_ratio", 0.1),
        split_seed=cfg["il_training"].get("split_seed", 42),
    )
    train_step_paths = subset_by_indices(step_paths, split["train_indices"])
    train_traces = build_trace_dataset(train_step_paths)
    print(f"[Phase5 RL] train trace counts: {summarize_family_trace_counts(train_traces)}")

    curriculum = build_family_curriculum(rl_cfg)
    if args.family_limit:
        allowed = {item.strip() for item in args.family_limit.split(",") if item.strip()}
        curriculum = [stage for stage in curriculum if stage["family"] in allowed]

    rng = random.Random(int(rl_cfg.get("split_seed", 42)))
    stage_reports = []
    best_reward = float("-inf")

    for stage in curriculum:
        family_name = stage["family"]
        family_traces = filter_trace_dataset(train_traces, [family_name])
        if not family_traces:
            print(f"[Phase5 RL] skip family={family_name}: no training traces")
            continue

        print(
            f"\n[Phase5 RL] stage family={family_name} episodes={stage['episodes']} "
            f"rollout_batch={stage['rollout_batch_size']} traces={len(family_traces)}"
        )
        family_rewards = []
        for episode_start in range(0, int(stage["episodes"]), int(stage["rollout_batch_size"])):
            batch_size = min(int(stage["rollout_batch_size"]), int(stage["episodes"]) - episode_start)
            trace_batch = sample_trace_batch(family_traces, batch_size, rng)
            agent.buffer.clear()
            batch_rewards = []

            for trace in trace_batch:
                target = trace["target"]
                z_c = encode_target(curve_encoder, target, device)
                env = MechanismEnv(
                    surrogate,
                    reward_cfg,
                    max_steps=int(trace["expected_j_steps"]) + 1,
                    device=device,
                    constraint_cfg=cfg.get("constraints", {}),
                )
                obs = env.reset(
                    target,
                    copy.deepcopy(trace["base_data"]),
                    z_c,
                    family_id=trace["family_id"],
                    family_index=trace["family_index"],
                    expected_j_steps=trace["expected_j_steps"],
                )

                transition_start = len(agent.buffer.states)
                while not env.done:
                    context = {
                        "family_index": trace["family_index"],
                        "step_index": obs["step"],
                        "expected_j_steps": trace["expected_j_steps"],
                        "can_stop": obs["can_stop"],
                        "stop_threshold": reward_cfg.get("stop_threshold", 0.5),
                    }
                    actions, log_probs, values = agent.batch_select_actions(
                        [obs["graph"]],
                        z_c,
                        deterministic=False,
                        contexts=[context],
                    )
                    action = actions[0]
                    if action is None:
                        action = {"stop": True}
                    next_obs, _, done, _ = env.step(action)
                    if not bool(action.get("stop", False)):
                        agent.buffer.store(
                            copy.deepcopy(obs["graph"]),
                            z_c.squeeze(0).detach().cpu(),
                            action,
                            0.0,
                            log_probs[0],
                            values[0],
                            done,
                        )
                    obs = next_obs

                rewards, reward_payloads = env.compute_episode_rewards()
                merged_rewards = _coalesce_transition_rewards(env, rewards)
                for local_idx, reward_value in enumerate(merged_rewards):
                    if transition_start + local_idx < len(agent.buffer.rewards):
                        agent.buffer.rewards[transition_start + local_idx] = float(reward_value)
                episode_reward = float(sum(merged_rewards)) if merged_rewards else 0.0
                if not merged_rewards and rewards:
                    episode_reward = float(sum(reward for reward, _ in rewards))
                batch_rewards.append(episode_reward)

            if agent.buffer.states:
                agent.update(agent.buffer, n_epochs=int(stage["ppo_epochs"]))

            family_rewards.extend(batch_rewards)
            mean_reward = float(np.mean(batch_rewards)) if batch_rewards else 0.0
            print(
                f"[Phase5 RL] {family_name} episodes "
                f"{episode_start + 1}-{episode_start + batch_size}: mean_reward={mean_reward:.4f}"
            )
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(_checkpoint_payload(policy, curve_encoder, agent), paths_cfg["rl_model_output"])
                print(f"[Phase5 RL] saved best checkpoint mean_reward={best_reward:.4f}")

        stage_reports.append(
            {
                "family": family_name,
                "episodes": int(stage["episodes"]),
                "trace_count": len(family_traces),
                "episode_rewards": [float(value) for value in family_rewards],
                "mean_reward": float(np.mean(family_rewards)) if family_rewards else 0.0,
                "max_reward": float(np.max(family_rewards)) if family_rewards else 0.0,
                "min_reward": float(np.min(family_rewards)) if family_rewards else 0.0,
            }
        )

    reward_plot_path = str((demo_root_from_workspace(WORKSPACE_ROOT) / "outputs" / "rl" / "phase5_reward_curve.png").resolve())
    _save_reward_plot(stage_reports, reward_plot_path)

    report = {
        "phase": "phase5_rl_refinement",
        "config": str(config_path),
        "multistep_dataset_path": multistep_dataset_path,
        "split_source": split.get("split_source"),
        "curriculum": stage_reports,
        "best_checkpoint": paths_cfg["rl_model_output"],
        "reward_plot": reward_plot_path,
    }
    report_path = str((demo_root_from_workspace(WORKSPACE_ROOT) / "outputs" / "rl" / "phase5_rl_report.json").resolve())
    ensure_parent_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"[Phase5 RL] reward plot: {reward_plot_path}")
    print(f"[Phase5 RL] report: {report_path}")
    print(f"[Phase5 RL] best checkpoint: {paths_cfg['rl_model_output']}")


if __name__ == "__main__":
    main()
