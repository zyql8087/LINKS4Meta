from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Iterable, Sequence

from src.inverse.phase4_il import family_name_to_index, group_paths_by_trace
from src.inverse.rl_env import apply_j_operator

FAMILY_STAGE_ORDER = ("6bar", "7bar", "8bar", "9bar")
FAMILY_EXPECTED_J_STEPS = {
    "6bar": 1,
    "7bar": 1,
    "8bar": 2,
    "9bar": 2,
}


def expected_j_steps_for_family(family_name: str, default: int = 1) -> int:
    return int(FAMILY_EXPECTED_J_STEPS.get(str(family_name), default))


def build_trace_dataset(step_paths: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    trace_dataset = []
    for trace_items in group_paths_by_trace(step_paths):
        first = trace_items[0]
        family_name = str(first["family_id"])
        expected_j_steps = int(first.get("step_count", expected_j_steps_for_family(family_name)))
        trace_dataset.append(
            {
                "trace_id": int(first["trace_id"]),
                "sample_id": int(first["sample_id"]),
                "family_id": family_name,
                "family_index": family_name_to_index(family_name),
                "expected_j_steps": expected_j_steps,
                "base_data": copy.deepcopy(first["base_data"]),
                "target": {
                    "y_foot": first["y_foot"],
                    "y_knee": first["y_knee"],
                    "y_ankle": first["y_ankle"],
                },
                "step_paths": trace_items,
            }
        )
    return trace_dataset


def reconstruct_expert_final_graph(trace_record: dict[str, object]):
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


def filter_trace_dataset(trace_dataset: Sequence[dict[str, object]], families: Iterable[str]) -> list[dict[str, object]]:
    family_set = {str(name) for name in families}
    return [record for record in trace_dataset if str(record["family_id"]) in family_set]


def build_family_curriculum(rl_cfg: dict) -> list[dict[str, object]]:
    configured = rl_cfg.get("family_curriculum")
    episodes_default = int(rl_cfg.get("episodes_per_family", rl_cfg.get("episodes", 500)))
    update_epochs_default = int(rl_cfg.get("ppo_epochs", 4))
    rollout_batch_default = int(rl_cfg.get("rollout_batch_size", 8))
    if configured:
        curriculum = []
        for stage in configured:
            curriculum.append(
                {
                    "family": str(stage["family"]),
                    "episodes": int(stage.get("episodes", episodes_default)),
                    "ppo_epochs": int(stage.get("ppo_epochs", update_epochs_default)),
                    "rollout_batch_size": int(stage.get("rollout_batch_size", rollout_batch_default)),
                }
            )
        return curriculum

    return [
        {
            "family": family_name,
            "episodes": episodes_default,
            "ppo_epochs": update_epochs_default,
            "rollout_batch_size": rollout_batch_default,
        }
        for family_name in FAMILY_STAGE_ORDER
    ]


def sample_trace_batch(trace_dataset: Sequence[dict[str, object]], batch_size: int, rng: random.Random) -> list[dict[str, object]]:
    if not trace_dataset:
        return []
    if len(trace_dataset) >= batch_size:
        return rng.sample(list(trace_dataset), batch_size)
    return [rng.choice(list(trace_dataset)) for _ in range(batch_size)]


def summarize_family_trace_counts(trace_dataset: Sequence[dict[str, object]]) -> dict[str, int]:
    counts = defaultdict(int)
    for record in trace_dataset:
        counts[str(record["family_id"])] += 1
    return dict(sorted(counts.items()))
