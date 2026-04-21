from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


FAMILY_INDEX_TO_NAME = {
    0: "6bar",
    1: "7bar",
    2: "8bar",
    3: "9bar",
}


def family_name_from_index(family_index: int | None) -> str:
    return FAMILY_INDEX_TO_NAME.get(int(family_index or -1), "unknown")


def step_role_for_index(step_index: int, expected_j_steps: int) -> str:
    if int(step_index) >= max(0, int(expected_j_steps) - 1):
        return "semantic"
    return "aux"


def codebook_bucket_for_step(family_name: str, step_role: str) -> str:
    family_name = str(family_name)
    step_role = str(step_role)
    if step_role == "aux":
        return "aux_shared"
    if family_name in {"6bar", "7bar"}:
        return "semantic_67"
    if family_name == "8bar":
        return "semantic_8bar"
    if family_name == "9bar":
        return "semantic_9bar"
    return f"{step_role}_{family_name}"


def default_action_codebook_path(dataset_path: str | Path) -> str:
    path = Path(dataset_path)
    return str(path.with_suffix(".codebook.pt"))


def default_action_codebook_json_path(dataset_path: str | Path) -> str:
    path = Path(dataset_path)
    return str(path.with_name("geom_codebook_v1.json"))


def _family_scope_for_bucket(bucket: str) -> tuple[list[str], str]:
    bucket = str(bucket)
    if bucket == "aux_shared":
        return list(FAMILY_INDEX_TO_NAME.values()), "aux"
    if bucket == "semantic_67":
        return ["6bar", "7bar"], "semantic"
    if bucket == "semantic_8bar":
        return ["8bar"], "semantic"
    if bucket == "semantic_9bar":
        return ["9bar"], "semantic"
    parts = bucket.split("_", maxsplit=1)
    if len(parts) == 2:
        return [parts[1]], parts[0]
    return ["unknown"], "unknown"


def _circle_intersections(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
    *,
    eps: float = 1.0e-8,
) -> list[np.ndarray]:
    delta = center_b - center_a
    dist = float(np.linalg.norm(delta))
    if dist <= eps:
        return []
    if dist > radius_a + radius_b + eps:
        return []
    if dist < abs(radius_a - radius_b) - eps:
        return []

    a = (radius_a**2 - radius_b**2 + dist**2) / (2.0 * dist)
    h_sq = max(radius_a**2 - a**2, 0.0)
    h = math.sqrt(h_sq)
    mid = center_a + a * delta / dist
    perp = np.array([-delta[1], delta[0]], dtype=np.float32) / dist
    if h <= eps:
        return [mid.astype(np.float32)]
    return [
        (mid + h * perp).astype(np.float32),
        (mid - h * perp).astype(np.float32),
    ]


def _branch_sign(origin_a: np.ndarray, origin_b: np.ndarray, point: np.ndarray, *, eps: float = 1.0e-8) -> float:
    cross = (origin_b[0] - origin_a[0]) * (point[1] - origin_a[1]) - (origin_b[1] - origin_a[1]) * (point[0] - origin_a[0])
    if abs(float(cross)) <= eps:
        return 1.0
    return 1.0 if float(cross) > 0.0 else -1.0


def _pick_oriented_point(
    origin_a: np.ndarray,
    origin_b: np.ndarray,
    candidates: Sequence[np.ndarray],
    sigma: float,
) -> np.ndarray:
    if not candidates:
        raise ValueError("no circle intersection candidates")
    sigma = 1.0 if float(sigma) >= 0.0 else -1.0
    scored = sorted(
        (
            (abs(_branch_sign(origin_a, origin_b, candidate) - sigma), candidate)
            for candidate in candidates
        ),
        key=lambda item: item[0],
    )
    return np.asarray(scored[0][1], dtype=np.float32)


def encode_local_dyad_code(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_w: np.ndarray,
    pos_n1: np.ndarray,
    pos_n2: np.ndarray,
) -> np.ndarray:
    pos_i = np.asarray(pos_i, dtype=np.float32)
    pos_j = np.asarray(pos_j, dtype=np.float32)
    pos_w = np.asarray(pos_w, dtype=np.float32)
    pos_n1 = np.asarray(pos_n1, dtype=np.float32)
    pos_n2 = np.asarray(pos_n2, dtype=np.float32)

    scale = float(np.linalg.norm(pos_j - pos_i))
    if scale <= 1.0e-8:
        raise ValueError("anchor pair has zero span")

    rho_i = float(np.linalg.norm(pos_n1 - pos_i) / scale)
    rho_j = float(np.linalg.norm(pos_n1 - pos_j) / scale)
    tau = float(np.linalg.norm(pos_n2 - pos_n1) / scale)
    gamma = float(np.linalg.norm(pos_n2 - pos_w) / scale)
    sigma_1 = _branch_sign(pos_i, pos_j, pos_n1)
    sigma_2 = _branch_sign(pos_n1, pos_w, pos_n2)
    return np.asarray([rho_i, rho_j, tau, gamma, sigma_1, sigma_2], dtype=np.float32)


def decode_local_dyad_code(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_w: np.ndarray,
    code: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    code_arr = np.asarray(code, dtype=np.float32).reshape(-1)
    if code_arr.size != 6:
        raise ValueError(f"expected 6-dim code, got shape {tuple(code_arr.shape)}")
    rho_i, rho_j, tau, gamma, sigma_1, sigma_2 = code_arr.tolist()
    pos_i = np.asarray(pos_i, dtype=np.float32)
    pos_j = np.asarray(pos_j, dtype=np.float32)
    pos_w = np.asarray(pos_w, dtype=np.float32)

    scale = float(np.linalg.norm(pos_j - pos_i))
    if scale <= 1.0e-8:
        raise ValueError("anchor pair has zero span")

    n1_candidates = _circle_intersections(pos_i, rho_i * scale, pos_j, rho_j * scale)
    n1 = _pick_oriented_point(pos_i, pos_j, n1_candidates, sigma_1)
    n2_candidates = _circle_intersections(n1, tau * scale, pos_w, gamma * scale)
    n2 = _pick_oriented_point(n1, pos_w, n2_candidates, sigma_2)
    return n1.astype(np.float32), n2.astype(np.float32)


def build_action_codebook(
    step_paths: Sequence[dict[str, object]],
    *,
    cluster_radius: float = 0.075,
    max_codes_per_bucket: int = 24,
) -> dict[str, object]:
    grouped: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for item_idx, item in enumerate(step_paths):
        grouped[str(item["action_code_bucket"])].append((item_idx, np.asarray(item["action_code_vec"], dtype=np.float32)))

    entries: list[dict[str, object]] = []
    item_assignments: dict[int, int] = {}
    bucket_to_ids: dict[str, list[int]] = {}

    for bucket, samples in sorted(grouped.items()):
        clusters: list[dict[str, object]] = []
        for item_idx, vector in samples:
            if not clusters:
                clusters.append({"center": vector.copy(), "items": [item_idx], "vectors": [vector.copy()]})
                continue
            distances = [float(np.linalg.norm(vector - cluster["center"])) for cluster in clusters]
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= float(cluster_radius):
                cluster = clusters[best_idx]
                cluster["items"].append(item_idx)
                cluster["vectors"].append(vector.copy())
                cluster["center"] = np.mean(np.stack(cluster["vectors"], axis=0), axis=0).astype(np.float32)
            else:
                clusters.append({"center": vector.copy(), "items": [item_idx], "vectors": [vector.copy()]})

        clusters.sort(key=lambda cluster: len(cluster["items"]), reverse=True)
        clusters = clusters[: max(1, int(max_codes_per_bucket))]
        bucket_ids: list[int] = []
        bucket_centers = [np.asarray(cluster["center"], dtype=np.float32) for cluster in clusters]

        for cluster in clusters:
            global_id = len(entries)
            entries.append(
                {
                    "id": global_id,
                    "bucket": bucket,
                    "vector": np.asarray(cluster["center"], dtype=np.float32).tolist(),
                    "count": int(len(cluster["items"])),
                }
            )
            bucket_ids.append(global_id)
        bucket_to_ids[bucket] = bucket_ids

        for item_idx, vector in samples:
            distances = [float(np.linalg.norm(vector - center)) for center in bucket_centers]
            bucket_local_id = int(np.argmin(distances))
            item_assignments[item_idx] = bucket_ids[bucket_local_id]

    return {
        "strategy": "strategy_a_local_ratio_dyad_v1",
        "code_dim": 6,
        "cluster_radius": float(cluster_radius),
        "max_codes_per_bucket": int(max_codes_per_bucket),
        "entries": entries,
        "bucket_to_ids": bucket_to_ids,
        "item_assignments": item_assignments,
    }


def attach_action_codebook(
    step_paths: Sequence[dict[str, object]],
    codebook: dict[str, object],
) -> list[dict[str, object]]:
    entries = codebook.get("entries", [])
    entry_vectors = {
        int(entry["id"]): torch.tensor(entry["vector"], dtype=torch.float32)
        for entry in entries
    }
    assignments = {int(key): int(value) for key, value in dict(codebook.get("item_assignments", {})).items()}

    updated = []
    for item_idx, item in enumerate(step_paths):
        new_item = dict(item)
        code_id = assignments[item_idx]
        new_item["action_code_id"] = code_id
        new_item["action_code_target"] = entry_vectors[code_id].clone()
        updated.append(new_item)
    return updated


def load_action_codebook(dataset_path: str | Path) -> dict[str, object]:
    return torch.load(default_action_codebook_path(dataset_path), map_location="cpu", weights_only=False)


def save_action_codebook(dataset_path: str | Path, codebook: dict[str, object]) -> str:
    output_path = default_action_codebook_path(dataset_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(codebook, output_path)
    export_action_codebook_v1_json(dataset_path, codebook)
    return output_path


def export_action_codebook_v1_json(
    dataset_path: str | Path,
    codebook: dict[str, object],
    *,
    step_paths: Sequence[dict[str, object]] | None = None,
    output_path: str | Path | None = None,
) -> str:
    bucket_counts: dict[str, int] = defaultdict(int)
    bucket_family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    entry_family_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    assignments = {int(key): int(value) for key, value in dict(codebook.get("item_assignments", {})).items()}

    if step_paths is not None:
        for item_idx, item in enumerate(step_paths):
            bucket = str(item.get("action_code_bucket", "unknown"))
            family_name = str(item.get("family_id") or item.get("family") or "unknown")
            bucket_counts[bucket] += 1
            bucket_family_counts[bucket][family_name] += 1
            if item_idx in assignments:
                entry_family_counts[int(assignments[item_idx])][family_name] += 1

    buckets_payload = []
    for bucket, ids in sorted(dict(codebook.get("bucket_to_ids", {})).items()):
        families, step_role = _family_scope_for_bucket(bucket)
        observed = bucket_family_counts.get(bucket)
        if observed:
            families = sorted(observed.keys())
        buckets_payload.append(
            {
                "bucket": bucket,
                "step_role": step_role,
                "family_scope": families,
                "num_codes": int(len(ids)),
                "sample_count": int(bucket_counts.get(bucket, 0)),
                "code_ids": [int(idx) for idx in ids],
            }
        )

    codes_payload = []
    for entry in sorted(codebook.get("entries", []), key=lambda item: int(item["id"])):
        code_id = int(entry["id"])
        families, step_role = _family_scope_for_bucket(str(entry["bucket"]))
        observed = entry_family_counts.get(code_id)
        if observed:
            families = sorted(observed.keys())
        codes_payload.append(
            {
                "id": code_id,
                "bucket": str(entry["bucket"]),
                "step_role": step_role,
                "family_scope": families,
                "vector": [float(value) for value in entry["vector"]],
                "count": int(entry.get("count", 0)),
            }
        )

    payload = {
        "version": "geom_codebook_v1",
        "strategy": str(codebook.get("strategy", "strategy_a_local_ratio_dyad_v1")),
        "parameterization": "rho_rho_tau_gamma_sigma_sigma",
        "code_dim": int(codebook.get("code_dim", 6)),
        "cluster_radius": float(codebook.get("cluster_radius", 0.0)),
        "max_codes_per_bucket": int(codebook.get("max_codes_per_bucket", 0)),
        "source_dataset": str(Path(dataset_path)),
        "buckets": buckets_payload,
        "codes": codes_payload,
    }

    final_path = Path(output_path) if output_path is not None else Path(default_action_codebook_json_path(dataset_path))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(final_path)


def allowed_code_ids_for_context(
    codebook: dict[str, object],
    *,
    family_name: str,
    step_role: str,
) -> list[int]:
    bucket = codebook_bucket_for_step(family_name, step_role)
    return [int(idx) for idx in codebook.get("bucket_to_ids", {}).get(bucket, [])]


def codebook_tensor(codebook: dict[str, object]) -> torch.Tensor:
    entries = sorted(codebook.get("entries", []), key=lambda entry: int(entry["id"]))
    if not entries:
        return torch.zeros((0, 6), dtype=torch.float32)
    return torch.tensor([entry["vector"] for entry in entries], dtype=torch.float32)
