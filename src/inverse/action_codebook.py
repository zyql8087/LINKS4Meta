# 动作码本：管理 dyad 几何配置的编解码、聚类构建、存储加载和查询。
# 动作码为 6 维向量 [rho_i, rho_j, tau, gamma, sigma_1, sigma_2]，描述二元杆组局部几何。
# 采用基于半径的贪心聚类构建码本，支持按机构家族和步骤角色分桶和细粒度桶策略。

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# 机构家族索引到名称的映射
FAMILY_INDEX_TO_NAME = {
    0: "6bar",
    1: "7bar",
    2: "8bar",
    3: "9bar",
}

CODEBOOK_VERSION = "geom_codebook_v2_validity_context_bucket"

# 默认细粒度桶策略配置
DEFAULT_FINE_BUCKET_POLICY = {
    "enabled": True,
    "complex_families": ["8bar", "9bar"],
    "semantic_only": True,
    "include_step_index": True,
    "include_action_topo": True,
    "include_topology_signature_when_available": False,
}


def family_name_from_index(family_index: int | None) -> str:
    """将家族索引转为名称字符串，None 时返回 "unknown"。"""
    if family_index is None:
        return "unknown"
    return FAMILY_INDEX_TO_NAME.get(int(family_index), "unknown")


def step_role_for_index(step_index: int, expected_j_steps: int) -> str:
    """最后一步（或超出范围）为 "semantic"，其余为 "aux"。"""
    if int(step_index) >= max(0, int(expected_j_steps) - 1):
        return "semantic"
    return "aux"


def _action_topo_suffix(action_topo) -> str | None:
    """从动作拓扑提取前三个节点索引生成 "0_1_2" 格式后缀。"""
    if action_topo is None:
        return None
    if hasattr(action_topo, "detach"):
        values = action_topo.detach().cpu().view(-1).tolist()
    elif hasattr(action_topo, "tolist"):
        values = action_topo.tolist()
    else:
        values = list(action_topo)
    if len(values) < 3:
        return None
    return "_".join(str(int(value)) for value in values[:3])


def _normalise_fine_bucket_policy(policy: dict | None) -> dict[str, object]:
    """将用户桶策略与默认策略合并，确保所有必需字段存在。"""
    merged = dict(DEFAULT_FINE_BUCKET_POLICY)
    if policy:
        merged.update(dict(policy))
        if "complex_families" in policy:
            merged["complex_families"] = [str(item) for item in policy["complex_families"]]
    return merged


def _normalise_bucket_code_overrides(overrides: dict | None) -> dict[str, int]:
    """规范化桶码数量覆盖配置，过滤掉 <=0 的限制。"""
    normalised: dict[str, int] = {}
    for key, value in dict(overrides or {}).items():
        limit = int(value)
        if limit <= 0:
            continue
        normalised[str(key)] = limit
    return normalised


def _max_codes_for_bucket(bucket: str, default_limit: int, overrides: dict[str, int]) -> int:
    """获取桶的最大码数：精确匹配 > 通配符匹配（最长前缀优先）> 默认限制。"""
    bucket = str(bucket)
    if bucket in overrides:
        return max(1, int(overrides[bucket]))
    wildcard_matches = [
        (len(pattern), limit)
        for pattern, limit in overrides.items()
        if pattern.endswith("*") and bucket.startswith(pattern[:-1])
    ]
    if wildcard_matches:
        wildcard_matches.sort(reverse=True)
        return max(1, int(wildcard_matches[0][1]))
    return max(1, int(default_limit))


def _bucket_token(value: object) -> str:
    """将任意值转为桶名中安全的 token（仅保留字母数字和下划线）。"""
    text = str(value).strip()
    safe = []
    for char in text:
        if char.isalnum():
            safe.append(char)
        elif char in {"-", "_"}:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def codebook_bucket_for_step(
    family_name: str, step_role: str, *, step_index: int | None = None,
    action_topo=None, topology_signature=None, fine_topology_buckets: bool = False,
    fine_bucket_policy: dict | None = None,
) -> str:
    """为指定步骤计算码本桶名称。
    辅助步默认 "aux_shared"；语义步中 6bar/7bar 共享 "semantic_67"，8bar/9bar 各自独立。
    启用细粒度桶时附加 step_index 和 action_topo 信息。
    """
    family_name = str(family_name)
    step_role = str(step_role)

    if step_role == "aux":
        policy = _normalise_fine_bucket_policy(fine_bucket_policy)
        complex_families = set(str(item) for item in policy.get("complex_families", []))
        use_fine = (
            bool(fine_topology_buckets)
            and bool(policy.get("enabled", True))
            and family_name in complex_families
            and not bool(policy.get("semantic_only", True))
        )
        if not use_fine:
            return "aux_shared"

        parts = [f"aux_{family_name}"]
        if bool(policy.get("include_step_index", True)) and step_index is not None:
            parts.append(f"step{int(step_index)}")
        if (
            bool(policy.get("include_topology_signature_when_available", False))
            and topology_signature not in (None, "")
        ):
            parts.append(f"sig_{_bucket_token(topology_signature)}")
        suffix = _action_topo_suffix(action_topo)
        if bool(policy.get("include_action_topo", True)) and suffix is not None:
            parts.append(f"topo_{suffix}")
        if len(parts) == 1:
            return "aux_shared"
        return "_".join(parts)

    if family_name in {"6bar", "7bar"}:
        return "semantic_67"

    if family_name in {"8bar", "9bar"}:
        base = f"semantic_{family_name}"
        policy = _normalise_fine_bucket_policy(fine_bucket_policy)
        complex_families = set(str(item) for item in policy.get("complex_families", []))
        use_fine = (
            bool(fine_topology_buckets)
            and bool(policy.get("enabled", True))
            and family_name in complex_families
            and (step_role == "semantic" or not bool(policy.get("semantic_only", True)))
        )
        if not use_fine:
            return base

        parts = [base]
        if bool(policy.get("include_step_index", True)) and step_index is not None:
            parts.append(f"step{int(step_index)}")
        if (
            bool(policy.get("include_topology_signature_when_available", False))
            and topology_signature not in (None, "")
        ):
            parts.append(f"sig_{_bucket_token(topology_signature)}")
        suffix = _action_topo_suffix(action_topo)
        if bool(policy.get("include_action_topo", True)) and suffix is not None:
            parts.append(f"topo_{suffix}")
        if len(parts) == 1:
            return base
        return "_".join(parts)

    return f"{step_role}_{family_name}"


def codebook_bucket_candidates_for_step(
    family_name: str, step_role: str, *, step_index: int | None = None,
    action_topo=None, topology_signature=None, fine_topology_buckets: bool = True,
    fine_bucket_policy: dict | None = None,
) -> list[str]:
    """生成候选桶名列表（从具体到通用）：细粒度桶 > 仅拓扑桶 > 基础桶 > 通用桶。"""
    base = codebook_bucket_for_step(family_name, step_role)
    if not fine_topology_buckets:
        return [base]

    fine = codebook_bucket_for_step(
        family_name, step_role, step_index=step_index, action_topo=action_topo,
        topology_signature=topology_signature, fine_topology_buckets=True,
        fine_bucket_policy=fine_bucket_policy,
    )
    topo_only = codebook_bucket_for_step(
        family_name, step_role, action_topo=action_topo, fine_topology_buckets=True,
        fine_bucket_policy={**_normalise_fine_bucket_policy(fine_bucket_policy), "include_step_index": False},
    )
    legacy = "aux_shared" if str(step_role) == "aux" else base
    return list(dict.fromkeys([fine, topo_only, base, legacy]))


def resolve_codebook_bucket_for_step(
    bucket_map: dict[str, object], family_name: str, step_role: str,
    *, step_index: int | None = None, action_topo=None, topology_signature=None,
    fine_topology_buckets: bool = True, fine_bucket_policy: dict | None = None,
) -> str:
    """按候选桶优先级查找实际存在的桶，无匹配则回退到基础桶。"""
    for bucket in codebook_bucket_candidates_for_step(
        family_name, step_role, step_index=step_index, action_topo=action_topo,
        topology_signature=topology_signature, fine_topology_buckets=fine_topology_buckets,
        fine_bucket_policy=fine_bucket_policy,
    ):
        if bucket in bucket_map:
            return bucket
    return codebook_bucket_for_step(family_name, step_role)


def default_action_codebook_path(dataset_path: str | Path) -> str:
    """数据集路径扩展名替换为 .codebook.pt。"""
    return str(Path(dataset_path).with_suffix(".codebook.pt"))


def default_action_codebook_json_path(dataset_path: str | Path) -> str:
    """同目录下生成 geom_codebook_v1.json。"""
    return str(Path(dataset_path).with_name("geom_codebook_v1.json"))


def _family_scope_for_bucket(bucket: str) -> tuple[list[str], str]:
    """根据桶名推断覆盖的机构家族列表和步骤角色。"""
    bucket = str(bucket)
    if bucket == "aux_shared":
        return list(FAMILY_INDEX_TO_NAME.values()), "aux"
    if bucket == "semantic_67":
        return ["6bar", "7bar"], "semantic"
    if bucket == "semantic_8bar" or bucket.startswith("semantic_8bar_"):
        return ["8bar"], "semantic"
    if bucket == "semantic_9bar" or bucket.startswith("semantic_9bar_"):
        return ["9bar"], "semantic"
    parts = bucket.split("_", maxsplit=1)
    if len(parts) == 2:
        return [parts[1]], parts[0]
    return ["unknown"], "unknown"


def _circle_intersections(
    center_a: np.ndarray, radius_a: float, center_b: np.ndarray, radius_b: float, *, eps: float = 1.0e-8,
) -> list[np.ndarray]:
    """计算两个圆的交点，返回 0/1/2 个交点坐标列表。"""
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
    """用叉积判断点在有向线段 AB 左侧 (+1) 还是右侧 (-1)。"""
    cross = (origin_b[0] - origin_a[0]) * (point[1] - origin_a[1]) - (origin_b[1] - origin_a[1]) * (point[0] - origin_a[0])
    if abs(float(cross)) <= eps:
        return 1.0
    return 1.0 if float(cross) > 0.0 else -1.0


def _pick_oriented_point(
    origin_a: np.ndarray, origin_b: np.ndarray, candidates: Sequence[np.ndarray], sigma: float,
) -> np.ndarray:
    """从候选点中选择与目标分支方向 sigma (+1/-1) 最一致的点。"""
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
    pos_i: np.ndarray, pos_j: np.ndarray, pos_w: np.ndarray,
    pos_n1: np.ndarray, pos_n2: np.ndarray,
) -> np.ndarray:
    """将 dyad 的 5 个节点位置编码为 6 维动作码 [rho_i, rho_j, tau, gamma, sigma_1, sigma_2]。
    rho_i/rho_j 为 n1 到 i/j 的归一化距离，tau/gamma 为 n2 到 n1/w 的归一化距离，
    sigma_1/sigma_2 为分支方向。
    """
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
    pos_i: np.ndarray, pos_j: np.ndarray, pos_w: np.ndarray, code: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """解码 6 维动码为 n1、n2 坐标：先通过 i/j 两圆交点定位 n1，再通过 n1/w 两圆交点定位 n2。"""
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


def _cluster_medoid(members: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    """计算聚类质心（到所有成员距离之和最小的实际成员点）。"""
    return _cluster_medoid_member(members)[1].copy()


def _cluster_medoid_member(members: Sequence[tuple[int, np.ndarray]]) -> tuple[int, np.ndarray]:
    """计算聚类质心及其原始样本索引。"""
    if not members:
        raise ValueError("cannot choose medoid from empty cluster")
    vectors = [np.asarray(vector, dtype=np.float32) for _, vector in members]
    if len(vectors) == 1:
        return int(members[0][0]), vectors[0].copy()
    stacked = np.stack(vectors, axis=0)
    distances = np.linalg.norm(stacked[:, None, :] - stacked[None, :, :], axis=-1).sum(axis=1)
    medoid_idx = int(np.argmin(distances))
    return int(members[medoid_idx][0]), vectors[medoid_idx].copy()


def _evenly_spaced_members(
    members: Sequence[tuple[int, np.ndarray]], max_items: int,
) -> list[tuple[int, np.ndarray]]:
    """从成员列表中均匀采样指定数量的元素。"""
    if len(members) <= max_items:
        return list(members)
    if max_items <= 1:
        return [members[0]]
    indices = np.linspace(0, len(members) - 1, num=int(max_items), dtype=int).tolist()
    return [members[int(idx)] for idx in dict.fromkeys(indices)]


def _vector_valid_for_step(vector: np.ndarray, step: dict[str, object], constraint_cfg: dict | None) -> bool:
    """验证动作码在给定步骤上下文中是否有效：解码 -> J 算子 -> 图结构验证。"""
    try:
        from src.inverse.rl_env import apply_j_operator, validate_graph_structure

        graph = step["base_data"]
        topo = step["action_topo"]
        if hasattr(topo, "detach"):
            u, v, w = [int(value) for value in topo.detach().cpu().view(-1).tolist()[:3]]
        else:
            u, v, w = [int(value) for value in list(topo)[:3]]
        n1, n2 = decode_local_dyad_code(
            graph.pos[u].detach().cpu().numpy(),
            graph.pos[v].detach().cpu().numpy(),
            graph.pos[w].detach().cpu().numpy(),
            vector,
        )
        next_graph = apply_j_operator(graph, u, v, w, n1, n2)
        is_valid, _ = validate_graph_structure(next_graph, constraint_cfg or {})
        return bool(is_valid)
    except Exception:
        return False


def _choose_cluster_representative_info(
    members: Sequence[tuple[int, np.ndarray]], step_paths: Sequence[dict[str, object]],
    *, strategy: str, constraint_cfg: dict | None,
    max_validation_candidates: int, max_validation_contexts: int,
) -> dict[str, object]:
    """选择聚类代表点。策略 "mean" 用均值；"validity_best_medoid" 选验证通过最多且距中心最近的成员。"""
    if not members:
        raise ValueError("cannot choose representative from empty cluster")
    strategy = str(strategy or "validity_best_medoid")

    if strategy == "mean":
        vector = np.mean([np.asarray(vector, dtype=np.float32) for _, vector in members], axis=0).astype(np.float32)
        return {"vector": vector, "representative_item_idx": None, "validity_pass_count": 0, "validity_context_count": 0}

    if not all("base_data" in step_paths[item_idx] and "action_topo" in step_paths[item_idx] for item_idx, _ in members):
        item_idx, vector = _cluster_medoid_member(members)
        return {"vector": vector, "representative_item_idx": int(item_idx), "validity_pass_count": 0, "validity_context_count": 0}

    validation_members = _evenly_spaced_members(list(members), max(1, int(max_validation_contexts)))
    candidate_members = _evenly_spaced_members(list(members), max(1, int(max_validation_candidates)))
    member_vectors = [np.asarray(vector, dtype=np.float32) for _, vector in members]

    best_key = None
    best_vector = None
    best_item_idx = None
    best_valid_count = 0

    for candidate_idx, (candidate_item_idx, candidate_vector) in enumerate(candidate_members):
        candidate_vector = np.asarray(candidate_vector, dtype=np.float32)
        valid_count = sum(
            int(_vector_valid_for_step(candidate_vector, step_paths[item_idx], constraint_cfg))
            for item_idx, _ in validation_members
        )
        distance_sum = float(sum(np.linalg.norm(candidate_vector - vector) for vector in member_vectors))
        key = (valid_count, -distance_sum, -candidate_idx)
        if best_key is None or key > best_key:
            best_key = key
            best_vector = candidate_vector
            best_item_idx = int(candidate_item_idx)
            best_valid_count = int(valid_count)

    if best_vector is None:
        best_item_idx, best_vector = _cluster_medoid_member(members)
        best_valid_count = 0

    return {
        "vector": np.asarray(best_vector, dtype=np.float32).copy(),
        "representative_item_idx": int(best_item_idx) if best_item_idx is not None else None,
        "validity_pass_count": int(best_valid_count),
        "validity_context_count": int(len(validation_members)),
    }


def _choose_cluster_representative(
    members: Sequence[tuple[int, np.ndarray]], step_paths: Sequence[dict[str, object]],
    *, strategy: str, constraint_cfg: dict | None,
    max_validation_candidates: int, max_validation_contexts: int,
) -> np.ndarray:
    """选择聚类代表点向量（简化接口），返回 (6,) float32 向量。"""
    return np.asarray(
        _choose_cluster_representative_info(
            members, step_paths, strategy=strategy, constraint_cfg=constraint_cfg,
            max_validation_candidates=max_validation_candidates,
            max_validation_contexts=max_validation_contexts,
        )["vector"],
        dtype=np.float32,
    ).copy()


def build_action_codebook(
    step_paths: Sequence[dict[str, object]], *, cluster_radius: float = 0.075,
    max_codes_per_bucket: int = 24, max_codes_per_bucket_overrides: dict | None = None,
    representative_strategy: str = "validity_best_medoid", constraint_cfg: dict | None = None,
    max_validation_candidates: int = 32, max_validation_contexts: int = 96,
    fine_bucket_policy: dict | None = None,
) -> dict[str, object]:
    """构建动作码本：按桶分组 -> 半径贪心聚类 -> 保留最大聚类 -> 选择代表点 -> 分配样本到最近码字。
    返回包含 entries、bucket_to_ids、item_assignments 等字段的码本字典。
    """
    fine_bucket_policy = _normalise_fine_bucket_policy(fine_bucket_policy)
    bucket_limit_overrides = _normalise_bucket_code_overrides(max_codes_per_bucket_overrides)

    grouped: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for item_idx, item in enumerate(step_paths):
        grouped[str(item["action_code_bucket"])].append((item_idx, np.asarray(item["action_code_vec"], dtype=np.float32)))

    entries: list[dict[str, object]] = []
    item_assignments: dict[int, int] = {}
    bucket_to_ids: dict[str, list[int]] = {}

    for bucket, samples in sorted(grouped.items()):
        print(f"[*] Building action codebook bucket '{bucket}' from {len(samples)} samples ...")
        clusters: list[dict[str, object]] = []

        for item_idx, vector in samples:
            if not clusters:
                clusters.append({"center": vector.copy(), "count": 1, "members": [(item_idx, vector.copy())]})
                continue
            distances = [float(np.linalg.norm(vector - cluster["center"])) for cluster in clusters]
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= float(cluster_radius):
                cluster = clusters[best_idx]
                count = int(cluster["count"])
                cluster["center"] = ((cluster["center"] * count) + vector) / float(count + 1)
                cluster["center"] = np.asarray(cluster["center"], dtype=np.float32)
                cluster["count"] = count + 1
                cluster["members"].append((item_idx, vector.copy()))
            else:
                clusters.append({"center": vector.copy(), "count": 1, "members": [(item_idx, vector.copy())]})

        raw_cluster_count = len(clusters)
        clusters.sort(key=lambda cluster: int(cluster["count"]), reverse=True)
        bucket_limit = _max_codes_for_bucket(bucket, int(max_codes_per_bucket), bucket_limit_overrides)
        clusters = clusters[:bucket_limit]
        bucket_ids: list[int] = []

        representative_infos = [
            _choose_cluster_representative_info(
                cluster["members"], step_paths, strategy=representative_strategy,
                constraint_cfg=constraint_cfg, max_validation_candidates=max_validation_candidates,
                max_validation_contexts=max_validation_contexts,
            )
            for cluster in clusters
        ]
        bucket_centers = [np.asarray(info["vector"], dtype=np.float32) for info in representative_infos]
        print(
            f"[*] Bucket '{bucket}' clustering complete: raw_clusters={raw_cluster_count}, "
            f"kept={len(clusters)}, limit={bucket_limit}"
        )

        for cluster, info in zip(clusters, representative_infos):
            representative = np.asarray(info["vector"], dtype=np.float32)
            global_id = len(entries)
            entries.append({
                "id": global_id,
                "bucket": bucket,
                "vector": np.asarray(representative, dtype=np.float32).tolist(),
                "count": int(cluster["count"]),
                "representative_item_idx": (
                    None if info["representative_item_idx"] is None else int(info["representative_item_idx"])
                ),
                "validity_pass_count": int(info["validity_pass_count"]),
                "validity_context_count": int(info["validity_context_count"]),
            })
            bucket_ids.append(global_id)
        bucket_to_ids[bucket] = bucket_ids

        for item_idx, vector in samples:
            distances = [float(np.linalg.norm(vector - center)) for center in bucket_centers]
            bucket_local_id = int(np.argmin(distances))
            item_assignments[item_idx] = bucket_ids[bucket_local_id]

    return {
        "version": CODEBOOK_VERSION,
        "strategy": "strategy_a_local_ratio_dyad_v1",
        "code_dim": 6,
        "cluster_radius": float(cluster_radius),
        "max_codes_per_bucket": int(max_codes_per_bucket),
        "max_codes_per_bucket_overrides": dict(sorted(bucket_limit_overrides.items())),
        "representative_strategy": str(representative_strategy),
        "max_validation_candidates": int(max_validation_candidates),
        "max_validation_contexts": int(max_validation_contexts),
        "fine_bucket_policy": fine_bucket_policy,
        "entries": entries,
        "bucket_to_ids": bucket_to_ids,
        "item_assignments": item_assignments,
    }


def attach_action_codebook(
    step_paths: Sequence[dict[str, object]], codebook: dict[str, object],
) -> list[dict[str, object]]:
    """为每个步骤附加 action_code_id 和 action_code_target 字段，返回新列表。"""
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
    """从 .codebook.pt 文件加载码本。"""
    return torch.load(default_action_codebook_path(dataset_path), map_location="cpu", weights_only=False)


def save_action_codebook(
    dataset_path: str | Path, codebook: dict[str, object],
    *, step_paths: Sequence[dict[str, object]] | None = None,
) -> str:
    """保存码本到 .pt 文件并导出 .json 可视化文件，返回 .pt 路径。"""
    output_path = default_action_codebook_path(dataset_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(codebook, output_path)
    export_action_codebook_v1_json(dataset_path, codebook, step_paths=step_paths)
    return output_path


def export_action_codebook_v1_json(
    dataset_path: str | Path, codebook: dict[str, object],
    *, step_paths: Sequence[dict[str, object]] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """将码本导出为 JSON 格式，包含桶统计和码字详情，便于可视化分析。"""
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
            families = sorted(observed)
        buckets_payload.append({
            "bucket": bucket, "step_role": step_role, "family_scope": families,
            "num_codes": int(len(ids)), "sample_count": int(bucket_counts.get(bucket, 0)),
            "code_ids": [int(idx) for idx in ids],
        })

    codes_payload = []
    for entry in sorted(codebook.get("entries", []), key=lambda item: int(item["id"])):
        code_id = int(entry["id"])
        families, step_role = _family_scope_for_bucket(str(entry["bucket"]))
        observed = entry_family_counts.get(code_id)
        if observed:
            families = sorted(observed)
        codes_payload.append({
            "id": code_id, "bucket": str(entry["bucket"]), "step_role": step_role,
            "family_scope": families, "vector": [float(value) for value in entry["vector"]],
            "count": int(entry.get("count", 0)),
            "representative_item_idx": (
                None if entry.get("representative_item_idx") is None else int(entry.get("representative_item_idx"))
            ),
            "validity_pass_count": int(entry.get("validity_pass_count", 0)),
            "validity_context_count": int(entry.get("validity_context_count", 0)),
        })

    payload = {
        "version": str(codebook.get("version", "geom_codebook_v1")),
        "strategy": str(codebook.get("strategy", "strategy_a_local_ratio_dyad_v1")),
        "parameterization": "rho_rho_tau_gamma_sigma_sigma",
        "code_dim": int(codebook.get("code_dim", 6)),
        "cluster_radius": float(codebook.get("cluster_radius", 0.0)),
        "max_codes_per_bucket": int(codebook.get("max_codes_per_bucket", 0)),
        "max_codes_per_bucket_overrides": dict(codebook.get("max_codes_per_bucket_overrides", {})),
        "representative_strategy": str(codebook.get("representative_strategy", "unknown")),
        "max_validation_candidates": int(codebook.get("max_validation_candidates", 0)),
        "max_validation_contexts": int(codebook.get("max_validation_contexts", 0)),
        "fine_bucket_policy": dict(codebook.get("fine_bucket_policy", {})),
        "source_dataset": str(Path(dataset_path)),
        "buckets": buckets_payload,
        "codes": codes_payload,
    }

    final_path = Path(output_path) if output_path is not None else Path(default_action_codebook_json_path(dataset_path))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(final_path)


def allowed_code_ids_for_context(
    codebook: dict[str, object], *, family_name: str, step_role: str,
    step_index: int | None = None, action_topo=None, topology_signature=None,
) -> list[int]:
    """查询指定上下文允许使用的码字 ID 列表。"""
    bucket = resolve_codebook_bucket_for_step(
        codebook.get("bucket_to_ids", {}), family_name, step_role,
        step_index=step_index, action_topo=action_topo,
        topology_signature=topology_signature,
        fine_bucket_policy=codebook.get("fine_bucket_policy", {}),
    )
    return [int(idx) for idx in codebook.get("bucket_to_ids", {}).get(bucket, [])]


def codebook_tensor(codebook: dict[str, object]) -> torch.Tensor:
    """将码本转为 PyTorch 张量 (num_codes, code_dim)，按 ID 排序。空码本返回 (0,6) 零张量。"""
    entries = sorted(codebook.get("entries", []), key=lambda entry: int(entry["id"]))
    if not entries:
        return torch.zeros((0, 6), dtype=torch.float32)
    return torch.tensor([entry["vector"] for entry in entries], dtype=torch.float32)
