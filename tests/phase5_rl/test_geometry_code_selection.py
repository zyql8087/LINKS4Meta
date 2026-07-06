"""几何码选择改进的单元测试：sigma 翻转候选扩展 + 增量 J-Operator 验证。

覆盖：
  1. sigma_flip_code_variants 的变体生成与去重；
  2. decode_local_dyad_code_candidates 的分支解码、原始分支在前、几何去重；
  3. validate_j_operator_candidate 与全量 validate_graph_structure 在合法前缀上的等价性。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
GMM_ROOT = WORKSPACE_ROOT / "GraphMetaMat-LINKS"
for root in (GMM_ROOT, GMM_ROOT / "code"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.inverse.action_codebook import (
    decode_local_dyad_code,
    decode_local_dyad_code_candidates,
    encode_local_dyad_code,
    sigma_flip_code_variants,
)
from src.inverse.rl_env import (
    apply_j_operator,
    validate_graph_structure,
    validate_j_operator_candidate,
)


def _base_4bar_graph() -> Data:
    x0 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    types = np.array([1, 0, 0, 1], dtype=np.float32)
    grounded = np.array([1, 0, 0, 0], dtype=np.float32)
    x_feat = np.column_stack([x0, types, grounded])
    edges = np.array(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 0], [0, 3]],
        dtype=np.int64,
    )
    return Data(
        x=torch.tensor(x_feat, dtype=torch.float32),
        pos=torch.tensor(x0, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).T,
    )


class TestSigmaFlipVariants(unittest.TestCase):
    def test_returns_four_distinct_sign_combos_with_original_first(self):
        code = np.array([0.4, 0.6, 0.5, 0.7, 1.0, -1.0], dtype=np.float32)
        variants = sigma_flip_code_variants(code)
        self.assertEqual(len(variants), 4)
        # 原始码按位保留在最前。
        np.testing.assert_array_equal(variants[0], code)
        # 4 个符号组合互不相同。
        sign_pairs = {(float(np.sign(v[4]) or 1.0), float(np.sign(v[5]) or 1.0)) for v in variants}
        self.assertEqual(sign_pairs, {(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)})
        # 非符号分量不被改动。
        for v in variants:
            np.testing.assert_allclose(v[:4], code[:4])

    def test_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            sigma_flip_code_variants(np.zeros(5, dtype=np.float32))


class TestDecodeCandidates(unittest.TestCase):
    def test_original_branch_decodes_first_and_matches_plain_decode(self):
        pos_i = np.array([0.0, 0.0], dtype=np.float32)
        pos_j = np.array([2.0, 0.0], dtype=np.float32)
        pos_w = np.array([1.0, 2.0], dtype=np.float32)
        # 用一个真实 dyad 编码出码，保证可解。
        n1_true = np.array([1.0, 1.3], dtype=np.float32)
        n2_true = np.array([0.7, 2.4], dtype=np.float32)
        code = encode_local_dyad_code(pos_i, pos_j, pos_w, n1_true, n2_true)

        plain_n1, plain_n2 = decode_local_dyad_code(pos_i, pos_j, pos_w, code)
        candidates = decode_local_dyad_code_candidates(pos_i, pos_j, pos_w, code)
        self.assertGreaterEqual(len(candidates), 1)
        # 第一个候选 == 原始分支解码结果。
        np.testing.assert_allclose(candidates[0][0], plain_n1, atol=1e-5)
        np.testing.assert_allclose(candidates[0][1], plain_n2, atol=1e-5)
        # 原始分支应能还原出真实几何。
        np.testing.assert_allclose(candidates[0][0], n1_true, atol=1e-4)
        np.testing.assert_allclose(candidates[0][1], n2_true, atol=1e-4)

    def test_sigma_flip_yields_opposite_branch(self):
        pos_i = np.array([0.0, 0.0], dtype=np.float32)
        pos_j = np.array([2.0, 0.0], dtype=np.float32)
        # w 取在 i-j 连线附近，使 y=±sqrt(3) 两个 n1 分支都能再解出 n2。
        pos_w = np.array([1.0, 0.2], dtype=np.float32)
        # rho_i = rho_j = 1.0 → n1 两个交点关于 i-j 连线对称（y = ±sqrt(3)）。
        code = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        candidates = decode_local_dyad_code_candidates(pos_i, pos_j, pos_w, code)
        n1_ys = sorted({round(float(n1[1]), 3) for n1, _n2, _v in candidates})
        # 翻转 sigma_1 应得到 i-j 连线另一侧的 n1。
        self.assertTrue(any(y < 0 for y in n1_ys))
        self.assertTrue(any(y > 0 for y in n1_ys))

    def test_disable_flips_returns_single_branch(self):
        pos_i = np.array([0.0, 0.0], dtype=np.float32)
        pos_j = np.array([2.0, 0.0], dtype=np.float32)
        pos_w = np.array([1.0, 5.0], dtype=np.float32)
        code = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        candidates = decode_local_dyad_code_candidates(
            pos_i, pos_j, pos_w, code, include_sigma_flips=False,
        )
        self.assertEqual(len(candidates), 1)


class TestIncrementalValidatorEquivalence(unittest.TestCase):
    """增量验证与全量验证在合法前缀上必须给出一致的 is_valid 与 reason。"""

    def _random_valid_prefixes(self, rng, count=12):
        """从 4bar 基图出发随机施加 J-Operator，收集若干合法前缀图（不带 keypoints）。"""
        prefixes = [_base_4bar_graph()]
        attempts = 0
        while len(prefixes) < count and attempts < count * 60:
            attempts += 1
            base = prefixes[rng.integers(0, len(prefixes))]
            n_nodes = int(base.pos.size(0))
            u, v, w = rng.choice(n_nodes, size=3, replace=False).tolist()
            lo = base.pos.min(dim=0).values.numpy() - 1.0
            hi = base.pos.max(dim=0).values.numpy() + 1.0
            n1 = rng.uniform(lo, hi).astype(np.float32)
            n2 = rng.uniform(lo, hi).astype(np.float32)
            cand = apply_j_operator(base, int(u), int(v), int(w), n1, n2)
            is_valid, _ = validate_graph_structure(cand, {})
            if is_valid:
                prefixes.append(cand)
        return prefixes

    def test_matches_full_validator_over_random_steps(self):
        rng = np.random.default_rng(20260626)
        prefixes = self._random_valid_prefixes(rng)
        constraint_cfg = {}
        compared = 0
        valid_seen = 0
        invalid_seen = 0
        for prefix in prefixes:
            # 确保前缀本身合法（增量验证的前提）。
            base_valid, _ = validate_graph_structure(prefix, constraint_cfg)
            self.assertTrue(base_valid)
            n_nodes = int(prefix.pos.size(0))
            lo = prefix.pos.min(dim=0).values.numpy() - 1.5
            hi = prefix.pos.max(dim=0).values.numpy() + 1.5
            for _ in range(80):
                u, v, w = rng.choice(n_nodes, size=3, replace=False).tolist()
                n1 = rng.uniform(lo, hi).astype(np.float32)
                n2 = rng.uniform(lo, hi).astype(np.float32)

                full_graph = apply_j_operator(prefix, int(u), int(v), int(w), n1, n2)
                full_valid, full_info = validate_graph_structure(full_graph, constraint_cfg)
                inc_valid, inc_info = validate_j_operator_candidate(
                    prefix, int(u), int(v), int(w), n1, n2, constraint_cfg,
                )
                self.assertEqual(
                    bool(full_valid), bool(inc_valid),
                    msg=f"is_valid mismatch full={full_info} inc={inc_info}",
                )
                self.assertEqual(
                    full_info.get('reason'), inc_info.get('reason'),
                    msg=f"reason mismatch full={full_info} inc={inc_info}",
                )
                compared += 1
                valid_seen += int(bool(full_valid))
                invalid_seen += int(not bool(full_valid))
        # 健全性：测试集应同时覆盖合法与非法两类，否则等价性测试没有意义。
        self.assertGreater(compared, 100)
        self.assertGreater(valid_seen, 0)
        self.assertGreater(invalid_seen, 0)


if __name__ == "__main__":
    unittest.main()
