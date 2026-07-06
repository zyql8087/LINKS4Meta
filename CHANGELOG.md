# LINKS4Meta 修改日志

本文记录 LINKS4Meta 代码层面的主要变更，实验输出、模型权重和临时报告不纳入版本库变更记录。

## 2026-07-06

Commit: `e15c614` - `Add new architecture diagnostics and stricter surrogate inputs`

### 新增

- 新增 IL/诊断脚本:
  - `scripts/diagnostics/run_il_greedy_code_rank_decomp.py`
  - `scripts/diagnostics/run_il_surrogate_target_matching.py`
  - `scripts/diagnostics/run_overfit_zc_ab.py`
  - `scripts/diagnostics/run_zc_conditioning_ablation.py`
- 新增新架构配置草案:
  - `src/_config_newarch_full.yaml`
  - `src/_config_newarch_oracle_merge.yaml`
- 新增 geometry code selection 回归测试:
  - `tests/phase5_rl/test_geometry_code_selection.py`

### 修改

- Forward surrogate 输入校验收紧:
  - `BioKinematicsGNN` 现在显式要求 node feature 宽度与配置一致。
  - forward 输入必须具备合法的 `family_id`、`step_context` 和每图唯一的 `hip/knee/ankle/foot` semantic role。
  - inverse graph 在评分前必须通过 surrogate graph adapter 补齐语义 mask。
- `src/config_model_bio.yaml` 的 encoder 类型切换为 `MPNN`，并将 encoder type 传入 `GNNEncoder`。
- 强化 target-conditioned readout、MCTS、RL agent 和 reward 路径中的 geometry code / surrogate scoring 处理。
- 扩展 IL geometry validity diagnostic，支持更细粒度的 code validity 和 ranking 分析。
- 更新 phase3 forward、phase5 RL 相关测试，覆盖更严格的 surrogate 输入契约和 geometry code 选择逻辑。

### 验证

- `python -m compileall -q src scripts tests`
- `tests/phase3_forward/test_forward.py`: 9 tests OK
- `tests/phase5_rl/test_rl.py`: 11 tests OK
- `tests/phase5_rl/test_geometry_code_selection.py`: 6 tests OK
- `tests/readout/test_readout_assignment.py`: 11 tests OK
- `tests/regression/test_source_style.py`: 5 tests OK

## 2026-06-22

Commit: `20a1207` - `Refactor LINKS4Meta experiment scripts and diagnostics`

### 新增

- 将原先 `code/` 下的大型实验脚本迁移为分层 `scripts/` 入口:
  - `scripts/benchmarks/`
  - `scripts/data/`
  - `scripts/diagnostics/`
  - `scripts/family_index/`
  - `scripts/forward/`
  - `scripts/inference/`
  - `scripts/inverse_il/`
  - `scripts/inverse_rl/`
  - `scripts/pretraining/`
  - `scripts/reports/`
- 重组测试目录:
  - `tests/phase3_forward/`
  - `tests/phase4_il/`
  - `tests/phase5_rl/`
  - `tests/readout/`
  - `tests/regression/`
  - 以及 family index、integration、pretraining 等分层目录。
- 新增 `tests/regression/test_source_style.py`，约束脚本入口与源码组织风格。

### 修改

- `code/*.py` 保留为兼容入口，核心实现迁移到 `scripts/*`。
- 增加 `src/config_utils.py` 统一配置加载、路径解析和输出目录创建。
- 更新 IL、readout、MCTS、forward surrogate、family index 等模块以适配新的脚本组织。
- `.gitignore` 增加 `output/` 和 `*.docx`，避免误提交实验输出和 Word 文档。

### 验证

- `python -m compileall -q src code scripts tests`
- `tests/phase4_il/test_il.py`: 31 tests OK
- `tests/phase5_rl/test_rl.py`: 11 tests OK
- `tests/regression/test_source_style.py`: 5 tests OK
