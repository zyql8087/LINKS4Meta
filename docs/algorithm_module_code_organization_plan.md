# LINKS4Meta algorithm-module code organization plan

Date: 2026-06-03

This document began as a review-only organization proposal. The first cleanup
pass has now been applied: executable entrypoints were moved from `code/` to
algorithm-oriented `scripts/` folders, `code/` now contains compatibility
wrappers, and tests were moved into phase-oriented folders. Importable `src/`
module migration remains future work.

## Scope

The workspace currently contains several code-bearing roots:

- `LINKS-main/`: LINKS-side mechanism generation, J-operator expansion, sampling,
  kinematics checks, and raw dataset export.
- `GraphMetaMat-LINKS/`: the active Git repository for GraphMetaMat-style forward
  and inverse learning.
- `GraphMetaMat-LINKS/scripts/`: executable entrypoints and experiment scripts.
- `GraphMetaMat-LINKS/code/`: compatibility wrappers for old entrypoint paths.
- `GraphMetaMat-LINKS/src/`: importable training, model, data, inverse-design,
  and utility modules.
- `GraphMetaMat-LINKS/tests/`: phase-oriented unit and regression tests.
- `demo/`: runnable demos, smoke configs, generated artifacts, logs, and legacy
  scripts.

Non-code or generated roots such as `DL/`, `output/`, `tmp/`, `__pycache__/`,
checkpoints, reports, plots, and `.pt/.pkl/.json` artifacts should stay outside
the algorithm code layout. They should be documented as artifacts, not treated as
source modules.

## Current algorithm framework

The project README describes the algorithm pipeline as:

1. Phase 0-1: multibase family dataset generation.
2. Phase 2: family-aware group split.
3. Phase 3: forward surrogate before policy.
4. LINKS pretraining and family index export.
5. Phase 4: multistep imitation learning.
6. Phase 5: RL refinement only.
7. Phase 6: inference-time MCTS reranking.
8. Experiment, benchmark, diagnostics, reporting, and demo workflows.

This is the most useful organizing axis. Some files are shared across phases,
so the target layout should separate reusable library modules from executable
scripts.

## Organization approaches

### Approach A: Pure phase-based folders

Example: `phase0_dataset/`, `phase2_split/`, `phase3_forward/`, `phase4_il/`,
`phase5_rl/`, `phase6_mcts/`.

Trade-off: easy to map to papers and reports, but shared code gets duplicated or
placed awkwardly. Files such as `gnn_policy.py`, `curve_encoder.py`,
`experiment_utils.py`, and `config_utils.py` serve more than one phase.

### Approach B: Pure software-domain folders

Example: `data/`, `models/`, `training/`, `evaluation/`, `experiments/`,
`scripts/`.

Trade-off: clean as software architecture, but less aligned with the actual
algorithm story. It becomes harder for a researcher to find "Phase 4 IL" or
"Phase 6 MCTS" code directly.

### Approach C: Hybrid algorithm modules with shared infrastructure

Recommended.

Keep the algorithm phases visible, but extract cross-cutting code into shared
modules. Keep executable scripts separate from importable libraries. This gives
research-readable structure without forcing artificial duplication.

## Recommended target layout

```text
GraphMetaMat-LINKS/
|- src/
|  |- links4meta/
|  |  |- common/                  # config, utility, graph encoder/pooler, kinematics helpers
|  |  |- data/                    # PyG conversion, data loading, split helpers
|  |  |- generation/              # optional future home if LINKS-main is merged
|  |  |- forward/                 # Phase 3 surrogate model, training helpers, metrics
|  |  |- pretraining/             # LINKS pretraining modules
|  |  |- family_index/            # family step index and action/codebook materialization
|  |  |- inverse/
|  |  |  |- action_space/         # J-operator action codebook and action decoding
|  |  |  |- policy/               # GNN policy and curve encoder
|  |  |  |- imitation/            # Phase 4 expert traces, IL datasets, IL losses
|  |  |  |- reinforcement/        # Phase 5 environment and PPO agent
|  |  |  |- search/               # Phase 6 MCTS
|  |  |  |- inference/            # runtime bundle loading, rollout, visualization helpers
|  |  |  `- readout/              # readout assignment and target mapping
|  |  `- experiments/             # metrics, benchmark helpers, target selection
|  |
|- scripts/
|  |- data/
|  |- forward/
|  |- pretraining/
|  |- family_index/
|  |- inverse_il/
|  |- inverse_rl/
|  |- inference/
|  |- diagnostics/
|  |- benchmarks/
|  `- reports/
|
|- tests/
|  |- phase0_phase1_generation/
|  |- phase2_split/
|  |- phase3_forward/
|  |- pretraining/
|  |- family_index/
|  |- phase4_il/
|  |- phase5_rl/
|  |- phase6_mcts/
|  |- readout/
|  `- integration/
```

The first cleanup pass keeps the current `src/` namespace and reorganizes
entrypoints into `scripts/` subfolders. A deeper pass can later rename `src`
imports to `src.links4meta.*` or an installed package name.

## Current file classification

### Phase 0-1: multibase generation and J-operator dataset construction

Current files:

- `LINKS-main/data_gen_v2/base_generators.py`
- `LINKS-main/data_gen_v2/constraints.py`
- `LINKS-main/data_gen_v2/j_operator.py`
- `LINKS-main/data_gen_v2/family_policies.py`
- `LINKS-main/data_gen_v2/descriptors.py`
- `LINKS-main/data_gen_v2/semantic_branch.py`
- `LINKS-main/data_gen_v2/kinematics_eval.py`
- `LINKS-main/data_gen_v2/dataset_builder_v2.py`
- `LINKS-main/data_gen_v2/dataset_report_v2.py`
- `LINKS-main/data_gen_v2/specs.py`
- `LINKS-main/data_gen_v2/diversity_sampler.py`
- `LINKS-main/run_generate_80k_v2.py`
- `demo/run_phase0_phase1_pipeline.py`
- `GraphMetaMat-LINKS/tests/phase0_phase1_generation/test_pipeline.py`

Suggested target:

- Keep in `LINKS-main/data_gen_v2/` for now, because it is a separate upstream
  generation subsystem.
- If merged later, place under `src/links4meta/generation/` and expose generation
  scripts through `scripts/data/`.

Notes:

- `diversity_sampler.py` serves both Phase 0-1 sampling and Phase 2 split logic.
  If moved, split it into generation sampling and group-split utilities only when
  tests prove the boundary.
- Older files such as `LINKS-main/dataset_builder.py`,
  `LINKS-main/combine_family_generations.py`, `LINKS-main/sim.py`, and
  `LINKS-main/utils.py` should be marked as legacy or adapter code before moving.

### Phase 2: family-aware split and data loading

Current files:

- `GraphMetaMat-LINKS/src/data_load.py`
- `GraphMetaMat-LINKS/src/forward_dataset_utils.py`
- `GraphMetaMat-LINKS/src/config_dataset.yaml`
- `GraphMetaMat-LINKS/scripts/data/dataset_tool.py`
- `GraphMetaMat-LINKS/code/dataset_tool.py` compatibility wrapper
- `GraphMetaMat-LINKS/tests/phase2_split/test_family_split.py`

Suggested target:

- `src/links4meta/data/loaders.py`
- `src/links4meta/data/pyg_conversion.py`
- `src/links4meta/data/splits.py`
- `scripts/data/dataset_tool.py`

Notes:

- `dataset_tool.py` is an executable conversion/inspection tool, not a library
  module.
- `forward_dataset_utils.py` is used by both data loading and the forward model;
  keep it in data utilities, not under forward training.

### Phase 3: forward surrogate

Current files:

- `GraphMetaMat-LINKS/src/generative_curve/GNN_model_biokinematics.py`
- `GraphMetaMat-LINKS/src/generative_curve/GNN_train_biokinematics.py`
- `GraphMetaMat-LINKS/src/forward_metrics.py`
- `GraphMetaMat-LINKS/src/config_model_bio.yaml`
- `GraphMetaMat-LINKS/scripts/forward/train_forward_bio.py`
- `GraphMetaMat-LINKS/scripts/forward/evaluate_forward_bio.py`
- `GraphMetaMat-LINKS/scripts/forward/test_forward_bio.py`
- `GraphMetaMat-LINKS/scripts/benchmarks/run_surrogate_ranking_quality.py`
- `GraphMetaMat-LINKS/code/*.py` compatibility wrappers for these entrypoints
- `demo/run_phase3_forward_suite.py`
- `GraphMetaMat-LINKS/tests/phase3_forward/test_forward.py`

Suggested target:

- `src/links4meta/forward/model.py`
- `src/links4meta/forward/training.py`
- `src/links4meta/forward/metrics.py`
- `scripts/forward/train.py`
- `scripts/forward/evaluate.py`
- `scripts/forward/test.py`
- `scripts/benchmarks/surrogate_ranking_quality.py`

Notes:

- `run_surrogate_ranking_quality.py` is a benchmark/diagnostic script, not core
  forward training.
- `forward_metrics.py` is also used by experiments, so imports should remain
  stable during migration.

### Shared model and utility infrastructure

Current files:

- `GraphMetaMat-LINKS/src/config_utils.py`
- `GraphMetaMat-LINKS/src/utils.py`
- `GraphMetaMat-LINKS/src/layers_encoder.py`
- `GraphMetaMat-LINKS/src/layers_pooler.py`
- `GraphMetaMat-LINKS/src/kinematics_extract.py`
- `GraphMetaMat-LINKS/src/__init__.py`
- `GraphMetaMat-LINKS/tests/common/test_kinematics.py`

Suggested target:

- `src/links4meta/common/config.py`
- `src/links4meta/common/nn.py`
- `src/links4meta/common/graph_encoder.py`
- `src/links4meta/common/pooling.py`
- `src/links4meta/common/kinematics.py`

Notes:

- These files should move before or together with forward and inverse modules,
  because many modules import them.
- `layers_encoder.py` is used by the forward surrogate and inverse policy.

### LINKS pretraining

Current files:

- `GraphMetaMat-LINKS/src/inverse/pretrain_links.py`
- `GraphMetaMat-LINKS/src/inverse/curve_encoder.py`
- `GraphMetaMat-LINKS/src/pretrain_links.yaml`
- `GraphMetaMat-LINKS/scripts/pretraining/pretrain_inverse_bio.py`
- `GraphMetaMat-LINKS/code/pretrain_inverse_bio.py` compatibility wrapper
- `GraphMetaMat-LINKS/tests/pretraining/test_links_pretrain.py`

Suggested target:

- `src/links4meta/pretraining/links_pretrain.py`
- `src/links4meta/inverse/policy/curve_encoder.py`
- `scripts/pretraining/pretrain_inverse.py`

Notes:

- `curve_encoder.py` is not only pretraining code. It is part of the inverse
  policy stack and is also used by IL/RL/inference.

### Family index and action/codebook materialization

Current files:

- `GraphMetaMat-LINKS/src/inverse/action_codebook.py`
- `GraphMetaMat-LINKS/src/inverse/family_index_builder.py`
- `GraphMetaMat-LINKS/scripts/family_index/build_family_index.py`
- `GraphMetaMat-LINKS/scripts/family_index/rebuild_il_codebook.py`
- `GraphMetaMat-LINKS/code/*.py` compatibility wrappers for these entrypoints
- `GraphMetaMat-LINKS/tests/family_index/test_family_index_builder.py`

Suggested target:

- `src/links4meta/family_index/builder.py`
- `src/links4meta/inverse/action_space/codebook.py`
- `scripts/family_index/build.py`
- `scripts/family_index/rebuild_codebook.py`

Notes:

- `action_codebook.py` is large and important. It likely deserves later internal
  splitting into action schema, codebook IO, quantization/bucketing, and action
  decoding.
- `rebuild_il_codebook.py` is a maintenance script and should stay outside
  importable package code.

### Phase 4: multistep imitation learning

Current files:

- `GraphMetaMat-LINKS/src/inverse/phase4_il.py`
- `GraphMetaMat-LINKS/src/inverse/train_il.py`
- `GraphMetaMat-LINKS/src/inverse/gnn_policy.py`
- `GraphMetaMat-LINKS/src/train_links4meta_il.yaml`
- `GraphMetaMat-LINKS/scripts/inverse_il/train_inverse_bio.py`
- `GraphMetaMat-LINKS/scripts/diagnostics/run_il_diagnostics.py`
- `GraphMetaMat-LINKS/scripts/diagnostics/run_il_geometry_validity_diagnostics.py`
- `GraphMetaMat-LINKS/scripts/diagnostics/run_il_oracle_coverage_diagnostics.py`
- `GraphMetaMat-LINKS/code/*.py` compatibility wrappers for these entrypoints
- `GraphMetaMat-LINKS/tests/phase4_il/test_il.py`
- `GraphMetaMat-LINKS/tests/regression/test_high_priority_fixes.py`

Suggested target:

- `src/links4meta/inverse/imitation/expert_paths.py`
- `src/links4meta/inverse/imitation/dataset.py`
- `src/links4meta/inverse/imitation/losses.py`
- `src/links4meta/inverse/imitation/training.py`
- `src/links4meta/inverse/policy/gnn_policy.py`
- `scripts/inverse_il/train.py`
- `scripts/diagnostics/il_diagnostics.py`
- `scripts/diagnostics/il_geometry_validity.py`
- `scripts/diagnostics/il_oracle_coverage.py`

Notes:

- `phase4_il.py` is the highest-priority split candidate because it is very
  large and mixes expert trace extraction, dataset construction, family mapping,
  decoding, loss helpers, and evaluation utilities.
- `train_inverse_bio.py` is an orchestration entrypoint and currently exports
  helpers imported directly by tests. If moved, create compatibility wrappers or
  move those helpers into importable library modules first.

### Phase 5: RL refinement

Current files:

- `GraphMetaMat-LINKS/src/inverse/phase5_rl.py`
- `GraphMetaMat-LINKS/src/inverse/rl_env.py`
- `GraphMetaMat-LINKS/src/inverse/rl_agent.py`
- `GraphMetaMat-LINKS/src/config_inverse.yaml`
- `GraphMetaMat-LINKS/scripts/inverse_rl/rl_refine_bio.py`
- `GraphMetaMat-LINKS/code/rl_refine_bio.py` compatibility wrapper
- `GraphMetaMat-LINKS/tests/phase5_rl/test_rl.py`

Suggested target:

- `src/links4meta/inverse/reinforcement/curriculum.py`
- `src/links4meta/inverse/reinforcement/env.py`
- `src/links4meta/inverse/reinforcement/ppo.py`
- `scripts/inverse_rl/refine.py`

Notes:

- `rl_env.py` also contains graph validation, surrogate loading, semantic mask,
  reward, and J-operator application utilities. Some of those may belong in
  `common/graph_validation.py`, `experiments/rewards.py`, or
  `inverse/action_space/apply.py` after a careful refactor.

### Phase 6: inference-time MCTS and inverse inference

Current files:

- `GraphMetaMat-LINKS/src/inverse/mcts.py`
- `GraphMetaMat-LINKS/src/inverse/inference_runtime.py`
- `GraphMetaMat-LINKS/scripts/inference/inference_inverse.py`
- `GraphMetaMat-LINKS/code/inference_inverse.py` compatibility wrapper
- `GraphMetaMat-LINKS/tests/phase6_mcts/test_mcts.py`

Suggested target:

- `src/links4meta/inverse/search/mcts.py`
- `src/links4meta/inverse/inference/runtime.py`
- `scripts/inference/inverse.py`

Notes:

- `inference_runtime.py` depends on policy, RL agent, MCTS, and codebook logic.
  Move it after those lower-level modules are stable.

### Readout assignment and target mapping

Current files:

- `GraphMetaMat-LINKS/src/inverse/readout_assignment.py`
- `GraphMetaMat-LINKS/scripts/benchmarks/run_readout_assignment_demo.py`
- `GraphMetaMat-LINKS/scripts/benchmarks/run_real_readout_benchmark.py`
- `GraphMetaMat-LINKS/scripts/diagnostics/run_phase5_readout_diagnostics.py`
- `GraphMetaMat-LINKS/code/*.py` compatibility wrappers for these entrypoints
- `GraphMetaMat-LINKS/tests/readout/test_readout_assignment.py`

Suggested target:

- `src/links4meta/inverse/readout/assignment.py`
- `scripts/diagnostics/phase5_readout.py`
- `scripts/benchmarks/real_readout.py`
- `scripts/benchmarks/readout_assignment_demo.py`

Notes:

- `readout_assignment.py` is another large split candidate. Separate rule-based
  assignment, surrogate-target assignment, metrics, and diagnostics only after
  behavior is covered by tests.

### Experiment orchestration, metrics, reporting

Current files:

- `GraphMetaMat-LINKS/src/inverse/experiment_utils.py`
- `GraphMetaMat-LINKS/scripts/benchmarks/run_experiment_bio.py`
- `GraphMetaMat-LINKS/scripts/reports/gen_experiment_report.py`
- `GraphMetaMat-LINKS/code/*.py` compatibility wrappers for these entrypoints
- `GraphMetaMat-LINKS/tests/integration/test_experiment_pipeline.py`

Suggested target:

- `src/links4meta/experiments/metrics.py`
- `src/links4meta/experiments/targets.py`
- `src/links4meta/experiments/splits.py`
- `scripts/benchmarks/run_experiment.py`
- `scripts/reports/gen_experiment_report.py`

Notes:

- `experiment_utils.py` should not become a catch-all. During refactor, split by
  responsibility: target feature building, reward/metrics, fixed split handling,
  and hard-sample selection.
- Report generation is not algorithm core. Keep it in `scripts/reports/`.

## Test classification

Current test folders:

- `tests/phase0_phase1_generation/test_pipeline.py`
- `tests/phase2_split/test_family_split.py`
- `tests/phase3_forward/test_forward.py`
- `tests/pretraining/test_links_pretrain.py`
- `tests/family_index/test_family_index_builder.py`
- `tests/phase4_il/test_il.py`
- `tests/phase5_rl/test_rl.py`
- `tests/phase6_mcts/test_mcts.py`
- `tests/readout/test_readout_assignment.py`
- `tests/common/test_kinematics.py`
- `tests/integration/test_experiment_pipeline.py`
- `tests/regression/test_high_priority_fixes.py`

## Recommended migration order

1. Documentation-only pass. Completed.
   - Keep this document and update README with a high-level map.
   - Do not move files.

2. Script-only pass. Completed.
   - Move `GraphMetaMat-LINKS/code/*.py` into `scripts/*`.
   - Add temporary wrapper files or update run commands and tests in the same
     commit.
   - Verify entrypoint commands and tests that import script helpers.

3. Shared-library pass.
   - Move `config_utils.py`, `utils.py`, `layers_encoder.py`,
     `layers_pooler.py`, `kinematics_extract.py`.
   - Update imports mechanically.

4. Forward/data/pretraining pass.
   - Move forward surrogate and data-loading modules.
   - Verify Phase 2, Phase 3, and pretraining tests.

5. Inverse core pass.
   - Move action codebook, family index, policy, IL, RL, MCTS, inference, and
     readout modules.
   - Keep compatibility wrappers until all tests and scripts use the new imports.

6. Large-file decomposition pass.
   - Split `phase4_il.py`, `readout_assignment.py`, `rl_env.py`,
     `train_inverse_bio.py`, and `experiment_utils.py` by responsibility.
   - Do this after file moves, not at the same time, to keep behavioral review
     simple.

## Migration risks

- Several tests import helpers directly from script files, especially
  `train_inverse_bio.py`, `run_experiment_bio.py`, and `dataset_tool.py`.
  Compatibility wrappers under `code/` preserve those imports after script
  relocation.
- `src/inverse/phase4_il.py`, `src/inverse/rl_env.py`, and
  `src/inverse/readout_assignment.py` contain cross-phase helper functions.
- `LINKS-main` is outside the active `GraphMetaMat-LINKS` Git repository. Moving
  it into the active package is a larger repository-structure decision.
- Generated artifacts and source files are mixed under `demo/` and `output/`.
  They should be cleaned by policy, not merged into source layout.
- Renaming the package from `src.*` to `links4meta.*` is desirable long-term but
  should be a dedicated import-migration pass.

## Verification checklist for future code movement

After any actual movement, run at least:

- `python -m unittest tests/phase0_phase1_generation/test_pipeline.py`
- `python -m unittest tests/phase2_split/test_family_split.py`
- `python -m unittest tests/phase3_forward/test_forward.py`
- `python -m unittest tests/pretraining/test_links_pretrain.py`
- `python -m unittest tests/family_index/test_family_index_builder.py`
- `python -m unittest tests/phase4_il/test_il.py`
- `python -m unittest tests/phase5_rl/test_rl.py`
- `python -m unittest tests/phase6_mcts/test_mcts.py`
- `python -m unittest tests/readout/test_readout_assignment.py`
- `python -m unittest tests/integration/test_experiment_pipeline.py`

Also smoke-test the key entrypoints after script relocation:

- dataset conversion
- forward train/evaluate
- LINKS pretraining
- family index build
- inverse IL train
- RL refinement
- inverse inference
- benchmark/report scripts

## Immediate recommendation

For the next source-package cleanup step, keep the compatibility wrappers in
place and migrate importable `src/` modules in small batches. The lowest-risk
implementation plan is:

1. Keep `code/` wrappers until all external commands have moved to `scripts/`.
2. Move shared `src` infrastructure first.
3. Move forward/data/pretraining modules next.
4. Move inverse core modules last.
5. Split large files only after package imports are stable.
