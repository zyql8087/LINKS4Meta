# LINKS4Meta

LINKS4Meta is a research codebase for **bio-inspired planar linkage generation and inverse design**.
It combines:

- a redesigned **LINKS-based multibase dataset generator**
- a **GraphMetaMat-style forward / inverse learning pipeline**
- a staged training route for **pretraining -> imitation learning -> RL refinement -> inference-time MCTS reranking**

The repository targets mechanisms whose motion matches three biological targets at the same time:

- foot trajectory
- knee angle curve
- ankle angle curve

## What Is Implemented

The current codebase includes the full engineering path from dataset generation to inverse inference:

1. **Phase 0-1: multibase family dataset generation**
   - 4-bar / 5-bar seeds
   - 1-step and 2-step J-operator expansion
   - 6 / 7 / 8 / 9-bar family construction
   - fail-fast structural and kinematic filtering

2. **Phase 2: family-aware group split**
   - split by family plus topology / geometry / motion neighborhood structure
   - leakage audit fields and split reports

3. **Phase 3: forward surrogate before policy**
   - graph-to-motion forward model
   - per-family evaluation
   - retrieval and ablation baselines
   - RL gate based on surrogate quality

4. **Phase 4: multistep imitation learning**
   - curriculum from 6/7-bar to 8/9-bar to unified family-conditioned IL
   - step-level expert traces extracted from `generation_trace`
   - discrete geometry codebook for J-operator local dyad actions

5. **Phase 5: RL as refinement only**
   - multistep episode environment
   - terminal reward based on foot / knee / ankle target match
   - retrieval / IL / IL+RL / IL+RL+MCTS comparison hooks

6. **Phase 6: MCTS only at inference**
   - top-k rollout reranking
   - no training-time tree search

7. **LINKS pretraining and family index export**
   - graph encoder pretraining
   - forward surrogate style pretraining head
   - validity classifier pretraining
   - exported family IL assets:
     - `geom_codebook_v1.json`
     - `family_group_split_v1.json`
     - `family_step_index_v1.pt`
     - `family_step_index_v1.jsonl`

## Repository Layout

```text
LINKS4Meta/
|- LINKS-main/
|  |- data_gen_v2/                     # multibase mechanism generator
|  |- output/                          # generated datasets and reports
|  `- run_generate_80k_v2.py
|
|- GraphMetaMat-LINKS/
|  |- code/
|  |  |- dataset_tool.py
|  |  |- train_forward_bio.py
|  |  |- pretrain_inverse_bio.py
|  |  |- build_family_index.py
|  |  |- train_inverse_bio.py
|  |  |- rl_refine_bio.py
|  |  |- run_experiment_bio.py
|  |  `- inference_inverse.py
|  |- src/
|  |  |- inverse/                      # IL / RL / MCTS / pretrain modules
|  |  |- generative_curve/             # forward surrogate model
|  |  |- config_dataset.yaml
|  |  |- config_model_bio.yaml
|  |  |- config_inverse.yaml
|  |  |- pretrain_links.yaml
|  |  `- train_links4meta_il.yaml
|  `- tests/
|
|- demo/
|  |- outputs/                         # checkpoints, reports, family index, smoke outputs
|  |- diagnostics/
|  |- smoke/
|  `- legacy/
|
`- README.md
```

## Core Design

### 1. Inverse design is a J-operator expansion process

The inverse model does not generate a full mechanism in one shot.
Instead, it predicts how to expand a valid base graph through one or two J-operator steps.

This makes the pipeline:

- more structured
- easier to supervise with expert traces
- compatible with RL refinement
- compatible with inference-time search

### 2. Geometry is discretized as a local dyad codebook

The current inverse action is built around:

- anchor pair selection
- support anchor selection
- local geometry code selection
- optional stop behavior at the sequence level

The geometry codebook uses a local ratio-based parameterization rather than raw global coordinates.
This is saved to `geom_codebook_v1.json` and reused across IL / RL / inference.

### 3. LINKS data is used for pretraining, not as the main IL target

The training route is intentionally split:

- **LINKS pretraining** learns generic structure-motion priors
- **LINKS4Meta family traces** provide the actual family-aware IL supervision
- **RL / MCTS** only refine target matching after the supervised pipeline is already stable

### 4. Splits are group-aware, not random

Train / val / test are built as family-aware group splits using topology / geometry / motion neighborhoods.
This is meant to reduce leakage and avoid overly optimistic surrogate or inverse results.

## Environment

This project is expected to run inside the **GMM Python environment**.

Typical dependencies include:

- Python 3.9+
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- PyYAML
- tqdm

If you are using the same local setup as the development workspace, the interpreter is typically:

```bash
F:\Anaconda\envs\GMM\python.exe
```

In the examples below, `python` assumes the GMM environment is already activated.

## Main Data Artifacts

Typical generated inputs:

- `LINKS-main/output/.../diverse_dataset_v2.pkl`
- `LINKS-main/output/.../diverse_dataset_v2_with_curves.pt`
- `LINKS-main/output/.../split_indices_v2.json`

Typical learning artifacts:

- `demo/outputs/pretrain/links_pretrain_cache.pt`
- `demo/outputs/checkpoints/graphmetamat_links/model_inverse_links_pretrain.pt`
- `demo/outputs/checkpoints/graphmetamat_links/model_bio_best.pt`
- `demo/outputs/checkpoints/graphmetamat_links/model_inverse_il.pt`
- `demo/outputs/checkpoints/graphmetamat_links/model_inverse_rl.pt`
- `demo/outputs/family_index/family_step_index_v1.pt`
- `demo/outputs/family_index/family_step_index_v1.jsonl`
- `demo/outputs/family_index/family_group_split_v1.json`
- `demo/outputs/family_index/geom_codebook_v1.json`

## Quick Start

### 1. Generate or prepare the dataset

Use the LINKS-side generator to build the multibase dataset:

```bash
python LINKS-main/run_generate_80k_v2.py
```

If needed, convert the `.pkl` dataset into the `.pt` format used by the forward pipeline:

```bash
python GraphMetaMat-LINKS/code/dataset_tool.py ^
  --input_pkl LINKS-main/output/data_gen_v2_final80k_20260331/diverse_dataset_v2.pkl ^
  convert ^
  --output_pt LINKS-main/output/data_gen_v2_final80k_20260331/diverse_dataset_v2_with_curves.pt
```

### 2. Train the forward surrogate

```bash
python GraphMetaMat-LINKS/code/train_forward_bio.py ^
  --config_dataset GraphMetaMat-LINKS/src/config_dataset.yaml
```

### 3. Run LINKS pretraining

This uses the dedicated pretraining config:

```bash
python GraphMetaMat-LINKS/code/pretrain_inverse_bio.py ^
  --config GraphMetaMat-LINKS/src/pretrain_links.yaml
```

This stage exports:

- `graph_encoder.pt`
- `forward_backbone.pt`
- `validity_head.pt`
- the full bundled pretrain checkpoint

### 4. Build family-aware IL assets

This materializes the step-level family index, split file, and codebook JSON:

```bash
python GraphMetaMat-LINKS/code/build_family_index.py ^
  --config GraphMetaMat-LINKS/src/train_links4meta_il.yaml ^
  --export-jsonl
```

### 5. Train the inverse IL policy

```bash
python GraphMetaMat-LINKS/code/train_inverse_bio.py ^
  --config GraphMetaMat-LINKS/src/train_links4meta_il.yaml
```

The IL entrypoint will:

- extract multistep expert paths if needed
- load or run LINKS pretraining
- build family index assets if enabled in config
- train the staged family-conditioned IL policy

### 6. Run RL refinement

```bash
python GraphMetaMat-LINKS/code/rl_refine_bio.py ^
  --config GraphMetaMat-LINKS/src/config_inverse.yaml
```

### 7. Run experiment comparison

```bash
python GraphMetaMat-LINKS/code/run_experiment_bio.py ^
  --config GraphMetaMat-LINKS/src/config_inverse.yaml
```

### 8. Run inverse inference

```bash
python GraphMetaMat-LINKS/code/inference_inverse.py ^
  --config GraphMetaMat-LINKS/src/config_inverse.yaml ^
  --model_type rl
```

## Important Config Files

- `GraphMetaMat-LINKS/src/pretrain_links.yaml`
  - LINKS-side pretraining only
  - graph encoder / forward / validity tasks

- `GraphMetaMat-LINKS/src/train_links4meta_il.yaml`
  - family-aware imitation learning
  - multistep curriculum
  - family index export

- `GraphMetaMat-LINKS/src/config_inverse.yaml`
  - integrated inverse / RL / experiment config
  - current general-purpose runtime config

## Testing

Unit and regression tests live under:

- `GraphMetaMat-LINKS/tests/`

Examples:

```bash
python -m unittest GraphMetaMat-LINKS/tests/test_phase4_il.py
python -m unittest GraphMetaMat-LINKS/tests/test_links_pretrain.py
python -m unittest GraphMetaMat-LINKS/tests/test_family_index_builder.py
```

## Current Status

The project is already set up to support:

- reproducible multibase data generation
- family-aware split construction
- forward surrogate training
- LINKS pretraining
- multistep imitation learning
- RL refinement
- inference-time MCTS reranking

The main research challenge now is not basic engineering completeness, but improving:

- 8 / 9-bar stability
- surrogate calibration on hard families
- inverse success rate on harder motion targets
- final benchmark quality under strict group splits

## Acknowledgement

This repository builds on:

- the original LINKS mechanism generation / simulation workflow
- GraphMetaMat-style graph learning ideas for inverse design

The current implementation adapts both directions into a single pipeline for **bio-inspired linkage design under foot / knee / ankle motion targets**.
