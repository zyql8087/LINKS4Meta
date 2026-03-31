# LINKS4Meta

LINKS4Meta is a research-oriented codebase for **bio-inspired planar linkage generation and inverse kinematic design**.  
It combines a redesigned **LINKS-based mechanism dataset pipeline** with a **GraphMetaMat-style learning pipeline** for:

- mechanism generation
- kinematic simulation
- biological curve extraction
- forward surrogate modeling
- inverse design by imitation learning
- RL-based refinement
- search-based inference with MCTS

The project is organized around two main folders:

- `LINKS-main`: mechanism generation, simulation, dataset construction, and multibase J-operator data generation
- `GraphMetaMat-LINKS`: forward/inverse learning, RL refinement, experiment evaluation, and inference

## Project Goal

The goal of this project is to design planar linkage mechanisms that match three target motion signals at the same time:

- foot trajectory
- knee angle curve
- ankle angle curve

Instead of directly generating a full mechanism in one step, the project represents inverse design as a **graph expansion process** based on the **J-operator**.  
This makes the generation process more structured, more interpretable, and easier to combine with imitation learning, reinforcement learning, and tree search.

## Repository Structure

```text
LINKS4Meta/
|- LINKS-main/
|  |- data_gen_v2/                  # multibase dataset generator
|  |- Dataset/                      # original LINKS dataset notes / notebook
|  |- run_generate_80k_v2.py        # v2 generation entrypoint
|  |- dataset_builder.py            # legacy generator
|  |- sim.py                        # kinematic solver
|  `- *.pkl / *.pt                  # generated datasets and derived artifacts
|
|- GraphMetaMat-LINKS/
|  |- src/
|  |  |- generative_curve/          # forward surrogate model
|  |  |- inverse/                   # inverse IL / RL / MCTS modules
|  |  |- config_dataset.yaml
|  |  |- config_model_bio.yaml
|  |  `- config_inverse.yaml
|  |- train_forward_bio.py          # train forward model
|  |- train_inverse_bio.py          # train inverse IL model
|  |- rl_refine_bio.py              # RL refinement
|  |- run_experiment_bio.py         # experiment evaluation
|  |- inference_inverse.py          # inference + visualization
|  |- dataset_tool.py               # dataset conversion / visualization utility
|  `- tests/                        # regression and pipeline tests
|
|- output/
|  |- reports/                      # formal IL reports
|  |- comparison/                   # RL before/after comparisons
|  `- doc/                          # generated Word reports
|
`- README.md
```

## End-to-End Workflow

### 1. Mechanism seed generation

`LINKS-main/data_gen_v2` extends the original LINKS pipeline from a single 4-bar seed into a **multibase generator** with:

- 4-bar templates: `T4-A`, `T4-B`, `T4-C`, `T4-D`
- 5-bar templates: `T5-A`, `T5-B`, `T5-C`

These base mechanisms are used as valid 1-DOF seeds before further expansion.

### 2. J-operator expansion

The generator expands seed mechanisms by applying one or two dyad insertions through the J-operator:

- `4-bar -> 6-bar`
- `5-bar -> 7-bar`
- `4-bar -> 8-bar`
- `5-bar -> 9-bar`

The expansion is controlled by family-specific policies to keep the generated topologies physically meaningful and semantically consistent.

### 3. Kinematic simulation and semantic analysis

Each candidate mechanism is validated through full-cycle simulation. The pipeline extracts:

- foot trajectory
- knee angle
- ankle angle
- ROM and motion quality statistics

Only mechanisms that are structurally valid and kinematically meaningful are kept.

### 4. Dataset construction

Two types of learning-ready data are produced:

- **forward data**: graph -> `(y_foot, y_knee, y_ankle)`
- **inverse expert paths**: `base graph + J-operator action + target curves`

The dataset code also performs:

- geometry deduplication
- motion deduplication
- family-aware split generation
- hard-sample selection

### 5. Forward surrogate training

`GraphMetaMat-LINKS/train_forward_bio.py` trains a graph neural network surrogate that predicts:

- 2D foot trajectory
- knee angle sequence
- ankle angle sequence

This surrogate is later frozen and reused as a reward model in RL and as a fast evaluator during inference.

### 6. Inverse design by imitation learning

`GraphMetaMat-LINKS/train_inverse_bio.py` trains an inverse policy consisting of:

- `CurveEncoder`: encodes target curves
- `GNNPolicy`: predicts topology choices
- `GeometryHead` (conditional VAE): predicts new node coordinates

The model learns how to reconstruct the expert J-operator action from the target motion curves.

### 7. RL refinement

`GraphMetaMat-LINKS/rl_refine_bio.py` refines the inverse policy with PPO-style updates using:

- frozen forward surrogate rewards
- geometry validity priors
- batched action selection
- differentiable coordinate refinement

### 8. Inference and evaluation

The project supports:

- experiment comparison: retrieval / IL / RL / MCTS
- hard-set evaluation
- inference visualization for generated mechanisms vs target curves

## Main Design Choices

### Unified biological semantics

The project fixes a Strategy-B-style semantic mapping so that inverse learning always works with the same conceptual chain:

- `knee`
- `ankle`
- `foot`

This is critical because the project is not only generating mechanisms, but generating mechanisms with **consistent biological interpretation**.

### Group-aware split instead of naive random split

To reduce leakage between train and test, the dataset is split by:

- family
- motion cluster

This makes evaluation more trustworthy than a plain random sample split.

### Structured inverse design instead of one-shot generation

Rather than predicting an entire mechanism from scratch, the project predicts **how to expand a base graph**.  
This is more compatible with:

- expert demonstrations
- RL policies
- MCTS search
- geometric validity checks

## Key Entry Points

### Generate a pilot dataset

```bash
python LINKS-main/run_generate_80k_v2.py --pilot-only --pilot-per-family 10 --output-dir LINKS-main/output/data_gen_v2_pilot10
```

### Convert a `.pkl` dataset into `.pt` graph data with curves

```bash
python GraphMetaMat-LINKS/dataset_tool.py convert --input_pkl LINKS-main/biological_6bar_dataset_80k_with_geninfo.pkl
```

### Train the forward surrogate

```bash
python GraphMetaMat-LINKS/train_forward_bio.py
```

### Train the inverse IL model

```bash
python GraphMetaMat-LINKS/train_inverse_bio.py --config GraphMetaMat-LINKS/src/config_inverse.yaml
```

### Run RL refinement

```bash
python GraphMetaMat-LINKS/rl_refine_bio.py --config GraphMetaMat-LINKS/src/config_inverse.yaml
```

### Run experiment evaluation

```bash
python GraphMetaMat-LINKS/run_experiment_bio.py --config GraphMetaMat-LINKS/src/config_inverse.yaml
```

### Run inference and save visualizations

```bash
python GraphMetaMat-LINKS/inference_inverse.py --config GraphMetaMat-LINKS/src/config_inverse.yaml --model_type rl
```

## Important Data / Model Artifacts

Common artifacts already present in the workspace include:

- `LINKS-main/biological_6bar_dataset_80k_with_geninfo.pkl`
- `LINKS-main/biological_6bar_dataset_80k_with_curves.pt`
- `LINKS-main/il_expert_paths_80k.pt`
- `GraphMetaMat-LINKS/model_bio_best.pt`
- `GraphMetaMat-LINKS/model_inverse_il.pt`
- `GraphMetaMat-LINKS/model_inverse_rl.pt`

Formal reports and comparisons are stored under:

- `output/reports/`
- `output/comparison/`
- `output/doc/`

## Environment Notes

This repository mixes code from different stages of development. In practice, the learning pipeline expects a Python environment with:

- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- PyYAML
- tqdm
- scikit-learn
- svgpath2mpl

The original `LINKS-main` README also mentions TensorFlow because the upstream LINKS project included additional tooling.  
For the current `GraphMetaMat-LINKS` learning pipeline, PyTorch and PyG are the primary requirements.

## Current Status

The codebase already contains:

- a multibase mechanism generator
- biological curve extraction
- forward surrogate training
- inverse IL training
- RL refinement
- experiment scripts
- regression tests for several high-priority fixes

The main remaining research challenge is improving the stability and efficiency of higher-complexity families such as `9-bar`, while pushing the inverse pipeline toward stronger formal results.

## Acknowledgement

This project builds on ideas from:

- the original LINKS dataset and simulation pipeline
- GraphMetaMat-style graph-based learning for inverse design

The current repository adapts and extends those ideas toward **bio-inspired linkage design with foot/knee/ankle motion targets**.
