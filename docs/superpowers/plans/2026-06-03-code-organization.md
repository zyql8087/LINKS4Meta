# LINKS4Meta Code Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize LINKS4Meta code files by algorithm module while preserving existing runtime behavior.

**Architecture:** Perform the refactor in narrow layers: first move executable scripts from `code/` into algorithm-oriented `scripts/` folders, then add compatibility wrappers in `code/` so existing commands and tests keep working. Move tests into phase-oriented folders and update import bootstrapping so direct script imports resolve through the wrappers. Update README command examples and the organization document to point to the new layout.

**Tech Stack:** Python, PyTorch/PyTorch Geometric, unittest, PowerShell, Git.

---

### Task 1: Create Script Folder Layout

**Files:**
- Create directories under `GraphMetaMat-LINKS/scripts/`
- Move files from `GraphMetaMat-LINKS/code/`

- [ ] **Step 1: Create target folders**

Run:

```powershell
New-Item -ItemType Directory -Force -Path scripts\data,scripts\forward,scripts\pretraining,scripts\family_index,scripts\inverse_il,scripts\inverse_rl,scripts\inference,scripts\diagnostics,scripts\benchmarks,scripts\reports | Out-Null
```

Expected: directories exist.

- [ ] **Step 2: Move script files by algorithm module**

Move:

```text
code/dataset_tool.py -> scripts/data/dataset_tool.py
code/train_forward_bio.py -> scripts/forward/train_forward_bio.py
code/evaluate_forward_bio.py -> scripts/forward/evaluate_forward_bio.py
code/test_forward_bio.py -> scripts/forward/test_forward_bio.py
code/pretrain_inverse_bio.py -> scripts/pretraining/pretrain_inverse_bio.py
code/build_family_index.py -> scripts/family_index/build_family_index.py
code/rebuild_il_codebook.py -> scripts/family_index/rebuild_il_codebook.py
code/train_inverse_bio.py -> scripts/inverse_il/train_inverse_bio.py
code/rl_refine_bio.py -> scripts/inverse_rl/rl_refine_bio.py
code/inference_inverse.py -> scripts/inference/inference_inverse.py
code/run_il_diagnostics.py -> scripts/diagnostics/run_il_diagnostics.py
code/run_il_geometry_validity_diagnostics.py -> scripts/diagnostics/run_il_geometry_validity_diagnostics.py
code/run_il_oracle_coverage_diagnostics.py -> scripts/diagnostics/run_il_oracle_coverage_diagnostics.py
code/run_phase5_readout_diagnostics.py -> scripts/diagnostics/run_phase5_readout_diagnostics.py
code/run_readout_assignment_demo.py -> scripts/benchmarks/run_readout_assignment_demo.py
code/run_real_readout_benchmark.py -> scripts/benchmarks/run_real_readout_benchmark.py
code/run_surrogate_ranking_quality.py -> scripts/benchmarks/run_surrogate_ranking_quality.py
code/run_experiment_bio.py -> scripts/benchmarks/run_experiment_bio.py
code/gen_experiment_report.py -> scripts/reports/gen_experiment_report.py
```

Expected: `code/` no longer owns implementations.

### Task 2: Add Compatibility Wrappers

**Files:**
- Create `GraphMetaMat-LINKS/code/*.py` wrappers for every moved script.

- [ ] **Step 1: Add wrapper template**

Each wrapper should add the repository root to `sys.path`, import all public names
from the moved script, and call its `main()` function when executed directly:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forward.train_forward_bio import *  # noqa: F401,F403,E402
from scripts.forward.train_forward_bio import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
```

Expected: old commands such as `python code/train_forward_bio.py` still work,
and tests importing from `train_inverse_bio`, `dataset_tool`, or
`run_experiment_bio` still resolve.

### Task 3: Reorganize Tests

**Files:**
- Move `GraphMetaMat-LINKS/tests/test_*.py` into phase folders.

- [ ] **Step 1: Create test folders and move files**

Move:

```text
tests/test_phase0_phase1_pipeline.py -> tests/phase0_phase1_generation/test_pipeline.py
tests/test_phase2_split.py -> tests/phase2_split/test_family_split.py
tests/test_phase3_forward.py -> tests/phase3_forward/test_forward.py
tests/test_links_pretrain.py -> tests/pretraining/test_links_pretrain.py
tests/test_family_index_builder.py -> tests/family_index/test_family_index_builder.py
tests/test_phase4_il.py -> tests/phase4_il/test_il.py
tests/test_phase5_rl.py -> tests/phase5_rl/test_rl.py
tests/test_phase6_mcts.py -> tests/phase6_mcts/test_mcts.py
tests/test_readout_assignment.py -> tests/readout/test_readout_assignment.py
tests/test_kinematics.py -> tests/common/test_kinematics.py
tests/test_experiment_pipeline.py -> tests/integration/test_experiment_pipeline.py
tests/test_high_priority_fixes.py -> tests/regression/test_high_priority_fixes.py
```

Expected: each algorithm phase has an obvious test folder.

- [ ] **Step 2: Update test import bootstrapping**

For tests that directly insert `code/` or `LINKS-main/` paths, verify the paths
still resolve from the deeper folder nesting. Use `Path(__file__).resolve()` based
repo discovery if any test relies on fixed parent counts.

### Task 4: Update Docs and Commands

**Files:**
- Modify `README.md`
- Modify `docs/algorithm_module_code_organization_plan.md`

- [ ] **Step 1: Replace command examples**

Update README examples from `GraphMetaMat-LINKS/code/...` to the new
`GraphMetaMat-LINKS/scripts/...` paths.

- [ ] **Step 2: Update test examples**

Update README test examples from flat `tests/test_*.py` paths to phase-folder
paths.

- [ ] **Step 3: Mark script movement complete**

Update the organization document so it records that the first cleanup pass moved
scripts and tests, while source-package migration remains future work.

### Task 5: Verify

**Files:**
- No new files expected.

- [ ] **Step 1: Run targeted import smoke checks**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Expected: all discoverable tests pass or fail only for pre-existing environment
issues such as missing PyTorch Geometric.

- [ ] **Step 2: Run old wrapper entrypoint smoke check**

Run a lightweight command such as:

```powershell
python code/build_family_index.py --help
```

Expected: help text prints from the moved script.

- [ ] **Step 3: Run new entrypoint smoke check**

Run:

```powershell
python scripts/family_index/build_family_index.py --help
```

Expected: help text prints from the moved script.
