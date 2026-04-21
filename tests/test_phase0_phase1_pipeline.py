from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LINKS_ROOT = WORKSPACE_ROOT / "LINKS-main"
if str(LINKS_ROOT) not in sys.path:
    sys.path.insert(0, str(LINKS_ROOT))

from data_gen_v2.dataset_builder_v2 import generate_dataset_v2
from data_gen_v2.dataset_report_v2 import build_dataset_report
from data_gen_v2.kinematics_eval import analyze_semantics
from data_gen_v2.specs import LINKS_ROOT as SPEC_LINKS_ROOT, GenerationSpec


def test_generation_spec_uses_phase1_pilot_targets_and_repo_relative_root(tmp_path):
    spec = GenerationSpec(output_dir=tmp_path / "pilot")

    assert spec.output_dir == tmp_path / "pilot"
    assert SPEC_LINKS_ROOT == LINKS_ROOT.resolve()
    assert spec.families["6bar"].pilot_target == 2000
    assert spec.families["7bar"].pilot_target == 1200
    assert spec.families["8bar"].pilot_target == 1600
    assert spec.families["9bar"].pilot_target == 600


def test_generated_sample_contains_phase1_required_fields(tmp_path):
    spec = GenerationSpec(seed=7, pilot_only=True, output_dir=tmp_path / "pilot")
    spec.families = {"6bar": replace(spec.families["6bar"], pilot_target=1)}

    result = generate_dataset_v2(spec)

    assert len(result["final_samples"]) == 1
    sample = result["final_samples"][0]
    required_keys = {
        "family_id",
        "seed_type",
        "step_count",
        "step_roles",
        "gen_info",
        "generation_trace",
        "topology_signature",
        "node_coords_init",
        "foot_curve",
        "knee_curve",
        "ankle_curve",
        "failure_reason",
    }
    assert required_keys.issubset(sample.keys())
    assert sample["family_id"] == "6bar"
    assert sample["step_count"] == 1
    assert sample["step_roles"] == ["semantic"]
    assert sample["failure_reason"] == ""
    assert sample["generation_trace"][0]["step_role"] == "semantic"
    assert sample["analysis"]["gen_info"] == sample["gen_info"]
    assert sample["foot_curve"].shape == (200, 2)
    assert sample["knee_curve"].shape == (200,)
    assert sample["ankle_curve"].shape == (200,)


def test_analyze_semantics_rejects_low_complexity_motion():
    steps = 200
    x_sol = np.zeros((4, steps, 2), dtype=float)
    x_sol[1, :, 0] = np.linspace(0.0, 1.0, steps)
    x_sol[2, :, 0] = np.linspace(1.0, 2.0, steps)
    x_sol[3, :, 0] = np.linspace(2.0, 3.0, steps)
    semantic_keypoints = {"hip": 0, "knee": 1, "ankle": 2, "foot": 3}

    _, failures = analyze_semantics(
        sample_id=0,
        x_sol=x_sol,
        semantic_keypoints=semantic_keypoints,
        rom_x_min=0.1,
        rom_y_min=0.0,
        knee_amp_min=0.0,
        ankle_amp_min=0.0,
        trajectory_complexity_min=0.05,
    )

    assert "low_trajectory_complexity" in failures


def test_dataset_report_surfaces_step_failure_stats(tmp_path):
    spec = GenerationSpec(seed=7, pilot_only=True, output_dir=tmp_path / "pilot")
    spec.families = {"6bar": replace(spec.families["6bar"], pilot_target=1)}
    result = generate_dataset_v2(spec)
    sample = result["final_samples"][0]

    record = {
        "attempts": 3,
        "accepted_raw": 1,
        "accepted_final": 1,
        "failure_reasons": Counter({"self_intersection": 2}),
        "failure_reasons_by_stage": {"step_1": Counter({"self_intersection": 2})},
        "template_attempts": Counter({sample["base_template"]: 3}),
        "template_failure_reasons": {sample["base_template"]: Counter({"self_intersection": 2})},
        "template_final_counts": Counter({sample["base_template"]: 1}),
    }
    report = build_dataset_report(
        spec=spec,
        pilot_samples=[sample],
        pilot_records={"6bar": record},
        final_records={"6bar": record},
        final_samples=[sample],
        splits={"train": [0], "val": [], "test": []},
        leakage={"train_val_overlap": 0, "train_test_overlap": 0, "val_test_overlap": 0},
    )

    assert report["family_stats"]["6bar"]["step_failure_reasons"]["step_1"]["self_intersection"] == 2
