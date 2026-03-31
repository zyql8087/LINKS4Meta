from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_gen_v2.dataset_builder_v2 import generate_dataset_v2
from data_gen_v2.specs import GenerationSpec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LINKS multibase v2 generator.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--pilot-per-family", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--recommendation-report", type=str, default=None)
    parser.add_argument("--use-t5c-tight-mode", action="store_true")
    parser.add_argument("--t5c-template-target", type=int, default=0)
    parser.add_argument("--t5c-template-attempt-multiplier", type=float, default=1.0)
    return parser


def _apply_report_recommendations(spec: GenerationSpec, report_path: str) -> None:
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    recommendations = report.get("recommendations", {})
    family_targets = recommendations.get("family_targets", {}).get("recommended_final_targets")
    if family_targets:
        spec.final_targets_override = {key: int(value) for key, value in family_targets.items()}
    dedup = recommendations.get("dedup_thresholds", {})
    if "recommended_geometry_radius" in dedup:
        spec.thresholds.geometry_radius = float(dedup["recommended_geometry_radius"])
    if "recommended_motion_radius" in dedup:
        spec.thresholds.motion_radius = float(dedup["recommended_motion_radius"])


def main() -> None:
    args = build_arg_parser().parse_args()
    default_output_dir = GenerationSpec().output_dir
    spec = GenerationSpec(
        seed=args.seed,
        pilot_only=args.pilot_only,
        pilot_per_family=args.pilot_per_family,
        output_dir=Path(args.output_dir) if args.output_dir else default_output_dir,
        use_t5c_tight_mode=args.use_t5c_tight_mode,
    )
    if args.t5c_template_target > 0:
        spec.template_final_target_overrides.setdefault("9bar", {})["T5-C"] = int(
            args.t5c_template_target
        )
    if args.t5c_template_attempt_multiplier > 1.0:
        spec.template_attempt_multiplier_overrides.setdefault("9bar", {})["T5-C"] = float(
            args.t5c_template_attempt_multiplier
        )
    if args.recommendation_report:
        _apply_report_recommendations(spec, args.recommendation_report)
    result = generate_dataset_v2(spec)
    print("Generation completed.")
    print(f"Final samples: {len(result['final_samples'])}")
    print(f"Report: {spec.report_file}")
    print(f"Dataset: {spec.final_dataset_file}")


if __name__ == "__main__":
    main()
