from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.inverse.readout_assignment import (  # noqa: E402
        LearnedChainScorerReadoutAssignment,
        RuleBasedReadoutAssignment,
        SlotPointerReadoutAssignment,
        build_synthetic_readout_records,
        evaluate_assignment_module,
    )
except Exception as exc:  # pragma: no cover - runtime environment hint
    print(f"[ERROR] Failed to import readout assignment demo dependencies: {exc}")
    print("[HINT] Run this demo in the GMM environment.")
    print("       Example: run_gmm.cmd GraphMetaMat-LINKS\\code\\run_readout_assignment_demo.py")
    raise


def main() -> None:
    train_records = build_synthetic_readout_records(num_records=64, seed=17)
    test_records = build_synthetic_readout_records(num_records=24, seed=117)

    scheme_a = RuleBasedReadoutAssignment()
    scheme_b = LearnedChainScorerReadoutAssignment()
    scheme_c = SlotPointerReadoutAssignment()

    train_summary_b = scheme_b.fit(train_records, epochs=60, lr=8e-3, seed=23)
    train_summary_c = scheme_c.fit(train_records, epochs=80, lr=8e-3, seed=23)

    summary = {
        "dataset": {
            "train_records": len(train_records),
            "test_records": len(test_records),
        },
        "train": {
            "scheme_b": train_summary_b,
            "scheme_c": train_summary_c,
        },
        "eval": {
            "scheme_a": evaluate_assignment_module(scheme_a, test_records),
            "scheme_b": evaluate_assignment_module(scheme_b, test_records),
            "scheme_c": evaluate_assignment_module(scheme_c, test_records),
        },
    }

    output_dir = WORKSPACE_ROOT / "demo" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "readout_assignment_demo_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
