import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.action_codebook import load_action_codebook
from src.inverse.family_index_builder import build_family_index_artifacts
from src.inverse.phase4_il import ensure_multistep_expert_paths, load_step_split


def main():
    parser = argparse.ArgumentParser(description="Build LINKS4Meta family-aware IL step index artifacts")
    parser.add_argument("--config", type=str, default="src/train_links4meta_il.yaml")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--export-jsonl", action="store_true")
    args = parser.parse_args()

    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    paths_cfg = cfg["paths"]
    resolve_mapping_paths(
        paths_cfg,
        (
            "pkl_dataset",
            "precomputed_split_input",
            "il_multistep_dataset_output",
            "il_split_output",
            "family_index_output_dir",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )

    ensure_parent_dir(Path(paths_cfg["family_index_output_dir"]) / "family_index_manifest_v1.json")
    multistep_dataset_path = paths_cfg.get("il_multistep_dataset_output", paths_cfg["il_dataset_output"])
    step_paths = ensure_multistep_expert_paths(
        pkl_path=paths_cfg["pkl_dataset"],
        output_path=multistep_dataset_path,
        use_cached=args.skip_extract,
    )
    split = load_step_split(
        step_paths,
        split_path=paths_cfg["il_split_output"],
        precomputed_split_path=paths_cfg.get("precomputed_split_input"),
        val_ratio=cfg["il_training"].get("val_ratio", 0.1),
        test_ratio=cfg["il_training"].get("test_ratio", 0.1),
        split_seed=cfg["il_training"].get("split_seed", 42),
    )
    codebook = load_action_codebook(multistep_dataset_path)
    manifest = build_family_index_artifacts(
        step_paths,
        split=split,
        codebook=codebook,
        output_dir=paths_cfg["family_index_output_dir"],
        dataset_path=multistep_dataset_path,
        export_jsonl=args.export_jsonl,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
