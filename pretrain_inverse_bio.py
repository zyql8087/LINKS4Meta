import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config_utils import ensure_parent_dir, load_yaml_config, resolve_mapping_paths
from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy
from src.inverse.pretrain_links import ensure_links_pretrain_cache, run_links_pretraining


def main():
    parser = argparse.ArgumentParser(description="LINKS pretraining for inverse encoder and validity head")
    parser.add_argument("--config", type=str, default="src/pretrain_links.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cfg, config_path = load_yaml_config(args.config, SCRIPT_DIR, WORKSPACE_ROOT)
    paths_cfg = cfg["paths"]
    resolve_mapping_paths(
        paths_cfg,
        (
            "links_pretrain_input",
            "links_pretrain_cache_output",
            "links_pretrain_model_output",
            "links_pretrain_report_output",
            "precomputed_split_input",
        ),
        config_dir=config_path.parent,
        workspace_root=WORKSPACE_ROOT,
    )
    for key in (
        "links_pretrain_cache_output",
        "links_pretrain_model_output",
        "links_pretrain_report_output",
    ):
        ensure_parent_dir(paths_cfg[key])

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[LINKS Pretrain] device={device}")

    curve_encoder = CurveEncoder(
        input_dim=cfg["curve_encoder"]["input_dim"],
        hidden_dims=cfg["curve_encoder"]["hidden_dims"],
        latent_dim=cfg["curve_encoder"]["latent_dim"],
    ).to(device)
    policy = GNNPolicy(cfg).to(device)

    pretrain_cfg = cfg.get("links_pretrain", {})
    cache = ensure_links_pretrain_cache(
        dataset_path=paths_cfg["links_pretrain_input"],
        cache_path=paths_cfg["links_pretrain_cache_output"],
        split_path=paths_cfg.get("precomputed_split_input"),
        max_samples=int(pretrain_cfg.get("max_samples", 0)),
        seed=int(pretrain_cfg.get("seed", 42)),
        constraint_cfg=cfg.get("constraints", {}),
        use_cached=not args.rebuild_cache,
    )
    report = run_links_pretraining(
        policy=policy,
        curve_encoder=curve_encoder,
        cache=cache,
        cfg=cfg,
        device=device,
        output_model_path=paths_cfg["links_pretrain_model_output"],
        output_report_path=paths_cfg["links_pretrain_report_output"],
    )
    print(f"[LINKS Pretrain] checkpoint: {paths_cfg['links_pretrain_model_output']}")
    print(f"[LINKS Pretrain] report: {paths_cfg['links_pretrain_report_output']}")
    print(f"[LINKS Pretrain] summary: {report}")


if __name__ == "__main__":
    main()
