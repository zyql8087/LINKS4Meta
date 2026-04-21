from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


def load_yaml_config(config_arg: str | Path, *search_roots: Path) -> tuple[dict, Path]:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        for root in (Path.cwd(), *search_roots):
            candidate = (root / config_path).resolve()
            if candidate.exists():
                config_path = candidate
                break
        else:
            config_path = (search_roots[0] / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, config_path


def resolve_path(path_value: str | Path, config_dir: Path, workspace_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates = [
        (config_dir / path).resolve(),
        (workspace_root / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_mapping_paths(
    mapping: dict[str, object],
    keys: Iterable[str],
    *,
    config_dir: Path,
    workspace_root: Path,
) -> None:
    for key in keys:
        value = mapping.get(key)
        if not value:
            continue
        mapping[key] = str(resolve_path(value, config_dir, workspace_root))


def ensure_parent_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
