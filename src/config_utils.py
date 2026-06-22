"""
配置文件加载与路径解析工具模块。

本模块提供 YAML 配置文件的加载、路径解析、批量路径映射等功能，
是整个项目所有脚本加载配置的基础设施。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


def load_yaml_config(config_arg: str | Path, *search_roots: Path) -> tuple[dict, Path]:
    """
    加载 YAML 配置文件，并返回解析后的字典和配置文件的绝对路径。

    查找策略：
    1. 如果 config_arg 是绝对路径，直接加载。
    2. 如果是相对路径，依次在以下目录中查找：
       - 当前工作目录 (Path.cwd())
       - search_roots 中传入的额外搜索根目录（按顺序）
    3. 若所有目录都找不到，则默认使用第一个 search_root 拼接路径。

    参数:
        config_arg: 配置文件路径（绝对或相对路径均可）
        *search_roots: 额外的搜索根目录，用于解析相对路径

    返回:
        (config_dict, config_path):
            - config_dict: 从 YAML 解析得到的配置字典
            - config_path: 配置文件的绝对路径（Path 对象）
    """
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        # 相对路径：依次在当前目录和搜索根目录中查找
        for root in (Path.cwd(), *search_roots):
            candidate = (root / config_path).resolve()
            if candidate.exists():
                config_path = candidate
                break
        else:
            # 所有目录都找不到时，默认使用第一个 search_root
            config_path = (search_roots[0] / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, config_path


def resolve_path(path_value: str | Path, config_dir: Path, workspace_root: Path) -> Path:
    """
    解析一个路径值，将其转为绝对路径。

    解析策略（按优先级）：
    1. 如果 path_value 已是绝对路径，直接返回。
    2. 相对路径时，依次尝试以下基准目录：
       - config_dir: 配置文件所在目录（最常见，配置中路径通常相对于配置文件）
       - workspace_root: 工作区根目录（项目级路径）
    3. 若两个候选路径都不存在，返回第一个（config_dir 优先）。

    参数:
        path_value:    待解析的路径字符串或 Path 对象
        config_dir:    配置文件所在的目录，用于解析相对路径
        workspace_root: 工作区根目录，作为备选基准

    返回:
        解析后的绝对路径（Path 对象）
    """
    path = Path(path_value)
    if path.is_absolute():
        return path

    # 按优先级排列候选基准目录：配置文件目录 > 工作区根目录
    candidates = [
        (config_dir / path).resolve(),
        (workspace_root / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # 都不存在时，返回第一个候选（config_dir 优先）
    return candidates[0]


def resolve_mapping_paths(
    mapping: dict[str, object],
    keys: Iterable[str],
    *,
    config_dir: Path,
    workspace_root: Path,
) -> None:
    """
    批量解析字典中指定键对应的路径值，将相对路径转为绝对路径（原地修改）。

    典型用法：加载 YAML 配置后，将配置字典中的文件路径字段统一转为绝对路径，
    便于后续代码直接使用。

    参数:
        mapping:       配置字典（会被原地修改）
        keys:          需要解析路径的键名列表
        config_dir:    配置文件所在目录
        workspace_root: 工作区根目录

    示例:
        config, cfg_path = load_yaml_config("config_inverse.yaml")
        resolve_mapping_paths(
            config,
            ["dataset_path", "model_path"],
            config_dir=cfg_path.parent,
            workspace_root=Path.cwd(),
        )
        # 此后 config["dataset_path"] 和 config["model_path"] 均为绝对路径
    """
    for key in keys:
        value = mapping.get(key)
        if not value:
            continue
        mapping[key] = str(resolve_path(value, config_dir, workspace_root))


def ensure_parent_dir(path_value: str | Path) -> Path:
    """
    确保路径的父目录存在，若不存在则递归创建。

    常用于输出文件写入前，确保目标目录已就绪。

    参数:
        path_value: 目标文件路径

    返回:
        原始路径的 Path 对象（未做修改）
    """
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
