from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from dsi.utils.bridge import portableize_serialized_paths


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def dump_config(
    config: dict[str, Any],
    path: str | Path,
    *,
    portable_paths: bool = True,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = portableize_serialized_paths(config) if portable_paths else config
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            serialized,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
