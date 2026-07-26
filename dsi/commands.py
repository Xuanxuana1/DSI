from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

from dsi import DSI_MODEL
from dsi.utils.bridge import resolve_path


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:

    from dsi.utils.config import load_config as package_load_config

    resolved = resolve_path(config_path).resolve()
    config = package_load_config(resolved)
    return normalize_config_paths(config, resolved), resolved


def normalize_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:

    dsi_root = Path(__file__).resolve().parents[1]

    for key in ["contract_path"]:
        if key in config:
            config[key] = str(resolve_path(config[key]).resolve())

    for key in ["raw_dir", "zip_path", "processed_dir"]:
        if key in config.get("data", {}):
            config["data"][key] = str((dsi_root / str(config["data"][key])).expanduser().resolve())

    for key in ["captions_path", "categories_path", "caption_category_path"]:
        if key in config.get("content_features", {}):
            config["content_features"][key] = str((dsi_root / str(config["content_features"][key])).expanduser().resolve())

    for key in ["root_dir", "prepare_log", "train_log"]:
        if key in config.get("outputs", {}):
            config["outputs"][key] = str((dsi_root / str(config["outputs"][key])).expanduser().resolve())

    return config


def apply_training_overrides(
    config: dict[str, Any],
    seed: int | None,
    device: str | None,
    output_root: str | None,
    train_log: str | None,
    epochs: int | None,
    hidden_dim: int | None,
) -> dict[str, Any]:

    config.setdefault("training", {})
    config.setdefault("model", {})
    config.setdefault("outputs", {})

    if seed is not None:
        config["training"]["seed"] = int(seed)
    if device is not None:
        config["training"]["device"] = str(device)
    if output_root is not None:
        config["outputs"]["root_dir"] = str(resolve_path(output_root))
    if train_log is not None:
        config["outputs"]["train_log"] = str(resolve_path(train_log))
    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    if hidden_dim is not None:
        config["model"]["hidden_dim"] = int(hidden_dim)

    return config


def config_slug(config_path: Path) -> str:

    return config_path.stem


def report_path(config_path: Path, kind: str, seed: int, output: str | None) -> Path:

    if output is not None:
        return resolve_path(output)

    root = Path(__file__).resolve().parents[1]
    return root / "artifacts" / "reports" / f"{config_slug(config_path)}_{kind}_seed{int(seed)}.json"


def log_path(config_path: Path, kind: str, seed: int, log_path_value: str | None) -> Path:

    if log_path_value is not None:
        return resolve_path(log_path_value)

    root = Path(__file__).resolve().parents[1]
    return root / "artifacts" / "logs" / f"{config_slug(config_path)}_{kind}_seed{int(seed)}.log"


def dsi_feature_source(config: dict[str, Any], config_path: Path) -> str:

    checkpoint_root = Path(config["outputs"]["root_dir"]).expanduser().resolve()

    return f"dsi:{DSI_MODEL}:{config_path}:{checkpoint_root}"


def module_argv(module_name: str, args: list[str]) -> list[str]:

    return [module_name, *args]


def append_many(command: list[str], option: str, values: Iterable[Any]) -> None:

    normalized = [str(value) for value in values]
    if not normalized:
        return

    command.append(option)
    command.extend(normalized)


def run_module_main(main_func, argv: list[str]) -> None:

    old_argv = sys.argv
    try:
        sys.argv = argv
        main_func()
    finally:
        sys.argv = old_argv
