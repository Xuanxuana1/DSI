from __future__ import annotations

from pathlib import Path
from typing import Any


def package_root() -> Path:

    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:

    value = Path(path).expanduser()
    if value.is_absolute():
        return value

    parts = value.parts
    root = package_root()
    if parts and parts[0] == root.name:
        value = Path(*parts[1:]) if len(parts) > 1 else Path(".")

    return root / value


def portable_path(path: str | Path, root: str | Path | None = None) -> str:
    """
    Convert an absolute path into a repository-portable serialized value.

    Paths inside the repository are stored relative to its root. External
    absolute paths are redacted because their parent directories may disclose
    machine or author identity.
    """
    value = Path(path).expanduser()
    if not value.is_absolute():
        return value.as_posix()

    repository_root = package_root() if root is None else Path(root).expanduser().resolve()
    try:
        return value.resolve(strict=False).relative_to(repository_root).as_posix()
    except ValueError:
        return "<external>"


def portableize_serialized_paths(value: Any, root: str | Path | None = None) -> Any:
    """Recursively redact or relativize absolute paths before serialization."""
    if isinstance(value, dict):
        return {
            key: portableize_serialized_paths(item, root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portableize_serialized_paths(item, root) for item in value]
    if isinstance(value, tuple):
        return tuple(portableize_serialized_paths(item, root) for item in value)
    if isinstance(value, Path):
        return portable_path(value, root)
    if isinstance(value, str) and Path(value).expanduser().is_absolute():
        return portable_path(value, root)
    return value
