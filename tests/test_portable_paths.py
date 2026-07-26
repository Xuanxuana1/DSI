from pathlib import Path

import yaml

from dsi.utils.bridge import (
    package_root,
    portable_path,
    portableize_serialized_paths,
    resolve_path,
)
from dsi.utils.config import dump_config


def test_portable_path_relativizes_repository_paths(tmp_path: Path) -> None:
    repository = tmp_path / "anonymous-repository"
    artifact = repository / "artifacts" / "reports" / "result.json"

    assert portable_path(artifact, repository) == "artifacts/reports/result.json"


def test_portable_path_redacts_external_paths(tmp_path: Path) -> None:
    repository = tmp_path / "anonymous-repository"
    external = tmp_path / "private-user" / "dataset"

    assert portable_path(external, repository) == "<external>"


def test_portable_repository_path_can_be_resolved_after_serialization() -> None:
    artifact = package_root() / "artifacts" / "cache" / "features.npz"
    serialized = portable_path(artifact)

    assert resolve_path(serialized) == artifact


def test_portableize_serialized_paths_is_recursive(tmp_path: Path) -> None:
    repository = tmp_path / "anonymous-repository"
    payload = {
        "processed_dir": repository / "artifacts" / "processed",
        "nested": [str(repository / "configs" / "paper.yaml")],
        "external": str(tmp_path / "private-user" / "raw"),
        "label": "validation_mae",
    }

    serialized = portableize_serialized_paths(payload, repository)

    assert serialized == {
        "processed_dir": "artifacts/processed",
        "nested": ["configs/paper.yaml"],
        "external": "<external>",
        "label": "validation_mae",
    }


def test_dump_config_does_not_write_absolute_paths(tmp_path: Path) -> None:
    output = tmp_path / "resolved_config.yaml"
    config = {
        "_config_path": str(package_root() / "configs" / "datasets" / "kuairec.yaml"),
        "data": {
            "processed_dir": str(package_root() / "artifacts" / "processed" / "kuairec"),
            "raw_dir": str(tmp_path / "private-user" / "raw"),
        },
    }

    dump_config(config, output)
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert loaded["_config_path"] == "configs/datasets/kuairec.yaml"
    assert loaded["data"]["processed_dir"] == "artifacts/processed/kuairec"
    assert loaded["data"]["raw_dir"] == "<external>"
    assert str(package_root()) not in output.read_text(encoding="utf-8")
