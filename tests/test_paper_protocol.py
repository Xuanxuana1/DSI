from pathlib import Path

import yaml


def test_interface_width_protocol_selects_by_validation_mae() -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs" / "experiments" / "paper.yaml"
    paper = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert paper["controlled_decision"]["train_rows"] == 1_000_000
    assert paper["controlled_decision"]["validation_select"] == "mae"
