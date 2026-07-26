from tests.smoke import run_smoke


def test_public_pipeline_smoke() -> None:
    result = run_smoke()
    assert result["readout"]["mae"] >= 0.0
    assert result["baseline"]["mae"] >= 0.0
