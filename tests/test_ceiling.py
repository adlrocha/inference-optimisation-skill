"""Checks for the hardware ceiling calculator."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills/inference-optimisation/scripts/ceiling.py"
)
SPEC = importlib.util.spec_from_file_location("ceiling", SCRIPT)
assert SPEC and SPEC.loader
ceiling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ceiling)


def test_main_reads_model_and_prints_ceilings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = {
        "bandwidth_gbs": 100,
        "peak_tflops": 20,
        "mfu": 0.5,
        "dense_gb": 1,
        "lm_head_gb": 1,
        "active_params_b": 5,
        "contexts": [4096],
        "kv_bytes_per_token": 1024,
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    ceiling.main(path)
    output = capsys.readouterr().out
    assert "bandwidth 100 GB/s" in output
    assert "Decode ceiling, concurrency 1" in output
    assert "Prefill ceiling" in output
    assert "KV:" in output


def test_main_propagates_missing_and_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    with pytest.raises(FileNotFoundError):
        ceiling.main(path)
    path.write_text("invalid json")
    with pytest.raises(json.JSONDecodeError):
        ceiling.main(path)
