"""Behavior tests for the benchmark's input, network and watchdog paths."""

import glob
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills/inference-optimisation/scripts/bench.py"
)
SPEC = importlib.util.spec_from_file_location("bench", SCRIPT)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_health_handles_network_and_bad_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = bench.Server({})
    monkeypatch.setattr(server, "get", lambda path: {"data": ["model"]})
    assert server.healthy()

    def fail(path: str) -> None:
        raise OSError("offline")

    monkeypatch.setattr(server, "get", fail)
    assert not server.healthy()
    monkeypatch.setattr(server, "get", lambda path: [])
    assert not server.healthy()


def test_power_probe_reads_supply_files_and_handles_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supply = tmp_path / "online"
    supply.write_text("0\n")
    monkeypatch.setattr(bench.sys, "platform", "linux")
    monkeypatch.setattr(glob, "glob", lambda pattern: [str(supply)])
    assert not bench.on_ac()
    supply.write_text("1\n")
    assert bench.on_ac()
    supply.write_bytes(b"\xff")
    assert bench.on_ac()
    supply.unlink()
    assert bench.on_ac()


def test_watchdog_logs_probe_failure_and_trips_on_memory_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    values = iter([ValueError("bad probe"), "2048"])

    def probe(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        value = next(values)
        if isinstance(value, ValueError):
            raise value
        return subprocess.CompletedProcess([], 0, stdout=value, stderr="")

    trips: list[bool] = []
    watch = bench.MemWatch(
        {"probe": ["fake"], "cap_gib": 1024 / 2**30}, lambda: trips.append(True)
    )
    waits = iter([False, False, True])
    monkeypatch.setattr(watch.stop, "wait", lambda interval: next(waits))
    monkeypatch.setattr(bench.subprocess, "run", probe)
    watch.run()
    assert watch.tripped and watch.peak == 2048 and trips == [True]
    assert "bad probe" in capsys.readouterr().out


def test_run_records_failed_request_and_show_reads_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"name": "case", "model": "model"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bench.Server, "healthy", lambda self: True)
    monkeypatch.setattr(bench, "on_ac", lambda: True)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("request failed")

    monkeypatch.setattr(bench.Server, "chat", fail)
    result = bench.run(path)
    assert not result["ok"] and result["flags"] == ["REQ_FAIL"]
    bench.show()
    assert "case" in capsys.readouterr().out
    assert json.loads((tmp_path / "ledger.jsonl").read_text())["name"] == "case"
