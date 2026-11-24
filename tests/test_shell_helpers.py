"""Exercise the queue and memory helpers without a GPU or Docker daemon."""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/inference-optimisation/scripts"


@pytest.mark.parametrize("flags", ["--long 123 --skip-cN", "", "--literal *"])
def test_queue_passes_extra_flags_as_separate_arguments(
    tmp_path: Path, flags: str
) -> None:
    bench = tmp_path / "bench.py"
    bench.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    env = os.environ | {
        "BENCH": str(bench),
        "BENCH_FLAGS": flags,
        "QUEUE_LOCK": str(tmp_path / "lock"),
    }
    result = subprocess.run(
        ["bash", str(SCRIPTS / "queue.sh"), "first", "second"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "QUEUE_DONE first second" in result.stdout
    assert json.loads((tmp_path / "runs-first.log").read_text()) == [
        "run",
        "configs/first.json",
        *flags.split(),
    ]
    assert (tmp_path / "runs-second.log").is_file()


@pytest.mark.parametrize("group", ["none", "docker", "system", "both"])
def test_mem_sums_gpu_and_cgroup_memory(tmp_path: Path, group: str) -> None:
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/sh\nif [ "$1" = top ]; then printf "PID\\n123\\n"; else echo nonexistent; fi\n'
    )
    nvidia = tmp_path / "nvidia-smi"
    nvidia.write_text('#!/bin/sh\necho "123, 50"\n')
    docker.chmod(0o755)
    nvidia.chmod(0o755)
    cgroups = tmp_path / "cgroups"
    if group in ("docker", "both"):
        path = cgroups / "docker/nonexistent"
        path.mkdir(parents=True)
        (path / "memory.stat").write_text("anon 42\n")
    if group in ("system", "both"):
        path = cgroups / "system.slice/docker-nonexistent.scope"
        path.mkdir(parents=True)
        (path / "memory.stat").write_text("anon 84\n")
    script = tmp_path / "mem.sh"
    script.write_text(
        (SCRIPTS / "mem.sh").read_text().replace("/sys/fs/cgroup", str(cgroups))
    )
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(script), "fake"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    anon = {"none": 0, "docker": 42, "system": 84, "both": 42}[group]
    assert result.stdout.strip() == str(50 * 1048576 + anon)
