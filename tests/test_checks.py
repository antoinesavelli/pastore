"""Pre-start host checks against scripted Host doubles; no real nvidia-smi, dpkg, /proc or Ollama.

Depends on: fucone.checks, tests.fakes (RecordingRunner).
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from fucone.checks import (
    GPU_ALLOWLIST,
    MEMINFO,
    MIN_AVAILABLE_BYTES,
    MODULE_VERSION_FILE,
    CheckError,
    Host,
    driver_ok,
    foreign_gpu_process,
    host_idle_reason,
    mem_available_bytes,
    ollama_ok,
    run_argv,
    stopped,
)
from tests.fakes import RecordingRunner

PKG_VERSION = "580.178.04-0ubuntu0.24.04.1"
GOOD_APPS = "1234, /usr/bin/cosmic-monitor\n5678, /usr/local/lib/ollama/llama-server\n"
MEM_OK = f"MemTotal: 65000000 kB\nMemAvailable: {MIN_AVAILABLE_BYTES // 1024} kB\n"
MEM_LOW = f"MemTotal: 65000000 kB\nMemAvailable: {MIN_AVAILABLE_BYTES // 1024 - 1} kB\n"


class Env:
    def __init__(
        self,
        module: str = "580.178.04\n",
        package: str = PKG_VERSION,
        smi_code: int = 0,
        apps: str = GOOD_APPS,
        meminfo: str = MEM_OK,
        http: int | OSError = 200,
    ) -> None:
        self.http_calls: list[tuple[str, float]] = []
        self._http = http
        self.runner = RecordingRunner(
            {
                "dpkg-query": RecordingRunner.ok(package),
                "nvidia-smi": RecordingRunner.ok(apps, smi_code),
            }
        )
        files = {MODULE_VERSION_FILE: module, MEMINFO: meminfo}
        self.host = Host(run=self.runner, read_text=files.__getitem__, http_status=self._status)

    def _status(self, url: str, timeout_s: float) -> int:
        self.http_calls.append((url, timeout_s))
        if isinstance(self._http, OSError):
            raise self._http
        return self._http


def test_invariant_driver_mismatch_refuses_dispatch() -> None:
    stale = Env(module="580.178.04\n", package="580.159.03-0ubuntu0.24.04.1")
    assert host_idle_reason(stale.host) == "driver-mismatch"

    smi_down = Env(smi_code=1)
    assert host_idle_reason(smi_down.host) == "driver-mismatch"

    assert host_idle_reason(Env().host) is None


def test_driver_mismatch_short_circuits_every_later_check() -> None:
    env = Env(module="580.159.03\n", http=OSError("down"), meminfo=MEM_LOW)
    assert host_idle_reason(env.host) == "driver-mismatch"
    assert env.http_calls == []
    assert [c for c in env.runner.calls if c[0] == "nvidia-smi"] == []


def test_driver_ok_fails_closed_on_empty_output() -> None:
    assert not driver_ok(Env(package="").host)
    assert not driver_ok(Env(module="\n").host)
    assert driver_ok(Env(package="580.178.04").host)


def test_driver_ok_false_on_dpkg_failure_and_oserror() -> None:
    env = Env()
    env.runner.results["dpkg-query"] = RecordingRunner.ok("", 1)
    assert not driver_ok(env.host)

    def missing(argv: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi")

    broken = Host(run=missing, read_text=env.host.read_text, http_status=env.host.http_status)
    assert not driver_ok(broken)


def test_cosmic_monitor_and_llama_server_are_not_foreign() -> None:
    assert foreign_gpu_process(Env().host) is None
    assert GPU_ALLOWLIST >= {"/usr/bin/cosmic-monitor", "/usr/local/lib/ollama/llama-server"}


def test_foreign_gpu_process_is_an_idle_reason() -> None:
    apps = GOOD_APPS + "999, /some/path\n"
    assert host_idle_reason(Env(apps=apps).host) == "gpu-foreign:/some/path"


def test_foreign_gpu_query_failure_and_garbage_raise() -> None:
    with pytest.raises(CheckError):
        foreign_gpu_process(Env(smi_code=9).host)
    with pytest.raises(CheckError):
        foreign_gpu_process(Env(apps="no comma here\n").host)


def test_ollama_down_on_refusal_and_on_non_200() -> None:
    assert host_idle_reason(Env(http=OSError("refused")).host) == "ollama-down"
    assert host_idle_reason(Env(http=503).host) == "ollama-down"
    env = Env()
    assert ollama_ok(env.host, "http://x:1")
    assert env.http_calls == [("http://x:1/api/tags", 5.0)]


def test_memory_floor_is_exactly_15_gib() -> None:
    assert mem_available_bytes(Env().host) == MIN_AVAILABLE_BYTES
    assert host_idle_reason(Env(meminfo=MEM_LOW).host) == "low-memory"
    with pytest.raises(CheckError):
        mem_available_bytes(Env(meminfo="MemTotal: 1 kB\n").host)
    with pytest.raises(CheckError):
        mem_available_bytes(Env(meminfo="MemAvailable: lots kB\n").host)


def test_first_failing_check_wins_in_r12_order() -> None:
    env = Env(http=OSError("down"), apps="1, /some/path\n", meminfo=MEM_LOW)
    assert host_idle_reason(env.host) == "ollama-down"
    env = Env(apps="1, /some/path\n", meminfo=MEM_LOW)
    assert host_idle_reason(env.host) == "gpu-foreign:/some/path"


def test_stopped_is_file_existence(tmp_path: Path) -> None:
    stop_file = tmp_path / "STOP"
    assert not stopped(stop_file)
    stop_file.touch()
    assert stopped(stop_file)


def test_multiarch_package_lines_must_all_match_the_module() -> None:
    both = f"{PKG_VERSION}\n{PKG_VERSION}\n"
    assert host_idle_reason(Env(package=both).host) is None
    split = f"{PKG_VERSION}\n580.159.03-0ubuntu0.24.04.1\n"
    assert host_idle_reason(Env(package=split).host) == "driver-mismatch"


def test_unexpected_host_output_fails_closed_instead_of_raising() -> None:
    env = Env()

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if any(arg.startswith("--query-compute-apps") for arg in argv):
            return subprocess.CompletedProcess([], 9, "", "")
        return env.runner(argv)

    broken_smi = Host(run=run, read_text=env.host.read_text, http_status=env.host.http_status)
    assert host_idle_reason(broken_smi) == "driver-mismatch"
    assert host_idle_reason(Env(meminfo="garbage\n").host) == "low-memory"


def test_run_argv_turns_a_missing_binary_into_exit_127(tmp_path: Path) -> None:
    assert run_argv([str(tmp_path / "missing")]).returncode == 127
