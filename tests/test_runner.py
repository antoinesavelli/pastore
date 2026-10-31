"""Run directories, job environment, scope argv, stop and leftover-scope recovery with doubles; no systemd.

Depends on: pastore.runner, pastore.spec, tests.fakes (RecordingRunner).
"""

import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from pastore.runner import (
    RunHandle,
    has_result,
    job_env,
    leftover_scopes,
    make_run_dir,
    new_run_id,
    scope_argv,
    start_run,
    stop_leftovers,
)
from pastore.spec import RunReader, TaskSpec
from tests.fakes import RecordingRunner

REPO = Path(__file__).resolve().parents[1]


def make_spec(tmp_path: Path, env: dict[str, str] | None = None) -> TaskSpec:
    return TaskSpec(
        consumer="torra",
        id="bench",
        kind="eval",
        argv=("python3", "bench.py", "--all"),
        cwd=tmp_path,
        env=env or {},
        models=(),
        timeout_s=60,
        ok_exit=frozenset({0}),
        ollama_env="OLLAMA_HOST",
        model_env="PASTORE_MODEL",
        reader=RunReader(argv=("true",)),
        ready=None,
        root=tmp_path,
        fingerprint="f" * 64,
    )


def test_invariant_job_env_has_run_dir_and_no_leaked_vars(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path / "runs", "r1")
    base = {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "LANG": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "ANTHROPIC_API_KEY": "secret",
        "PYTHONPATH": "/elsewhere",
    }
    env = job_env(make_spec(tmp_path, {"MODE": "x"}), run_dir, "http://127.0.0.1:1", "m1", base)
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "LANG": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "MODE": "x",
        "PASTORE_RUN_DIR": str(run_dir.out),
        "OLLAMA_HOST": "http://127.0.0.1:1",
        "PASTORE_MODEL": "m1",
    }


def test_job_env_passes_the_model_in_model_env_and_the_manifest_cannot_override_it(
    tmp_path: Path,
) -> None:
    run_dir = make_run_dir(tmp_path / "runs", "r1")
    spec = replace(make_spec(tmp_path, {"PASTORE_MODEL": "evil", "OTHER": "o"}), model_env="WHICH")
    env = job_env(spec, run_dir, "http://u", "qwen3:8b", {})
    assert env["WHICH"] == "qwen3:8b" and env["OTHER"] == "o"
    named = job_env(make_spec(tmp_path, {"PASTORE_MODEL": "evil"}), run_dir, "http://u", "m1", {})
    assert named["PASTORE_MODEL"] == "m1"


def test_run_dir_is_outside_repo_and_never_reused(tmp_path: Path) -> None:
    runs = tmp_path / "share" / "runs"
    run_dir = make_run_dir(runs, "20261024T010000-1")
    assert run_dir.out == run_dir.root / "out" and run_dir.out.is_dir()
    assert run_dir.stdout.parent == run_dir.root and run_dir.stderr.parent == run_dir.root
    assert not run_dir.root.is_relative_to(REPO)
    with pytest.raises(FileExistsError):
        make_run_dir(runs, "20261024T010000-1")


def test_new_run_id_is_utc() -> None:
    local = datetime(2026, 10, 24, 1, 5, 9, tzinfo=timezone(timedelta(hours=-4)))
    assert new_run_id(local, 7) == "20261024T050509-7"
    assert new_run_id(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC), 12) == "20260102T030405-12"


def test_scope_argv_is_user_scope_in_heavy_jobs_slice() -> None:
    argv = scope_argv("r1", ["python3", "x.py"], 7200)
    assert argv == [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--slice=heavy-jobs.slice",
        "--unit=pastore-r1",
        "-p",
        "CPUWeight=20",
        "-p",
        "RuntimeMaxSec=7200",
        "--",
        "python3",
        "x.py",
    ]
    assert "sudo" not in argv and "--user" in argv


class FakeProc:
    def __init__(self, hangs: bool = False) -> None:
        self.hangs = hangs
        self.killed = False

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        if self.hangs and not self.killed:
            raise subprocess.TimeoutExpired("job", timeout or 0)
        return -9 if self.killed else 0

    def kill(self) -> None:
        self.killed = True


def test_stop_stops_the_scope_then_waits_and_kills_only_on_timeout() -> None:
    run = RecordingRunner()
    clean = FakeProc()
    assert RunHandle(clean, "r1").stop(run) == 0 and not clean.killed
    assert run.calls == [["systemctl", "--user", "stop", "pastore-r1.scope"]]

    stuck = FakeProc(hangs=True)
    assert RunHandle(stuck, "r2").stop(run, wait_s=0.01) == -9 and stuck.killed


def test_start_run_passes_wrapped_argv_explicit_env_and_new_session(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    run_dir = make_run_dir(tmp_path / "runs", "r1")
    seen: dict[str, Any] = {}

    def popen(argv: list[str], **kwargs: Any) -> FakeProc:
        seen["argv"] = argv
        seen.update(kwargs)
        return FakeProc()

    handle = start_run(
        spec, run_dir, "r1", "http://u", "m1", {"PATH": "/bin", "SECRET": "s"}, popen=popen
    )
    assert handle.run_id == "r1"
    assert seen["argv"] == scope_argv("r1", spec.argv, spec.timeout_s)
    assert seen["cwd"] == tmp_path and seen["start_new_session"] is True
    assert seen["stdin"] == subprocess.DEVNULL
    assert "SECRET" not in seen["env"] and seen["env"]["OLLAMA_HOST"] == "http://u"
    assert seen["env"]["PASTORE_MODEL"] == "m1"
    assert run_dir.stdout.exists() and run_dir.stderr.exists()
    assert seen["stdout"].closed and seen["stderr"].closed


def test_leftover_scopes_keeps_only_pastore_scope_names_and_stop_leftovers_stops_them() -> None:
    listing = (
        "pastore-20261024T010000-3.scope loaded active running /bin/x\n"
        "\n"
        "pastore-foo.service loaded active running not a scope\n"
        "other.scope loaded active running no\n"
    )
    run = RecordingRunner({"systemctl": RecordingRunner.ok(listing)})
    assert stop_leftovers(run) == ["pastore-20261024T010000-3.scope"]
    assert run.calls[0] == [
        "systemctl",
        "--user",
        "list-units",
        "--type=scope",
        "--all",
        "--plain",
        "--no-legend",
        "pastore-*",
    ]
    assert run.calls[1] == ["systemctl", "--user", "stop", "pastore-20261024T010000-3.scope"]


def test_leftover_scopes_fails_loud_when_systemctl_fails() -> None:
    run = RecordingRunner({"systemctl": RecordingRunner.ok("", 1)})
    with pytest.raises(RuntimeError):
        leftover_scopes(run)


def test_has_result_is_false_for_empty_or_failed_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    assert not has_result(out)
    (out / "results.json").write_text("{}")
    assert has_result(out)
    (out / "FAILED").write_text("")
    assert not has_result(out)
