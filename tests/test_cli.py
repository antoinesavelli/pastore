"""Pins the CLI surface: the add gate, remove, status, stop, check, rerun/ack/reset and the not-wired commands.

Depends on: fucone.cli, fucone.state, fucone.manifest, tests.fakes and pytest only.
"""

import io
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fucone.cli import main
from fucone.manifest import load_manifest
from fucone.policy import load_policy
from fucone.state import DispatcherLock, JobRow, RunRow, default_paths, open_state
from tests.fakes import FakeClock

T0 = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
TASK = """  - id: bench
    kind: eval
    argv: [.venv/bin/python, -m, bench]
    cwd: .
    model: qwen3:8b
    timeout: 2h
    resumable: true
    ollama_env: OLLAMA_HOST
    reader: {run: [.venv/bin/python, read.py]}
"""


class Env:
    """A tmp home, a consumer repo with one task, and a policy file that names it."""

    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.root = tmp_path / "torra"
        (self.root / ".venv" / "bin").mkdir(parents=True)
        (self.root / ".venv" / "bin" / "python").symlink_to("/usr/bin/python3")
        self.manifest = self.root / "gpu-tasks.yaml"
        self.manifest.write_text("version: 1\ntasks:\n" + TASK)
        self.policy = tmp_path / "policy.yaml"
        self.policy.write_text(
            f'version: 1\nwindow: {{start: "01:00", end: "07:00"}}\n'
            f"consumers:\n  torra: {self.manifest}\njobs: []\n"
        )
        self.paths = default_paths(self.home)
        self.clock = FakeClock(T0)

    def run(self, *argv: str, tty: bool = False) -> tuple[int, str]:
        out = io.StringIO()
        code = main(
            [*argv],
            home=self.home,
            policy_path=self.policy,
            clock=self.clock,
            out=out,
            stdin_isatty=lambda: tty,
        )
        return code, out.getvalue()

    def fingerprint(self) -> str:
        return load_manifest(self.manifest, "torra")["bench"].fingerprint

    def pass_contract(self) -> None:
        state = open_state(self.paths)
        state.record_contract_pass("torra/bench", self.fingerprint(), T0)
        state.close()

    def jobs(self) -> tuple[str, ...]:
        return load_policy(self.policy).jobs


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_invariant_add_refuses_without_contract_pass(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = env.run("add", "torra/bench", "--yes")
    assert code == 1 and "no passing contract test" in capsys.readouterr().err
    assert env.jobs() == ()
    env.pass_contract()
    code, out = env.run("add", "torra/bench", "--yes")
    assert code == 0 and "+  - torra/bench" in out
    assert env.jobs() == ("torra/bench",)
    env.run("remove", "torra/bench", "--yes")
    env.manifest.write_text(env.manifest.read_text().replace("timeout: 2h", "timeout: 3h"))
    code, _ = env.run("add", "torra/bench", "--yes")
    assert code == 1 and "no passing contract test" in capsys.readouterr().err
    assert env.jobs() == ()


def test_add_refuses_unknown_task_consumer_and_duplicate(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.pass_contract()
    assert env.run("add", "torra/missing", "--yes")[0] == 1
    assert "no such task" in capsys.readouterr().err
    assert env.run("add", "ghost/bench", "--yes")[0] == 1
    assert "not listed under consumers" in capsys.readouterr().err
    assert env.run("add", "torra/bench", "--yes")[0] == 0
    assert env.run("add", "torra/bench", "--yes")[0] == 1
    assert "already listed" in capsys.readouterr().err


def test_add_needs_yes_or_a_tty_answer(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.pass_contract()
    before = env.policy.read_text()
    code, out = env.run("add", "torra/bench")
    assert code == 1 and "+  - torra/bench" in out and "pass --yes" in capsys.readouterr().err
    assert env.policy.read_text() == before
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    assert env.run("add", "torra/bench", tty=True)[0] == 1
    assert env.policy.read_text() == before
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert env.run("add", "torra/bench", tty=True)[0] == 0
    assert env.jobs() == ("torra/bench",)


def test_remove_needs_no_contract_and_refuses_unlisted(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    assert env.run("remove", "torra/bench", "--yes")[0] == 1
    assert "not listed" in capsys.readouterr().err
    env.policy.write_text(env.policy.read_text().replace("jobs: []", "jobs:\n  - torra/bench"))
    code, out = env.run("remove", "torra/bench", "--yes")
    assert code == 0 and "-  - torra/bench" in out and env.jobs() == ()


def test_status_without_state_is_not_an_error(env: Env) -> None:
    assert env.run("status") == (0, "no state yet\n")
    assert not env.paths.state_dir.exists()


def test_status_reports_dispatcher_jobs_and_reviews(env: Env) -> None:
    env.pass_contract()
    env.run("add", "torra/bench", "--yes")
    state = open_state(env.paths)
    state.heartbeat(env.clock())
    state.set_idle_reason(T0, "no-ready-task")
    state.upsert_job(JobRow("torra/bench", "fp", "pending", attempts=1, starved_flagged=1))
    state.insert_run(
        RunRow("r1", "torra/bench", "fp", "awaiting-review", "2026-10-25T01:00:00+00:00")
    )
    code, out = env.run("status")
    assert code == 0 and "dispatcher: dispatcher-down" in out
    with DispatcherLock(env.paths.lock):
        _, out = env.run("status")
        assert "dispatcher: up" in out
        env.clock.advance(61)
        _, stale = env.run("status")
        assert "dispatcher: dispatcher-down" in stale
    state.close()
    assert "idle: no-ready-task" in out
    assert "torra/bench: pending attempts=1 last_run=awaiting-review starved" in out
    assert "awaiting review: r1 (torra/bench)" in out


def test_status_works_while_a_writer_holds_the_db(env: Env) -> None:
    state = open_state(env.paths)
    state.heartbeat(T0)
    assert env.run("status")[0] == 0
    state.close()


def test_stop_writes_the_request(env: Env) -> None:
    assert env.run("stop")[0] == 0
    state = open_state(env.paths)
    assert state.stop_requested() == "graceful"
    assert env.run("stop", "--now")[0] == 0
    assert state.stop_requested() == "now"
    state.close()


def test_check_path_prints_ref_and_fingerprint_or_errors(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = env.run("check", str(env.manifest))
    assert code == 0 and out == f"torra/bench {env.fingerprint()}\n"
    env.manifest.write_text(env.manifest.read_text().replace("resumable: true", "resumable: false"))
    assert env.run("check", str(env.manifest))[0] == 1
    assert "resumable must be literally true" in capsys.readouterr().err


def test_check_other_path_uses_directory_name_as_consumer(env: Env, tmp_path: Path) -> None:
    other = tmp_path / "vault" / "gpu-tasks.yaml"
    other.parent.mkdir()
    other.write_text(env.manifest.read_text().replace(".venv/bin/python", "/usr/bin/python3"))
    assert env.run("check", str(other))[0] == 1
    other.write_text(env.manifest.read_text())
    (other.parent / ".venv" / "bin").mkdir(parents=True)
    assert env.run("check", str(other))[0] == 0


def test_check_needs_exactly_one_target(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert env.run("check")[0] == 1
    assert env.run("check", str(env.manifest), "--contract", "torra/bench")[0] == 1
    assert "exactly one" in capsys.readouterr().err


def test_run_now_refuses_an_unlisted_job(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert env.run("run-now", "torra/bench")[0] == 1
    assert "not listed" in capsys.readouterr().err


def test_usage_errors_exit_2(env: Env) -> None:
    assert env.run()[0] == 2
    assert env.run("bogus")[0] == 2


def test_rerun_and_ack(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    state = open_state(env.paths)
    state.upsert_job(JobRow("torra/bench", "fp", "failed", attempts=3))
    state.insert_run(RunRow("r1", "torra/bench", "fp", "awaiting-review", "t"))
    state.close()
    assert env.run("ack", "r1")[0] == 0
    assert env.run("ack", "r1")[0] == 1
    assert "not awaiting review" in capsys.readouterr().err
    assert env.run("rerun", "torra/bench")[0] == 0
    assert env.run("rerun", "torra/ghost")[0] == 1
    state = open_state(env.paths)
    job = state.get_job("torra/bench")
    assert job is not None and job.status == "pending" and job.attempts == 0
    state.close()


def test_reset_deletes_state_and_refuses_while_locked(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    open_state(env.paths).close()
    with DispatcherLock(env.paths.lock):
        assert env.run("reset")[0] == 1
        assert "dispatcher is running" in capsys.readouterr().err
    code, out = env.run("reset")
    assert code == 0 and "state.db" in out and not env.paths.db.exists()
    assert env.run("reset") == (0, "nothing to delete\n")


def test_policy_flag_overrides_default(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 7\n")
    assert env.run("--policy", str(bad), "add", "torra/bench", "--yes")[0] == 1
    assert "version must be 1" in capsys.readouterr().err
