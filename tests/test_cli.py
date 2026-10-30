"""Pins the CLI surface: the add gate (model, evidence, contract), remove, status, stop, check, rerun/ack/reset, run-now and contract gates, start wiring.

Depends on: pastore.cli, pastore.dispatcher (Deps), pastore.policy, pastore.spec (Pair), pastore.state, pastore.manifest, tests.fakes.
"""

import hashlib
import io
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pastore import cli
from pastore.cli import _run_in, main
from pastore.dispatcher import Deps
from pastore.manifest import load_manifest
from pastore.policy import load_policy
from pastore.spec import Pair
from pastore.state import DispatcherLock, JobRow, RunRow, default_paths, open_state
from tests.fakes import FakeClock

T0 = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
MODEL = "qwen3:8b"
OTHER = "gemma3:4b"
EVIDENCE = b"qwen3:8b passes bench on the owner's holdout\n"
FAKE_SHA = "0" * 64
TASK = """  - id: bench
    kind: eval
    argv: [.venv/bin/python, -m, bench]
    cwd: .
    models: [qwen3:8b, gemma3:4b]
    timeout: 2h
    resumable: true
    ollama_env: OLLAMA_HOST
    model_env: BENCH_MODEL
    reader: {run: [.venv/bin/python, read.py]}
"""


class Env:
    """A tmp home, a consumer repo with one task, an evidence file, and a v2 policy file that names the repo."""

    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.root = tmp_path / "torra"
        (self.root / ".venv" / "bin").mkdir(parents=True)
        (self.root / ".venv" / "bin" / "python").symlink_to("/usr/bin/python3")
        self.manifest = self.root / "gpu-tasks.yaml"
        self.manifest.write_text("version: 1\ntasks:\n" + TASK)
        self.evidence = tmp_path / "evidence.md"
        self.evidence.write_bytes(EVIDENCE)
        self.policy = tmp_path / "policy.yaml"
        self.write_policy()
        self.paths = default_paths(self.home)
        self.clock = FakeClock(T0)

    def write_policy(self, *listed: tuple[str, str]) -> None:
        """Write a valid v2 policy listing these (task, model) pairs in order."""
        pairs = "".join(
            f'\n  - {{task: "{t}", model: "{m}", evidence: "{self.evidence}", sha256: "{FAKE_SHA}"}}'
            for t, m in listed
        )
        self.policy.write_text(
            "version: 2\n"
            "hosts:\n"
            '  workstation: {url: "http://127.0.0.1:11435", vram_gb: 24, max_models: 3, parallel: 3}\n'
            "sources:\n"
            '  - {level: hearth, cgroup: ["pastore-*.scope"]}\n'
            f"consumers:\n  torra: {self.manifest}\n"
            f"jobs:{pairs or ' []'}\n"
        )

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

    def add(
        self,
        *extra: str,
        ref: str = "torra/bench",
        model: str = MODEL,
        evidence: Path | None = None,
        tty: bool = False,
    ) -> tuple[int, str]:
        """Run `pastore add` with the fixture's evidence file unless another is given."""
        ev = str(evidence or self.evidence)
        return self.run("add", ref, "--model", model, "--evidence", ev, *extra, tty=tty)

    def fingerprint(self) -> str:
        return load_manifest(self.manifest, "torra")["bench"].fingerprint

    def pass_contract(self) -> None:
        state = open_state(self.paths)
        state.record_contract_pass("torra/bench", self.fingerprint(), T0)
        state.close()

    def jobs(self) -> tuple[Pair, ...]:
        return load_policy(self.policy).jobs

    def listed(self) -> list[tuple[str, str]]:
        return [(p.task, p.model) for p in self.jobs()]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_invariant_add_refuses_without_contract_pass(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = env.add("--yes")
    assert code == 1 and "no passing contract test" in capsys.readouterr().err
    assert env.jobs() == ()
    env.pass_contract()
    code, out = env.add("--yes")
    assert code == 0 and '+  - {task: "torra/bench"' in out
    assert env.listed() == [("torra/bench", MODEL)]
    env.run("remove", "torra/bench", "--model", MODEL, "--yes")
    env.manifest.write_text(env.manifest.read_text().replace("timeout: 2h", "timeout: 3h"))
    code, _ = env.add("--yes")
    assert code == 1 and "no passing contract test" in capsys.readouterr().err
    assert env.jobs() == ()


def test_add_refuses_unknown_task_consumer_and_duplicate(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.pass_contract()
    assert env.add("--yes", ref="torra/missing")[0] == 1
    assert "no such task" in capsys.readouterr().err
    assert env.add("--yes", ref="ghost/bench")[0] == 1
    assert "is not in policy.yaml" in capsys.readouterr().err
    assert env.add("--yes")[0] == 0
    assert env.add("--yes")[0] == 1
    assert "already listed" in capsys.readouterr().err
    assert env.add("--yes", model=OTHER)[0] == 0
    assert env.listed() == [("torra/bench", MODEL), ("torra/bench", OTHER)]


def test_add_refuses_a_model_the_manifest_does_not_list(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.pass_contract()
    code, _ = env.add("--yes", model="llama3:70b")
    err = capsys.readouterr().err
    assert code == 1 and "'llama3:70b' is not in its manifest models" in err
    assert env.jobs() == ()


def test_add_refuses_a_missing_evidence_path_a_directory_and_a_symlink_loop(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env.pass_contract()
    loop = tmp_path / "loop.md"
    loop.symlink_to(loop)
    for bad in (tmp_path / "nope.md", tmp_path, loop):
        code, _ = env.add("--yes", evidence=bad)
        assert code == 1 and "evidence must be an existing regular file" in capsys.readouterr().err
    assert env.jobs() == ()


def test_add_checks_run_in_order_model_then_evidence_then_contract(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.md"
    assert env.add("--yes", model="llama3:70b", evidence=missing)[0] == 1
    assert "is not in its manifest models" in capsys.readouterr().err
    assert env.add("--yes", evidence=missing)[0] == 1
    assert "evidence must be an existing regular file" in capsys.readouterr().err
    assert env.add("--yes")[0] == 1
    assert "no passing contract test" in capsys.readouterr().err
    assert env.jobs() == ()


def test_add_writes_the_pair_with_the_absolute_evidence_path_and_its_sha256(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.pass_contract()
    monkeypatch.chdir(tmp_path)
    code, out = env.add("--yes", evidence=Path(os.path.relpath(env.evidence, tmp_path)))
    assert code == 0 and hashlib.sha256(EVIDENCE).hexdigest() in out
    assert env.jobs() == (
        Pair("torra/bench", MODEL, env.evidence.resolve(), hashlib.sha256(EVIDENCE).hexdigest()),
    )
    assert env.jobs()[0].evidence.is_absolute()


def test_add_stores_a_symlinked_evidence_file_as_its_target(env: Env, tmp_path: Path) -> None:
    env.pass_contract()
    link = tmp_path / "link.md"
    link.symlink_to(env.evidence)
    assert env.add("--yes", evidence=link)[0] == 0
    (pair,) = env.jobs()
    assert pair.evidence == env.evidence.resolve() and pair.evidence != link
    assert pair.sha256 == hashlib.sha256(EVIDENCE).hexdigest()


def test_add_needs_yes_or_a_tty_answer(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.pass_contract()
    before = env.policy.read_text()
    code, out = env.add()
    assert code == 1 and '+  - {task: "torra/bench"' in out
    assert "pass --yes" in capsys.readouterr().err
    assert env.policy.read_text() == before
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    assert env.add(tty=True)[0] == 1
    assert env.policy.read_text() == before
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert env.add(tty=True)[0] == 0
    assert env.listed() == [("torra/bench", MODEL)]


def test_remove_needs_no_contract_and_refuses_an_unlisted_pair(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    assert env.run("remove", "torra/bench", "--model", MODEL, "--yes")[0] == 1
    assert "not listed" in capsys.readouterr().err
    env.write_policy(("torra/bench", MODEL))
    assert env.run("remove", "torra/bench", "--model", OTHER, "--yes")[0] == 1
    assert "not listed" in capsys.readouterr().err
    code, out = env.run("remove", "torra/bench", "--model", MODEL, "--yes")
    assert code == 0 and '-  - {task: "torra/bench"' in out and env.jobs() == ()


def test_status_without_state_is_not_an_error(env: Env) -> None:
    assert env.run("status") == (0, "no state yet\n")
    assert not env.paths.state_dir.exists()


def test_status_reports_dispatcher_jobs_and_reviews(env: Env) -> None:
    env.write_policy(("torra/bench", MODEL))
    state = open_state(env.paths)
    state.heartbeat(env.clock())
    state.set_idle_reason(T0, "no-ready-task")
    state.upsert_job(JobRow("torra/bench", MODEL, "fp", "pending", attempts=1, starved_flagged=1))
    state.insert_run(
        RunRow(
            "r1",
            "torra/bench",
            MODEL,
            "workstation",
            "fp",
            "awaiting-review",
            "2026-10-25T01:00:00+00:00",
        )
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
    assert "torra/bench qwen3:8b: pending attempts=1 last_run=awaiting-review starved" in out
    assert "awaiting review: r1 (torra/bench qwen3:8b)" in out


def test_status_shows_one_line_per_pair_and_no_window_line(env: Env) -> None:
    env.write_policy(("torra/bench", MODEL), ("torra/bench", OTHER))
    state = open_state(env.paths)
    state.upsert_job(JobRow("torra/bench", MODEL, "fp", "done", attempts=2))
    state.close()
    code, out = env.run("status")
    lines = out.splitlines()
    assert code == 0 and "window" not in out
    assert "torra/bench qwen3:8b: done attempts=2 last_run=none" in lines
    assert "torra/bench gemma3:4b: unseen attempts=0 last_run=none" in lines
    assert sum(line.startswith("torra/bench ") for line in lines) == 2


def test_status_last_run_is_per_model(env: Env) -> None:
    env.write_policy(("torra/bench", MODEL), ("torra/bench", OTHER))
    state = open_state(env.paths)
    state.insert_run(RunRow("r1", "torra/bench", MODEL, "workstation", "fp", "done", "t"))
    state.close()
    lines = env.run("status")[1].splitlines()
    assert "torra/bench qwen3:8b: unseen attempts=0 last_run=done" in lines
    assert "torra/bench gemma3:4b: unseen attempts=0 last_run=none" in lines


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
    assert env.run("run-now", "torra/bench", "--model", MODEL)[0] == 1
    assert "not listed" in capsys.readouterr().err


def test_run_now_refuses_a_pair_listed_only_with_another_model(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.write_policy(("torra/bench", OTHER))
    env.pass_contract()
    assert env.run("run-now", "torra/bench", "--model", MODEL)[0] == 1
    err = capsys.readouterr().err
    assert "not listed" in err and "only listed pairs run" in err


def test_usage_errors_exit_2(env: Env) -> None:
    assert env.run()[0] == 2
    assert env.run("bogus")[0] == 2


@pytest.mark.parametrize("cmd", ["add", "remove", "rerun", "run-now"])
def test_model_is_required_where_a_pair_is_named(env: Env, cmd: str) -> None:
    extra = ["--evidence", str(env.evidence)] if cmd == "add" else []
    assert env.run(cmd, "torra/bench", *extra)[0] == 2


def test_add_requires_an_evidence_path(env: Env) -> None:
    assert env.run("add", "torra/bench", "--model", MODEL)[0] == 2


def test_rerun_and_ack(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    state = open_state(env.paths)
    state.upsert_job(JobRow("torra/bench", MODEL, "fp", "failed", attempts=3))
    state.insert_run(
        RunRow("r1", "torra/bench", MODEL, "workstation", "fp", "awaiting-review", "t")
    )
    state.close()
    assert env.run("ack", "r1")[0] == 0
    assert env.run("ack", "r1")[0] == 1
    assert "not awaiting review" in capsys.readouterr().err
    assert env.run("rerun", "torra/bench", "--model", OTHER)[0] == 1
    assert env.run("rerun", "torra/ghost", "--model", MODEL)[0] == 1
    code, out = env.run("rerun", "torra/bench", "--model", MODEL)
    assert code == 0 and out == "torra/bench qwen3:8b: back to pending\n"
    state = open_state(env.paths)
    job = state.get_job("torra/bench", MODEL)
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
    assert (
        env.run(
            "--policy",
            str(bad),
            "add",
            "torra/bench",
            "--model",
            MODEL,
            "--evidence",
            str(env.evidence),
            "--yes",
        )[0]
        == 1
    )
    assert "version must be 2" in capsys.readouterr().err


def test_run_now_needs_a_contract_pass_and_a_running_dispatcher(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.write_policy(("torra/bench", MODEL))
    assert env.run("run-now", "torra/bench", "--model", MODEL)[0] == 1
    assert "no contract pass" in capsys.readouterr().err
    env.pass_contract()
    assert env.run("run-now", "torra/bench", "--model", MODEL)[0] == 1
    assert "dispatcher-down" in capsys.readouterr().err


def test_run_now_queues_a_pending_or_unseen_pair_and_refuses_any_other_status(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.write_policy(("torra/bench", MODEL), ("torra/bench", OTHER))
    env.pass_contract()
    state = open_state(env.paths)
    state.heartbeat(env.clock())
    state.upsert_job(JobRow("torra/bench", MODEL, "fp", "failed", attempts=3))
    with DispatcherLock(env.paths.lock):
        assert env.run("run-now", "torra/bench", "--model", MODEL)[0] == 1
        assert "is failed" in capsys.readouterr().err
        state.rerun("torra/bench", MODEL)
        code, out = env.run("run-now", "torra/bench", "--model", MODEL)
        assert code == 0 and "run-now queued for torra/bench qwen3:8b" in out
        assert state.run_now_requested() == ("torra/bench", MODEL)
        assert env.run("run-now", "torra/bench", "--model", OTHER)[0] == 0
        assert state.run_now_requested() == ("torra/bench", OTHER)
    state.close()


def test_run_now_judges_each_model_of_one_task_on_its_own_state(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.write_policy(("torra/bench", MODEL), ("torra/bench", OTHER))
    env.pass_contract()
    state = open_state(env.paths)
    state.heartbeat(env.clock())
    state.upsert_job(JobRow("torra/bench", MODEL, "fp", "failed", attempts=3))
    with DispatcherLock(env.paths.lock):
        assert env.run("run-now", "torra/bench", "--model", OTHER)[0] == 0
        assert state.run_now_requested() == ("torra/bench", OTHER)
        assert env.run("run-now", "torra/bench", "--model", MODEL)[0] == 1
        assert "qwen3:8b: is failed" in capsys.readouterr().err
        assert state.run_now_requested() == ("torra/bench", OTHER)
    state.close()


def test_contract_refuses_while_the_dispatcher_holds_the_lock(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    env.paths.state_dir.mkdir(parents=True)
    with DispatcherLock(env.paths.lock):
        assert env.run("check", "--contract", "torra/bench")[0] == 1
    assert "stop it first" in capsys.readouterr().err


def test_add_keeps_the_policy_file_mode(env: Env) -> None:
    env.pass_contract()
    env.policy.chmod(0o644)
    assert env.add("--yes")[0] == 0
    assert stat.S_IMODE(env.policy.stat().st_mode) == 0o644


def test_run_in_turns_a_missing_binary_into_exit_127(tmp_path: Path) -> None:
    result = _run_in([str(tmp_path / "nope")], tmp_path, {"PATH": "/usr/bin"})
    assert result.returncode == 127


def test_start_passes_the_state_paths_and_the_real_environment_to_serve(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Deps] = {}

    def fake_serve(deps: Deps, policy_path: Path, **_kw: object) -> int:
        seen["deps"] = deps
        return 0

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert env.run("start")[0] == 0
    assert seen["deps"].paths == env.paths
    assert seen["deps"].base_env == dict(os.environ)
