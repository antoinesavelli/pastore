"""Pins the CLI surface: the add gate (model, evidence, contract), remove, status (state lines and the gateway read), stop, check, rerun/ack/reset, run-now and contract gates, start wiring.

Depends on: pastore.cli, pastore.dispatcher (Deps), pastore.hosts (HttpJson), pastore.policy, pastore.spec (Pair), pastore.state, pastore.manifest, tests.fakes; the gateway is always an injected HttpJson, never 127.0.0.1:11434.
"""

import hashlib
import io
import json
import os
import stat
import sys
import urllib.error
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path

import pytest

from pastore import cli, hosts
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
GATEWAY_URL = "http://127.0.0.1:11434/pastore/status"
LEVELS = ("reins", "webui", "coding", "corsu", "routine", "hearth")
REFUSED = "gateway: not answering ([Errno 111] Connection refused)"
WORKSTATION_LINE = (
    "gateway workstation: slots=1 in_flight=1 "
    "queued reins=0 webui=0 coding=1 corsu=0 routine=0 hearth=2 oldest_wait_s=12.4"
)
EXECUTOR_LINE = (
    "gateway executor: slots=9 in_flight=0 "
    "queued reins=0 webui=0 coding=0 corsu=0 routine=0 hearth=0 oldest_wait_s=none"
)
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


class Gateway:
    """An injected HttpJson: returns `answer`, or raises it when it is an exception; records every call."""

    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.calls: list[tuple[str, Mapping[str, object] | None, float]] = []

    def __call__(self, url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
        self.calls.append((url, body, timeout_s))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def host_entry(**changes: object) -> dict[str, object]:
    """A valid /pastore/status host entry (idle, nothing queued) with these keys replaced."""
    entry: dict[str, object] = {
        "slots": 1,
        "in_flight": 0,
        "queued": dict.fromkeys(LEVELS, 0),
        "oldest_wait_s": None,
    }
    return entry | changes


def without(entry: dict[str, object], key: str) -> dict[str, object]:
    return {k: v for k, v in entry.items() if k != key}


def hosts_answer(**entries: object) -> dict[str, object]:
    return {"hosts": entries}


def two_hosts() -> dict[str, object]:
    """The Phase 3a example answer."""
    return hosts_answer(
        workstation=host_entry(
            in_flight=1,
            queued={**dict.fromkeys(LEVELS, 0), "coding": 1, "hearth": 2},
            oldest_wait_s=12.4,
        ),
        executor=host_entry(slots=9),
    )


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
        self.gateway = Gateway(ConnectionRefusedError(111, "Connection refused"))

    def write_policy(self, *listed: tuple[str, str]) -> None:
        """Write a valid v2 policy listing these (task, model) pairs in order."""
        pairs = "".join(
            f'\n  - {{task: "{t}", model: "{m}", evidence: "{self.evidence}", sha256: "{FAKE_SHA}"}}'
            for t, m in listed
        )
        self.policy.write_text(
            "version: 2\n"
            "hosts:\n"
            '  workstation: {url: "http://127.0.0.1:11435", vram_gib: 24, max_models: 3, parallel: 3, context: 32768}\n'
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
            http_json=self.gateway,
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
    assert env.run("status") == (0, f"no state yet\n{REFUSED}\n")
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
    assert (
        "torra/bench qwen3:8b: pending attempts=1 last_run=awaiting-review host=workstation starved"
        in out
    )
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
    assert "torra/bench qwen3:8b: unseen attempts=0 last_run=done host=workstation" in lines
    assert "torra/bench gemma3:4b: unseen attempts=0 last_run=none" in lines


def test_status_works_while_a_writer_holds_the_db(env: Env) -> None:
    state = open_state(env.paths)
    state.heartbeat(T0)
    assert env.run("status")[0] == 0
    state.close()


def test_status_shows_only_the_newest_tick_error(env: Env) -> None:
    env.write_policy()
    state = open_state(env.paths)
    assert "tick-error" not in env.run("status")[1]
    state.record_event(T0, "tick-error", detail="first")
    state.record_event(T0 + timedelta(seconds=1), "tick-error", detail="second")
    state.close()
    lines = env.run("status")[1].splitlines()
    assert lines[2] == "last tick-error: 2026-10-25T03:00:01+00:00 second"
    assert sum("tick-error" in line for line in lines) == 1


def test_status_shows_the_host_of_the_last_run_on_each_pair_line(env: Env) -> None:
    env.write_policy(("torra/bench", MODEL), ("torra/bench", OTHER))
    state = open_state(env.paths)
    state.insert_run(RunRow("r1", "torra/bench", MODEL, "workstation", "fp", "failed", "t1"))
    state.insert_run(RunRow("r2", "torra/bench", MODEL, "executor", "fp", "done", "t2"))
    state.insert_run(RunRow("r3", "torra/bench", OTHER, "workstation", "fp", "done", "t3"))
    state.close()
    lines = env.run("status")[1].splitlines()
    assert "torra/bench qwen3:8b: unseen attempts=0 last_run=done host=executor" in lines
    assert "torra/bench gemma3:4b: unseen attempts=0 last_run=done host=workstation" in lines


def test_status_lists_running_and_awaiting_review_runs_before_the_gateway(env: Env) -> None:
    env.write_policy(("torra/bench", MODEL), ("torra/bench", OTHER))
    env.gateway.answer = two_hosts()
    state = open_state(env.paths)
    state.upsert_job(JobRow("torra/bench", MODEL, "fp", "running", attempts=1))
    state.insert_run(
        RunRow("r1", "torra/bench", OTHER, "workstation", "fp", "awaiting-review", "t1")
    )
    state.insert_run(RunRow("r2", "torra/bench", MODEL, "executor", "fp", "running", "t2"))
    state.close()
    assert env.run("status") == (
        0,
        "dispatcher: dispatcher-down\n"
        "idle: none\n"
        "workload watch: not run yet\n"
        "torra/bench qwen3:8b: running attempts=1 last_run=running host=executor\n"
        "torra/bench gemma3:4b: unseen attempts=0 last_run=awaiting-review host=workstation\n"
        "awaiting review: r1 (torra/bench gemma3:4b)\n"
        "running: r2 (torra/bench qwen3:8b on executor)\n"
        f"{WORKSTATION_LINE}\n{EXECUTOR_LINE}\n",
    )


def test_status_prints_one_gateway_line_per_host_from_the_status_endpoint(env: Env) -> None:
    env.write_policy(("torra/bench", MODEL))
    open_state(env.paths).close()
    env.gateway.answer = two_hosts()
    code, out = env.run("status")
    assert code == 0 and out.splitlines()[-2:] == [WORKSTATION_LINE, EXECUTOR_LINE]
    assert env.gateway.calls == [(GATEWAY_URL, None, 2.0)]


def test_status_prints_the_gateway_lines_even_with_no_state_yet(env: Env) -> None:
    env.gateway.answer = two_hosts()
    assert env.run("status") == (0, f"no state yet\n{WORKSTATION_LINE}\n{EXECUTOR_LINE}\n")
    assert not env.paths.state_dir.exists()


def test_status_gateway_wait_is_rounded_to_a_tenth_and_ints_pass(env: Env) -> None:
    env.gateway.answer = hosts_answer(
        a=host_entry(oldest_wait_s=12.44),
        b=host_entry(oldest_wait_s=3),
        c=host_entry(oldest_wait_s=0.0),
    )
    waits = [line.rsplit(" ", 1)[1] for line in env.run("status")[1].splitlines()[1:]]
    assert waits == ["oldest_wait_s=12.4", "oldest_wait_s=3", "oldest_wait_s=0.0"]
    env.gateway.answer = hosts_answer(a=host_entry(oldest_wait_s=10**400))
    code, out = env.run("status")
    assert code == 0 and out.endswith(f"oldest_wait_s={10**400}\n")


def test_status_defaults_to_the_real_http_json_reading_the_gateway_url(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Gateway(two_hosts())
    monkeypatch.setattr(hosts, "http_json", fake)
    out = io.StringIO()
    code = main(["status"], home=env.home, policy_path=env.policy, clock=env.clock, out=out)
    assert code == 0 and out.getvalue().splitlines()[-2:] == [WORKSTATION_LINE, EXECUTOR_LINE]
    assert fake.calls == [(GATEWAY_URL, None, 2.0)]


HTTP_403 = urllib.error.HTTPError(GATEWAY_URL, 403, "Forbidden", Message(), None)
HTTP_404 = urllib.error.HTTPError(GATEWAY_URL, 404, "Not Found", Message(), None)
UNREACHABLE = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (ConnectionRefusedError(111, "Connection refused"), "[Errno 111] Connection refused"),
        (UNREACHABLE, "<urlopen error [Errno 111] Connection refused>"),
        (TimeoutError("timed out"), "timed out"),
        (HTTP_403, "HTTP Error 403: Forbidden"),
        (HTTP_404, "HTTP Error 404: Not Found"),
        (
            json.JSONDecodeError("Expecting value", "", 0),
            "Expecting value: line 1 column 1 (char 0)",
        ),
        (
            ValueError(f"{GATEWAY_URL}: answer over 8388608 bytes"),
            f"{GATEWAY_URL}: answer over 8388608 bytes",
        ),
        (OSError("two\n  lines"), "two lines"),
        (OSError(), "OSError"),
        (OSError("x" * 79), "x" * 79),
        (OSError("x" * 80), "x" * 80),
        (OSError("x" * 81), "x" * 77 + "..."),
        (OSError("x" * 200), "x" * 77 + "..."),
        (OSError("\x1b[31mred"), "[31mred"),
        (OSError("a\u202eb\x00c\x7fd\u00a0e\re"), "a b c d e e"),
        (OSError("\x1b\x07"), "OSError"),
    ],
)
def test_status_reports_a_gateway_that_does_not_answer_and_still_exits_0(
    env: Env, failure: Exception, reason: str
) -> None:
    open_state(env.paths).close()
    env.gateway.answer = failure
    code, out = env.run("status")
    assert code == 0 and out.splitlines()[-1] == f"gateway: not answering ({reason})"
    assert "gateway workstation" not in out and "Traceback" not in out
    assert all(c.isprintable() or c == "\n" for c in out)
    assert env.gateway.calls == [(GATEWAY_URL, None, 2.0)]


def one_host(entry: object) -> dict[str, object]:
    return {"hosts": {"workstation": entry}}


HOSTS_BAD = "hosts is missing, empty or not an object"
QUEUED_BAD = "workstation: queued is missing or not an object"
BAD_SHAPES: dict[str, tuple[object, str]] = {
    "answer-a-list": ([], HOSTS_BAD),
    "answer-a-string": ("ok", HOSTS_BAD),
    "answer-null": (None, HOSTS_BAD),
    "no-hosts-key": ({}, HOSTS_BAD),
    "hosts-a-list": ({"hosts": []}, HOSTS_BAD),
    "hosts-empty": ({"hosts": {}}, HOSTS_BAD),
    "host-not-an-object": (one_host(3), "workstation: not an object"),
    "queued-missing": (one_host(without(host_entry(), "queued")), QUEUED_BAD),
    "queued-a-list": (one_host(host_entry(queued=[])), QUEUED_BAD),
    "queued-a-full-list": (one_host(host_entry(queued=[0] * 6)), QUEUED_BAD),
    "slots-missing": (
        one_host(without(host_entry(), "slots")),
        "workstation: slots is missing or not a non-negative integer",
    ),
    "slots-bool": (
        one_host(host_entry(slots=True)),
        "workstation: slots is missing or not a non-negative integer",
    ),
    "slots-float": (
        one_host(host_entry(slots=1.0)),
        "workstation: slots is missing or not a non-negative integer",
    ),
    "slots-string": (
        one_host(host_entry(slots="1")),
        "workstation: slots is missing or not a non-negative integer",
    ),
    "slots-negative": (
        one_host(host_entry(slots=-1)),
        "workstation: slots is missing or not a non-negative integer",
    ),
    "in-flight-missing": (
        one_host(without(host_entry(), "in_flight")),
        "workstation: in_flight is missing or not a non-negative integer",
    ),
    "in-flight-bool": (
        one_host(host_entry(in_flight=False)),
        "workstation: in_flight is missing or not a non-negative integer",
    ),
    "wait-missing": (
        one_host(without(host_entry(), "oldest_wait_s")),
        "workstation: oldest_wait_s is missing or not a non-negative number or null",
    ),
    "wait-string": (
        one_host(host_entry(oldest_wait_s="12.4")),
        "workstation: oldest_wait_s is missing or not a non-negative number or null",
    ),
    "wait-bool": (
        one_host(host_entry(oldest_wait_s=True)),
        "workstation: oldest_wait_s is missing or not a non-negative number or null",
    ),
    "wait-nan": (
        one_host(host_entry(oldest_wait_s=float("nan"))),
        "workstation: oldest_wait_s is missing or not a non-negative number or null",
    ),
    "wait-inf": (
        one_host(host_entry(oldest_wait_s=float("inf"))),
        "workstation: oldest_wait_s is missing or not a non-negative number or null",
    ),
    "wait-negative": (
        one_host(host_entry(oldest_wait_s=-0.5)),
        "workstation: oldest_wait_s is missing or not a non-negative number or null",
    ),
}


@pytest.mark.parametrize(("answer", "what"), BAD_SHAPES.values(), ids=BAD_SHAPES)
def test_status_reports_a_bad_gateway_shape_as_unexpected_and_still_exits_0(
    env: Env, answer: object, what: str
) -> None:
    env.gateway.answer = answer
    assert env.run("status") == (
        0,
        f"no state yet\ngateway: unexpected /pastore/status answer ({what})\n",
    )


@pytest.mark.parametrize("level", LEVELS)
def test_status_needs_every_level_name_as_a_plain_count(env: Env, level: str) -> None:
    full = dict.fromkeys(LEVELS, 0)
    bad = f"gateway: unexpected /pastore/status answer (workstation: queued.{level} is missing or not a non-negative integer)\n"
    for broken in (
        {k: v for k, v in full.items() if k != level},
        {**full, level: True},
        {**full, level: 1.5},
        {**full, level: None},
        {**full, level: -1},
    ):
        env.gateway.answer = one_host(host_entry(queued=broken))
        assert env.run("status") == (0, f"no state yet\n{bad}")


def unexpected(what: str) -> str:
    return f"gateway: unexpected /pastore/status answer ({what})\n"


NAME_RULE = "is not 1 to 64 of A-Z a-z 0-9 . _ -"


def test_status_refuses_a_host_name_that_would_forge_a_second_line(env: Env) -> None:
    env.gateway.answer = hosts_answer(**{"ws\ngateway fake: slots=99": host_entry()})
    assert env.run("status") == (
        0,
        "no state yet\n" + unexpected(rf"host name 'ws\ngateway fake: slots=99' {NAME_RULE}"),
    )


@pytest.mark.parametrize(
    ("name", "shown"),
    [
        ("", "''"),
        ("has space", "'has space'"),
        ("a\x1b[31m", r"'a\x1b[31m'"),
        ("caf\u00e9", "'caf\u00e9'"),
        ("a/b", "'a/b'"),
        ("a\nb", r"'a\nb'"),
        ("ws\n", r"'ws\n'"),
    ],
)
def test_status_refuses_host_names_outside_the_plain_name_alphabet(
    env: Env, name: str, shown: str
) -> None:
    env.gateway.answer = hosts_answer(**{name: host_entry()})
    assert env.run("status") == (0, f"no state yet\n{unexpected(f'host name {shown} {NAME_RULE}')}")


@pytest.mark.parametrize("name", ["a" * 65, "a" * 299 + "\n", "\u00e9" * 300])
def test_status_caps_the_unexpected_text_for_a_long_host_name(env: Env, name: str) -> None:
    env.gateway.answer = hosts_answer(**{name: host_entry()})
    code, out = env.run("status")
    line = out.splitlines()[-1]
    prefix = "gateway: unexpected /pastore/status answer ("
    assert code == 0 and out.count("\n") == 2
    assert line.startswith(prefix + "host name '") and line.endswith("...)")
    assert len(line) == len(prefix) + 80 + 1


def test_status_prints_a_host_name_of_up_to_64_plain_characters(env: Env) -> None:
    name = "Ab9._-" + "a" * 58
    env.gateway.answer = hosts_answer(**{name: host_entry()})
    code, out = env.run("status")
    assert code == 0 and out.splitlines()[-1].startswith(f"gateway {name}: slots=1 in_flight=0 ")


def test_status_with_an_invalid_policy_exits_1_and_prints_no_gateway_line(env: Env) -> None:
    open_state(env.paths).close()
    env.policy.write_text("version: 7\n")
    env.gateway.answer = two_hosts()
    code, out = env.run("status")
    lines = out.splitlines()
    assert code == 1 and lines[:3] == [
        "dispatcher: dispatcher-down",
        "idle: none",
        "workload watch: not run yet",
    ]
    assert lines[3].startswith("policy error: ") and "version must be 2" in lines[3]
    assert len(lines) == 4 and "gateway" not in out
    assert env.gateway.calls == []


def test_status_prints_no_gateway_host_line_when_a_later_host_is_malformed(env: Env) -> None:
    env.gateway.answer = hosts_answer(workstation=host_entry(), executor=host_entry(slots=True))
    code, out = env.run("status")
    assert code == 0 and "gateway workstation" not in out
    assert out.splitlines()[-1] == (
        "gateway: unexpected /pastore/status answer (executor: slots is missing or not a non-negative integer)"
    )


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


def test_start_gives_the_dispatcher_the_gateway_url_and_the_real_http_json(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Deps] = {}

    def fake_serve(deps: Deps, policy_path: Path, **_kw: object) -> int:
        seen["deps"] = deps
        return 0

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert env.run("start")[0] == 0
    assert seen["deps"].ollama_url == "http://127.0.0.1:11434"
    assert seen["deps"].http_json is hosts.http_json
    assert env.gateway.calls == []
