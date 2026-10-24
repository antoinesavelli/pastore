"""End-to-end contract scenarios: real FakeOllama, real child processes running tiny job scripts, no systemd.

Depends on: fucone.contract, fucone.runner, fucone.spec, tests.fakes (RecordingRunner).
"""

import os
import sys
import textwrap
import time
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from fucone.contract import CONTRACT_TIMEOUT_CAP_S, SCENARIOS, ContractResult, run_contract
from fucone.runner import RunHandle, start_run
from fucone.spec import RunReader, TaskSpec
from tests.fakes import FakeClock, RecordingRunner

PRELUDE = """
import os, pathlib, urllib.request
out = pathlib.Path(os.environ["FUCONE_RUN_DIR"])
url = os.environ["MY_OLLAMA"] + "/api/generate"
"""
CLIENT_TIMEOUT = "timeout=1"


def make_spec(tmp_path: Path) -> TaskSpec:
    return TaskSpec(
        consumer="demo",
        id="job",
        kind="eval",
        argv=("unused",),
        cwd=tmp_path,
        env={},
        models=(),
        timeout_s=3,
        ok_exit=frozenset({0}),
        ollama_env="MY_OLLAMA",
        reader=RunReader(argv=("true",)),
        ready=None,
        root=tmp_path,
        fingerprint="a" * 64,
    )


def contract_for(tmp_path: Path, body: str, client_timeout: str = CLIENT_TIMEOUT) -> ContractResult:
    script = tmp_path / "job.py"
    script.write_text(PRELUDE + textwrap.dedent(body).replace("TIMEOUT", client_timeout))
    spec = replace(make_spec(tmp_path), argv=(sys.executable, str(script)))
    handles: list[RunHandle] = []

    def start(*args: object) -> RunHandle:
        handle = partial(start_run, wrap=lambda run_id, argv: list(argv))(*args)  # type: ignore[arg-type]
        handles.append(handle)
        return handle

    result = run_contract(
        spec,
        tmp_path / "runs",
        {"PATH": os.environ["PATH"]},
        clock=lambda: datetime.now(UTC),
        sleep=time.sleep,
        start=start,
        run=RecordingRunner(),
        poll_s=0.05,
    )
    assert len(handles) == len(SCENARIOS)
    assert all(h.poll() is not None for h in handles), "a job process was left running"
    return result


def test_contract_fails_a_job_that_exits_0_on_ollama_errors(tmp_path: Path) -> None:
    result = contract_for(
        tmp_path,
        """
        try:
            urllib.request.urlopen(url, TIMEOUT)
        except Exception as err:
            (out / "results.json").write_text('{"ok": false, "error": "%s"}' % err)
        """,
    )
    assert not result.passed
    assert all(s.exit_code == 0 and s.result_left and not s.passed for s in result.scenarios)


def test_contract_passes_a_job_that_fails_loud(tmp_path: Path) -> None:
    result = contract_for(tmp_path, "urllib.request.urlopen(url, TIMEOUT)")
    assert result.passed
    assert [s.name for s in result.scenarios] == list(SCENARIOS)
    assert all(s.exit_code == 1 and not s.result_left and not s.timed_out for s in result.scenarios)


def test_contract_fails_a_job_that_leaves_a_result_behind(tmp_path: Path) -> None:
    result = contract_for(
        tmp_path,
        """
        (out / "results.json").write_text("{}")
        urllib.request.urlopen(url, TIMEOUT)
        """,
    )
    assert not result.passed
    assert all(s.exit_code == 1 and s.result_left for s in result.scenarios)


def test_contract_passes_a_job_that_marks_failed(tmp_path: Path) -> None:
    result = contract_for(
        tmp_path,
        """
        import sys
        try:
            urllib.request.urlopen(url, TIMEOUT)
        except Exception:
            (out / "FAILED").write_text("ollama error")
            sys.exit(1)
        """,
    )
    assert result.passed
    assert all(not s.result_left for s in result.scenarios)


def test_contract_kills_a_hanging_job_at_the_timeout_and_passes_if_no_result(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result = contract_for(tmp_path, "urllib.request.urlopen(url)")
    assert time.monotonic() - started < 15
    assert result.passed
    hang = {s.name: s for s in result.scenarios}["hang"]
    assert hang.timed_out and hang.exit_code is None and not hang.result_left
    assert not any(s.timed_out for s in result.scenarios if s.name != "hang")


def test_contract_timeout_is_capped_at_300_seconds(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 24, 1, 0, tzinfo=UTC))
    stops: list[float] = []

    class Stuck:
        def poll(self) -> int | None:
            return None

        def stop(self, run: object, wait_s: float = 30.0) -> int:
            stops.append((clock() - t0).total_seconds())
            return -9

    t0 = clock()
    spec = replace(make_spec(tmp_path), timeout_s=7200)
    result = run_contract(
        spec,
        tmp_path / "runs",
        {},
        clock=clock,
        sleep=clock.advance,
        start=lambda *args: Stuck(),
        run=RecordingRunner(),
        poll_s=10.0,
    )
    assert stops == [CONTRACT_TIMEOUT_CAP_S * n for n in (1, 2, 3)]
    assert result.passed and all(s.timed_out for s in result.scenarios)
