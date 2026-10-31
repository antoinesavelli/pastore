"""Check 14 (R34), dispatcher SIGKILL mid-run: the restart recovers the run, charges only a second consecutive crash, and runs the pair again.

Depends on: tests/interrupt_harness.py, a real dispatcher process on stubbed host seams whose job outlives it as a scope would.
"""

import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from pastore.spec import WORKSTATION
from pastore.state import lock_is_held
from tests.interrupt_harness import (
    CALLS,
    CONSUMER,
    MODEL,
    QUICK,
    WS_URL,
    Fleet,
    events,
    harness_alive,
    host_calls,
    idle_reason,
    job,
    job_alive,
    job_pid,
    job_started,
    prepare,
    read_pid,
    running_job,
    runs,
    wait_until,
)

PAIR = f"{CONSUMER}/a"
# Starts the real harness (argv[2:]) in a session of its own, records its pid, and stays as its parent for 60 s.
MIDDLE = """\
import pathlib, subprocess, sys, time
proc = subprocess.Popen(sys.argv[2:], start_new_session=True, stdin=subprocess.DEVNULL)
pathlib.Path(sys.argv[1]).write_text(str(proc.pid))
time.sleep(60)
"""
FAILURES = ("refused", "failed", "tick-error", "policy-error", "reader-failed")


@pytest.fixture
def fleet(tmp_path: Path) -> Iterator[Fleet]:
    with Fleet(tmp_path) as started:
        yield started


def stops(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if call[:3] == ["systemctl", "--user", "stop"]]


def test_check_14_sigkill_mid_run_restart_recovers_uncharged_and_runs_the_pair_again(
    tmp_path: Path, fleet: Fleet
) -> None:
    prepared = prepare(tmp_path)
    first = fleet.spawn(prepared)
    old = wait_until(lambda: running_job(prepared), 15, "the first run to be running")
    old_pid = job_pid(prepared, old.run_id)
    assert old_pid is not None

    assert first.kill() == -signal.SIGKILL
    assert job_alive(prepared.root, old_pid)  # the scope outlives the dispatcher
    assert not lock_is_held(prepared.paths.lock)

    second = fleet.spawn(prepared)
    new = wait_until(lambda: running_job(prepared, {old.run_id}), 15, "the pair to run again")

    ended = next(run for run in runs(prepared) if run.run_id == old.run_id)
    assert (ended.status, ended.ref, ended.model, ended.host) == (
        "crashed",
        PAIR,
        MODEL,
        WORKSTATION,
    )
    assert ended.ended_at is not None and ended.exit_code is None
    row = job(prepared)
    assert row is not None and (row.attempts, row.crash_streak) == (0, 1)
    assert lock_is_held(prepared.paths.lock)

    calls = host_calls(prepared, second.proc.pid)
    assert stops(calls) == [["systemctl", "--user", "stop", f"pastore-{old.run_id}.scope"]]
    assert calls[0][:3] == ["systemctl", "--user", "list-units"]
    assert stops(host_calls(prepared, first.proc.pid)) == []
    assert not job_alive(prepared.root, old_pid)

    started = job_started(prepared, new.run_id)
    assert started is not None and new.run_id != old.run_id
    assert (started["ollama"], started["model"]) == (WS_URL, MODEL)
    assert started["run_dir"] == str(prepared.paths.runs_dir / new.run_id / "out")
    log = events(prepared)
    recovers = [e for e in log if e.kind == "recover"]
    starts = [e for e in log if e.kind == "dispatcher-start"]
    run_starts = [e for e in log if e.kind == "run-start"]
    assert [e.detail for e in recovers] == ["", f"{PAIR} {MODEL}"]
    assert len(starts) == 2 and len(run_starts) == 2
    assert recovers[1].id < starts[1].id < run_starts[1].id
    assert run_starts[1].detail.split()[0] == new.run_id
    assert [e for e in log if e.kind in FAILURES] == [], second.stderr()


def test_check_14_a_second_consecutive_crash_is_charged_once_and_the_pair_still_runs(
    tmp_path: Path, fleet: Fleet
) -> None:
    prepared = prepare(tmp_path)
    first = fleet.spawn(prepared)
    one = wait_until(lambda: running_job(prepared), 15, "run 1")
    assert first.kill() == -signal.SIGKILL

    second = fleet.spawn(prepared)
    two = wait_until(lambda: running_job(prepared, {one.run_id}), 15, "run 2")
    row = job(prepared)
    assert row is not None and (row.attempts, row.crash_streak) == (0, 1)
    assert second.kill() == -signal.SIGKILL
    assert not lock_is_held(prepared.paths.lock)

    # STOP keeps the third dispatcher from placing anything, so the recovered row can be read as pending
    prepared.paths.stop_file.touch()
    third = fleet.spawn(prepared)
    wait_until(lambda: idle_reason(prepared) == "stopped", 15, "the third dispatcher to idle")
    row = job(prepared)
    assert row is not None
    assert (row.status, row.attempts, row.crash_streak) == ("pending", 1, 2)
    assert [(r.run_id, r.status) for r in runs(prepared)] == [
        (one.run_id, "crashed"),
        (two.run_id, "crashed"),
    ]
    assert stops(host_calls(prepared, third.proc.pid)) == [
        ["systemctl", "--user", "stop", f"pastore-{two.run_id}.scope"]
    ]

    prepared.paths.stop_file.unlink()
    three = wait_until(lambda: running_job(prepared, {one.run_id, two.run_id}), 15, "run 3")
    row = job(prepared)
    assert row is not None
    assert (row.status, row.attempts, row.crash_streak) == ("running", 1, 2)
    assert [r.status for r in runs(prepared)] == ["crashed", "crashed", "running"]
    assert len({one.run_id, two.run_id, three.run_id}) == 3
    recovers = [e.detail for e in events(prepared) if e.kind == "recover"]
    assert recovers == ["", f"{PAIR} {MODEL}", f"{PAIR} {MODEL}"]
    assert [e for e in events(prepared) if e.kind in FAILURES] == [], third.stderr()


def test_the_harness_exits_by_itself_when_its_parent_is_killed(
    tmp_path: Path, fleet: Fleet
) -> None:
    prepared = prepare(tmp_path)
    pidfile = tmp_path / "harness.pid"
    middle = fleet.spawn(prepared, launcher=[sys.executable, "-c", MIDDLE, str(pidfile)])
    pid = wait_until(lambda: read_pid(pidfile), 15, "the harness to be started")
    try:
        wait_until(lambda: running_job(prepared), 15, "the harness to be ticking")
        assert middle.kill() == -signal.SIGKILL
        wait_until(lambda: not harness_alive(pid), 10, "the orphaned harness to exit")
    finally:
        if harness_alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_the_harness_exits_by_itself_after_its_life_cap(tmp_path: Path, fleet: Fleet) -> None:
    prepared = prepare(tmp_path)
    harness = fleet.spawn(prepared, life_s=2)
    wait_until(lambda: running_job(prepared), 15, "the harness to be ticking")
    assert harness.proc.wait(timeout=10) == -signal.SIGALRM


def test_a_recurring_task_runs_again_after_it_is_done(tmp_path: Path, fleet: Fleet) -> None:
    ready = {"ready": {"every": "1s", "check": ["./ready"]}}
    prepared = prepare(tmp_path, QUICK, fields=ready)
    fleet.spawn(prepared)
    done = wait_until(
        lambda: [r for r in runs(prepared) if r.status == "done"][1:], 20, "two done runs"
    )
    assert done[0].ref == PAIR and [e for e in events(prepared) if e.kind == "refused"] == []


def test_host_calls_ignores_a_line_the_harness_is_still_writing(tmp_path: Path) -> None:
    prepared = prepare(tmp_path)
    whole = '{"pid": 7, "argv": ["a", "b"]}\n'
    (prepared.home / CALLS).write_text(whole + '{"pid": 7, "argv": ["c"')
    assert host_calls(prepared) == [["a", "b"]]
