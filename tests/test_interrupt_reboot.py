"""Check 14 (R34), reboot with a running row and no scope: the restart recovers the run with nothing to stop, charges only a second consecutive crash, and runs the pair again.

Depends on: tests/interrupt_harness.py, a real dispatcher process on stubbed host seams; a power loss kills it and its job together and systemd lists no scope.
"""

import signal
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pastore.spec import WORKSTATION
from pastore.state import lock_is_held, open_state
from tests.interrupt_harness import (
    CONSUMER,
    MODEL,
    QUICK,
    WS_URL,
    Fleet,
    Harness,
    Prepared,
    events,
    host_calls,
    idle_reason,
    job,
    job_alive,
    job_pid,
    job_pids,
    job_started,
    kill_group,
    prepare,
    read_state,
    running_job,
    runs,
    wait_until,
)

PAIR = f"{CONSUMER}/a"
GHOST = f"{CONSUMER}/ghost"
FAILURES = ("refused", "failed", "tick-error", "policy-error", "reader-failed", "dispatcher-stop")
# Counts its executions before it writes started.json, so a kill after started.json always follows the count; the
# second execution ends at once and every other one sleeps until it is killed.
SECOND_ENDS = (
    """\
import pathlib
count = pathlib.Path("runs.count")
n = int(count.read_text()) + 1 if count.exists() else 1
count.write_text(str(n))
"""
    + QUICK
    + "if n != 2:\n    time.sleep(90)\n"
)


@pytest.fixture
def fleet(tmp_path: Path) -> Iterator[Fleet]:
    with Fleet(tmp_path) as started:
        yield started


def stops(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if call[:3] == ["systemctl", "--user", "stop"]]


def power_loss(prepared: Prepared, harness: Harness, run_id: str) -> None:
    """Cut the power: the dispatcher's group first, so it cannot see its job die and charge it, then the job's; wait until both are gone."""
    pid = job_pid(prepared, run_id)
    assert pid is not None
    kill_group(harness.proc.pid)
    assert harness.proc.wait(timeout=5) == -signal.SIGKILL
    kill_group(pid)
    wait_until(lambda: not job_pids(prepared.root), 5, "the job and its children to die")
    assert not job_alive(prepared.root, pid)
    assert not lock_is_held(prepared.paths.lock)


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under `root` with its bytes and mtime, to show a run dir was left alone."""
    files = (path for path in sorted(root.rglob("*")) if path.is_file())
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) for p in files}


def test_check_14_reboot_with_a_running_row_and_no_scope_recovers_uncharged_and_runs_the_pair_again(
    tmp_path: Path, fleet: Fleet
) -> None:
    prepared = prepare(tmp_path)
    first = fleet.spawn(prepared)
    old = wait_until(lambda: running_job(prepared), 15, "the first run to be running")
    old_dir = prepared.paths.runs_dir / old.run_id
    power_loss(prepared, first, old.run_id)
    assert [r.status for r in runs(prepared)] == ["running"]  # the database still believes in it
    before = snapshot(old_dir)
    assert "out/started.json" in before

    second = fleet.spawn(prepared, list_scopes=False)
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
    assert calls[0][:3] == ["systemctl", "--user", "list-units"]  # it asked, and systemd forgot
    assert stops(calls) == []
    assert stops(host_calls(prepared, first.proc.pid)) == []

    started = job_started(prepared, new.run_id)
    assert started is not None and new.run_id != old.run_id
    assert (started["ollama"], started["model"]) == (WS_URL, MODEL)
    assert started["run_dir"] == str(prepared.paths.runs_dir / new.run_id / "out")
    assert started["run_dir"] != str(old_dir / "out")
    assert snapshot(old_dir) == before

    log = events(prepared)
    recovers = [e for e in log if e.kind == "recover"]
    starts = [e for e in log if e.kind == "dispatcher-start"]
    run_starts = [e for e in log if e.kind == "run-start"]
    assert [e.detail for e in recovers] == ["", f"{PAIR} {MODEL}"]
    assert [e.detail for e in starts] == ["", ""]
    assert len(run_starts) == 2
    assert recovers[1].id < starts[1].id < run_starts[1].id
    assert run_starts[1].detail.split()[0] == new.run_id
    assert [e for e in log if e.kind in FAILURES] == [], second.stderr()


def test_check_14_stale_heartbeat_stop_and_run_now_from_before_the_reboot_are_dropped_and_not_acted_on(
    tmp_path: Path, fleet: Fleet
) -> None:
    prepared = prepare(tmp_path)
    first = fleet.spawn(prepared)
    old = wait_until(lambda: running_job(prepared), 15, "the first run to be running")
    power_loss(prepared, first, old.run_id)
    stale_beat = datetime.now(UTC) - timedelta(hours=1)
    # The ghost pair is unlisted: a dispatcher that acted on the run-now would leave a `refused` event.
    with closing(open_state(prepared.paths)) as state:
        state.heartbeat(stale_beat)
        state.request_stop("graceful")
        state.request_run_now(GHOST, MODEL)

    second = fleet.spawn(prepared, list_scopes=False)
    wait_until(lambda: running_job(prepared, {old.run_id}), 15, "the pair to run again")

    log = events(prepared)
    starts = [e for e in log if e.kind == "dispatcher-start"]
    assert len(starts) == 2 and starts[0].detail == ""
    assert starts[1].detail.startswith("dropped stale")
    assert "graceful" in starts[1].detail and GHOST in starts[1].detail
    assert read_state(prepared, lambda state: state.stop_requested()) is None
    assert read_state(prepared, lambda state: state.run_now_requested()) is None
    beat = read_state(prepared, lambda state: state.last_heartbeat())
    assert beat is not None and datetime.now(UTC) - beat < timedelta(seconds=30)
    assert second.proc.poll() is None  # a stop request acted on would have ended it
    assert [e for e in log if e.kind in FAILURES] == [], second.stderr()
    assert stops(host_calls(prepared, second.proc.pid)) == []


def test_check_14_a_second_power_loss_in_a_row_charges_the_job_once_and_the_pair_still_runs(
    tmp_path: Path, fleet: Fleet
) -> None:
    prepared = prepare(tmp_path)
    first = fleet.spawn(prepared)
    one = wait_until(lambda: running_job(prepared), 15, "run 1")
    power_loss(prepared, first, one.run_id)

    second = fleet.spawn(prepared, list_scopes=False)
    two = wait_until(lambda: running_job(prepared, {one.run_id}), 15, "run 2")
    row = job(prepared)
    assert row is not None and (row.attempts, row.crash_streak) == (0, 1)
    power_loss(prepared, second, two.run_id)

    # STOP keeps the third dispatcher from placing anything, so the recovered row can be read as pending
    prepared.paths.stop_file.touch()
    third = fleet.spawn(prepared, list_scopes=False)
    wait_until(lambda: idle_reason(prepared) == "stopped", 15, "the third dispatcher to idle")
    row = job(prepared)
    assert row is not None
    assert (row.status, row.attempts, row.crash_streak) == ("pending", 1, 2)
    assert [(r.run_id, r.status) for r in runs(prepared)] == [
        (one.run_id, "crashed"),
        (two.run_id, "crashed"),
    ]

    prepared.paths.stop_file.unlink()
    three = wait_until(lambda: running_job(prepared, {one.run_id, two.run_id}), 15, "run 3")
    row = job(prepared)
    assert row is not None
    assert (row.status, row.attempts, row.crash_streak) == ("running", 1, 2)
    assert len({one.run_id, two.run_id, three.run_id}) == 3
    assert stops(host_calls(prepared)) == []
    recovers = [e.detail for e in events(prepared) if e.kind == "recover"]
    assert recovers == ["", f"{PAIR} {MODEL}", f"{PAIR} {MODEL}"]
    assert [e for e in events(prepared) if e.kind in FAILURES] == [], third.stderr()


def test_check_14_a_run_that_ended_done_between_two_power_losses_resets_the_streak(
    tmp_path: Path, fleet: Fleet
) -> None:
    ready = {"ready": {"every": "1s", "check": ["./ready"]}}
    prepared = prepare(tmp_path, SECOND_ENDS, fields=ready)
    first = fleet.spawn(prepared)
    one = wait_until(lambda: running_job(prepared), 15, "run 1")
    power_loss(prepared, first, one.run_id)

    second = fleet.spawn(prepared, list_scopes=False)
    two = wait_until(
        lambda: next((r for r in runs(prepared) if r.status == "done"), None), 20, "run 2 to end"
    )
    three = wait_until(lambda: running_job(prepared, {one.run_id, two.run_id}), 20, "run 3")
    row = job(prepared)
    assert row is not None
    assert (row.status, row.attempts, row.crash_streak) == ("running", 0, 0)
    power_loss(prepared, second, three.run_id)

    prepared.paths.stop_file.touch()
    third = fleet.spawn(prepared, list_scopes=False)
    wait_until(lambda: idle_reason(prepared) == "stopped", 15, "the third dispatcher to idle")
    row = job(prepared)
    assert row is not None
    assert (row.status, row.attempts, row.crash_streak) == ("pending", 0, 1)
    assert [r.status for r in runs(prepared)] == ["crashed", "done", "crashed"]

    prepared.paths.stop_file.unlink()
    four = wait_until(
        lambda: running_job(prepared, {one.run_id, two.run_id, three.run_id}), 20, "run 4"
    )
    assert four.run_id not in (one.run_id, two.run_id, three.run_id)
    assert stops(host_calls(prepared)) == []
    assert [e for e in events(prepared) if e.kind in FAILURES] == [], third.stderr()
