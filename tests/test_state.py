"""Pins the SQLite state: schema version refusal, read-only access, idle, stop and contract bookkeeping, the lock, reset.

Depends on: pastore.state, tests.fakes and pytest only.
"""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pastore.state import (
    DispatcherLock,
    JobRow,
    LockHeld,
    RunRow,
    StateError,
    StatePaths,
    default_paths,
    lock_is_held,
    open_state,
    reset,
)
from tests.fakes import FakeClock

T0 = datetime(2026, 10, 25, 1, 0, tzinfo=UTC)


@pytest.fixture
def paths(tmp_path: Path) -> StatePaths:
    return default_paths(tmp_path)


def test_default_paths_keep_outputs_outside_the_state_dir(tmp_path: Path) -> None:
    p = default_paths(tmp_path)
    assert p.db == tmp_path / ".local/state/pastore/state.db"
    assert p.stop_file == p.state_dir / "STOP" and p.lock == p.state_dir / "dispatcher.lock"
    assert p.runs_dir == tmp_path / ".local/share/pastore/runs"
    assert p.state_dir not in p.runs_dir.parents


def test_open_creates_wal_schema_and_reopens(paths: StatePaths) -> None:
    state = open_state(paths)
    state.close()
    conn = sqlite3.connect(paths.db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"jobs", "runs", "events", "contracts", "meta"} <= tables
    conn.close()
    open_state(paths).close()


@pytest.mark.parametrize("readonly", [False, True])
def test_invariant_schema_mismatch_refuses_to_start(paths: StatePaths, readonly: bool) -> None:
    open_state(paths).close()
    conn = sqlite3.connect(paths.db)
    conn.execute("PRAGMA user_version=99")
    conn.close()
    with pytest.raises(StateError, match="schema version 99"):
        open_state(paths, readonly=readonly)


def test_unversioned_database_with_tables_is_refused(paths: StatePaths) -> None:
    paths.state_dir.mkdir(parents=True)
    conn = sqlite3.connect(paths.db)
    conn.execute("CREATE TABLE stray(x)")
    conn.close()
    with pytest.raises(StateError, match="schema version 0"):
        open_state(paths)


def test_readonly_without_state_says_no_state_yet(paths: StatePaths) -> None:
    with pytest.raises(StateError, match="no state yet"):
        open_state(paths, readonly=True)
    assert not paths.state_dir.exists()


def test_readonly_reads_while_a_writer_holds_the_db_and_refuses_writes(paths: StatePaths) -> None:
    writer = open_state(paths)
    writer.heartbeat(T0)
    writer.set_idle_reason(T0, "outside-window")
    reader = open_state(paths, readonly=True)
    assert reader.last_heartbeat() == T0 and reader.idle_reason() == "outside-window"
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.heartbeat(T0)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.request_stop("now")
    reader.close()
    writer.close()


def test_idle_reason_logs_an_event_only_on_change(paths: StatePaths) -> None:
    state = open_state(paths)
    state.set_idle_reason(T0, "outside-window")
    state.set_idle_reason(T0, "outside-window")
    state.set_idle_reason(T0, "gpu-foreign:blender")
    events = state.recent_events(10)
    assert [(e.kind, e.detail) for e in events] == [
        ("idle", "gpu-foreign:blender"),
        ("idle", "outside-window"),
    ]
    state.clear_idle_reason()
    assert state.idle_reason() is None
    for bad in ("sleepy", "gpu-foreign", "gpu-foreign:", "stopped:x"):
        with pytest.raises(ValueError):
            state.set_idle_reason(T0, bad)
    state.close()


def test_events_validate_kind_and_naive_times_are_refused(paths: StatePaths) -> None:
    state = open_state(paths)
    state.record_event(T0, "refused", "torra/bench", "no contract")
    state.record_event(T0, "dispatcher-start")
    with pytest.raises(ValueError, match="unknown event kind"):
        state.record_event(T0, "bogus")
    with pytest.raises(ValueError, match="timezone-aware"):
        state.heartbeat(datetime(2026, 1, 1))
    assert len(state.recent_events(1)) == 1 and len(state.recent_events(10)) == 2
    state.close()


def test_contract_pass_is_per_fingerprint(paths: StatePaths) -> None:
    state = open_state(paths)
    assert not state.has_contract_pass("torra/bench", "fp1")
    state.record_contract_pass("torra/bench", "fp1", T0)
    assert state.has_contract_pass("torra/bench", "fp1")
    assert not state.has_contract_pass("torra/bench", "fp2")
    assert not state.has_contract_pass("torra/other", "fp1")
    state.close()


def test_job_and_run_rows_round_trip_and_checks_hold(paths: StatePaths) -> None:
    state = open_state(paths)
    job = JobRow("torra/bench", "fp", "running", attempts=2, ready_since="x")
    state.upsert_job(job)
    assert state.get_job("torra/bench") == job and state.get_job("nope") is None
    with pytest.raises(sqlite3.IntegrityError):
        state.upsert_job(JobRow("torra/bad", "fp", "weird"))
    run = RunRow("r1", "torra/bench", "fp", "running", "2026-10-25T01:00:00+00:00")
    state.insert_run(run)
    done = replace(run, status="ok", ended_at="end", exit_code=0, detail="fine")
    state.update_run(done)
    assert state.get_run("r1") == done and state.last_run("torra/bench") == done
    assert state.runs_with_status("ok") == [done] and state.runs_with_status("running") == []
    with pytest.raises(StateError, match="no run"):
        state.update_run(RunRow("ghost", "x", "fp", "ok", "t"))
    state.close()


def test_stop_request_round_trip(paths: StatePaths) -> None:
    state = open_state(paths)
    assert state.stop_requested() is None
    state.request_stop("now")
    assert state.stop_requested() == "now"
    with pytest.raises(ValueError):
        state.request_stop("later")  # type: ignore[arg-type]
    state.clear_stop()
    assert state.stop_requested() is None
    state.close()


def test_rerun_and_ack(paths: StatePaths) -> None:
    state = open_state(paths)
    state.upsert_job(JobRow("torra/bench", "fp", "failed", attempts=3, crash_streak=2))
    state.rerun("torra/bench")
    assert state.get_job("torra/bench") == JobRow("torra/bench", "fp", "pending")
    with pytest.raises(StateError, match="no job"):
        state.rerun("ghost/x")
    state.insert_run(RunRow("r1", "torra/bench", "fp", "awaiting-review", "t0"))
    state.ack("r1", T0)
    run = state.get_run("r1")
    job = state.get_job("torra/bench")
    assert run is not None and run.status == "done" and job is not None and job.status == "done"
    assert state.recent_events(1)[0].kind == "done"
    with pytest.raises(StateError, match="not awaiting review"):
        state.ack("r1", T0)
    with pytest.raises(StateError, match="no such run"):
        state.ack("nope", T0)
    state.close()


def test_heartbeat_round_trip_with_fake_clock(paths: StatePaths) -> None:
    clock = FakeClock(T0)
    state = open_state(paths)
    assert state.last_heartbeat() is None
    state.heartbeat(clock())
    clock.advance(30)
    assert state.last_heartbeat() == T0
    state.close()


def test_second_lock_on_the_same_file_raises_lock_held(paths: StatePaths) -> None:
    with DispatcherLock(paths.lock):
        assert lock_is_held(paths.lock)
        assert paths.lock.read_text().strip().isdigit()
        with pytest.raises(LockHeld), DispatcherLock(paths.lock):
            pass
    assert not lock_is_held(paths.lock)
    with DispatcherLock(paths.lock):
        pass


def test_lock_is_held_is_false_for_a_missing_file(paths: StatePaths) -> None:
    assert not lock_is_held(paths.lock)


def test_reset_never_touches_outputs_or_stop(paths: StatePaths) -> None:
    open_state(paths).close()
    state = open_state(paths)
    state.heartbeat(T0)
    paths.stop_file.write_text("")
    out = paths.runs_dir / "20261025T010000-1" / "out"
    out.mkdir(parents=True)
    (out / "results.json").write_text("{}")
    deleted = reset(paths)
    state.close()
    assert paths.db in deleted and not paths.db.exists()
    assert not paths.db.with_name("state.db-wal").exists()
    assert paths.stop_file.exists() and (out / "results.json").read_text() == "{}"
    assert reset(paths) == []


def test_reset_refuses_while_the_dispatcher_lock_is_held(paths: StatePaths) -> None:
    open_state(paths).close()
    with DispatcherLock(paths.lock), pytest.raises(LockHeld):
        reset(paths)
    assert paths.db.exists()


def test_ack_of_a_run_from_an_older_manifest_entry_leaves_the_job_alone(paths: StatePaths) -> None:
    state = open_state(paths)
    state.insert_run(
        RunRow("r1", "torra/bench", "fp-old", "awaiting-review", "2026-10-24T01:00:00+00:00")
    )
    state.upsert_job(JobRow("torra/bench", "fp-new", "pending"))
    state.ack("r1", T0)
    run, job = state.get_run("r1"), state.get_job("torra/bench")
    assert run is not None and run.status == "done"
    assert job is not None and job.status == "pending"
