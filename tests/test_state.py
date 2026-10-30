"""Pins the SQLite state: schema 2 and its refusals, task-plus-model keys, read-only access, bookkeeping, the lock, reset.

Depends on: pastore.state, tests.fakes and pytest only.
"""

import json
import shutil
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import pastore.state as state_module
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
REF = "torra/bench"
HOST = "workstation"


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
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"jobs", "runs", "events", "contracts", "meta"} <= tables
    conn.close()
    open_state(paths).close()


def test_schema_2_has_model_and_host_columns_and_a_ref_model_key(paths: StatePaths) -> None:
    open_state(paths).close()
    conn = sqlite3.connect(paths.db)
    jobs = conn.execute("PRAGMA table_info(jobs)").fetchall()
    runs = conn.execute("PRAGMA table_info(runs)").fetchall()
    conn.close()
    assert [c[1] for c in jobs[:3]] == ["ref", "model", "fingerprint"]
    assert [c[1] for c in runs[:6]] == ["run_id", "ref", "model", "host", "fingerprint", "status"]
    assert {c[1] for c in jobs if c[5]} == {"ref", "model"}


def _make_v1_database(paths: StatePaths, scratch: Path, *, wal_frames_left: bool) -> None:
    """A v1 state.db in WAL mode; with `wal_frames_left` its -wal still holds frames, as after a crash."""
    paths.state_dir.mkdir(parents=True)
    scratch.mkdir()
    live = (scratch if wal_frames_left else paths.state_dir) / "state.db"
    conn = sqlite3.connect(live)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE jobs(ref TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)")
    conn.execute("PRAGMA user_version=1")
    if wal_frames_left:
        wal = live.with_name("state.db-wal")
        assert wal.stat().st_size > 0
        shutil.copy(wal, paths.db.with_name("state.db-wal"))
        shutil.copy(live, paths.db)
    conn.close()


def _snapshot(paths: StatePaths) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in paths.state_dir.iterdir()}


@pytest.mark.parametrize("wal_frames_left", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
def test_a_v1_database_is_refused_and_left_untouched(
    paths: StatePaths, tmp_path: Path, readonly: bool, wal_frames_left: bool
) -> None:
    _make_v1_database(paths, tmp_path / "scratch", wal_frames_left=wal_frames_left)
    before = _snapshot(paths)
    with pytest.raises(StateError, match="schema version 1, this pastore needs 2"):
        open_state(paths, readonly=readonly)
    after = _snapshot(paths)
    assert after["state.db"] == before["state.db"]
    assert after.get("state.db-wal", b"") == before.get("state.db-wal", b"")
    assert set(after) <= {"state.db", "state.db-wal", "state.db-shm"}


@pytest.mark.parametrize("version", [1, 3, 99])
@pytest.mark.parametrize("readonly", [False, True])
def test_invariant_schema_mismatch_refuses_to_start(
    paths: StatePaths, readonly: bool, version: int
) -> None:
    open_state(paths).close()
    conn = sqlite3.connect(paths.db)
    conn.execute(f"PRAGMA user_version={version}")
    conn.close()
    with pytest.raises(StateError, match=f"schema version {version}"):
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


def test_readonly_of_an_existing_empty_database_says_no_state_yet(paths: StatePaths) -> None:
    paths.state_dir.mkdir(parents=True)
    paths.db.write_bytes(b"")
    with pytest.raises(StateError, match="no state yet"):
        open_state(paths, readonly=True)
    assert paths.db.read_bytes() == b""


@pytest.mark.parametrize(
    ("name", "broken"),
    [
        ("SCHEMA", state_module.SCHEMA + "CREATE TABLE jobs(x);"),
        ("SCHEMA_VERSION", "2;SELECT * FROM missing"),
    ],
    ids=["fails-after-the-tables", "fails-after-the-version"],
)
def test_schema_creation_is_one_transaction(
    paths: StatePaths, monkeypatch: pytest.MonkeyPatch, name: str, broken: str
) -> None:
    monkeypatch.setattr(state_module, name, broken)
    with pytest.raises(sqlite3.OperationalError, match="already exists|no such table"):
        open_state(paths)
    conn = sqlite3.connect(paths.db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
    conn.close()
    monkeypatch.undo()
    open_state(paths).close()


def test_version_check_reads_the_version_and_the_tables_in_one_statement(paths: StatePaths) -> None:
    open_state(paths).close()
    conn = sqlite3.connect(paths.db)
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    assert state_module._check_version(conn, paths) is False
    assert len([sql for sql in seen if not sql.startswith("--")]) == 1
    conn.close()


def test_a_reset_between_the_probe_and_the_open_still_creates_the_schema(
    paths: StatePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_state(paths).close()
    real_check = state_module._check_version
    probed: list[bool] = []

    def check_then_reset(conn: sqlite3.Connection, p: StatePaths) -> bool:
        result = real_check(conn, p)
        if not probed:
            probed.append(True)
            reset(p)
        return result

    monkeypatch.setattr(state_module, "_check_version", check_then_reset)
    state = open_state(paths)
    state.heartbeat(T0)
    assert state.last_heartbeat() == T0
    state.close()
    assert probed


def test_readonly_reads_while_a_writer_holds_the_db_and_refuses_writes(paths: StatePaths) -> None:
    writer = open_state(paths)
    writer.heartbeat(T0)
    writer.set_idle_reason(T0, "no-ready-task")
    reader = open_state(paths, readonly=True)
    assert reader.last_heartbeat() == T0 and reader.idle_reason() == "no-ready-task"
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.heartbeat(T0)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.request_stop("now")
    reader.close()
    writer.close()


def test_idle_reason_logs_an_event_only_on_change(paths: StatePaths) -> None:
    state = open_state(paths)
    state.set_idle_reason(T0, "no-ready-task")
    state.set_idle_reason(T0, "no-ready-task")
    state.set_idle_reason(T0, "gpu-foreign:blender")
    events = state.recent_events(10)
    assert [(e.kind, e.detail) for e in events] == [
        ("idle", "gpu-foreign:blender"),
        ("idle", "no-ready-task"),
    ]
    state.clear_idle_reason()
    assert state.idle_reason() is None
    for bad in ("sleepy", "outside-window", "gpu-foreign", "gpu-foreign:", "stopped:x"):
        with pytest.raises(ValueError):
            state.set_idle_reason(T0, bad)
    state.close()


def test_events_validate_kind_and_naive_times_are_refused(paths: StatePaths) -> None:
    state = open_state(paths)
    state.record_event(T0, "refused", REF, "no contract")
    state.record_event(T0, "dispatcher-start")
    with pytest.raises(ValueError, match="unknown event kind"):
        state.record_event(T0, "bogus")
    with pytest.raises(ValueError, match="timezone-aware"):
        state.heartbeat(datetime(2026, 1, 1))
    assert len(state.recent_events(1)) == 1 and len(state.recent_events(10)) == 2
    state.close()


def test_contract_pass_is_per_fingerprint(paths: StatePaths) -> None:
    state = open_state(paths)
    assert not state.has_contract_pass(REF, "fp1")
    state.record_contract_pass(REF, "fp1", T0)
    assert state.has_contract_pass(REF, "fp1")
    assert not state.has_contract_pass(REF, "fp2")
    assert not state.has_contract_pass("torra/other", "fp1")
    state.close()


def test_job_and_run_rows_round_trip_and_checks_hold(paths: StatePaths) -> None:
    state = open_state(paths)
    job = JobRow(REF, "m-a", "fp", "running", attempts=2, ready_since="x")
    state.upsert_job(job)
    assert state.get_job(REF, "m-a") == job and state.get_job("nope", "m-a") is None
    with pytest.raises(sqlite3.IntegrityError):
        state.upsert_job(JobRow("torra/bad", "m-a", "fp", "weird"))
    run = RunRow("r1", REF, "m-a", HOST, "fp", "running", "2026-10-25T01:00:00+00:00")
    state.insert_run(run)
    assert state.get_run("r1") == run
    done = replace(
        run, model="m-b", host="executor", status="ok", ended_at="end", exit_code=0, detail="fine"
    )
    state.update_run(done)
    assert state.get_run("r1") == done and state.last_run(REF, "m-b") == done
    assert state.last_run(REF, "m-a") is None
    assert state.runs_with_status("ok") == [done] and state.runs_with_status("running") == []
    with pytest.raises(StateError, match="no run"):
        state.update_run(RunRow("ghost", "x", "m-a", HOST, "fp", "ok", "t"))
    state.close()


@pytest.mark.parametrize("column", ["model", "host"])
def test_a_run_without_model_or_host_is_refused_by_the_database(
    paths: StatePaths, column: str
) -> None:
    state = open_state(paths)
    run = RunRow("r1", REF, "m-a", HOST, "fp", "running", "t")
    with pytest.raises(sqlite3.IntegrityError, match=f"runs.{column}"):
        state.insert_run(replace(run, **{column: None}))
    state.close()


def test_a_job_without_a_model_is_refused_by_the_database(paths: StatePaths) -> None:
    state = open_state(paths)
    with pytest.raises(sqlite3.IntegrityError, match="jobs.model"):
        state.upsert_job(JobRow(REF, None, "fp", "pending"))  # type: ignore[arg-type]
    state.close()


def test_jobs_of_one_ref_and_two_models_coexist(paths: StatePaths) -> None:
    state = open_state(paths)
    a = JobRow(REF, "m-a", "fp", "pending")
    b = JobRow(REF, "m-b", "fp", "failed", attempts=2)
    state.upsert_job(b)
    state.upsert_job(a)
    assert state.get_job(REF, "m-a") == a and state.get_job(REF, "m-b") == b
    assert state.get_job(REF, "m-c") is None
    running = replace(a, status="running", attempts=1)
    state.upsert_job(running)
    assert state.get_job(REF, "m-a") == running and state.get_job(REF, "m-b") == b
    state.close()


def test_jobs_with_status_orders_by_ref_then_model(paths: StatePaths) -> None:
    state = open_state(paths)
    for ref, model in ((REF, "m-b"), (REF, "m-a"), ("torra/alpha", "m-z")):
        state.upsert_job(JobRow(ref, model, "fp", "pending"))
    state.upsert_job(JobRow("torra/aaa", "m-a", "fp", "failed"))
    pairs = [(j.ref, j.model) for j in state.jobs_with_status("pending")]
    assert pairs == [("torra/alpha", "m-z"), (REF, "m-a"), (REF, "m-b")]
    state.close()


def test_last_run_is_per_task_and_model(paths: StatePaths) -> None:
    state = open_state(paths)
    old_a = RunRow("r1", REF, "m-a", HOST, "fp", "ok", "2026-10-25T01:00:00+00:00")
    new_a = RunRow("r2", REF, "m-a", HOST, "fp", "failed", "2026-10-25T02:00:00+00:00")
    only_b = RunRow("r3", REF, "m-b", "executor", "fp", "ok", "2026-10-25T00:00:00+00:00")
    for run in (new_a, only_b, old_a):
        state.insert_run(run)
    assert state.last_run(REF, "m-a") == new_a and state.last_run(REF, "m-b") == only_b
    assert state.last_run(REF, "m-c") is None and state.last_run("torra/other", "m-a") is None
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
    state.upsert_job(JobRow(REF, "m-a", "fp", "failed", attempts=3, crash_streak=2))
    state.rerun(REF, "m-a")
    assert state.get_job(REF, "m-a") == JobRow(REF, "m-a", "fp", "pending")
    with pytest.raises(StateError, match=r"no job ghost/x with model m-a"):
        state.rerun("ghost/x", "m-a")
    with pytest.raises(StateError, match=rf"no job {REF} with model m-z"):
        state.rerun(REF, "m-z")
    state.insert_run(RunRow("r1", REF, "m-a", HOST, "fp", "awaiting-review", "t0"))
    state.ack("r1", T0)
    run = state.get_run("r1")
    job = state.get_job(REF, "m-a")
    assert run is not None and run.status == "done" and job is not None and job.status == "done"
    assert state.recent_events(1)[0].kind == "done"
    with pytest.raises(StateError, match="not awaiting review"):
        state.ack("r1", T0)
    with pytest.raises(StateError, match="no such run"):
        state.ack("nope", T0)
    state.close()


def test_rerun_touches_only_the_matching_model(paths: StatePaths) -> None:
    state = open_state(paths)
    other = JobRow(REF, "m-b", "fp", "failed", attempts=3, crash_streak=2)
    state.upsert_job(JobRow(REF, "m-a", "fp", "failed", attempts=3, crash_streak=2))
    state.upsert_job(other)
    state.rerun(REF, "m-a")
    assert state.get_job(REF, "m-a") == JobRow(REF, "m-a", "fp", "pending")
    assert state.get_job(REF, "m-b") == other
    state.close()


def test_ack_touches_only_the_job_of_the_runs_model(paths: StatePaths) -> None:
    state = open_state(paths)
    state.upsert_job(JobRow(REF, "m-a", "fp", "pending"))
    state.upsert_job(JobRow(REF, "m-b", "fp", "pending"))
    state.insert_run(RunRow("r1", REF, "m-a", HOST, "fp", "awaiting-review", "t0"))
    state.insert_run(RunRow("r2", REF, "m-b", HOST, "fp", "awaiting-review", "t1"))
    state.ack("r1", T0)
    job_a, job_b = state.get_job(REF, "m-a"), state.get_job(REF, "m-b")
    assert job_a is not None and job_a.status == "done"
    assert job_b is not None and job_b.status == "pending"
    run_b = state.get_run("r2")
    assert run_b is not None and run_b.status == "awaiting-review"
    state.close()


def test_run_now_round_trips_a_pair_and_clears(paths: StatePaths) -> None:
    state = open_state(paths)
    assert state.run_now_requested() is None
    state.request_run_now(REF, "m-a")
    assert state.run_now_requested() == (REF, "m-a")
    conn = sqlite3.connect(paths.db)
    raw = conn.execute("SELECT value FROM meta WHERE key='run_now'").fetchone()[0]
    conn.close()
    assert json.loads(raw) == [REF, "m-a"]
    state.request_run_now(REF, "m-b")
    assert state.run_now_requested() == (REF, "m-b")
    state.clear_run_now()
    assert state.run_now_requested() is None
    state.close()


@pytest.mark.parametrize(
    "stored",
    [REF, "not json{", '["a"]', '["a", "b", "c"]', '["a", 1]', '{"a": "b"}', "null", '"ab"'],
)
def test_a_corrupt_run_now_value_raises_state_error(paths: StatePaths, stored: str) -> None:
    open_state(paths).close()
    conn = sqlite3.connect(paths.db)
    with conn:
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('run_now', ?)", (stored,))
    conn.close()
    state = open_state(paths)
    with pytest.raises(StateError, match="run_now"):
        state.run_now_requested()
    state.close()


def test_reset_crash_streaks_spares_exactly_the_listed_pairs(paths: StatePaths) -> None:
    state = open_state(paths)
    pairs = [("torra/a", "m-1"), ("torra/a", "m-2"), ("torra/b", "m-1"), ("torra/b", "m-2")]
    for ref, model in pairs:
        state.upsert_job(JobRow(ref, model, "fp", "failed", crash_streak=2))

    def streaks() -> dict[tuple[str, str], int]:
        rows = (state.get_job(ref, model) for ref, model in pairs)
        return {(j.ref, j.model): j.crash_streak for j in rows if j is not None}

    state.reset_crash_streaks({("torra/a", "m-1"), ("torra/b", "m-2")})
    assert streaks() == {
        ("torra/a", "m-1"): 2,
        ("torra/a", "m-2"): 0,
        ("torra/b", "m-1"): 0,
        ("torra/b", "m-2"): 2,
    }
    state.reset_crash_streaks(set())
    assert set(streaks().values()) == {0}
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
        RunRow("r1", REF, "m-a", HOST, "fp-old", "awaiting-review", "2026-10-24T01:00:00+00:00")
    )
    state.upsert_job(JobRow(REF, "m-a", "fp-new", "pending"))
    state.ack("r1", T0)
    run, job = state.get_run("r1"), state.get_job(REF, "m-a")
    assert run is not None and run.status == "done"
    assert job is not None and job.status == "pending"
