"""R34 "a write cut mid-transaction": SIGKILLed State writers leave state.db whole, never half-acked or half-created.

Depends on: pastore.state, pytest and stdlib only (real child processes under a tmp HOME, never a real unit).
"""

import contextlib
import itertools
import os
import random
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pastore.state import (
    SCHEMA_VERSION,
    State,
    StateError,
    StatePaths,
    default_paths,
    open_state,
)

WT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 31, 1, 0, tzinfo=UTC)
DEADLINE = 20.0
EVENT_LIMIT = 100_000
ROWS = 2000
TABLES = frozenset({"contracts", "events", "jobs", "meta", "runs"})

WRITER = r"""
import sys, time
from datetime import UTC, datetime
from pathlib import Path
from pastore.state import JobRow, RunRow, State, default_paths, open_state

home, progress, first, hang = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
state = open_state(default_paths(home))
out = open(progress, "a", buffering=1)


def say(line):
    out.write(line + "\n")
    out.flush()


if hang != "no":
    real = State._event

    def stuck(self, now, kind, ref, detail):
        if kind == "done" and hang == "before":
            say("in-ack")
            time.sleep(120)
        real(self, now, kind, ref, detail)
        if kind == "done" and hang == "after":
            say("in-ack")
            time.sleep(120)

    State._event = stuck
say("ready")
now = datetime(2026, 10, 31, 1, 0, tzinfo=UTC)
n = first
stop = time.monotonic() + 60
while time.monotonic() < stop:
    run_id, ref = f"r{n}", f"torra/job{n}"
    state.upsert_job(JobRow(ref, "m", f"fp{n}", "pending"))
    state.insert_run(RunRow(run_id, ref, "m", "workstation", f"fp{n}", "awaiting-review", "t0"))
    state.record_event(now, "awaiting-review", ref, run_id)
    state.ack(run_id, now)
    state.record_event(now, "run-end", None, run_id)
    say(str(n))
    n += 1
"""

CREATOR = r"""
import sys, time
from pathlib import Path
from pastore.state import default_paths, open_state

out = open(sys.argv[2], "w", buffering=1)
out.write("opening\n")
out.flush()
open_state(default_paths(Path(sys.argv[1])))
out.write("created\n")
out.flush()
time.sleep(120)
"""


HOT_JOURNAL = r"""
import sqlite3, sys, time
from pathlib import Path
from pastore.state import default_paths

conn = sqlite3.connect(default_paths(Path(sys.argv[1])).db)
conn.execute("PRAGMA journal_mode=DELETE")
conn.execute("PRAGMA cache_size=5")
conn.execute("BEGIN")
for i in range(int(sys.argv[4])):
    conn.execute(sys.argv[3], ("x" * 200 + str(i),))
out = open(sys.argv[2], "w", buffering=1)
out.write("ready\n")
out.flush()
time.sleep(120)
"""


@dataclass
class Child:
    """A started child process, its log file and its progress file."""

    proc: subprocess.Popen[bytes]
    log: Path
    progress: Path


Spawn = Callable[..., Child]


@pytest.fixture
def spawn(tmp_path: Path) -> Iterator[Spawn]:
    """Start a python child in its own session; every child is SIGKILLed and reaped at teardown."""
    started: list[Child] = []
    numbers = itertools.count()

    def start(script: str, home: Path, progress: Path, *extra: str) -> Child:
        log = tmp_path / f"child{next(numbers)}.log"
        env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        with log.open("wb") as sink:
            proc = subprocess.Popen(
                [sys.executable, "-c", script, str(home), str(progress), *extra],
                cwd=WT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=sink,
                start_new_session=True,
            )
        child = Child(proc, log, progress)
        started.append(child)
        return child

    yield start
    for child in started:
        if child.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.proc.pid, signal.SIGKILL)
        child.proc.wait(timeout=10)


def _lines(progress: Path) -> list[str]:
    text = progress.read_text() if progress.exists() else ""
    return text.split("\n")[:-1]


def _completed(progress: Path) -> list[int]:
    return [int(line) for line in _lines(progress) if line.isdigit()]


def _poll(child: Child, what: str, ready: Callable[[], bool]) -> None:
    deadline = time.monotonic() + DEADLINE
    while not ready():
        if child.proc.poll() is not None and not ready():
            pytest.fail(
                f"child exited {child.proc.returncode} before {what}:\n{child.log.read_text()}"
            )
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.0002)


def _wait_for_line(child: Child, line: str) -> None:
    _poll(child, f"line {line!r}", lambda: line in _lines(child.progress))


def _wait_for_acks(child: Child, count: int) -> None:
    _poll(child, f"{count} completed iterations", lambda: len(_completed(child.progress)) >= count)


def _sigkill(child: Child) -> None:
    os.killpg(child.proc.pid, signal.SIGKILL)
    assert child.proc.wait(timeout=10) == -signal.SIGKILL


def _raw(paths: StatePaths, sql: str) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(paths.db)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _assert_integrity(paths: StatePaths) -> None:
    assert _raw(paths, "PRAGMA integrity_check") == [("ok",)]


def _check_acks(state: State, completed: list[int]) -> int:
    """Assert every ack is whole or absent; return the next unused run index."""
    done = state.runs_with_status("done")
    waiting = state.runs_with_status("awaiting-review")
    events = [e for e in state.recent_events(EVENT_LIMIT) if e.kind == "done"]
    assert all(e.detail.startswith("acked ") for e in events)
    acked = [e.detail.removeprefix("acked ") for e in events]
    assert sorted(acked) == sorted(r.run_id for r in done)
    for run in done:
        job = state.get_job(run.ref, run.model)
        assert job is not None and job.status == "done", f"{run.run_id} done but its job is not"
    assert {(j.ref, j.model) for j in state.jobs_with_status("done")} == {
        (r.ref, r.model) for r in done
    }
    for run in waiting:
        job = state.get_job(run.ref, run.model)
        assert job is not None and job.status == "pending", f"{run.run_id} half-acked"
    assert {f"r{n}" for n in completed} <= {r.run_id for r in done}
    return max([-1, *(int(r.run_id[1:]) for r in (*done, *waiting))]) + 1


def _verify(paths: StatePaths, state: State, completed: list[int]) -> int:
    """Check a reopened database, redo every cut ack, and return the next unused run index."""
    _assert_integrity(paths)
    assert {row[0] for row in _raw(paths, "SELECT DISTINCT status FROM runs")} <= {
        "done",
        "awaiting-review",
    }
    _check_acks(state, completed)
    for run in state.runs_with_status("awaiting-review"):
        state.ack(run.run_id, NOW)
    _assert_integrity(paths)
    return _check_acks(state, completed)


def test_writers_killed_at_varied_moments_leave_a_whole_database(
    spawn: Spawn, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    paths = default_paths(home)
    rng = random.Random(34)
    first = 0
    for number, threshold in enumerate((0, 0, 1, 2, 4, 8, 16, 30, 55, 90)):
        progress = tmp_path / f"progress{number}.txt"
        writer = spawn(WRITER, home, progress, str(first), "no")
        _wait_for_line(writer, "ready")
        _wait_for_acks(writer, threshold)
        time.sleep(rng.uniform(0, 0.005))
        _sigkill(writer)
        state = open_state(paths)
        first = _verify(paths, state, _completed(progress))
        state.close()
    reopened = open_state(paths)
    assert len(reopened.runs_with_status("done")) == first
    reopened.close()


@pytest.mark.parametrize("hang", ["before", "after"])
def test_a_cut_ack_leaves_the_run_awaiting_review_and_can_be_redone(
    spawn: Spawn, tmp_path: Path, hang: str
) -> None:
    home = tmp_path / "home"
    paths = default_paths(home)
    progress = tmp_path / "progress.txt"
    writer = spawn(WRITER, home, progress, "0", hang)
    _wait_for_line(writer, "in-ack")
    _sigkill(writer)
    state = open_state(paths)
    _assert_integrity(paths)
    run = state.get_run("r0")
    job = state.get_job("torra/job0", "m")
    assert run is not None and run.status == "awaiting-review"
    assert job is not None and job.status == "pending"
    assert [e for e in state.recent_events(EVENT_LIMIT) if e.kind == "done"] == []
    assert _verify(paths, state, []) == 1
    redone = state.get_run("r0")
    assert redone is not None and redone.status == "done"
    state.close()


def test_a_held_read_snapshot_stays_whole_while_the_writer_is_killed(
    spawn: Spawn, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    paths = default_paths(home)
    progress = tmp_path / "progress.txt"
    writer = spawn(WRITER, home, progress, "0", "no")
    _wait_for_acks(writer, 3)
    before = _completed(progress)
    reader = open_state(paths, readonly=True)
    reader._conn.execute("BEGIN")
    held = len(reader.runs_with_status("done"))
    assert held >= len(before)
    _wait_for_acks(writer, len(before) + 5)
    _sigkill(writer)
    assert len(reader.runs_with_status("done")) == held
    _check_acks(reader, before)
    assert reader._conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    _assert_integrity(paths)
    reader._conn.execute("ROLLBACK")
    reader.close()
    state = open_state(paths)
    assert _verify(paths, state, _completed(progress)) > held
    state.close()


def _assert_complete(paths: StatePaths) -> None:
    """The database holds exactly the version 2 tables and passes the integrity check."""
    assert _raw(paths, "PRAGMA user_version") == [(SCHEMA_VERSION,)]
    names = _raw(
        paths,
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    )
    assert {row[0] for row in names} == TABLES
    _assert_integrity(paths)


def test_a_creation_cut_leaves_an_empty_or_a_complete_database(
    spawn: Spawn, tmp_path: Path
) -> None:
    for number in range(24):
        home = tmp_path / f"home{number}"
        paths = default_paths(home)
        creator = spawn(CREATOR, home, tmp_path / f"creator{number}.txt")
        _wait_for_line(creator, "opening")
        time.sleep(number * 0.0006)
        _sigkill(creator)
        state = open_state(paths)
        _assert_complete(paths)
        state.record_event(NOW, "dispatcher-start", None, "after a cut creation")
        assert [e.kind for e in state.recent_events(5)] == ["dispatcher-start"]
        state.close()


def _kill_in_spilled_transaction(spawn: Spawn, home: Path, progress: Path, update: str) -> Path:
    """Kill a child whose rollback-journal transaction rewrote every committed row and spilled; return the journal."""
    paths = default_paths(home)
    size_before = paths.db.stat().st_size
    child = spawn(HOT_JOURNAL, home, progress, update, str(ROWS))
    _wait_for_line(child, "ready")
    _sigkill(child)
    journal = paths.db.with_name(paths.db.name + "-journal")
    assert journal.stat().st_size > 0
    assert paths.db.stat().st_size > size_before
    return journal


def test_a_hot_rollback_journal_is_rolled_back_by_open_state(spawn: Spawn, tmp_path: Path) -> None:
    home = tmp_path / "home"
    paths = default_paths(home)
    state = open_state(paths)
    for i in range(ROWS):
        state.record_event(NOW, "dispatcher-start", None, f"committed {i}")
    state.close()
    update = (
        "UPDATE events SET detail = 'UNCOMMITTED ' || ?1 "
        "WHERE id = CAST(substr(?1, 201) AS INTEGER) + 1"
    )
    journal = _kill_in_spilled_transaction(spawn, home, tmp_path / "progress.txt", update)
    reopened = open_state(paths)
    assert not journal.exists()
    _assert_complete(paths)
    assert _raw(paths, "PRAGMA journal_mode") == [("wal",)]
    details = [e.detail for e in reopened.recent_events(EVENT_LIMIT)]
    assert sorted(details) == sorted(f"committed {i}" for i in range(ROWS))
    reopened.close()


def test_a_v1_database_with_a_hot_journal_is_refused_and_keeps_its_committed_rows(
    spawn: Spawn, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    paths = default_paths(home)
    paths.state_dir.mkdir(parents=True)
    conn = sqlite3.connect(paths.db)
    conn.execute("CREATE TABLE jobs(ref TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)")
    conn.executemany("INSERT INTO jobs VALUES(?, 'fp')", [(f"committed/{i}",) for i in range(ROWS)])
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()
    update = (
        "UPDATE jobs SET fingerprint = 'UNCOMMITTED' || ?1 "
        "WHERE ref = 'committed/' || CAST(substr(?1, 201) AS INTEGER)"
    )
    _kill_in_spilled_transaction(spawn, home, tmp_path / "progress.txt", update)
    with pytest.raises(StateError, match="schema version 1, this pastore needs 2"):
        open_state(paths)
    _assert_integrity(paths)
    assert _raw(paths, "SELECT COUNT(*) FROM jobs") == [(ROWS,)]
    assert _raw(paths, "SELECT DISTINCT fingerprint FROM jobs") == [("fp",)]
    assert _raw(paths, "PRAGMA user_version") == [(1,)]
    assert _raw(paths, "PRAGMA journal_mode") == [("delete",)]
