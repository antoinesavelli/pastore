"""SQLite WAL state (jobs, runs, events, contracts, meta), the dispatcher flock, and reset.

Depends on: pastore.spec and stdlib (sqlite3, fcntl, json) only; never reads a clock.
"""

import fcntl
import json
import os
import sqlite3
from dataclasses import astuple, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pastore.spec import TASK_EVENTS, WORKSTATION, idle_reason

SCHEMA_VERSION = 2
DISPATCHER_EVENTS = (
    "idle",
    "dispatcher-start",
    "dispatcher-stop",
    "recover",
    "policy-error",
    "run-start",
    "run-end",
    "tick-error",
    "workload-overlap",
    "workload-ack",
    "runs-pruned",
)
SCHEMA = """
CREATE TABLE jobs(
    ref TEXT NOT NULL, model TEXT NOT NULL, fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','done','failed','blocked')),
    attempts INT NOT NULL, crash_streak INT NOT NULL, ready_since TEXT, last_check_at TEXT,
    last_success_at TEXT, starved_flagged INT NOT NULL, PRIMARY KEY(ref, model));
CREATE TABLE runs(
    run_id TEXT PRIMARY KEY, ref TEXT NOT NULL, model TEXT NOT NULL, host TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running','ok','failed','timeout','stopped','crashed',
        'reader-failed','awaiting-review','done')),
    started_at TEXT NOT NULL, ended_at TEXT, exit_code INT, detail TEXT NOT NULL);
CREATE TABLE events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, ref TEXT, kind TEXT NOT NULL,
    detail TEXT NOT NULL);
CREATE TABLE contracts(
    ref TEXT NOT NULL, fingerprint TEXT NOT NULL, passed_at TEXT NOT NULL,
    PRIMARY KEY(ref, fingerprint));
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class StateError(Exception):
    """The state database is missing, of another schema version, or otherwise unusable."""


class LockHeld(StateError):
    """Another process holds the dispatcher lock."""


@dataclass(frozen=True)
class StatePaths:
    """Where pastore keeps its state (small, resettable) and run outputs (never reset)."""

    state_dir: Path
    db: Path
    lock: Path
    stop_file: Path
    runs_dir: Path


def default_paths(home: Path) -> StatePaths:
    """The standard layout under `home` (R2)."""
    state_dir = home / ".local" / "state" / "pastore"
    return StatePaths(
        state_dir=state_dir,
        db=state_dir / "state.db",
        lock=state_dir / "dispatcher.lock",
        stop_file=state_dir / "STOP",
        runs_dir=home / ".local" / "share" / "pastore" / "runs",
    )


@dataclass(frozen=True)
class JobRow:
    """One row of `jobs`, keyed by task ref and model."""

    ref: str
    model: str
    fingerprint: str
    status: str
    attempts: int = 0
    crash_streak: int = 0
    ready_since: str | None = None
    last_check_at: str | None = None
    last_success_at: str | None = None
    starved_flagged: int = 0


@dataclass(frozen=True)
class RunRow:
    """One row of `runs`."""

    run_id: str
    ref: str
    model: str
    host: str
    fingerprint: str
    status: str
    started_at: str
    ended_at: str | None = None
    exit_code: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class EventRow:
    """One row of `events`."""

    id: int
    at: str
    ref: str | None
    kind: str
    detail: str


def _iso(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return now.astimezone(UTC).isoformat()


def _check_version(conn: sqlite3.Connection, paths: StatePaths) -> bool:
    """Return True for a brand-new database; raise StateError on any other version mismatch."""
    version, tables = conn.execute(
        "SELECT (SELECT user_version FROM pragma_user_version), "
        "(SELECT COUNT(*) FROM sqlite_master)"
    ).fetchone()
    if version == 0 and tables == 0:
        return True
    if version != SCHEMA_VERSION:
        conn.close()
        raise StateError(
            f"{paths.db} has schema version {version}, this pastore needs {SCHEMA_VERSION}; "
            "refusing to start (pastore reset wipes state)"
        )
    return False


def _connect_ro(paths: StatePaths) -> sqlite3.Connection:
    return sqlite3.connect(f"{paths.db.resolve().as_uri()}?mode=ro", uri=True)


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create every table and set the version in one transaction, so a crash leaves it empty or complete."""
    try:
        conn.executescript(f"BEGIN;{SCHEMA}PRAGMA user_version={SCHEMA_VERSION};COMMIT;")
    except sqlite3.Error:
        conn.close()
        raise


def open_state(paths: StatePaths, *, readonly: bool = False) -> "State":
    """Open (and on first use create) the state database; a version mismatch is refused before any write."""
    if readonly:
        if not paths.db.exists():
            raise StateError("no state yet")
        conn = _connect_ro(paths)
        if _check_version(conn, paths):
            conn.close()
            raise StateError("no state yet")
    else:
        paths.state_dir.mkdir(parents=True, exist_ok=True)
        # a read-only connection cannot roll back a hot journal a kill left: the probe would fail forever
        if paths.db.exists() and not paths.db.with_name(paths.db.name + "-journal").exists():
            probe = _connect_ro(paths)
            _check_version(probe, paths)
            probe.close()
        conn = sqlite3.connect(paths.db)
        conn.execute("PRAGMA busy_timeout=5000")
        is_new = _check_version(conn, paths)
        conn.execute("PRAGMA journal_mode=WAL")
        if is_new:
            _create_schema(conn)
    conn.execute("PRAGMA busy_timeout=5000")
    return State(conn)


class State:
    """Typed access to the state database; callers pass timestamps, it never reads a clock."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def _one(self, sql: str, args: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        row: tuple[Any, ...] | None = self._conn.execute(sql, args).fetchone()
        return row

    def _meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key=?", (key,))
        return None if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    def _event(self, now: datetime, kind: str, ref: str | None, detail: str) -> None:
        if kind not in (*TASK_EVENTS, *DISPATCHER_EVENTS):
            raise ValueError(f"unknown event kind {kind!r}")
        self._conn.execute(
            "INSERT INTO events(at, ref, kind, detail) VALUES(?, ?, ?, ?)",
            (_iso(now), ref, kind, detail),
        )

    def heartbeat(self, now: datetime) -> None:
        """Record that the dispatcher loop is alive."""
        with self._conn:
            self._set_meta("heartbeat_at", _iso(now))

    def last_heartbeat(self) -> datetime | None:
        """The last recorded heartbeat, if any."""
        value = self._meta("heartbeat_at")
        return None if value is None else datetime.fromisoformat(value)

    def record_event(
        self, now: datetime, kind: str, ref: str | None = None, detail: str = ""
    ) -> None:
        """Append an event; the kind must be a known one."""
        with self._conn:
            self._event(now, kind, ref, detail)

    def recent_events(self, limit: int) -> list[EventRow]:
        """The newest events first."""
        rows = self._conn.execute(
            "SELECT id, at, ref, kind, detail FROM events ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [EventRow(*row) for row in rows]

    def set_idle_reason(self, now: datetime, reason: str) -> None:
        """Store the idle reason, logging an `idle` event only when it changes."""
        name, _, detail = reason.partition(":")
        idle_reason(name, detail)
        with self._conn:
            if self._meta("idle_reason") != reason:
                self._set_meta("idle_reason", reason)
                self._event(now, "idle", None, reason)

    def clear_idle_reason(self) -> None:
        """Forget the idle reason (a job is running)."""
        with self._conn:
            self._conn.execute("DELETE FROM meta WHERE key='idle_reason'")

    def idle_reason(self) -> str | None:
        """The current idle reason, if any."""
        return self._meta("idle_reason")

    def record_contract_pass(self, ref: str, fingerprint: str, now: datetime) -> None:
        """Record that `check --contract` passed for this task entry."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO contracts(ref, fingerprint, passed_at) VALUES(?, ?, ?)",
                (ref, fingerprint, _iso(now)),
            )

    def has_contract_pass(self, ref: str, fingerprint: str) -> bool:
        """Whether this exact task entry has a recorded contract pass."""
        row = self._one("SELECT 1 FROM contracts WHERE ref=? AND fingerprint=?", (ref, fingerprint))
        return row is not None

    def get_job(self, ref: str, model: str) -> JobRow | None:
        """The job row for this task and model, if any."""
        row = self._one("SELECT * FROM jobs WHERE ref=? AND model=?", (ref, model))
        return None if row is None else JobRow(*row)

    def upsert_job(self, job: JobRow) -> None:
        """Insert or replace the job row of this task and model."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                astuple(job),
            )

    def insert_run(self, run: RunRow) -> None:
        """Insert a new run row."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", astuple(run)
            )

    def update_run(self, run: RunRow) -> None:
        """Replace an existing run row; StateError if it does not exist."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE runs SET ref=?, model=?, host=?, fingerprint=?, status=?, started_at=?, "
                "ended_at=?, exit_code=?, detail=? WHERE run_id=?",
                (*astuple(run)[1:], run.run_id),
            )
        if cursor.rowcount == 0:
            raise StateError(f"no run {run.run_id}")

    def get_run(self, run_id: str) -> RunRow | None:
        """The run row, if any."""
        row = self._one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return None if row is None else RunRow(*row)

    def last_run(self, ref: str, model: str) -> RunRow | None:
        """The most recently started run of this task and model, if any."""
        row = self._one(
            "SELECT * FROM runs WHERE ref=? AND model=? ORDER BY started_at DESC, run_id DESC LIMIT 1",
            (ref, model),
        )
        return None if row is None else RunRow(*row)

    def latest_run_ids(self) -> set[str]:
        """The run id of the most recently started run of every task and model pair that has one."""
        rows = self._conn.execute(
            "SELECT run_id FROM (SELECT run_id, ROW_NUMBER() OVER (PARTITION BY ref, model "
            "ORDER BY started_at DESC, run_id DESC) AS n FROM runs) WHERE n = 1"
        )
        return {row[0] for row in rows}

    def jobs_with_status(self, status: str) -> list[JobRow]:
        """Every job in this status, ordered by ref, then model."""
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status=? ORDER BY ref, model", (status,)
        )
        return [JobRow(*row) for row in rows]

    def runs_with_status(self, status: str) -> list[RunRow]:
        """Every run in this status, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE status=? ORDER BY started_at, run_id", (status,)
        )
        return [RunRow(*row) for row in rows]

    def executor_runs(self, since: datetime) -> list[RunRow]:
        """Runs on a host other than the workstation that are still running or ended at or after `since`, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE host<>? AND (status='running' OR ended_at>=?) "
            "ORDER BY started_at, run_id",
            (WORKSTATION, _iso(since)),
        )
        return [RunRow(*row) for row in rows]

    def has_event(self, kind: str, detail: str) -> bool:
        """Whether an event of this kind has exactly this detail, or this detail followed by a space and more."""
        row = self._one(
            "SELECT 1 FROM events WHERE kind=? AND (detail=? OR substr(detail, 1, ?)=?) LIMIT 1",
            (kind, detail, len(detail) + 1, detail + " "),
        )
        return row is not None

    def set_workload_watch(self, text: str) -> None:
        """Store the R35 watch status (armed, down, no-signal, unreadable: ..., off)."""
        with self._conn:
            self._set_meta("workload_watch", text)

    def workload_watch(self) -> str | None:
        """The R35 watch status the dispatcher last stored, if any."""
        return self._meta("workload_watch")

    def _workload_acked_id(self) -> int:
        raw = self._meta("workload_acked_id")
        if raw is None:
            return 0
        if not (raw.isascii() and raw.isdigit()):
            raise StateError(f"meta workload_acked_id is not a number: {raw!r}")
        return int(raw)

    def unacked_workload_overlaps(self) -> list[EventRow]:
        """The `workload-overlap` events recorded since the last `ack_workload`, oldest first."""
        rows = self._conn.execute(
            "SELECT id, at, ref, kind, detail FROM events WHERE kind='workload-overlap' AND id>? "
            "ORDER BY id",
            (self._workload_acked_id(),),
        )
        return [EventRow(*row) for row in rows]

    def ack_workload(self, now: datetime) -> int:
        """Mark every recorded `workload-overlap` event reviewed, log a `workload-ack` event, and return how many were cleared."""
        acked = self._workload_acked_id()
        count, newest = self._conn.execute(
            "SELECT COUNT(*), MAX(id) FROM events WHERE kind='workload-overlap' AND id>?", (acked,)
        ).fetchone()
        with self._conn:
            self._set_meta("workload_acked_id", str(max(acked, newest or 0)))
            self._event(now, "workload-ack", None, f"cleared {count}")
        return int(count)

    def request_stop(self, mode: Literal["graceful", "now"]) -> None:
        """Ask the dispatcher to exit (after the running job, or now)."""
        if mode not in ("graceful", "now"):
            raise ValueError(f"stop mode must be graceful or now, got {mode!r}")
        with self._conn:
            self._set_meta("stop_requested", mode)

    def stop_requested(self) -> str | None:
        """The pending stop request, if any."""
        return self._meta("stop_requested")

    def clear_stop(self) -> None:
        """Drop any pending stop request."""
        with self._conn:
            self._conn.execute("DELETE FROM meta WHERE key='stop_requested'")

    def request_run_now(self, ref: str, model: str) -> None:
        """Ask the dispatcher to start this pending job next."""
        with self._conn:
            self._set_meta("run_now", json.dumps([ref, model]))

    def run_now_requested(self) -> tuple[str, str] | None:
        """The pending run-now (ref, model), if any; StateError if the stored value is corrupt."""
        raw = self._meta("run_now")
        if raw is None:
            return None
        try:
            pair = json.loads(raw)
        except json.JSONDecodeError:
            pair = None
        if not (
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)
        ):
            raise StateError(f"meta run_now is not a [ref, model] JSON list: {raw!r}")
        return pair[0], pair[1]

    def clear_run_now(self, expected: tuple[str, str] | None = None) -> None:
        """Drop the pending run-now request; with `expected`, only if it is still that (ref, model), in one statement."""
        with self._conn:
            if expected is None:
                self._conn.execute("DELETE FROM meta WHERE key='run_now'")
            else:
                self._conn.execute(
                    "DELETE FROM meta WHERE key='run_now' AND value=?",
                    (json.dumps(list(expected)),),
                )

    def reset_crash_streaks(self, except_keys: set[tuple[str, str]]) -> None:
        """Zero the crash streak of every job (ref, model) not running at this crash (D-012: only consecutive crashes count)."""
        sql = "UPDATE jobs SET crash_streak=0"
        with self._conn:
            if except_keys:
                marks = ",".join(["(?, ?)"] * len(except_keys))
                args = tuple(part for key in sorted(except_keys) for part in key)
                self._conn.execute(f"{sql} WHERE (ref, model) NOT IN (VALUES {marks})", args)
            else:
                self._conn.execute(sql)

    def rerun(self, ref: str, model: str) -> None:
        """Put a job back to pending with its counters cleared."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE jobs SET status='pending', attempts=0, crash_streak=0 WHERE ref=? AND model=?",
                (ref, model),
            )
        if cursor.rowcount == 0:
            raise StateError(f"no job {ref} with model {model} in state")

    def ack(self, run_id: str, now: datetime) -> None:
        """Accept a run awaiting review: the run becomes done, and its job (same ref and model) too unless the manifest entry changed since."""
        run = self.get_run(run_id)
        if run is None or run.status != "awaiting-review":
            state = "no such run" if run is None else f"status {run.status}"
            raise StateError(f"run {run_id} is not awaiting review ({state})")
        with self._conn:
            self._conn.execute("UPDATE runs SET status='done' WHERE run_id=?", (run_id,))
            self._conn.execute(
                "UPDATE jobs SET status='done' WHERE ref=? AND model=? AND fingerprint=?",
                (run.ref, run.model, run.fingerprint),
            )
            self._event(now, "done", run.ref, f"acked {run_id}")


class DispatcherLock:
    """Exclusive non-blocking flock on the lock file, holding the pid; context manager."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise LockHeld(f"dispatcher lock {self._path} is held by another process") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def lock_is_held(path: Path) -> bool:
    """Whether some process holds the lock file (probe with a non-blocking shared lock)."""
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def reset(paths: StatePaths) -> list[Path]:
    """Delete the state database files (never STOP or run outputs); refuse while the lock is held."""
    if lock_is_held(paths.lock):
        raise LockHeld("the dispatcher is running; stop it before reset")
    deleted = []
    for suffix in ("", "-wal", "-shm"):
        target = paths.db.with_name(paths.db.name + suffix)
        if target.exists():
            target.unlink()
            deleted.append(target)
    return deleted
