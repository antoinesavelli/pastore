"""SQLite WAL state (jobs, runs, events, contracts, meta), the dispatcher flock, and reset.

Depends on: pastore.spec and stdlib (sqlite3, fcntl) only; never reads a clock.
"""

import fcntl
import os
import sqlite3
from dataclasses import astuple, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pastore.spec import TASK_EVENTS, idle_reason

SCHEMA_VERSION = 1
DISPATCHER_EVENTS = (
    "idle",
    "dispatcher-start",
    "dispatcher-stop",
    "recover",
    "policy-error",
    "run-start",
    "run-end",
    "tick-error",
)
SCHEMA = """
CREATE TABLE jobs(
    ref TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','done','failed','blocked')),
    attempts INT NOT NULL, crash_streak INT NOT NULL, ready_since TEXT, last_check_at TEXT,
    last_success_at TEXT, starved_flagged INT NOT NULL);
CREATE TABLE runs(
    run_id TEXT PRIMARY KEY, ref TEXT NOT NULL, fingerprint TEXT NOT NULL,
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
    """One row of `jobs`."""

    ref: str
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


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _check_version(conn: sqlite3.Connection, paths: StatePaths) -> bool:
    """Return True for a brand-new database; raise StateError on any other version mismatch."""
    version = _user_version(conn)
    empty = conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
    if version == 0 and empty:
        return True
    if version != SCHEMA_VERSION:
        conn.close()
        raise StateError(
            f"{paths.db} has schema version {version}, this pastore needs {SCHEMA_VERSION}; "
            "refusing to start (pastore reset wipes state)"
        )
    return False


def open_state(paths: StatePaths, *, readonly: bool = False) -> "State":
    """Open (and on first use create) the state database; readonly never writes."""
    if readonly:
        if not paths.db.exists():
            raise StateError("no state yet")
        conn = sqlite3.connect(f"{paths.db.resolve().as_uri()}?mode=ro", uri=True)
        _check_version(conn, paths)
    else:
        paths.state_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(paths.db)
        conn.execute("PRAGMA busy_timeout=5000")
        is_new = _check_version(conn, paths)
        conn.execute("PRAGMA journal_mode=WAL")
        if is_new:
            conn.executescript(SCHEMA + f"PRAGMA user_version={SCHEMA_VERSION};")
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

    def get_job(self, ref: str) -> JobRow | None:
        """The job row, if any."""
        row = self._one("SELECT * FROM jobs WHERE ref=?", (ref,))
        return None if row is None else JobRow(*row)

    def upsert_job(self, job: JobRow) -> None:
        """Insert or replace a job row."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                astuple(job),
            )

    def insert_run(self, run: RunRow) -> None:
        """Insert a new run row."""
        with self._conn:
            self._conn.execute("INSERT INTO runs VALUES(?, ?, ?, ?, ?, ?, ?, ?)", astuple(run))

    def update_run(self, run: RunRow) -> None:
        """Replace an existing run row; StateError if it does not exist."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE runs SET ref=?, fingerprint=?, status=?, started_at=?, ended_at=?, "
                "exit_code=?, detail=? WHERE run_id=?",
                (*astuple(run)[1:], run.run_id),
            )
        if cursor.rowcount == 0:
            raise StateError(f"no run {run.run_id}")

    def get_run(self, run_id: str) -> RunRow | None:
        """The run row, if any."""
        row = self._one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return None if row is None else RunRow(*row)

    def last_run(self, ref: str) -> RunRow | None:
        """The most recently started run of this task, if any."""
        row = self._one(
            "SELECT * FROM runs WHERE ref=? ORDER BY started_at DESC, run_id DESC LIMIT 1", (ref,)
        )
        return None if row is None else RunRow(*row)

    def jobs_with_status(self, status: str) -> list[JobRow]:
        """Every job in this status."""
        rows = self._conn.execute("SELECT * FROM jobs WHERE status=? ORDER BY ref", (status,))
        return [JobRow(*row) for row in rows]

    def runs_with_status(self, status: str) -> list[RunRow]:
        """Every run in this status, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE status=? ORDER BY started_at, run_id", (status,)
        )
        return [RunRow(*row) for row in rows]

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

    def request_run_now(self, ref: str) -> None:
        """Ask the dispatcher to start `ref` at its next tick, outside the window."""
        with self._conn:
            self._set_meta("run_now", ref)

    def run_now_requested(self) -> str | None:
        """The pending run-now ref, if any."""
        return self._meta("run_now")

    def clear_run_now(self) -> None:
        """Drop any pending run-now request."""
        with self._conn:
            self._conn.execute("DELETE FROM meta WHERE key='run_now'")

    def reset_crash_streaks(self, except_refs: set[str]) -> None:
        """Zero the crash streak of every job not running at this crash (D-012: only consecutive crashes count)."""
        marks = ",".join("?" * len(except_refs))
        sql = "UPDATE jobs SET crash_streak=0"
        with self._conn:
            if except_refs:
                self._conn.execute(f"{sql} WHERE ref NOT IN ({marks})", tuple(sorted(except_refs)))
            else:
                self._conn.execute(sql)

    def rerun(self, ref: str) -> None:
        """Put a job back to pending with its counters cleared."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE jobs SET status='pending', attempts=0, crash_streak=0 WHERE ref=?", (ref,)
            )
        if cursor.rowcount == 0:
            raise StateError(f"no job {ref} in state")

    def ack(self, run_id: str, now: datetime) -> None:
        """Accept a run awaiting review: the run becomes done, and its job too unless the manifest entry changed since."""
        run = self.get_run(run_id)
        if run is None or run.status != "awaiting-review":
            state = "no such run" if run is None else f"status {run.status}"
            raise StateError(f"run {run_id} is not awaiting review ({state})")
        with self._conn:
            self._conn.execute("UPDATE runs SET status='done' WHERE run_id=?", (run_id,))
            self._conn.execute(
                "UPDATE jobs SET status='done' WHERE ref=? AND fingerprint=?",
                (run.ref, run.fingerprint),
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
