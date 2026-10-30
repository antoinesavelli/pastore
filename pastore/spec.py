"""Shared value types: a validated task, its reader and ready rule, and the closed idle-reason and event lists.

Depends on: nothing in pastore; manifest.py builds these, every other module only reads them.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

IDLE_REASONS = (
    "outside-window",
    "no-ready-task",
    "driver-mismatch",
    "ollama-down",
    "gpu-foreign",
    "low-memory",
    "stopped",
)
TASK_EVENTS = ("refused", "reader-failed", "awaiting-review", "starved", "done", "failed")


@dataclass(frozen=True)
class RunReader:
    """Machine reader: argv run against the finished run dir; non-zero means reader-failed."""

    argv: tuple[str, ...]


@dataclass(frozen=True)
class ValidateReader:
    """Validator argv, then the run waits in awaiting-review until the owner acks it."""

    argv: tuple[str, ...]
    review: str


@dataclass(frozen=True)
class Ready:
    """Recurring rule: run `check` at most every `every_s` seconds; exit 0 means ready."""

    every_s: int
    check: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    """One validated `gpu-tasks.yaml` entry, resolved against its consumer root."""

    consumer: str
    id: str
    kind: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    models: tuple[str, ...]
    timeout_s: int
    ok_exit: frozenset[int]
    ollama_env: str
    reader: RunReader | ValidateReader
    ready: Ready | None
    root: Path
    fingerprint: str

    @property
    def ref(self) -> str:
        """The task's name everywhere in pastore: `<consumer>/<id>`."""
        return f"{self.consumer}/{self.id}"


def idle_reason(reason: str, detail: str = "") -> str:
    """Format an idle reason, refusing anything outside the closed list."""
    if reason not in IDLE_REASONS:
        raise ValueError(f"not a closed-list idle reason: {reason!r}")
    if reason == "gpu-foreign":
        if not detail:
            raise ValueError("gpu-foreign needs the foreign process name")
        return f"gpu-foreign:{detail}"
    if detail:
        raise ValueError(f"{reason} takes no detail")
    return reason
