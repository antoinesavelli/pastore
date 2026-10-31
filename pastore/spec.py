"""Shared value types: tasks, hosts, source levels and rules, listed pairs, and the closed idle-reason and event lists.

Depends on: nothing in pastore; manifest.py and policy.py build these, every other module only reads them.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

IDLE_REASONS = (
    "no-ready-task",
    "no-fit",
    "driver-mismatch",
    "ollama-down",
    "executor-down",
    "gpu-foreign",
    "low-memory",
    "stopped",
)
TASK_EVENTS = ("refused", "reader-failed", "awaiting-review", "starved", "done", "failed")
WORKSTATION = "workstation"
BASE_ENV_KEYS = ("PATH", "HOME", "LANG", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
RUN_DIR_ENV = "PASTORE_RUN_DIR"
SOURCE_KINDS = ("cgroup", "peer", "ancestor", "under")
EXCLUDABLE_KINDS = ("cgroup", "peer")


class Level(IntEnum):
    """Request priority by source (D-026, R21): a lower value is served first; `policy.yaml` names it in lowercase."""

    REINS = 1
    WEBUI = 2
    CODING = 3
    CORSU = 4
    ROUTINE = 5
    HEARTH = 6


UNKNOWN_LEVEL = Level.ROUTINE


@dataclass(frozen=True)
class Host:
    """One machine from `policy.yaml` `hosts:` (R23, R25): Ollama URL, VRAM, model cap, per-model parallelism, context.

    `context` is the host Ollama's default context length in tokens (D-041: KV cache per slot is counted at it).
    `vram_gb` is read by placement as GiB (binary, 2**30 bytes each) despite its name; the name is the policy format.
    """

    name: str
    url: str
    vram_gb: int
    max_models: int
    parallel: int
    context: int


@dataclass(frozen=True)
class SourceRule:
    """One `policy.yaml` `sources:` row (R21, R22); rules are tried in order and the first match sets the level.

    `kind` is one of SOURCE_KINDS: `cgroup` globs the leaf unit of a loopback client's cgroup; `peer` holds the CIDRs a
    non-loopback peer address must fall in; `ancestor` holds `comm` names of the loopback client or any ancestor;
    `under` holds absolute directories the loopback client's exe or cwd must be at or below. `exclude` (globs or CIDRs,
    same form as `values`) is allowed only for EXCLUDABLE_KINDS. A request no rule matches gets UNKNOWN_LEVEL.
    """

    level: Level
    kind: str
    values: tuple[str, ...]
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pair:
    """One listed job (R29): a task `<consumer>/<id>`, the model it runs, and the evidence file the owner approved."""

    task: str
    model: str
    evidence: Path
    sha256: str


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
    model_env: str
    reader: RunReader | ValidateReader
    ready: Ready | None
    root: Path
    fingerprint: str

    @property
    def ref(self) -> str:
        """The task's name everywhere in pastore: `<consumer>/<id>`."""
        return f"{self.consumer}/{self.id}"


def canonical_model(name: str) -> str:
    """Ollama's name for a model: `:latest` is implied when the part after the last `/` has no tag."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


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
