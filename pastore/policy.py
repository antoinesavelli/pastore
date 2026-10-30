"""Parse and validate policy.yaml (the human gate), keep the last good copy, render and diff edits.

Depends on: PyYAML and stdlib only.
"""

import difflib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import yaml

KEYS = frozenset({"version", "window", "consumers", "jobs"})
MANIFEST_NAME = "gpu-tasks.yaml"
NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
HHMM = re.compile(r"([01]\d|2[0-3]):[0-5]\d")
HEADER = (
    "# The single human gate: only jobs listed under `jobs` ever run, in this order"
    " (first = highest priority).\n"
    "# Adding a job is ask-first and needs a passing `pastore check --contract <consumer>/<id>`"
    " (docs/DESIGN.md R7).\n"
)


class PolicyError(Exception):
    """The policy file is unreadable or invalid; the message lists every problem."""


@dataclass(frozen=True)
class Policy:
    """A validated policy: quiet-hours window, consumer manifests, ordered job refs."""

    window_start: time
    window_end: time
    consumers: Mapping[str, Path]
    jobs: tuple[str, ...]


def _parse_window(raw: object, problems: list[str]) -> tuple[time, time]:
    if not isinstance(raw, dict) or set(raw) != {"start", "end"}:
        problems.append("window must be a mapping with exactly start and end")
        return time(0), time(0)
    ends: list[time] = []
    for key in ("start", "end"):
        value = raw[key]
        if isinstance(value, str) and HHMM.fullmatch(value):
            ends.append(time(int(value[:2]), int(value[3:])))
        else:
            problems.append(f'window.{key} must be a quoted "HH:MM" string, got {value!r}')
    if len(ends) == 2 and ends[0] == ends[1]:
        problems.append("window start and end must differ")
    return (ends[0], ends[1]) if len(ends) == 2 else (time(0), time(0))


def _parse_consumers(raw: object, problems: list[str]) -> dict[str, Path]:
    if not isinstance(raw, dict):
        problems.append("consumers must be a mapping (use {} for none)")
        return {}
    out: dict[str, Path] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not NAME.fullmatch(name):
            problems.append(f"consumer name {name!r} must match {NAME.pattern}")
        elif not isinstance(value, str) or not Path(value).is_absolute():
            problems.append(f"consumer {name}: path must be an absolute string, got {value!r}")
        elif Path(value).name != MANIFEST_NAME:
            problems.append(f"consumer {name}: path must end in {MANIFEST_NAME}, got {value!r}")
        else:
            out[name] = Path(value)
    return out


def check_ref(ref: object, consumers: Mapping[str, Path]) -> str | None:
    """Return the problem with a job ref, or None when it is `<known consumer>/<id>`."""
    parts = ref.split("/") if isinstance(ref, str) else []
    if len(parts) != 2 or not NAME.fullmatch(parts[1]):
        return f"job {ref!r} must look like <consumer>/<id>"
    if parts[0] not in consumers:
        return f"job {ref!r}: consumer {parts[0]!r} is not listed under consumers"
    return None


def _parse_jobs(raw: object, consumers: Mapping[str, Path], problems: list[str]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        problems.append("jobs must be a list (use [] for none)")
        return ()
    seen: set[str] = set()
    for ref in raw:
        problem = check_ref(ref, consumers)
        if problem is None and ref in seen:
            problem = f"job {ref!r} is listed twice"
        if problem is not None:
            problems.append(problem)
        else:
            seen.add(ref)
    return tuple(r for r in raw if r in seen)


def parse_policy(text: str) -> Policy:
    """Validate policy text; raise PolicyError listing every problem found."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyError("policy must be a mapping with version, window, consumers, jobs")
    problems = [f"unknown key {k!r}" for k in data if k not in KEYS]
    problems += [f"missing key {k!r}" for k in sorted(KEYS - set(data))]
    if type(data.get("version")) is not int or data["version"] != 1:
        problems.append(f"version must be 1, got {data.get('version')!r}")
    start, end = _parse_window(data.get("window"), problems)
    consumers = _parse_consumers(data.get("consumers"), problems)
    jobs = _parse_jobs(data.get("jobs"), consumers, problems)
    if problems:
        raise PolicyError("; ".join(problems))
    return Policy(start, end, consumers, jobs)


def load_policy(path: Path) -> Policy:
    """Read and validate the policy file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyError(f"cannot read {path}: {exc}") from exc
    return parse_policy(text)


class PolicyCache:
    """Re-reads the policy when its bytes change and falls back to the last good one on error."""

    def __init__(self, path: Path, log: Callable[[str], None]) -> None:
        self._path = path
        self._log = log
        self._seen: bytes | None = None
        self._good: Policy | None = None
        self._error: PolicyError | None = None

    def current(self) -> Policy:
        """Return the live policy, the last good one if the file is bad, or raise if never good."""
        try:
            data = self._path.read_bytes()
            error = None
        except OSError as exc:
            data, error = f"unreadable: {exc}".encode(), PolicyError(f"cannot read: {exc}")
        if data != self._seen:
            self._seen = data
            if error is None:
                error = self._try_parse(data)
            self._error = error
            if error is not None:
                self._log(f"policy error in {self._path}: {error}")
        if self._good is None:
            raise self._error or PolicyError("no policy loaded")
        return self._good

    def _try_parse(self, data: bytes) -> PolicyError | None:
        try:
            self._good = parse_policy(data.decode("utf-8"))
        except UnicodeDecodeError as exc:
            return PolicyError(f"not UTF-8: {exc}")
        except PolicyError as exc:
            return exc
        return None


def render_policy(policy: Policy) -> str:
    """Canonical policy.yaml text; load_policy of it equals the input."""
    lines = [
        HEADER.rstrip("\n"),
        "version: 1",
        "window:",
        f'  start: "{policy.window_start:%H:%M}"',
        f'  end: "{policy.window_end:%H:%M}"',
    ]
    if policy.consumers:
        lines.append("consumers:")
        lines += [f"  {n}: {json.dumps(str(p))}" for n, p in policy.consumers.items()]
    else:
        lines.append("consumers: {}")
    if policy.jobs:
        lines.append("jobs:")
        lines += [f"  - {ref}" for ref in policy.jobs]
    else:
        lines.append("jobs: []")
    return "\n".join(lines) + "\n"


def with_job_added(policy: Policy, ref: str) -> Policy:
    """A copy of the policy with the job appended; PolicyError if unknown consumer or listed."""
    problem = check_ref(ref, policy.consumers)
    if problem is None and ref in policy.jobs:
        problem = f"job {ref!r} is already listed"
    if problem is not None:
        raise PolicyError(problem)
    return Policy(policy.window_start, policy.window_end, policy.consumers, (*policy.jobs, ref))


def with_job_removed(policy: Policy, ref: str) -> Policy:
    """A copy of the policy without the job; PolicyError if it is not listed."""
    if ref not in policy.jobs:
        raise PolicyError(f"job {ref!r} is not listed")
    jobs = tuple(j for j in policy.jobs if j != ref)
    return Policy(policy.window_start, policy.window_end, policy.consumers, jobs)


def policy_diff(old_text: str, new_text: str, path: Path) -> str:
    """Unified diff of a proposed policy edit."""
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (proposed)",
    )
    return "".join(diff)
