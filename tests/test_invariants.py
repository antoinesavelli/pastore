"""Repo-wide invariant scans over pastore's own source: never sudo, and no real clock outside the CLI.

Depends on: the source files under pastore/ (scanned as AST), plus pastore.spec.idle_reason.
"""

import ast
from pathlib import Path

import pytest

from pastore.spec import idle_reason

PACKAGE = Path(__file__).resolve().parent.parent / "pastore"
CLOCK_CALLS = {
    ("time", "time"),
    ("time", "monotonic"),
    ("time", "perf_counter"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("date", "today"),
}
CLOCK_ALLOWED = {"cli.py"}


def _sources() -> list[Path]:
    files = sorted(PACKAGE.glob("*.py"))
    assert files, "no pastore sources found"
    return files


def sudo_violations(name: str, text: str) -> list[str]:
    """Literal sudo/pkexec strings, or a systemctl/systemd-run argv list without --user."""
    found = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and node.value in {"sudo", "pkexec"}:
            found.append(f"{name}:{node.lineno}: {node.value!r}")
        if isinstance(node, ast.List | ast.Tuple):
            items = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if items and items[0] in {"systemctl", "systemd-run"} and "--user" not in items:
                found.append(f"{name}:{node.lineno}: {items}")
    return found


def _owner(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def clock_violations(name: str, text: str) -> list[str]:
    """Clock reads: datetime.now, datetime.datetime.now, time.time, or importing such a function directly."""
    found = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute) and (_owner(node.value), node.attr) in CLOCK_CALLS:
            found.append(f"{name}:{node.lineno}: {_owner(node.value)}.{node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module in {"time", "datetime"}:
            found += [
                f"{name}:{node.lineno}: from {node.module} import {a.name}"
                for a in node.names
                if (node.module, a.name) in CLOCK_CALLS
            ]
    return found


def test_scanners_catch_violations() -> None:
    """A scanner that finds nothing in known-bad code would make the invariants vacuous."""
    assert sudo_violations("x.py", 'run(["sudo", "ls"])')
    assert sudo_violations("x.py", 'run(["systemctl", "stop", "u"])')
    assert not sudo_violations("x.py", 'run(["systemctl", "--user", "stop", "u"])')
    assert clock_violations("x.py", "t = datetime.now(UTC)")
    assert clock_violations("x.py", "t = time.time()")
    assert clock_violations("x.py", "t = datetime.datetime.now()")
    assert clock_violations("x.py", "from time import monotonic")
    assert sudo_violations("x.py", 'run(("sudo", "ls"))')
    assert not clock_violations(
        "x.py", "from datetime import UTC, datetime\nt = datetime(2026, 1, 1)"
    )


def test_invariant_never_sudo() -> None:
    """No pastore source builds a sudo/pkexec or system-manager systemd command."""
    found = [v for p in _sources() for v in sudo_violations(p.name, p.read_text())]
    assert found == []


def test_invariant_no_real_clock_in_decision_code() -> None:
    """time.time / datetime.now / date.today appear only in cli.py, which binds the real clock."""
    found = [
        v
        for p in _sources()
        if p.name not in CLOCK_ALLOWED
        for v in clock_violations(p.name, p.read_text())
    ]
    assert found == []


def test_invariant_idle_reasons_are_a_closed_list() -> None:
    """Anything outside the closed list, or gpu-foreign without its process, is refused."""
    assert idle_reason("gpu-foreign", "/opt/x") == "gpu-foreign:/opt/x"
    assert idle_reason("no-fit") == "no-fit" and idle_reason("executor-down") == "executor-down"
    for bad in (("busy", ""), ("gpu-foreign", ""), ("stopped", "why"), ("outside-window", "")):
        with pytest.raises(ValueError):
            idle_reason(*bad)
