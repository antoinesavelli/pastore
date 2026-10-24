"""Repo-wide invariant scans over fucone's own source: never sudo, and no real clock outside the CLI.

Depends on: the source files under fucone/ (scanned as AST), plus fucone.spec.idle_reason.
"""

import ast
from pathlib import Path

import pytest

from fucone.spec import idle_reason

PACKAGE = Path(__file__).resolve().parent.parent / "fucone"
CLOCK_CALLS = {
    ("time", "time"),
    ("time", "monotonic"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("date", "today"),
}
CLOCK_ALLOWED = {"cli.py"}


def _sources() -> list[Path]:
    files = sorted(PACKAGE.glob("*.py"))
    assert files, "no fucone sources found"
    return files


def sudo_violations(name: str, text: str) -> list[str]:
    """Literal sudo/pkexec strings, or a systemctl/systemd-run argv list without --user."""
    found = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and node.value in {"sudo", "pkexec"}:
            found.append(f"{name}:{node.lineno}: {node.value!r}")
        if isinstance(node, ast.List):
            items = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if items and items[0] in {"systemctl", "systemd-run"} and "--user" not in items:
                found.append(f"{name}:{node.lineno}: {items}")
    return found


def clock_violations(name: str, text: str) -> list[str]:
    """Attribute uses such as datetime.now or time.time."""
    return [
        f"{name}:{node.lineno}: {node.value.id}.{node.attr}"
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id, node.attr) in CLOCK_CALLS
    ]


def test_scanners_catch_violations() -> None:
    """A scanner that finds nothing in known-bad code would make the invariants vacuous."""
    assert sudo_violations("x.py", 'run(["sudo", "ls"])')
    assert sudo_violations("x.py", 'run(["systemctl", "stop", "u"])')
    assert not sudo_violations("x.py", 'run(["systemctl", "--user", "stop", "u"])')
    assert clock_violations("x.py", "t = datetime.now(UTC)")
    assert clock_violations("x.py", "t = time.time()")


def test_invariant_never_sudo() -> None:
    """No fucone source builds a sudo/pkexec or system-manager systemd command."""
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
    for bad in (("busy", ""), ("gpu-foreign", ""), ("stopped", "why")):
        with pytest.raises(ValueError):
            idle_reason(*bad)
