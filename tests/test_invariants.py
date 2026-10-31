"""Repo-wide invariant scans over pastore's own source: never sudo, never ssh, and no real clock outside the CLI.

Depends on: the source files under pastore/ (scanned as AST), plus pastore.spec.idle_reason.
"""

import ast
import re
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
SSH_COMMAND = re.compile(r"^(\S*/)?ssh(\s|$)")


def _sources() -> list[Path]:
    files = sorted(PACKAGE.glob("*.py"))
    assert files, "no pastore sources found"
    return files


def _all_sources() -> list[Path]:
    """Every module under pastore/, subpackages (the gateway) included."""
    files = sorted(PACKAGE.rglob("*.py"))
    assert any("gateway" in p.parts for p in files), "no pastore/gateway sources found"
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


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Identity of every module, class and function docstring node: prose, never an argv."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def ssh_violations(name: str, text: str) -> list[str]:
    """String constants starting with the ssh binary, bare or path-qualified, alone or as a shell command.

    Docstrings are skipped (prose may mention ssh); a lookalike such as "sshd_config" is not the binary.
    """
    tree = ast.parse(text)
    docstrings = _docstring_ids(tree)
    return [
        f"{name}:{node.lineno}: {node.value!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and SSH_COMMAND.match(node.value)
    ]


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
    assert ssh_violations("x.py", 'run(["ssh", "second-host", "true"])')
    assert ssh_violations("x.py", 'run(("ssh", "second-host", "true"))')
    assert ssh_violations("x.py", 'run(["/usr/bin/ssh", "-L", "1:2:3", "h"])')
    assert ssh_violations("x.py", 'run(["sh", "-c", "ssh second-host true"])')
    assert ssh_violations("x.py", 'run(["sh", "-c", "/usr/bin/ssh second-host true"])')
    assert ssh_violations("x.py", 'cmd = f"/usr/local/bin/ssh {host} true"')
    assert ssh_violations("x.py", 'cmd = "./ssh -L 1:2:3 host"')
    assert ssh_violations("x.py", 'cmd = f"ssh {host} true"')
    assert ssh_violations("x.py", 'shutil.which("ssh")')
    assert not ssh_violations("x.py", 'path = "/etc/ssh/sshd_config"\nunit = "sshd.service"')
    assert not ssh_violations("x.py", 'run(["sshpass", "-V"])')
    assert not ssh_violations("x.py", 'p = "/etc/ssh/sshd_config"\nq = "/etc/ssh/ssh_config"')
    assert not ssh_violations("x.py", 'm = "the ssh forward is a user unit"')
    assert not ssh_violations(
        "x.py",
        '"""ssh forward notes."""\n\n\nclass C:\n    """ssh is run by the user unit."""\n\n'
        '    def f(self) -> None:\n        """ssh local forward, never run here."""\n',
    )
    assert not clock_violations(
        "x.py", "from datetime import UTC, datetime\nt = datetime(2026, 1, 1)"
    )


def test_invariant_never_sudo() -> None:
    """No pastore source builds a sudo/pkexec or system-manager systemd command."""
    found = [v for p in _sources() for v in sudo_violations(p.name, p.read_text())]
    assert found == []


def test_invariant_never_ssh() -> None:
    """No module under pastore/ (the gateway included) builds an ssh command; the user unit owns the executor link."""
    found = [
        v
        for p in _all_sources()
        for v in ssh_violations(str(p.relative_to(PACKAGE)), p.read_text())
    ]
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
