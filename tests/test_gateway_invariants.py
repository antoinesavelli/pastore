"""The repo-wide invariant scans applied to pastore/gateway/ (never sudo, no real clock outside its entry point), plus its import isolation.

Depends on: tests.test_invariants (the scanners), the source files under pastore/gateway/, and a python subprocess.
"""

import ast
import subprocess
import sys
from pathlib import Path

from tests.test_invariants import clock_violations, sudo_violations

ROOT = Path(__file__).resolve().parent.parent
GATEWAY = ROOT / "pastore" / "gateway"
CLOCK_ALLOWED = {"__main__.py"}
DISPATCHER_ONLY = {
    "pastore.dispatcher",
    "pastore.runner",
    "pastore.cli",
    "pastore.contract",
    "pastore.checks",
    "pastore.hosts",
    "pastore.manifest",
    "pastore.placement",
    "pastore.run_end",
    "pastore.tick_bound",
    "pastore.workload_watch",
}
PROBE = (
    "import sys\n"
    "import pastore.gateway.__main__\n"
    "import pastore.gateway.server\n"
    "print('\\n'.join(sorted(n for n in sys.modules if n.split('.')[0] == 'pastore')))\n"
)


def _is_dispatcher_only(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in DISPATCHER_ONLY)


def forbidden_imports(name: str, text: str, package: str = "pastore.gateway") -> list[str]:
    """Import and ImportFrom nodes anywhere in the source, function bodies included, naming a dispatcher-only module."""
    found = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            parts = package.split(".")[: len(package.split(".")) - max(node.level - 1, 0)]
            base = ".".join((parts if node.level else []) + ([node.module] if node.module else []))
            targets = [base] + [f"{base}.{a.name}" for a in node.names]
        else:
            continue
        found += [f"{name}:{node.lineno}: {t}" for t in targets if _is_dispatcher_only(t)]
    return found


def _sources() -> list[Path]:
    files = sorted(GATEWAY.glob("*.py"))
    assert files, "no gateway sources found"
    return files


def test_invariant_never_sudo_in_gateway() -> None:
    """No gateway source builds a sudo/pkexec or system-manager systemd command."""
    assert [v for p in _sources() for v in sudo_violations(p.name, p.read_text())] == []


def test_invariant_no_real_clock_in_gateway_decision_code() -> None:
    """Only the gateway entry point binds the real clock; the queue, attribution and server take it injected."""
    found = [
        v
        for p in _sources()
        if p.name not in CLOCK_ALLOWED
        for v in clock_violations(p.name, p.read_text())
    ]
    assert found == []


def test_invariant_gateway_imports_no_dispatcher_write_path() -> None:
    """The gateway neither imports (even lazily, in a function body) nor loads at import time a dispatcher-only module (R20)."""
    assert forbidden_imports("x.py", "def f():\n    import pastore.runner\n")
    assert forbidden_imports("x.py", "def f():\n    from pastore import cli\n")
    assert forbidden_imports("x.py", "from pastore.dispatcher import run_one")
    assert forbidden_imports("x.py", "import pastore.hosts as h")
    assert forbidden_imports("x.py", "def f():\n    from .. import placement\n")
    assert forbidden_imports("x.py", "from ..workload_watch import WorkloadWatch")
    assert forbidden_imports("x.py", "def f():\n    from pastore.tick_bound import TickOverrun\n")
    assert forbidden_imports("x.py", "from .. import run_end")
    assert not forbidden_imports(
        "x.py",
        "import pastore.policy\nfrom pastore import spec\nfrom pastore.state import default_paths\n"
        "from . import queue\nfrom .queue import Queue\nfrom pastore.gateway.server import make_app\n"
        "import pastore.clients\n",
    )
    assert [v for p in _sources() for v in forbidden_imports(p.name, p.read_text())] == []
    probe = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    loaded = set(probe.stdout.split())
    assert {"pastore.gateway.__main__", "pastore.gateway.server"} <= loaded
    assert sorted(loaded & DISPATCHER_ONLY) == []
