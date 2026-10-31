"""The repo-wide invariant scans applied to pastore/gateway/ (never sudo, no real clock outside its entry point), plus its import isolation.

Depends on: tests.test_invariants (the scanners), the source files under pastore/gateway/, and a python subprocess.
"""

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
    "pastore.workload_watch",
}
PROBE = (
    "import sys\n"
    "import pastore.gateway.__main__\n"
    "import pastore.gateway.server\n"
    "print('\\n'.join(sorted(n for n in sys.modules if n.split('.')[0] == 'pastore')))\n"
)


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
    """Importing the gateway entry point and server loads none of the dispatcher-only modules (R20)."""
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
