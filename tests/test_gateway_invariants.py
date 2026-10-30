"""The repo-wide invariant scans applied to pastore/gateway/: never sudo, and no real clock outside its entry point.

Depends on: tests.test_invariants (the scanners) and the source files under pastore/gateway/.
"""

from pathlib import Path

from tests.test_invariants import clock_violations, sudo_violations

GATEWAY = Path(__file__).resolve().parent.parent / "pastore" / "gateway"
CLOCK_ALLOWED = {"__main__.py"}


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
