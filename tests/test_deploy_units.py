"""The dispatcher's user unit kept in deploy/: the directives a rebuilt host needs, and that it is a user unit.

Depends on: deploy/pastore.service.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "deploy" / "pastore.service"


def _directives() -> dict[str, str]:
    """Every `key=value` line of the unit, the last one winning."""
    lines = (line.strip() for line in UNIT.read_text().splitlines())
    pairs = (line.partition("=") for line in lines if line and not line.startswith(("#", "[")))
    return {key.strip(): value.strip() for key, _, value in pairs}


def test_dispatcher_unit_runs_the_repo_venv_in_the_repo() -> None:
    """The dispatcher starts from the repo's own .venv, which `uv sync` keeps, in the repo directory."""
    unit = _directives()
    assert unit["ExecStart"] == "/srv/projects/pastore/.venv/bin/python -m pastore start"
    assert unit["WorkingDirectory"] == "/srv/projects/pastore"


def test_dispatcher_unit_restarts_on_failure_only_and_pages() -> None:
    """A crash restarts after 60 s, a graceful stop stays stopped, five failures an hour give up and page."""
    unit = _directives()
    assert unit["Restart"] == "on-failure"
    assert unit["RestartSec"] == "60"
    assert unit["StartLimitBurst"] == "5"
    assert unit["StartLimitIntervalSec"] == "1h"
    assert unit["OnFailure"] == "unit-failed@%n.service"


def test_dispatcher_unit_is_a_user_unit_without_sudo() -> None:
    """It installs under default.target, names no User=, and nothing in it escalates privileges."""
    unit = _directives()
    assert unit["WantedBy"] == "default.target"
    assert "User" not in unit
    assert "sudo" not in UNIT.read_text()
