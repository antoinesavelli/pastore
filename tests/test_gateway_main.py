"""The gateway entry point and its unit: refuses a bad policy, wires hang-up cancellation, points at policy.yaml.

Depends on: pastore.gateway.__main__, pastore.policy, deploy/pastore-gateway.service and the repo's policy.yaml.
"""

import shlex
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from pastore.gateway import __main__ as entry
from pastore.policy import load_policy

ROOT = Path(__file__).resolve().parent.parent
REPO_POLICY = ROOT / "policy.yaml"
UNIT = ROOT / "deploy" / "pastore-gateway.service"


def test_refuses_a_v1_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A policy v5 cannot read exits 2 with the reason and never serves."""
    v1 = tmp_path / "policy.yaml"
    v1.write_text("version: 1\nwindow: {start: '01:00', end: '07:00'}\nconsumers: {}\njobs: []\n")
    assert entry.main(["--policy", str(v1), "--port", "0"]) == 2
    assert "version must be 2" in capsys.readouterr().err


def test_serves_with_handler_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry point runs the app with handler cancellation, which hang-up removal (check 7) relies on."""
    seen: dict[str, Any] = {}

    def fake_run_app(app: web.Application, **kwargs: Any) -> None:
        seen.update(kwargs, app=app)

    monkeypatch.setattr(entry.web, "run_app", fake_run_app)
    assert entry.main(["--policy", str(REPO_POLICY), "--host", "127.0.0.1", "--port", "0"]) == 0
    assert isinstance(seen["app"], web.Application)
    assert (seen["host"], seen["port"], seen["handler_cancellation"]) == ("127.0.0.1", 0, True)


def test_unit_policy_argument_is_the_repo_policy_yaml() -> None:
    """The unit's --policy names policy.yaml under its WorkingDirectory, and load_policy accepts that file."""
    lines = {k: v for k, _, v in (line.partition("=") for line in UNIT.read_text().splitlines())}
    argv = shlex.split(lines["ExecStart"])
    arg = Path(argv[argv.index("--policy") + 1])
    assert arg.is_absolute()
    assert arg.relative_to(lines["WorkingDirectory"]) == Path("policy.yaml")
    policy = load_policy(ROOT / arg.relative_to(lines["WorkingDirectory"]))
    assert {"workstation", "executor"} <= policy.hosts.keys()
