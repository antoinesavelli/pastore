"""The gateway entry point and its interim policy: refuses a bad policy, wires hang-up cancellation, mirrors policy-v2.

Depends on: pastore.gateway.__main__, pastore.policy, deploy/gateway-policy.yaml and policy-v2.yaml.
"""

from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from pastore.gateway import __main__ as entry
from pastore.policy import load_policy

ROOT = Path(__file__).resolve().parent.parent
GATEWAY_POLICY = ROOT / "deploy" / "gateway-policy.yaml"


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
    assert entry.main(["--policy", str(GATEWAY_POLICY), "--host", "127.0.0.1", "--port", "0"]) == 0
    assert isinstance(seen["app"], web.Application)
    assert (seen["host"], seen["port"], seen["handler_cancellation"]) == ("127.0.0.1", 0, True)


def test_gateway_policy_mirrors_policy_v2() -> None:
    """The interim gateway policy has policy-v2's sources and hosts; only the workstation's slot counts may differ."""
    gateway, v2 = load_policy(GATEWAY_POLICY), load_policy(ROOT / "policy-v2.yaml")
    assert gateway.sources == v2.sources
    assert gateway.hosts.keys() == v2.hosts.keys()
    for name, host in gateway.hosts.items():
        same = (host.url, host.vram_gb) == (v2.hosts[name].url, v2.hosts[name].vram_gb)
        assert same and (name == "workstation" or host == v2.hosts[name])
    assert gateway.jobs == ()
