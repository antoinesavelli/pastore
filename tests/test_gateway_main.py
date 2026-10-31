"""The gateway entry point and its unit: refuses a bad policy, falls back to the last good copy, wires hang-up cancellation, points at policy.yaml.

Depends on: pastore.gateway.__main__, pastore.policy, deploy/pastore-gateway.service and the repo's policy.example.yaml.
"""

import os
import shlex
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from pastore.gateway import __main__ as entry
from pastore.policy import Policy, PolicyError, load_policy, parse_policy

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_POLICY = ROOT / "policy.example.yaml"
UNIT = ROOT / "deploy" / "pastore-gateway.service"
GOOD = EXAMPLE_POLICY.read_text()
OTHER = GOOD.replace("context: 32768", "context: 16384")
WARNING = "<4>pastore gateway: "


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HOME under tmp_path, so no test reads or writes the real ~/.local/state/pastore."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Policy]:
    """Stub web.run_app (never binds) and record the policy make_app is given, in order."""
    policies: list[Policy] = []
    make_app = entry.make_app

    def recording(policy: Policy, *args: Any, **kwargs: Any) -> web.Application:
        policies.append(policy)
        return make_app(policy, *args, **kwargs)

    monkeypatch.setattr(entry, "make_app", recording)
    monkeypatch.setattr(entry.web, "run_app", lambda app, **kwargs: None)
    return policies


def copy_of(home: Path, make_dir: bool = True) -> Path:
    """The gateway's copy under the (optionally created) state directory of `home`."""
    state_dir = home / ".local" / "state" / "pastore"
    if make_dir:
        state_dir.mkdir(parents=True)
    return state_dir / "gateway-policy.yaml"


def start(policy: Path) -> int:
    """Run the entry point on `policy` with the run_app stub in place."""
    return entry.main(["--policy", str(policy), "--host", "127.0.0.1", "--port", "0"])


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
    assert entry.main(["--policy", str(EXAMPLE_POLICY), "--host", "127.0.0.1", "--port", "0"]) == 0
    assert isinstance(seen["app"], web.Application)
    assert (seen["host"], seen["port"], seen["handler_cancellation"]) == ("127.0.0.1", 0, True)


def test_unit_policy_argument_is_the_repo_policy_yaml() -> None:
    """The unit's --policy names policy.yaml under its WorkingDirectory (%h/pastore); load_policy accepts the example it is copied from."""
    lines = {k: v for k, _, v in (line.partition("=") for line in UNIT.read_text().splitlines())}
    argv = shlex.split(lines["ExecStart"])
    arg = Path(argv[argv.index("--policy") + 1])
    assert arg.parts[0] == "%h"
    assert arg.relative_to(lines["WorkingDirectory"]) == Path("policy.yaml")
    policy = load_policy(EXAMPLE_POLICY)
    assert "workstation" in policy.hosts and len(policy.hosts) >= 2


def test_a_good_policy_writes_the_copy_with_identical_bytes(
    tmp_path: Path, home: Path, served: list[Policy]
) -> None:
    """A policy that loads is kept byte for byte as the last good copy, replacing an older one without leaving a temp file."""
    policy = tmp_path / "policy.yaml"
    policy.write_bytes(GOOD.encode())
    copy = copy_of(home)
    copy.write_text(OTHER)
    assert start(policy) == 0
    assert copy.read_bytes() == policy.read_bytes()
    assert served[0].hosts["workstation"].context == 32768
    assert sorted(p.name for p in copy.parent.iterdir()) == [copy.name]


def test_a_missing_copy_is_written_from_the_policy(
    tmp_path: Path, home: Path, served: list[Policy]
) -> None:
    """With the state directory present and no copy yet, the first start creates the copy."""
    policy = tmp_path / "policy.yaml"
    policy.write_text(GOOD)
    copy = copy_of(home)
    assert start(policy) == 0
    assert copy.read_text() == GOOD


def test_identical_bytes_are_not_rewritten(
    tmp_path: Path, home: Path, served: list[Policy]
) -> None:
    """When the copy already equals the policy, the start leaves the file as it is (same inode, same mtime)."""
    policy = tmp_path / "policy.yaml"
    policy.write_text(GOOD)
    copy = copy_of(home)
    copy.write_text(GOOD)
    os.utime(copy, ns=(1_000_000_000, 1_000_000_000))
    before = copy.stat()
    assert start(policy) == 0
    after = copy.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_a_bad_policy_starts_on_the_good_copy_and_warns(
    tmp_path: Path,
    home: Path,
    served: list[Policy],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy that fails to load starts the gateway on the copy, with a journal warning, and leaves the copy untouched."""
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n")
    copy = copy_of(home)
    copy.write_text(OTHER)
    os.utime(copy, ns=(1_000_000_000, 1_000_000_000))
    before = copy.stat()
    logged: list[str] = []
    monkeypatch.setattr(entry, "_log", logged.append)
    assert start(policy) == 0
    err = capsys.readouterr().err
    assert err.startswith(WARNING) and err.count("\n") == 1
    assert f"policy {policy} failed to load (" in err and "version must be 2" in err
    assert f"started on the last good copy {copy}" in err
    assert served[0].hosts["workstation"].context == 16384
    assert (copy.read_text(), copy.stat().st_mtime_ns) == (OTHER, before.st_mtime_ns)
    assert logged[-1].endswith(f"policy={copy}")


@pytest.mark.parametrize("kind", ["missing", "directory", "bad-utf8"])
def test_an_unreadable_policy_file_falls_back_to_the_copy(
    tmp_path: Path, home: Path, served: list[Policy], capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """A missing file, a directory or bytes that are not UTF-8 count as a load failure, not a crash."""
    policy = tmp_path / "policy.yaml"
    if kind == "directory":
        policy.mkdir()
    elif kind == "bad-utf8":
        policy.write_bytes(b"\xff\xfe version: 2\n")
    copy = copy_of(home)
    copy.write_text(OTHER)
    assert start(policy) == 0
    assert f"started on the last good copy {copy}" in capsys.readouterr().err
    assert served[0].hosts["workstation"].context == 16384


def test_a_bad_policy_without_a_copy_exits_2(
    tmp_path: Path, home: Path, served: list[Policy], capsys: pytest.CaptureFixture[str]
) -> None:
    """With nothing to fall back to, a bad policy exits 2 as before and nothing is served."""
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n")
    copy = copy_of(home)
    assert start(policy) == 2
    err = capsys.readouterr().err
    assert "version must be 2" in err and f"last good copy {copy}: cannot read" in err
    assert served == []


def test_a_bad_policy_with_a_bad_copy_exits_2_naming_both(
    tmp_path: Path, home: Path, served: list[Policy], capsys: pytest.CaptureFixture[str]
) -> None:
    """When the copy does not load either, both reasons are printed and the exit is 2."""
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n")
    copy = copy_of(home)
    copy.write_text("jobs: []\n")
    assert start(policy) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "version must be 2" in err and f"last good copy {copy}: " in err and "missing key" in err
    assert served == []
    assert copy.read_text() == "jobs: []\n"


def test_a_missing_state_directory_warns_and_still_starts(
    tmp_path: Path, home: Path, served: list[Policy], capsys: pytest.CaptureFixture[str]
) -> None:
    """The gateway never creates the state directory: it warns that the copy cannot be kept and serves the policy."""
    policy = tmp_path / "policy.yaml"
    policy.write_text(GOOD)
    copy = copy_of(home, make_dir=False)
    assert start(policy) == 0
    err = capsys.readouterr().err
    assert err.startswith(f"{WARNING}could not keep the last good policy copy {copy}: ")
    assert err.count("\n") == 1
    assert not copy.parent.exists()
    assert served[0].hosts["workstation"].context == 32768


def test_a_failing_write_warns_and_still_starts(
    tmp_path: Path,
    home: Path,
    served: list[Policy],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write that fails (here the replace) warns once and starts on the policy; the old copy is intact."""
    policy = tmp_path / "policy.yaml"
    policy.write_text(GOOD)
    copy = copy_of(home)
    copy.write_text(OTHER)

    def deny(src: Path, dst: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(entry.os, "replace", deny)
    assert start(policy) == 0
    err = capsys.readouterr().err
    assert err.startswith(f"{WARNING}could not keep the last good policy copy {copy}: ")
    assert err.count("\n") == 1 and "denied" in err
    assert copy.read_text() == OTHER
    assert [p.name for p in copy.parent.iterdir()] == [copy.name]
    assert served[0].hosts["workstation"].context == 32768


def test_a_multi_line_load_error_is_one_warning_line(
    tmp_path: Path, home: Path, served: list[Policy], capsys: pytest.CaptureFixture[str]
) -> None:
    """A YAML syntax error spans several lines; the fallback warning must still be one `<4>` line naming the copy."""
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 2\nhosts: [\n")
    with pytest.raises(PolicyError) as info:
        parse_policy(policy.read_text())
    assert "\n" in str(info.value)
    copy = copy_of(home)
    copy.write_text(OTHER)
    assert start(policy) == 0
    err = capsys.readouterr().err
    assert err.startswith(WARNING) and err.count("\n") == 1
    assert f"started on the last good copy {copy}" in err


def test_a_multi_line_load_error_is_one_line_when_exiting_2(
    tmp_path: Path, home: Path, served: list[Policy], capsys: pytest.CaptureFixture[str]
) -> None:
    """With no copy to fall back to, a multi-line error is still one stderr line."""
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 2\nhosts: [\n")
    copy_of(home)
    assert start(policy) == 2
    err = capsys.readouterr().err
    assert err.startswith("pastore gateway: invalid YAML") and err.count("\n") == 1


def test_a_planted_symlink_is_never_followed(
    tmp_path: Path, home: Path, served: list[Policy]
) -> None:
    """A symlink named like the old fixed temp file is neither written through nor installed as the copy."""
    policy = tmp_path / "policy.yaml"
    policy.write_text(GOOD)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    copy = copy_of(home)
    for name in (copy.name + ".tmp", ".gateway-policy-.tmp"):
        (copy.parent / name).symlink_to(victim)
    assert start(policy) == 0
    assert victim.read_text() == "precious"
    assert not copy.is_symlink() and copy.read_text() == GOOD


def test_the_policy_file_is_read_once(
    tmp_path: Path, home: Path, served: list[Policy], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes parsed are the bytes written: a second read of the policy would see the swapped file and the copy would differ."""
    policy = tmp_path / "policy.yaml"
    policy.write_text(GOOD)
    swap = tmp_path / "swap.yaml"
    swap.write_text(OTHER)
    copy = copy_of(home)
    opens: list[Path] = []
    real_open = Path.open

    def spy(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        if self == policy:
            opens.append(self)
            if len(opens) == 1:
                os.replace(swap, policy)
        return handle

    monkeypatch.setattr(Path, "open", spy)
    assert start(policy) == 0
    assert len(opens) == 1
    assert policy.read_text() == OTHER
    assert (copy.read_text(), served[0].hosts["workstation"].context) == (GOOD, 32768)
