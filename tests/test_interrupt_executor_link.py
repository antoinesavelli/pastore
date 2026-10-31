"""R34 executor link drop mid-run: a run cut short by a dead executor is released uncharged the first time (D-052 (1)) and re-placed; the same end with the link up is charged.

Depends on: tests/test_dispatcher.py (Rig, HostSpec, task, host, INFO), the real hosts.http_json over fake-Ollama child processes, a real job via runner.start_run, FakeClock.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from subprocess import CompletedProcess
from typing import cast

import pytest

from pastore.checks import Host
from pastore.hosts import http_json
from pastore.runner import RunDir, RunHandle, start_run
from pastore.spec import WORKSTATION, TaskSpec
from pastore.state import RunRow
from tests.test_dispatcher import (
    EXECUTOR,
    GIB,
    INFO,
    MODEL,
    HostSpec,
    Rig,
    host,
    task,
)

ROOT = Path(__file__).resolve().parents[1]
DEADLINE_S = 10.0
LIVE_PORTS = (11434, 11435, 11436)
KILLED = -int(signal.SIGKILL)

# A fake Ollama in ok mode on the given port (0 = any free one); it writes its port once it listens and ends itself after 120 s.
FAKE_CHILD = """\
import json, os, sys, time
import pastore.fake_ollama as fake_ollama

port, portfile = int(sys.argv[1]), sys.argv[2]
models, show = json.loads(sys.argv[3]), json.loads(sys.argv[4])
server = fake_ollama.ThreadingHTTPServer
fake_ollama.ThreadingHTTPServer = lambda _addr, handler: server(("127.0.0.1", port), handler)
with fake_ollama.FakeOllama("ok", models=models, show=show) as fake:
    with open(portfile + ".tmp", "w") as out:
        out.write(fake.url.rsplit(":", 1)[1])
    os.replace(portfile + ".tmp", portfile)
    time.sleep(120)
"""

# The job polls its Ollama every 0.1 s. "poll" exits 3 when it cannot connect, "hang" never exits by itself, "fail" exits 1 after one good poll.
JOB = """\
import http.client, os, pathlib, sys, time, urllib.request

mode = sys.argv[1]
url = os.environ["OLLAMA_HOST"] + "/api/version"
out = pathlib.Path(os.environ["PASTORE_RUN_DIR"])
deadline = time.monotonic() + 90
while time.monotonic() < deadline:
    try:
        urllib.request.urlopen(url, timeout=1).read()
    except (OSError, http.client.HTTPException):
        if mode != "hang":
            sys.exit(3)
    else:
        (out / "polling").write_text("1")
        if mode == "fail":
            sys.exit(1)
    time.sleep(0.1)
sys.exit(99)
"""


def wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = DEADLINE_S) -> None:
    """Poll every 0.05 s; fail the test with `what` after `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.05)


class FakeChild:
    """A fake-Ollama child process and the port it listens on."""

    def __init__(self, proc: subprocess.Popen[bytes], port: int) -> None:
        self.proc = proc
        self.port = port

    @property
    def url(self) -> str:
        """The base URL the dispatcher's policy names."""
        return f"http://127.0.0.1:{self.port}"

    def kill(self) -> None:
        """SIGKILL it, as a dropped link looks from the dispatcher, and confirm it died by that signal."""
        os.killpg(self.proc.pid, signal.SIGKILL)
        assert self.proc.wait(timeout=DEADLINE_S) == KILLED


class Children:
    """Every process the test started; `reap` SIGKILLs the live ones' process groups and runs the cleanups."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.procs: list[subprocess.Popen[bytes]] = []
        self.cleanups: list[Callable[[], None]] = []
        self.spawned = 0
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        self.env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    def fake_ollama(self, port: int = 0) -> FakeChild:
        """Start a fake Ollama child (on `port`, or a free one) and wait until it listens."""
        assert port not in LIVE_PORTS
        self.spawned += 1
        portfile = self.tmp_path / f"fake-{self.spawned}.port"
        with (self.tmp_path / f"fake-{self.spawned}.log").open("wb") as log:
            proc = subprocess.Popen(
                [sys.executable, "-c", FAKE_CHILD, str(port), str(portfile)]
                + [json.dumps({MODEL: GIB}), json.dumps({MODEL: INFO})],
                cwd=ROOT,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.procs.append(proc)
        wait_for(lambda: portfile.exists() or proc.poll() is not None, "a fake Ollama to listen")
        assert portfile.exists(), f"fake Ollama exited {proc.returncode} before it listened"
        bound = int(portfile.read_text())
        assert bound not in LIVE_PORTS and (port in (0, bound))
        return FakeChild(proc, bound)

    def reap(self) -> None:
        """Kill every child's process group, then wait and run the cleanups; one failure never skips the rest."""
        for proc in self.procs:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        try:
            for proc in self.procs:
                proc.wait(timeout=DEADLINE_S)
        finally:
            for cleanup in self.cleanups:
                cleanup()


@pytest.fixture
def children(tmp_path: Path) -> Iterator[Children]:
    """The test's fake Ollama children, every one reaped at teardown."""
    kids = Children(tmp_path)
    try:
        yield kids
    finally:
        kids.reap()


class Fleet:
    """The test_dispatcher Rig over a workstation and an executor fake, with a real job placed on the host its run row names."""

    def __init__(self, tmp_path: Path, kids: Children, mode: str, timeout_s: int = 3600) -> None:
        self.kids = kids
        self.ws = kids.fake_ollama()
        self.ex = kids.fake_ollama()
        self.calls: list[list[str]] = []
        self.jobs: dict[str, subprocess.Popen[bytes]] = {}
        self.canned = host()
        script = tmp_path / "job.py"
        script.write_text(JOB)
        spec = task(tmp_path, "a", argv=(sys.executable, str(script), mode), timeout_s=timeout_s)
        hosts = {
            WORKSTATION: HostSpec(self.ws.url, 24, 3, 3, 32768),
            EXECUTOR: HostSpec(self.ex.url, 10, 3, 3, 8192),
        }
        stub = Host(
            run=self.run, read_text=self.canned.read_text, http_status=self.canned.http_status
        )
        self.rig = Rig(
            tmp_path,
            [spec],
            [("c/a", MODEL)],
            hosts=hosts,
            host=stub,
            start=self.start,
            http_json=http_json,
            ollama_url=self.ws.url,
            base_env=kids.env,
        )
        kids.cleanups.append(self.rig.state.close)

    def run(self, argv: Sequence[str]) -> CompletedProcess[str]:
        """The host.run stub: canned driver answers, and a scope stop SIGKILLs that run's real job group."""
        self.calls.append(list(argv))
        if argv[0] in ("dpkg-query", "nvidia-smi"):
            return self.canned.run(argv)
        if list(argv[:3]) == ["systemctl", "--user", "stop"]:
            job = self.jobs[argv[3].removeprefix("pastore-").removesuffix(".scope")]
            if job.poll() is None:
                os.killpg(job.pid, signal.SIGKILL)
        return CompletedProcess(list(argv), 0, "", "")

    def start(
        self,
        spec: TaskSpec,
        run_dir: RunDir,
        run_id: str,
        url: str,
        model: str,
        env: Mapping[str, str],
    ) -> RunHandle:
        """Launch the real job against the Ollama of the host the run row was placed on."""
        row = self.rig.state.get_run(run_id)
        assert row is not None
        handle = start_run(
            spec,
            run_dir,
            run_id,
            self.rig.hosts[row.host].url,
            model,
            env,
            wrap=lambda _r, argv, _t: list(argv),
        )
        job = cast(subprocess.Popen[bytes], handle.proc)
        self.kids.procs.append(job)
        self.jobs[run_id] = job
        return handle

    def tick(self) -> bool:
        """One dispatcher tick on the fake clock."""
        return self.rig.tick()

    def hold_placement(self, held: bool) -> None:
        """Create or remove the stop file, so a tick still ends runs but places nothing."""
        stop_file = self.rig.paths.stop_file
        if held:
            stop_file.write_text("")
        else:
            stop_file.unlink()

    def running(self) -> RunRow:
        """The one run in flight."""
        [run] = self.rig.state.runs_with_status("running")
        return run

    def start_on_executor(self) -> RunRow:
        """Tick once, expect the pair on the executor, and wait until its job has polled that executor."""
        self.tick()
        run = self.running()
        assert run.host == EXECUTOR
        self.wait_polling(run)
        return run

    def wait_polling(self, run: RunRow) -> None:
        """Wait for the run's job to have reached its Ollama once."""
        marker = self.rig.paths.runs_dir / run.run_id / "out" / "polling"
        wait_for(marker.exists, f"the job of {run.run_id} to reach {run.host}")

    def wait_exit(self, run: RunRow) -> int:
        """Wait for the run's job to exit by itself and return its exit code."""
        job = self.jobs[run.run_id]
        wait_for(lambda: job.poll() is not None, f"the job of {run.run_id} to exit")
        assert job.returncode is not None
        return job.returncode

    def journal_calls(self) -> list[list[str]]:
        """Every kernel-log read the dispatcher made (the R35 watch's `journalctl -u` reads are not these)."""
        return [c for c in self.calls if c[:2] == ["journalctl", "-k"]]

    def run_row(self, run: RunRow) -> RunRow:
        """The run's row as stored now."""
        row = self.rig.state.get_run(run.run_id)
        assert row is not None
        return row

    def job_counts(self) -> tuple[str, int, int]:
        """The pair's job row as (status, attempts, crash_streak)."""
        job = self.rig.state.get_job("c/a", MODEL)
        assert job is not None
        return job.status, job.attempts, job.crash_streak


def release_with_the_executor_dead(fleet: Fleet) -> RunRow:
    """Start on the executor, drop it, wait for the job to exit 3, and tick with placement held; returns the first run."""
    run = fleet.start_on_executor()
    fleet.ex.kill()
    assert fleet.wait_exit(run) == 3
    fleet.hold_placement(True)
    fleet.tick()
    return run


def assert_released_uncharged(fleet: Fleet, run: RunRow, exit_code: int) -> None:
    """The run is `stopped`/`executor-down` on its own host and model, and its pair is pending: no attempt charged, one outage end in its streak."""
    ended = fleet.run_row(run)
    assert (ended.status, ended.detail, ended.exit_code) == ("stopped", "executor-down", exit_code)
    assert (ended.host, ended.model, ended.ref) == (EXECUTOR, MODEL, "c/a")
    assert ended.ended_at is not None
    assert fleet.job_counts() == ("pending", 0, 1)
    assert fleet.rig.state.runs_with_status("failed") == []
    assert fleet.rig.state.runs_with_status("timeout") == []
    assert fleet.rig.events("failed") == []
    [end] = [e for e in fleet.rig.events("run-end") if e.detail.startswith(run.run_id)]
    assert end.detail == f"{run.run_id} stopped {exit_code} executor-down"
    assert fleet.rig.state.runs_with_status("running") == [] and not fleet.rig.d.active


def test_r34_a_run_cut_by_a_dropped_executor_link_is_released_uncharged_then_replaced(
    tmp_path: Path, children: Children
) -> None:
    fleet = Fleet(tmp_path, children, "poll")
    run = release_with_the_executor_dead(fleet)
    assert_released_uncharged(fleet, run, 3)
    # The OOM check comes first (D-052 (3)): one kernel-log read, and it finds no kill.
    assert len(fleet.journal_calls()) == 1
    assert fleet.rig.state.last_run("c/a", MODEL) == fleet.run_row(run)
    fleet.hold_placement(False)
    fleet.tick()
    again = fleet.running()
    assert again.run_id != run.run_id and (again.ref, again.model) == (run.ref, run.model)
    assert again.host == WORKSTATION
    fleet.wait_polling(again)
    assert fleet.job_counts() == ("running", 0, 1)


def test_r34_an_executor_back_on_the_same_port_takes_the_released_pair_again(
    tmp_path: Path, children: Children
) -> None:
    fleet = Fleet(tmp_path, children, "poll")
    port = fleet.ex.port
    run = release_with_the_executor_dead(fleet)
    assert_released_uncharged(fleet, run, 3)
    fleet.ex = children.fake_ollama(port)
    assert fleet.ex.port == port
    fleet.hold_placement(False)
    fleet.tick()
    again = fleet.running()
    assert again.run_id != run.run_id and again.host == EXECUTOR
    fleet.wait_polling(again)
    assert fleet.job_counts() == ("running", 0, 1)
    assert fleet.rig.events("failed") == []


def test_r34_the_same_nonzero_exit_with_the_executor_up_is_a_counted_failure(
    tmp_path: Path, children: Children
) -> None:
    fleet = Fleet(tmp_path, children, "fail")
    fleet.tick()
    run = fleet.running()
    assert run.host == EXECUTOR
    assert fleet.wait_exit(run) == 1
    assert fleet.ex.proc.poll() is None
    fleet.hold_placement(True)
    fleet.tick()
    ended = fleet.run_row(run)
    assert (ended.status, ended.detail, ended.exit_code) == ("failed", "", 1)
    assert fleet.job_counts() == ("pending", 1, 0)
    assert fleet.rig.events("failed") == []


@pytest.mark.parametrize("link_down", [True, False])
def test_r34_a_timeout_is_released_uncharged_only_while_the_executor_link_is_down(
    tmp_path: Path, children: Children, link_down: bool
) -> None:
    fleet = Fleet(tmp_path, children, "hang", timeout_s=60)
    run = fleet.start_on_executor()
    if link_down:
        fleet.ex.kill()
    fleet.rig.clock.advance(61)
    fleet.hold_placement(True)
    fleet.tick()
    stop = ["systemctl", "--user", "stop", f"pastore-{run.run_id}.scope"]
    assert fleet.calls.count(stop) == 1
    assert fleet.jobs[run.run_id].returncode == KILLED
    if link_down:
        assert_released_uncharged(fleet, run, KILLED)
        assert fleet.journal_calls() == []
        return
    ended = fleet.run_row(run)
    assert (ended.status, ended.detail, ended.exit_code) == ("timeout", "", KILLED)
    assert fleet.job_counts() == ("pending", 1, 0)
