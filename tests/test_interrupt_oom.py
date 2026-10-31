"""R34 "job OOM-killed": a job the kernel OOM killer takes is charged as any failure and its run is marked `oom`.

Depends on: pastore.dispatcher over the Rig of tests/test_dispatcher.py; real sleeping jobs started by runner.start_run and SIGKILLed; a scripted kernel log.
"""

import os
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from pastore.runner import RunDir, RunHandle, start_run
from pastore.spec import TaskSpec
from tests.fakes import RecordingRunner
from tests.test_dispatcher import (
    EX_URL,
    EXECUTOR,
    GIB,
    MODEL,
    T0,
    Rig,
    fleet_rig,
    net_of,
    task,
)

SLICE = "/user.slice/user-1000.slice/user@1000.service/app.slice"
CLAUDE_SCOPE = f"{SLICE}/app-com.anthropic.Claude-23510.scope"
OTHER_RUN = "20261024T010000-7"
NOISE = "Memory cgroup out of memory: Killed process 219992 (python) total-vm:9028kB"
Journal = Callable[[str], str]


def pastore_scope(run_id: str) -> str:
    """The cgroup path of a pastore job's scope, as the kernel names it."""
    return f"{SLICE}/heavy-jobs.slice/pastore-{run_id}.scope"


def global_oom(task_memcg: str) -> str:
    """A global OOM kill line, shaped like the 2026-10-30 18:24 one."""
    return (
        "oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=user.slice,mems_allowed=0,"
        f"global_oom,task_memcg={task_memcg},task=python,pid=184076,uid=1000"
    )


def memcg_oom(oom_memcg: str, task_memcg: str) -> str:
    """A cgroup-limit OOM kill line, shaped like the 2026-10-30 18:24 one."""
    return (
        "oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=user.slice,mems_allowed=0,"
        f"oom_memcg={oom_memcg},task_memcg={task_memcg},task=python,pid=219992,uid=1000"
    )


def log(*lines: str) -> str:
    """A kernel log answer: the lines, one per row."""
    return "".join(f"{line}\n" for line in lines)


def kernel_log(line_of: Journal) -> Journal:
    """A journal answer for a run id: an unrelated global OOM, the given line, noise."""
    return lambda rid: log(global_oom(CLAUDE_SCOPE), line_of(rid), NOISE)


KILLED: dict[str, Journal] = {
    "global": kernel_log(lambda rid: global_oom(pastore_scope(rid))),
    "memcg-limit-on-the-slice": kernel_log(
        lambda rid: memcg_oom(f"{SLICE}/heavy-jobs.slice", pastore_scope(rid))
    ),
    "memcg-limit-on-the-scope": kernel_log(
        lambda rid: memcg_oom(pastore_scope(rid), pastore_scope(rid))
    ),
}
NOT_KILLED: dict[str, Journal] = {
    "empty": lambda _rid: "",
    "only-noise": lambda _rid: log(NOISE),
    "another-scope": lambda _rid: log(
        global_oom(CLAUDE_SCOPE), memcg_oom(pastore_scope(OTHER_RUN), pastore_scope(OTHER_RUN))
    ),
    "prefix-trap": kernel_log(lambda rid: global_oom(pastore_scope(rid + "0"))),
    "leading-trap": kernel_log(
        lambda rid: global_oom(pastore_scope(rid).replace("/pastore-", "/not-pastore-"))
    ),
    "trailing-trap": kernel_log(lambda rid: global_oom(pastore_scope(rid) + ".bak")),
    "scope-only-in-oom_memcg": kernel_log(lambda rid: memcg_oom(pastore_scope(rid), CLAUDE_SCOPE)),
    "not-at-line-start": kernel_log(lambda rid: "kernel: " + global_oom(pastore_scope(rid))),
}


class KillRig:
    """A Rig whose job is a real sleeping process; `kill_all` is the teardown."""

    def __init__(self, tmp_path: Path) -> None:
        self.procs: list[subprocess.Popen[bytes]] = []
        self.rig = Rig(tmp_path, [], [], start=self.start)
        spec = task(
            self.rig.root,
            "a",
            argv=(sys.executable, "-c", "import time; time.sleep(60)"),
            env={"HOME": str(tmp_path / "home")},
        )
        self.rig.specs["a"] = spec
        self.rig.state.record_contract_pass(spec.ref, spec.fingerprint, T0)
        self.rig.list_jobs(["c/a"])
        runner = self.rig.deps.host.run
        assert isinstance(runner, RecordingRunner)
        self.runner = runner

    def popen(self, argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        """Start the job for real, remembering it for the teardown."""
        proc = subprocess.Popen(list(argv), **kwargs)
        self.procs.append(proc)
        return proc

    def start(
        self,
        spec: TaskSpec,
        run_dir: RunDir,
        run_id: str,
        url: str,
        model: str,
        env: Mapping[str, str],
    ) -> RunHandle:
        """The dispatcher's `start`: the real runner without the systemd scope."""
        return start_run(
            spec,
            run_dir,
            run_id,
            url,
            model,
            env,
            popen=self.popen,
            wrap=lambda _run_id, argv, _timeout_s: list(argv),
        )

    def sigkill_and_tick(
        self, answer: Journal = lambda _rid: "", rc: int = 0, *, hold: bool = False
    ) -> str:
        """SIGKILL the job as the OOM killer would, then tick; `answer` scripts the kernel log, `hold` blocks the next start."""
        [active] = self.rig.d.active.values()
        proc = self.procs[-1]
        os.killpg(proc.pid, signal.SIGKILL)
        assert proc.wait(timeout=10) == -signal.SIGKILL
        self.runner.results["journalctl"] = RecordingRunner.ok(answer(active.run_id), rc)
        if hold:
            self.rig.paths.stop_file.write_text("")
        self.rig.clock.advance(5)
        self.rig.tick()
        return active.run_id

    def kernel_log_reads(self) -> list[list[str]]:
        """Every kernel-log argv the dispatcher ran (the R35 watch's `journalctl -u` reads are not these)."""
        return [c for c in self.runner.calls if c[:2] == ["journalctl", "-k"]]

    def kill_all(self) -> None:
        """Kill every job still alive, with the rest of its process group."""
        for proc in self.procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


@pytest.fixture
def kill_rig(tmp_path: Path) -> Iterator[KillRig]:
    rig = KillRig(tmp_path)
    try:
        yield rig
    finally:
        rig.kill_all()


def journalctl_argv(started_at: float) -> list[str]:
    """The exact kernel-log read for a run that started at this epoch second."""
    return ["journalctl", "-k", "-o", "cat", "--no-pager", f"--since=@{int(started_at)}"]


def reads_of(rig: Rig) -> list[list[str]]:
    """Every kernel-log argv a plain Rig's host ran."""
    runner = rig.deps.host.run
    assert isinstance(runner, RecordingRunner)
    return [c for c in runner.calls if c[:2] == ["journalctl", "-k"]]


@pytest.mark.parametrize("shape", sorted(KILLED))
def test_r34_a_job_the_oom_killer_took_is_failed_oom_and_charged(
    kill_rig: KillRig, shape: str
) -> None:
    rig = kill_rig.rig
    rig.tick()
    run_id = kill_rig.sigkill_and_tick(KILLED[shape], hold=True)
    run = rig.state.get_run(run_id)
    job = rig.state.get_job("c/a", MODEL)
    assert run is not None and job is not None
    assert (run.status, run.detail, run.exit_code) == ("failed", "oom", -signal.SIGKILL)
    assert (run.host, run.model, run.ended_at is not None) == ("workstation", MODEL, True)
    assert (job.attempts, job.crash_streak, job.status) == (1, 0, "pending")
    assert [e.detail for e in rig.events("run-end")] == [f"{run_id} failed -9 oom"]
    assert kill_rig.kernel_log_reads() == [journalctl_argv(T0.timestamp())]


@pytest.mark.parametrize("shape", sorted(NOT_KILLED))
def test_r34_a_sigkill_without_an_oom_line_for_its_scope_is_a_plain_failure(
    kill_rig: KillRig, shape: str
) -> None:
    rig = kill_rig.rig
    rig.tick()
    run_id = kill_rig.sigkill_and_tick(NOT_KILLED[shape], hold=True)
    run = rig.state.get_run(run_id)
    job = rig.state.get_job("c/a", MODEL)
    assert run is not None and job is not None
    assert (run.status, run.detail, run.exit_code) == ("failed", "", -signal.SIGKILL)
    assert (job.attempts, job.status) == (1, "pending")
    assert [e.detail for e in rig.events("run-end")] == [f"{run_id} failed -9"]
    assert len(kill_rig.kernel_log_reads()) == 1


@pytest.mark.parametrize("rc", [1, 124, 127])
def test_r34_a_kernel_log_that_cannot_be_read_is_a_plain_failure(
    kill_rig: KillRig, rc: int
) -> None:
    rig = kill_rig.rig
    rig.tick()
    run_id = kill_rig.sigkill_and_tick(KILLED["global"], rc, hold=True)
    run = rig.state.get_run(run_id)
    job = rig.state.get_job("c/a", MODEL)
    assert run is not None and job is not None
    assert (run.status, run.detail) == ("failed", "")
    assert (job.attempts, job.status) == (1, "pending")
    assert [e.detail for e in rig.events("run-end")] == [f"{run_id} failed -9"]


def test_r34_the_third_oom_kill_marks_the_job_failed(kill_rig: KillRig) -> None:
    rig = kill_rig.rig
    rig.tick()
    run_ids = [kill_rig.sigkill_and_tick(KILLED["global"]) for _ in range(3)]
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None and (job.attempts, job.status) == (3, "failed")
    assert [(r.run_id, r.detail) for r in rig.state.runs_with_status("failed")] == [
        (run_id, "oom") for run_id in run_ids
    ]
    assert [e.detail for e in rig.events("failed")] == [f"{MODEL}: 3 attempts"]
    assert len(kill_rig.procs) == 3 and rig.d.active == {}
    assert len(kill_rig.kernel_log_reads()) == 3


def test_r34_each_run_is_matched_to_its_own_scope(kill_rig: KillRig) -> None:
    rig = kill_rig.rig
    rig.tick()
    first = kill_rig.sigkill_and_tick(NOT_KILLED["another-scope"])
    second = kill_rig.sigkill_and_tick(KILLED["global"], hold=True)
    ends = {r.run_id: r.detail for r in rig.state.runs_with_status("failed")}
    assert first != second and ends == {first: "", second: "oom"}
    assert [c[-1] for c in kill_rig.kernel_log_reads()] == [
        f"--since=@{int(T0.timestamp())}",
        f"--since=@{int(T0.timestamp()) + 5}",
    ]


def test_r34_an_ok_end_never_reads_the_kernel_log(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.next_codes = [0]
    rig.tick()
    rig.tick()
    assert [r.status for r in rig.state.runs_with_status("done")] == ["done"]
    assert reads_of(rig) == []


def test_r34_a_timeout_never_reads_the_kernel_log(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a", timeout_s=60)], ["c/a"])
    rig.next_codes = [None]
    rig.tick()
    rig.clock.advance(61)
    rig.tick()
    [timed_out] = rig.state.runs_with_status("timeout")
    assert (timed_out.detail, timed_out.exit_code) == ("", -15)
    assert reads_of(rig) == []


def test_r34_a_stop_never_reads_the_kernel_log(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.next_codes = [None]
    rig.tick()
    rig.state.request_stop("now")
    assert rig.tick() is False
    assert [r.status for r in rig.state.runs_with_status("stopped")] == ["stopped"]
    assert reads_of(rig) == []


def test_r34_a_release_for_a_lost_executor_never_reads_the_kernel_log(tmp_path: Path) -> None:
    rig = fleet_rig(tmp_path, net_of({}, {"only": GIB}), [("c/a", "only")], models=("only",))
    rig.next_codes = [1]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == EXECUTOR
    rig.net.version_down.add(EX_URL)
    rig.tick()
    ended = rig.state.get_run(run.run_id)
    assert ended is not None and (ended.status, ended.detail) == ("stopped", "executor-down")
    assert reads_of(rig) == []


def test_r34_a_job_that_never_started_is_not_an_oom_kill(tmp_path: Path) -> None:
    def start(*_args: object) -> object:
        raise FileNotFoundError("./job")

    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], start=start)
    rig.tick()
    run = rig.state.last_run("c/a", MODEL)
    assert run is not None and (run.status, run.detail, run.exit_code) == ("failed", "", 127)
    assert reads_of(rig) == []
