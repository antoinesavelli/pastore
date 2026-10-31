"""D-050 (6): a tick is bounded by a SIGALRM timer; an overrun is a tick-error, a start, an end or a stop in progress finishes first, and a running job is never touched.

Depends on: pastore.tick_bound, tests/test_dispatcher.py (Rig, FakeHandle, host, task), tests/test_dispatcher_gateway_outage.py (Gateway), the real `serve` loop with a fake sleep, real short bounds and blocking fakes.
"""

import contextlib
import inspect
import signal
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from pastore import dispatcher as dispatcher_module
from pastore import policy as policy_module
from pastore.checks import Host
from pastore.dispatcher import Dispatcher, serve
from pastore.policy import PolicyCache
from pastore.runner import RunDir
from pastore.spec import Ready, TaskSpec
from pastore.tick_bound import TickOverrun
from tests.test_dispatcher import MODEL, FakeHandle, FakeNet, Rig, host, task
from tests.test_dispatcher_gateway_outage import Gateway, not_a_gateway

BOUND_S = 0.3
BLOCK_S = 2.0  # what a fake blocks for when nothing interrupts it
LATE_S = 0.6  # a critical section that outlives the bound by 0.3 s
NO_TIMER = (0.0, 0.0)


@pytest.fixture(autouse=True)
def clean_alarm() -> Iterator[None]:
    """No test leaves a timer armed or a SIGALRM or SIGTERM handler of its own behind."""
    before = signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before[0])
        signal.signal(signal.SIGTERM, before[1])


class Blocker:
    """The injected run_in: blocks on the argv[0] values in `block` (for `delay_s`), and answers `rc` for everything else."""

    def __init__(self, block: Sequence[str] = (), delay_s: float = BLOCK_S, rc: int = 0) -> None:
        self.block = set(block)
        self.delay_s = delay_s
        self.rc = rc
        self.calls: list[str] = []

    def __call__(
        self, argv: Sequence[str], cwd: Path, env: Mapping[str, str]
    ) -> CompletedProcess[str]:
        self.calls.append(argv[0])
        if argv[0] in self.block:
            time.sleep(self.delay_s)
        return CompletedProcess(list(argv), self.rc, "", "")


class Failing:
    """The injected run_in: works for `work_s`, then fails."""

    def __init__(self, work_s: float) -> None:
        self.work_s = work_s

    def __call__(
        self, argv: Sequence[str], cwd: Path, env: Mapping[str, str]
    ) -> CompletedProcess[str]:
        time.sleep(self.work_s)
        raise RuntimeError("the check failed")


class SlowHandle(FakeHandle):
    """A handle whose stop takes `LATE_S`."""

    def stop(self, run: object, wait_s: float = 30.0) -> int:
        time.sleep(LATE_S)
        return super().stop(run, wait_s)


class Drive:
    """Runs the real `serve` over a Rig with a fake sleep: `on_nap(n)` runs after tick n, the clock moves 5 s a nap."""

    def __init__(self, rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rig = rig
        self.dispatchers: list[Dispatcher] = []
        self.timers: list[tuple[float, float]] = []
        self.clock_timers: list[tuple[float, float]] = []
        self.naps = 0
        spy = self.dispatchers

        class Spy(Dispatcher):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]
                spy.append(self)

        monkeypatch.setattr(dispatcher_module, "Dispatcher", Spy)

    @property
    def d(self) -> Dispatcher:
        """The dispatcher `serve` built."""
        return self.dispatchers[0]

    def run(
        self,
        on_nap: Callable[[int], None],
        *,
        bound_s: float = BOUND_S,
        real_nap_s: float = 0.0,
    ) -> None:
        """Serve until the callback has requested a stop; every nap and every clock read also records the timer state."""

        def sleep(_seconds: float) -> None:
            self.naps += 1
            assert self.naps < 8, "the loop did not stop"
            self.timers.append(signal.getitimer(signal.ITIMER_REAL))
            time.sleep(real_nap_s)
            self.rig.clock.advance(5)
            on_nap(self.naps)

        def clock() -> datetime:
            self.clock_timers.append(signal.getitimer(signal.ITIMER_REAL))
            return self.rig.clock()

        deps = self.rig.deps
        assert (
            serve(
                deps,
                self.rig.policy,
                clock=clock,
                sleep=sleep,
                tick_s=1.0,
                tick_bound_s=bound_s,
            )
            == 0
        )
        assert signal.getitimer(signal.ITIMER_REAL) == NO_TIMER

    def overruns(self) -> list[str]:
        """The tick-error details, oldest first."""
        return [e.detail for e in reversed(self.rig.events("tick-error"))]


def rig_with(tmp_path: Path, specs: list[TaskSpec], **deps: object) -> Rig:
    """A Rig over these tasks, each listed with the default model."""
    return Rig(tmp_path, specs, [s.ref for s in specs], **deps)


def test_a_ready_check_that_blocks_past_the_bound_ends_the_tick_and_the_next_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = Blocker(block=["./ready"])
    spec = task(tmp_path, "a", ready=Ready(1, ("./ready",)))
    rig = rig_with(tmp_path, [spec], run_in=blocker)
    drive = Drive(rig, monkeypatch)
    started = time.monotonic()

    def on_nap(n: int) -> None:
        if n == 1:
            assert drive.overruns() == ["TickOverrun: tick ran past 0.3 s"]
            assert rig.events("run-start") == []
            blocker.block.clear()  # the second tick's check answers at once
        else:
            rig.state.request_stop("now")

    drive.run(on_nap)
    assert time.monotonic() - started < 1.5  # cut at the 0.3 s bound, not after the 2 s block
    assert drive.overruns() == ["TickOverrun: tick ran past 0.3 s"]
    assert [e.ref for e in rig.events("run-start")] == ["c/a"]  # the tick after the overrun ran
    assert drive.naps == 2


def test_invariant_a_tick_overrun_never_touches_a_running_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = Blocker(rc=1)
    running = task(tmp_path, "a")
    waiting = task(tmp_path, "b", ready=Ready(1, ("./ready",)))
    rig = rig_with(tmp_path, [running, waiting], run_in=blocker)
    rig.next_codes = [None]
    drive = Drive(rig, monkeypatch)
    seen: list[tuple[int, bool, str]] = []

    def on_nap(n: int) -> None:
        run = rig.state.runs_with_status("running")
        seen.append((len(drive.d.active), rig.handles[0].stopped, run[0].status))
        if n == 1:
            blocker.block.add("./ready")  # b's next ready check outlives the bound
        elif n == 2:
            assert len(drive.overruns()) == 1
            rig.state.request_stop("now")

    drive.run(on_nap)
    assert drive.overruns() == ["TickOverrun: tick ran past 0.3 s"]
    assert seen[:2] == [(1, False, "running"), (1, False, "running")]
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None and job.attempts == 0 and job.crash_streak == 0
    assert len(rig.handles) == 1  # the run was never restarted either


def test_an_overrun_during_a_reader_lets_the_ended_run_finish_before_the_tick_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = Blocker(block=["./read"], delay_s=LATE_S)
    rig = rig_with(tmp_path, [task(tmp_path, "a")], run_in=blocker)
    rig.next_codes = [0]
    drive = Drive(rig, monkeypatch)

    def on_nap(n: int) -> None:
        if n == 2:
            [run] = rig.events("run-start")
            row = rig.state.get_run(run.detail.split()[0])
            assert row is not None and row.status == "done"
            assert rig.job_status("a") == "done" and drive.d.active == {}
            [end] = rig.events("run-end")
            [error] = rig.events("tick-error")
            assert end.id < error.id and error.detail == "TickOverrun: tick ran past 0.3 s"
            rig.state.request_stop("now")

    drive.run(on_nap)
    assert blocker.calls.count("./read") == 1  # the next tick read nothing again
    assert len(rig.events("run-end")) == 1 and len(rig.events("done")) == 1
    assert len(drive.overruns()) == 1


def test_an_overrun_during_a_failed_runs_end_still_charges_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned = host()

    def run(argv: Sequence[str]) -> CompletedProcess[str]:
        if argv[:2] == [
            "journalctl",
            "-k",
        ]:  # `_finish` reads the kernel log: the end outlives the bound
            time.sleep(LATE_S)
        return canned.run(argv)

    net = FakeNet()
    gateway = Gateway(net)
    gateway.answer = not_a_gateway()
    slow = Host(run=run, read_text=canned.read_text, http_status=canned.http_status)
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=slow, net=net, http_json=gateway)
    rig.next_codes = [1]
    drive = Drive(rig, monkeypatch)

    def on_nap(n: int) -> None:
        if n == 2:
            [error] = rig.events("tick-error")
            [end] = rig.events("run-end")
            assert end.id < error.id and drive.d.active == {}
            rig.state.request_stop("now")

    drive.run(on_nap)
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None and (job.attempts, job.status) == (1, "pending")
    [end] = rig.events("run-end")
    assert end.detail.endswith(" failed 1") and len(rig.handles) == 1


def test_an_overrun_during_a_start_leaves_the_run_active_with_consistent_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    def slow_start(
        spec: TaskSpec, run_dir: RunDir, run_id: str, url: str, model: str, env: Mapping[str, str]
    ) -> FakeHandle:
        time.sleep(LATE_S)
        return rig.start(spec, run_dir, run_id, url, model, env)

    rig = rig_with(tmp_path, [task(tmp_path, "a")], start=slow_start)
    rig.next_codes = [None]
    drive = Drive(rig, monkeypatch)

    def on_nap(n: int) -> None:
        if n == 1:
            [run] = rig.state.runs_with_status("running")
            assert list(drive.d.active) == [run.run_id] and rig.job_status("a") == "running"
            assert [e.detail.split()[0] for e in rig.events("run-start")] == [run.run_id]
            assert len(drive.overruns()) == 1 and not rig.handles[0].stopped
        else:
            rig.state.request_stop("now")

    drive.run(on_nap)
    assert len(rig.handles) == 1 and len(rig.events("run-start")) == 1


def test_an_overrun_during_a_stop_lets_every_run_be_stopped_and_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_start(*_args: object) -> SlowHandle:
        handle = SlowHandle([None])
        rig.handles.append(handle)
        return handle

    rig = rig_with(tmp_path, [task(tmp_path, "a")], start=slow_start)
    drive = Drive(rig, monkeypatch)

    def on_nap(n: int) -> None:
        if n == 1:
            rig.state.request_stop("now")

    drive.run(on_nap)  # serve returns: the tick after the overrun saw a drained dispatcher
    [handle] = rig.handles
    assert handle.stopped and drive.d.active == {} and drive.d.draining
    assert rig.state.stop_requested() is None and rig.job_status("a") == "pending"
    [end] = rig.events("run-end")
    [stop] = rig.events("dispatcher-stop")
    [error] = rig.events("tick-error")
    assert stop.id < end.id < error.id and error.detail.startswith("TickOverrun: ")


def test_the_timer_is_armed_in_a_served_tick_and_disarmed_after_a_normal_one_and_an_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inside: list[tuple[float, float]] = []
    blocker = Blocker(rc=1)

    def probe(argv: Sequence[str], cwd: Path, env: Mapping[str, str]) -> CompletedProcess[str]:
        inside.append(signal.getitimer(signal.ITIMER_REAL))
        return blocker(argv, cwd, env)

    rig = rig_with(tmp_path, [task(tmp_path, "a", ready=Ready(1, ("./ready",)))], run_in=probe)
    drive = Drive(rig, monkeypatch)

    def on_nap(n: int) -> None:
        if n == 1:
            blocker.block.add("./ready")
        if n == 2:
            rig.state.request_stop("now")

    drive.run(on_nap)
    assert len(drive.overruns()) == 1
    assert [t[1] for t in inside] == [0.0, 0.0] and all(0 < t[0] <= BOUND_S for t in inside)
    assert drive.timers == [NO_TIMER, NO_TIMER]  # after the normal tick and after the overrun


def test_a_direct_tick_arms_no_timer(tmp_path: Path) -> None:
    inside: list[tuple[float, float]] = []

    def probe(argv: Sequence[str], cwd: Path, env: Mapping[str, str]) -> CompletedProcess[str]:
        inside.append(signal.getitimer(signal.ITIMER_REAL))
        return CompletedProcess(list(argv), 1, "", "")

    rig = rig_with(tmp_path, [task(tmp_path, "a", ready=Ready(1, ("./ready",)))], run_in=probe)
    rig.tick()
    assert inside == [NO_TIMER]


def test_serve_restores_the_previous_sigalrm_handler_and_installs_its_own_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mine(_signum: int, _frame: object) -> None:
        raise AssertionError("the test's own SIGALRM handler ran")

    signal.signal(signal.SIGALRM, mine)
    rig = rig_with(tmp_path, [task(tmp_path, "a")])
    drive = Drive(rig, monkeypatch)
    during: list[object] = []

    def on_nap(n: int) -> None:
        during.append(signal.getsignal(signal.SIGALRM))
        rig.state.request_stop("now")

    drive.run(on_nap)
    assert during and during[0] is not mine and callable(during[0])
    assert signal.getsignal(signal.SIGALRM) is mine


def test_serve_restores_the_handler_when_the_loop_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = signal.getsignal(signal.SIGALRM)
    rig = rig_with(tmp_path, [task(tmp_path, "a")])

    def sleep(_seconds: float) -> None:
        raise RuntimeError("the nap failed")

    with pytest.raises(RuntimeError, match="the nap failed"):
        serve(rig.deps, rig.policy, clock=rig.clock, sleep=sleep, tick_bound_s=BOUND_S)
    assert signal.getsignal(signal.SIGALRM) is before
    assert signal.getitimer(signal.ITIMER_REAL) == NO_TIMER


def test_no_alarm_fires_during_the_nap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = rig_with(tmp_path, [task(tmp_path, "a")])
    drive = Drive(rig, monkeypatch)

    def on_nap(n: int) -> None:
        if n == 2:
            rig.state.request_stop("now")

    drive.run(on_nap, bound_s=0.2, real_nap_s=0.5)  # each nap outlasts the bound by 0.3 s
    assert drive.naps == 2 and rig.events("tick-error") == []


def test_an_overrun_outside_a_critical_section_raises_at_once(tmp_path: Path) -> None:
    rig = rig_with(tmp_path, [task(tmp_path, "a")])
    with pytest.raises(TickOverrun, match=r"^tick ran past 60 s$"):
        rig.d.overrun(60.0)


def test_an_overrun_inside_nested_critical_sections_raises_at_the_outermost_exit(
    tmp_path: Path,
) -> None:
    rig = rig_with(tmp_path, [task(tmp_path, "a")])
    done: list[str] = []
    with pytest.raises(TickOverrun, match=r"^tick ran past 0\.3 s$"):
        with rig.d._critical():
            with rig.d._critical():
                rig.d.overrun(0.3)
                done.append("inner")
            done.append("between")  # the inner exit does not raise
        done.append("never")
    assert done == ["inner", "between"]


def test_a_pending_overrun_is_dropped_when_the_next_tick_starts(tmp_path: Path) -> None:
    rig = rig_with(tmp_path, [task(tmp_path, "a")])
    with contextlib.suppress(RuntimeError), rig.d._critical():
        rig.d.overrun(60.0)
        raise RuntimeError("the section's own work failed")
    rig.tick()  # starts a job, a critical section that would raise the stale overrun
    assert len(rig.handles) == 1


def test_serve_bounds_a_tick_at_sixty_seconds_by_default() -> None:
    assert dispatcher_module.TICK_BOUND_S == 60.0
    assert inspect.signature(serve).parameters["tick_bound_s"].default == 60.0


def test_a_tick_that_fails_late_leaves_no_timer_to_fire_in_the_nap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = task(tmp_path, "a", ready=Ready(1, ("./ready",)))
    rig = rig_with(tmp_path, [spec], run_in=Failing(0.2))
    drive = Drive(rig, monkeypatch)
    # The tick fails 0.2 s into its 0.3 s bound and the nap lasts 0.4 s: a timer left armed would fire in the nap.
    drive.run(lambda _n: rig.state.request_stop("now"), real_nap_s=0.4)
    assert drive.timers == [NO_TIMER]
    assert drive.overruns() == ["RuntimeError: the check failed"]


def test_the_timer_is_disarmed_before_the_tick_error_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = task(tmp_path, "a", ready=Ready(1, ("./ready",)))
    rig = rig_with(tmp_path, [spec], run_in=Failing(0.0))
    drive = Drive(rig, monkeypatch)

    def on_nap(_n: int) -> None:
        armed, recording = drive.clock_timers[
            -2:
        ]  # the tick's own clock read, then the tick-error's
        assert 0 < armed[0] <= 5.0 and recording == NO_TIMER
        rig.state.request_stop("now")

    drive.run(on_nap, bound_s=5.0)
    assert drive.overruns() == ["RuntimeError: the check failed"]


def test_an_overrun_after_the_run_now_was_read_keeps_the_request_and_the_next_tick_starts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = Blocker(rc=1)
    specs = [
        task(tmp_path, "a", ready=Ready(3600, ("./ready",))),
        task(tmp_path, "b", ready=Ready(1, ("./ready",))),
    ]
    rig = rig_with(tmp_path, specs, run_in=blocker)
    rig.next_codes = [None]
    drive = Drive(rig, monkeypatch)
    key = ("c/a", MODEL)

    def on_nap(n: int) -> None:
        if n == 1:
            assert rig.handles == []  # neither pair is ready
            rig.state.request_run_now(*key)
            blocker.block.add(
                "./ready"
            )  # b's ready check, after the request was read, outlives the bound
        elif n == 2:
            assert drive.overruns() == ["TickOverrun: tick ran past 0.3 s"]
            assert rig.state.run_now_requested() == key and rig.handles == []
            assert rig.events("refused") == []
            blocker.block.clear()
        else:
            assert rig.state.run_now_requested() is None and len(rig.handles) == 1
            assert [e.ref for e in rig.events("run-start")] == ["c/a"]
            rig.state.request_stop("now")

    drive.run(on_nap)


def test_a_newer_run_now_for_another_pair_survives_the_start_of_the_first(tmp_path: Path) -> None:
    def newer_request(
        argv: Sequence[str], cwd: Path, env: Mapping[str, str]
    ) -> CompletedProcess[str]:
        rig.state.request_run_now("c/b", MODEL)  # arrives while the tick is under way
        return CompletedProcess(list(argv), 1, "", "")

    specs = [
        task(tmp_path, "a", ready=Ready(3600, ("./ready",))),
        task(tmp_path, "b", ready=Ready(3600, ("./ready",))),
    ]
    rig = rig_with(tmp_path, specs, run_in=newer_request)
    rig.state.request_run_now("c/a", MODEL)
    rig.tick()
    assert [e.ref for e in rig.events("run-start")] == ["c/a"]
    assert rig.state.run_now_requested() == ("c/b", MODEL)
    rig.tick()
    assert [e.ref for e in reversed(rig.events("run-start"))] == ["c/a", "c/b"]
    assert rig.state.run_now_requested() is None


class CutOnce:
    """Wraps `state.record_event` so the first event of `kind` raises TickOverrun, as an overrun between two writes would."""

    def __init__(self, rig: Rig, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.real = rig.state.record_event
        self.kind = kind
        self.cut = False
        monkeypatch.setattr(rig.state, "record_event", self)

    def __call__(self, now: datetime, kind: str, ref: str | None = None, detail: str = "") -> None:
        if kind == self.kind and not self.cut:
            self.cut = True
            raise TickOverrun("cut")
        self.real(now, kind, ref, detail)


def test_an_interruption_around_the_starved_event_repeats_it_and_never_loses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host(module="580.159.03"))
    rig.tick()
    rig.clock.advance(25 * 3600)
    cut = CutOnce(rig, "starved", monkeypatch)
    with pytest.raises(TickOverrun):
        rig.tick()
    job = rig.state.get_job("c/a", MODEL)
    assert cut.cut and rig.events("starved") == [] and job is not None and job.starved_flagged == 0
    rig.tick()
    rig.tick()
    assert len(rig.events("starved")) == 1


def test_an_interruption_around_a_refused_event_repeats_it_and_never_loses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.specs["a"] = task(
        tmp_path, "a", fingerprint="fp-edited"
    )  # no contract pass: refused each tick
    cut = CutOnce(rig, "refused", monkeypatch)
    with pytest.raises(TickOverrun):
        rig.tick()
    assert cut.cut and rig.events("refused") == []
    rig.tick()
    rig.tick()
    assert len(rig.events("refused")) == 1


def test_a_policy_edit_is_not_ignored_when_reading_it_was_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a"), task(tmp_path, "b")], ["c/a"])
    cache = PolicyCache(rig.policy, lambda _message: None)
    assert [p.task for p in cache.current().jobs] == ["c/a"]
    rig.list_jobs(["c/a", "c/b"])
    real, calls = policy_module.parse_policy, []

    def interrupted(text: str) -> policy_module.Policy:
        calls.append(text)
        if len(calls) == 1:
            raise TickOverrun("cut")
        return real(text)

    monkeypatch.setattr(policy_module, "parse_policy", interrupted)
    with pytest.raises(TickOverrun):
        cache.current()
    assert [p.task for p in cache.current().jobs] == ["c/a", "c/b"]


def test_serve_restores_both_handlers_when_its_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_with(tmp_path, [task(tmp_path, "a")])
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGALRM)

    def failing_recover(_self: Dispatcher, _now: datetime) -> None:
        raise RuntimeError("recover failed")

    monkeypatch.setattr(Dispatcher, "recover", failing_recover)
    with pytest.raises(RuntimeError, match="recover failed"):
        serve(rig.deps, rig.policy, clock=rig.clock, sleep=lambda _s: None, tick_bound_s=BOUND_S)
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGALRM)) == before
