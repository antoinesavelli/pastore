"""D-050 (4), D-052 (1): a run whose non-ok end an outage (gateway or executor) caused is stopped and free the first time, a second consecutive outage end of the pair charges an attempt as a second crash does; every other non-ok end is charged as before.

Depends on: pastore.run_end (MAX_ATTEMPTS, gateway_outage), tests/test_dispatcher.py (Rig, FakeNet, fleet_rig, net_of, task), a scripted /pastore/status answer over the Rig's fake http_json, FakeClock.
"""

import urllib.error
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from email.message import Message
from pathlib import Path

import pytest

from pastore.hosts import PROBE_TIMEOUT_S
from pastore.run_end import MAX_ATTEMPTS, gateway_outage
from pastore.spec import Ready
from pastore.state import RunRow
from tests.test_dispatcher import EX_URL, GIB, MODEL, T0, FakeNet, Rig, fleet_rig, net_of, task

STATUS_URL = "http://127.0.0.1:11434/pastore/status"
KILLED_BY_STOP = -15


def status(started_at: object) -> dict[str, object]:
    """The gateway's /pastore/status answer with this `started_at`."""
    return {"hosts": {}, "started_at": started_at}


def started(offset_s: float) -> dict[str, object]:
    """A status answer for a gateway that started `offset_s` seconds after the run did (T0)."""
    return status((T0 + timedelta(seconds=offset_s)).isoformat())


class Gateway:
    """The injected http_json: the status page answers as scripted, every other URL goes to the Rig's FakeNet."""

    def __init__(self, net: FakeNet) -> None:
        self.net = net
        self.answer: object = started(-3600)
        self.reads = 0

    def __call__(self, url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
        if url != STATUS_URL:
            return self.net(url, body, timeout_s)
        self.reads += 1
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


@dataclass
class Scene:
    """A rig after the tick that ended its one run, and the run as it was while running."""

    rig: Rig
    gateway: Gateway
    run: RunRow

    def ended(self) -> RunRow:
        """The run's row after the ending tick."""
        row = self.rig.state.get_run(self.run.run_id)
        assert row is not None
        return row

    def attempts_and_streak(self) -> tuple[int, int]:
        """The job's attempts and crash_streak after the ending tick."""
        job = self.rig.state.get_job("c/a", MODEL)
        assert job is not None
        return job.attempts, job.crash_streak


def end_a_run(
    tmp_path: Path,
    answer: object,
    *,
    code: int | None = 1,
    timed_out: bool = False,
    streak: int = 1,
    attempts: int = 0,
) -> Scene:
    """Start one run at T0, put the job's crash_streak and attempts as given, set the status answer, and tick when the run has ended."""
    net = FakeNet()
    gateway = Gateway(net)
    rig = Rig(tmp_path, [task(tmp_path, "a", timeout_s=60)], ["c/a"], net=net, http_json=gateway)
    rig.next_codes = [None] if timed_out else [code]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None
    rig.state.upsert_job(replace(job, crash_streak=streak, attempts=attempts))
    rig.paths.stop_file.write_text(
        ""
    )  # nothing is placed again, so the job's row is as the end left it
    gateway.answer = answer
    rig.clock.advance(61 if timed_out else 30)
    rig.tick()
    return Scene(rig, gateway, run)


def not_a_gateway() -> urllib.error.HTTPError:
    """What the real http_json raises when Ollama itself answers 404 to the status URL."""
    return urllib.error.HTTPError(STATUS_URL, 404, "Not Found", Message(), None)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(OSError("connection refused"), id="unreachable"),
        pytest.param(TimeoutError("timed out"), id="timeout"),
        pytest.param(started(10), id="restarted-after-the-run-started"),
    ],
)
def test_invariant_a_run_cut_by_a_gateway_outage_is_not_charged(
    tmp_path: Path, answer: object
) -> None:
    scene = end_a_run(tmp_path, answer, streak=0)
    rig = scene.rig
    ended = scene.ended()
    assert (ended.status, ended.detail, ended.exit_code) == ("stopped", "ollama-down", 1)
    assert rig.job_status("a") == "pending"
    assert scene.attempts_and_streak() == (0, 1)
    [event] = rig.events("run-end")
    assert event.detail == f"{scene.run.run_id} stopped 1 ollama-down"
    assert rig.events("failed") == [] and not rig.handles[0].stopped
    assert scene.gateway.reads == 1


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(started(-3600), id="up-before-the-run"),
        pytest.param(started(0), id="started-at-the-run-start"),
        pytest.param(not_a_gateway(), id="404-not-a-gateway"),
        pytest.param(ValueError("not JSON"), id="bad-json"),
        pytest.param([started(10)], id="not-an-object"),
        pytest.param({"hosts": {}}, id="no-started-at"),
        pytest.param(status(1761271200), id="started-at-not-a-string"),
        pytest.param(status("2026-10-24T02:00:10"), id="naive-started-at"),
        pytest.param(status("yesterday"), id="garbage-started-at"),
    ],
)
def test_any_other_non_ok_end_is_charged_as_before(tmp_path: Path, answer: object) -> None:
    scene = end_a_run(tmp_path, answer)
    ended = scene.ended()
    assert (ended.status, ended.detail, ended.exit_code) == ("failed", "", 1)
    assert scene.rig.job_status("a") == "pending"
    assert scene.attempts_and_streak() == (1, 0)
    assert scene.gateway.reads == 1


def test_a_clean_end_never_reads_the_status_page(tmp_path: Path) -> None:
    scene = end_a_run(tmp_path, OSError("down"), code=0)
    assert scene.ended().status == "done"
    assert scene.gateway.reads == 0
    assert scene.rig.job_status("a") == "done"


def test_a_timed_out_run_with_the_gateway_down_is_stopped_uncharged_after_the_scope_stop(
    tmp_path: Path,
) -> None:
    scene = end_a_run(tmp_path, OSError("down"), timed_out=True, streak=0)
    ended = scene.ended()
    assert (ended.status, ended.detail, ended.exit_code) == (
        "stopped",
        "ollama-down",
        KILLED_BY_STOP,
    )
    assert scene.rig.handles[0].stopped
    assert scene.rig.job_status("a") == "pending"
    assert scene.attempts_and_streak() == (0, 1)


def test_a_timed_out_run_with_the_gateway_up_is_charged_as_a_timeout(tmp_path: Path) -> None:
    scene = end_a_run(tmp_path, started(-3600), timed_out=True)
    ended = scene.ended()
    assert (ended.status, ended.detail) == ("timeout", "")
    assert scene.rig.handles[0].stopped and scene.attempts_and_streak() == (1, 0)


@pytest.mark.parametrize(
    ("executor_down", "reason"), [(True, "executor-down"), (False, "ollama-down")]
)
def test_an_executor_run_is_executor_down_first_and_ollama_down_when_only_the_gateway_is_(
    tmp_path: Path, executor_down: bool, reason: str
) -> None:
    net = net_of({}, {"only": GIB})
    gateway = Gateway(net)
    rig = fleet_rig(tmp_path, net, [("c/a", "only")], models=("only",), http_json=gateway)
    rig.next_codes = [1]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == "executor"
    rig.paths.stop_file.write_text("")
    gateway.answer = OSError("down")
    if executor_down:
        net.version_down.add(EX_URL)
    rig.tick()
    ended = rig.state.get_run(run.run_id)
    job = rig.state.get_job("c/a", "only")
    assert ended is not None and (ended.status, ended.detail) == ("stopped", reason)
    assert job is not None and (job.attempts, job.status) == (0, "pending")
    assert gateway.reads == (0 if executor_down else 1)


REASON = {"gateway": "ollama-down", "executor": "executor-down"}
Switch = Callable[[bool], None]


def outage_rig(tmp_path: Path, kind: str, **task_kw: object) -> tuple[Rig, Switch]:
    """A rig whose one pair runs on the executor, and a switch that puts the `kind` of outage (gateway or executor) down (True) or up."""
    net = net_of({}, {"only": GIB})
    gateway = Gateway(net)
    rig = fleet_rig(
        tmp_path, net, [("c/a", "only")], models=("only",), task_kw=task_kw, http_json=gateway
    )

    def switch(down: bool) -> None:
        if kind == "gateway":
            gateway.answer = OSError("down") if down else started(-3600)
        elif down:
            net.version_down.add(EX_URL)
        else:
            net.version_down.discard(EX_URL)

    return rig, switch


def place_and_end(rig: Rig, switch: Switch, *, outage: bool) -> RunRow:
    """Place the pair's next run with every outage up, then end it with the outage down or up; placement is held for the end, so nothing else starts."""
    stop_file = rig.paths.stop_file
    rig.clock.advance(30)
    stop_file.unlink(missing_ok=True)
    switch(False)
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    stop_file.write_text("")
    switch(outage)
    rig.tick()
    return run


def counts(rig: Rig) -> tuple[str, int, int]:
    """The pair's job row as (status, attempts, crash_streak)."""
    job = rig.state.get_job("c/a", "only")
    assert job is not None
    return job.status, job.attempts, job.crash_streak


def assert_stopped_by(rig: Rig, run: RunRow, kind: str) -> None:
    ended = rig.state.get_run(run.run_id)
    assert ended is not None and (ended.status, ended.detail) == ("stopped", REASON[kind])


@pytest.mark.parametrize("kind", ["gateway", "executor"])
def test_a_second_consecutive_outage_end_charges_an_attempt(tmp_path: Path, kind: str) -> None:
    rig, switch = outage_rig(tmp_path, kind)
    rig.next_codes = [1]
    first = place_and_end(rig, switch, outage=True)
    assert counts(rig) == ("pending", 0, 1)
    second = place_and_end(rig, switch, outage=True)
    assert counts(rig) == ("pending", 1, 2)
    third = place_and_end(rig, switch, outage=True)
    assert counts(rig) == ("pending", 2, 3)
    for run in (first, second, third):
        assert_stopped_by(rig, run, kind)
    assert rig.events("failed") == [] and rig.state.runs_with_status("failed") == []


@pytest.mark.parametrize(
    ("middle", "after_middle"), [(1, ("pending", 1, 0)), (0, ("pending", 0, 0))]
)
def test_a_normal_end_between_two_outage_ends_makes_the_second_free_again(
    tmp_path: Path, middle: int, after_middle: tuple[str, int, int]
) -> None:
    rig, switch = outage_rig(tmp_path, "gateway", ready=Ready(1, ("./ready",)))
    rig.next_codes = [1]
    first = place_and_end(rig, switch, outage=True)
    assert counts(rig) == ("pending", 0, 1)
    rig.next_codes = [middle]
    normal = place_and_end(rig, switch, outage=False)
    ended = rig.state.get_run(normal.run_id)
    assert ended is not None and ended.status == ("done" if middle == 0 else "failed")
    assert counts(rig) == after_middle
    rig.next_codes = [1]
    last = place_and_end(rig, switch, outage=True)
    assert counts(rig) == ("pending", after_middle[1], 1)
    assert_stopped_by(rig, first, "gateway")
    assert_stopped_by(rig, last, "gateway")
    assert rig.events("failed") == []


@pytest.mark.parametrize(
    ("streak", "status", "expected", "failed_events"),
    [(0, "pending", (2, 1), 0), (1, "failed", (3, 2), 1)],
)
def test_a_second_outage_end_at_the_last_attempt_fails_the_pair(
    tmp_path: Path, streak: int, status: str, expected: tuple[int, int], failed_events: int
) -> None:
    scene = end_a_run(tmp_path, OSError("down"), streak=streak, attempts=MAX_ATTEMPTS - 1)
    rig = scene.rig
    ended = scene.ended()
    assert (ended.status, ended.detail) == ("stopped", "ollama-down")
    assert rig.job_status("a") == status and scene.attempts_and_streak() == expected
    failed = [(e.ref, e.detail) for e in rig.events("failed")]
    assert failed == [("c/a", f"{MODEL}: 3 attempts")] * failed_events


def test_the_status_page_is_read_by_one_get_with_the_probe_timeout() -> None:
    calls: list[tuple[str, object, float]] = []

    def get(url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
        calls.append((url, body, timeout_s))
        return started(10)

    assert gateway_outage(get, STATUS_URL, T0) is True
    assert calls == [(STATUS_URL, None, PROBE_TIMEOUT_S)] and PROBE_TIMEOUT_S == 2.0
