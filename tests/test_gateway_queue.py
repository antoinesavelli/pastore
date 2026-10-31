"""Pins the gateway's per-host priority queue: level order, FIFO, hang-up removal, release rules and starvation.

Depends on: pastore.gateway.queue, pastore.spec and pytest only; async scenarios run through asyncio.run.
"""

import asyncio
import functools
import time
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any

import pytest

from pastore.gateway.queue import STARVE_AFTER_S, Grant, HostQueue
from pastore.spec import Level

Starved = list[tuple[str, Level, str, float]]


class Clock:
    """A float clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> Callable[[], None]:
    """Turn an async test function into a plain one that runs it on a fresh event loop."""

    @functools.wraps(scenario)
    def wrapper() -> None:
        asyncio.run(scenario())

    return wrapper


def make_queue(
    slots: int, starve_after_s: float = STARVE_AFTER_S
) -> tuple[HostQueue, Clock, Starved]:
    clock = Clock()
    calls: Starved = []

    def on_starved(host: str, level: Level, label: str, waited_s: float) -> None:
        calls.append((host, level, label, waited_s))

    return (
        HostQueue("workstation", slots, clock, on_starved, starve_after_s=starve_after_s),
        clock,
        calls,
    )


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def enqueue(queue: HostQueue, level: Level, label: str) -> "asyncio.Task[Grant]":
    return asyncio.create_task(queue.acquire(level, label))


async def granted_in_order(
    queue: HostQueue, held: Grant, waiters: list["asyncio.Task[Grant]"]
) -> list[int]:
    """Release `held`, then each next grant in turn; return the indexes of `waiters` in the order they were granted."""
    order: list[int] = []
    current = held
    for _ in waiters:
        queue.release(current)
        await settle()
        newly = [i for i, t in enumerate(waiters) if t.done() and i not in order]
        assert len(newly) == 1
        order.append(newly[0])
        current = waiters[newly[0]].result()
    return order


@run
async def test_a_free_slot_with_nobody_waiting_grants_at_once() -> None:
    queue, _, _ = make_queue(1)
    grant = await asyncio.wait_for(queue.acquire(Level.REINS, "reins"), 1)
    assert (grant.host, grant.level, grant.waited_s) == ("workstation", Level.REINS, 0.0)
    assert queue.snapshot().in_flight == 1


@run
async def test_a_higher_level_is_served_before_a_lower_one() -> None:
    queue, _, _ = make_queue(1)
    held = await queue.acquire(Level.ROUTINE, "running")
    levels = [Level.HEARTH, Level.ROUTINE, Level.REINS, Level.CODING]
    waiters = []
    for level in levels:
        waiters.append(enqueue(queue, level, level.name))
        await settle()
    order = await granted_in_order(queue, held, waiters)
    assert [levels[i] for i in order] == [Level.REINS, Level.CODING, Level.ROUTINE, Level.HEARTH]


@run
async def test_requests_of_one_level_are_served_in_arrival_order() -> None:
    queue, _, _ = make_queue(1)
    held = await queue.acquire(Level.HEARTH, "running")
    waiters = []
    for n in range(5):
        waiters.append(enqueue(queue, Level.CORSU, f"corsu{n}"))
        await settle()
    assert await granted_in_order(queue, held, waiters) == [0, 1, 2, 3, 4]
    seqs = [t.result().seq for t in waiters]
    assert seqs == sorted(seqs)


@run
async def test_several_slots_run_that_many_requests_at_once() -> None:
    queue, _, _ = make_queue(3)
    tasks = [enqueue(queue, Level.ROUTINE, f"r{n}") for n in range(5)]
    await settle()
    assert [t.done() for t in tasks] == [True, True, True, False, False]
    snap = queue.snapshot()
    assert (snap.slots, snap.in_flight, snap.queued["routine"]) == (3, 3, 2)
    queue.release(tasks[0].result())
    await settle()
    snap = queue.snapshot()
    assert (snap.in_flight, snap.queued["routine"]) == (3, 1)
    assert tasks[3].done() and not tasks[4].done()


@run
async def test_check2_coding_gets_the_next_slot_ahead_of_five_waiting_hearth() -> None:
    queue, _, _ = make_queue(1)
    running = await queue.acquire(Level.HEARTH, "hearth:r0")
    hearth = []
    for n in range(5):
        hearth.append(enqueue(queue, Level.HEARTH, f"hearth:r{n + 1}"))
        await settle()
    coding = enqueue(queue, Level.CODING, "claude")
    await settle()
    assert not coding.done()
    assert queue.snapshot().in_flight == 1
    queue.release(running)
    await settle()
    assert coding.done()
    assert not any(t.done() for t in hearth)
    assert queue.snapshot().queued["hearth"] == 5


@run
async def test_nothing_preempts_a_running_request() -> None:
    queue, _, _ = make_queue(1)
    running = await queue.acquire(Level.HEARTH, "hearth:r0")
    reins = enqueue(queue, Level.REINS, "reins")
    await settle()
    snap = queue.snapshot()
    assert (snap.in_flight, snap.queued["reins"]) == (1, 1)
    assert not reins.done()
    queue.release(running)
    await settle()
    assert reins.done()


@run
async def test_check7_a_cancelled_waiter_is_skipped_and_the_count_drops() -> None:
    queue, _, _ = make_queue(1)
    held = await queue.acquire(Level.ROUTINE, "running")
    first = enqueue(queue, Level.CODING, "first")
    middle = enqueue(queue, Level.CODING, "middle")
    last = enqueue(queue, Level.CODING, "last")
    await settle()
    assert queue.snapshot().queued["coding"] == 3
    middle.cancel()
    await settle()
    assert middle.cancelled()
    assert queue.snapshot().queued["coding"] == 2
    assert await granted_in_order(queue, held, [first, last]) == [0, 1]
    assert queue.snapshot().in_flight == 1


@run
async def test_check7_cancel_after_the_slot_was_handed_over_does_not_leak() -> None:
    queue, _, _ = make_queue(1)
    held = await queue.acquire(Level.ROUTINE, "running")
    waiter = enqueue(queue, Level.REINS, "waiter")
    await settle()
    queue.release(held)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    snap = queue.snapshot()
    assert (snap.in_flight, sum(snap.queued.values())) == (0, 0)
    again = await asyncio.wait_for(queue.acquire(Level.ROUTINE, "later"), 1)
    queue.release(again)
    assert queue.snapshot().in_flight == 0


@run
async def test_check7_a_slot_handed_to_a_cancelled_waiter_goes_to_the_next_one() -> None:
    queue, _, _ = make_queue(1)
    held = await queue.acquire(Level.ROUTINE, "running")
    doomed = enqueue(queue, Level.REINS, "doomed")
    next_up = enqueue(queue, Level.ROUTINE, "next")
    await settle()
    queue.release(held)
    doomed.cancel()
    await settle()
    assert doomed.cancelled()
    assert next_up.done()
    assert queue.snapshot().in_flight == 1
    queue.release(next_up.result())
    assert queue.snapshot().in_flight == 0


@run
async def test_releasing_a_grant_twice_raises_and_keeps_the_count() -> None:
    queue, _, _ = make_queue(2)
    first = await queue.acquire(Level.REINS, "a")
    await queue.acquire(Level.REINS, "b")
    queue.release(first)
    with pytest.raises(ValueError, match="already released"):
        queue.release(first)
    assert queue.snapshot().in_flight == 1


@run
async def test_releasing_a_grant_the_queue_never_gave_raises() -> None:
    queue, _, _ = make_queue(1)
    await queue.acquire(Level.REINS, "a")
    with pytest.raises(ValueError, match="unknown"):
        queue.release(Grant("workstation", Level.REINS, 99, 0.0))
    with pytest.raises(ValueError, match="unknown"):
        queue.release(Grant("executor", Level.REINS, 1, 0.0))
    assert queue.snapshot().in_flight == 1


@pytest.mark.parametrize(
    "forged", [{"level": Level.HEARTH}, {"waited_s": 123.0}, {"host": "executor"}]
)
def test_releasing_a_forged_copy_of_a_held_grant_raises(forged: dict[str, Any]) -> None:
    async def scenario() -> None:
        queue, _, _ = make_queue(1)
        held = await queue.acquire(Level.REINS, "a")
        with pytest.raises(ValueError, match="unknown"):
            queue.release(replace(held, **forged))
        assert queue.snapshot().in_flight == 1
        queue.release(held)
        assert queue.snapshot().in_flight == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("slots", [0, -1])
def test_a_queue_needs_at_least_one_slot(slots: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_queue(slots)


@run
async def test_waited_s_is_the_injected_clock_from_entry_to_grant() -> None:
    queue, clock, _ = make_queue(1)
    clock.now = 2.0
    held = await queue.acquire(Level.ROUTINE, "running")
    assert held.waited_s == 0.0
    waiter = enqueue(queue, Level.CODING, "claude")
    await settle()
    clock.now = 5.5
    queue.release(held)
    await settle()
    assert waiter.result().waited_s == 3.5


@run
async def test_check4_only_the_starved_waiter_is_reported_once_and_keeps_waiting() -> None:
    queue, clock, calls = make_queue(1, starve_after_s=0.05)
    held = await queue.acquire(Level.ROUTINE, "running")
    old = enqueue(queue, Level.CODING, "claude")
    await settle()
    clock.now = 1.0
    young = enqueue(queue, Level.ROUTINE, "cron")
    await settle()
    await asyncio.sleep(0.2)
    assert calls == [("workstation", Level.CODING, "claude", 1.0)]
    assert not old.done()
    await asyncio.sleep(0.2)
    assert len(calls) == 1
    queue.release(held)
    await settle()
    assert old.done()
    assert len(calls) == 1
    clock.now = 2.0
    await asyncio.sleep(0.2)
    assert calls[1:] == [("workstation", Level.ROUTINE, "cron", 1.0)]
    assert not young.done()


@run
async def test_check4_a_request_granted_before_the_limit_never_reports() -> None:
    queue, clock, calls = make_queue(1, starve_after_s=0.05)
    held = await queue.acquire(Level.ROUTINE, "running")
    waiter = enqueue(queue, Level.CODING, "claude")
    await settle()
    queue.release(held)
    await settle()
    assert waiter.done()
    clock.now = 10.0
    await asyncio.sleep(0.2)
    assert calls == []


@run
async def test_check4_a_request_granted_just_under_the_limit_never_reports() -> None:
    queue, clock, calls = make_queue(1, starve_after_s=0.05)
    held = await queue.acquire(Level.ROUTINE, "running")
    waiter = enqueue(queue, Level.CODING, "claude")
    await settle()
    clock.now = 0.04
    await asyncio.sleep(0.2)
    assert calls == []
    queue.release(held)
    await settle()
    assert waiter.result().waited_s == 0.04
    await asyncio.sleep(0.2)
    assert calls == []


@run
async def test_check4_a_grant_after_the_limit_before_the_timer_ran_still_reports_once() -> None:
    queue, clock, calls = make_queue(1, starve_after_s=1.0)
    held = await queue.acquire(Level.ROUTINE, "running")
    waiter = enqueue(queue, Level.CODING, "claude")
    await settle()
    clock.now = 5.0
    queue.release(held)
    grant = await asyncio.wait_for(waiter, 1)
    assert grant.waited_s == 5.0
    assert calls == [("workstation", Level.CODING, "claude", 5.0)]
    await settle()
    assert len(calls) == 1


@run
async def test_check4_a_blocked_real_loop_past_the_limit_still_reports_once() -> None:
    calls: Starved = []

    def on_starved(host: str, level: Level, label: str, waited_s: float) -> None:
        calls.append((host, level, label, waited_s))

    queue = HostQueue("workstation", 1, time.monotonic, on_starved, starve_after_s=0.05)
    held = await queue.acquire(Level.ROUTINE, "running")
    waiter = enqueue(queue, Level.CODING, "claude")
    await settle()
    time.sleep(0.12)
    queue.release(held)
    grant = await asyncio.wait_for(waiter, 1)
    assert grant.waited_s >= 0.12
    assert [c[:3] for c in calls] == [("workstation", Level.CODING, "claude")]
    assert calls[0][3] == grant.waited_s
    await settle()
    assert len(calls) == 1


@run
async def test_check4_the_real_clock_wakes_a_starved_waiter_without_help() -> None:
    calls: Starved = []

    def on_starved(host: str, level: Level, label: str, waited_s: float) -> None:
        calls.append((host, level, label, waited_s))

    queue = HostQueue("executor", 1, time.monotonic, on_starved, starve_after_s=0.05)
    held = await queue.acquire(Level.ROUTINE, "running")
    waiter = enqueue(queue, Level.HEARTH, "hearth:r7")
    await asyncio.sleep(0.25)
    assert [c[:3] for c in calls] == [("executor", Level.HEARTH, "hearth:r7")]
    assert calls[0][3] >= 0.05
    queue.release(held)
    await settle()
    assert waiter.done()


@run
async def test_a_failing_starved_callback_after_the_grant_gives_the_slot_back() -> None:
    def on_starved(host: str, level: Level, label: str, waited_s: float) -> None:
        raise RuntimeError("journal write failed")

    clock = Clock()
    queue = HostQueue("workstation", 1, clock, on_starved, starve_after_s=1.0)
    held = await queue.acquire(Level.ROUTINE, "running")
    waiter = enqueue(queue, Level.CODING, "claude")
    await settle()
    clock.now = 5.0
    queue.release(held)
    with pytest.raises(RuntimeError, match="journal write failed"):
        await waiter
    assert queue.snapshot().in_flight == 0


@run
async def test_a_failing_starved_callback_removes_the_waiter_and_propagates() -> None:
    def on_starved(host: str, level: Level, label: str, waited_s: float) -> None:
        raise RuntimeError("journal write failed")

    queue = HostQueue("workstation", 1, Clock(), on_starved, starve_after_s=0.0)
    await queue.acquire(Level.ROUTINE, "running")
    with pytest.raises(RuntimeError, match="journal write failed"):
        await queue.acquire(Level.CODING, "claude")
    snap = queue.snapshot()
    assert sum(snap.queued.values()) == 0
    assert snap.in_flight == 1


@run
async def test_snapshot_lists_every_level_and_the_longest_wait() -> None:
    queue, clock, _ = make_queue(2)
    empty = queue.snapshot()
    assert empty.oldest_wait_s is None
    assert empty.queued == {
        "reins": 0,
        "webui": 0,
        "coding": 0,
        "corsu": 0,
        "routine": 0,
        "hearth": 0,
    }
    assert (empty.slots, empty.in_flight) == (2, 0)
    for n in range(2):
        await queue.acquire(Level.ROUTINE, f"running{n}")
    enqueue(queue, Level.HEARTH, "hearth:r1")
    await settle()
    clock.now = 4.0
    enqueue(queue, Level.WEBUI, "webui")
    enqueue(queue, Level.WEBUI, "webui2")
    await settle()
    clock.now = 6.0
    snap = queue.snapshot()
    assert snap.queued == {
        "reins": 0,
        "webui": 2,
        "coding": 0,
        "corsu": 0,
        "routine": 0,
        "hearth": 1,
    }
    assert (snap.slots, snap.in_flight, snap.oldest_wait_s) == (2, 2, 6.0)
