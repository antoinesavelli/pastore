"""Per-host request slots and priority queue (R23, R32): level first, FIFO within a level, hang-up removal, starvation.

Depends on: pastore.spec (Level) and stdlib asyncio only.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pastore.spec import Level

STARVE_AFTER_S = 600.0
OnStarved = Callable[[str, Level, str, float], None]


@dataclass(frozen=True)
class Grant:
    """One held slot on one host; give it back with `HostQueue.release` exactly once."""

    host: str
    level: Level
    seq: int
    waited_s: float


@dataclass(frozen=True)
class Snapshot:
    """A host's slots, slots in use, waiting requests per level name (every level present), and the oldest wait."""

    slots: int
    in_flight: int
    queued: Mapping[str, int]
    oldest_wait_s: float | None


@dataclass(eq=False)
class _Waiter:
    level: Level
    label: str
    seq: int
    start: float
    future: "asyncio.Future[Grant]"
    starved: bool = False


class HostQueue:
    """The slots of one host; a request waits here until a slot frees and no better-placed request is waiting."""

    def __init__(
        self,
        host: str,
        slots: int,
        clock: Callable[[], float],
        on_starved: OnStarved,
        *,
        starve_after_s: float = STARVE_AFTER_S,
    ) -> None:
        if slots < 1:
            raise ValueError(f"host {host}: slots must be at least 1, got {slots}")
        self._host = host
        self._slots = slots
        self._clock = clock
        self._on_starved = on_starved
        self._starve_after_s = starve_after_s
        self._seq = 0
        self._held: dict[int, Grant] = {}
        self._waiting: list[_Waiter] = []

    async def acquire(self, level: Level, label: str) -> Grant:
        """Wait for a slot; cancelling the waiting task removes its entry and never leaks a slot."""
        start = self._clock()
        self._seq += 1
        if len(self._held) < self._slots:
            grant = Grant(self._host, level, self._seq, 0.0)
            self._held[self._seq] = grant
            return grant
        waiter = _Waiter(level, label, self._seq, start, asyncio.get_running_loop().create_future())
        self._waiting.append(waiter)
        try:
            return await self._wait(waiter)
        except BaseException:
            self._abandon(waiter)
            raise

    def release(self, grant: Grant) -> None:
        """Free the slot and hand it to the best waiting request."""
        if self._held.get(grant.seq) != grant:
            raise ValueError(f"grant {grant.seq} on {grant.host} is unknown or already released")
        del self._held[grant.seq]
        self._grant_next()

    def snapshot(self) -> Snapshot:
        """The queue's current state for `/pastore/status`."""
        now = self._clock()
        queued = {level.name.lower(): 0 for level in Level}
        for waiter in self._waiting:
            queued[waiter.level.name.lower()] += 1
        oldest = max((now - w.start for w in self._waiting), default=None)
        return Snapshot(self._slots, len(self._held), queued, oldest)

    async def _wait(self, waiter: _Waiter) -> Grant:
        while not waiter.future.done():
            timeout = None
            if not waiter.starved:
                timeout = self._starve_after_s - (self._clock() - waiter.start)
                if timeout <= 0:
                    self._starve(waiter, self._clock() - waiter.start)
                    continue
            await asyncio.wait({waiter.future}, timeout=timeout)
        grant = waiter.future.result()
        if not waiter.starved and grant.waited_s >= self._starve_after_s:
            self._starve(waiter, grant.waited_s)
        return grant

    def _starve(self, waiter: _Waiter, waited_s: float) -> None:
        waiter.starved = True
        self._on_starved(self._host, waiter.level, waiter.label, waited_s)

    def _abandon(self, waiter: _Waiter) -> None:
        if waiter.future.done():
            del self._held[waiter.future.result().seq]
            self._grant_next()
        else:
            self._waiting.remove(waiter)

    def _grant_next(self) -> None:
        while self._waiting and len(self._held) < self._slots:
            best = min(self._waiting, key=lambda w: (w.level, w.seq))
            self._waiting.remove(best)
            grant = Grant(self._host, best.level, best.seq, self._clock() - best.start)
            self._held[best.seq] = grant
            best.future.set_result(grant)
