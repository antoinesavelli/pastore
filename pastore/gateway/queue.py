"""Per-host request slots and priority queue (R23, R32): level first, FIFO within a level, hang-up removal, starvation.

Depends on: pastore.spec (Level) and stdlib asyncio only.
"""

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
        raise NotImplementedError

    async def acquire(self, level: Level, label: str) -> Grant:
        """Wait for a slot; cancelling the waiting task removes its entry and never leaks a slot."""
        raise NotImplementedError

    def release(self, grant: Grant) -> None:
        """Free the slot and hand it to the best waiting request."""
        raise NotImplementedError

    def snapshot(self) -> Snapshot:
        """The queue's current state for `/pastore/status`."""
        raise NotImplementedError
