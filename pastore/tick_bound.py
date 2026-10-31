"""The tick bound (D-050 (6)): a one-shot SIGALRM timer around a tick, and the critical sections an overrun waits for.

Depends on: stdlib signal and contextlib only.
"""

import signal
from collections.abc import Iterator
from contextlib import contextmanager

TICK_BOUND_S = 60.0


class TickOverrun(Exception):
    """A tick ran past its bound; the loop logs it and the next tick starts."""


@contextmanager
def bounded(seconds: float) -> Iterator[None]:
    """Run the body under a one-shot SIGALRM timer (0 arms none), disarmed the moment the body returns or raises."""
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class CriticalSections:
    """Mixin: an overrun raises TickOverrun at once, or when the outermost critical section in progress ends."""

    _depth = 0
    _late: float | None = None

    def overrun(self, bound_s: float) -> None:
        """The tick bound's alarm: raise now outside a critical section, else once the outermost one has finished."""
        if self._depth:
            self._late = bound_s
        else:
            raise TickOverrun(f"tick ran past {bound_s:g} s")

    @contextmanager
    def _critical(self) -> Iterator[None]:
        """Work that must finish together (a start, one run's end, a stop): an overrun waits for its end."""
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
        if not self._depth and self._late is not None:
            bound, self._late = self._late, None
            raise TickOverrun(f"tick ran past {bound:g} s")
