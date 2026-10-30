"""Source level of a request (R21, R22): the peer address, or for a local peer the client process found through /proc.

Depends on: pastore.spec (Level, SourceRule, UNKNOWN_LEVEL) and stdlib only; reads /proc under an injected root.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pastore.spec import Level, SourceRule

BUDGET_S = 0.020
Address = tuple[str, int]


@dataclass(frozen=True)
class Attribution:
    """Who sent a request: its level, a short space-free label for the journal, and the hearth run id, if any."""

    level: Level
    source: str
    run_id: str | None = None


class Attributor:
    """Maps a connection to its source level by the policy's rules, first match wins, within a time budget."""

    def __init__(
        self,
        rules: Sequence[SourceRule],
        clock: Callable[[], float],
        *,
        proc: Path = Path("/proc"),
        uid: int | None = None,
        budget_s: float = BUDGET_S,
    ) -> None:
        raise NotImplementedError

    def __call__(self, peer: Address, local: Address) -> Attribution:
        """Attribute the connection whose client end is `peer` and whose server end is `local`."""
        raise NotImplementedError
