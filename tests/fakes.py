"""Shared test doubles: an injectable clock and a recording command runner; tests import these, never edit them per test.

Depends on: stdlib datetime and subprocess only.
"""

import subprocess
from collections.abc import Sequence
from datetime import datetime, timedelta


class FakeClock:
    """A clock that only moves when a test advances it."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs a timezone-aware datetime")
        self.now_value = start

    def __call__(self) -> datetime:
        return self.now_value

    def advance(self, seconds: float) -> None:
        self.now_value = self.now_value + timedelta(seconds=seconds)


class RecordingRunner:
    """Records every argv it is called with and returns scripted results keyed by argv[0]."""

    def __init__(
        self,
        results: dict[str, subprocess.CompletedProcess[str]] | None = None,
        default_returncode: int = 0,
    ) -> None:
        self.results = dict(results or {})
        self.default_returncode = default_returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if isinstance(argv, str):
            raise TypeError("argv must be a list, not a shell string")
        self.calls.append(list(argv))
        if argv[0] in self.results:
            return self.results[argv[0]]
        return subprocess.CompletedProcess(
            args=list(argv), returncode=self.default_returncode, stdout="", stderr=""
        )

    @staticmethod
    def ok(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
        """A scripted result to put in `results`."""
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")
