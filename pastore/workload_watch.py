"""R35 foreign-workload watch: log lines of watched units in this machine's journal against executor run spans; records and alerts only.

Depends on: pastore.runner (Runner), pastore.spec (WatchConfig, Signal) and stdlib; the journal is read on the workstation through an injected runner.
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

from pastore.runner import Runner
from pastore.spec import Signal, ValueSignal, WatchConfig

AFTER_RUN = timedelta(minutes=10)
KEEP_AFTER_END = timedelta(minutes=30)
LOOKBACK = timedelta(minutes=15)
STATUS_WINDOW = timedelta(minutes=30)
MAX_DIGITS = 9

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class JournalError(Exception):
    """The journal could not be read or did not have the shape journalctl promises."""


@dataclass(frozen=True)
class Line:
    """One journal line of a watched unit."""

    at: datetime
    unit: str
    message: str


@dataclass(frozen=True)
class Span:
    """A pastore run on the executor; `end` is None while it runs."""

    run_id: str
    ref: str
    start: datetime
    end: datetime | None


@dataclass(frozen=True)
class Disruption:
    """A disruption of the watched workload; `end` is None while it lasts."""

    signal: str
    start: datetime
    end: datetime | None


def _iso(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Overlap:
    """A disruption that started while a run was on the executor, or within AFTER_RUN after it."""

    span: Span
    disruption: Disruption

    def key(self) -> str:
        """The dedupe key: the same overlap re-read on a later tick has the same key."""
        return f"{self.span.run_id} {self.disruption.signal} {_iso(self.disruption.start)}"

    def detail(self) -> str:
        """One line naming both time spans; only our own fields, never journal text."""
        s, d = self.span, self.disruption
        run_end = "running" if s.end is None else _iso(s.end)
        gone = "ongoing" if d.end is None else _iso(d.end)
        return (
            f"run={s.run_id} {s.ref} run={_iso(s.start)}..{run_end} "
            f"signal={d.signal} disruption={_iso(d.start)}..{gone}"
        )


@dataclass(frozen=True)
class Watch:
    """The watch result: "armed", "down", "no-signal" or "unreadable: <reason>", plus the overlaps found."""

    status: str
    overlaps: tuple[Overlap, ...]


JournalReader = Callable[[str, datetime], list[Line]]


def _parse_line(unit: str, raw: str) -> Line | None:
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError):
        raise JournalError("journalctl output is not JSON") from None
    if not isinstance(obj, dict):
        raise JournalError("journalctl line is not a JSON object")
    stamp = obj.get("__REALTIME_TIMESTAMP")
    if not isinstance(stamp, str) or not (stamp.isascii() and stamp.isdigit()):
        raise JournalError("journalctl line has no usable timestamp")
    message = obj.get("MESSAGE")
    if not isinstance(message, str):
        return None
    try:
        at = _EPOCH + timedelta(microseconds=int(stamp))
    except (OverflowError, ValueError):
        raise JournalError("journalctl timestamp out of range") from None
    return Line(at, unit, message)


def journal_reader(run: Runner) -> JournalReader:
    """A reader of `journalctl -u <unit>` (system units, read only) through `run`; it raises JournalError."""

    def read(unit: str, since: datetime) -> list[Line]:
        argv = [
            "journalctl",
            "-u",
            unit,
            f"--since=@{int(since.timestamp())}",
            "-o",
            "json",
            "--output-fields=MESSAGE",
            "--no-pager",
        ]
        proc = run(argv)
        if proc.returncode != 0:
            raise JournalError(f"journalctl exit {proc.returncode}")
        parsed = (_parse_line(unit, raw) for raw in proc.stdout.split("\n") if raw.strip())
        return sorted((line for line in parsed if line is not None), key=lambda line: line.at)

    return read


class _Sample(NamedTuple):
    at: datetime
    signal: str
    good: bool


def _judge(signal: Signal, message: str) -> bool | None:
    """Whether the message is a good sample of the signal, a bad one, or none; a number that is not digits is none."""
    if isinstance(signal, ValueSignal):
        hit = re.search(signal.value, message)
        digits = hit.group(1) if hit else None
        if not digits or not digits.isdecimal():
            return None
        digits = digits.lstrip("0") or "0"
        return len(digits) <= MAX_DIGITS and int(digits) <= signal.limit
    if re.search(signal.bad, message):
        return False
    return True if re.search(signal.good, message) else None


def _samples(config: WatchConfig, lines: Sequence[Line]) -> list[_Sample]:
    found: list[_Sample] = []
    for line in lines:
        for signal in config.signals:
            good = _judge(signal, line.message) if signal.unit == line.unit else None
            if good is not None:
                found.append(_Sample(line.at, signal.name, good))
    return sorted(found, key=lambda s: s.at)


def disruptions(config: WatchConfig, lines: Sequence[Line]) -> list[Disruption]:
    """Good-to-bad transitions per signal; a first bad sample counts only for a signal marked `first_bad`."""
    first_bad = {signal.name for signal in config.signals if signal.first_bad}
    previous: dict[str, bool] = {}
    open_at: dict[str, datetime] = {}
    found: list[Disruption] = []
    for at, signal, good in _samples(config, lines):
        before = previous.get(signal)
        if good and signal in open_at:
            found.append(Disruption(signal, open_at.pop(signal), at))
        elif not good and (before is True or (before is None and signal in first_bad)):
            open_at[signal] = at
        previous[signal] = good
    found += [Disruption(signal, at, None) for signal, at in open_at.items()]
    return sorted(found, key=lambda d: (d.start, d.signal))


def overlaps(spans: Sequence[Span], found: Sequence[Disruption]) -> list[Overlap]:
    """Each disruption that started during a span or within AFTER_RUN after it, per span."""
    hits = [
        Overlap(span, d)
        for span in spans
        for d in found
        if span.start <= d.start and (span.end is None or d.start <= span.end + AFTER_RUN)
    ]
    return sorted(hits, key=lambda o: (o.disruption.start, o.span.run_id, o.disruption.signal))


def read_since(now: datetime, spans: Sequence[Span]) -> datetime:
    """How far back to read: the status window, or LOOKBACK before the earliest span, whichever is earlier."""
    return min([now - STATUS_WINDOW, *(s.start - LOOKBACK for s in spans)])


def watch(config: WatchConfig, now: datetime, spans: Sequence[Span], read: JournalReader) -> Watch:
    """Read the watched units and report the status and any overlap with the spans; it never acts."""
    since = read_since(now, spans)
    try:
        lines = [line for unit in config.units for line in read(unit, since)]
    except JournalError as exc:
        return Watch(f"unreadable: {exc}", ())
    recent = [
        s
        for s in _samples(config, lines)
        if s.signal == config.status and s.at >= now - STATUS_WINDOW
    ]
    status = "no-signal" if not recent else "armed" if recent[-1].good else "down"
    return Watch(status, tuple(overlaps(spans, disruptions(config, lines))))
