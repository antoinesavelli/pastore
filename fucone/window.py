"""Quiet-hours arithmetic: is a local wall-clock time inside the window, and when does it next open.

Depends on: stdlib datetime only; the caller passes a timezone-aware local `now`.
"""

from datetime import datetime, time, timedelta


def _validate(now: datetime, start: time, end: time) -> None:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if start == end:
        raise ValueError("window start and end must differ")


def in_window(now: datetime, start: time, end: time) -> bool:
    """True when now's wall-clock time is in [start, end), crossing midnight when start > end."""
    _validate(now, start, end)
    t = now.time()
    if start < end:
        return start <= t < end
    return t >= start or t < end


def next_window_start(now: datetime, start: time, end: time) -> datetime:
    """The current window's opening if inside it, else the next opening strictly after now."""
    _validate(now, start, end)
    today = now.date()
    opening = datetime.combine(today, start, tzinfo=now.tzinfo)
    if in_window(now, start, end):
        return opening if now.time() >= start else opening - timedelta(days=1)
    return opening if opening > now else opening + timedelta(days=1)
