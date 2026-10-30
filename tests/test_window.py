"""Quiet-hours arithmetic with fixed tz-aware datetimes.

Depends on: pastore.window only.
"""

from datetime import UTC, datetime, time, timedelta, timezone

import pytest

from pastore.window import in_window, next_window_start

TZ = timezone(timedelta(hours=-4))
NIGHT = (time(22, 0), time(6, 0))
DEFAULT = (time(1, 0), time(7, 0))


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def test_window_crosses_midnight() -> None:
    assert in_window(at(24, 23), *NIGHT)
    assert in_window(at(25, 2), *NIGHT)
    assert not in_window(at(24, 12), *NIGHT)
    assert not in_window(at(25, 7), *NIGHT)


def test_window_boundaries_start_inclusive_end_exclusive() -> None:
    assert in_window(at(24, 1, 0), *DEFAULT)
    assert not in_window(at(24, 0, 59), *DEFAULT)
    assert in_window(at(24, 6, 59), *DEFAULT)
    assert not in_window(at(24, 7, 0), *DEFAULT)
    assert in_window(at(24, 22, 0), *NIGHT)
    assert not in_window(at(24, 6, 0), *NIGHT)


def test_window_with_equal_start_and_end_is_refused() -> None:
    with pytest.raises(ValueError):
        in_window(at(24, 3), time(1, 0), time(1, 0))
    with pytest.raises(ValueError):
        next_window_start(at(24, 3), time(1, 0), time(1, 0))


def test_window_refuses_a_naive_datetime() -> None:
    with pytest.raises(ValueError):
        in_window(datetime(2026, 10, 24, 3), *DEFAULT)
    with pytest.raises(ValueError):
        next_window_start(datetime(2026, 10, 24, 3), *DEFAULT)


def test_window_uses_the_wall_clock_of_the_given_zone_not_utc() -> None:
    instant = datetime(2026, 10, 24, 8, 0, tzinfo=UTC)
    assert not in_window(instant, *DEFAULT)
    assert in_window(instant.astimezone(TZ), *DEFAULT)


def test_next_window_start_before_inside_and_after_a_same_day_window() -> None:
    assert next_window_start(at(24, 0, 30), *DEFAULT) == at(24, 1)
    assert next_window_start(at(24, 3), *DEFAULT) == at(24, 1)
    assert next_window_start(at(24, 1), *DEFAULT) == at(24, 1)
    assert next_window_start(at(24, 7), *DEFAULT) == at(25, 1)
    assert next_window_start(at(24, 23), *DEFAULT) == at(25, 1)


def test_next_window_start_across_midnight() -> None:
    assert next_window_start(at(24, 12), *NIGHT) == at(24, 22)
    assert next_window_start(at(24, 23), *NIGHT) == at(24, 22)
    assert next_window_start(at(25, 2), *NIGHT) == at(24, 22)
    assert next_window_start(at(25, 6), *NIGHT) == at(25, 22)
    assert next_window_start(at(24, 12), *NIGHT).tzinfo == TZ
