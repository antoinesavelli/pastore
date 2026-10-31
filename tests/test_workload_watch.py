"""R35 and check 15: the foreign-workload watch over recorded journal lines, the journalctl reader and the spans.

Depends on: pastore.workload_watch and pastore.spec (WatchConfig); a scripted journalctl runner defined here (no real journalctl, systemd or ssh).
"""

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pastore import workload_watch as tw
from pastore.spec import PatternSignal, ValueSignal, WatchConfig
from pastore.workload_watch import Disruption, JournalError, Line, Overlap, Span

HEALTH_UNIT = "workload-health.service"
WORKLOAD_UNIT = "workload.service"
CONFIG = WatchConfig(
    "health",
    (
        PatternSignal("health", HEALTH_UNIT, r"^ok\b", r"^down\b"),
        PatternSignal("link", WORKLOAD_UNIT, "link restored", "link lost", first_bad=True),
        ValueSignal("lag", WORKLOAD_UNIT, r"lag (\d+)s", 900),
    ),
)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
T0 = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
UP = "ok 12ms"
DOWN = "down since 11:58, connection refused"
LOST = "link lost for 312s, resuming soon \u2014 retrying"
RECOVERED = "link restored \u2014 resuming"
LAG = "lag {}s (clamped)"


def at(minutes: float) -> datetime:
    """T0 plus minutes."""
    return T0 + timedelta(minutes=minutes)


def reach(minutes: float, up: bool) -> Line:
    """A recorded health line at T0 + minutes."""
    return Line(at(minutes), HEALTH_UNIT, UP if up else DOWN)


def engine(minutes: float, message: str) -> Line:
    """A recorded workload line at T0 + minutes."""
    return Line(at(minutes), WORKLOAD_UNIT, message)


def watch(now: datetime, spans: Sequence[Span], read: tw.JournalReader) -> tw.Watch:
    """The watch under CONFIG."""
    return tw.watch(CONFIG, now, spans, read)


def disruptions(lines: Sequence[Line]) -> list[Disruption]:
    """The disruptions under CONFIG."""
    return tw.disruptions(CONFIG, lines)


def span(start: float, end: float | None, run_id: str = "r1", ref: str = "example/eval") -> Span:
    """An executor run from T0 + start to T0 + end minutes (None = running)."""
    return Span(run_id, ref, at(start), None if end is None else at(end))


def fixed(lines: Sequence[Line]) -> tw.JournalReader:
    """A reader that serves the given lines per unit."""
    return lambda unit, since: [ln for ln in lines if ln.unit == unit]


def micros(when: datetime) -> int:
    """Microseconds since the epoch, as journald stamps them."""
    return (when - EPOCH) // timedelta(microseconds=1)


def entry(when: datetime, message: object, *, ascii_only: bool = True) -> str:
    """One `journalctl -o json` line."""
    body = {"__REALTIME_TIMESTAMP": str(micros(when)), "MESSAGE": message}
    return json.dumps(body, ensure_ascii=ascii_only)


class Journal:
    """A scripted journalctl: JSON lines per unit, one exit code, every argv recorded."""

    def __init__(self, by_unit: Mapping[str, Sequence[str]] | None = None, returncode: int = 0):
        self.by_unit = dict(by_unit or {})
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        unit = argv[list(argv).index("-u") + 1]
        out = "".join(raw + "\n" for raw in self.by_unit.get(unit, []))
        return subprocess.CompletedProcess(list(argv), self.returncode, out, "")


def one_line(raw: str) -> tw.JournalReader:
    """A reader over one health journal line."""
    return tw.journal_reader(Journal({HEALTH_UNIT: [raw]}))


def test_check_15_the_watched_units_are_each_read_once_in_first_seen_order() -> None:
    assert CONFIG.units == (HEALTH_UNIT, WORKLOAD_UNIT)
    signals = (
        PatternSignal("a", "y.service", "g", "b"),
        PatternSignal("b", "x.service", "g", "b"),
        ValueSignal("c", "y.service", r"(\d+)", 5),
    )
    assert WatchConfig("a", signals).units == ("y.service", "x.service")


# --- check 15: overlaps -----------------------------------------------------------------------


def test_check_15_up_to_down_during_a_run_is_one_overlap_naming_both_spans() -> None:
    lines = [reach(-5, True), reach(10, False)]
    result = watch(at(20), [span(0, 30)], fixed(lines))
    assert len(result.overlaps) == 1
    assert result.status == "down"
    only = result.overlaps[0]
    assert only.disruption == Disruption("health", at(10), None)
    assert only.detail() == (
        "run=r1 example/eval run=2026-10-31T12:00:00Z..2026-10-31T12:30:00Z "
        "signal=health disruption=2026-10-31T12:10:00Z..ongoing"
    )
    assert only.key() == "r1 health 2026-10-31T12:10:00Z"


@pytest.mark.parametrize(
    ("after_end_min", "expected"), [(0, 1), (9, 1), (10, 1), (10.5, 0), (11, 0), (60, 0)]
)
def test_check_15_a_disruption_after_the_run_ends_counts_for_ten_minutes(
    after_end_min: float, expected: int
) -> None:
    lines = [reach(-5, True), reach(30 + after_end_min, False)]
    result = watch(at(100), [span(0, 30)], fixed(lines))
    assert len(result.overlaps) == expected


@pytest.mark.parametrize(("offset_s", "expected"), [(-60, 0), (-1, 0), (0, 1), (1, 1)])
def test_check_15_a_transition_before_the_run_start_does_not_overlap(
    offset_s: int, expected: int
) -> None:
    down_at = at(0) + timedelta(seconds=offset_s)
    lines = [Line(at(-20), HEALTH_UNIT, UP), Line(down_at, HEALTH_UNIT, DOWN)]
    result = watch(at(20), [span(0, 30)], fixed(lines))
    assert len(result.overlaps) == expected


def test_check_15_up_throughout_is_armed_with_no_overlap() -> None:
    lines = [reach(m, True) for m in (-10, 0, 10, 20)]
    result = watch(at(25), [span(0, None)], fixed(lines))
    assert result == tw.Watch("armed", ())


def test_check_15_down_the_whole_window_is_down_with_no_overlap() -> None:
    lines = [reach(m, False) for m in (-10, 0, 10, 20)]
    result = watch(at(25), [span(-12, None)], fixed(lines))
    assert result == tw.Watch("down", ())
    assert disruptions([reach(0, False), reach(5, False)]) == []


def test_check_15_back_up_then_down_during_a_run_is_an_overlap() -> None:
    lines = [reach(-10, False), reach(5, True), reach(20, False)]
    spans = [span(0, None)]
    assert watch(at(10), spans, fixed(lines[:2])) == tw.Watch("armed", ())
    result = watch(at(25), spans, fixed(lines))
    assert result.status == "down"
    assert [o.disruption for o in result.overlaps] == [Disruption("health", at(20), None)]


def test_check_15_a_workload_that_stays_down_keeps_its_overlap_on_later_reads() -> None:
    lines = [reach(0, True), reach(10, False)] + [reach(m, False) for m in range(11, 60, 7)]
    first = watch(at(15), [span(5, 20)], fixed(lines[:3]))
    later = watch(at(60), [span(5, 20)], fixed(lines))
    assert [o.key() for o in first.overlaps] == [o.key() for o in later.overlaps]
    assert len(later.overlaps) == 1 and later.status == "down"


def test_check_15_a_disruption_ending_before_the_run_does_not_overlap() -> None:
    lines = [reach(-30, True), reach(-20, False), reach(-10, True)]
    result = watch(at(5), [span(0, None)], fixed(lines))
    assert result == tw.Watch("armed", ())


def test_check_15_lag_900_is_good_and_901_is_bad() -> None:
    ok = [engine(m, LAG.format(900)) for m in (-5, 5, 10)]
    assert watch(at(20), [span(0, 30)], fixed(ok)).overlaps == ()
    lines = [engine(-5, LAG.format(900)), engine(10, LAG.format(901))]
    result = watch(at(20), [span(0, 30)], fixed(lines))
    assert [o.disruption for o in result.overlaps] == [Disruption("lag", at(10), None)]
    assert result.overlaps[0].key() == "r1 lag 2026-10-31T12:10:00Z"


def test_check_15_the_lag_text_is_found_anywhere_in_the_message() -> None:
    lines = [engine(0, "WARN " + LAG.format(100)), engine(5, "WARN " + LAG.format(901))]
    assert disruptions(lines) == [Disruption("lag", at(5), None)]


def test_check_15_a_first_lag_over_the_limit_is_an_outage_from_before_the_window() -> None:
    lines = [engine(5, LAG.format(901)), engine(10, LAG.format(950))]
    assert watch(at(20), [span(0, 30)], fixed(lines)).overlaps == ()


def test_check_15_the_first_link_alarm_counts_repeats_are_one_and_recovery_ends_it() -> None:
    lines = [engine(10, LOST), engine(11, "link lost count=2"), engine(12, LOST)]
    ongoing = watch(at(20), [span(0, 30)], fixed(lines))
    assert [o.disruption for o in ongoing.overlaps] == [Disruption("link", at(10), None)]
    lines.append(engine(15, RECOVERED))
    ended = watch(at(20), [span(0, 30)], fixed(lines))
    assert [o.disruption for o in ended.overlaps] == [Disruption("link", at(10), at(15))]
    lines += [engine(18, LOST)]
    again = watch(at(20), [span(0, 30)], fixed(lines))
    assert [o.disruption.start for o in again.overlaps] == [at(10), at(18)]


@pytest.mark.parametrize("alarm", [LOST, "WARN link lost count=2"])
def test_check_15_either_link_alarm_text_starts_a_link_disruption(alarm: str) -> None:
    got = disruptions([engine(10, alarm)])
    assert got == [Disruption("link", at(10), None)]


def test_check_15_no_lines_is_no_signal() -> None:
    assert watch(at(20), [span(0, 30)], fixed([])) == tw.Watch("no-signal", ())
    assert watch(at(20), [], fixed([])) == tw.Watch("no-signal", ())


def test_check_15_status_comes_from_the_newest_sample_in_the_last_30_minutes() -> None:
    assert watch(at(40), [], fixed([reach(10, True)])).status == "armed"
    assert watch(at(40), [], fixed([reach(9.9, True)])).status == "no-signal"
    mixed = [reach(15, True), reach(20, False), reach(25, True)]
    assert watch(at(40), [], fixed(mixed)).status == "armed"
    assert watch(at(40), [], fixed(mixed[:2])).status == "down"
    old_bad = [reach(0, False), reach(35, True)]
    assert watch(at(40), [], fixed(old_bad)).status == "armed"


def test_check_15_engine_lines_do_not_set_the_status() -> None:
    assert watch(at(20), [], fixed([engine(10, LAG.format(100))])).status == "no-signal"


def test_check_15_overlaps_come_from_the_engine_even_when_the_status_signal_is_down() -> None:
    lines = [reach(-5, False), engine(5, RECOVERED), engine(10, LOST)]
    result = watch(at(20), [span(0, 30)], fixed(lines))
    assert result.status == "down"
    assert [o.disruption.signal for o in result.overlaps] == ["link"]


def test_check_15_unread_journal_is_unreadable_and_never_raises() -> None:
    def broken(unit: str, since: datetime) -> list[Line]:
        raise JournalError("journalctl exit 1")

    assert watch(at(20), [span(0, 30)], broken) == tw.Watch("unreadable: journalctl exit 1", ())


def test_check_15_end_to_end_through_the_journalctl_reader() -> None:
    journal = Journal(
        {
            HEALTH_UNIT: [entry(at(-5), UP), entry(at(10), DOWN)],
            WORKLOAD_UNIT: [entry(at(12), LOST)],
        }
    )
    result = watch(at(20), [span(0, 30)], tw.journal_reader(journal))
    assert result.status == "down"
    assert [o.disruption.signal for o in result.overlaps] == ["health", "link"]


# --- check 15: the signals -------------------------------------------------------------------


def test_check_15_unrelated_lines_are_ignored() -> None:
    noise = [
        Line(at(1), HEALTH_UNIT, "heartbeat ok"),
        Line(at(2), HEALTH_UNIT, "okay"),
        Line(at(3), HEALTH_UNIT, "downed"),
        Line(at(4), HEALTH_UNIT, ""),
        Line(at(5), WORKLOAD_UNIT, "lag soon"),
        Line(at(6), WORKLOAD_UNIT, "tick complete"),
        Line(at(7), "other.service", UP),
        Line(at(9), "other.service", DOWN),
        Line(at(8), "other.service", LOST),
    ]
    assert disruptions(noise) == []
    assert watch(at(20), [span(0, None)], fixed(noise)).status == "no-signal"


def test_check_15_a_signal_is_judged_against_its_own_previous_sample() -> None:
    lines = [reach(0, True), engine(1, LOST), reach(2, False), engine(3, LAG.format(901))]
    found = disruptions(lines)
    assert found == [Disruption("link", at(1), None), Disruption("health", at(2), None)]


def test_check_15_disruptions_sort_their_input_by_time() -> None:
    shuffled = [reach(10, False), reach(0, True), reach(20, True)]
    assert disruptions(shuffled) == [Disruption("health", at(10), at(20))]


def test_check_15_a_lag_number_too_large_to_parse_is_over_the_limit_not_an_error() -> None:
    lines = [engine(0, LAG.format(100)), engine(1, "lag " + "9" * 5000 + "s")]
    assert disruptions(lines) == [Disruption("lag", at(1), None)]
    zeros = [engine(0, LAG.format(100)), engine(1, "lag " + "0" * 5000 + "5s")]
    assert disruptions(zeros) == []


def test_check_15_overlap_order_is_disruption_start_then_run_id() -> None:
    found = [Disruption("link", at(12), None), Disruption("health", at(10), at(11))]
    spans = [span(0, None, "b-run"), span(5, 30, "a-run"), span(11, None, "c-run")]
    got = [(o.disruption.start, o.span.run_id) for o in tw.overlaps(spans, found)]
    assert got == [
        (at(10), "a-run"),
        (at(10), "b-run"),
        (at(12), "a-run"),
        (at(12), "b-run"),
        (at(12), "c-run"),
    ]


def test_check_15_a_run_that_started_after_the_disruption_does_not_overlap() -> None:
    assert tw.overlaps([span(11, None)], [Disruption("link", at(10), None)]) == []


def test_check_15_key_and_detail_hold_only_our_own_fields() -> None:
    running = Overlap(span(0, None), Disruption("lag", at(10), at(14)))
    assert running.detail() == (
        "run=r1 example/eval run=2026-10-31T12:00:00Z..running "
        "signal=lag disruption=2026-10-31T12:10:00Z..2026-10-31T12:14:00Z"
    )
    assert running.key() == "r1 lag 2026-10-31T12:10:00Z"
    assert "\n" not in running.detail()
    est = timezone(timedelta(hours=-4))
    start, end, down = (x.astimezone(est) for x in (at(0), at(30), at(10)))
    offset = Overlap(Span("r1", "example/eval", start, end), Disruption("link", down, None))
    assert offset.key() == "r1 link 2026-10-31T12:10:00Z"
    assert "run=2026-10-31T12:00:00Z..2026-10-31T12:30:00Z" in offset.detail()


# --- check 15: read_since --------------------------------------------------------------------


def test_check_15_read_since_is_the_earlier_of_the_status_window_and_the_lookback() -> None:
    now = at(60)
    assert tw.read_since(now, []) == at(30)
    assert tw.read_since(now, [span(55, None)]) == at(30)
    assert tw.read_since(now, [span(40, 50)]) == at(25)
    assert tw.read_since(now, [span(50, None), span(10, 20), span(40, 41)]) == at(-5)
    assert tw.read_since(now, [span(45, None)]) == at(30)


# --- check 15: the journalctl reader ---------------------------------------------------------


def test_check_15_the_reader_runs_exactly_the_documented_journalctl_argv() -> None:
    journal = Journal()
    since = datetime(2026, 10, 31, 12, 0, 7, 900000, tzinfo=UTC)
    assert tw.journal_reader(journal)(WORKLOAD_UNIT, since) == []
    assert journal.calls == [
        [
            "journalctl",
            "-u",
            WORKLOAD_UNIT,
            f"--since=@{int(since.timestamp())}",
            "-o",
            "json",
            "--output-fields=MESSAGE",
            "--no-pager",
        ]
    ]
    assert journal.calls[0][3] == "--since=@1793448007"


def test_check_15_watch_reads_both_units_with_no_ssh_and_no_user_scope() -> None:
    journal = Journal()
    watch(at(20), [span(0, 30)], tw.journal_reader(journal))
    assert [c[2] for c in journal.calls] == list(CONFIG.units)
    for argv in journal.calls:
        assert argv[0] == "journalctl"
        assert "ssh" not in argv and "--user" not in argv
        assert not any("sudo" in a or "pkexec" in a for a in argv)
        assert argv[3] == f"--since=@{int(at(-15).timestamp())}"


def test_check_15_the_module_source_has_no_ssh_user_scope_or_sudo() -> None:
    source = Path(tw.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\b(ssh|sudo|pkexec|systemctl)\b|--user", source)


@pytest.mark.parametrize("returncode", [1, 124, 127, -9])
def test_check_15_a_journalctl_failure_is_unreadable(returncode: int) -> None:
    journal = Journal({HEALTH_UNIT: [entry(at(0), UP)]}, returncode=returncode)
    with pytest.raises(JournalError, match=f"^journalctl exit {returncode}$"):
        tw.journal_reader(journal)(HEALTH_UNIT, at(-30))
    result = watch(at(20), [], tw.journal_reader(journal))
    assert result == tw.Watch(f"unreadable: journalctl exit {returncode}", ())


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "{",
        '["a"]',
        "42",
        '{"MESSAGE": "ok"}',
        '{"MESSAGE": "ok", "__REALTIME_TIMESTAMP": "abc"}',
        '{"MESSAGE": "ok", "__REALTIME_TIMESTAMP": "12.5"}',
        '{"MESSAGE": "ok", "__REALTIME_TIMESTAMP": "-5"}',
        '{"MESSAGE": "ok", "__REALTIME_TIMESTAMP": ""}',
        '{"MESSAGE": "ok", "__REALTIME_TIMESTAMP": 1793448000000000}',
        '{"MESSAGE": "ok", "__REALTIME_TIMESTAMP": "' + "9" * 30 + '"}',
        '{"MESSAGE": "ok", "__REALTIME_TIMESTAMP": "' + "1" * 5000 + '"}',
        "[" * 100000,
    ],
)
def test_check_15_a_malformed_journal_line_raises_a_journal_error(raw: str) -> None:
    with pytest.raises(JournalError):
        one_line(raw)(HEALTH_UNIT, at(-30))


def test_check_15_a_malformed_journal_makes_the_watch_unreadable_with_our_words_only() -> None:
    secret = "SECRET-journal-text"
    journal = Journal({HEALTH_UNIT: [entry(at(0), UP), "{" + secret]})
    result = watch(at(20), [], tw.journal_reader(journal))
    assert result.status.startswith("unreadable: ")
    assert secret not in result.status and len(result.status) < 80
    assert result.overlaps == ()


def test_check_15_a_message_that_is_not_text_is_skipped() -> None:
    binary = json.dumps({"__REALTIME_TIMESTAMP": str(micros(at(1))), "MESSAGE": [117, 112]})
    missing = json.dumps({"__REALTIME_TIMESTAMP": str(micros(at(2)))})
    null = entry(at(3), None)
    journal = Journal({HEALTH_UNIT: [binary, missing, null, entry(at(4), UP)]})
    got = tw.journal_reader(journal)(HEALTH_UNIT, at(-30))
    assert got == [Line(at(4), HEALTH_UNIT, UP)]


def test_check_15_the_reader_returns_sorted_aware_utc_lines_with_microseconds() -> None:
    late = at(5) + timedelta(microseconds=123456)
    journal = Journal({WORKLOAD_UNIT: [entry(late, "b"), entry(at(1), "a"), "", "   "]})
    got = tw.journal_reader(journal)(WORKLOAD_UNIT, at(-30))
    assert [ln.message for ln in got] == ["a", "b"]
    assert [ln.at for ln in got] == [at(1), late]
    assert all(ln.unit == WORKLOAD_UNIT and ln.at.utcoffset() == timedelta(0) for ln in got)
    assert micros(got[1].at) == micros(late)


def test_check_15_a_unicode_line_separator_inside_a_message_does_not_split_the_line() -> None:
    message = "lag 5s\u2028and\x85more\u2029text"
    raw = entry(at(1), message, ascii_only=False)
    got = one_line(raw)(HEALTH_UNIT, at(-30))
    assert [ln.message for ln in got] == [message]


def test_check_15_terminal_escapes_in_a_message_stay_out_of_the_overlap_detail() -> None:
    evil = "down \x1b[31mEVIL\x1b[0m"
    lines = [reach(0, True), engine(10, "\x1b]0;pwned\x07 " + LOST)]
    result = watch(at(20), [span(0, 30)], fixed(lines + [Line(at(11), HEALTH_UNIT, evil)]))
    assert {o.disruption.signal for o in result.overlaps} == {"health", "link"}
    for o in result.overlaps:
        assert "\x1b" not in o.detail() and "\x1b" not in o.key() and "EVIL" not in o.detail()


# --- check 15: the configured signals ------------------------------------------------------


def test_check_15_a_line_matching_both_good_and_bad_is_bad() -> None:
    both = WatchConfig("s", (PatternSignal("s", HEALTH_UNIT, "fine", "fine but slow"),))
    lines = [Line(at(0), HEALTH_UNIT, "fine"), Line(at(5), HEALTH_UNIT, "fine but slow")]
    assert tw.disruptions(both, lines) == [Disruption("s", at(5), None)]


def test_check_15_the_status_comes_from_the_signal_the_config_names() -> None:
    by_link = WatchConfig("link", CONFIG.signals)
    lines = [reach(10, True), engine(12, LOST)]
    assert tw.watch(by_link, at(20), [], fixed(lines)).status == "down"
    assert tw.watch(by_link, at(20), [], fixed(lines[:1])).status == "no-signal"
    assert watch(at(20), [], fixed(lines)).status == "armed"


def test_check_15_one_line_can_be_a_sample_of_every_signal_of_its_unit() -> None:
    lines = [engine(0, LAG.format(100)), engine(5, "link lost, " + LAG.format(950))]
    assert disruptions(lines) == [Disruption("lag", at(5), None), Disruption("link", at(5), None)]


def test_check_15_a_value_that_is_not_digits_or_not_captured_is_no_sample() -> None:
    word = WatchConfig("n", (ValueSignal("n", HEALTH_UNIT, r"n=(\w+)", 10),))
    odd = ["n=abc", "n=\u00b2", "n=", "n=-1", "n=1.5x"]
    lines = [Line(at(m), HEALTH_UNIT, text) for m, text in enumerate(["n=5", *odd])]
    assert tw.disruptions(word, lines) == []
    lines.append(Line(at(9), HEALTH_UNIT, "n=11"))
    assert tw.disruptions(word, lines) == [Disruption("n", at(9), None)]
    optional = WatchConfig("n", (ValueSignal("n", HEALTH_UNIT, r"m(=\d+)?", 10),))
    assert (
        tw.disruptions(optional, [Line(at(0), HEALTH_UNIT, "m=5"), Line(at(1), HEALTH_UNIT, "m")])
        == []
    )


def test_check_15_a_value_limit_is_inclusive_and_leading_zeros_do_not_count() -> None:
    config = WatchConfig("n", (ValueSignal("n", HEALTH_UNIT, r"n=(\d+)", 10),))
    ok = [
        Line(at(m), HEALTH_UNIT, f"n={v}")
        for m, v in enumerate(("0", "010", "10", "0000000000010"))
    ]
    assert tw.disruptions(config, ok) == []
    bad = [Line(at(0), HEALTH_UNIT, "n=1"), Line(at(1), HEALTH_UNIT, "n=11")]
    assert tw.disruptions(config, bad) == [Disruption("n", at(1), None)]
    nine = [Line(at(0), HEALTH_UNIT, "n=1"), Line(at(1), HEALTH_UNIT, "n=" + "9" * 9)]
    assert tw.disruptions(config, nine) == [Disruption("n", at(1), None)]
