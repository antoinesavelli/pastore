"""Parse and render the optional `watch:` section of policy.yaml (the foreign-workload watch signals).

Depends on: policy_fields (check_fields, matching, positive, quote, yaml_key), spec (the signal and WatchConfig types), stdlib only.
"""

import re
from collections.abc import Callable

from pastore.policy_fields import check_fields, matching, positive, quote, yaml_key
from pastore.spec import PatternSignal, Signal, ValueSignal, WatchConfig

WATCH_KEYS = frozenset({"status", "signals"})
MAX_SIGNALS = 8
SIGNAL_NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
UNIT_NAME = re.compile(r"[A-Za-z0-9:_.\\@][A-Za-z0-9:_.\\@-]*")


def _regex(groups: int | None = None) -> Callable[[object], str]:
    """A validator for a non-empty string that compiles as a regex, with exactly `groups` capture groups when given."""

    def check(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"must be a non-empty regex string, got {value!r}")
        try:
            compiled = re.compile(value)
        except (re.error, OverflowError, RecursionError) as exc:
            raise ValueError(f"is not a valid regex ({exc}): {value!r}") from None
        if groups is not None and compiled.groups != groups:
            raise ValueError(f"must have exactly {groups} capture group, got {compiled.groups}")
        return value

    return check


def _parse_signal(where: str, raw: object, problems: list[str]) -> Signal | None:
    if not isinstance(raw, dict):
        problems.append(f"{where} must be a mapping with name, unit, good + bad or value + limit")
        return None
    before = len(problems)
    first_bad = raw.get("first_bad", False)
    if type(first_bad) is not bool:
        problems.append(f"{where}: first_bad must be true or false, got {first_bad!r}")
    body = {k: v for k, v in raw.items() if k != "first_bad"}
    common = {
        "name": matching(SIGNAL_NAME, f"a name matching {SIGNAL_NAME.pattern}"),
        "unit": matching(UNIT_NAME, "a systemd unit name (not starting with '-')"),
    }
    kind = ValueSignal if "value" in body or "limit" in body else PatternSignal
    if kind is ValueSignal:
        shape = {**common, "value": _regex(1), "limit": positive}
    else:
        shape = {**common, "good": _regex(), "bad": _regex()}
    fields = check_fields(where, body, shape, problems)
    return None if fields is None or len(problems) > before else kind(**fields, first_bad=first_bad)


def _parse_signals(raw: object, problems: list[str]) -> tuple[Signal, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_SIGNALS:
        problems.append(f"watch.signals must be a list of 1 to {MAX_SIGNALS} signals")
        return ()
    parsed = [_parse_signal(f"watch.signals[{i}]", item, problems) for i, item in enumerate(raw)]
    signals = tuple(signal for signal in parsed if signal is not None)
    names = [signal.name for signal in signals]
    problems += [
        f"watch.signals: name {n!r} is used twice"
        for n in dict.fromkeys(names)
        if names.count(n) > 1
    ]
    return signals


def parse_watch(raw: object, problems: list[str]) -> WatchConfig | None:
    """Validate the `watch:` value, appending problems; the config, or None when there is one."""
    if not isinstance(raw, dict):
        problems.append(f"watch must be a mapping with {', '.join(sorted(WATCH_KEYS))}")
        return None
    before = len(problems)
    problems += [f"watch: unknown key {k!r}" for k in raw if k not in WATCH_KEYS]
    problems += [f"watch: missing key {k!r}" for k in sorted(WATCH_KEYS - raw.keys())]
    signals = _parse_signals(raw["signals"], problems) if "signals" in raw else ()
    status = raw.get("status")
    if len(problems) > before:
        return None
    if not isinstance(status, str) or not any(
        isinstance(s, PatternSignal) and s.name == status for s in signals
    ):
        problems.append(f"watch: status {status!r} must name a signal with good and bad patterns")
        return None
    return WatchConfig(status, signals)


def _single(value: str) -> str:
    """A single-quoted scalar, which keeps a regex's backslashes as written; text that is not printable ASCII is double-quoted."""
    if value.isascii() and value.isprintable():
        return "'" + value.replace("'", "''") + "'"
    return quote(value)


def _signal_line(signal: Signal) -> str:
    if isinstance(signal, ValueSignal):
        match = f"value: {_single(signal.value)}, limit: {signal.limit}"
    else:
        match = f"good: {_single(signal.good)}, bad: {_single(signal.bad)}"
    first_bad = ", first_bad: true" if signal.first_bad else ""
    return (
        f"    - {{name: {yaml_key(signal.name)}, unit: {quote(signal.unit)}, {match}{first_bad}}}"
    )


def render_watch(watch: WatchConfig) -> list[str]:
    """The `watch:` section as policy.yaml lines."""
    return [
        "watch:",
        f"  status: {yaml_key(watch.status)}",
        "  signals:",
        *(_signal_line(s) for s in watch.signals),
    ]
