"""Pins the watch section's own parser and renderer and the field helpers they share with policy.py, each imported directly.

Depends on: pastore.policy_watch, pastore.policy_fields, pastore.spec and PyYAML only.
"""

import re
from typing import Any

import yaml

from pastore.policy_fields import check_fields, matching, positive, quote, yaml_key
from pastore.policy_watch import parse_watch, render_watch
from pastore.spec import PatternSignal, ValueSignal, WatchConfig

AWKWARD = WatchConfig(
    "health",
    (
        PatternSignal("health", "a@b.service", r"^ok\b \[x\] \d+ '\w'", 'q"\\b', first_bad=True),
        ValueSignal("lag", "c.service", r"lag=(\d+)\s*\\", 7),
    ),
)


def test_render_watch_then_parse_watch_returns_the_config_with_no_problems() -> None:
    """The renderer's lines are YAML that the parser turns back into the same config."""
    problems: list[str] = []
    loaded: dict[str, Any] = yaml.safe_load("\n".join(render_watch(AWKWARD)))
    assert parse_watch(loaded["watch"], problems) == AWKWARD
    assert problems == []


def test_render_watch_lays_out_the_section_header_then_one_line_per_signal() -> None:
    lines = render_watch(AWKWARD)
    assert lines[:3] == ["watch:", "  status: health", "  signals:"]
    assert len(lines) == 3 + len(AWKWARD.signals)
    assert all(line.startswith("    - {") for line in lines[3:])


def test_check_fields_reports_unknown_missing_and_invalid_keys_in_order() -> None:
    problems: list[str] = []
    shape = {"size": positive, "name": matching(re.compile(r"[a-z]+"), "lowercase letters")}
    raw = {"size": 0, "extra": 1}
    assert check_fields("thing", raw, shape, problems) is None
    assert problems == [
        "thing: unknown key 'extra'",
        "thing: missing key 'name'",
        "thing: size must be a positive int, got 0",
    ]


def test_check_fields_names_the_keys_it_wants_when_given_no_mapping() -> None:
    problems: list[str] = []
    assert check_fields("thing", ["x"], {"size": positive, "name": str}, problems) is None
    assert problems == ["thing must be a mapping with size, name"]


def test_check_fields_returns_the_stored_values_of_a_valid_mapping() -> None:
    problems: list[str] = []
    shape = {"size": positive, "name": lambda value: str(value).upper()}
    assert check_fields("thing", {"name": "abc", "size": 3}, shape, problems) == {
        "size": 3,
        "name": "ABC",
    }
    assert problems == []


def test_yaml_key_leaves_a_plain_name_bare_and_quotes_one_yaml_reads_as_another_type() -> None:
    assert yaml_key("health") == "health"
    assert yaml_key("true") == '"true"'
    assert yaml_key("12") == '"12"'
    assert yaml.safe_load(yaml_key("true") + ": 1") == {"true": 1}


def test_quote_double_quotes_on_one_line_and_escapes_non_ascii() -> None:
    assert quote('a"b') == '"a\\"b"'
    assert quote("café") == '"caf\\xE9"'
    assert "\n" not in quote("a\nb")
    assert "\n" not in quote("word " * 40)
