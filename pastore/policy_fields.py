"""Field validators and YAML scalar quoting shared by the policy parser and the watch parser.

Depends on: PyYAML and stdlib only.
"""

import re
import sys
from collections.abc import Callable, Mapping
from typing import Any

import yaml


def check_fields(
    where: str, raw: object, shape: Mapping[str, Callable[[object], Any]], problems: list[str]
) -> dict[str, Any] | None:
    """Check a mapping with exactly the keys of `shape`; each validator returns the stored value or raises."""
    if not isinstance(raw, dict):
        problems.append(f"{where} must be a mapping with {', '.join(shape)}")
        return None
    before = len(problems)
    problems += [f"{where}: unknown key {k!r}" for k in raw if k not in shape]
    problems += [f"{where}: missing key {k!r}" for k in sorted(shape.keys() - raw.keys())]
    fields: dict[str, Any] = {}
    for key in (k for k in shape if k in raw):
        try:
            fields[key] = shape[key](raw[key])
        except ValueError as exc:
            problems.append(f"{where}: {key} {exc}")
    return fields if len(problems) == before else None


def matching(form: re.Pattern[str], what: str) -> Callable[[object], str]:
    """A validator for a string that fully matches `form`; `what` names the form in the message."""

    def check(value: object) -> str:
        if not isinstance(value, str) or not form.fullmatch(value):
            raise ValueError(f"must be {what}, got {value!r}")
        return value

    return check


def positive(value: object) -> int:
    """Return the value when it is an int above zero, else raise ValueError."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"must be a positive int, got {value!r}")
    return value


def quote(value: object) -> str:
    """The value as a double-quoted YAML scalar on one line."""
    dumped = yaml.safe_dump(str(value), default_style='"', width=sys.maxsize, allow_unicode=False)
    return dumped.removesuffix("\n")


def yaml_key(name: str) -> str:
    """The name as a YAML mapping key: bare when YAML reads it back as the same string, else quoted."""
    return name if yaml.safe_load(name) == name else quote(name)
