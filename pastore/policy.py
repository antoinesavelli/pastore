"""Parse and validate policy.yaml v2 (hosts, sources, consumers, listed pairs, the optional watch), keep the last good copy, render and edit.

Depends on: spec (Host, Level, Pair, SourceRule, the signal and WatchConfig types, WORKSTATION and the source-kind lists), PyYAML and stdlib only.
"""

import difflib
import ipaddress
import re
import sys
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError

from pastore.spec import (
    EXCLUDABLE_KINDS,
    SOURCE_KINDS,
    WORKSTATION,
    Host,
    Level,
    Pair,
    PatternSignal,
    Signal,
    SourceRule,
    ValueSignal,
    WatchConfig,
)

VERSION = 2
KEYS = frozenset({"version", "hosts", "sources", "consumers", "jobs"})
OPTIONAL_KEYS = frozenset({"watch"})
RULE_KEYS = frozenset({"level", "exclude", *SOURCE_KINDS})
WATCH_KEYS = frozenset({"status", "signals"})
MAX_SIGNALS = 8
SIGNAL_NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
UNIT_NAME = re.compile(r"[A-Za-z0-9:_.\\@][A-Za-z0-9:_.\\@-]*")
MANIFEST_NAME = "gpu-tasks.yaml"
NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
URL = re.compile(r"http://(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+):0*([0-9]{1,5})/?")
LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
NUMERIC = re.compile(r"[0-9.]+")
LEVELS = {level.name.lower(): level for level in Level}
MAX_COMM = 15
MERGE_TAG = "tag:yaml.org,2002:merge"
HEADER = (
    "# The single human gate: only the task+model pairs listed under `jobs` ever run, in this order"
    " (first = highest priority).\n"
    "# Adding a pair is ask-first: `pastore add <consumer>/<id> --model <name> --evidence <path>`"
    " with a passing\n"
    "# `pastore check --contract <consumer>/<id>` (docs/DESIGN.md R29).\n"
    "# Each host's context is its Ollama's default context length in tokens;"
    " placement counts KV cache per slot at it (D-041).\n"
    "# Each host's vram_gib is the usable VRAM its Ollama reports, in GiB (D-050).\n"
)


class PolicyError(Exception):
    """The policy file is unreadable or invalid; the message lists every problem."""


@dataclass(frozen=True)
class Policy:
    """A validated policy: hosts, source-level rules, consumer manifests, ordered task+model pairs, the optional watch."""

    hosts: Mapping[str, Host]
    sources: tuple[SourceRule, ...]
    consumers: Mapping[str, Path]
    jobs: tuple[Pair, ...]
    watch: WatchConfig | None = None


def _fields(
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


def _matching(form: re.Pattern[str], what: str) -> Callable[[object], str]:
    def check(value: object) -> str:
        if not isinstance(value, str) or not form.fullmatch(value):
            raise ValueError(f"must be {what}, got {value!r}")
        return value

    return check


def _positive(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"must be a positive int, got {value!r}")
    return value


def _positive_number(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 < value <= sys.float_info.max
    ):
        raise ValueError(f"must be a positive number, got {value!r}")
    return float(value)


def _host_ok(host: str) -> bool:
    try:
        if host.startswith("["):
            ipaddress.IPv6Address(host[1:-1])
        elif NUMERIC.fullmatch(host):
            ipaddress.IPv4Address(host)
        else:
            return len(host) <= 253 and all(LABEL.fullmatch(label) for label in host.split("."))
    except ValueError:
        return False
    return True


def _normalize_url(value: object) -> str:
    match = URL.fullmatch(value) if isinstance(value, str) else None
    if match is None or not _host_ok(match[1]) or not 0 < int(match[2]) < 65536:
        raise ValueError(
            f"must look like http://<host>:<port> (no path or credentials), got {value!r}"
        )
    return f"http://{match[1]}:{int(match[2])}"


def _parse_host(name: object, body: object, problems: list[str]) -> Host | None:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        problems.append(f"host {name!r}: name must match {NAME.pattern}")
        return None
    shape = {
        "url": _normalize_url,
        "vram_gib": _positive_number,
        "max_models": _positive,
        "parallel": _positive,
        "context": _positive,
    }
    fields = _fields(f"host {name!r}", body, shape, problems)
    return None if fields is None else Host(name, **fields)


def _parse_hosts(raw: object, problems: list[str]) -> dict[str, Host]:
    if not isinstance(raw, dict) or not raw:
        problems.append("hosts must be a non-empty mapping")
        return {}
    if WORKSTATION not in raw:
        problems.append(f"hosts must include {WORKSTATION!r}")
    parsed = {name: _parse_host(name, body, problems) for name, body in raw.items()}
    return {name: host for name, host in parsed.items() if host is not None}


def _glob(value: str) -> str:
    if "/" in value:
        raise ValueError(f"{value!r} must be one cgroup leaf name without '/'")
    return value


def _network(value: str) -> str:
    return str(ipaddress.ip_network(value, strict=True))


def _comm(value: str) -> str:
    if "/" in value or len(value.encode()) > MAX_COMM:
        raise ValueError(f"{value!r} must be a process name without '/', at most {MAX_COMM} bytes")
    return value


def _directory(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or value.startswith("//"):
        raise ValueError(
            f"{value!r} must be an absolute directory path without '..' or a leading '//'"
        )
    return str(path)


NORMALIZERS: dict[str, Callable[[str], str]] = {
    "cgroup": _glob,
    "peer": _network,
    "ancestor": _comm,
    "under": _directory,
}


def _normalize_list(raw: object, kind: str, where: str, problems: list[str]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        problems.append(f"{where} must be a non-empty list of strings")
        return ()
    out: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value:
            problems.append(f"{where}: {value!r} must be a non-empty string")
            continue
        try:
            out.append(NORMALIZERS[kind](value))
        except ValueError as exc:
            problems.append(f"{where}: {exc}")
    return tuple(out)


def _parse_exclude(
    raw: dict[object, object], kind: str, where: str, problems: list[str]
) -> tuple[str, ...]:
    if "exclude" not in raw:
        return ()
    if kind not in EXCLUDABLE_KINDS:
        problems.append(f"{where}: exclude is not allowed with {kind}")
        return ()
    return _normalize_list(raw["exclude"], kind, f"{where}.exclude", problems)


def _parse_rule(index: int, raw: object, problems: list[str]) -> SourceRule | None:
    where = f"sources[{index}]"
    if not isinstance(raw, dict):
        problems.append(
            f"{where} must be a mapping with level, one of {', '.join(SOURCE_KINDS)}, optional exclude"
        )
        return None
    before = len(problems)
    problems += [f"{where}: unknown key {k!r}" for k in raw if k not in RULE_KEYS]
    level = LEVELS.get(raw["level"]) if isinstance(raw.get("level"), str) else None
    if level is None:
        problems.append(
            f"{where}: level must be one of {', '.join(LEVELS)}, got {raw.get('level')!r}"
        )
    kinds = [k for k in SOURCE_KINDS if k in raw]
    if len(kinds) != 1:
        problems.append(
            f"{where}: needs exactly one of {', '.join(SOURCE_KINDS)}, got {len(kinds)}"
        )
        return None
    values = _normalize_list(raw[kinds[0]], kinds[0], f"{where}.{kinds[0]}", problems)
    exclude = _parse_exclude(raw, kinds[0], where, problems)
    if level is None or len(problems) > before:
        return None
    return SourceRule(level, kinds[0], values, exclude)


def _hearth_problems(raw: list[object], first: SourceRule | None) -> list[str]:
    problems: list[str] = []
    if first is not None and (first.level, first.kind) != (Level.HEARTH, "cgroup"):
        problems.append(
            "sources[0] must be level hearth with kind cgroup (hearth is matched first and exactly)"
        )
    return problems + [
        f"sources[{i}]: level hearth is only allowed at sources[0]"
        for i, item in enumerate(raw[1:], 1)
        if isinstance(item, dict) and item.get("level") == "hearth"
    ]


def _parse_sources(raw: object, problems: list[str]) -> tuple[SourceRule, ...]:
    if not isinstance(raw, list) or not raw:
        problems.append("sources must be a non-empty list")
        return ()
    parsed = [_parse_rule(i, item, problems) for i, item in enumerate(raw)]
    problems += _hearth_problems(raw, parsed[0])
    return tuple(rule for rule in parsed if rule is not None)


def _parse_consumers(raw: object, problems: list[str]) -> dict[str, Path]:
    if not isinstance(raw, dict):
        problems.append("consumers must be a mapping (use {} for none)")
        return {}
    out: dict[str, Path] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not NAME.fullmatch(name):
            problems.append(f"consumer name {name!r} must match {NAME.pattern}")
        elif not isinstance(value, str) or not Path(value).is_absolute():
            problems.append(f"consumer {name}: path must be an absolute string, got {value!r}")
        elif Path(value).name != MANIFEST_NAME:
            problems.append(f"consumer {name}: path must end in {MANIFEST_NAME}, got {value!r}")
        else:
            out[name] = Path(value)
    return out


def check_ref(ref: object, consumers: Mapping[str, Path]) -> str | None:
    """Return the problem with a job ref, or None when it is `<known consumer>/<id>`."""
    parts = ref.split("/") if isinstance(ref, str) else []
    if len(parts) != 2 or not NAME.fullmatch(parts[1]):
        return f"job {ref!r} must look like <consumer>/<id>"
    if parts[0] not in consumers:
        return f"job {ref!r}: consumer {parts[0]!r} is not listed under consumers"
    return None


def _task(consumers: Mapping[str, Path]) -> Callable[[object], str]:
    def check(value: object) -> str:
        if problem := check_ref(value, consumers):
            raise ValueError(problem)
        return str(value)

    return check


def _parse_pair(
    where: str, raw: object, consumers: Mapping[str, Path], problems: list[str]
) -> Pair | None:
    shape = {
        "task": _task(consumers),
        "model": _matching(re.compile(r"\S+"), "a non-empty string without whitespace"),
        "evidence": _matching(re.compile(r"/.*", re.DOTALL), "an absolute path string"),
        "sha256": _matching(re.compile(r"[0-9a-f]{64}"), "64 lowercase hex characters"),
    }
    fields = _fields(where, raw, shape, problems)
    if fields is None:
        return None
    return Pair(fields["task"], fields["model"], Path(fields["evidence"]), fields["sha256"])


def _parse_jobs(
    raw: object, consumers: Mapping[str, Path], problems: list[str]
) -> tuple[Pair, ...]:
    if not isinstance(raw, list):
        problems.append("jobs must be a list (use [] for none)")
        return ()
    seen: set[tuple[str, str]] = set()
    pairs: list[Pair] = []
    for index, item in enumerate(raw):
        pair = _parse_pair(f"jobs[{index}]", item, consumers, problems)
        if pair is None:
            continue
        if (pair.task, pair.model) in seen:
            problems.append(
                f"jobs[{index}]: {pair.task!r} with model {pair.model!r} is listed twice"
            )
        else:
            seen.add((pair.task, pair.model))
            pairs.append(pair)
    return tuple(pairs)


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
        "name": _matching(SIGNAL_NAME, f"a name matching {SIGNAL_NAME.pattern}"),
        "unit": _matching(UNIT_NAME, "a systemd unit name (not starting with '-')"),
    }
    kind = ValueSignal if "value" in body or "limit" in body else PatternSignal
    if kind is ValueSignal:
        shape = {**common, "value": _regex(1), "limit": _positive}
    else:
        shape = {**common, "good": _regex(), "bad": _regex()}
    fields = _fields(where, body, shape, problems)
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


def _parse_watch(raw: object, problems: list[str]) -> WatchConfig | None:
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


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses aliases (a few hundred bytes of nested ones expand to gigabytes), merge keys and a mapping key written twice (PyYAML keeps the last)."""

    def compose_node(self, parent: yaml.Node | None, index: Any) -> yaml.Node | None:
        event = self.peek_event()  # type: ignore[no-untyped-call]  # types-PyYAML leaves the parser untyped
        if isinstance(event, yaml.AliasEvent):
            raise ComposerError(None, None, "aliases are not allowed", event.start_mark)
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        if any(key.tag == MERGE_TAG for key, _ in node.value):
            raise ConstructorError(None, None, "merge keys are not allowed", node.start_mark)
        mapping = super().construct_mapping(node, deep=deep)
        keys = [self.constructed_objects[key] for key, _ in node.value]
        if len(keys) != len(mapping):
            dup = next(k for k in keys if keys.count(k) > 1)
            raise ConstructorError(None, None, f"duplicate key {dup!r}", node.start_mark)
        return mapping


# What PyYAML's constructors and the validators' own messages raise on crafted input: nested too deep, an int over
# 4300 digits, a bad explicit tag. None of it may stall the dispatcher's tick or stop the gateway from starting.
_CRAFTED = (RecursionError, ValueError, TypeError, KeyError, AttributeError, IndexError)


def parse_policy(text: str) -> Policy:
    """Validate policy text; raise PolicyError listing every problem found."""
    try:
        return _parse(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"invalid YAML: {exc}") from exc
    except _CRAFTED as exc:
        raise PolicyError(f"invalid policy: {type(exc).__name__}: {exc}") from exc


def _parse(text: str) -> Policy:
    data = yaml.load(text, Loader=_UniqueKeyLoader)
    if not isinstance(data, dict):
        raise PolicyError("policy must be a mapping with version, hosts, sources, consumers, jobs")
    problems = [f"unknown key {k!r}" for k in data if k not in KEYS | OPTIONAL_KEYS]
    problems += [f"missing key {k!r}" for k in sorted(KEYS - set(data))]
    if type(data.get("version")) is not int or data["version"] != VERSION:
        problems.append(f"version must be {VERSION}, got {data.get('version')!r}")
    hosts = _parse_hosts(data.get("hosts"), problems)
    sources = _parse_sources(data.get("sources"), problems)
    consumers = _parse_consumers(data.get("consumers"), problems)
    jobs = _parse_jobs(data.get("jobs"), consumers, problems)
    watch = _parse_watch(data["watch"], problems) if "watch" in data else None
    if problems:
        raise PolicyError("; ".join(problems))
    return Policy(hosts, sources, consumers, jobs, watch)


def load_policy(path: Path) -> Policy:
    """Read and validate the policy file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyError(f"cannot read {path}: {exc}") from exc
    return parse_policy(text)


class PolicyCache:
    """Re-reads the policy when its bytes change and falls back to the last good one on error."""

    def __init__(self, path: Path, log: Callable[[str], None]) -> None:
        self._path = path
        self._log = log
        self._seen: bytes | None = None
        self._good: Policy | None = None
        self._error: PolicyError | None = None

    def current(self) -> Policy:
        """Return the live policy, the last good one if the file is bad, or raise if never good."""
        try:
            data = self._path.read_bytes()
            error = None
        except OSError as exc:
            data, error = f"unreadable: {exc}".encode(), PolicyError(f"cannot read: {exc}")
        if data != self._seen:
            self._seen = data  # before the parse: an unforeseen failure in it must cost one tick, not every tick
            if error is None:
                error = self._try_parse(data)
            self._error = error
            if error is not None:
                self._log(f"policy error in {self._path}: {error}")
        if self._good is None:
            raise self._error or PolicyError("no policy loaded")
        return self._good

    def _try_parse(self, data: bytes) -> PolicyError | None:
        try:
            self._good = parse_policy(data.decode("utf-8"))
        except UnicodeDecodeError as exc:
            return PolicyError(f"not UTF-8: {exc}")
        except PolicyError as exc:
            return exc
        return None


def _quote(value: object) -> str:
    dumped = yaml.safe_dump(str(value), default_style='"', width=sys.maxsize, allow_unicode=False)
    return dumped.removesuffix("\n")


def _key(name: str) -> str:
    return name if yaml.safe_load(name) == name else _quote(name)


def _strings(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_quote(v) for v in values) + "]"


def _yaml_float(value: float) -> str:
    """`repr` of the float, with `.0` before a dotless exponent (`1e-05` -> `1.0e-05`): YAML 1.1 reads `1e-05` as a string."""
    text = repr(float(value))
    return text.replace("e", ".0e") if "e" in text and "." not in text else text


def _host_line(host: Host) -> str:
    fields = (
        f"url: {_quote(host.url)}, vram_gib: {_yaml_float(host.vram_gib)}, max_models: {host.max_models},"
        f" parallel: {host.parallel}, context: {host.context}"
    )
    return f"  {_key(host.name)}: {{{fields}}}"


def _rule_line(rule: SourceRule) -> str:
    fields = f"level: {rule.level.name.lower()}, {rule.kind}: {_strings(rule.values)}"
    if rule.exclude:
        fields += f", exclude: {_strings(rule.exclude)}"
    return f"  - {{{fields}}}"


def _pair_line(pair: Pair) -> str:
    fields = f"task: {_quote(pair.task)}, model: {_quote(pair.model)}, evidence: {_quote(pair.evidence)}, sha256: {_quote(pair.sha256)}"
    return f"  - {{{fields}}}"


def _single(value: str) -> str:
    """A single-quoted scalar, which keeps a regex's backslashes as written; text that is not printable ASCII is double-quoted."""
    if value.isascii() and value.isprintable():
        return "'" + value.replace("'", "''") + "'"
    return _quote(value)


def _signal_line(signal: Signal) -> str:
    if isinstance(signal, ValueSignal):
        match = f"value: {_single(signal.value)}, limit: {signal.limit}"
    else:
        match = f"good: {_single(signal.good)}, bad: {_single(signal.bad)}"
    first_bad = ", first_bad: true" if signal.first_bad else ""
    return f"    - {{name: {_key(signal.name)}, unit: {_quote(signal.unit)}, {match}{first_bad}}}"


def render_policy(policy: Policy) -> str:
    """Canonical policy.yaml text; load_policy of it equals the input."""
    lines = [HEADER.rstrip("\n"), f"version: {VERSION}", "hosts:"]
    lines += [_host_line(h) for h in policy.hosts.values()]
    lines += ["sources:", *(_rule_line(r) for r in policy.sources)]
    if policy.consumers:
        lines.append("consumers:")
        lines += [f"  {_key(n)}: {_quote(p)}" for n, p in policy.consumers.items()]
    else:
        lines.append("consumers: {}")
    if policy.jobs:
        lines += ["jobs:", *(_pair_line(p) for p in policy.jobs)]
    else:
        lines.append("jobs: []")
    if policy.watch is not None:
        lines += ["watch:", f"  status: {_key(policy.watch.status)}", "  signals:"]
        lines += [_signal_line(s) for s in policy.watch.signals]
    return "\n".join(lines) + "\n"


def with_job_added(policy: Policy, pair: Pair) -> Policy:
    """A copy of the policy with the pair appended; PolicyError if invalid, unknown consumer or listed."""
    raw = {
        "task": pair.task,
        "model": pair.model,
        "evidence": str(pair.evidence),
        "sha256": pair.sha256,
    }
    problems: list[str] = []
    _parse_pair("new pair", raw, policy.consumers, problems)
    if any((p.task, p.model) == (pair.task, pair.model) for p in policy.jobs):
        problems.append(f"job {pair.task!r} with model {pair.model!r} is already listed")
    if problems:
        raise PolicyError("; ".join(problems))
    return replace(policy, jobs=(*policy.jobs, pair))


def with_job_removed(policy: Policy, task: str, model: str) -> Policy:
    """A copy of the policy without the pair; PolicyError if it is not listed."""
    kept = tuple(p for p in policy.jobs if (p.task, p.model) != (task, model))
    if len(kept) == len(policy.jobs):
        raise PolicyError(f"job {task!r} with model {model!r} is not listed")
    return replace(policy, jobs=kept)


def policy_diff(old_text: str, new_text: str, path: Path) -> str:
    """Unified diff of a proposed policy edit."""
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (proposed)",
    )
    return "".join(diff)
