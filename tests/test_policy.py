"""Pins policy.yaml v2 parsing (with the optional watch section), canonical rendering, last-good caching and the add/remove edits.

Depends on: pastore.policy, pastore.spec and pytest only; reads the tracked policy.example.yaml, never a live policy.yaml.
"""

import copy
import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from pastore.policy import (
    Policy,
    PolicyCache,
    PolicyError,
    load_policy,
    parse_policy,
    policy_diff,
    render_policy,
    with_job_added,
    with_job_removed,
)
from pastore.spec import Host, Level, Pair, PatternSignal, SourceRule, ValueSignal, WatchConfig

EXAMPLE_POLICY = Path(__file__).resolve().parent.parent / "policy.example.yaml"
BENCH_SHA = hashlib.sha256(b"bench").hexdigest()
GOOD_DATA: dict[str, Any] = {
    "version": 2,
    "hosts": {
        "workstation": {
            "url": "http://127.0.0.1:11435",
            "vram_gib": 24,
            "max_models": 3,
            "parallel": 3,
            "context": 32768,
        },
        "executor": {
            "url": "http://192.0.2.101:11434",
            "vram_gib": 10,
            "max_models": 3,
            "parallel": 3,
            "context": 8192,
        },
    },
    "sources": [
        {"level": "hearth", "cgroup": ["pastore-*.scope"]},
        {"level": "webui", "peer": ["172.17.0.0/16"]},
        {"level": "reins", "peer": ["0.0.0.0/0", "::/0"], "exclude": ["192.0.2.101/32"]},
        {"level": "coding", "ancestor": ["claude", "opencode"]},
        {"level": "corsu", "under": ["/srv/projects"]},
        {"level": "routine", "cgroup": ["*.service", "*.scope"], "exclude": ["session-*.scope"]},
    ],
    "consumers": {"torra": "/srv/torra/gpu-tasks.yaml", "vault": "/srv/vault/gpu-tasks.yaml"},
    "jobs": [
        {
            "task": "torra/bench",
            "model": "qwen3:8b",
            "evidence": "/srv/torra/evidence/bench.json",
            "sha256": BENCH_SHA,
        }
    ],
}
WATCH_DATA: dict[str, Any] = {
    "status": "health",
    "signals": [
        {"name": "health", "unit": "workload-health.service", "good": r"^ok\b", "bad": r"^down\b"},
        {
            "name": "link",
            "unit": "workload.service",
            "good": "link restored",
            "bad": "link lost",
            "first_bad": True,
        },
        {"name": "lag", "unit": "workload.service", "value": r"lag (\d+)s", "limit": 900},
    ],
}
WATCHED_DATA: dict[str, Any] = {**GOOD_DATA, "watch": WATCH_DATA}
WATCH = WatchConfig(
    "health",
    (
        PatternSignal("health", "workload-health.service", r"^ok\b", r"^down\b"),
        PatternSignal("link", "workload.service", "link restored", "link lost", first_bad=True),
        ValueSignal("lag", "workload.service", r"lag (\d+)s", 900),
    ),
)
GOOD = yaml.safe_dump(GOOD_DATA, sort_keys=False)
DELETE = object()
V1_TEXT = """version: 1
window: {start: "01:00", end: "07:00"}
consumers:
  torra: /srv/torra/gpu-tasks.yaml
jobs:
  - torra/bench
"""


def mutated(path: tuple[str | int, ...], value: object, base: dict[str, Any] = GOOD_DATA) -> str:
    """`base` (GOOD_DATA unless given) with the value at `path` replaced (or deleted), as YAML text."""
    data = copy.deepcopy(base)
    target: Any = data
    for step in path[:-1]:
        target = target[step]
    if value is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return yaml.safe_dump(data, sort_keys=False)


def pair(task: str, model: str, name: str = "x") -> Pair:
    return Pair(
        task, model, Path(f"/srv/evidence/{name}.json"), hashlib.sha256(name.encode()).hexdigest()
    )


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(text)
    return path


def test_load_valid_policy(tmp_path: Path) -> None:
    policy = load_policy(write(tmp_path, GOOD))
    assert policy.hosts == {
        "workstation": Host("workstation", "http://127.0.0.1:11435", 24, 3, 3, 32768),
        "executor": Host("executor", "http://192.0.2.101:11434", 10, 3, 3, 8192),
    }
    assert [r.level for r in policy.sources] == [
        Level.HEARTH,
        Level.WEBUI,
        Level.REINS,
        Level.CODING,
        Level.CORSU,
        Level.ROUTINE,
    ]
    assert policy.sources[2] == SourceRule(
        Level.REINS, "peer", ("0.0.0.0/0", "::/0"), ("192.0.2.101/32",)
    )
    assert policy.sources[3] == SourceRule(Level.CODING, "ancestor", ("claude", "opencode"))
    assert policy.consumers == {
        "torra": Path("/srv/torra/gpu-tasks.yaml"),
        "vault": Path("/srv/vault/gpu-tasks.yaml"),
    }
    assert policy.jobs == (
        Pair("torra/bench", "qwen3:8b", Path("/srv/torra/evidence/bench.json"), BENCH_SHA),
    )


def test_values_are_stored_normalized() -> None:
    text = mutated(("hosts", "executor", "url"), "http://Exec1:011434/")
    policy = parse_policy(text)
    assert policy.hosts["executor"].url == "http://Exec1:11434"
    sources = copy.deepcopy(GOOD_DATA["sources"])
    sources[1]["peer"] = ["10.9.0.7", "fd00:0::/8"]
    sources[4]["under"] = ["/srv//projects/", "/"]
    policy = parse_policy(mutated(("sources",), sources))
    assert policy.sources[1].values == ("10.9.0.7/32", "fd00::/8")
    assert policy.sources[4].values == ("/srv/projects", "/")


def test_two_models_of_one_task_are_two_pairs() -> None:
    other = {**GOOD_DATA["jobs"][0], "model": "llama3:70b"}
    policy = parse_policy(mutated(("jobs",), [GOOD_DATA["jobs"][0], other]))
    assert [(p.task, p.model) for p in policy.jobs] == [
        ("torra/bench", "qwen3:8b"),
        ("torra/bench", "llama3:70b"),
    ]


def test_example_policy_loads_and_round_trips() -> None:
    policy = load_policy(EXAMPLE_POLICY)
    assert policy.hosts == {
        "workstation": Host("workstation", "http://127.0.0.1:11435", 20.0, 1, 1, 32768),
        "second": Host("second", "http://127.0.0.1:11436", 10.0, 1, 1, 8192),
    }
    assert policy.sources == (
        SourceRule(Level.HEARTH, "cgroup", ("pastore-*.scope",)),
        SourceRule(Level.WEBUI, "peer", ("172.17.0.0/16",)),
        SourceRule(Level.REINS, "peer", ("0.0.0.0/0", "::/0"), ("192.0.2.101/32",)),
        SourceRule(Level.CODING, "ancestor", ("claude", "opencode")),
        SourceRule(Level.CORSU, "under", ("/srv/projects",)),
        SourceRule(Level.ROUTINE, "cgroup", ("*.service", "*.scope"), ("session-*.scope",)),
    )
    assert set(policy.consumers) == {"example"}
    assert all(pair.task.split("/", 1)[0] in policy.consumers for pair in policy.jobs)
    assert len({(pair.task, pair.model) for pair in policy.jobs}) == len(policy.jobs)
    assert all(re.fullmatch("[0-9a-f]{64}", pair.sha256) for pair in policy.jobs)
    assert policy.watch == WATCH
    rendered = render_policy(policy)
    assert parse_policy(rendered) == policy
    example = re.sub(r" +#.*", "", EXAMPLE_POLICY.read_text().split("\nwatch:\n", 1)[1])
    assert rendered.split("\nwatch:\n", 1)[1] == example


def rich_policy() -> Policy:
    hosts = {
        "workstation": Host("workstation", "http://127.0.0.1:11435", 24.0, 3, 3, 32768),
        "executor": Host("executor", "http://192.0.2.101:11434", 10.0, 3, 3, 8192),
        "off": Host("off", "http://[::1]:11437", 16.0, 1, 4, 16384),
    }
    sources = (
        SourceRule(Level.HEARTH, "cgroup", ("pastore-*.scope",)),
        SourceRule(
            Level.ROUTINE, "cgroup", ("*.service", "a b", 'q"\\b\U0001f600*'), ("session-*.scope",)
        ),
        SourceRule(Level.WEBUI, "peer", ("172.17.0.0/16",)),
        SourceRule(Level.REINS, "peer", ("0.0.0.0/0", "::/0"), ("192.0.2.101/32", "fe80::/10")),
        SourceRule(Level.CODING, "ancestor", ("claude", "opencode")),
        SourceRule(Level.CORSU, "under", ("/srv/projects", "/")),
    )
    consumers = {
        "yes": Path("/srv/y/gpu-tasks.yaml"),
        "torra": Path("/srv/torra/gpu-tasks.yaml"),
        "odd": Path('/srv/\U0001f600 "q" \\/gpu-tasks.yaml'),
    }
    jobs = (
        pair("yes/index", "m:1", "a"),
        pair("torra/bench", "qwen3:8b", "b"),
        pair("torra/bench", "x/y:z", "c"),
        Pair("odd/x", 'm\U0001f600"q"\\\x01#', Path('/srv/e "v" \\\U0001f600\x01.json'), "0" * 64),
    )
    return Policy(hosts, sources, consumers, jobs)


def awkward_watch() -> WatchConfig:
    """Signals whose regexes need every kind of quoting: backslashes, quotes, brackets, non-ASCII, newlines."""
    return WatchConfig(
        "yes",
        (
            PatternSignal("yes", "a@b.service", r"^ok\b \[x\] \d+ '\w'", 'q"\\b'),
            PatternSignal("no", "d.service", "caf\u00e9\U0001f600", "a\nb\tc", first_bad=True),
            ValueSignal("on", "e-f_g.1:h.timer", r"n=(\d+)\s*\\", 7, first_bad=True),
        ),
    )


@pytest.mark.parametrize("variant", ["rich", "no-consumers-no-jobs", "watch", "awkward-watch"])
def test_render_round_trips(variant: str) -> None:
    policy = rich_policy()
    if variant == "no-consumers-no-jobs":
        policy = Policy(policy.hosts, policy.sources, {}, ())
    elif variant == "watch":
        policy = replace(policy, watch=WATCH)
    elif variant == "awkward-watch":
        policy = replace(policy, watch=awkward_watch())
    text = render_policy(policy)
    assert parse_policy(text) == policy
    assert render_policy(parse_policy(text)) == text


@pytest.mark.parametrize(("given", "stored"), [(21.5, 21.5), (5.6, 5.6), (24, 24.0), (24.0, 24.0)])
def test_vram_gib_takes_an_int_or_a_float_and_stores_a_float(given: float, stored: float) -> None:
    got = parse_policy(mutated(("hosts", "executor", "vram_gib"), given)).hosts["executor"].vram_gib
    assert got == stored and type(got) is float


def test_the_retired_key_is_refused_as_unknown_and_vram_gib_as_missing() -> None:
    retired = "vram_" + "gb"  # spelled in two parts so a grep for the old name finds no live use
    data = copy.deepcopy(GOOD_DATA)
    data["hosts"]["executor"][retired] = data["hosts"]["executor"].pop("vram_gib")
    with pytest.raises(PolicyError) as info:
        parse_policy(yaml.safe_dump(data, sort_keys=False))
    message = str(info.value)
    assert f"host 'executor': unknown key '{retired}'" in message
    assert "host 'executor': missing key 'vram_gib'" in message


@pytest.mark.parametrize(
    "vram",
    [
        21.5,
        5.6,
        24.0,
        0.1,
        15.999999999999998,
        1e-05,
        1.5e-07,
        2e-300,
        1e16,
        1.2e22,
        1.7976931348623157e308,
    ],
)
def test_render_round_trips_a_fractional_vram_gib_exactly(vram: float) -> None:
    host = Host("workstation", "http://127.0.0.1:11435", vram, 1, 1, 32768)
    policy = replace(rich_policy(), hosts={"workstation": host})
    text = render_policy(policy)
    assert parse_policy(text).hosts["workstation"].vram_gib == vram
    assert render_policy(parse_policy(text)) == text


def test_render_writes_context_after_parallel_and_keeps_it() -> None:
    policy = rich_policy()
    text = render_policy(policy)
    assert "parallel: 4, context: 16384}" in text and "parallel: 3, context: 32768}" in text
    assert {h.name: h.context for h in parse_policy(text).hosts.values()} == {
        "workstation": 32768,
        "executor": 8192,
        "off": 16384,
    }


def test_missing_context_is_reported_with_every_other_host_problem() -> None:
    data = copy.deepcopy(GOOD_DATA)
    del data["hosts"]["workstation"]["context"]
    data["hosts"]["workstation"]["parallel"] = 0
    data["hosts"]["executor"]["vram_gib"] = 0
    with pytest.raises(PolicyError) as info:
        parse_policy(yaml.safe_dump(data, sort_keys=False))
    message = str(info.value)
    for fragment in (
        "host 'workstation': missing key 'context'",
        "host 'workstation': parallel must be a positive int",
        "host 'executor': vram_gib must be a positive number, got 0",
    ):
        assert fragment in message


def test_render_escapes_unicode_quotes_backslashes_and_controls_as_ascii() -> None:
    policy = rich_policy()
    text = render_policy(policy)
    assert text.isascii() and "\\U0001F600" in text and "\\x01" in text
    assert parse_policy(text).jobs[-1] == policy.jobs[-1]


def test_render_quotes_names_yaml_would_read_as_booleans() -> None:
    text = render_policy(rich_policy())
    assert '  "off": {url: ' in text and '  "yes": "/srv/y/gpu-tasks.yaml"' in text
    assert text.startswith("# The single human gate: only the task+model pairs listed under `jobs`")


LONG_HOST = ".".join(["a" * 63] * 3 + ["a" * 61])
BAD_EDITS: list[tuple[tuple[str | int, ...], object, str]] = [
    (("version",), 1, "version must be 2"),
    (("version",), "2", "version must be 2"),
    (("version",), 2.0, "version must be 2"),
    (("version",), True, "version must be 2"),
    (("window",), {"start": "01:00", "end": "07:00"}, "unknown key 'window'"),
    (("extra",), 1, "unknown key 'extra'"),
    (("hosts",), DELETE, "missing key 'hosts'"),
    (("sources",), DELETE, "missing key 'sources'"),
    (("consumers",), DELETE, "missing key 'consumers'"),
    (("jobs",), DELETE, "missing key 'jobs'"),
    (("hosts",), {}, "hosts must be a non-empty mapping"),
    (("hosts",), ["workstation"], "hosts must be a non-empty mapping"),
    (("hosts",), {"executor": GOOD_DATA["hosts"]["executor"]}, "hosts must include 'workstation'"),
    (("hosts", "Big_Name"), GOOD_DATA["hosts"]["executor"], "name must match"),
    (("hosts", "aB"), GOOD_DATA["hosts"]["executor"], "host 'aB': name must match"),
    (("hosts", "a!"), GOOD_DATA["hosts"]["executor"], "host 'a!': name must match"),
    (("hosts", "executor"), "http://x:1", "host 'executor' must be a mapping"),
    (("hosts", "executor", "gpu"), 1, "host 'executor': unknown key 'gpu'"),
    (("hosts", "executor", "parallel"), DELETE, "host 'executor': missing key 'parallel'"),
    (("hosts", "executor", "url"), "https://192.0.2.101:11434", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://192.0.2.101", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://192.0.2.101:11434/api", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://u:p@192.0.2.101:11434", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://u@192.0.2.101:11434", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://192.0.2.101:11434?x=1", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://192.0.2.101:11434#f", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://192.0.2.101:65536", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://192.0.2.101:70000", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://192.0.2.101:0", "host 'executor': url"),
    (("hosts", "executor", "url"), "http://a b:11434", "host 'executor': url"),
    (("hosts", "executor", "url"), 11434, "host 'executor': url"),
    *[
        (("hosts", "executor", "url"), f"http://{host}:1", "host 'executor': url")
        for host in (
            "[:]",
            "[1:2:3]",
            "[::1",
            ".",
            "-",
            "a..b",
            "a-",
            "-a",
            "exec.",
            "host_1",
            "1.2.3.999",
            "10",
            "1.2.3",
            "a" * 64,
            LONG_HOST + "a",
        )
    ],
    (
        ("hosts", "executor", "vram_gib"),
        0,
        "host 'executor': vram_gib must be a positive number, got 0",
    ),
    (("hosts", "executor", "vram_gib"), -5.6, "vram_gib must be a positive number, got -5.6"),
    (("hosts", "executor", "vram_gib"), "10", "vram_gib must be a positive number, got '10'"),
    (("hosts", "executor", "vram_gib"), True, "vram_gib must be a positive number, got True"),
    (
        ("hosts", "executor", "vram_gib"),
        float("nan"),
        "vram_gib must be a positive number, got nan",
    ),
    (
        ("hosts", "executor", "vram_gib"),
        float("inf"),
        "vram_gib must be a positive number, got inf",
    ),
    (("hosts", "executor", "vram_gib"), 10**400, "vram_gib must be a positive number, got 1000"),
    (("hosts", "executor", "vram_gib"), DELETE, "host 'executor': missing key 'vram_gib'"),
    (("hosts", "executor", "max_models"), -1, "host 'executor': max_models must be a positive int"),
    (("hosts", "executor", "parallel"), 1.5, "host 'executor': parallel must be a positive int"),
    (("hosts", "executor", "context"), DELETE, "host 'executor': missing key 'context'"),
    (("hosts", "executor", "context"), 0, "host 'executor': context must be a positive int"),
    (("hosts", "executor", "context"), -8192, "host 'executor': context must be a positive int"),
    (("hosts", "executor", "context"), True, "host 'executor': context must be a positive int"),
    (("hosts", "executor", "context"), "8192", "host 'executor': context must be a positive int"),
    (("hosts", "executor", "context"), 8192.0, "host 'executor': context must be a positive int"),
    (("sources",), [], "sources must be a non-empty list"),
    (("sources",), "hearth", "sources must be a non-empty list"),
    (("sources", 3), "claude", "sources[3] must be a mapping"),
    (("sources", 3, "colour"), 1, "sources[3]: unknown key 'colour'"),
    (("sources", 1, "level"), "Reins", "sources[1]: level must be one of"),
    (("sources", 1, "level"), "nope", "sources[1]: level must be one of"),
    (("sources", 1, "level"), DELETE, "sources[1]: level must be one of"),
    (("sources", 1, "ancestor"), ["x"], "sources[1]: needs exactly one of"),
    (("sources", 1, "peer"), DELETE, "sources[1]: needs exactly one of"),
    (("sources", 1, "peer"), [], "sources[1].peer must be a non-empty list"),
    (("sources", 1, "peer"), "172.17.0.0/16", "sources[1].peer must be a non-empty list"),
    (("sources", 1, "peer"), [""], "sources[1].peer: '' must be a non-empty string"),
    (("sources", 1, "peer"), [5], "sources[1].peer: 5 must be a non-empty string"),
    (("sources", 1, "peer"), ["not-a-cidr"], "sources[1].peer: 'not-a-cidr'"),
    (("sources", 1, "peer"), ["192.0.2.1/24"], "sources[1].peer: 192.0.2.1/24 has host bits set"),
    (("sources", 2, "exclude"), [], "sources[2].exclude must be a non-empty list"),
    (("sources", 2, "exclude"), ["192.0.2.300/32"], "sources[2].exclude: '192.0.2.300/32'"),
    (("sources", 3, "exclude"), ["x"], "sources[3]: exclude is not allowed with ancestor"),
    (("sources", 4, "exclude"), ["/x"], "sources[4]: exclude is not allowed with under"),
    (
        ("sources", 0, "cgroup"),
        ["a/b.scope"],
        "sources[0].cgroup: 'a/b.scope' must be one cgroup leaf name",
    ),
    (("sources", 5, "exclude"), ["a/b"], "sources[5].exclude: 'a/b' must be one cgroup leaf name"),
    (("sources", 3, "ancestor"), ["a/b"], "sources[3].ancestor: 'a/b' must be a process name"),
    (("sources", 3, "ancestor"), ["x" * 16], "at most 15 bytes"),
    (("sources", 3, "ancestor"), ["\u00e9" * 8], "at most 15 bytes"),
    (
        ("sources", 4, "under"),
        ["srv/projects"],
        "sources[4].under: 'srv/projects' must be an absolute directory",
    ),
    (
        ("sources", 4, "under"),
        ["/srv/../etc"],
        "sources[4].under: '/srv/../etc' must be an absolute directory",
    ),
    (
        ("sources", 4, "under"),
        ["//srv/projects"],
        "sources[4].under: '//srv/projects' must be an absolute",
    ),
    (("sources", 4, "under"), ["//"], "sources[4].under: '//' must be an absolute"),
    (("sources", 0, "level"), "webui", "sources[0] must be level hearth with kind cgroup"),
    (
        ("sources",),
        [*GOOD_DATA["sources"], {"level": "hearth", "cgroup": ["x-*.scope"]}],
        "sources[6]: level hearth is only allowed at sources[0]",
    ),
    (
        ("sources", 3),
        {"level": "hearth", "ancestor": ["claude"]},
        "sources[3]: level hearth is only allowed at sources[0]",
    ),
    (("sources", 2, "level"), "hearth", "sources[2]: level hearth is only allowed at sources[0]"),
    (
        ("sources", 0),
        {"level": "hearth", "peer": ["192.0.2.0/24"]},
        "sources[0] must be level hearth with kind cgroup",
    ),
    (("consumers", "Torra"), "/srv/torra/gpu-tasks.yaml", "consumer name"),
    (("consumers", "torra"), "torra/gpu-tasks.yaml", "absolute"),
    (("consumers", "torra"), "/srv/torra/other.yaml", "gpu-tasks.yaml"),
    (("jobs",), "torra/bench", "jobs must be a list"),
    (("jobs", 0), "torra/bench", "jobs[0] must be a mapping"),
    (("jobs", 0, "extra"), 1, "jobs[0]: unknown key 'extra'"),
    (("jobs", 0, "task"), DELETE, "jobs[0]: missing key 'task'"),
    (
        ("jobs", 0, "task"),
        "nope/bench",
        "jobs[0]: task job 'nope/bench': consumer 'nope' is not listed under consumers",
    ),
    (("jobs", 0, "task"), "torra", "jobs[0]: task job 'torra' must look like <consumer>/<id>"),
    (
        ("jobs", 0, "model"),
        "two words",
        "jobs[0]: model must be a non-empty string without whitespace",
    ),
    (("jobs", 0, "model"), "", "jobs[0]: model must be a non-empty string without whitespace"),
    (("jobs", 0, "model"), 7, "jobs[0]: model must be a non-empty string without whitespace"),
    (
        ("jobs", 0, "evidence"),
        "evidence/bench.json",
        "jobs[0]: evidence must be an absolute path string",
    ),
    (("jobs", 0, "sha256"), BENCH_SHA.upper(), "jobs[0]: sha256 must be 64 lowercase hex"),
    (("jobs", 0, "sha256"), BENCH_SHA[:63], "jobs[0]: sha256 must be 64 lowercase hex"),
    (("jobs", 0, "sha256"), BENCH_SHA + "0", "jobs[0]: sha256 must be 64 lowercase hex"),
    (("jobs", 0, "sha256"), DELETE, "jobs[0]: missing key 'sha256'"),
    (
        ("jobs",),
        # Copies: one object listed twice would dump as an alias, which the loader refuses.
        [copy.deepcopy(GOOD_DATA["jobs"][0]) for _ in range(2)],
        "jobs[1]: 'torra/bench' with model 'qwen3:8b' is listed twice",
    ),
]


@pytest.mark.parametrize(("path", "value", "fragment"), BAD_EDITS)
def test_invalid_policy_is_refused(
    path: tuple[str | int, ...], value: object, fragment: str
) -> None:
    with pytest.raises(PolicyError) as info:
        parse_policy(mutated(path, value))
    assert fragment in str(info.value)


def with_duplicate_rule_level() -> str:
    return GOOD.replace("- level: webui\n", "- level: webui\n  level: reins\n")


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (GOOD + "jobs: []\n", "jobs"),
        (GOOD.replace("  executor:", "  workstation:", 1), "workstation"),
        (with_duplicate_rule_level(), "level"),
    ],
)
def test_duplicate_mapping_key_is_refused(text: str, key: str) -> None:
    assert text != GOOD
    with pytest.raises(PolicyError) as info:
        parse_policy(text)
    assert f"invalid YAML: duplicate key {key!r}" in str(info.value)


def test_boundary_values_are_accepted() -> None:
    data = copy.deepcopy(GOOD_DATA)
    data["hosts"]["executor"]["url"] = "http://Exec-1.lan:65535"
    data["hosts"]["workstation"]["url"] = f"http://{LONG_HOST}:1"
    assert len(LONG_HOST) == 253
    data["hosts"]["extra"] = {**data["hosts"]["executor"], "url": "http://[::1]:1/"}
    data["sources"][3]["ancestor"] = ["x" * 15, "\u00e9" * 7 + "a"]
    policy = parse_policy(yaml.safe_dump(data, sort_keys=False))
    assert policy.hosts["executor"].url == "http://Exec-1.lan:65535"
    assert policy.hosts["workstation"].url == f"http://{LONG_HOST}:1"
    assert policy.hosts["extra"].url == "http://[::1]:1"
    assert policy.sources[3].values == ("x" * 15, "\u00e9" * 7 + "a")


@pytest.mark.parametrize("text", ["y:\n  <<: {k: 1}\n", "y:\n  <<: {k: 1}\n  k: 2\n"])
def test_merge_keys_are_refused_plainly(text: str) -> None:
    with pytest.raises(PolicyError) as info:
        parse_policy(text)
    assert "merge keys are not allowed" in str(info.value)
    assert "duplicate" not in str(info.value)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (V1_TEXT, "version must be 2"),
        (V1_TEXT, "unknown key 'window'"),
        (V1_TEXT, "missing key 'hosts'"),
        ("- just\n- a list\n", "must be a mapping"),
        ("version: [\n", "invalid YAML"),
    ],
)
def test_v1_and_malformed_text_is_refused(text: str, fragment: str) -> None:
    with pytest.raises(PolicyError) as info:
        parse_policy(text)
    assert fragment in str(info.value)


def test_error_lists_every_problem() -> None:
    data = copy.deepcopy(GOOD_DATA)
    data["version"] = 1
    data["hosts"]["executor"]["vram_gib"] = 0
    data["sources"][3]["ancestor"] = ["a/b"]
    data["jobs"][0]["task"] = "nope/x"
    with pytest.raises(PolicyError) as info:
        parse_policy(yaml.safe_dump(data, sort_keys=False))
    message = str(info.value)
    for fragment in ("version must be 2", "vram_gib", "sources[3].ancestor", "nope/x"):
        assert fragment in message


def test_cache_keeps_last_good_on_parse_error(tmp_path: Path) -> None:
    path = write(tmp_path, GOOD)
    logged: list[str] = []
    cache = PolicyCache(path, logged.append)
    good = cache.current()
    path.write_text("version: 9\n")
    assert cache.current() == good and cache.current() == good
    assert len(logged) == 1 and "version" in logged[0]
    path.write_text(V1_TEXT)
    assert cache.current() == good
    assert len(logged) == 2
    path.write_text(mutated(("jobs", 0, "task"), "vault/index"))
    assert cache.current().jobs[0].task == "vault/index"
    path.unlink()
    assert cache.current().jobs[0].task == "vault/index"
    assert len(logged) == 3


def test_cache_raises_when_there_was_never_a_good_policy(tmp_path: Path) -> None:
    path = write(tmp_path, "version: 9\n")
    logged: list[str] = []
    cache = PolicyCache(path, logged.append)
    for _ in range(2):
        with pytest.raises(PolicyError, match="version must be 2"):
            cache.current()
    assert len(logged) == 1
    with pytest.raises(PolicyError, match="cannot read"):
        PolicyCache(tmp_path / "missing.yaml", logged.append).current()


def test_add_and_remove_edit_the_pair_list() -> None:
    policy = parse_policy(GOOD)
    vault = pair("vault/index", "m1", "v")
    added = with_job_added(policy, vault)
    assert added.jobs == (*policy.jobs, vault)
    assert parse_policy(render_policy(added)) == added
    assert added.hosts is policy.hosts and added.sources is policy.sources
    both = with_job_added(added, pair("vault/index", "m2", "w"))
    assert [(p.task, p.model) for p in both.jobs] == [
        ("torra/bench", "qwen3:8b"),
        ("vault/index", "m1"),
        ("vault/index", "m2"),
    ]
    assert [p.model for p in with_job_removed(both, "vault/index", "m1").jobs] == ["qwen3:8b", "m2"]
    with pytest.raises(PolicyError, match="already listed"):
        with_job_added(policy, pair("torra/bench", "qwen3:8b", "again"))
    with pytest.raises(PolicyError, match="not listed under consumers"):
        with_job_added(policy, pair("ghost/x", "m"))
    with pytest.raises(PolicyError, match="must look like"):
        with_job_added(policy, pair("torra", "m"))
    with pytest.raises(PolicyError, match="not listed"):
        with_job_removed(policy, "vault/index", "m1")
    with pytest.raises(PolicyError, match="not listed"):
        with_job_removed(policy, "torra/bench", "other-model")


VAULT = pair("vault/index", "m1", "v")
BAD_PAIRS = [
    (replace(VAULT, model="two words"), "model must be a non-empty string without whitespace"),
    (replace(VAULT, model=""), "model must be a non-empty string without whitespace"),
    (replace(VAULT, evidence=Path("rel/e.json")), "evidence must be an absolute path string"),
    (replace(VAULT, sha256="ABC"), "sha256 must be 64 lowercase hex characters"),
    (replace(VAULT, sha256="0" * 65), "sha256 must be 64 lowercase hex characters"),
    (
        replace(VAULT, task="ghost/x"),
        "task job 'ghost/x': consumer 'ghost' is not listed under consumers",
    ),
]


@pytest.mark.parametrize(("bad", "fragment"), BAD_PAIRS)
def test_add_refuses_a_pair_the_parser_would_refuse(bad: Pair, fragment: str) -> None:
    with pytest.raises(PolicyError) as info:
        with_job_added(parse_policy(GOOD), bad)
    assert fragment in str(info.value)


def test_add_reports_every_problem_of_a_bad_pair() -> None:
    bad = Pair("vault/index", "a b", Path("rel"), "ABC")
    with pytest.raises(PolicyError) as info:
        with_job_added(parse_policy(GOOD), bad)
    for fragment in ("model", "evidence", "sha256"):
        assert fragment in str(info.value)


def test_policy_diff_shows_the_added_pair() -> None:
    policy = parse_policy(GOOD)
    old = render_policy(policy)
    added = with_job_added(policy, pair("vault/index", "m1", "v"))
    diff = policy_diff(old, render_policy(added), Path("policy.yaml"))
    added_lines = [
        line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    assert len(added_lines) == 1 and "vault/index" in added_lines[0] and "--- " in diff


# --- the optional watch section ---------------------------------------------------------------


def test_a_policy_without_a_watch_section_has_none_and_renders_none() -> None:
    policy = parse_policy(GOOD)
    assert policy.watch is None and "watch" not in render_policy(policy)


def test_the_watch_section_parses_into_typed_signals() -> None:
    policy = parse_policy(yaml.safe_dump(WATCHED_DATA, sort_keys=False))
    assert policy.watch == WATCH
    assert policy.watch.units == ("workload-health.service", "workload.service")
    assert [type(s) for s in policy.watch.signals] == [PatternSignal, PatternSignal, ValueSignal]
    assert [s.first_bad for s in policy.watch.signals] == [False, True, False]


def test_the_watch_section_renders_regexes_in_single_quotes_with_backslashes_intact() -> None:
    text = render_policy(replace(parse_policy(GOOD), watch=WATCH))
    assert (
        "    - {name: health, unit: \"workload-health.service\", good: '^ok\\b', bad: '^down\\b'}"
        in text
    )
    assert "value: 'lag (\\d+)s', limit: 900}" in text
    assert "good: 'link restored', bad: 'link lost', first_bad: true}" in text
    text = render_policy(replace(parse_policy(GOOD), watch=awkward_watch()))
    assert r"""good: '^ok\b \[x\] \d+ ''\w''', bad: 'q"\b'}""" in text
    assert text.isascii() and r"""good: "caf\xE9\U0001F600",""" in text
    assert r'bad: "a\nb\tc", first_bad: true}' in text
    assert r"value: 'n=(\d+)\s*\\', limit: 7, first_bad: true" in text


def test_add_and_remove_keep_the_watch() -> None:
    policy = replace(parse_policy(GOOD), watch=WATCH)
    added = with_job_added(policy, VAULT)
    assert added.watch == WATCH and parse_policy(render_policy(added)) == added
    assert with_job_removed(added, VAULT.task, VAULT.model).watch == WATCH
    diff = policy_diff(render_policy(policy), render_policy(added), Path("policy.yaml"))
    changed = [x for x in diff.splitlines() if x[:1] in "+-" and not x.startswith(("+++", "---"))]
    assert len(changed) == 1 and "vault/index" in changed[0]


def signals(index: int, key: str) -> tuple[str | int, ...]:
    """The path to a key of the index-th watch signal."""
    return ("watch", "signals", index, key)


TOO_MANY = [{**WATCH_DATA["signals"][0], "name": f"s{i}"} for i in range(9)]
BAD_WATCH_EDITS: list[tuple[tuple[str | int, ...], object, str]] = [
    (("watch",), None, "watch must be a mapping with signals, status"),
    (("watch",), ["health"], "watch must be a mapping with signals, status"),
    (("watch", "extra"), 1, "watch: unknown key 'extra'"),
    (("watch", "status"), DELETE, "watch: missing key 'status'"),
    (("watch", "signals"), DELETE, "watch: missing key 'signals'"),
    (("watch", "status"), "nope", "watch: status 'nope' must name a signal with good and bad"),
    (("watch", "status"), "lag", "watch: status 'lag' must name a signal with good and bad"),
    (("watch", "status"), 5, "watch: status 5 must name a signal with good and bad"),
    (("watch", "status"), ["health"], "watch: status ['health'] must name a signal"),
    (("watch", "signals"), [], "watch.signals must be a list of 1 to 8 signals"),
    (("watch", "signals"), "health", "watch.signals must be a list of 1 to 8 signals"),
    (("watch", "signals"), TOO_MANY, "watch.signals must be a list of 1 to 8 signals"),
    (("watch", "signals", 1), "link", "watch.signals[1] must be a mapping"),
    (signals(0, "colour"), 1, "watch.signals[0]: unknown key 'colour'"),
    (signals(0, "name"), DELETE, "watch.signals[0]: missing key 'name'"),
    (signals(0, "unit"), DELETE, "watch.signals[0]: missing key 'unit'"),
    (signals(0, "good"), DELETE, "watch.signals[0]: missing key 'good'"),
    (signals(0, "bad"), DELETE, "watch.signals[0]: missing key 'bad'"),
    (signals(2, "limit"), DELETE, "watch.signals[2]: missing key 'limit'"),
    (signals(2, "value"), DELETE, "watch.signals[2]: missing key 'value'"),
    (signals(2, "good"), "x", "watch.signals[2]: unknown key 'good'"),
    (signals(0, "value"), r"(\d+)", "watch.signals[0]: unknown key 'good'"),
    *[
        (signals(0, "name"), name, "watch.signals[0]: name must be a name matching")
        for name in ("Health", "1x", "", "a b", "a_b", "-a", "a" * 33, 7)
    ],
    (signals(1, "name"), "health", "watch.signals: name 'health' is used twice"),
    *[
        (signals(0, "unit"), unit, "watch.signals[0]: unit must be a systemd unit name")
        for unit in ("-x.service", "a b.service", "a/b.service", "", "a\nb", "x\x00", "a*", 7)
    ],
    *[
        (signals(0, key), regex, f"watch.signals[0]: {key} must be a non-empty regex string")
        for key in ("good", "bad")
        for regex in ("", 5, None, ["a"])
    ],
    *[
        (signals(0, "bad"), regex, "watch.signals[0]: bad is not a valid regex")
        for regex in ("[", "(", "a{2,1}", "a{99999999999}", "(" * 1000 + ")" * 1000, "*a")
    ],
    (signals(2, "value"), r"lag \d+s", "value must have exactly 1 capture group, got 0"),
    (signals(2, "value"), r"(lag) (\d+)s", "value must have exactly 1 capture group, got 2"),
    (signals(2, "value"), r"((\d+))", "value must have exactly 1 capture group, got 2"),
    (signals(2, "value"), "(", "watch.signals[2]: value is not a valid regex"),
    *[
        (signals(2, "limit"), limit, "watch.signals[2]: limit must be a positive int")
        for limit in (0, -1, "900", True, 1.5, None)
    ],
    *[
        (signals(1, "first_bad"), flag, "watch.signals[1]: first_bad must be true or false")
        for flag in ("yes", 1, "true", None)
    ],
]


@pytest.mark.parametrize(("path", "value", "fragment"), BAD_WATCH_EDITS)
def test_an_invalid_watch_section_is_refused(
    path: tuple[str | int, ...], value: object, fragment: str
) -> None:
    with pytest.raises(PolicyError) as info:
        parse_policy(mutated(path, value, WATCHED_DATA))
    assert fragment in str(info.value)


def test_a_bad_watch_section_is_reported_with_every_other_problem() -> None:
    data = copy.deepcopy(WATCHED_DATA)
    data["version"] = 1
    data["watch"]["signals"][0]["good"] = "["
    data["watch"]["signals"][2]["limit"] = 0
    with pytest.raises(PolicyError) as info:
        parse_policy(yaml.safe_dump(data, sort_keys=False))
    message = str(info.value)
    for fragment in ("version must be 2", "signals[0]: good is not a valid regex", "limit"):
        assert fragment in message
    assert "status" not in message


def test_an_unknown_top_level_key_is_still_refused_next_to_a_watch_section() -> None:
    data = {**WATCHED_DATA, "wach": copy.deepcopy(WATCH_DATA)}
    with pytest.raises(PolicyError, match="unknown key 'wach'"):
        parse_policy(yaml.safe_dump(data, sort_keys=False))


def test_eight_signals_are_the_most_a_watch_takes() -> None:
    eight = TOO_MANY[:8]
    text = mutated(("watch", "signals"), eight, WATCHED_DATA).replace(
        "status: health", "status: s0"
    )
    policy = parse_policy(text)
    assert policy.watch is not None and len(policy.watch.signals) == 8
