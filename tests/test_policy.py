"""Pins policy.yaml parsing, canonical rendering, last-good caching and the add/remove edits.

Depends on: pastore.policy and pytest only.
"""

from datetime import time
from pathlib import Path

import pytest

from pastore.policy import (
    Policy,
    PolicyCache,
    PolicyError,
    load_policy,
    policy_diff,
    render_policy,
    with_job_added,
    with_job_removed,
)

REPO_POLICY = Path(__file__).resolve().parent.parent / "policy.yaml"
GOOD = """version: 1
window: {start: "22:30", end: "06:00"}
consumers:
  torra: /srv/torra/gpu-tasks.yaml
  vault: /srv/vault/gpu-tasks.yaml
jobs:
  - torra/bench
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(text)
    return path


def test_load_valid_policy(tmp_path: Path) -> None:
    policy = load_policy(write(tmp_path, GOOD))
    assert policy.window_start == time(22, 30) and policy.window_end == time(6, 0)
    assert policy.consumers == {
        "torra": Path("/srv/torra/gpu-tasks.yaml"),
        "vault": Path("/srv/vault/gpu-tasks.yaml"),
    }
    assert policy.jobs == ("torra/bench",)


def test_repo_policy_loads_and_render_keeps_its_header() -> None:
    policy = load_policy(REPO_POLICY)
    assert all(job.split("/")[0] in policy.consumers for job in policy.jobs)
    header = "".join(REPO_POLICY.read_text().splitlines(keepends=True)[:2])
    assert render_policy(policy).startswith(header)


@pytest.mark.parametrize("variant", ["good", "empty"])
def test_render_round_trips(tmp_path: Path, variant: str) -> None:
    policy = load_policy(write(tmp_path, GOOD))
    if variant == "empty":
        policy = Policy(time(1), time(7), {}, ())
    again = load_policy(write(tmp_path, render_policy(policy)))
    assert again == policy


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (GOOD.replace("version: 1", "version: 2"), "version must be 1"),
        (GOOD + "extra: 1\n", "unknown key 'extra'"),
        (GOOD.replace("jobs:\n  - torra/bench\n", ""), "missing key 'jobs'"),
        (GOOD.replace('"22:30"', "22:30"), "window.start"),
        (GOOD.replace('"06:00"', '"22:30"'), "must differ"),
        (GOOD.replace('"06:00"', '"24:00"'), "window.end"),
        (GOOD.replace("torra:", "Torra:"), "consumer name"),
        (GOOD.replace("/srv/torra/gpu-tasks.yaml", "torra/gpu-tasks.yaml"), "absolute"),
        (GOOD.replace("/srv/torra/gpu-tasks.yaml", "/srv/torra/other.yaml"), "gpu-tasks.yaml"),
        (GOOD.replace("torra/bench", "nope/bench"), "not listed under consumers"),
        (GOOD.replace("torra/bench", "torra"), "<consumer>/<id>"),
        (GOOD + "  - torra/bench\n", "listed twice"),
        ("- just\n- a list\n", "must be a mapping"),
        ("version: [\n", "invalid YAML"),
    ],
)
def test_invalid_policy_is_refused(tmp_path: Path, text: str, fragment: str) -> None:
    with pytest.raises(PolicyError, match=fragment):
        load_policy(write(tmp_path, text))


def test_error_lists_every_problem(tmp_path: Path) -> None:
    text = GOOD.replace("version: 1", "version: 2").replace("torra/bench", "nope/x")
    with pytest.raises(PolicyError) as info:
        load_policy(write(tmp_path, text))
    assert "version must be 1" in str(info.value) and "nope/x" in str(info.value)


def test_cache_keeps_last_good_on_parse_error(tmp_path: Path) -> None:
    path = write(tmp_path, GOOD)
    logged: list[str] = []
    cache = PolicyCache(path, logged.append)
    good = cache.current()
    path.write_text("version: 9\n")
    assert cache.current() == good and cache.current() == good
    assert len(logged) == 1 and "version" in logged[0]
    path.write_text("version: 8\n")
    assert cache.current() == good
    assert len(logged) == 2
    path.write_text(GOOD.replace("torra/bench", "vault/index"))
    assert cache.current().jobs == ("vault/index",)
    path.unlink()
    assert cache.current().jobs == ("vault/index",)
    assert len(logged) == 3


def test_cache_raises_when_there_was_never_a_good_policy(tmp_path: Path) -> None:
    path = write(tmp_path, "version: 9\n")
    logged: list[str] = []
    cache = PolicyCache(path, logged.append)
    for _ in range(2):
        with pytest.raises(PolicyError, match="version must be 1"):
            cache.current()
    assert len(logged) == 1
    with pytest.raises(PolicyError, match="cannot read"):
        PolicyCache(tmp_path / "missing.yaml", logged.append).current()


def test_add_and_remove_edit_the_job_list(tmp_path: Path) -> None:
    policy = load_policy(write(tmp_path, GOOD))
    added = with_job_added(policy, "vault/index")
    assert added.jobs == ("torra/bench", "vault/index")
    assert with_job_removed(added, "torra/bench").jobs == ("vault/index",)
    with pytest.raises(PolicyError, match="already listed"):
        with_job_added(policy, "torra/bench")
    with pytest.raises(PolicyError, match="not listed under consumers"):
        with_job_added(policy, "ghost/x")
    with pytest.raises(PolicyError, match="not listed"):
        with_job_removed(policy, "vault/index")


def test_policy_diff_shows_the_added_job(tmp_path: Path) -> None:
    policy = load_policy(write(tmp_path, GOOD))
    old = render_policy(policy)
    diff = policy_diff(old, render_policy(with_job_added(policy, "vault/index")), tmp_path / "p")
    assert "+  - vault/index" in diff and "--- " in diff
