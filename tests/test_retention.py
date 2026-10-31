"""Pins run-folder retention (D-050 (7)): old folders go except each pair's latest, run rows stay.

Depends on: pastore.runner, pastore.state and pytest only; fixed datetimes, no real clock.
"""

import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

import pastore.runner as runner_module
from pastore.runner import RUN_DIR_MAX_AGE, new_run_id, prune_run_dirs
from pastore.state import RunRow, default_paths, open_state

NOW = datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC)
HOST = "workstation"


def run_folder(runs_dir: Path, started: datetime, seq: int = 1, suffix: str = "") -> str:
    """Create a run folder named for `started` with a file inside; return its name."""
    name = new_run_id(started, seq) + suffix
    (runs_dir / name / "out").mkdir(parents=True)
    (runs_dir / name / "out" / "result.json").write_text("{}")
    return name


def row(run_id: str, ref: str, model: str, started: datetime) -> RunRow:
    return RunRow(run_id, ref, model, HOST, "fp", "done", started.isoformat())


def test_an_old_run_folder_that_is_not_a_latest_is_deleted(tmp_path: Path) -> None:
    old = run_folder(tmp_path, NOW - timedelta(days=45))
    assert prune_run_dirs(tmp_path, set(), NOW) == [old]
    assert not (tmp_path / old).exists()


def test_an_old_run_folder_that_is_its_pairs_latest_stays(tmp_path: Path) -> None:
    latest = run_folder(tmp_path, NOW - timedelta(days=90))
    other = run_folder(tmp_path, NOW - timedelta(days=91))
    assert prune_run_dirs(tmp_path, {latest}, NOW) == [other]
    assert (tmp_path / latest / "out" / "result.json").is_file()


def test_a_young_run_folder_stays(tmp_path: Path) -> None:
    young = run_folder(tmp_path, NOW - timedelta(days=29, hours=23))
    assert prune_run_dirs(tmp_path, set(), NOW) == []
    assert (tmp_path / young).is_dir()


def test_exactly_thirty_days_stays_and_one_second_more_goes(tmp_path: Path) -> None:
    assert RUN_DIR_MAX_AGE == timedelta(days=30)
    edge = run_folder(tmp_path, NOW - timedelta(days=30))
    over = run_folder(tmp_path, NOW - timedelta(days=30, seconds=1))
    assert prune_run_dirs(tmp_path, set(), NOW) == [over]
    assert (tmp_path / edge).is_dir() and not (tmp_path / over).exists()


def test_old_contract_folders_without_a_row_are_deleted(tmp_path: Path) -> None:
    old = NOW - timedelta(days=40)
    names = [
        run_folder(tmp_path, old, 1, f"-contract-{kind}") for kind in ("503", "hang", "refused")
    ]
    assert prune_run_dirs(tmp_path, set(), NOW) == sorted(names)
    assert list(tmp_path.iterdir()) == []


def test_the_stamp_is_read_as_utc_whatever_zone_now_carries(tmp_path: Path) -> None:
    over = run_folder(tmp_path, NOW - timedelta(days=30, hours=1), 1)
    kept = run_folder(tmp_path, NOW - timedelta(days=29, hours=23), 2)
    now_west = NOW.astimezone(timezone(timedelta(hours=-5)))
    assert prune_run_dirs(tmp_path, set(), now_west) == [over]
    assert (tmp_path / kept).is_dir()


@pytest.fixture
def local_zone() -> Iterator[Callable[[str], None]]:
    """Switch the process time zone for a test and restore it, with tzset on both sides."""
    saved = os.environ.get("TZ")

    def switch(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield switch
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()


@pytest.mark.parametrize("zone", ["Asia/Kolkata", "America/Los_Angeles", "UTC"])
def test_the_stamp_is_read_as_utc_whatever_the_local_zone(
    tmp_path: Path, local_zone: Callable[[str], None], zone: str
) -> None:
    local_zone(zone)
    over = run_folder(tmp_path, NOW - timedelta(days=30, hours=1), 1)
    kept = run_folder(tmp_path, NOW - timedelta(days=29, hours=23), 2)
    assert prune_run_dirs(tmp_path, set(), NOW) == [over]
    assert (tmp_path / kept).is_dir()


def test_a_symlink_child_is_never_followed_or_removed(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    target = run_folder(tmp_path, NOW - timedelta(days=60))
    link = runs / new_run_id(NOW - timedelta(days=60), 2)
    link.symlink_to(tmp_path / target, target_is_directory=True)
    assert prune_run_dirs(runs, set(), NOW) == []
    assert link.is_symlink() and (tmp_path / target / "out" / "result.json").is_file()


def test_a_symlink_inside_an_old_folder_does_not_delete_its_target(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    keepsake = tmp_path / "elsewhere"
    keepsake.mkdir()
    (keepsake / "data.txt").write_text("x")
    old = run_folder(runs, NOW - timedelta(days=60))
    (runs / old / "out" / "link").symlink_to(keepsake, target_is_directory=True)
    assert prune_run_dirs(runs, set(), NOW) == [old]
    assert (keepsake / "data.txt").is_file()


def test_rmtree_is_never_called_on_a_symlink_or_a_plain_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    target = run_folder(tmp_path, NOW - timedelta(days=60))
    runs.mkdir()
    (runs / new_run_id(NOW - timedelta(days=60), 2)).symlink_to(
        tmp_path / target, target_is_directory=True
    )
    (runs / new_run_id(NOW - timedelta(days=60), 3)).write_text("a file named like a run")
    old = run_folder(runs, NOW - timedelta(days=60), 4)
    called: list[str] = []
    monkeypatch.setattr(runner_module.shutil, "rmtree", lambda path: called.append(Path(path).name))
    assert prune_run_dirs(runs, set(), NOW) == [old]
    assert called == [old]


def test_plain_files_and_unparseable_names_are_left(tmp_path: Path) -> None:
    (tmp_path / "20200101T000000-1").write_text("a file named like a run")
    for name in ("notes", "20201301T000000-1", "20200101T000000", "x20200101T000000-1"):
        (tmp_path / name).mkdir()
    before = sorted(p.name for p in tmp_path.iterdir())
    assert prune_run_dirs(tmp_path, set(), NOW) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_missing_runs_dir_returns_nothing(tmp_path: Path) -> None:
    assert prune_run_dirs(tmp_path / "absent", set(), NOW) == []


def test_a_failing_rmtree_leaves_that_folder_and_the_rest_are_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stuck = run_folder(tmp_path, NOW - timedelta(days=50), 1)
    gone = run_folder(tmp_path, NOW - timedelta(days=50), 2)
    real_rmtree = runner_module.shutil.rmtree

    def rmtree(path: Path) -> None:
        if Path(path).name == stuck:
            raise PermissionError("denied")
        real_rmtree(path)

    monkeypatch.setattr(runner_module.shutil, "rmtree", rmtree)
    assert prune_run_dirs(tmp_path, set(), NOW) == [gone]
    assert (tmp_path / stuck).is_dir() and not (tmp_path / gone).exists()


def test_latest_run_ids_is_the_latest_per_pair_with_the_larger_id_breaking_ties(
    tmp_path: Path,
) -> None:
    state = open_state(default_paths(tmp_path))
    t = NOW - timedelta(days=5)
    for r in (
        row("a1", "torra/bench", "m1", t),
        row("a2", "torra/bench", "m1", t + timedelta(hours=1)),
        row("b1", "torra/bench", "m2", t),
        row("c1", "memory/index", "m1", t),
        row("c2", "memory/index", "m1", t),
        row("d1", "memory/index", "m1", t - timedelta(days=1)),
    ):
        state.insert_run(r)
    assert state.latest_run_ids() == {"a2", "b1", "c2"}
    state.close()


def test_latest_run_ids_stays_fast_and_matches_last_run_on_many_rows_with_ties(
    tmp_path: Path,
) -> None:
    paths = default_paths(tmp_path)
    state = open_state(paths)
    t = NOW - timedelta(days=200)
    rows = [
        (
            f"r{i:05d}",
            f"ref/{i % 40}",
            f"m{i % 7}",
            HOST,
            "fp",
            "done",
            (t + timedelta(hours=i % 13)).isoformat(),
            None,
            None,
            "",
        )
        for i in range(20_000)
    ]
    with sqlite3.connect(paths.db) as conn:
        conn.executemany("INSERT INTO runs VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.close()
    start = time.perf_counter()
    got = state.latest_run_ids()
    assert time.perf_counter() - start < 2
    pairs = {(r[1], r[2]) for r in rows}
    last = (state.last_run(ref, model) for ref, model in pairs)
    assert len(pairs) == 280 and got == {r.run_id for r in last if r is not None}
    state.close()


def test_latest_run_ids_of_an_empty_table_is_empty(tmp_path: Path) -> None:
    state = open_state(default_paths(tmp_path))
    assert state.latest_run_ids() == set()
    state.close()


def test_pruning_leaves_every_run_row(tmp_path: Path) -> None:
    state = open_state(default_paths(tmp_path))
    runs = tmp_path / "runs"
    for seq, age in enumerate((90, 60, 45), start=1):
        started = NOW - timedelta(days=age)
        state.insert_run(row(run_folder(runs, started, seq), "torra/bench", "m1", started))
    latest = new_run_id(NOW - timedelta(days=45), 3)
    assert state.latest_run_ids() == {latest}
    assert len(prune_run_dirs(runs, state.latest_run_ids(), NOW)) == 2
    assert [p.name for p in runs.iterdir()] == [latest]
    assert len(state.runs_with_status("done")) == 3
    state.close()
