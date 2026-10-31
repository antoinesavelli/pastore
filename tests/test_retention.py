"""Pins run-folder retention (D-050 (7), D-052 (5)): old folders go except each pair's latest and every run awaiting review, run rows stay.

Depends on: pastore.runner, pastore.state, pastore.dispatcher (the daily prune in a tick, over the dispatcher test Rig)
and pytest only; fixed datetimes, no real clock.
"""

import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

import pastore.dispatcher as dispatcher_module
import pastore.runner as runner_module
from pastore.dispatcher import PRUNE_EVERY, _guarded_tick
from pastore.runner import RUN_DIR_MAX_AGE, new_run_id, prune_run_dirs
from pastore.state import RunRow, default_paths, open_state
from pastore.tick_bound import TickOverrun
from tests.test_dispatcher import MODEL, Rig, task

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


def seed(rig: Rig, ref: str, age: timedelta, seq: int = 1, row_too: bool = True) -> str:
    """An old run folder under the rig's runs dir, with a `runs` row unless `row_too` is false (a contract folder)."""
    started = rig.clock() - age
    name = run_folder(rig.paths.runs_dir, started, seq)
    if row_too:
        rig.state.insert_run(row(name, ref, MODEL, started))
    return name


def idle_rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path, [task(tmp_path, "a")], [])


@pytest.fixture
def prune_calls(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    """Spy on the dispatcher's prune: the `now` of every call, the real function still runs."""
    calls: list[datetime] = []
    real = dispatcher_module.prune_run_dirs

    def spy(runs_dir: Path, keep: set[str], now: datetime) -> list[str]:
        calls.append(now)
        return real(runs_dir, keep, now)

    monkeypatch.setattr(dispatcher_module, "prune_run_dirs", spy)
    return calls


def test_the_first_tick_prunes_an_old_non_latest_folder_and_keeps_each_pairs_latest(
    tmp_path: Path, prune_calls: list[datetime]
) -> None:
    rig = idle_rig(tmp_path)
    old = seed(rig, "c/a", timedelta(days=90))
    latest = seed(rig, "c/a", timedelta(days=45))
    other = seed(rig, "c/b", timedelta(days=60))
    contract = new_run_id(rig.clock() - timedelta(days=50), 1) + "-contract-503"
    (rig.paths.runs_dir / contract).mkdir()
    young = seed(rig, "c/c", timedelta(days=5))
    rig.tick()
    assert sorted(p.name for p in rig.paths.runs_dir.iterdir()) == sorted([latest, other, young])
    assert len(prune_calls) == 1
    (event,) = rig.events("runs-pruned")
    assert event.detail == f"2 {old} {contract}" and event.ref is None
    assert len(rig.state.runs_with_status("done")) == 4


def test_an_awaiting_review_folder_is_never_pruned(tmp_path: Path) -> None:
    rig = idle_rig(tmp_path)
    reviewed = seed(rig, "c/a", timedelta(days=90), seq=1)
    plain = seed(rig, "c/a", timedelta(days=80), seq=2)
    latest = seed(rig, "c/a", timedelta(days=5), seq=3)
    run = rig.state.get_run(reviewed)
    assert run is not None
    rig.state.update_run(replace(run, status="awaiting-review"))
    rig.tick()
    assert sorted(p.name for p in rig.paths.runs_dir.iterdir()) == sorted([reviewed, latest])
    assert (rig.paths.runs_dir / reviewed / "out" / "result.json").is_file()
    assert [e.detail for e in rig.events("runs-pruned")] == [f"1 {plain} {plain}"]
    rig.state.ack(reviewed, rig.clock())
    rig.clock.advance(PRUNE_EVERY.total_seconds())
    rig.tick()
    assert [p.name for p in rig.paths.runs_dir.iterdir()] == [latest]
    assert rig.events("runs-pruned")[0].detail == f"1 {reviewed} {reviewed}"  # newest first
    assert len(rig.state.runs_with_status("done")) == 3


def test_a_second_tick_inside_24_hours_does_not_prune_and_one_24_hours_later_does(
    tmp_path: Path, prune_calls: list[datetime]
) -> None:
    rig = idle_rig(tmp_path)
    assert PRUNE_EVERY == timedelta(days=1)
    rig.tick()
    first = rig.clock()
    stale = seed(rig, "c/a", timedelta(days=60), row_too=False)
    rig.clock.advance(30)
    rig.tick()
    rig.clock.now_value = first + PRUNE_EVERY - timedelta(seconds=1)
    rig.tick()
    assert prune_calls == [first] and (rig.paths.runs_dir / stale).is_dir()
    rig.clock.now_value = first + PRUNE_EVERY
    rig.tick()
    assert prune_calls == [first, first + PRUNE_EVERY]
    assert not (rig.paths.runs_dir / stale).exists()


def test_runs_pruned_is_recorded_only_when_something_was_deleted(
    tmp_path: Path, prune_calls: list[datetime]
) -> None:
    rig = idle_rig(tmp_path)
    seed(rig, "c/a", timedelta(days=5))
    rig.tick()
    assert len(prune_calls) == 1 and rig.events("runs-pruned") == []
    seed(rig, "c/a", timedelta(days=60), seq=2, row_too=False)
    rig.clock.advance(PRUNE_EVERY.total_seconds())
    rig.tick()
    assert len(prune_calls) == 2 and len(rig.events("runs-pruned")) == 1


def test_a_running_jobs_folder_is_never_deleted(
    tmp_path: Path, prune_calls: list[datetime]
) -> None:
    spec = task(tmp_path, "a", timeout_s=200 * 86400)
    rig = Rig(tmp_path, [spec], ["c/a"])
    rig.next_codes = [None]
    rig.tick()
    (run_id,) = rig.d.active
    stale = seed(rig, "c/a", timedelta(days=90), seq=2, row_too=False)
    rig.clock.advance(timedelta(days=40).total_seconds())
    rig.tick()
    assert len(prune_calls) == 2 and not (rig.paths.runs_dir / stale).exists()
    assert (rig.paths.runs_dir / run_id).is_dir() and list(rig.d.active) == [run_id]
    assert rig.events("runs-pruned")[-1].detail.startswith("1 ")


def test_a_prune_that_raises_is_one_tick_error_and_the_next_attempt_waits_a_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = idle_rig(tmp_path)
    calls: list[datetime] = []

    def broken(runs_dir: Path, keep: set[str], now: datetime) -> list[str]:
        calls.append(now)
        raise PermissionError("denied")

    monkeypatch.setattr(dispatcher_module, "prune_run_dirs", broken)
    for _ in range(3):
        assert _guarded_tick(rig.d, rig.state, rig.clock)
        rig.clock.advance(30)
    assert len(calls) == 1
    assert [e.detail for e in rig.events("tick-error")] == ["PermissionError: denied"]
    rig.clock.now_value = calls[0] + PRUNE_EVERY
    assert _guarded_tick(rig.d, rig.state, rig.clock)
    assert len(calls) == 2 and len(rig.events("tick-error")) == 2


def test_the_prune_runs_after_the_workload_watch_outside_any_critical_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = idle_rig(tmp_path)
    order: list[str] = []
    depths: list[int] = []
    real_watch = rig.d._watch_workload

    def watch(now: datetime) -> None:
        order.append("watch")
        real_watch(now)

    def spy(runs_dir: Path, keep: set[str], now: datetime) -> list[str]:
        order.append("prune")
        depths.append(rig.d._depth)
        return []

    monkeypatch.setattr(rig.d, "_watch_workload", watch)
    monkeypatch.setattr(dispatcher_module, "prune_run_dirs", spy)
    rig.tick()
    assert order == ["watch", "prune"] and depths == [0]


def test_an_overrun_during_the_prune_ends_the_tick_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = idle_rig(tmp_path)

    def overrun(runs_dir: Path, keep: set[str], now: datetime) -> list[str]:
        rig.d.overrun(60.0)
        raise AssertionError("the overrun should have interrupted the prune")

    monkeypatch.setattr(dispatcher_module, "prune_run_dirs", overrun)
    with pytest.raises(TickOverrun):
        rig.tick()
