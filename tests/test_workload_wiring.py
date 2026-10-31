"""R35 wiring and check 15 end to end: the tick records foreign workload overlaps, `pastore status` shows them, `pastore ack-workload` clears them.

Depends on: pastore.dispatcher, pastore.cli and pastore.state over the fleet Rig of tests/test_dispatcher.py; a scripted journalctl on the host run seam (no real journalctl, systemd, ssh or Ollama).
"""

import io
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pastore.cli import main
from pastore.dispatcher import Dispatcher, _guarded_tick
from pastore.policy import PolicyCache
from pastore.runner import Runner
from pastore.spec import WORKSTATION
from pastore.state import RunRow, StateError, default_paths, open_state
from pastore.workload_watch import WORKLOAD_UNIT, REACHABILITY_UNIT, UNITS
from tests.fakes import RecordingRunner
from tests.test_dispatcher import (
    EXECUTOR,
    FLEET,
    GIB,
    MODEL,
    T0,
    WORKSTATION_ONLY,
    Rig,
    fleet_rig,
    host,
    net_of,
)

UP = "health=ok 12ms"
DOWN = (
    "health=down since 11:58, "
    "connection refused"
)
LOST = (
    "[LINK ALARM] link DOWN for 312s during an active window. "
    "Work is NOT being processed (paused). Check the other node."
)
EVIL = "\x1b[31mEVIL\x1b]0;pwned\x07"
RUN_A = "20261024T020000-1"
KEY = f"{RUN_A} reachability 2026-10-24T02:03:00Z"
DETAIL = (
    f"run={RUN_A} c/a run=2026-10-24T02:00:00Z..running "
    "signal=reachability disruption=2026-10-24T02:03:00Z..ongoing"
)
WARNING = f"<4>pastore: workload-overlap {DETAIL}\n"


def at(minutes: float) -> datetime:
    """T0 plus minutes."""
    return T0 + timedelta(minutes=minutes)


class Journal:
    """The host's run seam: answers `journalctl -u <unit>` from recorded lines up to the clock, anything else from `base`."""

    def __init__(self, base: Runner, now: Callable[[], datetime]) -> None:
        self.base = base
        self.now = now
        self.lines: dict[str, list[tuple[datetime, str]]] = {unit: [] for unit in UNITS}
        self.rc = 0
        self.raw: str | None = None
        self.on_read: Callable[[], None] | None = None
        self.reads: list[list[str]] = []

    def add(self, unit: str, minutes: float, message: str) -> None:
        """Record a line of this unit at T0 + minutes."""
        self.lines[unit].append((at(minutes), message))

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if list(argv[:2]) != ["journalctl", "-u"]:
            return self.base(argv)
        self.reads.append(list(argv))
        if self.on_read is not None:
            self.on_read()
        since = int(next(a for a in argv if a.startswith("--since=@")).removeprefix("--since=@"))
        now = self.now()
        body = "".join(
            json.dumps(
                {"__REALTIME_TIMESTAMP": str(int(when.timestamp()) * 1_000_000), "MESSAGE": m}
            )
            + "\n"
            for when, m in self.lines[argv[2]]
            if since <= when.timestamp() <= now.timestamp()
        )
        out = body if self.raw is None else self.raw
        return subprocess.CompletedProcess(list(argv), self.rc, out, "")


class Wired:
    """A fleet Rig whose host answers journalctl from recorded lines; c/a runs on the executor and keeps running."""

    def __init__(self, tmp_path: Path, *, fleet: bool = True) -> None:
        base = host()
        assert isinstance(base.run, RecordingRunner)
        self.recorder = base.run
        self.journal = Journal(base.run, lambda: self.rig.clock())
        self.rig: Rig = fleet_rig(
            tmp_path,
            net_of({MODEL: GIB}, {MODEL: GIB} if fleet else None),
            ["c/a"],
            ids=("a", "b"),
            hosts=FLEET if fleet else WORKSTATION_ONLY,
            host=replace(base, run=self.journal),
        )
        self.rig.next_codes = [None]
        self.home = tmp_path / "home"

    def tick_at(self, minutes: float) -> bool:
        """Set the clock to T0 + minutes and run one tick."""
        self.rig.clock.now_value = at(minutes)
        return self.rig.tick()

    def overlap_events(self) -> list[str]:
        """The detail of every `workload-overlap` event, oldest first."""
        return [e.detail for e in reversed(self.rig.events("workload-overlap"))]

    def cli(self, *argv: str) -> tuple[int, str]:
        """Run the pastore CLI over this rig's state; the gateway is a refused connection."""
        out = io.StringIO()
        code = main(
            [*argv],
            home=self.home,
            policy_path=self.rig.policy,
            clock=self.rig.clock,
            out=out,
            stdin_isatty=lambda: False,
            http_json=refuse_gateway,
        )
        return code, out.getvalue()

    def workload_status(self) -> list[str]:
        """The `pastore status` lines about the workload watch."""
        _, out = self.cli("status")
        return [x for x in out.splitlines() if x.startswith(("workload", "run pastore"))]


def refuse_gateway(url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
    """An injected HttpJson for the CLI: nothing answers."""
    raise ConnectionRefusedError(111, "Connection refused")


def overlap_scenario(tmp_path: Path, *, down: bool) -> Wired:
    """c/a on the executor, c/b joining on the workstation mid-way; the workload goes down at +3 min when `down`."""
    w = Wired(tmp_path)
    w.journal.add(REACHABILITY_UNIT, 1, UP)
    w.journal.add(REACHABILITY_UNIT, 3, DOWN if down else UP)
    w.tick_at(0)
    w.tick_at(2)
    w.rig.list_jobs(["c/a", "c/b"])
    for minutes in (4, 5, 6):
        w.tick_at(minutes)
    for handle in w.rig.handles:
        handle.codes = [0]
    w.tick_at(7)
    w.tick_at(8)
    return w


def test_check_15_an_overlap_records_one_event_over_many_ticks_and_warns_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = Wired(tmp_path)
    w.journal.add(REACHABILITY_UNIT, 1, UP)
    w.journal.add(REACHABILITY_UNIT, 3, DOWN)
    w.tick_at(0)
    w.tick_at(2)
    assert w.rig.events("workload-overlap") == [] and capsys.readouterr().err == ""
    w.tick_at(4)
    [event] = w.rig.events("workload-overlap")
    assert (event.ref, event.at, event.detail) == ("c/a", at(4).isoformat(), f"{KEY} {DETAIL}")
    assert capsys.readouterr().err == WARNING
    for minutes in (5, 6, 7):
        w.tick_at(minutes)
    assert len(w.rig.events("workload-overlap")) == 1 and capsys.readouterr().err == ""


def test_check_15_the_dedupe_survives_a_dispatcher_restart_and_the_run_ending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = overlap_scenario(tmp_path, down=True)
    capsys.readouterr()
    assert w.overlap_events() == [f"{KEY} {DETAIL}"]
    w.rig.d = Dispatcher(w.rig.deps, w.rig.state, PolicyCache(w.rig.policy, lambda _m: None))
    w.tick_at(9)
    assert w.overlap_events() == [f"{KEY} {DETAIL}"] and capsys.readouterr().err == ""


@dataclass(frozen=True)
class Outcome:
    """Everything an overlap must leave unchanged: the non-watch events, finished runs, starts and stops."""

    events: list[tuple[str, str | None, str]]
    done: list[tuple[str, str, str, int | None]]
    start_models: list[str]
    stopped: list[bool]
    idle: str | None


def outcome_of(w: Wired) -> Outcome:
    """Collect what the overlap must not change from a finished scenario."""
    rig = w.rig
    return Outcome(
        [
            (e.kind, e.ref, e.detail)
            for e in reversed(rig.state.recent_events(1000))
            if not e.kind.startswith("workload-")
        ],
        [(r.ref, r.host, r.status, r.exit_code) for r in rig.state.runs_with_status("done")],
        rig.start_models,
        [h.stopped for h in rig.handles],
        rig.state.idle_reason(),
    )


def test_check_15_an_overlap_never_stops_a_run_or_changes_work(tmp_path: Path) -> None:
    hit = overlap_scenario(tmp_path / "hit", down=True)
    clean = overlap_scenario(tmp_path / "clean", down=False)
    assert len(hit.rig.events("workload-overlap")) == 1 and clean.rig.events("workload-overlap") == []
    got = outcome_of(hit)
    assert got == outcome_of(clean)
    kinds = [kind for kind, _, _ in got.events]
    assert kinds.count("run-start") == 2 and kinds.count("done") == 2
    assert got.stopped == [False, False]
    assert {(ref, host_name) for ref, host_name, _, _ in got.done} == {
        ("c/a", EXECUTOR),
        ("c/b", WORKSTATION),
    }


def test_check_15_no_overlap_records_nothing_and_the_watch_is_armed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = overlap_scenario(tmp_path, down=False)
    assert w.rig.events("workload-overlap") == [] and w.rig.events("workload-ack") == []
    assert w.rig.state.workload_watch() == "armed" and capsys.readouterr().err == ""


def test_check_15_workload_down_the_whole_window_reads_workload_down_with_no_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = Wired(tmp_path)
    for minutes in (-20, -10, 1, 3):
        w.journal.add(REACHABILITY_UNIT, minutes, DOWN)
    w.tick_at(0)
    w.tick_at(4)
    assert w.rig.state.runs_with_status("running")[0].host == EXECUTOR
    assert w.rig.state.workload_watch() == "down"
    assert w.rig.events("workload-overlap") == [] and capsys.readouterr().err == ""


def test_check_15_with_no_journal_lines_the_watch_reads_no_signal(tmp_path: Path) -> None:
    w = Wired(tmp_path)
    w.tick_at(0)
    assert w.rig.state.workload_watch() == "no-signal" and w.rig.events("workload-overlap") == []


@pytest.mark.parametrize(
    ("rc", "raw", "status"),
    [
        (1, None, "unreadable: journalctl exit 1"),
        (124, None, "unreadable: journalctl exit 124"),
        (127, None, "unreadable: journalctl exit 127"),
        (0, "not json\n", "unreadable: journalctl output is not JSON"),
    ],
)
def test_check_15_an_unreadable_journal_is_stored_as_unreadable_and_work_goes_on(
    tmp_path: Path, rc: int, raw: str | None, status: str
) -> None:
    w = Wired(tmp_path)
    w.journal.rc, w.journal.raw = rc, raw
    w.tick_at(0)
    rig = w.rig
    assert _guarded_tick(rig.d, rig.state, rig.clock) is True
    assert rig.state.workload_watch() == status
    assert rig.events("tick-error") == [] and rig.events("workload-overlap") == []
    assert len(rig.handles) == 1 and [e.kind for e in rig.events("run-start")] == ["run-start"]
    w.journal.rc, w.journal.raw = 0, None
    w.journal.add(REACHABILITY_UNIT, 0.5, UP)
    w.tick_at(1)
    assert rig.state.workload_watch() == "armed"


def test_check_15_a_run_on_the_workstation_is_never_a_span(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = Wired(tmp_path, fleet=False)
    w.journal.add(REACHABILITY_UNIT, 1, UP)
    w.journal.add(REACHABILITY_UNIT, 3, DOWN)
    w.tick_at(0)
    w.tick_at(20)
    [run] = w.rig.state.runs_with_status("running")
    assert run.host == WORKSTATION
    assert w.rig.events("workload-overlap") == [] and capsys.readouterr().err == ""
    assert w.rig.state.workload_watch() == "down"
    # a span would reach back to 15 min before the run (T0 - 15); with none it is now - 30 min (T0 - 10)
    since = int(at(-10).timestamp())
    assert all(f"--since=@{since}" in read for read in w.journal.reads[-2:])


def test_check_15_the_watch_runs_after_placement_and_on_draining_ticks_only_while_work_remains(
    tmp_path: Path,
) -> None:
    w = Wired(tmp_path)
    started_at_read: list[int] = []
    w.journal.on_read = lambda: started_at_read.append(len(w.rig.handles))
    w.tick_at(0)
    assert started_at_read == [1, 1]
    assert [read[2] for read in w.journal.reads] == [REACHABILITY_UNIT, WORKLOAD_UNIT]
    w.rig.state.request_stop("graceful")
    assert w.tick_at(1) is True
    assert len(w.journal.reads) == 4
    w.rig.handles[0].codes = [0]
    assert w.tick_at(2) is False
    assert len(w.journal.reads) == 4
    assert all(read[0] == "journalctl" and "--user" not in read for read in w.journal.reads)
    assert all(call[0] not in ("ssh", "sudo") for call in w.recorder.calls)


@pytest.mark.parametrize(
    ("down_at", "seen_at", "recorded"),
    [(12, 20, 1), (14.9, 20, 1), (16, 20, 0), (12, 25, 1), (12, 40, 0)],
)
def test_check_15_a_finished_run_counts_for_ten_minutes_and_is_checked_for_thirty(
    tmp_path: Path, down_at: float, seen_at: float, recorded: int
) -> None:
    w = Wired(tmp_path)
    w.journal.add(REACHABILITY_UNIT, 1, UP)
    w.tick_at(0)
    w.rig.handles[0].codes = [0]
    w.tick_at(5)
    assert [r.run_id for r in w.rig.state.runs_with_status("done")] == [RUN_A]
    w.journal.add(REACHABILITY_UNIT, down_at, DOWN)
    w.tick_at(seen_at)
    assert len(w.rig.events("workload-overlap")) == recorded
    if recorded:
        start = at(down_at).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert w.overlap_events()[0].endswith(
            f"run=2026-10-24T02:00:00Z..2026-10-24T02:05:00Z signal=reachability disruption={start}..ongoing"
        )


def test_check_15_a_link_alarm_from_the_workload_is_an_overlap_too(tmp_path: Path) -> None:
    w = Wired(tmp_path)
    w.journal.add(WORKLOAD_UNIT, 2, LOST)
    w.tick_at(0)
    w.tick_at(3)
    [detail] = w.overlap_events()
    assert detail.startswith(f"{RUN_A} link 2026-10-24T02:02:00Z run={RUN_A} c/a ")
    assert "signal=link disruption=2026-10-24T02:02:00Z..ongoing" in detail
    assert "LOST" not in detail


def test_check_15_executor_runs_are_the_running_and_recently_ended_runs_off_the_workstation(
    tmp_path: Path,
) -> None:
    state = open_state(default_paths(tmp_path / "home"))

    def put(run_id: str, host_name: str, status: str, start: float, end: float | None) -> None:
        stamp = None if end is None else at(end).astimezone(UTC).isoformat()
        state.insert_run(
            RunRow(run_id, "c/a", MODEL, host_name, "fp", status, at(start).isoformat(), stamp)
        )

    put("old", EXECUTOR, "ok", 0, 10)
    put("edge", EXECUTOR, "ok", 1, 20)
    put("recent", EXECUTOR, "failed", 2, 40)
    put("live", EXECUTOR, "running", 3, None)
    put("ws-live", WORKSTATION, "running", 4, None)
    put("ws-recent", WORKSTATION, "ok", 5, 45)
    ids = [r.run_id for r in state.executor_runs(at(20))]
    assert ids == ["edge", "recent", "live"]
    assert [r.run_id for r in state.executor_runs(at(20) + timedelta(microseconds=1))] == [
        "recent",
        "live",
    ]
    assert [r.run_id for r in state.executor_runs(at(20) - timedelta(microseconds=1))] == ids
    other_zone = at(20).astimezone(timezone(timedelta(hours=3)))
    assert [r.run_id for r in state.executor_runs(other_zone)] == ids
    state.close()


def test_check_15_has_event_matches_the_whole_detail_or_a_prefix_ending_at_a_space(
    tmp_path: Path,
) -> None:
    state = open_state(default_paths(tmp_path / "home"))
    state.record_event(T0, "workload-overlap", "c/a", "r1 link t1 more words")
    state.record_event(T0, "workload-ack", None, "r2 link t2")
    assert state.has_event("workload-overlap", "r1 link t1")
    assert state.has_event("workload-overlap", "r1 link t1 more words")
    assert not state.has_event("workload-overlap", "r1 link t")
    assert not state.has_event("workload-overlap", "r1 link t1 more words and more")
    assert not state.has_event("workload-overlap", "r2 link t2")
    assert state.has_event("workload-ack", "r2 link t2")
    state.close()


def test_check_15_ack_workload_clears_what_was_recorded_and_logs_each_ack(tmp_path: Path) -> None:
    state = open_state(default_paths(tmp_path / "home"))
    assert state.workload_watch() is None and state.unacked_workload_overlaps() == []
    state.set_workload_watch("armed")
    assert state.workload_watch() == "armed"
    state.record_event(T0, "workload-overlap", "c/a", "k1 d1")
    state.record_event(T0, "idle", None, "no-ready-task")
    state.record_event(T0, "workload-overlap", "c/b", "k2 d2")
    assert [e.detail for e in state.unacked_workload_overlaps()] == ["k1 d1", "k2 d2"]
    assert state.ack_workload(at(1)) == 2 and state.unacked_workload_overlaps() == []
    state.record_event(T0, "workload-overlap", "c/a", "k3 d3")
    assert [e.detail for e in state.unacked_workload_overlaps()] == ["k3 d3"]
    assert state.ack_workload(at(2)) == 1
    assert state.ack_workload(at(3)) == 0
    assert state.unacked_workload_overlaps() == []
    state.record_event(T0, "workload-overlap", "c/a", "k4 d4")
    assert [e.detail for e in state.unacked_workload_overlaps()] == ["k4 d4"]
    acks = [(e.at, e.detail) for e in reversed(state.recent_events(50)) if e.kind == "workload-ack"]
    assert acks == [
        (at(1).isoformat(), "cleared 2"),
        (at(2).isoformat(), "cleared 1"),
        (at(3).isoformat(), "cleared 0"),
    ]
    state.close()


def test_check_15_ack_workload_refuses_a_corrupt_ack_mark(tmp_path: Path) -> None:
    state = open_state(default_paths(tmp_path / "home"))
    state.record_event(T0, "workload-overlap", "c/a", "k1 d1")
    state._set_meta("workload_acked_id", "soon")
    with pytest.raises(StateError, match="workload_acked_id"):
        state.unacked_workload_overlaps()
    with pytest.raises(StateError, match="workload_acked_id"):
        state.ack_workload(at(1))
    state.close()


def test_check_15_status_shows_the_watch_and_the_overlap_until_ack_workload(
    tmp_path: Path,
) -> None:
    w = overlap_scenario(tmp_path, down=True)
    [event] = w.rig.events("workload-overlap")
    assert w.workload_status() == [
        "workload watch: down",
        f"workload-overlap: {event.at} {DETAIL}",
        "run pastore ack-workload once reviewed",
    ]
    assert w.cli("ack-workload") == (0, "acked 1 workload-overlap event(s)\n")
    assert w.workload_status() == ["workload watch: down"]
    assert w.cli("ack-workload") == (0, "acked 0 workload-overlap event(s)\n")
    assert w.rig.state.unacked_workload_overlaps() == []
    assert w.workload_status() == ["workload watch: down"]
    assert [e.detail for e in w.rig.events("workload-ack")] == ["cleared 0", "cleared 1"]


def test_check_15_the_overlap_shows_until_acked_whatever_the_watch_status(tmp_path: Path) -> None:
    w = Wired(tmp_path)
    w.journal.add(REACHABILITY_UNIT, 1, UP)
    w.journal.add(REACHABILITY_UNIT, 3, DOWN)
    w.tick_at(0)
    w.tick_at(4)
    [event] = w.rig.events("workload-overlap")
    w.journal.add(REACHABILITY_UNIT, 5, UP)
    w.tick_at(6)
    assert w.rig.state.workload_watch() == "armed" and len(w.rig.events("workload-overlap")) == 1
    assert w.workload_status() == [
        "workload watch: armed",
        f"workload-overlap: {event.at} {DETAIL}",
        "run pastore ack-workload once reviewed",
    ]
    assert w.cli("ack-workload") == (0, "acked 1 workload-overlap event(s)\n")
    assert w.workload_status() == ["workload watch: armed"]


def test_check_15_status_prints_an_overlap_line_as_one_escape_free_line(tmp_path: Path) -> None:
    w = Wired(tmp_path)
    detail = f"r1 link 2026-10-24T02:00:00Z run=r1 c/a {EVIL}\nforged: line"
    w.rig.state.record_event(T0, "workload-overlap", "c/a", detail)
    _, out = w.cli("status")
    [line] = [x for x in out.splitlines() if "forged" in x]
    assert line.startswith("workload-overlap: ") and "run=r1 c/a" in line
    assert "\x1b" not in out and "\x07" not in out


def test_check_15_an_overlap_after_an_ack_shows_alone(tmp_path: Path) -> None:
    w = Wired(tmp_path)
    w.journal.add(REACHABILITY_UNIT, 1, UP)
    w.journal.add(REACHABILITY_UNIT, 3, DOWN)
    w.tick_at(0)
    w.tick_at(4)
    assert w.cli("ack-workload")[1] == "acked 1 workload-overlap event(s)\n"
    assert w.cli("ack-workload")[1] == "acked 0 workload-overlap event(s)\n"
    w.journal.add(REACHABILITY_UNIT, 5, UP)
    w.journal.add(REACHABILITY_UNIT, 7, DOWN)
    w.tick_at(8)
    shown = w.workload_status()
    assert len(shown) == 3 and shown[0] == "workload watch: down"
    assert "disruption=2026-10-24T02:07:00Z..ongoing" in shown[1] and "02:03:00Z" not in shown[1]
    assert w.cli("ack-workload")[1] == "acked 1 workload-overlap event(s)\n"


def test_check_15_status_before_any_tick_says_the_watch_has_not_run(tmp_path: Path) -> None:
    w = Wired(tmp_path)
    assert w.workload_status() == ["workload watch: not run yet"]


def test_check_15_ack_workload_with_no_state_refuses_and_creates_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["ack-workload"],
        home=tmp_path / "home",
        policy_path=tmp_path / "policy.yaml",
        clock=lambda: T0,
        out=io.StringIO(),
        stdin_isatty=lambda: False,
        http_json=refuse_gateway,
    )
    assert code == 1 and "no state yet" in capsys.readouterr().err
    assert not (tmp_path / "home").exists()


def test_check_15_no_journal_text_reaches_status_stderr_events_or_the_stored_watch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = Wired(tmp_path)
    w.journal.add(REACHABILITY_UNIT, 1, UP)
    w.journal.add(REACHABILITY_UNIT, 3, f"health=down {EVIL} since=x :: {EVIL}")
    w.journal.add(WORKLOAD_UNIT, 3, f"[LINK ALARM] {EVIL}")
    w.tick_at(0)
    w.tick_at(4)
    texts = [e.detail for e in w.rig.events("workload-overlap")]
    assert len(texts) == 2
    texts += [w.cli("status")[1], capsys.readouterr().err, w.rig.state.workload_watch() or ""]
    w.journal.raw = json.dumps({"__REALTIME_TIMESTAMP": "x" + EVIL, "MESSAGE": EVIL}) + "\n"
    w.tick_at(5)
    texts += [w.cli("status")[1], w.rig.state.workload_watch() or ""]
    assert w.rig.state.workload_watch() == "unreadable: journalctl line has no usable timestamp"
    for text in texts:
        assert "EVIL" not in text and "\x1b" not in text and "\x07" not in text


def test_check_15_status_prints_stored_watch_text_as_one_safe_line(tmp_path: Path) -> None:
    w = Wired(tmp_path)
    w.rig.state.set_workload_watch(f"unreadable: {EVIL}\nforged: line")
    [line] = w.workload_status()
    assert line.startswith("workload watch: unreadable: ")
    assert "\x1b" not in line and "\x07" not in line and "\n" not in line
