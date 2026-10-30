"""Dispatcher tests: listing gate, model pairs, closed idle reasons, driver refusal, attempts, crashes, readers, stop, starved.

Depends on: pastore.dispatcher with a fake Host, fake handles and a FakeClock; one end-to-end run uses real start_run.
"""

import json
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from pastore.checks import MODULE_VERSION_FILE, Host
from pastore.dispatcher import Deps, Dispatcher, _guarded_tick, serve
from pastore.policy import PolicyCache, PolicyError
from pastore.runner import RunDir, start_run
from pastore.spec import IDLE_REASONS, WORKSTATION, Ready, RunReader, TaskSpec, ValidateReader
from pastore.state import (
    DispatcherLock,
    EventRow,
    JobRow,
    LockHeld,
    State,
    default_paths,
    open_state,
)
from tests.fakes import FakeClock, RecordingRunner

T0 = datetime(2026, 10, 24, 2, 0, tzinfo=UTC)
MODEL = "m"
SHA256 = "a" * 64

Jobs = Sequence[str | tuple[str, str]]


class FakeHandle:
    """A started run whose poll results are scripted; None means still running."""

    def __init__(self, codes: list[int | None]) -> None:
        self.codes = codes
        self.stopped = False

    def poll(self) -> int | None:
        return self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]

    def stop(self, run: object, wait_s: float = 30.0) -> int:
        self.stopped = True
        return -15


def task(root: Path, task_id: str, **kw: object) -> TaskSpec:
    fields: dict[str, object] = {
        "consumer": "c",
        "id": task_id,
        "kind": "bench",
        "argv": ("./job",),
        "cwd": root,
        "env": {},
        "models": (MODEL,),
        "timeout_s": 3600,
        "ok_exit": frozenset({0}),
        "ollama_env": "OLLAMA_HOST",
        "model_env": "MODEL_NAME",
        "reader": RunReader(("./read",)),
        "ready": None,
        "root": root,
        "fingerprint": f"fp-{task_id}",
    }
    fields.update(kw)
    return TaskSpec(**fields)  # type: ignore[arg-type]


def host(module: str = "580.178.04", apps: str = "1, /usr/bin/cosmic-monitor\n") -> Host:
    runner = RecordingRunner(
        {
            "dpkg-query": RecordingRunner.ok("580.178.04-0ubuntu0.24.04.1"),
            "nvidia-smi": RecordingRunner.ok(apps),
        }
    )

    def read_text(path: Path) -> str:
        return module + "\n" if path == MODULE_VERSION_FILE else "MemAvailable: 40000000 kB\n"

    return Host(run=runner, read_text=read_text, http_status=lambda _url, _t: 200)


class Rig:
    """A dispatcher over tmp state, a fake host, scripted handles and a FakeClock."""

    def __init__(self, tmp_path: Path, specs: list[TaskSpec], jobs: Jobs, **deps: object) -> None:
        self.root = tmp_path / "repo"
        self.root.mkdir(parents=True)
        self.paths = default_paths(tmp_path / "home")
        self.state: State = open_state(self.paths)
        self.specs = {s.id: s for s in specs}
        self.policy = tmp_path / "policy.yaml"
        self.list_jobs(jobs)
        self.handles: list[FakeHandle] = []
        self.next_codes: list[int | None] = [0]
        self.reader_rc: dict[str, int] = {}
        self.run_in_calls: list[list[str]] = []
        self.run_in_envs: list[dict[str, str]] = []
        self.clock = FakeClock(T0)
        base: dict[str, object] = {
            "paths": self.paths,
            "load_manifest": lambda _p, _c: dict(self.specs),
            "host": host(),
            "run_in": self.run_in,
            "start": self.start,
            "base_env": {"PATH": "/usr/bin", "SECRET_TOKEN": "x"},
            "ollama_url": "http://127.0.0.1:11434",
        }
        base.update(deps)
        self.deps = Deps(**base)  # type: ignore[arg-type]
        self.d = Dispatcher(self.deps, self.state, PolicyCache(self.policy, lambda _m: None))
        for s in specs:
            self.state.record_contract_pass(s.ref, s.fingerprint, T0)

    def list_jobs(self, jobs: Jobs) -> None:
        """Write a valid v2 policy listing these jobs; a bare task ref uses the default model."""
        evidence = json.dumps(str(self.root / "evidence.json"))
        pairs = [(j, MODEL) if isinstance(j, str) else j for j in jobs]
        listed = "".join(
            f"  - {{task: {t}, model: {json.dumps(m)}, evidence: {evidence}, sha256: {SHA256}}}\n"
            for t, m in pairs
        )
        self.policy.write_text(
            "version: 2\n"
            'hosts:\n  workstation: {url: "http://127.0.0.1:11435", vram_gb: 24, max_models: 3, parallel: 3}\n'
            'sources:\n  - {level: hearth, cgroup: ["pastore-*.scope"]}\n'
            f"consumers:\n  c: {self.root / 'gpu-tasks.yaml'}\njobs:\n{listed or '  []'}\n"
        )

    def run_in(
        self, argv: Sequence[str], cwd: Path, env: Mapping[str, str]
    ) -> CompletedProcess[str]:
        self.run_in_calls.append(list(argv))
        self.run_in_envs.append(dict(env))
        return CompletedProcess(list(argv), self.reader_rc.get(argv[0], 0), "", "")

    def start(
        self, spec: TaskSpec, run_dir: RunDir, run_id: str, url: str, env: Mapping[str, str]
    ) -> FakeHandle:
        handle = FakeHandle(list(self.next_codes))
        self.handles.append(handle)
        return handle

    def tick(self) -> bool:
        return self.d.tick(self.clock())

    def events(self, kind: str) -> list[EventRow]:
        return [e for e in self.state.recent_events(1000) if e.kind == kind]

    def job_status(self, task_id: str, model: str = MODEL) -> str:
        job = self.state.get_job(f"c/{task_id}", model)
        assert job is not None
        return job.status


def test_invariant_only_policy_listed_jobs_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a"), task(tmp_path, "b")], ["c/a"])
    rig.state.request_run_now("c/b", MODEL)
    rig.tick()
    assert rig.events("refused")[0].ref == "c/b"
    assert [e.ref for e in rig.events("run-start")] == ["c/a"]
    for _ in range(5):
        rig.tick()
    assert [e.ref for e in rig.events("run-start")] == ["c/a"]


def test_invariant_contract_pass_is_rechecked_by_the_dispatcher(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.specs["a"] = task(tmp_path, "a", fingerprint="fp-edited")
    rig.tick()
    assert rig.handles == []
    assert "no contract pass" in rig.events("refused")[0].detail
    assert rig.state.idle_reason() == "no-ready-task"


def test_invariant_every_idle_period_has_a_closed_list_reason(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], [])
    rig.paths.stop_file.write_text("")
    seen = []
    rig.tick()
    seen.append(rig.state.idle_reason())
    rig.paths.stop_file.unlink()
    rig.tick()
    seen.append(rig.state.idle_reason())
    rig.list_jobs(["c/a"])
    mismatch = replace(rig.deps, host=host(module="580.159.03"))
    rig.d = Dispatcher(mismatch, rig.state, PolicyCache(rig.policy, lambda _m: None))
    rig.tick()
    seen.append(rig.state.idle_reason())
    assert seen == ["stopped", "no-ready-task", "driver-mismatch"]
    assert all(e.detail.split(":")[0] in IDLE_REASONS for e in rig.events("idle"))
    assert len(rig.events("idle")) == 3


def test_invariant_driver_mismatch_refuses_dispatch(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host(module="580.159.03"))
    rig.tick()
    assert rig.handles == []
    assert rig.state.idle_reason() == "driver-mismatch"


def test_foreign_gpu_process_blocks_start(tmp_path: Path) -> None:
    apps = "1, /usr/bin/cosmic-monitor\n2, /opt/game/bin/game\n"
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host(apps=apps))
    rig.tick()
    assert rig.state.idle_reason() == "gpu-foreign:/opt/game/bin/game"


def test_three_counted_failures_mark_the_job_failed(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.next_codes = [1]
    for _ in range(10):
        rig.tick()
    assert len(rig.handles) == 3
    assert rig.job_status("a") == "failed"
    [failed] = rig.events("failed")
    assert failed.ref == "c/a" and failed.detail == f"{MODEL}: 3 attempts"


def test_timeout_is_counted_and_stops_the_scope(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a", timeout_s=60)], ["c/a"])
    rig.next_codes = [None]
    rig.tick()
    rig.clock.advance(61)
    rig.tick()
    assert rig.handles[0].stopped
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None and job.attempts == 1 and job.status == "pending"


def test_crash_is_charged_only_on_the_second_consecutive_crash(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host())
    rig.next_codes = [None]
    rig.tick()
    for expected_attempts in (0, 1):
        rig.d.recover(rig.clock())
        job = rig.state.get_job("c/a", MODEL)
        assert job is not None and job.status == "pending" and job.attempts == expected_attempts
        rig.d = Dispatcher(rig.deps, rig.state, PolicyCache(rig.policy, lambda _m: None))
        rig.tick()


def test_crash_streak_resets_when_another_job_was_running(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a"), task(tmp_path, "b")], ["c/a", "c/b"])
    rig.next_codes = [None]
    rig.tick()
    rig.d.recover(rig.clock())
    a = rig.state.get_job("c/a", MODEL)
    assert a is not None and a.crash_streak == 1
    rig.state.upsert_job(replace(a, status="blocked"))
    rig.d = Dispatcher(rig.deps, rig.state, PolicyCache(rig.policy, lambda _m: None))
    rig.tick()
    assert rig.events("run-start")[0].ref == "c/b"
    rig.d.recover(rig.clock())
    a = rig.state.get_job("c/a", MODEL)
    assert a is not None and a.crash_streak == 0


def test_reader_failed_blocks_without_counting_an_attempt(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.reader_rc["./read"] = 3
    for _ in range(4):
        rig.tick()
    assert len(rig.handles) == 1
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None and job.status == "blocked" and job.attempts == 0
    assert rig.events("reader-failed")[0].detail == f"{MODEL}: reader exit 3"


def test_validate_reader_waits_for_owner_ack(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", reader=ValidateReader(("./validate",), "owner"))
    rig = Rig(tmp_path, [spec], ["c/a"])
    rig.tick()
    rig.tick()
    run_id = rig.events("awaiting-review")[0].detail
    assert rig.job_status("a") == "blocked"
    rig.state.ack(run_id, rig.clock())
    rig.tick()
    assert rig.job_status("a") == "done" and len(rig.handles) == 1


def test_recurring_job_checks_ready_at_most_every_interval(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", ready=Ready(3600, ("./ready",)))
    rig = Rig(tmp_path, [spec], ["c/a"])
    rig.reader_rc["./ready"] = 1
    rig.tick()
    rig.tick()
    assert rig.run_in_calls == [["./ready"]] and rig.handles == []
    rig.reader_rc["./ready"] = 2
    rig.clock.advance(3601)
    rig.tick()
    assert rig.events("refused")[0].detail == "ready-check-exit:2"
    rig.reader_rc["./ready"] = 0
    rig.clock.advance(3601)
    rig.tick()
    rig.tick()
    assert len(rig.handles) == 1 and rig.job_status("a") == "pending"


def test_ready_job_waiting_over_24h_is_flagged_starved_once(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host(module="580.159.03"))
    rig.tick()
    rig.clock.advance(23 * 3600)
    rig.tick()
    assert rig.events("starved") == []
    rig.clock.advance(2 * 3600)
    rig.tick()
    rig.tick()
    [starved] = rig.events("starved")
    assert starved.ref == "c/a" and starved.detail.startswith(f"{MODEL}: ready since ")


def test_graceful_stop_waits_for_the_job_and_stop_now_does_not(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.next_codes = [None, 0]
    rig.tick()
    rig.state.request_stop("graceful")
    assert rig.tick() is True
    assert rig.tick() is False
    assert rig.job_status("a") == "done"

    rig2 = Rig(tmp_path / "two", [task(tmp_path, "b")], ["c/b"])
    rig2.next_codes = [None]
    rig2.tick()
    rig2.state.request_stop("now")
    assert rig2.tick() is False
    job = rig2.state.get_job("c/b", MODEL)
    assert rig2.handles[0].stopped and job is not None and job.status == "pending"
    assert job.attempts == 0


def test_invariant_job_never_writes_into_a_repo(tmp_path: Path) -> None:
    script = tmp_path / "repo-src" / "job.py"
    script.parent.mkdir()
    script.write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['PASTORE_RUN_DIR'], 'results.json').write_text('{}')\n"
    )

    def real_start(
        spec: TaskSpec, run_dir: RunDir, run_id: str, url: str, env: Mapping[str, str]
    ) -> object:
        return start_run(spec, run_dir, run_id, url, env, wrap=lambda _r, argv, _t: list(argv))

    rig = Rig(tmp_path, [], [], start=real_start)
    spec = task(rig.root, "a", argv=(sys.executable, str(script)))
    rig.specs["a"] = spec
    rig.state.record_contract_pass(spec.ref, spec.fingerprint, T0)
    rig.list_jobs(["c/a"])
    before = sorted(rig.root.rglob("*"))
    rig.tick()
    assert rig.d.active is not None
    out = rig.d.active.run_dir.out
    rig.d.active.handle.poll()
    for _ in range(50):
        if rig.d.active is None:
            break
        rig.clock.advance(1)
        rig.tick()
        time.sleep(0.05)
    assert (out / "results.json").read_text() == "{}"
    assert out.is_relative_to(rig.paths.runs_dir)
    assert not out.is_relative_to(rig.root)
    assert sorted(rig.root.rglob("*")) == before


def test_run_dirs_never_collide_within_one_second(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a"), task(tmp_path, "b")], ["c/a", "c/b"])
    rig.tick()
    rig.tick()
    rig.tick()
    ids = [e.detail.split()[0] for e in rig.events("run-start")]
    assert len(ids) == 2 and len(set(ids)) == 2


@pytest.mark.parametrize("kind", ["refused", "run-start", "run-end", "done"])
def test_event_kinds_used_by_the_dispatcher_are_accepted_by_state(
    tmp_path: Path, kind: str
) -> None:
    state = open_state(default_paths(tmp_path))
    state.record_event(T0 + timedelta(seconds=1), kind, "c/a", "x")
    assert state.recent_events(1)[0].kind == kind


def test_recover_stops_leftover_scopes_and_unsticks_job_rows(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.state.upsert_job(JobRow("c/a", MODEL, "fp-a", "running"))
    rig.d.recover(rig.clock())
    runner = rig.deps.host.run
    assert isinstance(runner, RecordingRunner)
    assert any(c[:3] == ["systemctl", "--user", "list-units"] for c in runner.calls)
    assert rig.job_status("a") == "pending"


def test_readers_and_checks_get_no_dispatcher_secrets(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", ready=Ready(3600, ("./ready",)), env={"MODEL": "m"})
    rig = Rig(tmp_path, [spec], ["c/a"])
    rig.tick()
    rig.tick()
    check_env, reader_env = rig.run_in_envs[0], rig.run_in_envs[-1]
    assert check_env == {"PATH": "/usr/bin", "MODEL": "m"}
    assert "SECRET_TOKEN" not in reader_env and reader_env["PASTORE_RUN_DIR"].endswith("/out")
    assert reader_env["FUCONE_RUN_DIR"] == reader_env["PASTORE_RUN_DIR"]


def test_a_manifest_edit_reruns_a_done_one_shot(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.tick()
    rig.tick()
    assert rig.job_status("a") == "done"
    rig.tick()
    assert len(rig.handles) == 1
    edited = task(tmp_path, "a", fingerprint="fp-a2")
    rig.specs["a"] = edited
    rig.state.record_contract_pass(edited.ref, edited.fingerprint, T0)
    rig.tick()
    assert len(rig.handles) == 2


def test_run_now_never_restarts_a_done_job_and_a_blocked_one_is_recorded(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.tick()
    rig.tick()
    rig.state.request_run_now("c/a", MODEL)
    rig.tick()
    refused = rig.events("refused")[0].detail
    assert len(rig.handles) == 1 and refused.startswith(f"{MODEL}: run-now: not listed")
    assert "not pending" in refused
    rig.state.rerun("c/a", MODEL)
    rig.paths.stop_file.write_text("")
    rig.state.request_run_now("c/a", MODEL)
    rig.tick()
    assert rig.events("refused")[0].detail == f"{MODEL}: run-now: stopped"
    rig.paths.stop_file.unlink()
    rig.state.request_run_now("c/a", MODEL)
    rig.tick()
    assert len(rig.handles) == 2


def test_a_job_that_cannot_start_is_a_counted_failure(tmp_path: Path) -> None:
    def broken_start(*_args: object) -> FakeHandle:
        raise FileNotFoundError("./job")

    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], start=broken_start)
    rig.tick()
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None and job.status == "pending" and job.attempts == 1
    assert rig.d.active is None and "could not start" in rig.events("refused")[0].detail


def test_guarded_tick_records_an_error_and_keeps_the_loop_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])

    def explode(_now: datetime) -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(rig.d, "tick", explode)
    assert _guarded_tick(rig.d, rig.state, rig.clock) is True
    assert rig.events("tick-error")[0].detail == "RuntimeError: boom"


def test_serve_recovers_drops_stale_requests_and_stops_on_request(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.state.request_stop("now")
    rig.state.request_run_now("c/a", MODEL)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        rig.clock.advance(seconds)
        if len(sleeps) == 3:
            rig.state.request_stop("graceful")

    assert serve(rig.deps, rig.policy, clock=rig.clock, sleep=sleep, tick_s=1.0) == 0
    assert len(sleeps) == 3
    assert rig.events("recover") and "dropped stale" in rig.events("dispatcher-start")[0].detail


def test_sigterm_stops_now_logs_the_stop_and_restores_the_handler(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.next_codes = [None, None, None]
    before = signal.getsignal(signal.SIGTERM)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        rig.clock.advance(seconds)
        if len(sleeps) == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    assert serve(rig.deps, rig.policy, clock=rig.clock, sleep=sleep) == 0
    assert sleeps == [1.0, 1.0]  # woke mid-tick, not after the rest of the 30 s
    [stop] = rig.events("dispatcher-stop")
    [end] = rig.events("run-end")
    assert stop.detail == "sigterm" and stop.id < end.id  # logged before the job is stopped
    job = rig.state.get_job("c/a", MODEL)
    assert rig.handles[0].stopped and job is not None and job.status == "pending"
    assert job.attempts == 0
    assert signal.getsignal(signal.SIGTERM) is before


def test_serve_refuses_a_bad_policy_and_a_held_lock(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    with DispatcherLock(rig.paths.lock), pytest.raises(LockHeld):
        serve(rig.deps, rig.policy, clock=rig.clock, sleep=lambda _s: None)
    rig.policy.write_text("version: 2\n")
    with pytest.raises(PolicyError):
        serve(rig.deps, rig.policy, clock=rig.clock, sleep=lambda _s: None)


@pytest.mark.parametrize("hour", [3, 14])
def test_a_ready_pair_starts_at_any_local_hour(tmp_path: Path, hour: int) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.clock.now_value = datetime(2026, 10, 24, hour, 0, tzinfo=UTC)
    rig.tick()
    assert len(rig.handles) == 1 and rig.state.idle_reason() is None


def test_a_pair_whose_model_is_not_in_the_manifest_is_refused_and_never_starts(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], [("c/a", "other")])
    rig.state.request_run_now("c/a", "other")
    for _ in range(3):
        rig.tick()
    refused = rig.events("refused")
    assert rig.handles == [] and rig.state.runs_with_status("running") == []
    assert {e.ref for e in refused} == {"c/a"} and len(refused) == 2
    assert sum("'other'" in e.detail for e in refused) == 1
    assert rig.state.get_job("c/a", "other") is None


def test_two_pairs_of_one_task_are_separate_jobs_run_one_at_a_time_in_policy_order(
    tmp_path: Path,
) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m2"), ("c/a", "m1")])
    rig.next_codes = [None, 0]
    for _ in range(8):
        rig.tick()
        assert len(rig.state.runs_with_status("running")) <= 1
        rig.clock.advance(1)
    starts = [e.detail.split() for e in reversed(rig.events("run-start"))]
    assert [model for _id, model in starts] == ["m2", "m1"]
    assert {e.ref for e in rig.events("run-start")} == {"c/a"}
    done = rig.state.runs_with_status("done")
    assert [(r.model, r.host) for r in done] == [("m2", WORKSTATION), ("m1", WORKSTATION)]
    assert rig.job_status("a", "m1") == "done" and rig.job_status("a", "m2") == "done"


def test_run_now_starts_exactly_the_requested_pair(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"), ready=Ready(3600, ("./ready",)))
    rig = Rig(tmp_path, [spec], [("c/a", "m1"), ("c/a", "m2")])
    rig.reader_rc["./ready"] = 1
    rig.state.request_run_now("c/a", "m2")
    rig.tick()
    assert [r.model for r in rig.state.runs_with_status("running")] == ["m2"]
    assert rig.job_status("a", "m2") == "running"
    assert rig.run_in_calls == [] and len(rig.handles) == 1
    assert rig.events("run-start")[0].detail.endswith(" m2")


def test_recover_charges_only_the_pair_that_was_running(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m1"), ("c/a", "m2")])
    rig.next_codes = [None]
    rig.tick()
    idle = rig.state.get_job("c/a", "m2")
    assert idle is not None
    rig.state.upsert_job(replace(idle, crash_streak=1))
    rig.d.recover(rig.clock())
    ran, other = rig.state.get_job("c/a", "m1"), rig.state.get_job("c/a", "m2")
    assert ran is not None and ran.crash_streak == 1 and ran.status == "pending"
    assert other is not None and other.crash_streak == 0 and other.status == "pending"
    assert rig.events("recover")[0].detail == "c/a m1"


def test_stop_now_returns_only_the_running_pair_to_pending(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m2"), ("c/a", "m1")])
    rig.next_codes = [None]
    rig.tick()
    waiting = rig.state.get_job("c/a", "m1")
    assert waiting is not None and rig.job_status("a", "m2") == "running"
    rig.state.request_stop("now")
    assert rig.tick() is False
    assert rig.handles[0].stopped and rig.job_status("a", "m2") == "pending"
    assert rig.state.jobs_with_status("running") == []
    assert rig.state.get_job("c/a", "m1") == waiting


def test_a_pair_that_cannot_start_is_charged_to_that_pair(tmp_path: Path) -> None:
    def broken_start(*_args: object) -> FakeHandle:
        raise FileNotFoundError("./job")

    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m2"), ("c/a", "m1")], start=broken_start)
    rig.tick()
    started, other = rig.state.get_job("c/a", "m2"), rig.state.get_job("c/a", "m1")
    assert started is not None and started.attempts == 1 and started.status == "pending"
    assert other is None or (other.attempts == 0 and other.status == "pending")
    assert rig.events("refused")[0].detail.startswith("m2: could not start: ")


def test_run_now_still_respects_the_host_checks(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host(module="580.159.03"))
    rig.state.request_run_now("c/a", MODEL)
    rig.tick()
    [refused] = rig.events("refused")
    assert rig.handles == [] and rig.job_status("a") == "pending"
    assert refused.ref == "c/a" and refused.detail == f"{MODEL}: run-now: driver-mismatch"
    assert rig.state.idle_reason() == "driver-mismatch"


def test_the_model_must_match_the_manifest_exactly_tag_included(tmp_path: Path) -> None:
    models = ("qwen3:8b", "hf.co/org/model:Q4_K_M")
    spec = task(tmp_path, "a", models=models)
    jobs = [("c/a", "qwen3:14b"), ("c/a", "qwen3"), ("c/a", "hf.co/org/model:Q4_K_M")]
    rig = Rig(tmp_path, [spec], jobs)
    rig.tick()
    refused = rig.events("refused")
    assert sorted(e.detail for e in refused) == [
        "model 'qwen3' is not in the manifest's models",
        "model 'qwen3:14b' is not in the manifest's models",
    ]
    assert [r.model for r in rig.state.runs_with_status("running")] == ["hf.co/org/model:Q4_K_M"]
    assert (
        rig.state.get_job("c/a", "qwen3:14b") is None and rig.state.get_job("c/a", "qwen3") is None
    )


def test_an_unknown_task_is_refused_as_a_manifest_problem(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/zz"])
    rig.tick()
    [refused] = rig.events("refused")
    assert rig.handles == [] and refused.ref == "c/zz" and refused.detail.startswith("manifest:")


def test_recover_leaves_the_other_models_of_the_task_alone(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", models=("m0", "m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m1"), ("c/a", "m2"), ("c/a", "m0")])
    rig.next_codes = [None]
    rig.tick()
    settled = JobRow("c/a", "m2", spec.fingerprint, "blocked", attempts=2)
    orphan = JobRow("c/a", "m0", spec.fingerprint, "running", crash_streak=1)
    rig.state.upsert_job(settled)
    rig.state.upsert_job(orphan)
    rig.d.recover(rig.clock())
    ran = rig.state.get_job("c/a", "m1")
    assert ran is not None and ran.status == "pending" and ran.crash_streak == 1
    assert ran.attempts == 0
    assert rig.state.get_job("c/a", "m2") == settled
    assert rig.state.get_job("c/a", "m0") == replace(orphan, status="pending", crash_streak=0)
    assert [r.model for r in rig.state.runs_with_status("crashed")] == ["m1"]
    assert rig.events("recover")[0].detail == "c/a m1"


def test_pair_specific_events_name_the_pairs_own_model(
    tmp_path: Path,
) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m1"), ("c/a", "m2")])
    rig.state.upsert_job(JobRow("c/a", "m1", spec.fingerprint, "blocked"))
    rig.paths.stop_file.write_text("")
    rig.state.request_run_now("c/a", "m2")
    rig.tick()
    rig.paths.stop_file.unlink()
    rig.reader_rc["./read"] = 3
    rig.tick()
    rig.tick()
    rig.state.rerun("c/a", "m2")
    rig.next_codes = [1]
    for _ in range(8):
        rig.tick()
    [refused] = rig.events("refused")
    [reader] = rig.events("reader-failed")
    [failed] = rig.events("failed")
    assert [e.ref for e in (refused, reader, failed)] == ["c/a", "c/a", "c/a"]
    assert refused.detail == "m2: run-now: stopped"
    assert reader.detail == "m2: reader exit 3"
    assert failed.detail == "m2: 3 attempts"
    assert {r.model for r in rig.state.runs_with_status("failed")} == {"m2"}
