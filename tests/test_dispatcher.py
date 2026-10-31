"""Dispatcher tests: listing gate, pairs, placement across hosts, checks 8 and 9, idle reasons, attempts, crashes, readers, stop.

Depends on: pastore.dispatcher with a fake Host, a fake `http_json` per host URL, fake handles and a FakeClock; one test
runs the real `hosts.http_json` and placement over FakeOllama servers, one runs the real start_run.
"""

import json
import os
import signal
import sys
import time
import urllib.error
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from pastore.checks import MODULE_VERSION_FILE, Host
from pastore.dispatcher import Deps, Dispatcher, _guarded_tick, serve
from pastore.fake_ollama import FakeOllama
from pastore.hosts import http_json
from pastore.policy import PolicyCache, PolicyError
from pastore.runner import RunDir, start_run
from pastore.spec import (
    IDLE_REASONS,
    WORKSTATION,
    Ready,
    RunReader,
    TaskSpec,
    ValidateReader,
    canonical_model,
)
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
LATEST = canonical_model(MODEL)
SHA256 = "a" * 64
GIB = 1024**3
EXECUTOR = "executor"
WS_URL = "http://ws.test:11435"
EX_URL = "http://exec.test:11435"
INFO: dict[str, object] = {
    "general.architecture": "llama",
    "llama.block_count": 2,
    "llama.attention.head_count_kv": 2,
    "llama.attention.key_length": 64,
}

Jobs = Sequence[str | tuple[str, str]]


@dataclass(frozen=True)
class HostSpec:
    """One `hosts:` entry of the test policy."""

    url: str
    vram_gib: float
    max_models: int
    parallel: int
    context: int


WORKSTATION_ONLY = {WORKSTATION: HostSpec(WS_URL, 24, 3, 3, 32768)}
ONE_AT_A_TIME = {WORKSTATION: replace(WORKSTATION_ONLY[WORKSTATION], max_models=1)}
FLEET = {**WORKSTATION_ONLY, EXECUTOR: HostSpec(EX_URL, 10, 3, 3, 8192)}


class FakeNet:
    """The injected `http_json`: per-URL tags, ps, show and version answers, with scripted outages."""

    def __init__(self) -> None:
        self.tags: dict[str, dict[str, int]] = {}
        self.ps: dict[str, dict[str, tuple[int, int]]] = {}
        self.info: dict[str, dict[str, object]] = {}
        self.link_down: set[str] = set()
        self.version_down: set[str] = set()
        self.calls: list[str] = []

    def __call__(self, url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
        base, _, path = url.partition("/api/")
        self.calls.append(url)
        # The job endpoint answers but is no gateway, so a failed run stays charged (D-050 (4)).
        if url.endswith("/pastore/status"):
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        if base in self.link_down or (path == "version" and base in self.version_down):
            raise OSError(f"{base} is down")
        if path == "version":
            return {"version": "0.0.0-fake"}
        if path == "tags":
            return {"models": [{"name": n, "size": s} for n, s in self.tags.get(base, {}).items()]}
        if path == "ps":
            rows = self.ps.get(base, {})
            return {
                "models": [{"name": n, "size_vram": v, "size": s} for n, (v, s) in rows.items()]
            }
        assert path == "show" and body is not None
        return {"model_info": self.info.get(str(body["model"]), INFO)}


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


def host(
    module: str = "580.178.04",
    apps: str = "1, /usr/bin/cosmic-monitor\n",
    seen: list[str] | None = None,
    refused: set[str] | None = None,
) -> Host:
    """The workstation seams; `seen` collects the URL of every HTTP status check, `refused` names endpoints that refuse."""
    runner = RecordingRunner(
        {
            "dpkg-query": RecordingRunner.ok("580.178.04-0ubuntu0.24.04.1"),
            "nvidia-smi": RecordingRunner.ok(apps),
        }
    )

    def read_text(path: Path) -> str:
        return module + "\n" if path == MODULE_VERSION_FILE else "MemAvailable: 40000000 kB\n"

    def http_status(url: str, _timeout_s: float) -> int:
        if seen is not None:
            seen.append(url)
        if refused and url.partition("/api/")[0] in refused:
            raise OSError(f"{url} refused")
        return 200

    return Host(run=runner, read_text=read_text, http_status=http_status)


class Rig:
    """A dispatcher over tmp state, a fake host, scripted handles and a FakeClock."""

    def __init__(
        self,
        tmp_path: Path,
        specs: list[TaskSpec],
        jobs: Jobs,
        *,
        hosts: Mapping[str, HostSpec] = WORKSTATION_ONLY,
        net: FakeNet | None = None,
        **deps: object,
    ) -> None:
        self.root = tmp_path / "repo"
        self.root.mkdir(parents=True)
        self.paths = default_paths(tmp_path / "home")
        self.state: State = open_state(self.paths)
        self.specs = {s.id: s for s in specs}
        self.policy = tmp_path / "policy.yaml"
        self.hosts = dict(hosts)
        self.net = net or FakeNet()
        self.list_jobs(jobs)
        everything = {MODEL, *(m for s in specs for m in s.models), *(p[1] for p in self.pairs)}
        for host_spec in self.hosts.values():
            self.net.tags.setdefault(host_spec.url, dict.fromkeys(everything, GIB))
        self.handles: list[FakeHandle] = []
        self.start_models: list[str] = []
        self.start_urls: list[str] = []
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
            "http_json": self.net,
        }
        base.update(deps)
        self.deps = Deps(**base)  # type: ignore[arg-type]
        self.d = Dispatcher(self.deps, self.state, PolicyCache(self.policy, lambda _m: None))
        for s in specs:
            self.state.record_contract_pass(s.ref, s.fingerprint, T0)

    def list_jobs(self, jobs: Jobs) -> None:
        """Write a valid v2 policy listing these jobs; a bare task ref uses the default model."""
        evidence = json.dumps(str(self.root / "evidence.json"))
        self.pairs = [(j, MODEL) if isinstance(j, str) else j for j in jobs]
        listed = "".join(
            f"  - {{task: {t}, model: {json.dumps(m)}, evidence: {evidence}, sha256: {SHA256}}}\n"
            for t, m in self.pairs
        )
        hosts = "".join(
            f'  {name}: {{url: "{h.url}", vram_gib: {h.vram_gib}, max_models: {h.max_models},'
            f" parallel: {h.parallel}, context: {h.context}}}\n"
            for name, h in self.hosts.items()
        )
        self.policy.write_text(
            "version: 2\n"
            f"hosts:\n{hosts}"
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
        self,
        spec: TaskSpec,
        run_dir: RunDir,
        run_id: str,
        url: str,
        model: str,
        env: Mapping[str, str],
    ) -> FakeHandle:
        self.start_models.append(model)
        self.start_urls.append(url)
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
    assert [r.exit_code for r in rig.state.runs_with_status("timeout")] == [-15]
    job = rig.state.get_job("c/a", MODEL)
    assert job is not None and job.attempts == 1
    assert (
        len(rig.handles) == 2 and job.status == "running"
    )  # the freed slot is refilled in the same tick


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
        spec: TaskSpec, run_dir: RunDir, run_id: str, url: str, model: str, env: Mapping[str, str]
    ) -> object:
        return start_run(
            spec, run_dir, run_id, url, model, env, wrap=lambda _r, argv, _t: list(argv)
        )

    rig = Rig(tmp_path, [], [], start=real_start)
    spec = task(rig.root, "a", argv=(sys.executable, str(script)))
    rig.specs["a"] = spec
    rig.state.record_contract_pass(spec.ref, spec.fingerprint, T0)
    rig.list_jobs(["c/a"])
    before = sorted(rig.root.rglob("*"))
    rig.tick()
    [active] = rig.d.active.values()
    out = active.run_dir.out
    for _ in range(50):
        if not rig.d.active:
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
    assert reader_env.keys() == {*check_env, "PASTORE_RUN_DIR"}


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
    assert rig.d.active == {} and "could not start" in rig.events("refused")[0].detail


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


def test_a_failing_tick_after_sigterm_is_retried_once_a_second_and_never_spins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    terminated_at_start: list[bool] = []

    def tick(self: Dispatcher, _now: datetime) -> bool:
        terminated_at_start.append(self.terminated)
        if len(terminated_at_start) == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        if len(terminated_at_start) <= 3:
            raise RuntimeError("boom")
        return False

    monkeypatch.setattr(Dispatcher, "tick", tick)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 20:
            raise RuntimeError("the loop is spinning")
        rig.clock.advance(seconds)

    assert serve(rig.deps, rig.policy, clock=rig.clock, sleep=sleep) == 0
    assert terminated_at_start == [False, True, True, True]
    assert sleeps == [1.0, 1.0, 1.0]  # one 1 s sleep after each failed tick, none after the last
    assert [e.detail for e in rig.events("tick-error")] == ["RuntimeError: boom"] * 3


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


def test_the_dispatcher_starts_the_listed_pairs_model_with_the_deps_ollama_url(
    tmp_path: Path,
) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m2")])
    rig.tick()
    assert rig.start_models == ["m2"]
    assert rig.start_urls == [rig.deps.ollama_url]


def test_with_one_model_per_host_two_pairs_of_one_task_run_one_at_a_time_in_policy_order(
    tmp_path: Path,
) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m2"), ("c/a", "m1")], hosts=ONE_AT_A_TIME)
    rig.next_codes = [None, 0]
    for _ in range(8):
        rig.tick()
        assert len(rig.state.runs_with_status("running")) <= 1
        rig.clock.advance(1)
    starts = [e.detail.split() for e in reversed(rig.events("run-start"))]
    assert [(model, host) for _id, model, host in starts] == [
        ("m2", WORKSTATION),
        ("m1", WORKSTATION),
    ]
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
    assert rig.run_in_calls == [["./ready"]] and len(rig.handles) == 1
    skipped, checked = rig.state.get_job("c/a", "m2"), rig.state.get_job("c/a", "m1")
    assert skipped is not None and skipped.last_check_at is None
    assert checked is not None and checked.last_check_at is not None
    assert rig.events("run-start")[0].detail.split()[1:] == ["m2", WORKSTATION]


def test_recover_charges_only_the_pair_that_was_running(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m1"), ("c/a", "m2")], hosts=ONE_AT_A_TIME)
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
    rig = Rig(tmp_path, [spec], [("c/a", "m2"), ("c/a", "m1")], hosts=ONE_AT_A_TIME)
    rig.next_codes = [None]
    rig.tick()
    waiting = rig.state.get_job("c/a", "m1")
    assert waiting is not None and rig.job_status("a", "m2") == "running"
    rig.state.request_stop("now")
    assert rig.tick() is False
    assert rig.handles[0].stopped and rig.job_status("a", "m2") == "pending"
    assert rig.state.jobs_with_status("running") == []
    assert rig.state.get_job("c/a", "m1") == waiting


def test_a_pair_that_cannot_start_is_charged_to_that_pair_and_the_next_still_starts(
    tmp_path: Path,
) -> None:
    def start(
        spec: TaskSpec, run_dir: RunDir, run_id: str, url: str, model: str, env: Mapping[str, str]
    ) -> FakeHandle:
        if model == "m2":
            raise FileNotFoundError("./job")
        return FakeHandle([None])

    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m2"), ("c/a", "m1")], start=start)
    rig.tick()
    broken, other = rig.state.get_job("c/a", "m2"), rig.state.get_job("c/a", "m1")
    assert broken is not None and broken.attempts == 1 and broken.status == "pending"
    assert other is not None and other.attempts == 0 and other.status == "running"
    assert rig.events("refused")[0].detail.startswith("m2: could not start: ")
    assert [r.model for r in rig.d.active.values()] == ["m1"]


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
    rig = Rig(tmp_path, [spec], [("c/a", "m1"), ("c/a", "m2"), ("c/a", "m0")], hosts=ONE_AT_A_TIME)
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


def net_of(workstation: Mapping[str, int], executor: Mapping[str, int] | None = None) -> FakeNet:
    """A FakeNet whose workstation and executor serve exactly these models (name -> tag size)."""
    net = FakeNet()
    net.tags[WS_URL] = dict(workstation)
    net.tags[EX_URL] = dict(executor or {})
    return net


def fleet_rig(
    tmp_path: Path,
    net: FakeNet,
    jobs: Jobs,
    *,
    models: tuple[str, ...] = (MODEL,),
    ids: Sequence[str] = ("a",),
    hosts: Mapping[str, HostSpec] = FLEET,
    task_kw: Mapping[str, object] | None = None,
    **deps: object,
) -> Rig:
    """A Rig over the two-host fleet: one task per id, all offering `models`."""
    specs = [task(tmp_path, i, models=models, **(task_kw or {})) for i in ids]
    return Rig(tmp_path, specs, jobs, hosts=hosts, net=net, **deps)


def test_check_8_a_model_needing_over_10_gib_goes_to_the_workstation_never_the_executor(
    tmp_path: Path,
) -> None:
    net = net_of({"big": 9 * GIB}, {"big": 9 * GIB})
    rig = fleet_rig(tmp_path, net, [("c/a", "big")], models=("big",))
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert (run.model, run.host) == ("big", WORKSTATION)
    assert rig.state.idle_reason() is None


def test_check_8_a_model_loaded_with_a_cpu_part_over_10_gib_never_goes_to_the_executor(
    tmp_path: Path,
) -> None:
    net = net_of({"big": 5 * GIB}, {"big": 5 * GIB})  # the tag estimate (7 GiB) would fit 10 GiB
    net.ps[EX_URL] = {"big": (6 * GIB, 11 * GIB)}  # size_vram 6 GiB, size 11 GiB with the CPU part
    rig = fleet_rig(tmp_path, net, [("c/a", "big")], models=("big",))
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert (run.model, run.host) == ("big", WORKSTATION)


def test_check_8_with_the_workstation_full_a_big_model_waits_with_no_fit_and_no_run(
    tmp_path: Path,
) -> None:
    net = net_of({"big": 9 * GIB}, {"big": 9 * GIB})
    net.ps[WS_URL] = {"other": (20 * GIB, 20 * GIB)}
    rig = fleet_rig(tmp_path, net, [("c/a", "big")], models=("big",))
    for _ in range(3):
        rig.tick()
    job = rig.state.get_job("c/a", "big")
    assert rig.handles == [] and rig.state.last_run("c/a", "big") is None
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert rig.state.idle_reason() == "no-fit" and rig.events("failed") == []
    assert [e.detail for e in rig.events("idle")] == ["no-fit"]


def test_check_9_a_lost_executor_link_waits_with_the_reason_and_no_failed_run(
    tmp_path: Path,
) -> None:
    net = net_of({}, {"only": GIB})
    net.version_down.add(EX_URL)  # tags still answer: only the probe says the link is down
    rig = fleet_rig(tmp_path, net, [("c/a", "only")], models=("only",))
    for _ in range(3):
        rig.tick()
    job = rig.state.get_job("c/a", "only")
    assert rig.handles == [] and rig.state.last_run("c/a", "only") is None
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert rig.state.idle_reason() == "executor-down" and rig.events("failed") == []
    assert [e.detail for e in rig.events("idle")] == ["executor-down"]
    net.version_down.clear()
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == EXECUTOR and rig.state.idle_reason() is None


@pytest.mark.parametrize("timed_out", [False, True])
def test_check_9_a_run_cut_short_by_a_lost_executor_is_stopped_and_not_charged(
    tmp_path: Path, timed_out: bool
) -> None:
    rig = fleet_rig(
        tmp_path,
        net_of({}, {"only": GIB}),
        [("c/a", "only")],
        models=("only",),
        task_kw={"timeout_s": 60},
    )
    rig.next_codes = [None] if timed_out else [1]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == EXECUTOR
    rig.net.version_down.add(EX_URL)
    rig.clock.advance(61 if timed_out else 0)
    rig.tick()
    ended = rig.state.get_run(run.run_id)
    job = rig.state.get_job("c/a", "only")
    assert ended is not None and (ended.status, ended.detail) == ("stopped", "executor-down")
    assert job is not None and job.attempts == 0 and job.status == "pending"
    assert rig.events("failed") == [] and len(rig.handles) == 1
    assert rig.state.idle_reason() == "executor-down"
    rig.net.version_down.clear()
    rig.tick()
    assert len(rig.handles) == 2 and rig.state.runs_with_status("running")[0].host == EXECUTOR


@pytest.mark.parametrize("timed_out", [False, True])
def test_check_9_the_same_end_with_the_executor_up_is_charged(
    tmp_path: Path, timed_out: bool
) -> None:
    rig = fleet_rig(
        tmp_path,
        net_of({}, {"only": GIB}),
        [("c/a", "only")],
        models=("only",),
        task_kw={"timeout_s": 60},
    )
    rig.next_codes = [None] if timed_out else [1]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    rig.clock.advance(61 if timed_out else 0)
    rig.tick()
    ended = rig.state.get_run(run.run_id)
    job = rig.state.get_job("c/a", "only")
    assert ended is not None and ended.status == ("timeout" if timed_out else "failed")
    assert ended.detail == "" and job is not None and job.attempts == 1


def test_check_9_a_run_that_ended_ok_is_read_even_if_the_executor_is_down_by_then(
    tmp_path: Path,
) -> None:
    rig = fleet_rig(tmp_path, net_of({}, {"only": GIB}), [("c/a", "only")], models=("only",))
    rig.next_codes = [0]
    rig.tick()
    rig.net.version_down.add(EX_URL)
    rig.tick()
    assert rig.job_status("a", "only") == "done"
    assert [r.status for r in rig.state.runs_with_status("done")] == ["done"]


def test_check_9_a_failing_workstation_run_is_charged_whatever_a_version_probe_would_say(
    tmp_path: Path,
) -> None:
    net = net_of({"m": GIB}, {})
    net.version_down.add(WS_URL)
    rig = fleet_rig(tmp_path, net, ["c/a"])
    rig.next_codes = [1]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == WORKSTATION
    rig.tick()
    job = rig.state.get_job("c/a", MODEL)
    assert [r.status for r in rig.state.runs_with_status("failed")] == ["failed"]
    assert job is not None and job.attempts == 1


def test_the_workstation_checks_gate_a_run_on_the_executor_too(tmp_path: Path) -> None:
    rig = fleet_rig(
        tmp_path,
        net_of({}, {"only": GIB}),
        [("c/a", "only")],
        models=("only",),
        host=host(module="580.159.03"),
    )
    rig.tick()
    assert rig.handles == [] and rig.state.idle_reason() == "driver-mismatch"


def test_three_pairs_with_three_models_start_in_one_tick_and_a_fourth_waits_for_a_slot(
    tmp_path: Path,
) -> None:
    models = ("m1", "m2", "m3", "m4")
    rig = Rig(tmp_path, [task(tmp_path, "a", models=models)], [("c/a", m) for m in models])
    rig.next_codes = [None]
    rig.tick()
    rows = rig.state.runs_with_status("running")
    assert [(r.model, r.host) for r in rows] == [(m, WORKSTATION) for m in models[:3]]
    starts = [e.detail.split() for e in reversed(rig.events("run-start"))]
    assert starts == [[r.run_id, r.model, r.host] for r in rows]
    assert rig.job_status("a", "m4") == "pending" and rig.state.idle_reason() is None
    rig.tick()
    assert len(rig.handles) == 3
    assert rig.state.idle_reason() is None and rig.events("idle") == []  # runs active: not idle
    rig.handles[0].codes = [0]
    rig.tick()
    assert [a.model for a in rig.d.active.values()] == ["m2", "m3", "m4"]
    assert rig.job_status("a", "m1") == "done"


def test_the_executor_is_preferred_to_the_workstation_when_the_model_fits_there(
    tmp_path: Path,
) -> None:
    rig = fleet_rig(tmp_path, net_of({MODEL: GIB}, {MODEL: GIB}), ["c/a"])
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert (run.model, run.host) == (MODEL, EXECUTOR)
    assert rig.events("run-start")[0].detail.split() == [run.run_id, MODEL, EXECUTOR]


def test_a_model_runs_once_per_host_so_a_second_pair_of_it_takes_the_workstation(
    tmp_path: Path,
) -> None:
    rig = fleet_rig(
        tmp_path, net_of({MODEL: GIB}, {MODEL: GIB}), ["c/a", "c/b", "c/c"], ids=("a", "b", "c")
    )
    rig.next_codes = [None]
    rig.tick()
    rows = rig.state.runs_with_status("running")
    assert [(r.ref, r.host) for r in rows] == [("c/a", EXECUTOR), ("c/b", WORKSTATION)]
    assert rig.job_status("c") == "pending"


def test_a_model_already_resident_on_the_workstation_goes_there(tmp_path: Path) -> None:
    net = net_of({MODEL: GIB}, {MODEL: GIB})
    net.ps[WS_URL] = {MODEL: (GIB, GIB)}
    rig = fleet_rig(tmp_path, net, ["c/a"])
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == WORKSTATION


def test_the_last_loaded_size_replaces_the_estimate_even_after_the_model_unloads(
    tmp_path: Path,
) -> None:
    net = net_of({MODEL: 5 * GIB}, {MODEL: 5 * GIB})
    net.ps[EX_URL] = {MODEL: (6 * GIB, 11 * GIB)}
    rig = fleet_rig(tmp_path, net, ["c/a"], ids=("a", "b"))
    rig.next_codes = [None]
    rig.tick()
    assert [r.host for r in rig.state.runs_with_status("running")] == [WORKSTATION]
    net.ps.clear()  # unloaded from the executor; the 11 GiB seen there is remembered
    rig.list_jobs(["c/a", "c/b"])
    rig.tick()
    assert len(rig.handles) == 1 and rig.job_status("b") == "pending"


def test_a_pair_listed_later_is_placed_next_to_a_run_already_in_flight(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m1")])
    rig.next_codes = [None]
    rig.tick()
    rig.list_jobs([("c/a", "m1"), ("c/a", "m2")])
    rig.tick()
    assert [r.model for r in rig.state.runs_with_status("running")] == ["m1", "m2"]


@pytest.mark.parametrize(
    ("pair_model", "listed"),
    [
        ("nomic-embed-text", "nomic-embed-text:latest"),
        ("devstral:24b", "devstral:24b"),
        ("hf.co/bartowski/x-GGUF:IQ4_XS", "hf.co/bartowski/x-GGUF:IQ4_XS"),
        ("registry.local:5000/foo", "registry.local:5000/foo:latest"),
    ],
)
def test_a_pair_places_on_the_host_that_lists_its_model_by_ollamas_name_and_runs_under_its_own(
    tmp_path: Path, pair_model: str, listed: str
) -> None:
    net = net_of({listed: GIB})
    rig = fleet_rig(tmp_path, net, [("c/a", pair_model)], models=(pair_model,))
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert (run.model, run.host) == (pair_model, WORKSTATION)
    assert rig.start_models == [pair_model]
    assert rig.job_status("a", pair_model) == "running"


def test_a_pair_prefers_the_host_where_ollama_lists_its_model_as_latest_in_ps(
    tmp_path: Path,
) -> None:
    net = net_of({"x:latest": GIB}, {"x:latest": GIB})
    net.ps[WS_URL] = {"x:latest": (GIB, GIB)}
    rig = fleet_rig(tmp_path, net, [("c/a", "x")], models=("x",))
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == WORKSTATION


def test_the_loaded_size_of_a_latest_model_replaces_the_estimate_of_its_untagged_pair(
    tmp_path: Path,
) -> None:
    net = net_of({"x:latest": 5 * GIB}, {"x:latest": 5 * GIB})
    net.ps[EX_URL] = {"x:latest": (6 * GIB, 11 * GIB)}  # 11 GiB loaded: over the 10 GiB card
    rig = fleet_rig(tmp_path, net, [("c/a", "x")], models=("x",))
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == WORKSTATION


def test_d023_one_run_of_an_untagged_model_per_host_among_pairs_placed_in_one_tick(
    tmp_path: Path,
) -> None:
    net = net_of({"x:latest": GIB}, {"x:latest": GIB})
    net.ps[EX_URL] = {"x:latest": (GIB, GIB)}  # loaded there, so only the held check keeps it away
    jobs = [("c/a", "x"), ("c/b", "x"), ("c/c", "x")]
    rig = fleet_rig(tmp_path, net, jobs, models=("x",), ids=("a", "b", "c"))
    rig.next_codes = [None]
    rig.tick()
    rows = rig.state.runs_with_status("running")
    assert [(r.ref, r.host) for r in rows] == [("c/a", EXECUTOR), ("c/b", WORKSTATION)]
    assert rig.job_status("c", "x") == "pending"


def test_d023_a_pair_listed_later_does_not_double_up_the_untagged_model_of_a_run_in_flight(
    tmp_path: Path,
) -> None:
    net = net_of({"x:latest": GIB}, {"x:latest": GIB})
    net.ps[EX_URL] = {"x:latest": (GIB, GIB)}  # loaded there, so only the held check keeps it away
    rig = fleet_rig(tmp_path, net, [("c/a", "x")], models=("x",), ids=("a", "b"))
    rig.next_codes = [None]
    rig.tick()
    rig.list_jobs([("c/a", "x"), ("c/b", "x")])
    rig.tick()
    rows = rig.state.runs_with_status("running")
    assert [(r.ref, r.host) for r in rows] == [("c/a", EXECUTOR), ("c/b", WORKSTATION)]


@pytest.mark.parametrize(("context", "starts"), [(32768, False), (8192, True)])
def test_the_kv_cache_is_counted_at_the_hosts_context(
    tmp_path: Path, context: int, starts: bool
) -> None:
    net = net_of({MODEL: GIB})
    net.ps[WS_URL] = {"other": (20 * GIB, 20 * GIB)}  # 4 GiB free
    net.info[LATEST] = {**INFO, "llama.block_count": 32, "llama.attention.head_count_kv": 8}
    net.info[LATEST]["llama.attention.key_length"] = 64  # 2 GiB a slot at 32768, 0.5 GiB at 8192
    hosts = {WORKSTATION: replace(WORKSTATION_ONLY[WORKSTATION], context=context)}
    rig = fleet_rig(tmp_path, net, ["c/a"], hosts=hosts)
    rig.tick()
    assert (len(rig.handles) == 1) is starts
    assert rig.state.idle_reason() == (None if starts else "no-fit")


@pytest.mark.parametrize(("version_down", "expected"), [(False, "no-fit"), (True, "executor-down")])
def test_a_pair_that_fits_nowhere_waits_with_the_first_unreadable_hosts_reason_else_no_fit(
    tmp_path: Path, version_down: bool, expected: str
) -> None:
    net = net_of({"big": 9 * GIB})
    net.ps[WS_URL] = {"other": (20 * GIB, 20 * GIB)}
    if version_down:
        net.version_down.add(EX_URL)
    rig = fleet_rig(tmp_path, net, [("c/a", "big")], models=("big",))
    rig.tick()
    assert rig.handles == [] and rig.state.idle_reason() == expected


def test_run_now_is_tried_first_and_starts_though_not_ready(tmp_path: Path) -> None:
    specs = [
        task(tmp_path, "a", models=("m1",)),
        task(tmp_path, "b", models=("m2",), ready=Ready(3600, ("./ready",))),
    ]
    rig = Rig(tmp_path, specs, [("c/a", "m1"), ("c/b", "m2")], hosts=ONE_AT_A_TIME)
    rig.reader_rc["./ready"] = 1
    rig.state.request_run_now("c/b", "m2")
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert (run.ref, run.model) == ("c/b", "m2")
    assert rig.run_in_calls == [] and rig.job_status("a", "m1") == "pending"
    assert rig.state.run_now_requested() is None and rig.events("refused") == []


@pytest.mark.parametrize(("version_down", "reason"), [(False, "no-fit"), (True, "executor-down")])
def test_a_run_now_that_fits_nowhere_is_refused_with_the_reason_and_the_rest_still_start(
    tmp_path: Path, version_down: bool, reason: str
) -> None:
    net = net_of({"m1": GIB}, {"m2": 9 * GIB})  # m2 is only on the executor and too big for it
    if version_down:
        net.version_down.add(EX_URL)
    rig = fleet_rig(
        tmp_path, net, [("c/a", "m1"), ("c/b", "m2")], ids=("a", "b"), models=("m1", "m2")
    )
    rig.state.request_run_now("c/b", "m2")
    rig.tick()
    [refused] = rig.events("refused")
    assert refused.ref == "c/b" and refused.detail == f"m2: run-now: {reason}"
    assert [r.model for r in rig.state.runs_with_status("running")] == ["m1"]
    assert rig.state.run_now_requested() is None
    rig.tick()
    assert len(rig.events("refused")) == 1


def test_graceful_stop_with_two_runs_starts_nothing_and_exits_only_after_both_end(
    tmp_path: Path,
) -> None:
    models = ("m1", "m2", "m3")
    hosts = {WORKSTATION: replace(WORKSTATION_ONLY[WORKSTATION], max_models=2)}
    spec = task(tmp_path, "a", models=models)
    rig = Rig(tmp_path, [spec], [("c/a", m) for m in models], hosts=hosts)
    rig.next_codes = [None]
    rig.tick()
    assert len(rig.handles) == 2
    rig.state.request_stop("graceful")
    rig.handles[0].codes = [0]
    assert rig.tick() is True  # the first run ended and freed a slot; the third pair still waits
    assert len(rig.handles) == 2 and rig.job_status("a", "m3") == "pending"
    assert rig.tick() is True
    rig.handles[1].codes = [0]
    assert rig.tick() is False
    assert rig.job_status("a", "m1") == "done" and rig.job_status("a", "m2") == "done"
    assert len(rig.handles) == 2


def test_stop_now_stops_every_run_and_returns_each_pair_to_pending(tmp_path: Path) -> None:
    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m1"), ("c/a", "m2")])
    rig.next_codes = [None]
    rig.tick()
    assert len(rig.handles) == 2
    rig.state.request_stop("now")
    assert rig.tick() is False
    assert all(h.stopped for h in rig.handles) and rig.d.active == {}
    assert [r.model for r in rig.state.runs_with_status("stopped")] == ["m1", "m2"]
    assert rig.state.jobs_with_status("running") == []
    for model in ("m1", "m2"):
        job = rig.state.get_job("c/a", model)
        assert job is not None and job.status == "pending" and job.attempts == 0


def test_end_to_end_over_two_fake_ollama_hosts_with_the_real_http_and_placement(
    tmp_path: Path,
) -> None:
    sizes = {"small": GIB, "big": 9 * GIB}
    shows = {"small": INFO, "big": INFO}
    with (
        FakeOllama("ok", models=sizes, show=shows) as ws,
        FakeOllama("ok", models=sizes, show=shows) as ex,
    ):
        hosts = {
            WORKSTATION: HostSpec(ws.url, 24, 3, 3, 32768),
            EXECUTOR: HostSpec(ex.url, 10, 3, 3, 8192),
        }
        spec = task(tmp_path, "a", models=("small", "big"))
        rig = Rig(
            tmp_path, [spec], [("c/a", "small"), ("c/a", "big")], hosts=hosts, http_json=http_json
        )
        rig.next_codes = [None]
        rig.tick()
        placed = {(r.model, r.host) for r in rig.state.runs_with_status("running")}
        assert placed == {("small", EXECUTOR), ("big", WORKSTATION)}
        assert "/api/version" in ex.requests and "/api/version" not in ws.requests
        assert {"/api/tags", "/api/ps", "/api/show"} <= set(ws.requests)
    assert rig.net.calls == []


def test_a_rerun_while_the_pair_runs_does_not_start_it_a_second_time_on_the_other_host(
    tmp_path: Path,
) -> None:
    rig = fleet_rig(tmp_path, net_of({MODEL: GIB}, {MODEL: GIB}), ["c/a"])
    rig.next_codes = [None]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == EXECUTOR
    rig.state.rerun("c/a", MODEL)
    rig.tick()
    rig.tick()
    assert [r.run_id for r in rig.state.runs_with_status("running")] == [run.run_id]
    assert len(rig.handles) == 1


def test_a_run_now_of_a_pair_that_runs_is_refused_even_after_a_rerun(tmp_path: Path) -> None:
    rig = fleet_rig(tmp_path, net_of({MODEL: GIB}, {MODEL: GIB}), ["c/a"])
    rig.next_codes = [None]
    rig.tick()
    rig.state.rerun("c/a", MODEL)
    rig.state.request_run_now("c/a", MODEL)
    rig.tick()
    [refused] = rig.events("refused")
    assert refused.detail.startswith(f"{MODEL}: run-now: not listed")
    assert len(rig.handles) == 1 and len(rig.state.runs_with_status("running")) == 1


def test_a_manifest_edit_while_the_pair_runs_neither_restarts_it_nor_resets_its_job(
    tmp_path: Path,
) -> None:
    rig = fleet_rig(tmp_path, net_of({MODEL: GIB}, {MODEL: GIB}), ["c/a"])
    rig.next_codes = [None]
    rig.tick()
    edited = task(tmp_path, "a", fingerprint="fp-a2")
    rig.specs["a"] = edited
    rig.state.record_contract_pass(edited.ref, edited.fingerprint, T0)
    rig.tick()
    job = rig.state.get_job("c/a", MODEL)
    assert len(rig.handles) == 1 and len(rig.state.runs_with_status("running")) == 1
    assert job is not None and (job.status, job.fingerprint) == ("running", "fp-a")
    rig.handles[0].codes = [0]
    rig.tick()  # the old run ends and is read; the new fingerprint then starts, once
    assert [r.fingerprint for r in rig.state.runs_with_status("done")] == ["fp-a"]
    assert [r.fingerprint for r in rig.state.runs_with_status("running")] == ["fp-a2"]
    rig.tick()
    rig.tick()
    assert len(rig.handles) == 2


def test_a_run_on_one_host_does_not_block_the_same_model_on_the_other_host(tmp_path: Path) -> None:
    rig = fleet_rig(tmp_path, net_of({MODEL: GIB}, {MODEL: GIB}), ["c/a"], ids=("a", "b"))
    rig.next_codes = [None]
    rig.tick()
    assert [r.host for r in rig.state.runs_with_status("running")] == [EXECUTOR]
    rig.list_jobs(["c/a", "c/b"])
    rig.tick()
    rows = rig.state.runs_with_status("running")
    assert [(r.ref, r.host) for r in rows] == [("c/a", EXECUTOR), ("c/b", WORKSTATION)]


def test_a_run_on_the_executor_does_not_use_up_a_workstation_slot(tmp_path: Path) -> None:
    hosts = {WORKSTATION: replace(FLEET[WORKSTATION], max_models=1), EXECUTOR: FLEET[EXECUTOR]}
    spec = task(tmp_path, "a", models=("m1", "m2"))
    net = net_of({"m1": GIB, "m2": GIB}, {"m1": GIB})
    rig = Rig(tmp_path, [spec], [("c/a", "m1")], hosts=hosts, net=net)
    rig.next_codes = [None]
    rig.tick()
    rig.list_jobs([("c/a", "m1"), ("c/a", "m2")])
    rig.tick()
    rows = rig.state.runs_with_status("running")
    assert [(r.model, r.host) for r in rows] == [("m1", EXECUTOR), ("m2", WORKSTATION)]


def test_a_models_loaded_size_on_the_executor_does_not_leak_to_the_workstation(
    tmp_path: Path,
) -> None:
    net = net_of({MODEL: GIB}, {MODEL: GIB})
    net.ps[EX_URL] = {MODEL: (6 * GIB, 11 * GIB)}
    net.ps[WS_URL] = {"other": (20 * GIB, 20 * GIB)}  # 4 GiB free: the 2.2 GiB estimate fits
    rig = fleet_rig(tmp_path, net, ["c/a"])
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == WORKSTATION


@pytest.mark.parametrize(("context", "starts"), [(8192, True), (16384, False)])
def test_the_kv_cache_is_counted_at_the_executors_own_context(
    tmp_path: Path, context: int, starts: bool
) -> None:
    net = net_of({}, {MODEL: GIB // 2})
    net.ps[EX_URL] = {"other": (7 * GIB, 7 * GIB)}  # 3 GiB free of 10
    info = {**INFO, "llama.block_count": 32, "llama.attention.head_count_kv": 8}
    net.info[LATEST] = {**info, "llama.attention.key_length": 128}  # 1 GiB per slot at 8192
    hosts = {**WORKSTATION_ONLY, EXECUTOR: replace(FLEET[EXECUTOR], context=context, parallel=1)}
    rig = fleet_rig(tmp_path, net, ["c/a"], hosts=hosts)
    rig.tick()
    assert (len(rig.handles) == 1) is starts
    assert rig.state.idle_reason() == (None if starts else "no-fit")


@pytest.mark.parametrize("mode", ["now", "graceful"])
def test_a_stop_reaches_a_run_on_the_executor(tmp_path: Path, mode: str) -> None:
    rig = fleet_rig(tmp_path, net_of({}, {"only": GIB}), [("c/a", "only")], models=("only",))
    rig.next_codes = [None]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == EXECUTOR
    rig.state.request_stop("now" if mode == "now" else "graceful")
    if mode == "now":
        assert rig.tick() is False
        ended = rig.state.get_run(run.run_id)
        job = rig.state.get_job("c/a", "only")
        assert rig.handles[0].stopped and ended is not None and ended.status == "stopped"
        assert job is not None and job.status == "pending" and job.attempts == 0
        return
    assert rig.tick() is True and not rig.handles[0].stopped
    rig.handles[0].codes = [0]
    assert rig.tick() is False
    assert rig.job_status("a", "only") == "done"


def test_a_run_whose_host_left_the_policy_ends_as_a_normal_charged_failure(tmp_path: Path) -> None:
    rig = fleet_rig(tmp_path, net_of({}, {"only": GIB}), [("c/a", "only")], models=("only",))
    rig.next_codes = [1]
    rig.tick()
    [run] = rig.state.runs_with_status("running")
    assert run.host == EXECUTOR
    rig.net.version_down.add(EX_URL)  # a listed executor that probes down would release the run
    rig.hosts = dict(WORKSTATION_ONLY)
    rig.list_jobs(rig.pairs)
    rig.tick()
    ended = rig.state.get_run(run.run_id)
    job = rig.state.get_job("c/a", "only")
    assert ended is not None and (ended.status, ended.detail) == ("failed", "")
    assert job is not None and job.attempts == 1 and rig.events("failed") == []


def test_the_global_gate_asks_the_endpoint_jobs_use(tmp_path: Path) -> None:
    urls: list[str] = []
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host(seen=urls))
    rig.tick()
    assert urls == [rig.deps.ollama_url + "/api/tags"]
    assert WS_URL not in "".join(urls)


def test_a_refusing_job_endpoint_starts_nothing_while_the_workstation_ollama_answers(
    tmp_path: Path,
) -> None:
    refused: set[str] = set()
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"], host=host(refused=refused))
    refused.add(rig.deps.ollama_url)
    rig.tick()
    job = rig.state.get_job("c/a", MODEL)
    assert rig.handles == [] and rig.state.idle_reason() == "ollama-down"
    assert job is not None and (job.status, job.attempts) == ("pending", 0)
    refused.clear()
    rig.tick()
    assert len(rig.handles) == 1


def test_recover_crashes_every_running_row_and_charges_each_on_its_second_crash(
    tmp_path: Path,
) -> None:
    net = net_of({"m1": GIB}, {"m2": GIB})
    rig = fleet_rig(
        tmp_path, net, [("c/a", "m1"), ("c/b", "m2")], ids=("a", "b"), models=("m1", "m2")
    )
    rig.next_codes = [None]
    rig.tick()
    assert {r.host for r in rig.state.runs_with_status("running")} == {WORKSTATION, EXECUTOR}
    for crashes, (attempts, streak) in enumerate([(0, 1), (1, 2)], start=1):
        rig.d.recover(rig.clock())
        assert len(rig.state.runs_with_status("crashed")) == 2 * crashes
        assert rig.state.runs_with_status("running") == []
        assert rig.events("recover")[0].detail == "c/a m1,c/b m2"
        for ref, model in (("c/a", "m1"), ("c/b", "m2")):
            job = rig.state.get_job(ref, model)
            assert job is not None and job.status == "pending"
            assert (job.attempts, job.crash_streak) == (attempts, streak)
        rig.d = Dispatcher(rig.deps, rig.state, PolicyCache(rig.policy, lambda _m: None))
        rig.tick()
        assert len(rig.state.runs_with_status("running")) == 2


@pytest.mark.parametrize(
    ("order", "expected"),
    [((WORKSTATION, EXECUTOR), "ollama-down"), ((EXECUTOR, WORKSTATION), "executor-down")],
)
def test_the_first_unhealthy_host_in_policy_order_names_the_idle_reason(
    tmp_path: Path, order: tuple[str, str], expected: str
) -> None:
    net = net_of({MODEL: GIB}, {MODEL: GIB})
    net.link_down.add(WS_URL)
    net.version_down.add(EX_URL)
    rig = fleet_rig(tmp_path, net, ["c/a"], hosts={name: FLEET[name] for name in order})
    rig.tick()
    assert rig.handles == [] and rig.state.idle_reason() == expected


def test_two_runs_ending_in_one_tick_are_both_finished_and_both_slots_refilled(
    tmp_path: Path,
) -> None:
    models = ("m1", "m2", "m3", "m4")
    hosts = {WORKSTATION: replace(FLEET[WORKSTATION], max_models=2)}
    rig = Rig(
        tmp_path, [task(tmp_path, "a", models=models)], [("c/a", m) for m in models], hosts=hosts
    )
    rig.next_codes = [None]
    rig.tick()
    assert len(rig.handles) == 2
    for handle in rig.handles:
        handle.codes = [0]
    rig.tick()
    assert [r.model for r in rig.state.runs_with_status("done")] == ["m1", "m2"]
    assert [r.model for r in rig.state.runs_with_status("running")] == ["m3", "m4"]
    assert len(rig.handles) == 4


def test_a_start_that_fails_does_not_occupy_the_slot_for_the_next_pair(tmp_path: Path) -> None:
    def start(
        spec: TaskSpec, run_dir: RunDir, run_id: str, url: str, model: str, env: Mapping[str, str]
    ) -> FakeHandle:
        if model == "m2":
            raise FileNotFoundError("./job")
        return FakeHandle([None])

    spec = task(tmp_path, "a", models=("m1", "m2"))
    rig = Rig(tmp_path, [spec], [("c/a", "m2"), ("c/a", "m1")], hosts=ONE_AT_A_TIME, start=start)
    rig.tick()
    broken = rig.state.get_job("c/a", "m2")
    assert broken is not None and broken.attempts == 1 and broken.status == "pending"
    assert [a.model for a in rig.d.active.values()] == ["m1"]


def test_a_start_failure_on_the_executor_is_charged_even_when_its_probe_would_fail(
    tmp_path: Path,
) -> None:
    net = net_of({}, {"only": GIB})

    def start(*_args: object) -> FakeHandle:
        net.version_down.add(EX_URL)  # the link drops just as the job fails to start
        raise FileNotFoundError("./job")

    rig = fleet_rig(tmp_path, net, [("c/a", "only")], models=("only",), start=start)
    rig.tick()
    run = rig.state.last_run("c/a", "only")
    job = rig.state.get_job("c/a", "only")
    assert run is not None and (run.host, run.status, run.exit_code) == (EXECUTOR, "failed", 127)
    assert run.detail == "" and job is not None and job.attempts == 1


def test_a_run_stays_active_when_finishing_it_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.next_codes = [1]
    rig.tick()

    def explode(*_args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(rig.d, "_finish", explode)
    with pytest.raises(RuntimeError):
        rig.tick()
    assert len(rig.d.active) == 1 and len(rig.state.runs_with_status("running")) == 1
    monkeypatch.undo()
    rig.tick()
    assert [r.status for r in rig.state.runs_with_status("failed")] == ["failed"]


def test_a_run_stays_active_when_the_executor_probe_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = fleet_rig(tmp_path, net_of({}, {"only": GIB}), [("c/a", "only")], models=("only",))
    rig.next_codes = [1]
    rig.tick()

    def explode(*_args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("pastore.run_end.executor_reason", explode)
    with pytest.raises(RuntimeError):
        rig.tick()
    assert len(rig.d.active) == 1 and len(rig.state.runs_with_status("running")) == 1


def test_a_run_stays_active_when_stopping_it_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, [task(tmp_path, "a")], ["c/a"])
    rig.next_codes = [None]
    rig.tick()

    def explode(_run: object, wait_s: float = 30.0) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(rig.handles[0], "stop", explode)
    rig.state.request_stop("now")
    with pytest.raises(RuntimeError):
        rig.tick()
    assert len(rig.d.active) == 1 and rig.state.stop_requested() == "now"
