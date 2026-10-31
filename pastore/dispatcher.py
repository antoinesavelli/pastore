"""The fleet dispatcher: each tick heartbeats, honours stop and run-now requests, watches every running job, places the ready ones, and records any foreign workload disruption that overlaps an executor run (R35); a tick is bounded at 60 s (D-050 (6)).

Depends on: policy (PolicyCache), manifest, state (State, rows, lock), checks (Host), hosts (read_host), placement (place), runner, run_end (Active, charge_crash, gateway_outage, oom_killed), tick_bound, contract (Started), spec, workload_watch.
"""

import math
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from subprocess import CompletedProcess
from types import FrameType

from pastore import workload_watch
from pastore.checks import Host, host_idle_reason, stopped
from pastore.contract import Started
from pastore.hosts import HostState, HttpJson, executor_reason, read_host
from pastore.manifest import ManifestError
from pastore.placement import HostLoad, ModelFacts, kv_bytes_per_slot, place
from pastore.policy import Policy, PolicyCache
from pastore.run_end import MAX_ATTEMPTS, Active, NoHandle, charge_crash, gateway_outage, oom_killed
from pastore.runner import RunDir, fresh_run_dir, stop_leftovers
from pastore.spec import (
    BASE_ENV_KEYS,
    RUN_DIR_ENV,
    WORKSTATION,
    RunReader,
    TaskSpec,
    canonical_model,
    idle_reason,
)
from pastore.state import DispatcherLock, JobRow, RunRow, State, StatePaths, open_state
from pastore.tick_bound import TICK_BOUND_S, CriticalSections, bounded

STARVED_AFTER = timedelta(hours=24)
TICK_S = 30.0

RunIn = Callable[[Sequence[str], Path, Mapping[str, str]], CompletedProcess[str]]
LoadManifest = Callable[[Path, str], dict[str, TaskSpec]]
StartRun = Callable[[TaskSpec, RunDir, str, str, str, Mapping[str, str]], Started]
Unit = tuple[TaskSpec, str]


@dataclass(frozen=True)
class Deps:
    """Everything the dispatcher touches outside its own logic, injected."""

    paths: StatePaths
    load_manifest: LoadManifest
    host: Host
    run_in: RunIn
    start: StartRun
    base_env: Mapping[str, str]
    ollama_url: str
    http_json: HttpJson


def iso(now: datetime) -> str:
    """UTC ISO-8601 text, the timestamp form stored in state."""
    return now.astimezone(UTC).isoformat()


class Dispatcher(CriticalSections):
    """One dispatcher; `serve` holds the lock and owns the loop and the clock."""

    def __init__(self, deps: Deps, state: State, policy: PolicyCache) -> None:
        self.deps = deps
        self.state = state
        self.policy = policy
        self.active: dict[str, Active] = {}
        self.loaded_seen: dict[tuple[str, str], int] = {}
        self.draining = False
        self.terminated = False
        self._refused: set[tuple[str, str]] = set()

    def recover(self, now: datetime) -> None:
        """Stop leftover scopes, return running rows to pending, charge only a second consecutive crash."""
        stop_leftovers(self.deps.host.run)
        crashed: set[tuple[str, str]] = set()
        for run in self.state.runs_with_status("running"):
            crashed.add((run.ref, run.model))
            self.state.update_run(replace(run, status="crashed", ended_at=iso(now)))
            job = self.state.get_job(run.ref, run.model)
            if job is not None:
                self.state.upsert_job(charge_crash(job))
        for job in self.state.jobs_with_status("running"):
            if (job.ref, job.model) not in crashed:
                self.state.upsert_job(replace(job, status="pending"))
        self.state.reset_crash_streaks(crashed)
        detail = ",".join(f"{ref} {model}" for ref, model in sorted(crashed))
        self.state.record_event(now, "recover", detail=detail)

    def tick(self, now: datetime) -> bool:
        """Run one decision step; False means the dispatcher should exit."""
        self._late = None
        self.state.heartbeat(now)
        policy = self.policy.current()
        if not self._handle_stop_request(now):
            return False
        self._watch_all(now, policy)
        if self.draining:
            if not self.active:
                return False
        else:
            self._dispatch(now, policy)
        self._watch_workload(now)
        return True

    def _watch_workload(self, now: datetime) -> None:
        """R35: read the workload's journal against the executor runs and record a new overlap; it never acts on a run."""
        spans = [
            workload_watch.Span(
                row.run_id,
                row.ref,
                datetime.fromisoformat(row.started_at),
                None if row.ended_at is None else datetime.fromisoformat(row.ended_at),
            )
            for row in self.state.executor_runs(now - workload_watch.KEEP_AFTER_END)
        ]
        found = workload_watch.watch(now, spans, workload_watch.journal_reader(self.deps.host.run))
        self.state.set_workload_watch(found.status)
        for overlap in found.overlaps:
            if not self.state.has_event("workload-overlap", overlap.key()):
                self.state.record_event(
                    now,
                    "workload-overlap",
                    overlap.span.ref,
                    f"{overlap.key()} {overlap.detail()}",
                )
                print(f"<4>pastore: workload-overlap {overlap.detail()}", file=sys.stderr)

    def _handle_stop_request(self, now: datetime) -> bool:
        mode = "sigterm" if self.terminated else self.state.stop_requested()
        if mode is None:
            return True
        with self._critical():
            # Logged first: on shutdown, systemd may SIGKILL us while the jobs are being stopped.
            self.state.record_event(now, "dispatcher-stop", detail=mode)
            if mode != "graceful":
                self._stop_all(now)
            self.draining = True
            self.state.clear_stop()
        return bool(self.active)

    def _stop_all(self, now: datetime) -> None:
        for run_id, active in list(self.active.items()):
            code = active.handle.stop(self.deps.host.run)
            self._end_run(active, now, "stopped", code)
            self.state.upsert_job(replace(self._job(active.spec, active.model), status="pending"))
            del self.active[run_id]

    def _watch_all(self, now: datetime, policy: Policy) -> None:
        for run_id, active in list(self.active.items()):
            with self._critical():
                outcome = self._poll(now, active)
                if outcome is None:
                    continue
                code, timed_out = outcome
                lost = self._outage(active, policy, code, timed_out)
                if lost is None:
                    self._finish(active, now, code, timed_out)
                else:
                    self._release(active, now, code, lost)
                del self.active[run_id]

    def _poll(self, now: datetime, active: Active) -> tuple[int, bool] | None:
        """The exit code and whether the run timed out, or None while it goes on."""
        code = active.handle.poll()
        limit = timedelta(seconds=active.spec.timeout_s)
        timed_out = code is None and now - active.started_at > limit
        if timed_out:
            code = active.handle.stop(self.deps.host.run)
        return None if code is None else (code, timed_out)

    def _outage(self, active: Active, policy: Policy, code: int, timed_out: bool) -> str | None:
        """Check 9 and D-050 (4): the idle reason of the outage that cut a non-ok end short, or None when the run is charged."""
        if not timed_out and code in active.spec.ok_exit:
            return None
        host = policy.hosts.get(active.host)
        if host is not None and host.name != WORKSTATION:
            if executor_reason(host, self.deps.http_json) is not None:
                return idle_reason("executor-down")
        status_url = self.deps.ollama_url + "/pastore/status"
        down = gateway_outage(self.deps.http_json, status_url, active.started_at)
        return idle_reason("ollama-down") if down else None

    def _release(self, active: Active, now: datetime, code: int, reason: str) -> None:
        """End a run an outage cut short: stopped, back to pending, no attempt charged."""
        self._end_run(active, now, "stopped", code, reason)
        self.state.upsert_job(replace(self._job(active.spec, active.model), status="pending"))

    def _finish(self, active: Active, now: datetime, code: int, timed_out: bool) -> None:
        spec = active.spec
        job = replace(self._job(spec, active.model), crash_streak=0)
        if not timed_out and code in spec.ok_exit:
            self._read(active, now, job, code)
            return
        detail = "oom" if not timed_out and oom_killed(self.deps.host.run, active) else ""
        self._end_run(active, now, "timeout" if timed_out else "failed", code, detail)
        attempts = job.attempts + 1
        status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
        self.state.upsert_job(replace(job, attempts=attempts, status=status))
        if status == "failed":
            self.state.record_event(now, "failed", spec.ref, f"{active.model}: {attempts} attempts")

    def _read(self, active: Active, now: datetime, job: JobRow, code: int) -> None:
        spec = active.spec
        env = {**self._env(spec), RUN_DIR_ENV: str(active.run_dir.out)}
        reader_rc = self.deps.run_in(spec.reader.argv, spec.cwd, env).returncode
        if reader_rc != 0 or not isinstance(spec.reader, RunReader):
            status = "reader-failed" if reader_rc != 0 else "awaiting-review"
            self._end_run(active, now, status, code)
            self.state.upsert_job(replace(job, attempts=0, status="blocked"))
            detail = f"{active.model}: reader exit {reader_rc}" if reader_rc != 0 else active.run_id
            self.state.record_event(now, status, spec.ref, detail)
            return
        self._end_run(active, now, "done", code)
        self.state.upsert_job(
            replace(
                job,
                attempts=0,
                status="done" if spec.ready is None else "pending",
                ready_since=None,
                last_success_at=iso(now),
                starved_flagged=0,
            )
        )
        self.state.record_event(now, "done", spec.ref, active.run_id)

    def _end_run(
        self, active: Active, now: datetime, status: str, code: int | None, detail: str = ""
    ) -> None:
        run = self.state.get_run(active.run_id)
        if run is None:
            raise RuntimeError(f"run {active.run_id} has no state row")
        ended = replace(run, status=status, ended_at=iso(now), exit_code=code, detail=detail)
        self.state.update_run(ended)
        text = f"{active.run_id} {status} {code}" + (f" {detail}" if detail else "")
        self.state.record_event(now, "run-end", active.spec.ref, text)

    def _env(self, spec: TaskSpec) -> dict[str, str]:
        base = {k: self.deps.base_env[k] for k in BASE_ENV_KEYS if k in self.deps.base_env}
        return {**base, **spec.env}

    def _dispatch(self, now: datetime, policy: Policy) -> None:
        units = self._not_running(self._listed(now, policy))
        forced = self._run_now(now, units)
        ready = [] if forced is None else [forced]
        ready += [u for u in units if u is not forced and self._is_ready(now, *u)]
        reason = self._gate(ready)
        placed = 0
        if reason is None:
            states = self._read_hosts(policy, ready)
            reason = _unplaced_reason(states)
            placed = self._place_all(now, self._loads(states), ready, forced, reason)
        elif forced is not None:
            self._refuse_run_now(now, forced[0].ref, forced[1], reason)
        if placed or self.active:
            self.state.clear_idle_reason()
        else:
            self.state.set_idle_reason(now, reason)

    def _not_running(self, units: list[Unit]) -> list[Unit]:
        """Drop the pairs with a run in flight, so they are never ready-checked, run-now'd or started twice."""
        running = {(a.spec.ref, a.model) for a in self.active.values()}
        return [u for u in units if (u[0].ref, u[1]) not in running]

    def _run_now(self, now: datetime, units: list[Unit]) -> Unit | None:
        """The requested pair if it is listed and pending; the request stays until it is acted on, so an overrun cannot lose it."""
        key = self.state.run_now_requested()
        if key is None:
            return None
        match = [u for u in units if (u[0].ref, u[1]) == key and self._job(*u).status == "pending"]
        if not match:
            self._refuse_run_now(
                now, *key, "not listed, no contract pass, or not pending (use rerun first)"
            )
        return match[0] if match else None

    def _refuse_run_now(self, now: datetime, ref: str, model: str, reason: str) -> None:
        with self._critical():
            self.state.record_event(now, "refused", ref, f"{model}: run-now: {reason}")
            self.state.clear_run_now((ref, model))

    def _gate(self, ready: list[Unit]) -> str | None:
        """The idle reason that keeps every ready unit waiting, or None when placement may go on."""
        if stopped(self.deps.paths.stop_file):
            return idle_reason("stopped")
        if not ready:
            return idle_reason("no-ready-task")
        return host_idle_reason(self.deps.host, self.deps.ollama_url)

    def _read_hosts(self, policy: Policy, ready: list[Unit]) -> list[HostState]:
        """Every policy host's facts for the ready units' models and the held ones; refreshes `loaded_seen`."""
        wanted = [model for _, model in ready] + [a.model for a in self.active.values()]
        states = [read_host(host, self.deps.http_json, wanted) for host in policy.hosts.values()]
        for state in states:
            for model, size in state.loaded.items():
                self.loaded_seen[(state.host.name, model)] = size
        return states

    def _loads(self, states: list[HostState]) -> list[HostLoad]:
        """The healthy hosts as placement sees them, one `held` entry per run active there."""
        return [
            HostLoad(
                state.host,
                {
                    model: ModelFacts(
                        size,
                        _kv_per_slot(state, model),
                        self.loaded_seen.get((state.host.name, model)),
                    )
                    for model, size in state.sizes.items()
                },
                state.resident,
                tuple(
                    canonical_model(a.model)
                    for a in self.active.values()
                    if a.host == state.host.name
                ),
            )
            for state in states
            if state.reason is None
        ]

    def _place_all(
        self,
        now: datetime,
        loads: list[HostLoad],
        ready: list[Unit],
        forced: Unit | None,
        reason: str,
    ) -> int:
        """Start each ready unit where it fits, in order; the number of starts attempted."""
        attempted = 0
        for unit in ready:
            model = canonical_model(unit[1])
            host = place(model, loads)
            if host is None:
                if unit is forced:
                    self._refuse_run_now(now, unit[0].ref, unit[1], reason)
                continue
            attempted += 1
            with self._critical():
                started = self._start(now, *unit, host)
                if unit is forced:
                    self.state.clear_run_now((unit[0].ref, unit[1]))
            if started:
                _hold(loads, host, model)
        return attempted

    def _listed(self, now: datetime, policy: Policy) -> list[Unit]:
        units = []
        manifests: dict[str, dict[str, TaskSpec]] = {}
        for pair in policy.jobs:
            consumer, task_id = pair.task.split("/", 1)
            try:
                if consumer not in manifests:
                    path = policy.consumers[consumer]
                    manifests[consumer] = self.deps.load_manifest(path, consumer)
                spec = manifests[consumer][task_id]
            except (ManifestError, OSError, KeyError) as exc:
                self._refuse(now, pair.task, f"manifest: {exc}")
                continue
            if pair.model not in spec.models:
                self._refuse(
                    now, pair.task, f"model {pair.model!r} is not in the manifest's models"
                )
                continue
            if not self.state.has_contract_pass(pair.task, spec.fingerprint):
                detail = f"no contract pass for fingerprint {spec.fingerprint[:12]}"
                self._refuse(now, pair.task, detail)
                continue
            units.append((spec, pair.model))
        return units

    def _refuse(self, now: datetime, ref: str, detail: str) -> None:
        if (ref, detail) not in self._refused:
            self.state.record_event(now, "refused", ref, detail)
            self._refused.add((ref, detail))

    def _job(self, spec: TaskSpec, model: str) -> JobRow:
        job = self.state.get_job(spec.ref, model)
        if job is None or job.fingerprint != spec.fingerprint:
            job = JobRow(spec.ref, model, spec.fingerprint, "pending")
            self.state.upsert_job(job)
        if spec.ready is not None and job.status == "done":
            job = replace(job, status="pending", ready_since=None)
            self.state.upsert_job(job)
        return job

    def _is_ready(self, now: datetime, spec: TaskSpec, model: str) -> bool:
        job = self._job(spec, model)
        if job.status != "pending":
            return False
        if job.ready_since is None:
            job = self._mark_ready(job, now) if spec.ready is None else self._check(job, now, spec)
        self._flag_starved(job, now)
        return job.ready_since is not None

    def _mark_ready(self, job: JobRow, now: datetime) -> JobRow:
        job = replace(job, ready_since=iso(now))
        self.state.upsert_job(job)
        return job

    def _check(self, job: JobRow, now: datetime, spec: TaskSpec) -> JobRow:
        if spec.ready is None:
            raise RuntimeError(f"{spec.ref} has no ready check")
        every = timedelta(seconds=spec.ready.every_s)
        if (
            job.last_check_at is not None
            and now - datetime.fromisoformat(job.last_check_at) < every
        ):
            return job
        rc = self.deps.run_in(spec.ready.check, spec.cwd, self._env(spec)).returncode
        job = replace(job, last_check_at=iso(now), ready_since=iso(now) if rc == 0 else None)
        if rc not in (0, 1):
            self._refuse(now, spec.ref, f"ready-check-exit:{rc}")
        self.state.upsert_job(job)
        return job

    def _flag_starved(self, job: JobRow, now: datetime) -> None:
        if job.ready_since is None or job.starved_flagged:
            return
        if now - datetime.fromisoformat(job.ready_since) > STARVED_AFTER:
            detail = f"{job.model}: ready since {job.ready_since}"
            self.state.record_event(now, "starved", job.ref, detail)
            self.state.upsert_job(replace(job, starved_flagged=1))

    def _start(self, now: datetime, spec: TaskSpec, model: str, host: str) -> bool:
        with self._critical():
            run_id, run_dir = fresh_run_dir(self.deps.paths.runs_dir, now)
            run = RunRow(run_id, spec.ref, model, host, spec.fingerprint, "running", iso(now))
            self.state.insert_run(run)
            self.state.upsert_job(replace(self._job(spec, model), status="running"))
            self.state.record_event(now, "run-start", spec.ref, f"{run_id} {model} {host}")
            try:
                handle = self.deps.start(
                    spec, run_dir, run_id, self.deps.ollama_url, model, self.deps.base_env
                )
            except OSError as exc:
                failed = Active(spec, model, host, run_id, run_dir, now, NoHandle())
                self._finish(failed, now, 127, False)
                detail = f"{model}: could not start: {exc}"
                self.state.record_event(now, "refused", spec.ref, detail)
                return False
            self.active[run_id] = Active(spec, model, host, run_id, run_dir, now, handle)
            return True


def _kv_per_slot(state: HostState, model: str) -> int | None:
    shape = state.kv.get(model)
    if shape is None:
        return None
    return kv_bytes_per_slot(shape.layers, shape.kv_heads, shape.head_dim, state.host.context)


def _unplaced_reason(states: list[HostState]) -> str:
    """Why a unit that fits nowhere waits: the first unreadable host's reason, else no-fit."""
    return next((s.reason for s in states if s.reason is not None), idle_reason("no-fit"))


def _hold(loads: list[HostLoad], host: str, model: str) -> None:
    """Count a run just started on `host` as held there, for the units placed after it this tick."""
    index = next(i for i, load in enumerate(loads) if load.host.name == host)
    loads[index] = replace(loads[index], held=(*loads[index].held, model))


def serve(
    deps: Deps,
    policy_path: Path,
    *,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    tick_s: float = TICK_S,
    tick_bound_s: float = TICK_BOUND_S,
) -> int:
    """Hold the lock, recover, and tick until a stop request, each tick bounded by a SIGALRM timer; the only loop in pastore."""
    with DispatcherLock(deps.paths.lock), closing(open_state(deps.paths)) as state:

        def log(message: str) -> None:
            print(message, file=sys.stderr)
            state.record_event(clock(), "policy-error", detail=message)

        dispatcher = Dispatcher(deps, state, PolicyCache(policy_path, log))
        dispatcher.policy.current()

        def on_sigterm(_signum: int, _frame: FrameType | None) -> None:
            dispatcher.terminated = True  # `_nap` wakes within 1 s; a handler must not touch sqlite

        def on_sigalrm(_signum: int, _frame: FrameType | None) -> None:
            dispatcher.overrun(tick_bound_s)

        previous = signal.signal(signal.SIGTERM, on_sigterm)
        previous_alarm = signal.signal(signal.SIGALRM, on_sigalrm)
        try:
            stale = [state.stop_requested(), state.run_now_requested()]
            state.clear_stop()
            state.clear_run_now()
            dispatcher.recover(clock())
            dropped = [r for r in stale if r is not None]
            detail = f"dropped stale {dropped}" if dropped else ""
            state.record_event(clock(), "dispatcher-start", detail=detail)
            while _guarded_tick(dispatcher, state, clock, tick_bound_s):
                _nap(dispatcher, sleep, tick_s)
        finally:
            signal.signal(signal.SIGTERM, previous)
            signal.signal(signal.SIGALRM, previous_alarm)
    return 0


def _nap(dispatcher: Dispatcher, sleep: Callable[[float], None], tick_s: float) -> None:
    """Sleep one tick in 1 s steps, so a SIGTERM is acted on before a shutdown's SIGKILL."""
    if dispatcher.terminated:
        # A tick after SIGTERM failed (a clean one would have ended the loop): retry once a second, never spin.
        sleep(1.0)
        return
    for _ in range(math.ceil(tick_s)):
        if dispatcher.terminated:
            return
        sleep(1.0)


def _guarded_tick(
    dispatcher: Dispatcher, state: State, clock: Callable[[], datetime], tick_bound_s: float = 0.0
) -> bool:
    """One tick under a one-shot SIGALRM timer (0 arms none); an error or a TickOverrun is logged as a tick-error event and the loop goes on."""
    try:
        with bounded(tick_bound_s):
            return dispatcher.tick(clock())
    except Exception as exc:  # noqa: BLE001 — the night must not end on one bad tick; the error is recorded
        message = f"{type(exc).__name__}: {exc}"
        print(f"tick-error: {message}", file=sys.stderr)
        try:
            state.record_event(clock(), "tick-error", detail=message)
        except Exception as inner:  # noqa: BLE001 — state itself failing must still not kill the loop
            print(f"tick-error not recorded: {inner}", file=sys.stderr)
        return True
