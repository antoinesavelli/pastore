"""The dispatcher: each tick heartbeats, honours stop and run-now requests, watches the running job, or starts the next.

Depends on: policy (PolicyCache), manifest, state (State, rows, lock), window, checks (Host), runner, contract (Started), spec.
"""

import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from subprocess import CompletedProcess
from types import FrameType

from fucone.checks import Host, host_idle_reason, stopped
from fucone.contract import Started
from fucone.manifest import ManifestError
from fucone.policy import Policy, PolicyCache
from fucone.runner import BASE_ENV_KEYS, RunDir, fresh_run_dir, stop_leftovers
from fucone.spec import RunReader, TaskSpec, idle_reason
from fucone.state import DispatcherLock, JobRow, RunRow, State, StatePaths, open_state
from fucone.window import in_window

MAX_ATTEMPTS = 3
STARVED_AFTER = timedelta(hours=24)
TICK_S = 30.0

RunIn = Callable[[Sequence[str], Path, Mapping[str, str]], CompletedProcess[str]]
LoadManifest = Callable[[Path, str], dict[str, TaskSpec]]
StartRun = Callable[[TaskSpec, RunDir, str, str, Mapping[str, str]], Started]


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
    local_tz: tzinfo | None


@dataclass(frozen=True)
class Active:
    """The one running job."""

    spec: TaskSpec
    run_id: str
    run_dir: RunDir
    started_at: datetime
    handle: Started


def iso(now: datetime) -> str:
    """UTC ISO-8601 text, the timestamp form stored in state."""
    return now.astimezone(UTC).isoformat()


class Dispatcher:
    """One dispatcher; `serve` holds the lock and owns the loop and the clock."""

    def __init__(self, deps: Deps, state: State, policy: PolicyCache) -> None:
        self.deps = deps
        self.state = state
        self.policy = policy
        self.active: Active | None = None
        self.draining = False
        self.terminated = False
        self._refused: set[tuple[str, str]] = set()

    def recover(self, now: datetime) -> None:
        """Stop leftover scopes, return running rows to pending, charge only a second consecutive crash."""
        stop_leftovers(self.deps.host.run)
        crashed = set()
        for run in self.state.runs_with_status("running"):
            crashed.add(run.ref)
            self.state.update_run(replace(run, status="crashed", ended_at=iso(now)))
            job = self.state.get_job(run.ref)
            if job is not None:
                self.state.upsert_job(_charge_crash(job))
        for job in self.state.jobs_with_status("running"):
            if job.ref not in crashed:
                self.state.upsert_job(replace(job, status="pending"))
        self.state.reset_crash_streaks(crashed)
        self.state.record_event(now, "recover", detail=",".join(sorted(crashed)))

    def tick(self, now: datetime) -> bool:
        """Run one decision step; False means the dispatcher should exit."""
        self.state.heartbeat(now)
        policy = self.policy.current()
        if not self._handle_stop_request(now):
            return False
        if self.active is not None:
            self._watch(now)
            return not (self.active is None and self.draining)
        if self.draining:
            return False
        self._dispatch(now, policy)
        return True

    def _handle_stop_request(self, now: datetime) -> bool:
        mode = "sigterm" if self.terminated else self.state.stop_requested()
        if mode is None:
            return True
        if mode != "graceful" and self.active is not None:
            self._stop_active(now)
        self.draining = True
        self.state.clear_stop()
        self.state.record_event(now, "dispatcher-stop", detail=mode)
        return self.active is not None

    def _stop_active(self, now: datetime) -> None:
        assert self.active is not None
        active, self.active = self.active, None
        code = active.handle.stop(self.deps.host.run)
        self._end_run(active, now, "stopped", code)
        self.state.upsert_job(replace(self._job(active.spec), status="pending"))

    def _watch(self, now: datetime) -> None:
        assert self.active is not None
        code = self.active.handle.poll()
        limit = timedelta(seconds=self.active.spec.timeout_s)
        timed_out = code is None and now - self.active.started_at > limit
        if timed_out:
            code = self.active.handle.stop(self.deps.host.run)
        if code is None:
            return
        active, self.active = self.active, None
        self._finish(active, now, code, timed_out)

    def _finish(self, active: Active, now: datetime, code: int, timed_out: bool) -> None:
        spec = active.spec
        job = replace(self._job(spec), crash_streak=0)
        if not timed_out and code in spec.ok_exit:
            self._read(active, now, job, code)
            return
        self._end_run(active, now, "timeout" if timed_out else "failed", code)
        attempts = job.attempts + 1
        status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
        self.state.upsert_job(replace(job, attempts=attempts, status=status))
        if status == "failed":
            self.state.record_event(now, "failed", spec.ref, f"{attempts} attempts")

    def _read(self, active: Active, now: datetime, job: JobRow, code: int) -> None:
        spec = active.spec
        env = {**self._env(spec), "FUCONE_RUN_DIR": str(active.run_dir.out)}
        reader_rc = self.deps.run_in(spec.reader.argv, spec.cwd, env).returncode
        if reader_rc != 0 or not isinstance(spec.reader, RunReader):
            status = "reader-failed" if reader_rc != 0 else "awaiting-review"
            self._end_run(active, now, status, code)
            self.state.upsert_job(replace(job, attempts=0, status="blocked"))
            detail = f"reader exit {reader_rc}" if reader_rc != 0 else active.run_id
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

    def _end_run(self, active: Active, now: datetime, status: str, code: int | None) -> None:
        run = self.state.get_run(active.run_id)
        assert run is not None
        self.state.update_run(replace(run, status=status, ended_at=iso(now), exit_code=code))
        self.state.record_event(now, "run-end", active.spec.ref, f"{active.run_id} {status} {code}")

    def _env(self, spec: TaskSpec) -> dict[str, str]:
        base = {k: self.deps.base_env[k] for k in BASE_ENV_KEYS if k in self.deps.base_env}
        return {**base, **spec.env}

    def _dispatch(self, now: datetime, policy: Policy) -> None:
        specs = self._listed_specs(now, policy)
        forced = self._run_now(now, specs)
        ready = [forced] if forced else [s for s in specs if self._is_ready(now, s)]
        reason = self._idle_reason(now, policy, ready, forced is not None)
        if reason is not None:
            if forced is not None:
                self.state.record_event(now, "refused", forced.ref, f"run-now: {reason}")
            self.state.set_idle_reason(now, reason)
            return
        self.state.clear_idle_reason()
        self._start(now, ready[0])

    def _run_now(self, now: datetime, specs: list[TaskSpec]) -> TaskSpec | None:
        ref = self.state.run_now_requested()
        if ref is None:
            return None
        self.state.clear_run_now()
        match = [s for s in specs if s.ref == ref and self._job(s).status == "pending"]
        if not match:
            detail = "run-now: not listed, no contract pass, or not pending (use rerun first)"
            self.state.record_event(now, "refused", ref, detail)
        return match[0] if match else None

    def _idle_reason(
        self, now: datetime, policy: Policy, ready: list[TaskSpec], forced: bool
    ) -> str | None:
        if stopped(self.deps.paths.stop_file):
            return idle_reason("stopped")
        local = now.astimezone(self.deps.local_tz)
        if not forced and not in_window(local, policy.window_start, policy.window_end):
            return idle_reason("outside-window")
        if not ready:
            return idle_reason("no-ready-task")
        return host_idle_reason(self.deps.host, self.deps.ollama_url)

    def _listed_specs(self, now: datetime, policy: Policy) -> list[TaskSpec]:
        specs = []
        manifests: dict[str, dict[str, TaskSpec]] = {}
        for ref in policy.jobs:
            consumer, task_id = ref.split("/", 1)
            try:
                if consumer not in manifests:
                    path = policy.consumers[consumer]
                    manifests[consumer] = self.deps.load_manifest(path, consumer)
                spec = manifests[consumer][task_id]
            except (ManifestError, OSError, KeyError) as exc:
                self._refuse(now, ref, f"manifest: {exc}")
                continue
            if not self.state.has_contract_pass(ref, spec.fingerprint):
                self._refuse(now, ref, f"no contract pass for fingerprint {spec.fingerprint[:12]}")
                continue
            specs.append(spec)
        return specs

    def _refuse(self, now: datetime, ref: str, detail: str) -> None:
        if (ref, detail) not in self._refused:
            self._refused.add((ref, detail))
            self.state.record_event(now, "refused", ref, detail)

    def _job(self, spec: TaskSpec) -> JobRow:
        job = self.state.get_job(spec.ref)
        if job is None or job.fingerprint != spec.fingerprint:
            job = JobRow(spec.ref, spec.fingerprint, "pending")
            self.state.upsert_job(job)
        if spec.ready is not None and job.status == "done":
            job = replace(job, status="pending", ready_since=None)
            self.state.upsert_job(job)
        return job

    def _is_ready(self, now: datetime, spec: TaskSpec) -> bool:
        job = self._job(spec)
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
        assert spec.ready is not None
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
            self.state.upsert_job(replace(job, starved_flagged=1))
            self.state.record_event(now, "starved", job.ref, f"ready since {job.ready_since}")

    def _start(self, now: datetime, spec: TaskSpec) -> None:
        run_id, run_dir = fresh_run_dir(self.deps.paths.runs_dir, now)
        self.state.insert_run(RunRow(run_id, spec.ref, spec.fingerprint, "running", iso(now)))
        self.state.upsert_job(replace(self._job(spec), status="running"))
        self.state.record_event(now, "run-start", spec.ref, run_id)
        try:
            handle = self.deps.start(
                spec, run_dir, run_id, self.deps.ollama_url, self.deps.base_env
            )
        except OSError as exc:
            self._finish(Active(spec, run_id, run_dir, now, _NoHandle()), now, 127, False)
            self.state.record_event(now, "refused", spec.ref, f"could not start: {exc}")
            return
        self.active = Active(spec, run_id, run_dir, now, handle)


class _NoHandle:
    """Stands in for a run whose process never started."""

    def poll(self) -> int | None:
        return 127

    def stop(self, run: object, wait_s: float = 30.0) -> int:
        return 127


def _charge_crash(job: JobRow) -> JobRow:
    streak = job.crash_streak + 1
    attempts = job.attempts + (1 if streak >= 2 else 0)
    status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
    return replace(job, crash_streak=streak, attempts=attempts, status=status)


def serve(
    deps: Deps,
    policy_path: Path,
    *,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    tick_s: float = TICK_S,
) -> int:
    """Hold the lock, recover, and tick until a stop request; the only loop in fucone."""
    with DispatcherLock(deps.paths.lock), closing(open_state(deps.paths)) as state:

        def log(message: str) -> None:
            print(message, file=sys.stderr)
            state.record_event(clock(), "policy-error", detail=message)

        dispatcher = Dispatcher(deps, state, PolicyCache(policy_path, log))
        dispatcher.policy.current()

        def on_sigterm(_signum: int, _frame: FrameType | None) -> None:
            dispatcher.terminated = True  # the next tick stops; a handler must not touch sqlite

        previous = signal.signal(signal.SIGTERM, on_sigterm)
        stale = [state.stop_requested(), state.run_now_requested()]
        state.clear_stop()
        state.clear_run_now()
        dispatcher.recover(clock())
        dropped = [r for r in stale if r is not None]
        state.record_event(
            clock(), "dispatcher-start", detail=f"dropped stale {dropped}" if dropped else ""
        )
        try:
            while _guarded_tick(dispatcher, state, clock):
                sleep(tick_s)
        finally:
            signal.signal(signal.SIGTERM, previous)
    return 0


def _guarded_tick(dispatcher: Dispatcher, state: State, clock: Callable[[], datetime]) -> bool:
    """One tick; an unexpected error is logged as a tick-error event and the loop goes on."""
    try:
        return dispatcher.tick(clock())
    except Exception as exc:  # noqa: BLE001 — the night must not end on one bad tick; the error is recorded
        message = f"{type(exc).__name__}: {exc}"
        print(f"tick-error: {message}", file=sys.stderr)
        try:
            state.record_event(clock(), "tick-error", detail=message)
        except Exception as inner:  # noqa: BLE001 — state itself failing must still not kill the loop
            print(f"tick-error not recorded: {inner}", file=sys.stderr)
        return True
