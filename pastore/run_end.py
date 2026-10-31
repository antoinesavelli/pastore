"""A running job (`Active`), its poll, the reads that explain how its run ended (the kernel log's OOM kill, R34; an outage of the gateway or the executor, D-050 (4), D-052 (1)) and the attempt charges (D-012, D-052 (1)).

Depends on: contract (Started), hosts (HttpJson, PROBE_TIMEOUT_S, executor_reason), policy (Policy), runner (RunDir, Runner), spec (TaskSpec, WORKSTATION, idle_reason), state (JobRow).
"""

import urllib.error
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from pastore.contract import Started
from pastore.hosts import PROBE_TIMEOUT_S, HttpJson, executor_reason
from pastore.policy import Policy
from pastore.runner import RunDir, Runner
from pastore.spec import WORKSTATION, TaskSpec, idle_reason
from pastore.state import JobRow

MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Active:
    """One running job and the host it was placed on."""

    spec: TaskSpec
    model: str
    host: str
    run_id: str
    run_dir: RunDir
    started_at: datetime
    handle: Started


class NoHandle:
    """Stands in for a run whose process never started."""

    def poll(self) -> int | None:
        return 127

    def stop(self, run: object, wait_s: float = 30.0) -> int:
        return 127


def poll_run(active: Active, now: datetime, run: Runner) -> tuple[int, bool] | None:
    """The exit code and whether the run timed out (its scope is stopped then), or None while it goes on."""
    code = active.handle.poll()
    timed_out = code is None and now - active.started_at > timedelta(seconds=active.spec.timeout_s)
    if timed_out:
        code = active.handle.stop(run)
    return None if code is None else (code, timed_out)


def oom_killed(run: Runner, active: Active) -> bool:
    """True when this boot's kernel log holds an OOM kill whose victim sat in the run's own scope."""
    if isinstance(active.handle, NoHandle):
        return False
    since = int(active.started_at.timestamp())
    result = run(["journalctl", "-k", "-o", "cat", "--no-pager", f"--since=@{since}"])
    if result.returncode != 0:
        return False
    scope = f"/pastore-{active.run_id}.scope"
    return any(
        field.startswith("task_memcg=") and field.endswith(scope)
        for line in result.stdout.splitlines()
        if line.startswith("oom-kill:")
        for field in line.split(",")
    )


def gateway_outage(get: HttpJson, status_url: str, since: datetime) -> bool:
    """True when `status_url` refuses connections or answers a `started_at` after `since`; an answer that is no gateway is no outage."""
    try:
        answer = get(status_url, None, PROBE_TIMEOUT_S)
    except urllib.error.HTTPError:
        return False
    except OSError:
        return True
    except ValueError:
        return False
    started = answer.get("started_at") if isinstance(answer, dict) else None
    if not isinstance(started, str):
        return False
    try:
        restarted = datetime.fromisoformat(started)
    except ValueError:
        return False
    return restarted.tzinfo is not None and restarted > since


def outage_reason(active: Active, policy: Policy, get: HttpJson, ollama_url: str) -> str | None:
    """Check 9 and D-050 (4): the idle reason of the outage that cut a non-ok end short (`executor-down` for a lost non-workstation host, else `ollama-down` for a gateway that is down or restarted since the run began), or None."""
    host = policy.hosts.get(active.host)
    if host is not None and host.name != WORKSTATION and executor_reason(host, get) is not None:
        return idle_reason("executor-down")
    down = gateway_outage(get, ollama_url + "/pastore/status", active.started_at)
    return idle_reason("ollama-down") if down else None


def _counted(job: JobRow, attempts: int, streak: int) -> JobRow:
    """The job with these counters, `failed` once attempts reach MAX_ATTEMPTS and `pending` otherwise."""
    status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
    return replace(job, crash_streak=streak, attempts=attempts, status=status)


def charge_failure(job: JobRow) -> JobRow:
    """A failed or timed-out end: one attempt, and the crash streak ends."""
    return _counted(job, job.attempts + 1, 0)


def charge_crash(job: JobRow) -> JobRow:
    """D-012, D-052 (1): a crash or an outage end extends the streak; an attempt is charged from the second in a row."""
    streak = job.crash_streak + 1
    return _counted(job, job.attempts + (1 if streak >= 2 else 0), streak)
