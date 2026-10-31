"""A running job (`Active`), the reads that explain how its run ended (the kernel log's OOM kill, R34; the gateway's outage, D-050 (4)) and the crash charge (D-012).

Depends on: contract (Started), hosts (HttpJson, PROBE_TIMEOUT_S), runner (RunDir, Runner), spec (TaskSpec), state (JobRow).
"""

import urllib.error
from dataclasses import dataclass, replace
from datetime import datetime

from pastore.contract import Started
from pastore.hosts import PROBE_TIMEOUT_S, HttpJson
from pastore.runner import RunDir, Runner
from pastore.spec import TaskSpec
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


def charge_crash(job: JobRow) -> JobRow:
    streak = job.crash_streak + 1
    attempts = job.attempts + (1 if streak >= 2 else 0)
    status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
    return replace(job, crash_streak=streak, attempts=attempts, status=status)
