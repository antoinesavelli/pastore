"""Run directories, the `systemd-run --user` scope, the job environment, stop, and leftover-scope recovery.

Depends on: fucone.spec (TaskSpec); every systemctl call goes through an injected `run` callable.
"""

import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fucone.spec import TaskSpec

BASE_ENV_KEYS = ("PATH", "HOME", "LANG", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
_SCOPE_NAME = re.compile(r"^fucone-.+\.scope$")
Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class PopenLike(Protocol):
    """The slice of subprocess.Popen the runner uses."""

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


@dataclass(frozen=True)
class RunDir:
    """One run's directory: `out` is FUCONE_RUN_DIR; the logs sit beside it."""

    root: Path
    out: Path
    stdout: Path
    stderr: Path


def new_run_id(now: datetime, seq: int) -> str:
    """`<UTC yyyymmddTHHMMSS>-<seq>`."""
    return now.astimezone(UTC).strftime("%Y%m%dT%H%M%S") + f"-{seq}"


def make_run_dir(runs_dir: Path, run_id: str) -> RunDir:
    """Create `runs_dir/run_id/out`; an existing run directory is an error."""
    root = runs_dir / run_id
    root.mkdir(parents=True, exist_ok=False)
    out = root / "out"
    out.mkdir()
    return RunDir(root=root, out=out, stdout=root / "stdout.log", stderr=root / "stderr.log")


def job_env(
    spec: TaskSpec, run_dir: RunDir, ollama_url: str, base: Mapping[str, str]
) -> dict[str, str]:
    """The explicit job environment: base keys, manifest env, FUCONE_RUN_DIR, the Ollama URL."""
    env = {key: base[key] for key in BASE_ENV_KEYS if key in base}
    env.update(spec.env)
    env["FUCONE_RUN_DIR"] = str(run_dir.out)
    env[spec.ollama_env] = ollama_url
    return env


def scope_argv(run_id: str, argv: Sequence[str]) -> list[str]:
    """Wrap argv in a transient user scope inside heavy-jobs.slice."""
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--slice=heavy-jobs.slice",
        f"--unit=fucone-{run_id}",
        "-p",
        "CPUWeight=20",
        "--",
        *argv,
    ]


class RunHandle:
    """A started run: poll it, or stop its scope."""

    def __init__(self, proc: PopenLike, run_id: str) -> None:
        self.proc = proc
        self.run_id = run_id

    def poll(self) -> int | None:
        """The exit code, or None while running."""
        return self.proc.poll()

    def stop(self, run: Runner, wait_s: float = 30.0) -> int:
        """Stop the scope, wait for the process, kill it if it outlives `wait_s`."""
        run(["systemctl", "--user", "stop", f"fucone-{self.run_id}.scope"])
        try:
            return self.proc.wait(wait_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(None)


def start_run(
    spec: TaskSpec,
    run_dir: RunDir,
    run_id: str,
    ollama_url: str,
    base_env: Mapping[str, str],
    *,
    popen: Callable[..., PopenLike] = subprocess.Popen,
    wrap: Callable[[str, Sequence[str]], list[str]] = scope_argv,
) -> RunHandle:
    """Start the job in its own session with logs beside the run's out/ directory."""
    with run_dir.stdout.open("wb") as out_log, run_dir.stderr.open("wb") as err_log:
        proc = popen(
            wrap(run_id, spec.argv),
            cwd=spec.cwd,
            env=job_env(spec, run_dir, ollama_url, base_env),
            stdout=out_log,
            stderr=err_log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return RunHandle(proc, run_id)


def leftover_scopes(run: Runner) -> list[str]:
    """Names of fucone-*.scope units systemd still knows about."""
    result = run(
        ["systemctl", "--user", "list-units", "--type=scope", "--all", "--plain", "--no-legend"]
        + ["fucone-*"]
    )
    if result.returncode != 0:
        raise RuntimeError(f"systemctl list-units exited {result.returncode}: {result.stderr}")
    names = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
    return [name for name in names if _SCOPE_NAME.match(name)]


def stop_leftovers(run: Runner) -> list[str]:
    """Stop every leftover fucone scope by name and return the names."""
    names = leftover_scopes(run)
    for name in names:
        result = run(["systemctl", "--user", "stop", name])
        if result.returncode != 0:
            raise RuntimeError(f"systemctl stop {name} exited {result.returncode}")
    return names


def has_result(out: Path) -> bool:
    """True when out/ holds something and no file named FAILED."""
    entries = list(out.iterdir())
    return bool(entries) and not any(entry.name == "FAILED" for entry in entries)
