"""Run directories, the `systemd-run --user` scope, the job environment, stop, and leftover-scope recovery.

Depends on: pastore.spec (TaskSpec); every systemctl call goes through an injected `run` callable.
"""

import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pastore.spec import BASE_ENV_KEYS, RUN_DIR_ENV, TaskSpec

_SCOPE_NAME = re.compile(r"^pastore-.+\.scope$")
Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class PopenLike(Protocol):
    """The slice of subprocess.Popen the runner uses."""

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


@dataclass(frozen=True)
class RunDir:
    """One run's directory: `out` is PASTORE_RUN_DIR; the logs sit beside it."""

    root: Path
    out: Path
    stdout: Path
    stderr: Path


def new_run_id(now: datetime, seq: int) -> str:
    """`<UTC yyyymmddTHHMMSS>-<seq>`."""
    return now.astimezone(UTC).strftime("%Y%m%dT%H%M%S") + f"-{seq}"


def fresh_run_dir(runs_dir: Path, now: datetime, suffix: str = "") -> tuple[str, RunDir]:
    """The first free `<UTC stamp>-<seq><suffix>` run id and its new directory."""
    for seq in range(1, 1000):
        run_id = new_run_id(now, seq) + suffix
        try:
            return run_id, make_run_dir(runs_dir, run_id)
        except FileExistsError:
            continue
    raise RuntimeError(f"no free run id under {runs_dir} at {now.isoformat()}")


def make_run_dir(runs_dir: Path, run_id: str) -> RunDir:
    """Create `runs_dir/run_id/out`; an existing run directory is an error."""
    root = runs_dir / run_id
    root.mkdir(parents=True, exist_ok=False)
    out = root / "out"
    out.mkdir()
    return RunDir(root=root, out=out, stdout=root / "stdout.log", stderr=root / "stderr.log")


def job_env(
    spec: TaskSpec, run_dir: RunDir, ollama_url: str, model: str, base: Mapping[str, str]
) -> dict[str, str]:
    """The explicit job environment: base keys, manifest env, the run dir, the Ollama URL, the model."""
    env = {key: base[key] for key in BASE_ENV_KEYS if key in base}
    env.update(spec.env)
    env[RUN_DIR_ENV] = str(run_dir.out)
    env[spec.ollama_env] = ollama_url
    env[spec.model_env] = model
    return env


def scope_argv(run_id: str, argv: Sequence[str], timeout_s: int) -> list[str]:
    """Wrap argv in a transient user scope inside heavy-jobs.slice; systemd kills it at timeout_s even if pastore dies."""
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--slice=heavy-jobs.slice",
        f"--unit=pastore-{run_id}",
        "-p",
        "CPUWeight=20",
        "-p",
        f"RuntimeMaxSec={timeout_s}",
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
        run(["systemctl", "--user", "stop", f"pastore-{self.run_id}.scope"])
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
    model: str,
    base_env: Mapping[str, str],
    *,
    popen: Callable[..., PopenLike] = subprocess.Popen,
    wrap: Callable[[str, Sequence[str], int], list[str]] = scope_argv,
) -> RunHandle:
    """Start the job in its own session with logs beside the run's out/ directory."""
    with run_dir.stdout.open("wb") as out_log, run_dir.stderr.open("wb") as err_log:
        proc = popen(
            wrap(run_id, spec.argv, spec.timeout_s),
            cwd=spec.cwd,
            env=job_env(spec, run_dir, ollama_url, model, base_env),
            stdout=out_log,
            stderr=err_log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return RunHandle(proc, run_id)


def leftover_scopes(run: Runner) -> list[str]:
    """Names of pastore-*.scope units systemd still knows about."""
    result = run(
        ["systemctl", "--user", "list-units", "--type=scope", "--all", "--plain", "--no-legend"]
        + ["pastore-*"]
    )
    if result.returncode != 0:
        raise RuntimeError(f"systemctl list-units exited {result.returncode}: {result.stderr}")
    names = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
    return [name for name in names if _SCOPE_NAME.match(name)]


def stop_leftovers(run: Runner) -> list[str]:
    """Stop every leftover pastore scope by name and return the names."""
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
