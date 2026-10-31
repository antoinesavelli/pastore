"""Interruption harness: a real dispatcher process on stubbed host seams, and the test-side helpers to kill it and read what it left.

Depends on: pastore dispatcher, manifest, runner, state and the stdlib; run as `python -m tests.interrupt_harness <config.json>`.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from types import TracebackType
from typing import Any, Self

import yaml

from pastore.checks import MODULE_VERSION_FILE, Host
from pastore.dispatcher import Deps, serve
from pastore.manifest import load_manifest
from pastore.runner import RunDir, RunHandle, start_run
from pastore.spec import TaskSpec
from pastore.state import (
    EventRow,
    JobRow,
    RunRow,
    State,
    StatePaths,
    default_paths,
    open_state,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "m"
CONSUMER = "c"
WS_URL = "http://ws.test:19000"
GIB = 1024**3
SHA256 = "a" * 64
DRIVER = "580.178.04"
CALLS = "host-calls.jsonl"
LIFE_S = 120
INFO: dict[str, object] = {
    "general.architecture": "llama",
    "llama.block_count": 2,
    "llama.attention.head_count_kv": 2,
    "llama.attention.key_length": 64,
}
# A job that writes started.json into its run's out/ and exits 0; SLEEPER then sleeps, and its 90 s cap ends an orphan.
QUICK = """\
import json, os, pathlib, time
out = pathlib.Path(os.environ["PASTORE_RUN_DIR"])
info = {"pid": os.getpid(), "run_dir": str(out), "model": os.environ["MODEL_NAME"],
        "ollama": os.environ["OLLAMA_HOST"]}
(out / "started.tmp").write_text(json.dumps(info))
(out / "started.tmp").replace(out / "started.json")
"""
SLEEPER = QUICK + "time.sleep(90)\n"


@dataclass(frozen=True)
class Config:
    """What the harness process is told, as JSON; paths are strings."""

    home: str
    policy: str
    root: str
    list_scopes: bool
    ollama_url: str
    life_s: int


def proc_cwd(pid: int) -> str | None:
    """The working directory of a live process, None once it is gone or a zombie."""
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def job_alive(root: Path, pid: int) -> bool:
    """Whether `pid` is still a live process in the consumer root; a reused pid is not."""
    return proc_cwd(pid) == str(root)


def job_pids(root: Path) -> list[int]:
    """Every live process whose working directory is the consumer root: the jobs and what they started."""
    pids = [int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    return [pid for pid in pids if job_alive(root, pid)]


def harness_alive(pid: int) -> bool:
    """Whether `pid` is a live harness process (a zombie has no command line)."""
    try:
        return b"tests.interrupt_harness" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def read_pid(path: Path) -> int | None:
    """The pid in a job.pid file, None while the file is missing or half written."""
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def kill_group(pid: int) -> None:
    """SIGKILL the process group of `pid`, never the caller's own."""
    try:
        group = os.getpgid(pid)
        if group != os.getpgrp():
            os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _done(argv: Sequence[str], stdout: str = "") -> CompletedProcess[str]:
    return CompletedProcess(list(argv), 0, stdout, "")


def _live_runs(runs_dir: Path, root: Path) -> list[str]:
    """Run ids whose job process is alive: the scopes systemd would still list."""
    pids = {path.parent.name: read_pid(path) for path in runs_dir.glob("*/job.pid")}
    return sorted(run_id for run_id, pid in pids.items() if pid and job_alive(root, pid))


def _stop_scope(runs_dir: Path, root: Path, unit: str) -> None:
    """What `systemctl --user stop pastore-<run>.scope` does: kill the job's group and wait until it is gone."""
    run_id = unit.removeprefix("pastore-").removesuffix(".scope")
    pid = read_pid(runs_dir / run_id / "job.pid")
    if pid is None or not job_alive(root, pid):
        return
    kill_group(pid)
    deadline = time.monotonic() + 5
    while job_alive(root, pid) and time.monotonic() < deadline:
        time.sleep(0.01)


def _host_run(config: Config, runs_dir: Path) -> Callable[[Sequence[str]], CompletedProcess[str]]:
    """The harness's stand-in for systemd, nvidia-smi and dpkg-query; every call is logged as one JSON line."""
    root = Path(config.root)
    log = Path(config.home) / CALLS

    def run(argv: Sequence[str]) -> CompletedProcess[str]:
        with log.open("a") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "argv": list(argv)}) + "\n")
        if list(argv[:3]) == ["systemctl", "--user", "list-units"]:
            runs = _live_runs(runs_dir, root) if config.list_scopes else []
            lines = "".join(f"pastore-{r}.scope loaded active running pastore job\n" for r in runs)
            return _done(argv, lines)
        if list(argv[:3]) == ["systemctl", "--user", "stop"]:
            _stop_scope(runs_dir, root, argv[3])
        elif argv[0] == "dpkg-query":
            return _done(argv, f"{DRIVER}-0ubuntu0.24.04.1\n")
        elif argv[0] == "nvidia-smi":
            return _done(argv, "1, /usr/bin/cosmic-monitor\n")
        return _done(argv)

    return run


def _read_text(path: Path) -> str:
    return DRIVER + "\n" if path == MODULE_VERSION_FILE else "MemAvailable: 40000000 kB\n"


def _http_json(url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
    """Answers for the policy's one host, no network: version, tags (the model), ps (none), show."""
    base, _, path = url.partition("/api/")
    if base != WS_URL:
        raise OSError(f"{base} is not a host of the harness policy")
    if path == "version":
        return {"version": "0.0.0-fake"}
    if path == "tags":
        return {"models": [{"name": MODEL, "size": GIB}]}
    if path == "ps":
        return {"models": []}
    if path == "show":
        return {"model_info": INFO}
    raise OSError(f"no fake answer for {url}")


def _start(
    spec: TaskSpec,
    run_dir: RunDir,
    run_id: str,
    url: str,
    model: str,
    env: Mapping[str, str],
) -> RunHandle:
    """Start the job in its own session without a scope, so it outlives the harness as a scope would."""
    handle = start_run(spec, run_dir, run_id, url, model, env, wrap=lambda _r, argv, _t: list(argv))
    assert isinstance(handle.proc, subprocess.Popen)
    (run_dir.root / "job.pid").write_text(f"{handle.proc.pid}\n")
    return handle


def main(argv: Sequence[str]) -> int:
    """Run the real `serve` loop over the stubbed seams described by the config file; it ends with its parent or after `life_s`."""
    config = Config(**json.loads(Path(argv[0]).read_text()))
    signal.alarm(config.life_s)
    parent = os.getppid()
    paths = default_paths(Path(config.home))
    host = Host(
        run=_host_run(config, paths.runs_dir), read_text=_read_text, http_status=lambda _u, _t: 200
    )

    def run_in(argv: Sequence[str], _cwd: Path, _env: Mapping[str, str]) -> CompletedProcess[str]:
        return _done(argv)

    def sleep(seconds: float) -> None:
        if os.getppid() != parent:
            os._exit(1)  # the test that started us is gone; a leaked harness must not tick forever
        time.sleep(seconds)
        if os.getppid() != parent:
            os._exit(1)  # nor tick once more after a parent that died in the nap

    deps = Deps(
        paths=paths,
        load_manifest=load_manifest,
        host=host,
        run_in=run_in,
        start=_start,
        base_env={"PATH": "/usr/bin:/bin", "HOME": config.home},
        ollama_url=config.ollama_url,
        http_json=_http_json,
    )
    return serve(
        deps, Path(config.policy), clock=lambda: datetime.now(UTC), sleep=sleep, tick_s=1.0
    )


@dataclass(frozen=True)
class Prepared:
    """One test's consumer, policy and tmp home, ready for a harness."""

    home: Path
    paths: StatePaths
    policy: Path
    root: Path


def prepare(
    tmp_path: Path,
    job_body: str = SLEEPER,
    *,
    also: Mapping[str, str] | None = None,
    fields: Mapping[str, object] | None = None,
) -> Prepared:
    """Write a consumer (task `a` runs `job_body`, each `also` id its body; `fields`, e.g. a `ready` rule, join every task), a policy listing them, the contract passes."""
    home = tmp_path / "home"
    root = (tmp_path / "consumer").resolve()
    root.mkdir(parents=True)
    bodies = {"a": job_body, **(also or {})}
    for task_id, body in bodies.items():
        script = root / f"{task_id}.py"
        script.write_text(f"#!{sys.executable}\n{body}")
        script.chmod(0o755)
    tasks = [
        {
            "id": task_id,
            "kind": "bench",
            "argv": [f"./{task_id}.py"],
            "cwd": ".",
            "model": MODEL,
            "timeout": "10m",
            "resumable": True,
            "ollama_env": "OLLAMA_HOST",
            "model_env": "MODEL_NAME",
            "reader": {"run": ["./read"]},
            **(fields or {}),
        }
        for task_id in bodies
    ]
    manifest = root / "gpu-tasks.yaml"
    manifest.write_text(yaml.safe_dump({"version": 1, "tasks": tasks}))
    evidence = json.dumps(str(root / "evidence.json"))
    listed = "".join(
        f"  - {{task: {CONSUMER}/{t}, model: {MODEL}, evidence: {evidence}, sha256: {SHA256}}}\n"
        for t in bodies
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "version: 2\n"
        f'hosts:\n  workstation: {{url: "{WS_URL}", vram_gib: 24, max_models: 3, parallel: 3,'
        " context: 32768}\n"
        'sources:\n  - {level: hearth, cgroup: ["pastore-*.scope"]}\n'
        f"consumers:\n  {CONSUMER}: {manifest}\njobs:\n{listed}"
    )
    paths = default_paths(home)
    with closing(open_state(paths)) as state:
        for spec in load_manifest(manifest, CONSUMER).values():
            state.record_contract_pass(spec.ref, spec.fingerprint, datetime.now(UTC))
    return Prepared(home, paths, policy, root)


@dataclass(frozen=True)
class Harness:
    """A running harness process and where its output went."""

    proc: subprocess.Popen[bytes]
    out: Path
    err: Path

    def kill(self) -> int:
        """SIGKILL the dispatcher process only; its job, in a session of its own, keeps running. Returns the exit code."""
        self.proc.kill()
        return self.proc.wait(timeout=5)

    def stderr(self) -> str:
        """The harness's stderr so far, for assertion messages."""
        return self.err.read_text(errors="replace")


class Fleet:
    """Every harness a test started; leaving the context SIGKILLs them, then every job left in their consumer roots."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.harnesses: list[Harness] = []
        self.roots: set[Path] = set()

    def spawn(
        self,
        prepared: Prepared,
        *,
        list_scopes: bool = True,
        life_s: int = LIFE_S,
        launcher: Sequence[str] = (),
    ) -> Harness:
        """Start a harness (its own session, output to files); `list_scopes=False` makes the systemctl stub list none, `life_s` is its hard cap, `launcher` an argv that starts it."""
        n = len(self.harnesses) + 1
        config = Config(
            str(prepared.home),
            str(prepared.policy),
            str(prepared.root),
            list_scopes,
            WS_URL,
            life_s,
        )
        config_file = self.tmp / f"harness-{n}.json"
        config_file.write_text(json.dumps(asdict(config)))
        env = {"HOME": str(prepared.home), "PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1"}
        self.roots.add(prepared.root)
        out, err = self.tmp / f"harness-{n}.out", self.tmp / f"harness-{n}.err"
        with out.open("wb") as out_log, err.open("wb") as err_log:
            proc = subprocess.Popen(
                [*launcher, sys.executable, "-m", "tests.interrupt_harness", str(config_file)],
                cwd=ROOT,
                env=env,
                stdout=out_log,
                stderr=err_log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self.harnesses.append(Harness(proc, out, err))
        return self.harnesses[-1]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for harness in self.harnesses:
            if harness.proc.poll() is None:
                kill_group(harness.proc.pid)
            harness.proc.wait(timeout=5)
        deadline = time.monotonic() + 5
        clean = 0
        while clean < 2:  # two empty scans in a row: a job forked as its harness died shows up late
            left = [pid for root in self.roots for pid in job_pids(root)]
            clean = 0 if left else clean + 1
            for pid in left:
                kill_group(pid)
            assert not left or time.monotonic() < deadline, f"jobs survived teardown: {left}"
            time.sleep(0.05)


def wait_until[T](probe: Callable[[], T | None], timeout_s: float, what: str) -> T:
    """Poll every 0.05 s until `probe()` is truthy and return it; fail naming `what` after `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = probe()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.05)


def read_state[T](prepared: Prepared, read: Callable[[State], T]) -> T:
    """Apply `read` to a read-only view of the tmp state database."""
    with closing(open_state(prepared.paths, readonly=True)) as state:
        return read(state)


def events(prepared: Prepared) -> list[EventRow]:
    """Every event, oldest first."""
    return read_state(prepared, lambda state: state.recent_events(100_000)[::-1])


def job(prepared: Prepared, ref: str = f"{CONSUMER}/a", model: str = MODEL) -> JobRow | None:
    """The job row of a pair."""
    return read_state(prepared, lambda state: state.get_job(ref, model))


def idle_reason(prepared: Prepared) -> str | None:
    """The stored idle reason."""
    return read_state(prepared, lambda state: state.idle_reason())


def runs(prepared: Prepared) -> list[RunRow]:
    """Every run row, oldest first."""
    uri = f"{prepared.paths.db.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at, run_id")
        return [RunRow(*row) for row in rows]


def job_pid(prepared: Prepared, run_id: str) -> int | None:
    """The pid the harness recorded for a run's job."""
    return read_pid(prepared.paths.runs_dir / run_id / "job.pid")


def job_started(prepared: Prepared, run_id: str) -> dict[str, Any] | None:
    """What the job wrote to its out/started.json (pid, run_dir, model, ollama), None before it has."""
    try:
        data: dict[str, Any] = json.loads(
            (prepared.paths.runs_dir / run_id / "out" / "started.json").read_text()
        )
    except (OSError, ValueError):
        return None
    return data


def running_job(prepared: Prepared, seen: Collection[str] = ()) -> RunRow | None:
    """A `running` run not in `seen` whose job has started and is alive."""
    for run in runs(prepared):
        started = job_started(prepared, run.run_id)
        if (
            run.status == "running"
            and run.run_id not in seen
            and started is not None
            and job_alive(prepared.root, started["pid"])
        ):
            return run
    return None


def host_calls(prepared: Prepared, pid: int | None = None) -> list[list[str]]:
    """Every argv the stubbed host saw, in order; only those of one harness process when `pid` is given."""
    path = prepared.home / CALLS
    if not path.exists():
        return []
    whole = [line for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]
    lines = [json.loads(line) for line in whole]
    return [line["argv"] for line in lines if pid is None or line["pid"] == pid]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
