"""The pastore command line: start, stop, status (state plus the gateway's /pastore/status), add/remove (gated), check [--contract], run-now, rerun, ack, reset.

Depends on: spec (Level, Pair, TaskSpec), policy, manifest, state, dispatcher (serve), contract, hosts (HttpJson, injected for the gateway read), checks/runner real bindings; binds the real clock only here.
"""

import argparse
import hashlib
import math
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TextIO

from pastore import hosts
from pastore.checks import OLLAMA_URL, real_host
from pastore.contract import run_contract
from pastore.dispatcher import Deps, serve
from pastore.hosts import HttpJson
from pastore.manifest import ManifestError, load_manifest
from pastore.policy import (
    Policy,
    PolicyError,
    load_policy,
    policy_diff,
    render_policy,
    with_job_added,
    with_job_removed,
)
from pastore.runner import start_run
from pastore.spec import Level, Pair, TaskSpec
from pastore.state import (
    DispatcherLock,
    LockHeld,
    State,
    StateError,
    StatePaths,
    default_paths,
    lock_is_held,
    open_state,
    reset,
)

HEARTBEAT_STALE = timedelta(seconds=60)
SUBPROCESS_TIMEOUT_S = 45
GATEWAY_TIMEOUT_S = 2.0
LEVEL_NAMES = tuple(level.name.lower() for level in Level)
TEXT_MAX_CHARS = 80
HOST_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "policy.yaml"


@dataclass(frozen=True)
class Ctx:
    """Everything a command needs from its environment."""

    paths: StatePaths
    policy_path: Path
    clock: Callable[[], datetime]
    out: TextIO
    isatty: Callable[[], bool]
    http_json: HttpJson


def _say(ctx: Ctx, text: str = "") -> None:
    print(text, file=ctx.out)


def _refuse(text: str) -> int:
    print(text, file=sys.stderr)
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pastore", description="GPU job queue.")
    parser.add_argument("--policy", type=Path, help="policy.yaml to use (default: the repo's)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start")
    sub.add_parser("status")
    sub.add_parser("reset")
    sub.add_parser("stop").add_argument("--now", action="store_true")
    for name in ("add", "remove", "rerun", "run-now"):
        cmd = sub.add_parser(name)
        cmd.add_argument("ref")
        cmd.add_argument("--model", required=True)
        if name in ("add", "remove"):
            cmd.add_argument("--yes", action="store_true")
        if name == "add":
            cmd.add_argument("--evidence", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("path", nargs="?", type=Path)
    check.add_argument("--contract", metavar="REF")
    sub.add_parser("ack").add_argument("run_id")
    return parser


def _dispatcher_up(ctx: Ctx) -> bool:
    if not ctx.paths.db.exists() or not lock_is_held(ctx.paths.lock):
        return False
    with closing(open_state(ctx.paths, readonly=True)) as state:
        beat = state.last_heartbeat()
    return beat is not None and ctx.clock() - beat <= HEARTBEAT_STALE


def _pair_line(state: State, pair: Pair) -> str:
    """One status line for a listed pair: job status, attempts, last run and its host, starved flag."""
    job, run = state.get_job(pair.task, pair.model), state.last_run(pair.task, pair.model)
    line = f"{pair.task} {pair.model}: {job.status if job else 'unseen'}"
    line += f" attempts={job.attempts if job else 0}"
    line += f" last_run={run.status} host={run.host}" if run else " last_run=none"
    return line + (" starved" if job and job.starved_flagged else "")


def _run_lines(state: State) -> list[str]:
    """One line per run awaiting review, then one per running run with its host."""
    review = [
        f"awaiting review: {r.run_id} ({r.ref} {r.model})"
        for r in state.runs_with_status("awaiting-review")
    ]
    running = [
        f"running: {r.run_id} ({r.ref} {r.model} on {r.host})"
        for r in state.runs_with_status("running")
    ]
    return review + running


class _Unexpected(Exception):
    """The gateway answered something other than the documented /pastore/status shape."""


def _count(host: str, key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _Unexpected(f"{host}: {key} is missing or not a non-negative integer")
    return value


def _oldest_wait(host: str, entry: dict[str, object]) -> str:
    value = entry.get("oldest_wait_s")
    if value is None and "oldest_wait_s" in entry:
        return "none"
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value < math.inf:
        raise _Unexpected(f"{host}: oldest_wait_s is missing or not a non-negative number or null")
    return str(round(value, 1))


def _gateway_line(host: str, entry: object) -> str:
    if not HOST_NAME.fullmatch(host):
        raise _Unexpected(f"host name {host!r} is not 1 to 64 of A-Z a-z 0-9 . _ -")
    if not isinstance(entry, dict):
        raise _Unexpected(f"{host}: not an object")
    queued = entry.get("queued")
    if not isinstance(queued, dict):
        raise _Unexpected(f"{host}: queued is missing or not an object")
    levels = " ".join(
        f"{name}={_count(host, f'queued.{name}', queued.get(name))}" for name in LEVEL_NAMES
    )
    return (
        f"gateway {host}: slots={_count(host, 'slots', entry.get('slots'))} "
        f"in_flight={_count(host, 'in_flight', entry.get('in_flight'))} queued {levels} "
        f"oldest_wait_s={_oldest_wait(host, entry)}"
    )


def _gateway_lines(answer: object) -> list[str]:
    table = answer.get("hosts") if isinstance(answer, dict) else None
    if not isinstance(table, dict) or not table:
        raise _Unexpected("hosts is missing, empty or not an object")
    return [_gateway_line(host, entry) for host, entry in table.items()]


def _oneline(text: str) -> str:
    """Gateway text made safe to print: control characters become spaces, whitespace collapses, 80 characters at most."""
    flat = " ".join("".join(c if c.isprintable() else " " for c in text).split())
    return flat if len(flat) <= TEXT_MAX_CHARS else flat[: TEXT_MAX_CHARS - 3] + "..."


def _say_gateway(ctx: Ctx) -> None:
    """Print the gateway's per-host queue lines, or one line saying why there are none."""
    try:
        answer = ctx.http_json(OLLAMA_URL + "/pastore/status", None, GATEWAY_TIMEOUT_S)
    except (OSError, ValueError) as exc:
        _say(ctx, f"gateway: not answering ({_oneline(str(exc)) or type(exc).__name__})")
        return
    try:
        lines = _gateway_lines(answer)
    except _Unexpected as exc:
        lines = [f"gateway: unexpected /pastore/status answer ({_oneline(str(exc))})"]
    for line in lines:
        _say(ctx, line)


def _tick_error_lines(state: State) -> list[str]:
    """The newest tick-error event as a status line, or nothing."""
    errors = [e for e in state.recent_events(200) if e.kind == "tick-error"]
    return [f"last tick-error: {errors[0].at} {errors[0].detail}"] if errors else []


def _status(ctx: Ctx, _args: argparse.Namespace) -> int:
    if not ctx.paths.db.exists():
        _say(ctx, "no state yet")
        _say_gateway(ctx)
        return 0
    up = _dispatcher_up(ctx)
    with closing(open_state(ctx.paths, readonly=True)) as state:
        _say(ctx, f"dispatcher: {'up' if up else 'dispatcher-down'}")
        _say(ctx, f"idle: {state.idle_reason() or 'none'}")
        for line in _tick_error_lines(state):
            _say(ctx, line)
        try:
            jobs = load_policy(ctx.policy_path).jobs
        except PolicyError as exc:
            _say(ctx, f"policy error: {exc}")
            return 1
        for pair in jobs:
            _say(ctx, _pair_line(state, pair))
        for line in _run_lines(state):
            _say(ctx, line)
    _say_gateway(ctx)
    return 0


def _stop(ctx: Ctx, args: argparse.Namespace) -> int:
    mode: Literal["graceful", "now"] = "now" if args.now else "graceful"
    with closing(open_state(ctx.paths)) as state:
        state.request_stop(mode)
    _say(ctx, f"stop requested ({mode})")
    return 0


def _write_atomic(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".policy-")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _confirm_and_write(ctx: Ctx, new: Policy, assume_yes: bool) -> int:
    old_text = ctx.policy_path.read_text(encoding="utf-8")
    new_text = render_policy(new)
    _say(ctx, policy_diff(old_text, new_text, ctx.policy_path))
    if not assume_yes:
        if not ctx.isatty():
            return _refuse("not a terminal: pass --yes to apply this edit")
        ctx.out.write("apply this edit? [y/N] ")
        ctx.out.flush()
        if sys.stdin.readline().strip().lower() != "y":
            return _refuse("not applied")
    _write_atomic(ctx.policy_path, new_text)
    _say(ctx, f"wrote {ctx.policy_path}")
    return 0


def _has_contract_pass(ctx: Ctx, ref: str, fingerprint: str) -> bool:
    if not ctx.paths.db.exists():
        return False
    with closing(open_state(ctx.paths, readonly=True)) as state:
        return state.has_contract_pass(ref, fingerprint)


def _spec_for(policy: Policy, ref: str) -> TaskSpec:
    consumer, _, task_id = ref.partition("/")
    if consumer not in policy.consumers:
        raise PolicyError(f"{ref}: consumer {consumer!r} is not in policy.yaml")
    tasks = load_manifest(policy.consumers[consumer], consumer)
    if task_id not in tasks:
        raise ManifestError(f"{ref}: no such task in {policy.consumers[consumer]}")
    return tasks[task_id]


def _add(ctx: Ctx, args: argparse.Namespace) -> int:
    policy = load_policy(ctx.policy_path)
    spec = _spec_for(policy, args.ref)
    if args.model not in spec.models:
        return _refuse(
            f"{args.ref}: model {args.model!r} is not in its manifest models {list(spec.models)}"
        )
    evidence = Path(os.path.realpath(args.evidence))
    if not evidence.is_file():
        return _refuse(f"{evidence}: evidence must be an existing regular file")
    if not _has_contract_pass(ctx, args.ref, spec.fingerprint):
        return _refuse(
            f"{args.ref}: no passing contract test for its current manifest entry; "
            f"run: pastore check --contract {args.ref}"
        )
    with evidence.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    pair = Pair(args.ref, args.model, evidence, digest)
    return _confirm_and_write(ctx, with_job_added(policy, pair), args.yes)


def _remove(ctx: Ctx, args: argparse.Namespace) -> int:
    new = with_job_removed(load_policy(ctx.policy_path), args.ref, args.model)
    return _confirm_and_write(ctx, new, args.yes)


def _check(ctx: Ctx, args: argparse.Namespace) -> int:
    if (args.path is None) == (args.contract is None):
        return _refuse("check needs exactly one of <path> or --contract <ref>")
    if args.contract is not None:
        return _contract(ctx, args.contract)
    path = args.path.resolve()
    named = [n for n, p in load_policy(ctx.policy_path).consumers.items() if p.resolve() == path]
    for task in load_manifest(path, named[0] if named else path.parent.name).values():
        _say(ctx, f"{task.ref} {task.fingerprint}")
    return 0


def _rerun(ctx: Ctx, args: argparse.Namespace) -> int:
    with closing(open_state(ctx.paths)) as state:
        state.rerun(args.ref, args.model)
    _say(ctx, f"{args.ref} {args.model}: back to pending")
    return 0


def _ack(ctx: Ctx, args: argparse.Namespace) -> int:
    with closing(open_state(ctx.paths)) as state:
        state.ack(args.run_id, ctx.clock())
    _say(ctx, f"{args.run_id}: acked")
    return 0


def _reset(ctx: Ctx, _args: argparse.Namespace) -> int:
    deleted = reset(ctx.paths)
    _say(ctx, "\n".join(f"deleted {p}" for p in deleted) or "nothing to delete")
    return 0


def _contract(ctx: Ctx, ref: str) -> int:
    spec = _spec_for(load_policy(ctx.policy_path), ref)
    ctx.paths.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with DispatcherLock(ctx.paths.lock):
            result = run_contract(
                spec, ctx.paths.runs_dir, os.environ, clock=ctx.clock, sleep=time.sleep
            )
    except LockHeld:
        return _refuse("the dispatcher is running; stop it first (one job at a time)")
    for s in result.scenarios:
        verdict = "pass" if s.passed else "FAIL"
        _say(
            ctx,
            f"{verdict} {s.name}: exit={s.exit_code} timed_out={s.timed_out} "
            f"result_left={s.result_left} contacted={s.contacted} run={s.run_id}",
        )
    if not result.passed:
        return _refuse(f"{ref}: contract test failed; it cannot be listed")
    with closing(open_state(ctx.paths)) as state:
        state.record_contract_pass(ref, spec.fingerprint, ctx.clock())
    _say(ctx, f"{ref}: contract passed for fingerprint {spec.fingerprint[:12]}")
    return 0


def _run_in(
    argv: Sequence[str], cwd: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(list(argv), 124, "", "timeout")
    except OSError as exc:
        return subprocess.CompletedProcess(list(argv), 127, "", str(exc))


def _start(ctx: Ctx, _args: argparse.Namespace) -> int:
    deps = Deps(
        paths=ctx.paths,
        load_manifest=load_manifest,
        host=real_host(),
        run_in=_run_in,
        start=start_run,
        base_env=dict(os.environ),
        ollama_url=OLLAMA_URL,
        http_json=hosts.http_json,
    )
    for attempt in range(2):
        try:
            return serve(deps, ctx.policy_path, clock=ctx.clock, sleep=time.sleep)
        except LockHeld:
            if attempt:
                return _refuse("a dispatcher is already running")
            time.sleep(1.0)
    return 1


def _run_now(ctx: Ctx, args: argparse.Namespace) -> int:
    policy = load_policy(ctx.policy_path)
    if not any((p.task, p.model) == (args.ref, args.model) for p in policy.jobs):
        return _refuse(f"{args.ref} {args.model}: not listed in policy.yaml; only listed pairs run")
    if not _has_contract_pass(ctx, args.ref, _spec_for(policy, args.ref).fingerprint):
        return _refuse(f"{args.ref}: no contract pass for its current manifest entry")
    if not _dispatcher_up(ctx):
        return _refuse("dispatcher-down: start it first (pastore start)")
    with closing(open_state(ctx.paths)) as state:
        job = state.get_job(args.ref, args.model)
        if job is not None and job.status != "pending":
            return _refuse(
                f"{args.ref} {args.model}: is {job.status}; "
                f"run `pastore rerun {args.ref} --model {args.model}` first"
            )
        state.request_run_now(args.ref, args.model)
    _say(
        ctx,
        f"run-now queued for {args.ref} {args.model}; the dispatcher picks it up within one tick",
    )
    return 0


HANDLERS: dict[str, Callable[[Ctx, argparse.Namespace], int]] = {
    "status": _status,
    "stop": _stop,
    "add": _add,
    "remove": _remove,
    "check": _check,
    "rerun": _rerun,
    "ack": _ack,
    "reset": _reset,
    "start": _start,
    "run-now": _run_now,
}


def _context(
    args: argparse.Namespace,
    home: Path | None,
    policy_path: Path | None,
    clock: Callable[[], datetime] | None,
    out: TextIO | None,
    stdin_isatty: Callable[[], bool] | None,
    http_json: HttpJson | None,
) -> Ctx:
    """The command environment: each injected seam, or its real binding."""
    return Ctx(
        paths=default_paths(home or Path.home()),
        policy_path=args.policy or policy_path or DEFAULT_POLICY,
        clock=clock or (lambda: datetime.now(UTC)),
        out=out or sys.stdout,
        isatty=stdin_isatty or sys.stdin.isatty,
        http_json=http_json or hosts.http_json,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    home: Path | None = None,
    policy_path: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    out: TextIO | None = None,
    stdin_isatty: Callable[[], bool] | None = None,
    http_json: HttpJson | None = None,
) -> int:
    """Run the CLI and return its exit code: 0 ok, 1 refused or invalid, 2 usage."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    ctx = _context(args, home, policy_path, clock, out, stdin_isatty, http_json)
    try:
        return HANDLERS[args.cmd](ctx, args)
    except (PolicyError, ManifestError, StateError, sqlite3.Error, OSError) as exc:
        return _refuse(f"{type(exc).__name__}: {exc}")
