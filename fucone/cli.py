"""The fucone command line: start, stop, status, add/remove (gated), check [--contract], run-now, rerun, ack, reset.

Depends on: policy, manifest, state, dispatcher (serve), contract, checks/runner real bindings; binds the real clock only here.
"""

import argparse
import os
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

from fucone.checks import OLLAMA_URL, real_host
from fucone.contract import run_contract
from fucone.dispatcher import Deps, serve
from fucone.manifest import ManifestError, load_manifest
from fucone.policy import (
    Policy,
    PolicyError,
    load_policy,
    policy_diff,
    render_policy,
    with_job_added,
    with_job_removed,
)
from fucone.runner import start_run
from fucone.spec import TaskSpec
from fucone.state import (
    LockHeld,
    StateError,
    StatePaths,
    default_paths,
    lock_is_held,
    open_state,
    reset,
)

HEARTBEAT_STALE = timedelta(seconds=60)
SUBPROCESS_TIMEOUT_S = 600
DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "policy.yaml"


@dataclass(frozen=True)
class Ctx:
    """Everything a command needs from its environment."""

    paths: StatePaths
    policy_path: Path
    clock: Callable[[], datetime]
    out: TextIO
    isatty: Callable[[], bool]


def _say(ctx: Ctx, text: str = "") -> None:
    print(text, file=ctx.out)


def _refuse(text: str) -> int:
    print(text, file=sys.stderr)
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fucone", description="Night-shift GPU job queue.")
    parser.add_argument("--policy", type=Path, help="policy.yaml to use (default: the repo's)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start")
    sub.add_parser("status")
    sub.add_parser("reset")
    sub.add_parser("stop").add_argument("--now", action="store_true")
    for name in ("add", "remove"):
        cmd = sub.add_parser(name)
        cmd.add_argument("ref")
        cmd.add_argument("--yes", action="store_true")
    check = sub.add_parser("check")
    check.add_argument("path", nargs="?", type=Path)
    check.add_argument("--contract", metavar="REF")
    for name, arg in (("run-now", "ref"), ("rerun", "ref"), ("ack", "run_id")):
        sub.add_parser(name).add_argument(arg)
    return parser


def _dispatcher_up(ctx: Ctx) -> bool:
    if not ctx.paths.db.exists() or not lock_is_held(ctx.paths.lock):
        return False
    with closing(open_state(ctx.paths, readonly=True)) as state:
        beat = state.last_heartbeat()
    return beat is not None and ctx.clock() - beat <= HEARTBEAT_STALE


def _status(ctx: Ctx, _args: argparse.Namespace) -> int:
    if not ctx.paths.db.exists():
        _say(ctx, "no state yet")
        return 0
    up = _dispatcher_up(ctx)
    with closing(open_state(ctx.paths, readonly=True)) as state:
        _say(ctx, f"dispatcher: {'up' if up else 'dispatcher-down'}")
        _say(ctx, f"idle: {state.idle_reason() or 'none'}")
        try:
            jobs = load_policy(ctx.policy_path).jobs
        except PolicyError as exc:
            _say(ctx, f"policy error: {exc}")
            return 1
        for ref in jobs:
            job, run = state.get_job(ref), state.last_run(ref)
            line = f"{ref}: {job.status if job else 'unseen'} attempts={job.attempts if job else 0}"
            line += f" last_run={run.status if run else 'none'}"
            _say(ctx, line + (" starved" if job and job.starved_flagged else ""))
        for run in state.runs_with_status("awaiting-review"):
            _say(ctx, f"awaiting review: {run.run_id} ({run.ref})")
    return 0


def _stop(ctx: Ctx, args: argparse.Namespace) -> int:
    mode: Literal["graceful", "now"] = "now" if args.now else "graceful"
    with closing(open_state(ctx.paths)) as state:
        state.request_stop(mode)
    _say(ctx, f"stop requested ({mode})")
    return 0


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".policy-")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
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
    new = with_job_added(policy, args.ref)
    if not _has_contract_pass(ctx, args.ref, _spec_for(policy, args.ref).fingerprint):
        return _refuse(
            f"{args.ref}: no passing contract test for its current manifest entry; "
            f"run: fucone check --contract {args.ref}"
        )
    return _confirm_and_write(ctx, new, args.yes)


def _remove(ctx: Ctx, args: argparse.Namespace) -> int:
    return _confirm_and_write(
        ctx, with_job_removed(load_policy(ctx.policy_path), args.ref), args.yes
    )


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
        state.rerun(args.ref)
    _say(ctx, f"{args.ref}: back to pending")
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
    result = run_contract(spec, ctx.paths.runs_dir, os.environ, clock=ctx.clock, sleep=time.sleep)
    for s in result.scenarios:
        verdict = "pass" if s.passed else "FAIL"
        _say(
            ctx,
            f"{verdict} {s.name}: exit={s.exit_code} timed_out={s.timed_out} result_left={s.result_left} run={s.run_id}",
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


def _start(ctx: Ctx, _args: argparse.Namespace) -> int:
    deps = Deps(
        paths=ctx.paths,
        load_manifest=load_manifest,
        host=real_host(),
        run_in=_run_in,
        start=start_run,
        base_env=dict(os.environ),
        ollama_url=OLLAMA_URL,
        local_tz=None,
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
    if args.ref not in policy.jobs:
        return _refuse(f"{args.ref}: not listed in policy.yaml; only listed jobs run")
    if not _dispatcher_up(ctx):
        return _refuse("dispatcher-down: start it first (fucone start)")
    with closing(open_state(ctx.paths)) as state:
        state.request_run_now(args.ref)
    _say(ctx, f"run-now queued for {args.ref}; the dispatcher picks it up within one tick")
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


def main(
    argv: Sequence[str] | None = None,
    *,
    home: Path | None = None,
    policy_path: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    out: TextIO | None = None,
    stdin_isatty: Callable[[], bool] | None = None,
) -> int:
    """Run the CLI and return its exit code: 0 ok, 1 refused or invalid, 2 usage."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    ctx = Ctx(
        paths=default_paths(home or Path.home()),
        policy_path=args.policy or policy_path or DEFAULT_POLICY,
        clock=clock or (lambda: datetime.now(UTC)),
        out=out or sys.stdout,
        isatty=stdin_isatty or sys.stdin.isatty,
    )
    try:
        return HANDLERS[args.cmd](ctx, args)
    except (PolicyError, ManifestError, StateError) as exc:
        return _refuse(str(exc))
