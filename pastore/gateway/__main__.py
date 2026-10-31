"""`python -m pastore.gateway`: load the policy, bind the real clock, `/proc` and stdout, and serve the front door (R20).

Depends on: pastore.policy, pastore.state, pastore.gateway.attribution and server, aiohttp.web.
"""

import argparse
import sys
import time
from pathlib import Path

from aiohttp import web

from pastore.gateway.attribution import Attributor
from pastore.gateway.server import make_app
from pastore.policy import PolicyError, load_policy
from pastore.state import default_paths

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "policy.yaml"


def _log(line: str) -> None:
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Serve until stopped; exit 2 when the policy is unreadable or invalid."""
    parser = argparse.ArgumentParser(
        prog="python -m pastore.gateway", description="pastore's Ollama front door."
    )
    parser.add_argument(
        "--policy", type=Path, default=DEFAULT_POLICY, help="v2 policy (default: the repo's)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
    except PolicyError as exc:
        print(f"pastore gateway: {exc}", file=sys.stderr)
        return 2
    app = make_app(
        policy,
        Attributor(policy.sources, time.monotonic),
        state_paths=default_paths(Path.home()),
        clock=time.monotonic,
        log=_log,
    )
    _log(f"gateway listening {args.host}:{args.port} policy={args.policy}")
    web.run_app(
        app, host=args.host, port=args.port, handler_cancellation=True, access_log=None, print=None
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
