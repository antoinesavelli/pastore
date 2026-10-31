"""`python -m pastore.gateway`: load the policy (or its last good copy), bind the real clock, `/proc` and stdout, and serve the front door (R20).

Depends on: pastore.policy, pastore.state, pastore.gateway.attribution and server, aiohttp.web.
"""

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aiohttp import web

from pastore.gateway.attribution import Attributor
from pastore.gateway.server import make_app
from pastore.policy import Policy, PolicyError, parse_policy
from pastore.state import default_paths

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "policy.yaml"
COPY_NAME = "gateway-policy.yaml"


def _log(line: str) -> None:
    print(line, flush=True)


def _warn(text: str) -> None:
    """Print to stderr at journald's warning level (`<4>`)."""
    print(f"<4>pastore gateway: {text}", file=sys.stderr)


def _read(path: Path) -> tuple[bytes, Policy]:
    """Read the file once and parse exactly those bytes; PolicyError when it cannot be read or is invalid."""
    try:
        data = path.read_bytes()
        return data, parse_policy(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyError(f"cannot read {path}: {exc}") from exc


def _keep_copy(copy: Path, data: bytes) -> None:
    """Make `copy` hold `data` (atomic replace, only when it differs); warn and go on if the directory is missing or the write fails."""
    try:
        try:
            if copy.read_bytes() == data:
                return
        except FileNotFoundError:
            pass
        tmp = copy.with_name(copy.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, copy)
    except OSError as exc:
        _warn(f"could not keep the last good policy copy {copy}: {exc}")


def _load(path: Path, copy: Path) -> tuple[Policy, Path] | None:
    """Load `path` and refresh `copy`, else load `copy` (D-050 (5)); None, after printing both reasons, when neither loads."""
    try:
        data, policy = _read(path)
    except PolicyError as exc:
        try:
            policy = _read(copy)[1]
        except PolicyError as copy_exc:
            print(
                f"pastore gateway: {exc}\npastore gateway: last good copy: {copy_exc}",
                file=sys.stderr,
            )
            return None
        _warn(f"policy {path} failed to load ({exc}); started on the last good copy {copy}")
        return policy, copy
    _keep_copy(copy, data)
    return policy, path


def main(argv: list[str] | None = None) -> int:
    """Serve until stopped; exit 2 when neither the policy nor its last good copy loads."""
    parser = argparse.ArgumentParser(
        prog="python -m pastore.gateway", description="pastore's Ollama front door."
    )
    parser.add_argument(
        "--policy", type=Path, default=DEFAULT_POLICY, help="v2 policy (default: the repo's)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args(argv)
    paths = default_paths(Path.home())
    loaded = _load(args.policy, paths.state_dir / COPY_NAME)
    if loaded is None:
        return 2
    policy, source = loaded
    app = make_app(
        policy,
        Attributor(policy.sources, time.monotonic),
        state_paths=paths,
        clock=time.monotonic,
        log=_log,
        started_at=datetime.now(UTC).isoformat(),
    )
    _log(f"gateway listening {args.host}:{args.port} policy={source}")
    web.run_app(
        app, host=args.host, port=args.port, handler_cancellation=True, access_log=None, print=None
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
