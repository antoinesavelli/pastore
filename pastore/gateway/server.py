"""The Ollama gateway's streaming proxy (R20 to R24, R32): attribute, route, queue, stream through, journal.

Depends on: aiohttp, pastore.gateway.attribution and .queue, pastore.policy, pastore.spec and pastore.state (read-only).
"""

import asyncio
import ipaddress
import json
import posixpath
import re
import sqlite3
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
from aiohttp import web
from yarl import URL

from pastore.gateway.attribution import Address, Attribution, is_local
from pastore.gateway.queue import STARVE_AFTER_S, HostQueue
from pastore.policy import Policy
from pastore.spec import WORKSTATION, Level
from pastore.state import StateError, StatePaths, open_state

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
DROP_REQUEST = HOP_BY_HOP | {"host", "content-length"}
DROP_RESPONSE = HOP_BY_HOP | {"content-length", "date", "server"}
GATED_PATHS = frozenset({"/api/generate", "/api/chat", "/api/embed", "/api/embeddings"})
GATED_PREFIX = "/v1/"
REFUSED_PATHS = frozenset(
    {
        "/api/pull",
        "/api/delete",
        "/api/create",
        "/api/copy",
        "/api/push",
        "/api/signout",
        "/api/experimental/web_fetch",
        "/api/experimental/web_search",
    }
)
REFUSED_PREFIXES = ("/api/blobs/", "/api/user/keys/")
DOCKER_NET = ipaddress.IPv4Network("172.17.0.0/16")
SKIP_AUTO_HEADERS = ("Accept", "Accept-Encoding", "Content-Type", "User-Agent")
CONNECT_TIMEOUT_S = 3.0
UPSTREAM_IDLE_S = 300.0
MAX_BODY_BYTES = 64 * 1024 * 1024
MODEL_PARSE_MAX_BYTES = 1024 * 1024
CLIENT_CLOSED = 499
STATUS_PATH = "/pastore/status"
UNSAFE = re.compile(r"[^A-Za-z0-9._:/@+-]")
SESSION = web.AppKey("session", aiohttp.ClientSession)
UPSTREAM_IDLE = web.AppKey("upstream_idle_s", float)


@dataclass
class _Journal:
    """What one request knows about itself so far; `_Gateway._gated` writes it as the R32 line when the request ends."""

    level: str = "-"
    source: str = "-"
    model: str = "-"
    host: str = "-"
    granted: float | None = None
    status: str = "500"


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """`host` as an address, an IPv4-mapped IPv6 one read as its IPv4 address; None when it does not parse."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    return ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip


def is_loopback(host: str) -> bool:
    """True for 127.0.0.0/8, ::1 and ::ffff:127.x.x.x; an address that does not parse is not loopback."""
    ip = _ip(host)
    return ip is not None and ip.is_loopback


def on_this_machine(peer: str, local: str, proc: Path) -> bool:
    """D-050 (9c): True for a peer in 172.17.0.0/16 or one the attribution calls local (`is_local`, reading `proc`)."""
    ip = _ip(peer)
    return ip is not None and (ip in DOCKER_NET or is_local(ip, _ip(local), proc))


def _token(value: str) -> str:
    """A space-free journal value: unsafe characters become `_`, at most 100 characters, `-` when empty."""
    return UNSAFE.sub("_", value)[:100] or "-"


def _model_of(body: bytes) -> str:
    """The body's `model` as a journal token; `-` when it is absent, not a string, not JSON or too big to parse."""
    if len(body) > MODEL_PARSE_MAX_BYTES:
        return "-"
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return "-"
    model = data.get("model") if isinstance(data, dict) else None
    return _token(model) if isinstance(model, str) else "-"


def _passed(headers: Mapping[str, str], drop: frozenset[str]) -> list[tuple[str, str]]:
    return [(name, value) for name, value in headers.items() if name.lower() not in drop]


def _endpoints(request: web.Request) -> tuple[Address, Address]:
    transport = request.transport
    if transport is None:
        raise ConnectionResetError("the client is already gone")
    peer, local = transport.get_extra_info("peername"), transport.get_extra_info("sockname")
    return (peer[0], peer[1]), (local[0], local[1])


def _failure(exc: Exception) -> str:
    """The journal status of an upstream failure: `upstream-idle` for the idle limit, else `upstream-error`."""
    return "upstream-idle" if isinstance(exc, aiohttp.SocketTimeoutError) else "upstream-error"


async def _pump(
    request: web.Request,
    upstream: aiohttp.ClientResponse,
    response: web.StreamResponse,
    journal: _Journal,
) -> None:
    """Send the headers, then copy the upstream body to the client as it arrives; record how it ended."""
    try:
        await response.prepare(request)
        async for chunk in upstream.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
    except ConnectionResetError:
        journal.status = "hangup"
    except (aiohttp.ClientError, TimeoutError) as exc:
        journal.status = _failure(exc)
        if request.transport is not None:
            request.transport.close()
    else:
        journal.status = str(response.status)


class _Gateway:
    """The per-app state of the gateway: a queue per host, the attribution callable, the journal."""

    def __init__(
        self,
        policy: Policy,
        attribute: Callable[[Address, Address], Attribution],
        state_paths: StatePaths,
        clock: Callable[[], float],
        log: Callable[[str], None],
        starve_after_s: float,
        started_at: str,
        proc: Path,
    ) -> None:
        self._policy = policy
        self._started_at = started_at
        self._proc = proc
        self._attribute = attribute
        self._state_paths = state_paths
        self._clock = clock
        self._log = log
        self._queues = {
            name: HostQueue(
                name,
                host.parallel * host.max_models,
                clock,
                self._starved,
                starve_after_s=starve_after_s,
            )
            for name, host in policy.hosts.items()
        }

    def _starved(self, host: str, level: Level, label: str, waited_s: float) -> None:
        name = level.name.lower()
        self._log(
            f"starved host={host} level={name} source={label} waited_ms={int(waited_s * 1000)}"
        )

    async def handle(self, request: web.Request) -> web.StreamResponse:
        """Serve one request: the status page, a gated inference request, a refusal or an ungated pass-through."""
        if not request.raw_path.startswith("/"):
            raise web.HTTPBadRequest()
        path = request.path
        if path == "/pastore" or path.startswith("/pastore/"):
            return self._pastore(request)
        if request.method == "POST" and (path in GATED_PATHS or path.startswith(GATED_PREFIX)):
            return await self._gated(request)
        if (refusal := await self._refuse_off_machine(request, path)) is not None:
            return refusal
        data = request.content if request.can_read_body else None
        return await self._relay(request, self._policy.hosts[WORKSTATION].url, data, _Journal())

    def _pastore(self, request: web.Request) -> web.Response:
        if request.method != "GET" or request.path != STATUS_PATH:
            raise web.HTTPNotFound()
        peername = request.transport.get_extra_info("peername") if request.transport else None
        if not peername or not is_loopback(peername[0]):
            raise web.HTTPForbidden()
        hosts = {name: asdict(queue.snapshot()) for name, queue in self._queues.items()}
        return web.json_response({"hosts": hosts, "started_at": self._started_at})

    async def _refuse_off_machine(self, request: web.Request, path: str) -> web.Response | None:
        """403 for a model-changing, account or web-fetching route from a peer that is not on this machine (D-050 (9c), D-052 (4)); None for anything else."""
        route = posixpath.normpath(re.sub("/+", "/", path))
        if route not in REFUSED_PATHS and not f"{route}/".startswith(REFUSED_PREFIXES):
            return None
        try:
            (peer, _), (local, _) = _endpoints(request)
        except ConnectionResetError:
            peer = local = ""
        if await asyncio.get_running_loop().run_in_executor(
            None, on_this_machine, peer, local, self._proc
        ):
            return None
        self._log(f"refused route={_token(path)} peer={_token(peer)}")
        error = "this route is refused from off this machine"
        return web.json_response({"error": error}, status=403)

    async def _gated(self, request: web.Request) -> web.StreamResponse:
        started = self._clock()
        journal = _Journal()
        try:
            peer, local = _endpoints(request)
            body = await request.read()
            loop = asyncio.get_running_loop()
            journal.model = await loop.run_in_executor(None, _model_of, body)
            who = await loop.run_in_executor(None, self._attribute, peer, local)
            journal.level, journal.source = who.level.name.lower(), _token(who.source)
            journal.host = await loop.run_in_executor(None, self._host_for, who)
            queue = self._queues[journal.host]
            grant = await queue.acquire(who.level, journal.source)
            journal.granted = self._clock()
            try:
                return await self._relay(
                    request, self._policy.hosts[journal.host].url, body, journal
                )
            finally:
                queue.release(grant)
        except asyncio.CancelledError:
            journal.status = "hangup"
            raise
        except ConnectionResetError:
            journal.status = "hangup"
            return web.Response(status=CLIENT_CLOSED)
        except web.HTTPException as exc:
            journal.status = str(exc.status)
            raise
        finally:
            self._write_journal(journal, started)

    def _write_journal(self, journal: _Journal, started: float) -> None:
        now = self._clock()
        granted = now if journal.granted is None else journal.granted
        self._log(
            f"request level={journal.level} source={journal.source} model={journal.model}"
            f" host={journal.host} wait_ms={int((granted - started) * 1000)}"
            f" run_ms={int((now - granted) * 1000)} status={journal.status}"
        )

    def _host_for(self, who: Attribution) -> str:
        """R24: levels 1-5 go to the workstation; hearth goes to its run's host, else the workstation."""
        if who.level is not Level.HEARTH or who.run_id is None:
            return WORKSTATION
        try:
            state = open_state(self._state_paths, readonly=True)
            try:
                run = state.get_run(who.run_id)
            finally:
                state.close()
        except (StateError, sqlite3.Error, OSError):
            return WORKSTATION
        return run.host if run is not None and run.host in self._policy.hosts else WORKSTATION

    async def _relay(
        self,
        request: web.Request,
        base_url: str,
        data: bytes | aiohttp.StreamReader | None,
        journal: _Journal,
    ) -> web.StreamResponse:
        """Forward the request to the host and stream its answer back; 502 if it cannot be reached."""
        url = URL(base_url + request.raw_path, encoded=True)
        try:
            upstream = await request.app[SESSION].request(
                request.method,
                url,
                headers=_passed(request.headers, DROP_REQUEST),
                data=data,
                allow_redirects=False,
            )
        except (aiohttp.ClientError, TimeoutError) as exc:
            journal.status = _failure(exc)
            if journal.status == "upstream-idle":
                idle = f"upstream idle: no bytes from Ollama for {request.app[UPSTREAM_IDLE]:g} s"
                return web.json_response({"error": idle}, status=504)
            return web.json_response({"error": f"upstream unreachable: {exc}"}, status=502)
        async with upstream:
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=_passed(upstream.headers, DROP_RESPONSE),
            )
            await _pump(request, upstream, response, journal)
        return response


async def _upstream_session(app: web.Application) -> AsyncIterator[None]:
    """One upstream client for the app's life: no pool limit, 3 s to connect, no total timeout, the idle limit per read."""
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(
            total=None, connect=CONNECT_TIMEOUT_S, sock_read=app[UPSTREAM_IDLE]
        ),
        auto_decompress=False,
        skip_auto_headers=SKIP_AUTO_HEADERS,
    )
    app[SESSION] = session
    yield
    await session.close()


def make_app(
    policy: Policy,
    attribute: Callable[[Address, Address], Attribution],
    *,
    state_paths: StatePaths,
    clock: Callable[[], float],
    log: Callable[[str], None],
    started_at: str,
    starve_after_s: float = STARVE_AFTER_S,
    upstream_idle_s: float = UPSTREAM_IDLE_S,
    proc: Path = Path("/proc"),
) -> web.Application:
    """Build the gateway application; the caller runs it with `handler_cancellation=True`.

    `started_at` is the gateway's start as UTC ISO-8601; `/pastore/status` reports it, so the dispatcher can tell a
    run that a gateway restart cut short (D-050 (4)). A relayed reply that gets no bytes from the upstream for
    `upstream_idle_s` is ended. `proc` is where the refused routes' peer check reads the host's own addresses.
    """
    gateway = _Gateway(policy, attribute, state_paths, clock, log, starve_after_s, started_at, proc)
    app = web.Application(client_max_size=MAX_BODY_BYTES, handler_args={"auto_decompress": False})
    app[UPSTREAM_IDLE] = upstream_idle_s
    app.cleanup_ctx.append(_upstream_session)
    app.router.add_route("*", "/{tail:.*}", gateway.handle)
    return app
