"""Test doubles for the gateway: an Ollama-like upstream on 127.0.0.1, a scripted attributor, a refused URL.

Depends on: aiohttp, pastore.gateway.attribution and stdlib only; binds port 0 on 127.0.0.1, never 11434 to 11436.
"""

import asyncio
import gzip
import json
import socket
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Self

from aiohttp import web

from pastore.gateway.attribution import Address, Attribution

Mode = Literal["ok", "503", "cut"]
STREAMS = {"/api/generate", "/api/chat", "/v1/chat/completions"}
SERVER = "fake-ollama"
INFERENCE = STREAMS | {"/api/embed", "/api/embeddings"}


@dataclass(frozen=True)
class Recorded:
    """One request the fake received: method, raw path with query, headers and body."""

    method: str
    target: str
    headers: dict[str, str]
    body: bytes


class ScriptedAttribution:
    """An attribution callable that answers each call with the next scripted `Attribution`, or raises a scripted error."""

    def __init__(self, script: Sequence[Attribution | Exception]) -> None:
        self._script = deque(script)
        self.calls: list[tuple[Address, Address]] = []
        self.threads: list[int] = []

    def __call__(self, peer: Address, local: Address) -> Attribution:
        self.calls.append((peer, local))
        self.threads.append(threading.get_ident())
        if not self._script:
            raise AssertionError("the attribution script ran out")
        step = self._script.popleft()
        if isinstance(step, Exception):
            raise step
        return step


def refused_url() -> str:
    """A 127.0.0.1 URL whose port nothing listens on: bind a free port, then close it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


async def serve(
    app: web.Application, *, handler_cancellation: bool = False
) -> tuple[web.AppRunner, str]:
    """Start `app` on 127.0.0.1 port 0 and return its runner and base URL."""
    runner = web.AppRunner(
        app, handler_cancellation=handler_cancellation, access_log=None, shutdown_timeout=1.0
    )
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


class FakeStream(web.StreamResponse):
    """The fake's streaming response; a distinct type so a test can patch the gateway's plain StreamResponse only."""


def _wants_stream(body: bytes) -> bool:
    try:
        data = json.loads(body)
    except ValueError:
        return True
    return not (isinstance(data, dict) and data.get("stream") is False)


class FakeUpstream:
    """An Ollama-like server: NDJSON and SSE streams, JSON answers, a 503 mode, a cut mode, a hold mode.

    `compress` gzips JSON answers; `hold` makes an inference request wait on `release` before it answers and `stall`
    does the same for every request; `pause_after=k` makes a stream wait on `release` after k chunks; `aborted` counts
    streams whose reader went away.
    """

    def __init__(
        self, tag: str = "fake", *, mode: Mode = "ok", chunks: int = 4, delay_s: float = 0.005
    ):
        self.tag = tag
        self.mode: Mode = mode
        self.chunks = chunks
        self.delay_s = delay_s
        self.hold = False
        self.stall = False
        self.compress = False
        self.pause_after: int | None = None
        self.release = asyncio.Event()
        self.requests: list[Recorded] = []
        self.aborted = 0
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> Self:
        app = web.Application(client_max_size=0, handler_args={"auto_decompress": False})
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner, self.url = await serve(app)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release.set()
        if self._runner is not None:
            await self._runner.cleanup()

    async def wait_for_requests(self, count: int, timeout_s: float = 5.0) -> None:
        """Return once the fake has received `count` requests; fail the test after `timeout_s`."""
        async with asyncio.timeout(timeout_s):
            while len(self.requests) < count:
                await asyncio.sleep(0.005)

    def prompts(self) -> list[str]:
        """The `prompt` of each JSON body the fake received, in arrival order."""
        return [json.loads(r.body)["prompt"] for r in self.requests]

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        self.requests.append(
            Recorded(request.method, request.raw_path, dict(request.headers), body)
        )
        inference = request.method == "POST" and request.path in INFERENCE
        if self.stall or (self.hold and inference):
            await self.release.wait()
        if self.mode == "503":
            return web.json_response(
                {"error": "overloaded"}, status=503, headers={"Server": SERVER}
            )
        streams = request.method == "POST" and request.path in STREAMS
        if streams and _wants_stream(body):
            return await self._stream(request)
        answer = json.dumps({"tag": self.tag, "target": request.raw_path}).encode()
        headers = {"X-Upstream-Tag": self.tag, "Server": SERVER, "Content-Type": "application/json"}
        if self.compress:
            headers["Content-Encoding"] = "gzip"
            answer = gzip.compress(answer, mtime=0)
        return web.Response(body=answer, headers=headers)

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        sse = request.path.startswith("/v1/")
        kind = "text/event-stream" if sse else "application/x-ndjson"
        response = FakeStream(
            headers={"Content-Type": kind, "X-Upstream-Tag": self.tag, "Server": SERVER}
        )
        await response.prepare(request)
        try:
            for index in range(self.chunks):
                if index == self.pause_after:
                    await self.release.wait()
                await response.write(self._chunk(index, sse))
                await asyncio.sleep(self.delay_s)
            if self.mode == "cut":
                self._abort(request)
                return response
            await response.write(b"data: [DONE]\n\n" if sse else b'{"done": true}\n')
        except ConnectionResetError:
            self.aborted += 1
        return response

    def _chunk(self, index: int, sse: bool) -> bytes:
        payload = json.dumps({"tag": self.tag, "token": f"t{index}"})
        return f"data: {payload}\n\n".encode() if sse else f"{payload}\n".encode()

    @staticmethod
    def _abort(request: web.Request) -> None:
        if request.transport is not None:
            request.transport.abort()
