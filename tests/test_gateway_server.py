"""Pins the gateway proxy (R20-R24, R32; checks 1-5 and 7): pass-through, queueing, routing, the journal, the status page.

Depends on: pastore.gateway.server with the real queue, tests.gateway_fakes, aiohttp and pytest; binds 127.0.0.1 port 0 only.
"""

import asyncio
import gzip
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import Mock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

import pastore.gateway.server as server
from pastore.gateway.attribution import Attribution
from pastore.gateway.server import (
    CLIENT_CLOSED,
    SESSION,
    UPSTREAM_IDLE_S,
    is_loopback,
    make_app,
    on_this_machine,
)
from pastore.policy import Policy
from pastore.spec import WORKSTATION, Host, Level
from pastore.state import RunRow, StatePaths, default_paths, open_state
from tests.gateway_fakes import FakeUpstream, ScriptedAttribution, refused_url, serve

LEVEL_NAMES = ("reins", "webui", "coding", "corsu", "routine", "hearth")
CHAT = {"model": "m", "prompt": "hi"}
MIB = 1024 * 1024
STARTED_AT = "2026-10-31T12:00:00+00:00"


class Reply(NamedTuple):
    status: int
    content_type: str | None
    body: bytes
    headers: Any


class Clock:
    """A float clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def who(level: Level, source: str = "src", run_id: str | None = None) -> Attribution:
    return Attribution(level, source, run_id)


def run(scenario: Coroutine[Any, Any, None]) -> None:
    asyncio.run(asyncio.wait_for(scenario, 30))


async def raw_exchange(url: str, data: bytes) -> bytes:
    """Send raw bytes to `url` (the request must say `Connection: close`) and return everything it answers."""
    host, port = url.removeprefix("http://").split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(data)
    answer = await asyncio.wait_for(reader.read(), 10)
    writer.close()
    return answer


@dataclass
class Rig:
    """A gateway on 127.0.0.1 in front of fake upstreams, plus a client and everything the tests read back."""

    url: str
    fakes: dict[str, FakeUpstream]
    lines: list[str]
    client: aiohttp.ClientSession
    attribution: ScriptedAttribution
    app: web.Application
    cancelled: list[str]
    tasks: list["asyncio.Task[Reply]"] = field(default_factory=list)

    async def send(
        self,
        method: str,
        path: str,
        body: object = None,
        *,
        base: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Reply:
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        url = (base or self.url) + path
        async with self.client.request(method, url, data=data, headers=headers) as resp:
            return Reply(
                resp.status, resp.headers.get("Content-Type"), await resp.read(), resp.headers
            )

    def spawn(self, method: str, path: str, body: object = None) -> "asyncio.Task[Reply]":
        task = asyncio.create_task(self.send(method, path, body))
        self.tasks.append(task)
        return task

    def chat(self, prompt: str, model: str = "m") -> "asyncio.Task[Reply]":
        return self.spawn("POST", "/api/chat", {"model": model, "prompt": prompt})

    async def status(self) -> dict[str, Any]:
        reply = await self.send("GET", "/pastore/status")
        assert reply.status == 200
        result: dict[str, Any] = json.loads(reply.body)
        return result

    async def wait_queued(self, level: str, count: int, host: str = WORKSTATION) -> None:
        async with asyncio.timeout(5):
            while (await self.status())["hosts"][host]["queued"][level] != count:
                await asyncio.sleep(0.005)

    async def wait_idle(self, host: str = WORKSTATION) -> None:
        async with asyncio.timeout(5):
            while (state := (await self.status())["hosts"][host])["in_flight"] or any(
                state["queued"].values()
            ):
                await asyncio.sleep(0.005)

    async def wait_journal(self, count: int) -> None:
        async with asyncio.timeout(5):
            while len(self.journal()) < count:
                await asyncio.sleep(0.005)

    def journal(self) -> list[dict[str, str]]:
        return [
            dict(part.split("=", 1) for part in line.split()[1:])
            for line in self.lines
            if line.startswith("request ")
        ]

    def entry(self, model: str) -> dict[str, str]:
        (found,) = (e for e in self.journal() if e["model"] == model)
        return found


@asynccontextmanager
async def rig(
    tmp_path: Path,
    script: list[Attribution | Exception] | None = None,
    *,
    extra_hosts: tuple[str, ...] = (),
    parallel: int = 1,
    max_models: int = 1,
    workstation_url: str | None = None,
    starve_after_s: float = 600.0,
    clock: Callable[[], float] = time.monotonic,
    state_paths: StatePaths | None = None,
    upstream_idle_s: float = UPSTREAM_IDLE_S,
    proc: Path | None = None,
) -> AsyncIterator[Rig]:
    async with AsyncExitStack() as stack:
        fakes = {
            name: await stack.enter_async_context(FakeUpstream(name))
            for name in (WORKSTATION, *extra_hosts)
        }
        urls = {name: fake.url for name, fake in fakes.items()}
        urls[WORKSTATION] = workstation_url or urls[WORKSTATION]
        hosts = {n: Host(n, u, 24, max_models, parallel, 32768) for n, u in urls.items()}
        lines: list[str] = []
        attribution = ScriptedAttribution(script or [])
        app = make_app(
            Policy(hosts, (), {}, ()),
            attribution,
            state_paths=state_paths or default_paths(tmp_path),
            clock=clock,
            log=lines.append,
            started_at=STARTED_AT,
            starve_after_s=starve_after_s,
            upstream_idle_s=upstream_idle_s,
            proc=proc or tmp_path / "no-proc",
        )
        cancelled: list[str] = []

        @web.middleware
        async def spy(request: web.Request, handler: Any) -> web.StreamResponse:
            try:
                response: web.StreamResponse = await handler(request)
            except asyncio.CancelledError:
                cancelled.append(request.path)
                raise
            return response

        app.middlewares.append(spy)
        runner, url = await serve(app, handler_cancellation=True)
        stack.push_async_callback(runner.cleanup)
        client = await stack.enter_async_context(aiohttp.ClientSession(auto_decompress=False))
        current = Rig(url, fakes, lines, client, attribution, app, cancelled)

        async def finish() -> None:
            for fake in fakes.values():
                fake.release.set()
            for task in current.tasks:
                task.cancel()
            await asyncio.gather(*current.tasks, return_exceptions=True)

        stack.push_async_callback(finish)
        yield current


# --- check 1: pass-through -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body", "kind"),
    [
        ("/api/generate?a=1&b=%20x", CHAT, "application/x-ndjson"),
        ("/api/chat", CHAT, "application/x-ndjson"),
        ("/v1/chat/completions", CHAT, "text/event-stream"),
        ("/api/chat", {**CHAT, "stream": False}, "application/json"),
        ("/api/embed", {"model": "m", "input": "x"}, "application/json"),
    ],
)
def test_check1_answers_through_the_gateway_are_byte_equal_to_a_direct_call(
    tmp_path: Path, path: str, body: dict[str, Any], kind: str
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            fake = r.fakes[WORKSTATION]
            direct = await r.send("POST", path, body, base=fake.url)
            via = await r.send("POST", path, body)
            assert via.body == direct.body and via.body
            assert (via.status, via.content_type) == (direct.status, direct.content_type)
            assert via.content_type is not None and via.content_type.startswith(kind)
            assert via.headers["X-Upstream-Tag"] == "workstation"
            assert via.headers["Server"] != "fake-ollama" and "Content-Length" not in via.headers
            first, second = fake.requests
            assert (first.method, first.target, first.body) == (
                second.method,
                second.target,
                second.body,
            )
            assert first.headers.keys() == second.headers.keys()
            assert {**first.headers, "Host": ""} == {**second.headers, "Host": ""}

    run(scenario())


def test_check1_the_upstream_gets_the_callers_headers_and_no_others(tmp_path: Path) -> None:
    sent = {
        "X-Trace": "abc",
        "Authorization": "Bearer t",
        "Keep-Alive": "timeout=5",
        "Proxy-Connection": "keep-alive",
        "TE": "trailers",
        "Trailer": "X-End",
        "Upgrade": "h2c",
    }

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            fake = r.fakes[WORKSTATION]
            bare = aiohttp.ClientSession(
                skip_auto_headers=("Accept", "Accept-Encoding", "Content-Type", "User-Agent")
            )
            async with bare, bare.post(r.url + "/api/chat", data=b"{}", headers=sent) as resp:
                await resp.read()
            (seen,) = fake.requests
            assert {k.lower(): v for k, v in seen.headers.items()} == {
                "host": fake.url.removeprefix("http://"),
                "content-length": "2",
                "x-trace": "abc",
                "authorization": "Bearer t",
            }

    run(scenario())


def test_check1_the_first_token_reaches_the_client_while_the_upstream_is_still_streaming(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            fake = r.fakes[WORKSTATION]
            fake.pause_after = 1
            async with r.client.post(r.url + "/api/chat", data=json.dumps(CHAT)) as resp:
                async with asyncio.timeout(3):
                    first = await resp.content.readline()
                assert json.loads(first)["token"] == "t0"
                fake.release.set()
                rest = await resp.read()
            assert rest.endswith(b'{"done": true}\n')

    run(scenario())


def test_the_attribution_gets_both_socket_ends_and_runs_off_the_event_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            await r.send("POST", "/api/chat", CHAT)
            ((peer, local),) = r.attribution.calls
            assert peer[0] == "127.0.0.1" and local == ("127.0.0.1", int(r.url.rsplit(":", 1)[1]))
            assert r.attribution.threads != [threading.get_ident()]

    run(scenario())


def test_check1_a_gzip_answer_is_passed_on_still_compressed(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            fake = r.fakes[WORKSTATION]
            fake.compress = True
            direct = await r.send("GET", "/api/tags", base=fake.url)
            via = await r.send("GET", "/api/tags")
            assert via.body == direct.body and via.body[:2] == b"\x1f\x8b"
            assert via.headers["Content-Encoding"] == "gzip"

    run(scenario())


def test_the_raw_path_and_query_reach_the_upstream_unchanged(tmp_path: Path) -> None:
    target = "/a/%2e%2e/./b%2Fc/d?x=%zz&y=a+b&z=%E2%82%AC&&k"

    async def scenario() -> None:
        async with rig(tmp_path) as r:
            request = f"GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
            assert (await raw_exchange(r.url, request.encode())).startswith(b"HTTP/1.1 200")
            assert [x.target for x in r.fakes[WORKSTATION].requests] == [target]

    run(scenario())


# --- request bodies and targets --------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/chat", "/api/create"])
def test_a_gzip_request_body_reaches_the_upstream_byte_equal_on_gated_and_ungated_routes(
    tmp_path: Path, path: str
) -> None:
    packed = gzip.compress(json.dumps({**CHAT, "stream": False}).encode(), mtime=0)
    headers = {"Content-Encoding": "gzip", "Content-Type": "application/json"}

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            fake = r.fakes[WORKSTATION]
            await r.send("POST", path, packed, base=fake.url, headers=headers)
            await r.send("POST", path, packed, headers=headers)
            direct, via = fake.requests
            assert direct.body == via.body == packed
            assert direct.headers["Content-Encoding"] == via.headers["Content-Encoding"] == "gzip"

    run(scenario())


@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.parametrize(("extra", "parsed"), [(0, True), (1, False)])
def test_the_journal_model_is_read_from_a_body_of_exactly_one_mib_but_not_one_byte_more(
    tmp_path: Path, extra: int, parsed: bool
) -> None:
    head, tail = b'{"model": "m-edge", "pad": "', b'"}'
    body = head + b"x" * (MIB - len(head) - len(tail) + extra) + tail

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            assert (await r.send("POST", "/api/embed", body)).status == 200
            (entry,) = r.journal()
            assert len(body) == MIB + extra
            assert entry["model"] == ("m-edge" if parsed else "-")

    run(scenario())


def test_the_body_is_parsed_for_its_model_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    real = server._model_of

    def spy(body: bytes) -> str:
        seen.append(threading.get_ident())
        return real(body)

    monkeypatch.setattr(server, "_model_of", spy)

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            await r.send("POST", "/api/chat", CHAT)
            assert seen and threading.get_ident() not in seen
            assert r.journal()[0]["model"] == "m"

    run(scenario())


def test_a_gated_body_over_64_mib_gets_413_with_one_journal_line(tmp_path: Path) -> None:
    async def body() -> AsyncIterator[bytes]:
        chunk = b"x" * MIB
        for _ in range(65):
            yield chunk

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            async with r.client.post(r.url + "/api/embed", data=body()) as resp:
                assert resp.status == 413
            await r.wait_journal(1)
            (entry,) = r.journal()
            assert entry["status"] == "413" and entry["host"] == "-"
            assert not r.attribution.calls and not r.fakes[WORKSTATION].requests
            await r.wait_idle()

    run(scenario())


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_a_gated_body_of_exactly_64_mib_is_accepted_and_reaches_the_upstream(
    tmp_path: Path,
) -> None:
    async def body() -> AsyncIterator[bytes]:
        chunk = b"x" * MIB
        for _ in range(64):
            yield chunk

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            async with r.client.post(r.url + "/api/embed", data=body()) as resp:
                assert resp.status == 200
            (seen,) = r.fakes[WORKSTATION].requests
            assert len(seen.body) == 64 * MIB
            assert r.journal()[0]["status"] == "200"

    run(scenario())


@pytest.mark.parametrize(
    "request_line",
    ["GET http://{host}/api/tags HTTP/1.1", "POST http://{host}/api/chat HTTP/1.1"],
)
def test_a_target_that_does_not_start_with_a_slash_gets_400_before_anything_else(
    tmp_path: Path, request_line: str
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)], extra_hosts=("executor",)) as r:
            other = r.fakes["executor"].url.removeprefix("http://")
            body = json.dumps(CHAT)
            line = request_line.format(host=other)
            head = f"{line}\r\nHost: {other}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            answer = await raw_exchange(r.url, (head + body).encode())
            assert answer.startswith(b"HTTP/1.1 400")
            assert not any(f.requests for f in r.fakes.values())
            assert r.journal() == [] and not r.attribution.calls

    run(scenario())


def test_an_attribution_oserror_is_a_500_not_a_hangup(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [PermissionError("cannot read /proc")]) as r:
            assert (await r.send("POST", "/api/chat", CHAT)).status == 500
            (entry,) = r.journal()
            assert entry["status"] == "500" and r.cancelled == []

    run(scenario())


def test_an_attribution_failure_answers_500_and_still_writes_one_journal_line(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            assert (await r.send("POST", "/api/chat", CHAT)).status == 500
            (entry,) = r.journal()
            assert (entry["status"], entry["level"], entry["model"]) == ("500", "-", "m")
            await r.wait_idle()

    run(scenario())


def test_connection_and_transfer_encoding_headers_are_not_forwarded(tmp_path: Path) -> None:
    async def chunked() -> AsyncIterator[bytes]:
        yield json.dumps({**CHAT, "stream": False}).encode()

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS), who(Level.REINS)]) as r:
            await r.send(
                "POST",
                "/api/chat",
                json.dumps({**CHAT, "stream": False}).encode(),
                headers={"Connection": "keep-alive, X-Hop", "X-Hop": "1"},
            )
            async with r.client.post(r.url + "/api/chat", data=chunked()) as resp:
                assert resp.status == 200
            hopped, streamed = r.fakes[WORKSTATION].requests
            assert hopped.headers.get("Connection") != "keep-alive, X-Hop"
            assert "Transfer-Encoding" not in streamed.headers
            assert streamed.headers["Content-Length"] == str(len(streamed.body))

    run(scenario())


def test_a_percent_encoded_gated_path_sent_raw_is_still_gated(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.HEARTH), who(Level.HEARTH)]) as r:
            fake = r.fakes[WORKSTATION]
            fake.hold = True
            held = r.chat("A")
            await fake.wait_for_requests(1)
            body = json.dumps(CHAT)
            head = f"POST /api/%63hat HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n"
            reader, writer = await asyncio.open_connection(
                *r.url.removeprefix("http://").split(":")
            )
            writer.write(f"{head}Connection: close\r\n\r\n{body}".encode())
            await r.wait_queued("hearth", 1)
            assert len(fake.requests) == 1
            fake.release.set()
            await held
            assert (await asyncio.wait_for(reader.read(), 5)).startswith(b"HTTP/1.1 200")
            writer.close()
            assert fake.requests[1].target == "/api/%63hat"

    run(scenario())


@pytest.mark.parametrize(("method", "path"), [("GET", "/api/tags"), ("POST", "/api/chat")])
def test_an_upstream_redirect_is_passed_on_not_followed(
    tmp_path: Path, method: str, path: str
) -> None:
    async def scenario() -> None:
        hits: list[str] = []

        async def redirect(request: web.Request) -> web.Response:
            hits.append(request.path)
            raise web.HTTPTemporaryRedirect("/elsewhere")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", redirect)
        runner, url = await serve(app)
        try:
            async with rig(tmp_path, [who(Level.REINS)], workstation_url=url) as r:
                async with r.client.request(method, r.url + path, allow_redirects=False) as resp:
                    assert (resp.status, resp.headers["Location"]) == (307, "/elsewhere")
                assert hits == [path]
        finally:
            await runner.cleanup()

    run(scenario())


def test_an_ungated_upload_is_streamed_to_the_upstream_while_the_client_is_still_sending(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first_seen: list[int] = []
        finished = asyncio.Event()

        async def sink(request: web.Request) -> web.Response:
            first = len(await request.content.readany())
            first_seen.append(first)
            await finished.wait()
            return web.json_response({"bytes": first + len(await request.read())})

        app = web.Application(client_max_size=0)
        app.router.add_route("*", "/{tail:.*}", sink)
        runner, url = await serve(app)
        try:
            async with rig(tmp_path, workstation_url=url) as r:
                gate = asyncio.Event()

                async def upload() -> AsyncIterator[bytes]:
                    yield b"a" * 1024
                    await gate.wait()
                    yield b"b" * 1024

                task = asyncio.create_task(
                    r.client.post(r.url + "/api/blobs/sha256:x", data=upload())
                )
                async with asyncio.timeout(5):
                    while not first_seen:
                        await asyncio.sleep(0.005)
                gate.set()
                finished.set()
                async with await task as resp:
                    assert (await resp.json())["bytes"] == 2048
        finally:
            await runner.cleanup()

    run(scenario())


# --- checks 2 and 3: priority; check 4: starvation -------------------------------------------------------------


def test_checks_2_and_3_a_higher_level_gets_the_next_slot_ahead_of_a_waiting_hearth(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        script = [who(Level.HEARTH), who(Level.HEARTH), who(Level.REINS)]
        async with rig(tmp_path, script) as r:
            fake = r.fakes[WORKSTATION]
            fake.hold = True
            tasks = [r.chat("A")]
            await fake.wait_for_requests(1)
            tasks.append(r.chat("B"))
            await r.wait_queued("hearth", 1)
            tasks.append(r.chat("R"))
            await r.wait_queued("reins", 1)
            fake.release.set()
            await asyncio.gather(*tasks)
            assert fake.prompts() == ["A", "R", "B"]

    run(scenario())


def test_check4_a_request_queued_past_the_limit_logs_one_starved_line(tmp_path: Path) -> None:
    async def scenario() -> None:
        script = [who(Level.HEARTH, "run-7"), who(Level.CORSU, "torra")]
        async with rig(tmp_path, script, starve_after_s=0.05) as r:
            fake = r.fakes[WORKSTATION]
            fake.hold = True
            first = r.chat("A")
            await fake.wait_for_requests(1)
            second = r.chat("B")
            async with asyncio.timeout(5):
                while not any(line.startswith("starved ") for line in r.lines):
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.15)
            fake.release.set()
            await asyncio.gather(first, second)
        starved = [line for line in r.lines if line.startswith("starved ")]
        assert len(starved) == 1
        assert re.fullmatch(
            r"starved host=workstation level=corsu source=torra waited_ms=\d+", starved[0]
        )

    run(scenario())


# --- check 5: upstream errors ----------------------------------------------------------------------------------


def test_check5_a_refused_upstream_gives_a_json_502_within_five_seconds(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)], workstation_url=refused_url()) as r:
            began = time.monotonic()
            gated = await r.send("POST", "/api/chat", CHAT)
            ungated = await r.send("GET", "/api/tags")
            assert time.monotonic() - began < 5
            for reply in (gated, ungated):
                assert reply.status == 502
                assert "error" in json.loads(reply.body)
            assert r.entry("m")["status"] == "upstream-error"
            await r.wait_idle()

    run(scenario())


def test_check5_an_upstream_503_passes_through_unchanged(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            r.fakes[WORKSTATION].mode = "503"
            reply = await r.send("POST", "/api/chat", CHAT)
            assert reply.status == 503 and json.loads(reply.body) == {"error": "overloaded"}
            assert r.entry("m")["status"] == "503"

    run(scenario())


def test_an_upstream_that_dies_mid_stream_closes_the_client_connection(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            r.fakes[WORKSTATION].mode = "cut"
            async with r.client.post(r.url + "/api/chat", data=json.dumps(CHAT)) as resp:
                assert resp.status == 200
                with pytest.raises(aiohttp.ClientPayloadError):
                    await resp.read()
            await r.wait_journal(1)
            assert r.entry("m")["status"] == "upstream-error"
            await r.wait_idle()

    run(scenario())


# --- the upstream idle limit (D-050 (5)) -----------------------------------------------------------------------


def test_a_reply_with_no_bytes_before_the_headers_gets_504_and_frees_the_slot(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS), who(Level.REINS)], upstream_idle_s=1.0) as r:
            fake = r.fakes[WORKSTATION]
            fake.hold = True
            first = r.chat("A", "m-A")
            await fake.wait_for_requests(1)
            second = r.chat("B", "m-B")
            await r.wait_queued("reins", 1)
            fake.hold = False
            stalled, next_in_line = await asyncio.gather(first, second)
            assert stalled.status == 504
            assert json.loads(stalled.body) == {
                "error": "upstream idle: no bytes from Ollama for 1 s"
            }
            assert next_in_line.status == 200 and fake.prompts() == ["A", "B"]
            await r.wait_idle()
            assert (r.entry("m-A")["status"], r.entry("m-B")["status"]) == ("upstream-idle", "200")

    run(scenario())


def test_a_reply_that_goes_quiet_mid_body_is_ended_and_frees_the_slot(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS), who(Level.REINS)], upstream_idle_s=1.0) as r:
            fake = r.fakes[WORKSTATION]
            fake.pause_after = 1
            async with r.client.post(r.url + "/api/chat", data=json.dumps(CHAT)) as resp:
                assert resp.status == 200 and json.loads(await resp.content.readline())["token"]
                second = r.chat("B", "m-B")
                await r.wait_queued("reins", 1)
                fake.pause_after = None
                with pytest.raises(aiohttp.ClientPayloadError):
                    await resp.read()
            assert (await second).status == 200 and fake.prompts()[1] == "B"
            await r.wait_idle()
            assert [e["status"] for e in r.journal()] == ["upstream-idle", "200"]

    run(scenario())


def test_an_ungated_reply_with_no_bytes_gets_504_and_writes_no_journal_line(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, upstream_idle_s=0.3) as r:
            r.fakes[WORKSTATION].stall = True
            reply = await r.send("GET", "/api/tags")
            assert reply.status == 504 and "0.3 s" in json.loads(reply.body)["error"]
            assert r.journal() == []

    run(scenario())


def test_a_body_that_keeps_arriving_within_the_limit_is_not_ended(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)], upstream_idle_s=0.6) as r:
            fake = r.fakes[WORKSTATION]
            fake.chunks, fake.delay_s = 12, 0.1
            reply = await r.send("POST", "/api/chat", CHAT)
            assert reply.status == 200 and reply.body.endswith(b'{"done": true}\n')
            await r.wait_journal(1)
            assert r.journal()[0]["status"] == "200"

    run(scenario())


@pytest.mark.parametrize(
    ("error", "status", "journalled"),
    [
        (aiohttp.ConnectionTimeoutError("c"), 502, "upstream-error"),
        (aiohttp.SocketTimeoutError("r"), 504, "upstream-idle"),
    ],
)
def test_a_connect_timeout_stays_502_and_only_a_read_timeout_is_504(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: int,
    journalled: str,
) -> None:
    async def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            monkeypatch.setattr(r.app[SESSION], "request", fail)
            reply = await r.send("POST", "/api/chat", CHAT)
            assert reply.status == status and "error" in json.loads(reply.body)
            await r.wait_journal(1)
            assert r.entry("m")["status"] == journalled

    run(scenario())


# --- check 7: hang-ups -----------------------------------------------------------------------------------------


def test_check7_a_client_that_hangs_up_while_queued_frees_its_place(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = Clock()
        script = [who(Level.HEARTH), who(Level.HEARTH), who(Level.HEARTH)]
        async with rig(tmp_path, script, clock=clock) as r:
            fake = r.fakes[WORKSTATION]
            fake.hold = True
            first = r.chat("A", "m-A")
            await fake.wait_for_requests(1)
            gone = r.chat("B", "m-B")
            await r.wait_queued("hearth", 1)
            third = r.chat("C", "m-C")
            await r.wait_queued("hearth", 2)
            clock.now = 1.0
            gone.cancel()
            await r.wait_queued("hearth", 1)
            await r.wait_journal(1)
            assert r.entry("m-B") == {
                "level": "hearth",
                "source": "src",
                "model": "m-B",
                "host": WORKSTATION,
                "wait_ms": "1000",
                "run_ms": "0",
                "status": "hangup",
            }
            clock.now = 2.5
            fake.release.set()
            await asyncio.gather(first, third)
            assert fake.prompts() == ["A", "C"]
            assert r.cancelled == ["/api/chat"]
            assert (await r.status())["hosts"][WORKSTATION]["in_flight"] == 0
            assert (r.entry("m-A")["wait_ms"], r.entry("m-A")["run_ms"]) == ("0", "2500")
            assert (r.entry("m-C")["wait_ms"], r.entry("m-C")["run_ms"]) == ("2500", "0")

    run(scenario())


def test_check7_a_client_that_hangs_up_mid_stream_releases_the_slot_and_the_upstream(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS), who(Level.REINS)]) as r:
            fake = r.fakes[WORKSTATION]
            fake.chunks, fake.delay_s = 500, 0.01
            resp = await r.client.post(r.url + "/api/chat", data=json.dumps(CHAT))
            await resp.content.readline()
            resp.close()
            async with asyncio.timeout(5):
                while not fake.aborted:
                    await asyncio.sleep(0.01)
            await r.wait_idle()
            await r.wait_journal(1)
            assert r.journal()[0]["status"] == "hangup"
            assert r.cancelled == ["/api/chat"]
            fake.chunks = 2
            assert (await r.send("POST", "/api/chat", CHAT)).status == 200

    run(scenario())


def test_a_client_already_gone_when_the_request_starts_is_journalled_as_hangup(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            request = make_mocked_request("POST", "/api/chat", app=r.app)
            request.protocol.transport = None
            handler = (await r.app.router.resolve(request)).handler
            assert (await handler(request)).status == CLIENT_CLOSED
            (entry,) = r.journal()
            assert entry["status"] == "hangup" and not r.attribution.calls

    run(scenario())


def test_a_client_reset_while_the_headers_are_sent_is_a_hangup_and_logs_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = web.StreamResponse.prepare

    async def prepare(self: web.StreamResponse, request: web.BaseRequest) -> Any:
        if type(self) is web.StreamResponse:
            raise aiohttp.ClientConnectionResetError("client gone")
        return await real(self, request)

    monkeypatch.setattr(web.StreamResponse, "prepare", prepare)

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            with pytest.raises(aiohttp.ClientError):
                await r.send("POST", "/api/chat", {**CHAT, "stream": False})
            await r.wait_journal(1)
            assert r.journal()[0]["status"] == "hangup"
            await r.wait_idle()

    run(scenario())
    assert [x for x in caplog.records if x.levelno >= logging.ERROR] == []


def test_a_client_reset_mid_body_is_a_hangup_not_an_upstream_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = web.StreamResponse.write

    async def write(self: web.StreamResponse, data: bytes) -> None:
        if type(self) is web.StreamResponse:
            raise aiohttp.ClientConnectionResetError("client gone")
        await real(self, data)

    monkeypatch.setattr(web.StreamResponse, "write", write)

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            fake = r.fakes[WORKSTATION]
            fake.chunks, fake.delay_s = 500, 0.01
            await r.send("POST", "/api/chat", CHAT)
            await r.wait_journal(1)
            assert r.journal()[0]["status"] == "hangup"
            async with asyncio.timeout(5):
                while not fake.aborted:
                    await asyncio.sleep(0.01)
            await r.wait_idle()

    run(scenario())


# --- routing (R24) ---------------------------------------------------------------------------------------------


def _write_run(paths: StatePaths, host: str) -> None:
    state = open_state(paths)
    state.insert_run(
        RunRow("r1", "torra/bench", "m", host, "f" * 8, "running", "2026-10-30T01:00:00+00:00")
    )
    state.close()


def test_a_hearth_run_goes_to_the_host_its_run_was_assigned(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = default_paths(tmp_path)
        _write_run(paths, "executor")
        script = [who(Level.HEARTH, "pastore-r1", "r1")]
        async with rig(tmp_path, script, extra_hosts=("executor",)) as r:
            reply = await r.send("POST", "/api/chat", CHAT)
            assert reply.headers["X-Upstream-Tag"] == "executor"
            assert not r.fakes[WORKSTATION].requests and len(r.fakes["executor"].requests) == 1
            assert r.entry("m")["host"] == "executor"

    run(scenario())


def _no_database(paths: StatePaths) -> str:
    return "r1"


def _other_schema(paths: StatePaths) -> str:
    paths.state_dir.mkdir(parents=True)
    conn = sqlite3.connect(paths.db)
    conn.execute("PRAGMA user_version=1")
    conn.execute("CREATE TABLE runs(x)")
    conn.commit()
    conn.close()
    return "r1"


def _not_a_database(paths: StatePaths) -> str:
    paths.state_dir.mkdir(parents=True)
    paths.db.write_bytes(b"this is not sqlite" * 100)
    return "r1"


def _unknown_run(paths: StatePaths) -> str:
    _write_run(paths, "executor")
    return "r9"


def _host_not_in_policy(paths: StatePaths) -> str:
    _write_run(paths, "elsewhere")
    return "r1"


@pytest.mark.parametrize(
    "prepare", [_no_database, _other_schema, _not_a_database, _unknown_run, _host_not_in_policy]
)
def test_a_hearth_request_whose_run_cannot_be_resolved_goes_to_the_workstation(
    tmp_path: Path, prepare: Callable[[StatePaths], str]
) -> None:
    async def scenario() -> None:
        run_id = prepare(default_paths(tmp_path))
        script = [who(Level.HEARTH, "pastore-r1", run_id)]
        async with rig(tmp_path, script, extra_hosts=("executor",)) as r:
            reply = await r.send("POST", "/api/chat", CHAT)
            assert reply.headers["X-Upstream-Tag"] == WORKSTATION
            assert not r.fakes["executor"].requests

    run(scenario())


def test_a_hearth_request_without_a_run_id_goes_to_the_workstation(tmp_path: Path) -> None:
    async def scenario() -> None:
        _write_run(default_paths(tmp_path), "executor")
        async with rig(tmp_path, [who(Level.HEARTH)], extra_hosts=("executor",)) as r:
            reply = await r.send("POST", "/api/chat", CHAT)
            assert reply.headers["X-Upstream-Tag"] == WORKSTATION

    run(scenario())


def test_levels_one_to_five_go_to_the_workstation_even_with_a_run_id_and_another_host(
    tmp_path: Path,
) -> None:
    levels = [Level.REINS, Level.WEBUI, Level.CODING, Level.CORSU, Level.ROUTINE]

    async def scenario() -> None:
        _write_run(default_paths(tmp_path), "executor")
        async with rig(
            tmp_path, [who(x, "s", "r1") for x in levels], extra_hosts=("executor",)
        ) as r:
            for _ in levels:
                reply = await r.send("POST", "/api/chat", CHAT)
                assert reply.headers["X-Upstream-Tag"] == WORKSTATION
            assert not r.fakes["executor"].requests
            assert [e["level"] for e in r.journal()] == [x.name.lower() for x in levels]

    run(scenario())


def test_each_host_has_its_own_slots(tmp_path: Path) -> None:
    async def scenario() -> None:
        script = [who(Level.HEARTH, "pastore-r1", "r1"), who(Level.REINS)]
        _write_run(default_paths(tmp_path), "executor")
        async with rig(tmp_path, script, extra_hosts=("executor",)) as r:
            executor = r.fakes["executor"]
            executor.hold = True
            held = r.chat("A")
            await executor.wait_for_requests(1)
            assert (await r.send("POST", "/api/chat", CHAT)).status == 200
            hosts = (await r.status())["hosts"]
            assert (hosts["executor"]["in_flight"], hosts[WORKSTATION]["in_flight"]) == (1, 0)
            executor.release.set()
            await held

    run(scenario())


def test_the_state_lookup_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    real = server.open_state

    def spy(paths: StatePaths, *, readonly: bool = False) -> Any:
        seen.append(threading.get_ident())
        return real(paths, readonly=readonly)

    monkeypatch.setattr(server, "open_state", spy)

    async def scenario() -> None:
        _write_run(default_paths(tmp_path), "executor")
        script = [who(Level.HEARTH, "pastore-r1", "r1")]
        async with rig(tmp_path, script, extra_hosts=("executor",)) as r:
            await r.send("POST", "/api/chat", CHAT)
            assert seen and threading.get_ident() not in seen

    run(scenario())


def test_a_missing_state_db_is_not_created_by_the_lookup(tmp_path: Path) -> None:
    async def scenario() -> None:
        script = [who(Level.HEARTH, "pastore-r1", "r1")]
        async with rig(tmp_path, script, extra_hosts=("executor",)) as r:
            assert (await r.send("POST", "/api/chat", CHAT)).status == 200

    run(scenario())
    assert not default_paths(tmp_path).state_dir.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unreadable_state_dir_routes_hearth_to_the_workstation(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = default_paths(tmp_path)
        _write_run(paths, "executor")
        script = [who(Level.HEARTH, "pastore-r1", "r1")]
        paths.state_dir.chmod(0)
        try:
            async with rig(tmp_path, script, extra_hosts=("executor",)) as r:
                reply = await r.send("POST", "/api/chat", CHAT)
                assert reply.headers["X-Upstream-Tag"] == WORKSTATION
        finally:
            paths.state_dir.chmod(0o755)

    run(scenario())


# --- the gated set ---------------------------------------------------------------------------------------------

GATED = [
    "/api/generate",
    "/api/chat",
    "/api/embed",
    "/api/embeddings",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
]
UNGATED = [
    ("GET", "/api/tags", None),
    ("GET", "/api/ps", None),
    ("POST", "/api/show", b'{"model": "m"}'),
    ("GET", "/api/version", None),
    ("POST", "/api/pull", b'{"model": "m"}'),
    ("POST", "/api/push", b'{"model": "m"}'),
    ("POST", "/api/create", b'{"model": "m"}'),
    ("POST", "/api/copy", b'{"source": "a", "destination": "b"}'),
    ("DELETE", "/api/delete", b'{"model": "m"}'),
    ("POST", "/api/blobs/sha256:abc", bytes(range(256)) * 4096),
    ("GET", "/v1/models", None),
    ("GET", "/api/chat", None),
    ("GET", "/no/such/route?x=1", None),
]


def test_gated_routes_wait_for_a_slot_and_everything_else_passes_while_it_is_held(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        script = [who(Level.HEARTH) for _ in range(len(GATED) + 1)]
        async with rig(tmp_path, script, extra_hosts=("executor",)) as r:
            fake = r.fakes[WORKSTATION]
            fake.hold = True
            held = r.chat("A")
            await fake.wait_for_requests(1)
            queued = [r.spawn("POST", path, CHAT) for path in GATED]
            await r.wait_queued("hearth", len(GATED))
            assert len(fake.requests) == 1
            for method, path, body in UNGATED:
                reply = await r.send(method, path, body)
                assert reply.status == 200 and reply.headers["X-Upstream-Tag"] == WORKSTATION
            sent = {(x.method, x.target): x.body for x in fake.requests[1:]}
            assert sent == {(m, p): b or b"" for m, p, b in UNGATED}
            assert not r.fakes["executor"].requests
            assert r.journal() == []
            fake.release.set()
            await asyncio.gather(held, *queued)
            assert len(r.journal()) == len(GATED) + 1

    run(scenario())


# --- the journal (R32) -----------------------------------------------------------------------------------------


def test_one_journal_line_per_gated_request_with_every_field(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.CORSU, "torra-bench")]) as r:
            await r.send("POST", "/api/generate", {"model": "qwen3:8b", "prompt": "x"})
            await r.send("GET", "/api/tags")
        (line,) = (x for x in r.lines if x.startswith("request "))
        assert re.fullmatch(
            r"request level=corsu source=torra-bench model=qwen3:8b host=workstation"
            r" wait_ms=\d+ run_ms=\d+ status=200",
            line,
        )

    run(scenario())


@pytest.mark.parametrize(
    ("body", "model"),
    [
        (b'{"model": 5}', "-"),
        (b"not json at all", "-"),
        (b"[1, 2]", "-"),
        (b"", "-"),
        (b'{"model": "a b\\nc"}', "a_b_c"),
        (b'{"model": "hf.co/u/m:Q4_K_M"}', "hf.co/u/m:Q4_K_M"),
    ],
)
def test_the_journal_model_is_space_free_and_dash_when_unknown(
    tmp_path: Path, body: bytes, model: str
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS, "a b")]) as r:
            await r.send("POST", "/api/embed", body)
            (entry,) = r.journal()
            assert entry["model"] == model and entry["source"] == "a_b"

    run(scenario())


# --- /pastore/status -------------------------------------------------------------------------------------------


def test_status_lists_every_host_with_slots_of_parallel_times_models(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, extra_hosts=("executor",), parallel=3, max_models=4) as r:
            assert await r.status() == {
                "hosts": {
                    name: {
                        "slots": 12,
                        "in_flight": 0,
                        "queued": dict.fromkeys(LEVEL_NAMES, 0),
                        "oldest_wait_s": None,
                    }
                    for name in (WORKSTATION, "executor")
                },
                "started_at": STARTED_AT,
            }

    run(scenario())


def test_status_counts_in_flight_and_queued_requests(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = Clock()
        script = [who(Level.HEARTH), who(Level.HEARTH), who(Level.REINS)]
        async with rig(tmp_path, script, clock=clock) as r:
            fake = r.fakes[WORKSTATION]
            fake.hold = True
            r.chat("A")
            await fake.wait_for_requests(1)
            r.chat("B")
            await r.wait_queued("hearth", 1)
            clock.now = 4.0
            r.chat("R")
            await r.wait_queued("reins", 1)
            clock.now = 5.0
            state = (await r.status())["hosts"][WORKSTATION]
            assert state == {
                "slots": 1,
                "in_flight": 1,
                "queued": {**dict.fromkeys(LEVEL_NAMES, 0), "reins": 1, "hearth": 1},
                "oldest_wait_s": 5.0,
            }

    run(scenario())


def test_status_404s_for_other_pastore_paths_and_methods(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            for method, path in [
                ("GET", "/pastore"),
                ("GET", "/pastore/other"),
                ("POST", "/pastore/status"),
            ]:
                assert (await r.send(method, path)).status == 404
            assert not r.fakes[WORKSTATION].requests

    run(scenario())


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.8.9.10", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("192.0.2.101", False),
        ("172.17.0.2", False),
        ("::ffff:192.0.2.101", False),
        ("fe80::1", False),
        ("localhost", False),
        ("", False),
    ],
)
def test_is_loopback(host: str, expected: bool) -> None:
    assert is_loopback(host) is expected


def test_status_refuses_a_peer_that_is_not_loopback(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            for peer, allowed in [(("127.0.0.1", 5000), True), (("192.0.2.101", 5000), False)]:
                transport = Mock()
                transport.get_extra_info.return_value = peer
                request = make_mocked_request(
                    "GET", "/pastore/status", transport=transport, app=r.app
                )
                handler = (await r.app.router.resolve(request)).handler
                if allowed:
                    assert (await handler(request)).status == 200
                else:
                    with pytest.raises(web.HTTPForbidden):
                        await handler(request)

    run(scenario())


def test_the_upstream_session_has_no_total_timeout_a_connect_timeout_and_the_idle_limit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            session = r.app[SESSION]
            assert (session.timeout.total, session.timeout.sock_read) == (None, 300.0)
            assert session.timeout.connect == 3.0
            assert session.connector is not None and session.connector.limit == 0
        async with rig(tmp_path, upstream_idle_s=7.5) as r:
            assert r.app[SESSION].timeout.sock_read == 7.5

    run(scenario())


# --- model-changing, account and web-fetching routes only from this machine (D-050 (9c), D-052 (4)) -----------

LAN, EXECUTOR_IP = "192.0.2.5", "192.0.2.101"
REFUSED_ROUTES = (
    "/api/pull",
    "/api/delete",
    "/api/create",
    "/api/copy",
    "/api/push",
    "/api/blobs/sha256-abc",
    "/api/signout",
    "/api/user/keys/abc",
    "/api/experimental/web_fetch",
    "/api/experimental/web_search",
)
REFUSED = {"error": "this route is refused from off this machine"}


HOST_FIB_TRIE = """\
Local:
  +-- 0.0.0.0/0 3 1 5
     +-- 192.0.2.128/25 2 0 2
        |-- 192.0.2.167
           /32 host LOCAL
        |-- 192.0.2.255
           /32 link BROADCAST
"""


def host_proc(tmp_path: Path) -> Path:
    """A /proc whose fib_trie lists 192.0.2.167 as an address of this host."""
    root = tmp_path / "host-proc"
    (root / "net").mkdir(parents=True)
    (root / "net" / "fib_trie").write_text(HOST_FIB_TRIE)
    return root


def peer_is(monkeypatch: pytest.MonkeyPatch, peer: str, local: str = "192.0.2.1") -> None:
    """Make the gateway see every connection as coming from `peer` to `local`, whatever the socket says."""
    monkeypatch.setattr(server, "_endpoints", lambda request: ((peer, 40000), (local, 11434)))


async def raw_status(r: Rig, method: str, target: str) -> int:
    head = (
        f"{method} {target} HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
    return int((await raw_exchange(r.url, head.encode())).split(b" ", 2)[1])


def test_invariant_model_changing_routes_are_refused_off_this_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path, extra_hosts=("executor",)) as r:
            for peer in (LAN, EXECUTOR_IP, f"::ffff:{LAN}"):
                peer_is(monkeypatch, peer)
                for path in REFUSED_ROUTES:
                    reply = await r.send("POST", path, {"model": "m"})
                    assert reply.status == 403 and json.loads(reply.body) == REFUSED
                    assert f"refused route={path} peer={peer}" in r.lines
            assert len(r.lines) == 3 * len(REFUSED_ROUTES) and r.journal() == []
            assert not any(fake.requests for fake in r.fakes.values())

    run(scenario())


@pytest.mark.parametrize(
    ("peer", "local"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "::1"),
        ("::ffff:127.0.0.1", "::ffff:127.0.0.1"),
        ("172.17.0.9", "192.0.2.1"),
        (LAN, LAN),
    ],
)
def test_refused_routes_are_relayed_from_this_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peer: str, local: str
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            peer_is(monkeypatch, peer, local)
            for path in REFUSED_ROUTES:
                reply = await r.send("POST", path, {"model": "m"})
                assert reply.status == 200 and reply.headers["X-Upstream-Tag"] == WORKSTATION
            assert [x.target for x in r.fakes[WORKSTATION].requests] == list(REFUSED_ROUTES)
            assert r.lines == []

    run(scenario())


@pytest.mark.parametrize(
    ("method", "target"),
    [
        ("POST", "/api/pull/"),
        ("POST", "//api/pull"),
        ("POST", "/api//pull"),
        ("POST", "/api/%70ull"),
        ("POST", "/api%2Fpull"),
        ("POST", "/api/./pull"),
        ("POST", "///api/delete//"),
        ("POST", "/api/pull?stream=false"),
        ("DELETE", "/api/delete"),
        ("GET", "/api/pull"),
        ("PUT", "/api/copy"),
        ("POST", "/api/blobs"),
        ("HEAD", "/api/blobs/sha256-abc"),
        ("POST", "//api/blobs/x"),
        ("POST", "/api/blobs/"),
        ("POST", "/api/blobs%2Fsha256-abc"),
        ("POST", "/api/blobs/x/../y"),
        ("POST", "/api/./signout"),
        ("GET", "/api/%73ignout"),
        ("POST", "/api/user/keys"),
        ("GET", "/api/user/keys/"),
        ("DELETE", "/api/user/keys/abc"),
        ("POST", "/api/user%2Fkeys/abc"),
        ("GET", "/api/experimental/web_fetch"),
        ("POST", "/api/experimental//web_search/"),
        ("POST", "/api/experimental/web%5Ffetch?x=1"),
    ],
)
def test_refused_variants_and_any_method_are_refused_off_this_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, target: str
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            peer_is(monkeypatch, LAN)
            assert await raw_status(r, method, target) == 403
            assert not r.fakes[WORKSTATION].requests

    run(scenario())


@pytest.mark.parametrize(
    ("peer", "proc_state", "relayed"),
    [
        ("192.0.2.167", "listed", True),
        ("192.0.2.5", "listed", False),
        ("192.0.2.255", "listed", False),
        ("192.0.2.167", "missing", False),
        ("192.0.2.167", "unreadable", False),
    ],
)
def test_a_host_address_peer_on_the_bridge_address_is_relayed_and_an_unlisted_one_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peer: str, proc_state: str, relayed: bool
) -> None:
    proc = host_proc(tmp_path)
    if proc_state == "missing":
        (proc / "net" / "fib_trie").unlink()
    elif proc_state == "unreadable":
        (proc / "net" / "fib_trie").unlink()
        (proc / "net" / "fib_trie").mkdir()

    async def scenario() -> None:
        async with rig(tmp_path, proc=proc) as r:
            peer_is(monkeypatch, peer, "172.17.0.1")
            reply = await r.send("POST", "/api/pull", {"model": "m"})
            assert reply.status == (200 if relayed else 403)
            assert len(r.fakes[WORKSTATION].requests) == relayed

    run(scenario())


def test_the_peer_check_reads_proc_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    real = server.on_this_machine

    def spy(peer: str, local: str, proc: Path) -> bool:
        threads.append(threading.get_ident())
        return real(peer, local, proc)

    monkeypatch.setattr(server, "on_this_machine", spy)

    async def scenario() -> None:
        async with rig(tmp_path) as r:
            peer_is(monkeypatch, LAN)
            assert (await r.send("POST", "/api/pull", {"model": "m"})).status == 403
            assert threads and threading.get_ident() not in threads

    run(scenario())


def test_every_other_route_still_goes_through_from_a_lan_peer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    others = [
        ("GET", "/api/tags"),
        ("GET", "/api/ps"),
        ("POST", "/api/show"),
        ("GET", "/api/version"),
        ("POST", "/api/pulled"),
        ("POST", "/api/pull/x"),
        ("POST", "/api/blobsx"),
        ("POST", "/api/blobsx/y"),
        ("GET", "/api/user"),
        ("POST", "/api/user/keysx"),
        ("GET", "/api/user/keysx/abc"),
        ("POST", "/api/signoutx"),
        ("POST", "/api/experimental/web_fetchx"),
    ]

    async def scenario() -> None:
        async with rig(tmp_path, [who(Level.REINS)]) as r:
            peer_is(monkeypatch, LAN)
            for method, target in others:
                assert await raw_status(r, method, target) == 200
            assert (await r.send("POST", "/api/chat", CHAT)).status == 200
            assert [x.target for x in r.fakes[WORKSTATION].requests] == [t for _, t in others] + [
                "/api/chat"
            ]
            assert not [line for line in r.lines if line.startswith("refused")]

    run(scenario())


def test_the_refusal_line_is_one_sanitised_line_and_a_gone_client_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with rig(tmp_path) as r:
            peer_is(monkeypatch, LAN)
            assert await raw_status(r, "GET", "/api/pull/%20X/..") == 403
            assert r.lines == [f"refused route=/api/pull/_X/.. peer={LAN}"]
            monkeypatch.undo()
            request = make_mocked_request("POST", "/api/pull", app=r.app)
            request.protocol.transport = None
            handler = (await r.app.router.resolve(request)).handler
            assert (await handler(request)).status == 403
            assert r.lines[-1] == "refused route=/api/pull peer=-"
            assert not r.fakes[WORKSTATION].requests

    run(scenario())


@pytest.mark.parametrize(
    ("peer", "local", "expected"),
    [
        ("127.0.0.1", "192.0.2.1", True),
        ("127.8.9.10", "192.0.2.1", True),
        ("::1", "fe80::1", True),
        ("::ffff:127.0.0.1", "192.0.2.1", True),
        ("192.0.2.5", "192.0.2.5", True),
        ("::ffff:192.0.2.5", "192.0.2.5", True),
        ("192.0.2.5", "::ffff:192.0.2.5", True),
        ("fe80::1", "fe80::1", True),
        ("172.17.0.1", "192.0.2.1", True),
        ("172.17.255.255", "192.0.2.1", True),
        ("::ffff:172.17.0.9", "192.0.2.1", True),
        ("172.18.0.1", "192.0.2.1", False),
        ("172.16.255.255", "192.0.2.1", False),
        ("192.0.2.5", "192.0.2.6", False),
        ("::ffff:192.0.2.5", "192.0.2.6", False),
        ("fe80::1", "fe80::2", False),
        ("192.0.2.5", "", False),
        ("", "192.0.2.1", False),
        ("", "", False),
        ("localhost", "192.0.2.1", False),
        ("not-an-ip", "not-an-ip", False),
        ("192.0.2.167", "172.17.0.1", True),
        ("192.0.2.167", "192.0.2.1", True),
        ("192.0.2.5", "172.17.0.1", False),
        ("192.0.2.255", "172.17.0.1", False),
    ],
)
def test_on_this_machine(tmp_path: Path, peer: str, local: str, expected: bool) -> None:
    assert on_this_machine(peer, local, host_proc(tmp_path)) is expected
