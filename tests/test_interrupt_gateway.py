"""R34 "gateway killed mid-stream" (check 14): SIGKILL a real gateway with one stream in flight and one caller queued.

Depends on: pastore.gateway run as a child process, tests.gateway_fakes (the fake Ollama child), the repo's policy.yaml, PyYAML and pytest; binds 127.0.0.1 port 0 only.
"""

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO_POLICY = ROOT / "policy.yaml"
READY_S = 10.0
SOCKET_GUARD_S = 15.0
SEEN_WITHIN_S = 2.0
JOIN_AFTER_KILL_S = 5.0
URL_PREFIX = "http://127.0.0.1:"
UPSTREAM_LIFETIME_S = 90
UPSTREAM_SCRIPT = f"""
import asyncio, os
from tests.gateway_fakes import FakeUpstream

async def main():
    async with FakeUpstream(tag=str(os.getpid()), chunks=600, delay_s=0.1) as fake:
        print(fake.url, flush=True)
        await asyncio.sleep({UPSTREAM_LIFETIME_S})

asyncio.run(main())
"""
CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn(argv: list[str], env: dict[str, str], log: Path) -> "subprocess.Popen[bytes]":
    with log.open("wb") as out:
        return subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _reap(procs: "list[subprocess.Popen[bytes]]") -> None:
    """SIGKILL every child's process group, even one whose leader is already reaped, then wait for each leader."""
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for proc in procs:
        proc.wait(timeout=5)


def _wait_for_url(proc: "subprocess.Popen[bytes]", log: Path) -> str:
    """The upstream's first output line, once it is a whole `http://127.0.0.1:<port>` line; fail if the child dies."""

    def printed() -> bool:
        assert proc.poll() is None, (
            f"the upstream exited {proc.returncode}: {log.read_text()[-300:]}"
        )
        text = log.read_text()
        return text.startswith(URL_PREFIX) and text.endswith("\n")

    _wait_until("the upstream prints its url", printed)
    return log.read_text().splitlines()[0]


def _join_after_kill(callers: list["Caller"], killed_at: float) -> None:
    """Wait for each caller's thread until `JOIN_AFTER_KILL_S` after the kill, no longer."""
    for caller in callers:
        caller.join(timeout=max(0.0, killed_at + JOIN_AFTER_KILL_S - time.monotonic()))


def _post_chat(port: int) -> bytes:
    payload = json.dumps({**CHAT, "stream": True}).encode()
    head = (
        f"POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    )
    return head.encode() + payload


def _get_json(port: int, path: str) -> dict[str, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    try:
        conn.request("GET", path)
        answer = json.loads(conn.getresponse().read())
    finally:
        conn.close()
    assert isinstance(answer, dict)
    return answer


def _wait_until(what: str, ready: Callable[[], bool], deadline_s: float = 5.0) -> None:
    deadline = time.monotonic() + deadline_s
    while not ready():
        assert time.monotonic() < deadline, f"not within {deadline_s} s: {what}"
        time.sleep(0.05)


class Caller(threading.Thread):
    """One client on a raw socket: sends a chat, reads until the connection ends, records how and when it ended."""

    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self._port = port
        self._request = _post_chat(port)
        self.first_data = threading.Event()
        self.received = bytearray()
        self.outcome = "open"
        self.ended_at: float | None = None

    def run(self) -> None:
        """Read until EOF, a reset or the socket guard; each is an outcome, a guard timeout is the hang."""
        try:
            with socket.create_connection(
                ("127.0.0.1", self._port), timeout=SOCKET_GUARD_S
            ) as sock:
                sock.sendall(self._request)
                while chunk := sock.recv(65536):
                    self.received += chunk
                    if b'"token"' in self.received:
                        self.first_data.set()
                self.outcome = "eof"
        except ConnectionError:
            self.outcome = "reset"
        except TimeoutError:
            self.outcome = "hang"
        self.ended_at = time.monotonic()


@dataclass
class Rig:
    """A fake Ollama child, a policy that points at it, and the gateway children started so far."""

    tmp_path: Path
    upstream: "subprocess.Popen[bytes]"
    port: int
    policy: Path
    env: dict[str, str]
    gateways: "list[subprocess.Popen[bytes]]" = field(default_factory=list)

    def start_gateway(self) -> "subprocess.Popen[bytes]":
        """Start a gateway on the rig's port; its output goes to a file under tmp_path."""
        argv = [sys.executable, "-m", "pastore.gateway", "--policy", str(self.policy)]
        argv += ["--host", "127.0.0.1", "--port", str(self.port)]
        proc = _spawn(argv, self.env, self.tmp_path / f"gateway-{len(self.gateways)}.log")
        self.gateways.append(proc)
        return proc

    def status(self) -> dict[str, Any]:
        """The gateway's `/pastore/status` hosts."""
        hosts = _get_json(self.port, "/pastore/status")["hosts"]
        assert isinstance(hosts, dict)
        return hosts

    def wait_ready(self, proc: "subprocess.Popen[bytes]") -> dict[str, Any]:
        """Return the status once the gateway answers it; fail if it exits or is silent for 10 s."""
        deadline = time.monotonic() + READY_S
        while True:
            assert proc.poll() is None, f"the gateway exited {proc.returncode} before it was ready"
            try:
                return self.status()
            except (OSError, http.client.HTTPException):
                assert time.monotonic() < deadline, f"no /pastore/status within {READY_S} s"
                time.sleep(0.05)

    def counts(self) -> tuple[int, int]:
        """The workstation's slots in use and callers queued."""
        host = self.status()["workstation"]
        return host["in_flight"], sum(host["queued"].values())

    def occupy_the_slot_and_queue_one(self) -> tuple[Caller, Caller]:
        """Start a stream that holds the only slot and a second chat behind it; return both once /status shows it."""
        streaming = Caller(self.port)
        streaming.start()
        assert streaming.first_data.wait(5.0), "no NDJSON line from the stream within 5 s"
        queued = Caller(self.port)
        queued.start()
        _wait_until("one in flight and one queued", lambda: self.counts() == (1, 1))
        assert not queued.received, "the queued caller was answered before the kill"
        return streaming, queued

    def kill(self, proc: "subprocess.Popen[bytes]") -> float:
        """SIGKILL the gateway, check that it died of it, and return the `time.monotonic` just before the signal."""
        killed_at = time.monotonic()
        os.kill(proc.pid, signal.SIGKILL)
        assert proc.wait(timeout=5) == -signal.SIGKILL
        return killed_at


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    """The fake Ollama child, the one-slot policy and the env; kills every child it started, on any exit."""
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    upstream_log = tmp_path / "upstream.log"
    upstream = _spawn([sys.executable, "-c", UPSTREAM_SCRIPT], env, upstream_log)
    rig = Rig(tmp_path, upstream, _free_port(), tmp_path / "policy.yaml", env)
    try:
        upstream_url = _wait_for_url(upstream, upstream_log)
        policy = yaml.safe_load(REPO_POLICY.read_text())
        policy["hosts"]["workstation"].update(url=upstream_url, parallel=1, max_models=1)
        policy["hosts"]["executor"]["url"] = f"http://127.0.0.1:{_free_port()}"
        assert set(policy["hosts"]) == {"workstation", "executor"}, "a host here keeps a live url"
        rig.policy.write_text(yaml.safe_dump(policy))
        yield rig
    finally:
        _reap([upstream, *rig.gateways])


def test_gateway_killed_mid_stream_fails_every_caller_at_once(rig: Rig) -> None:
    """R34: SIGKILL with a stream in flight and a caller queued ends both at once, and the dead port refuses."""
    gateway = rig.start_gateway()
    rig.wait_ready(gateway)
    streaming, queued = rig.occupy_the_slot_and_queue_one()

    killed_at = rig.kill(gateway)
    _join_after_kill([streaming, queued], killed_at)

    for caller in (streaming, queued):
        assert caller.ended_at is not None, (
            f"a caller was open {JOIN_AFTER_KILL_S} s after the kill"
        )
        assert caller.outcome in {"eof", "reset"}, caller.outcome
        assert caller.ended_at - killed_at < SEEN_WITHIN_S
    head = bytes(streaming.received).partition(b"\r\n\r\n")[0].lower()
    assert head.startswith(b"http/1.1 200")
    assert b"transfer-encoding: chunked" in head
    assert not bytes(streaming.received).endswith(b"0\r\n\r\n"), "the body ended cleanly"
    assert not queued.received
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", rig.port), timeout=1.0).close()


def test_restarted_gateway_starts_with_every_slot_free(rig: Rig) -> None:
    """R34: after a kill that left a slot held and a caller queued, a new gateway has nothing held and serves at once."""
    first = rig.start_gateway()
    rig.wait_ready(first)
    streaming, queued = rig.occupy_the_slot_and_queue_one()
    _join_after_kill([streaming, queued], rig.kill(first))

    status = rig.wait_ready(rig.start_gateway())

    assert status["workstation"]["slots"] == 1
    for name, host in status.items():
        assert host["in_flight"] == 0, name
        assert sum(host["queued"].values()) == 0, name
        assert host["oldest_wait_s"] is None, name
    started = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=SOCKET_GUARD_S)
    try:
        conn.request("POST", "/api/chat", json.dumps({**CHAT, "stream": False}))
        reply = conn.getresponse()
        body = reply.read()
    finally:
        conn.close()
    assert time.monotonic() - started < SEEN_WITHIN_S
    assert reply.status == 200
    assert json.loads(body)["tag"] == str(rig.upstream.pid)
    assert rig.upstream.poll() is None
