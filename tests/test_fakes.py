"""Tests that pin the shared test doubles, so a broken fake cannot make a real test pass.

Depends on: pastore/fake_ollama.py, tests/fakes.py.
"""

import contextlib
import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import pytest

from pastore.fake_ollama import MAX_POST_BYTES, FakeOllama, Mode, refused_url
from tests.fakes import FakeClock, RecordingRunner


def test_ok_mode_answers_tags() -> None:
    with FakeOllama("ok") as fake:
        with urllib.request.urlopen(fake.url + "/api/tags", timeout=5) as resp:
            assert resp.status == 200
            assert resp.read() == b'{"models": []}'
        assert fake.requests == ["/api/tags"]


def get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def post_json(url: str, body: object) -> object:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=5) as resp:
        return json.loads(resp.read())


def test_ok_mode_reports_model_sizes_on_tags() -> None:
    with FakeOllama("ok", models={"a:1b": 100, "b:2b": 200}) as fake:
        assert get_json(fake.url + "/api/tags") == {
            "models": [
                {"name": "a:1b", "model": "a:1b", "size": 100},
                {"name": "b:2b", "model": "b:2b", "size": 200},
            ]
        }
        fake.models["c:3b"] = 300
        tags = get_json(fake.url + "/api/tags")
        assert isinstance(tags, dict)
        assert [m["name"] for m in tags["models"]] == ["a:1b", "b:2b", "c:3b"]


def test_ok_mode_reports_resident_sizes_on_ps() -> None:
    with FakeOllama("ok", models={"a:1b": 100}, loaded={"a:1b": 90}) as fake:
        assert get_json(fake.url + "/api/ps") == {
            "models": [{"name": "a:1b", "model": "a:1b", "size": 90, "size_vram": 90}]
        }
        fake.loaded.clear()
        assert get_json(fake.url + "/api/ps") == {"models": []}


def test_ok_mode_answers_show_from_the_posted_model() -> None:
    info = {"general.architecture": "llama", "llama.block_count": 32}
    with FakeOllama("ok", show={"a:1b": info}) as fake:
        assert post_json(fake.url + "/api/show", {"model": "a:1b"}) == {"model_info": info}
        assert post_json(fake.url + "/api/show", {"model": "other"}) == {"model_info": {}}
        assert post_json(fake.url + "/api/show", {"name": "a:1b"}) == {"model_info": {}}
        fake.show["b:2b"] = {"x": 1}
        assert post_json(fake.url + "/api/show", {"model": "b:2b"}) == {"model_info": {"x": 1}}
        assert fake.requests == ["/api/show"] * 4


def test_ok_mode_answers_post_to_other_paths_with_an_empty_object() -> None:
    with FakeOllama("ok") as fake:
        assert post_json(fake.url + "/api/chat", {"model": "a"}) == {}
        assert fake.requests == ["/api/chat"]


def open_post(url: str, declared: int, sent: bytes | None) -> http.client.HTTPConnection:
    conn = http.client.HTTPConnection(urlsplit(url).netloc, timeout=5)
    conn.putrequest("POST", "/api/show")
    conn.putheader("Content-Length", str(declared))
    conn.endheaders(sent)
    return conn


def wait_until(predicate: Callable[[], bool], seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize("mode", ["ok", "hang"])
def test_post_is_recorded_on_its_headers_while_the_body_is_incomplete(mode: Mode) -> None:
    with FakeOllama(mode) as fake:
        conn = open_post(fake.url, 100, b'{"model"')
        try:
            assert wait_until(lambda: fake.requests == ["/api/show"])
        finally:
            conn.close()


def test_post_declaring_over_the_cap_gets_413_without_its_body_read() -> None:
    assert MAX_POST_BYTES == 1024 * 1024
    with FakeOllama("ok") as fake:
        conn = open_post(fake.url, MAX_POST_BYTES + 1, None)
        try:
            assert conn.getresponse().status == 413
        finally:
            conn.close()
        assert fake.requests == ["/api/show"]


def raw_post_answer(url: str, content_length: str) -> tuple[int, object]:
    """POST /api/show with the given Content-Length header and no body, the socket open until the answer or 2 s."""
    parts = urlsplit(url)
    request = f"POST /api/show HTTP/1.0\r\nContent-Length: {content_length}\r\n\r\n".encode(
        "latin-1"
    )
    data = b""
    with socket.create_connection((parts.hostname or "", parts.port or 0), timeout=2) as sock:
        sock.sendall(request)
        with contextlib.suppress(TimeoutError):
            while chunk := sock.recv(65536):
                data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    return (int(head.split()[1]) if head else 0), (json.loads(body) if body else None)


@pytest.mark.parametrize("declared", ["-1", "-2", "abc", "\u00b2", "1.5", " "])
def test_post_with_a_content_length_that_is_not_a_plain_number_is_answered_as_bodyless(
    declared: str,
) -> None:
    with FakeOllama("ok") as fake:
        assert raw_post_answer(fake.url, declared) == (200, {"model_info": {}})


@pytest.mark.parametrize("declared", ["9" * 5000, "1" + "0" * 9, "9" * 10, "0" * 10])
def test_post_with_a_content_length_of_ten_or_more_digits_gets_413(declared: str) -> None:
    with FakeOllama("ok") as fake:
        assert raw_post_answer(fake.url, declared) == (413, {"error": "fake 413"})


@pytest.mark.parametrize("declared", ["0", "000000000"])
def test_post_with_a_zero_content_length_is_answered(declared: str) -> None:
    with FakeOllama("ok") as fake:
        assert raw_post_answer(fake.url, declared) == (200, {"model_info": {}})


def test_post_of_exactly_the_cap_is_read_and_answered() -> None:
    empty = len(json.dumps({"model": "a:1b", "pad": ""}))
    body = json.dumps({"model": "a:1b", "pad": "x" * (MAX_POST_BYTES - empty)}).encode("utf-8")
    assert len(body) == MAX_POST_BYTES
    with FakeOllama("ok", show={"a:1b": {"k": 1}}) as fake:
        conn = open_post(fake.url, len(body), body)
        try:
            resp = conn.getresponse()
            assert (resp.status, json.loads(resp.read())) == (200, {"model_info": {"k": 1}})
        finally:
            conn.close()


def test_ok_mode_answers_version_and_other_paths() -> None:
    with FakeOllama("ok") as fake:
        assert get_json(fake.url + "/api/version") == {"version": "0.0.0-fake"}
        assert get_json(fake.url + "/api/ps") == {"models": []}
        assert get_json(fake.url + "/api/chat") == {}
        assert fake.requests == ["/api/version", "/api/ps", "/api/chat"]


def test_503_mode_returns_503() -> None:
    with FakeOllama("503") as fake:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(fake.url + "/api/chat", timeout=5)
        exc_info.value.close()
        assert exc_info.value.code == 503


def test_hang_mode_never_answers() -> None:
    with FakeOllama("hang") as fake:
        with pytest.raises((TimeoutError, urllib.error.URLError)):
            urllib.request.urlopen(fake.url + "/api/chat", timeout=0.5)


def test_refused_url_refuses() -> None:
    with pytest.raises(urllib.error.URLError) as exc_info:
        urllib.request.urlopen(refused_url() + "/api/tags", timeout=2)
    assert isinstance(exc_info.value.reason, ConnectionRefusedError)


def test_fake_clock_moves_only_when_advanced() -> None:
    start = datetime(2026, 10, 24, 1, 0, tzinfo=UTC)
    clock = FakeClock(start)
    assert clock() == start
    clock.advance(90)
    assert clock() == start + timedelta(seconds=90)


def test_fake_clock_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        FakeClock(datetime(2026, 10, 24))


def test_recording_runner_records_and_scripts() -> None:
    runner = RecordingRunner({"nvidia-smi": RecordingRunner.ok("580.178.04\n")})
    r = runner(["nvidia-smi", "-q"])
    assert r.stdout == "580.178.04\n"
    assert runner(["dpkg-query", "-W"]).returncode == 0
    assert runner.calls == [["nvidia-smi", "-q"], ["dpkg-query", "-W"]]


def test_recording_runner_refuses_shell_string() -> None:
    with pytest.raises(TypeError):
        RecordingRunner()("nvidia-smi -q")
