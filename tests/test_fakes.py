"""Tests that pin the shared test doubles, so a broken fake cannot make a real test pass.

Depends on: pastore/fake_ollama.py, tests/fakes.py.
"""

import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

import pytest

from pastore.fake_ollama import FakeOllama, refused_url
from tests.fakes import FakeClock, RecordingRunner


def test_ok_mode_answers_tags() -> None:
    with FakeOllama("ok") as fake:
        with urllib.request.urlopen(fake.url + "/api/tags", timeout=5) as resp:
            assert resp.status == 200
            assert resp.read() == b'{"models": []}'
        assert fake.requests == ["/api/tags"]


def test_503_mode_returns_503() -> None:
    with FakeOllama("503") as fake:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(fake.url + "/api/chat", timeout=5)
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
