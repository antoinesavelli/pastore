"""Tests for per-host model facts, KV shapes and the executor health probe, over FakeOllama and an injected HTTP call.

Depends on: pastore/hosts.py, pastore/fake_ollama.py, pastore/spec.py.
"""

import contextlib
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import Literal

import pytest

from pastore import hosts
from pastore.fake_ollama import FakeOllama, refused_url
from pastore.hosts import (
    PROBE_TIMEOUT_S,
    READ_TIMEOUT_S,
    HostError,
    HostState,
    HttpJson,
    KvShape,
    executor_reason,
    http_json,
    kv_shape,
    model_sizes,
    ps_sizes,
    read_host,
)
from pastore.spec import WORKSTATION, Host

EXECUTOR_URL = "http://executor.test:11434"
WS_URL = "http://workstation.test:11434"

LLAMA_INFO: dict[str, object] = {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 64,
    "llama.embedding_length": 4096,
    "llama.attention.head_count": 16,
}
LLAMA_SHAPE = KvShape(32, 8, 64)
HEALTHY: dict[str, object] = {
    "/api/version": {"version": "0.9.0"},
    "/api/tags": {"models": [{"name": "a", "size": 100}]},
    "/api/ps": {"models": []},
}
ERRORS = [OSError("refused"), TimeoutError("slow"), ValueError("bad json")]


def make_host(name: str = "executor", url: str = EXECUTOR_URL) -> Host:
    return Host(name=name, url=url, vram_gb=24, max_models=3, parallel=1, context=8192)


def workstation(url: str = WS_URL) -> Host:
    return make_host(WORKSTATION, url)


HOSTS = [
    pytest.param(workstation(), "ollama-down", id="workstation"),
    pytest.param(make_host(), "executor-down", id="executor"),
]


class Recorder:
    """An injected HttpJson: a healthy host serving model `a` unless overridden; records each (url, body, timeout)."""

    def __init__(
        self,
        answers: Mapping[str, object] | None = None,
        show: Mapping[str, object] | None = None,
        raises: Mapping[str, Exception] | None = None,
    ) -> None:
        self.answers = {**HEALTHY, **(answers or {})}
        self.show = {"a": {"model_info": LLAMA_INFO}, **(show or {})}
        self.raises = dict(raises or {})
        self.calls: list[tuple[str, Mapping[str, object] | None, float]] = []

    def __call__(self, url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
        self.calls.append((url, body, timeout_s))
        path = "/" + url.split("/", 3)[3]
        if path in self.raises:
            raise self.raises[path]
        if path == "/api/show":
            assert body is not None
            return self.show[str(body["model"])]
        return self.answers[path]


def down(host: Host, reason: str) -> HostState:
    return HostState(host, reason, {}, {}, {}, {})


Keys = tuple[str, ...]
TAGS_KEYS: Keys = ("size",)
PS_KEYS: Keys = ("size_vram", "size")
READERS: dict[str, tuple[Keys, Callable[[Host, HttpJson], object]]] = {
    "/api/tags": (TAGS_KEYS, model_sizes),
    "/api/ps": (PS_KEYS, ps_sizes),
}


def entry(keys: Keys, **changes: object) -> dict[str, object]:
    return {"name": "a", **dict.fromkeys(keys, 1), **changes}


def malformed(keys: Keys) -> dict[str, object]:
    good = entry(keys)
    cases: dict[str, object] = {
        "not-a-dict": [good],
        "string-answer": "models",
        "missing-models": {},
        "models-not-a-list": {"models": {"a": 1}},
        "models-null": {"models": None},
        "entry-not-a-dict": {"models": ["a"]},
        "missing-name": {"models": [{k: v for k, v in good.items() if k != "name"}]},
        "empty-name": {"models": [entry(keys, name="")]},
        "non-string-name": {"models": [entry(keys, name=7)]},
        "duplicate-name": {"models": [good, entry(keys)]},
    }
    for key in keys:
        cases[f"missing-{key}"] = {"models": [{k: v for k, v in good.items() if k != key}]}
        for label, value in (
            ("bool", True),
            ("negative", -1),
            ("float", 1.5),
            ("string", "1"),
            ("null", None),
        ):
            cases[f"{label}-{key}"] = {"models": [entry(keys, **{key: value})]}
    return cases


MALFORMED = [
    pytest.param(path, payload, id=f"{path}-{name}")
    for path, (keys, _) in READERS.items()
    for name, payload in malformed(keys).items()
]


def patched(
    changes: Mapping[str, object] | None = None, drop: Collection[str] = ()
) -> dict[str, object]:
    return {k: v for k, v in {**LLAMA_INFO, **(changes or {})}.items() if k not in drop}


def shape_of(model_info: object) -> KvShape | None:
    return kv_shape(make_host(), "a", Recorder(show={"a": {"model_info": model_info}}))


@contextlib.contextmanager
def serving(mode: Literal["refused", "503", "hang"]) -> Iterator[str]:
    if mode == "refused":
        yield refused_url()
    else:
        with FakeOllama(mode) as fake:
            yield fake.url


def test_check9_executor_link_down_gives_executor_down() -> None:
    for mode in ("refused", "503", "hang"):
        started = time.monotonic()
        with serving(mode) as url:
            state = read_host(make_host(url=url), http_json, ["a"])
        assert state == down(make_host(url=url), "executor-down"), mode
        assert time.monotonic() - started < 4.0, mode


def test_executor_down_reads_neither_tags_ps_nor_show() -> None:
    with FakeOllama("503") as fake:
        state = read_host(make_host(url=fake.url), http_json, ["a"])
    assert state.reason == "executor-down"
    assert fake.requests == ["/api/version"]


def test_workstation_with_ollama_refused_is_ollama_down() -> None:
    host = workstation(refused_url())
    assert read_host(host, http_json, ["a"]) == down(host, "ollama-down")


def test_probe_uses_the_two_second_timeout_on_the_version_path() -> None:
    assert PROBE_TIMEOUT_S == 2.0
    recorder = Recorder()
    assert executor_reason(make_host(), recorder) is None
    assert recorder.calls == [(EXECUTOR_URL + "/api/version", None, 2.0)]


def test_executor_host_is_probed_then_read_then_shown_with_the_read_timeout() -> None:
    assert READ_TIMEOUT_S == 5.0
    recorder = Recorder()
    state = read_host(make_host(), recorder, ["a"])
    assert state == HostState(make_host(), None, {"a": 100}, {}, {}, {"a": LLAMA_SHAPE})
    assert recorder.calls == [
        (EXECUTOR_URL + "/api/version", None, 2.0),
        (EXECUTOR_URL + "/api/tags", None, 5.0),
        (EXECUTOR_URL + "/api/ps", None, 5.0),
        (EXECUTOR_URL + "/api/show", {"model": "a"}, 5.0),
    ]


def test_workstation_is_not_probed_for_a_version() -> None:
    recorder = Recorder()
    read_host(workstation(), recorder, ["a"])
    assert recorder.calls == [
        (WS_URL + "/api/tags", None, 5.0),
        (WS_URL + "/api/ps", None, 5.0),
        (WS_URL + "/api/show", {"model": "a"}, 5.0),
    ]


def test_healthy_executor_reports_exact_facts() -> None:
    with FakeOllama(
        "ok", models={"a:1b": 100, "b:2b": 200}, loaded={"b:2b": 150}, show={"a:1b": LLAMA_INFO}
    ) as fake:
        host = make_host(url=fake.url)
        state = read_host(host, http_json, ["a:1b"])
    expected = HostState(
        host, None, {"a:1b": 100, "b:2b": 200}, {"b:2b": 150}, {"b:2b": 150}, {"a:1b": LLAMA_SHAPE}
    )
    assert state == expected
    assert fake.requests == ["/api/version", "/api/tags", "/api/ps", "/api/show"]


def test_healthy_workstation_reports_exact_facts_and_makes_no_version_request() -> None:
    with FakeOllama(
        "ok", models={"a:1b": 100}, loaded={"a:1b": 90}, show={"a:1b": LLAMA_INFO}
    ) as fake:
        host = workstation(fake.url)
        state = read_host(host, http_json, ["a:1b"])
    assert state == HostState(
        host, None, {"a:1b": 100}, {"a:1b": 90}, {"a:1b": 90}, {"a:1b": LLAMA_SHAPE}
    )
    assert fake.requests == ["/api/tags", "/api/ps", "/api/show"]


def test_ps_gives_size_vram_as_resident_and_size_as_loaded() -> None:
    ps = {
        "models": [
            {"name": "a", "size": 120, "size_vram": 90},
            {"name": "b", "size": 7, "size_vram": 7},
        ]
    }
    resident, loaded = ps_sizes(make_host(), Recorder({"/api/ps": ps}))
    assert resident == {"a": 90, "b": 7}
    assert loaded == {"a": 120, "b": 7}
    state = read_host(make_host(), Recorder({"/api/ps": ps}), [])
    assert (dict(state.resident), dict(state.loaded)) == ({"a": 90, "b": 7}, {"a": 120, "b": 7})


def test_tags_read_size_and_ignore_other_fields() -> None:
    tags = {
        "models": [
            {"name": "a", "size": 0, "digest": "x", "size_vram": 9},
            {"name": "b", "size": 5},
        ]
    }
    assert model_sizes(make_host(), Recorder({"/api/tags": tags})) == {"a": 0, "b": 5}


def test_ps_accepts_zero_sizes_and_ignores_other_fields() -> None:
    ps = {"models": [{"name": "a", "size_vram": 0, "size": 0, "expires_at": "later"}]}
    assert ps_sizes(make_host(), Recorder({"/api/ps": ps})) == ({"a": 0}, {"a": 0})


def test_empty_models_lists_are_valid() -> None:
    recorder = Recorder({"/api/tags": {"models": []}, "/api/ps": {"models": []}})
    assert model_sizes(make_host(), recorder) == {}
    assert ps_sizes(make_host(), recorder) == ({}, {})


@pytest.mark.parametrize(("path", "payload"), MALFORMED)
def test_malformed_listing_raises_host_error_naming_host_and_path(
    path: str, payload: object
) -> None:
    reader = READERS[path][1]
    with pytest.raises(HostError, match=f"executor.*{path}"):
        reader(make_host(), Recorder({path: payload}))


def test_tags_need_size_and_ps_need_both_size_fields() -> None:
    with pytest.raises(HostError):
        model_sizes(
            make_host(), Recorder({"/api/tags": {"models": [{"name": "a", "size_vram": 1}]}})
        )
    with pytest.raises(HostError):
        ps_sizes(make_host(), Recorder({"/api/ps": {"models": [{"name": "a", "size_vram": 1}]}}))
    with pytest.raises(HostError):
        ps_sizes(make_host(), Recorder({"/api/ps": {"models": [{"name": "a", "size": 1}]}}))


@pytest.mark.parametrize(("path", "payload"), MALFORMED)
@pytest.mark.parametrize(("host", "reason"), HOSTS)
def test_malformed_listing_takes_the_host_out_with_its_reason(
    path: str, payload: object, host: Host, reason: str
) -> None:
    assert read_host(host, Recorder({path: payload}), ["a"]) == down(host, reason)


@pytest.mark.parametrize("error", ERRORS)
@pytest.mark.parametrize("path", ["/api/tags", "/api/ps", "/api/show"])
@pytest.mark.parametrize(("host", "reason"), HOSTS)
def test_unreadable_host_takes_no_new_run(
    error: Exception, path: str, host: Host, reason: str
) -> None:
    assert read_host(host, Recorder(raises={path: error}), ["a"]) == down(host, reason)


@pytest.mark.parametrize(("host", "reason"), HOSTS)
def test_unusable_show_answer_takes_the_host_out_with_its_reason(host: Host, reason: str) -> None:
    assert read_host(host, Recorder(show={"a": ["not", "an", "object"]}), ["a"]) == down(
        host, reason
    )


@pytest.mark.parametrize(
    "answer",
    [{}, {"version": None}, {"version": 9}, {"version": ["0.9"]}, ["version"], "0.9.0", None],
)
def test_executor_without_a_string_version_is_down(answer: object) -> None:
    assert executor_reason(make_host(), Recorder({"/api/version": answer})) == "executor-down"


@pytest.mark.parametrize("error", ERRORS)
def test_executor_probe_failure_is_down(error: Exception) -> None:
    assert executor_reason(make_host(), Recorder(raises={"/api/version": error})) == "executor-down"


def test_executor_with_a_string_version_is_up() -> None:
    assert (
        executor_reason(make_host(), Recorder({"/api/version": {"version": "0.9.0", "x": 1}}))
        is None
    )


def test_kv_shape_posts_the_model_name_to_show() -> None:
    recorder = Recorder()
    assert kv_shape(make_host(), "a", recorder) == LLAMA_SHAPE
    assert recorder.calls == [(EXECUTOR_URL + "/api/show", {"model": "a"}, 5.0)]


def test_kv_shape_reads_the_fields_under_the_architecture_prefix() -> None:
    info = {
        "general.architecture": "gemma2",
        "gemma2.block_count": 26,
        "gemma2.attention.head_count_kv": 4,
        "gemma2.attention.key_length": 256,
        "llama.block_count": 99,
        "llama.attention.head_count_kv": 99,
        "llama.attention.key_length": 99,
    }
    assert shape_of(info) == KvShape(26, 4, 256)


def test_kv_shape_head_dim_falls_back_to_embedding_over_head_count() -> None:
    drop = ("llama.attention.key_length",)
    assert shape_of(patched(drop=drop)) == KvShape(32, 8, 256)
    assert shape_of(patched({"llama.embedding_length": 4100}, drop)) == KvShape(32, 8, 256)


def test_kv_shape_prefers_key_length_over_the_fallback() -> None:
    assert shape_of(patched({"llama.attention.key_length": 128})) == KvShape(32, 8, 128)


def test_kv_shape_accepts_one() -> None:
    changes = {
        "llama.block_count": 1,
        "llama.attention.head_count_kv": 1,
        "llama.attention.key_length": 1,
    }
    assert shape_of(patched(changes)) == KvShape(1, 1, 1)


def test_kv_shape_fallback_head_dim_of_zero_is_missing() -> None:
    changes = {"llama.embedding_length": 8, "llama.attention.head_count": 16}
    assert shape_of(patched(changes, ("llama.attention.key_length",))) is None


@pytest.mark.parametrize(
    "drop",
    [
        ("llama.block_count",),
        ("llama.attention.head_count_kv",),
        ("llama.attention.key_length", "llama.embedding_length"),
        ("llama.attention.key_length", "llama.attention.head_count"),
        ("general.architecture",),
    ],
)
def test_kv_shape_with_a_field_missing_is_none(drop: tuple[str, ...]) -> None:
    assert shape_of(patched(drop=drop)) is None


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "8", None, [8], [8, 8]], ids=repr)
@pytest.mark.parametrize(
    ("key", "drop"),
    [
        ("llama.block_count", ()),
        ("llama.attention.head_count_kv", ()),
        ("llama.attention.key_length", ()),
        ("llama.embedding_length", ("llama.attention.key_length",)),
        ("llama.attention.head_count", ("llama.attention.key_length",)),
        ("general.architecture", ()),
    ],
)
def test_kv_shape_with_a_field_in_a_bad_form_is_none(
    key: str, drop: tuple[str, ...], value: object
) -> None:
    assert shape_of(patched({key: value}, drop)) is None


@pytest.mark.parametrize("model_info", [None, [], "llama", 7])
def test_kv_shape_without_a_model_info_object_is_none(model_info: object) -> None:
    assert shape_of(model_info) is None
    assert kv_shape(make_host(), "a", Recorder(show={"a": {}})) is None


@pytest.mark.parametrize("answer", [[], None, "llama", 7])
def test_kv_shape_with_a_non_object_answer_raises_host_error(answer: object) -> None:
    with pytest.raises(HostError, match="executor.*a"):
        kv_shape(make_host(), "a", Recorder(show={"a": answer}))


def test_kv_shape_through_http_posts_the_body_fakeollama_reads() -> None:
    other = patched({"llama.block_count": 40})
    with FakeOllama("ok", show={"a": LLAMA_INFO, "b": other}) as fake:
        host = make_host(url=fake.url)
        assert kv_shape(host, "a", http_json) == LLAMA_SHAPE
        assert kv_shape(host, "b", http_json) == KvShape(40, 8, 64)
        assert kv_shape(host, "unknown", http_json) is None


def test_read_host_gives_kv_only_for_requested_models_present_in_tags() -> None:
    tags = {"models": [{"name": "a", "size": 1}, {"name": "b", "size": 2}]}
    show = {"b": {"model_info": patched({"llama.block_count": 40})}}
    recorder = Recorder({"/api/tags": tags}, show=show)
    state = read_host(workstation(), recorder, ["a", "c"])
    assert state.kv == {"a": LLAMA_SHAPE}
    assert state.sizes == {"a": 1, "b": 2}
    assert [c for c in recorder.calls if c[0].endswith("/api/show")] == [
        (WS_URL + "/api/show", {"model": "a"}, 5.0)
    ]


def test_read_host_asks_show_once_per_distinct_model() -> None:
    recorder = Recorder()
    state = read_host(workstation(), recorder, ["a", "a"])
    assert state.kv == {"a": LLAMA_SHAPE}
    assert sum(c[0].endswith("/api/show") for c in recorder.calls) == 1


def test_read_host_with_no_models_asks_show_nothing() -> None:
    recorder = Recorder()
    state = read_host(workstation(), recorder, [])
    assert state == HostState(workstation(), None, {"a": 100}, {}, {}, {})
    assert not any(c[0].endswith("/api/show") for c in recorder.calls)


def test_read_host_keeps_a_model_with_missing_kv_fields_as_none() -> None:
    state = read_host(workstation(), Recorder(show={"a": {"model_info": {}}}), ["a"])
    assert state.reason is None
    assert state.kv == {"a": None}


class Answer(io.BytesIO):
    """A urlopen response whose body was promised in full, recording the sizes asked of read()."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.length = 0
        self.asked: list[int | None] = []

    def read(self, size: int | None = -1) -> bytes:
        self.asked.append(size)
        return super().read(size)


def capture_urlopen(
    monkeypatch: pytest.MonkeyPatch, data: bytes
) -> tuple[list[tuple[urllib.request.Request, float]], Answer]:
    seen: list[tuple[urllib.request.Request, float]] = []
    answer = Answer(data)

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> Answer:
        seen.append((request, timeout))
        return answer

    monkeypatch.setattr(hosts.urllib.request, "urlopen", fake_urlopen)
    return seen, answer


def http_answer(body: bytes, length: int | None = None) -> bytes:
    declared = len(body) if length is None else length
    return (
        b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
        % declared
        + body
    )


@contextlib.contextmanager
def raw_server(payload: bytes) -> Iterator[str]:
    """A TCP server that answers every connection with the raw bytes `payload` once the request headers arrive."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.1)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn, contextlib.suppress(OSError):
                conn.settimeout(2)
                seen = b""
                while b"\r\n\r\n" not in seen:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    seen += chunk
                conn.sendall(payload)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        stop.set()
        thread.join(timeout=5)
        listener.close()


def padded_json(size: int) -> bytes:
    """A JSON answer that every host probe accepts, padded to exactly `size` bytes."""
    base = {"version": "1", "models": [], "p": ""}
    body = json.dumps({**base, "p": "x" * (size - len(json.dumps(base)))}).encode("utf-8")
    assert len(body) == size
    return body


OS_ERROR_ANSWERS = {
    "cut-short-valid-json": http_answer(b'{"version": "1"}', 1000),
    "cut-short-mid-json": http_answer(b'{"version": "1', 1000),
    "non-http-banner": b"SSH-2.0-OpenSSH_9.6\r\n",
    "header-line-too-long": b"HTTP/1.0 200 OK\r\nX-Pad: " + b"x" * 70000 + b"\r\n\r\n",
}
DEEP_NESTING = http_answer(b"[" * 200000)
BROKEN_ANSWERS = {**OS_ERROR_ANSWERS, "deep-nesting": DEEP_NESTING}


def test_http_json_without_a_body_is_a_get_with_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, _ = capture_urlopen(monkeypatch, b'{"version": "1"}')
    assert http_json("http://h.test/api/version", None, 3.5) == {"version": "1"}
    [(request, timeout)] = seen
    assert request.full_url == "http://h.test/api/version"
    assert (request.get_method(), request.data, timeout) == ("GET", None, 3.5)


def test_http_json_with_a_body_posts_it_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, _ = capture_urlopen(monkeypatch, b'{"model_info": {}}')
    assert http_json("http://h.test/api/show", {"model": "a"}, 5.0) == {"model_info": {}}
    [(request, timeout)] = seen
    assert request.full_url == "http://h.test/api/show"
    assert request.get_method() == "POST"
    assert request.data == b'{"model": "a"}'
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 5.0


def test_http_json_raises_value_error_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    capture_urlopen(monkeypatch, b"not json")
    with pytest.raises(ValueError):
        http_json("http://h.test/api/tags", None, 1.0)


def test_http_json_asks_for_one_byte_over_the_body_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hosts.MAX_BODY_BYTES == 8 * 1024 * 1024
    _, answer = capture_urlopen(monkeypatch, b"{}")
    http_json("http://h.test/api/tags", None, 1.0)
    assert answer.asked == [8 * 1024 * 1024 + 1]


def test_http_json_parses_the_body_of_a_2xx_over_real_http() -> None:
    with FakeOllama("ok") as fake:
        assert http_json(fake.url + "/api/version", None, 2.0) == {"version": "0.0.0-fake"}
        assert http_json(fake.url + "/api/show", {"model": "x"}, 2.0) == {"model_info": {}}


@pytest.mark.parametrize("body", [None, {"model": "a"}])
def test_http_json_raises_on_non_2xx_and_closes_the_error(
    body: Mapping[str, object] | None,
) -> None:
    with FakeOllama("503") as fake:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            http_json(fake.url + "/api/version", body, 2.0)
    assert exc_info.value.code == 503
    assert exc_info.value.closed


@pytest.mark.parametrize("payload", OS_ERROR_ANSWERS.values(), ids=OS_ERROR_ANSWERS.keys())
def test_http_json_turns_http_client_errors_and_cut_short_bodies_into_os_error(
    payload: bytes,
) -> None:
    with raw_server(payload) as url:
        with pytest.raises(OSError):
            http_json(url + "/api/version", None, 2.0)


def test_http_json_turns_deep_nesting_into_value_error() -> None:
    with raw_server(DEEP_NESTING) as url:
        with pytest.raises(ValueError, match="nested"):
            http_json(url + "/api/version", None, 2.0)


def test_http_json_accepts_a_body_of_exactly_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hosts, "MAX_BODY_BYTES", 1024)
    exact = padded_json(1024)
    with raw_server(http_answer(exact)) as url:
        assert http_json(url + "/api/tags", None, 2.0) == json.loads(exact)


@pytest.mark.parametrize("size", [1025, 2048])
def test_http_json_refuses_a_body_over_the_cap(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    monkeypatch.setattr(hosts, "MAX_BODY_BYTES", 1024)
    with raw_server(http_answer(padded_json(size))) as url:
        with pytest.raises(ValueError, match="over 1024 bytes"):
            http_json(url + "/api/tags", None, 2.0)


@pytest.mark.parametrize("payload", BROKEN_ANSWERS.values(), ids=BROKEN_ANSWERS.keys())
@pytest.mark.parametrize(
    ("name", "reason"), [(WORKSTATION, "ollama-down"), ("executor", "executor-down")]
)
def test_broken_http_answer_takes_the_host_out_with_its_reason(
    payload: bytes, name: str, reason: str
) -> None:
    with raw_server(payload) as url:
        host = make_host(name, url)
        assert read_host(host, http_json, ["a"]) == down(host, reason)


@pytest.mark.parametrize(
    ("name", "reason"), [(WORKSTATION, "ollama-down"), ("executor", "executor-down")]
)
def test_oversized_answer_takes_the_host_out_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, name: str, reason: str
) -> None:
    monkeypatch.setattr(hosts, "MAX_BODY_BYTES", 1024)
    with raw_server(http_answer(padded_json(2048))) as url:
        host = make_host(name, url)
        assert read_host(host, http_json, ["a"]) == down(host, reason)
