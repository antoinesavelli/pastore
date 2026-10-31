"""The arithmetic of the hand-run measurement scripts in deploy/ (checks 10, 11, 12), with injected input.

Depends on: deploy/measure_ttfb.py, deploy/measure_throughput.py, deploy/sample_executor.py.
"""

import argparse
import http.client
import importlib.util
import json
import subprocess
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"


def _import(name: str) -> ModuleType:
    """Load deploy/<name>.py as a module without running its main() (deploy/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"deploy_{name}", DEPLOY / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ttfb() -> ModuleType:
    return _import("measure_ttfb")


@pytest.fixture(scope="module")
def tp() -> ModuleType:
    return _import("measure_throughput")


@pytest.fixture(scope="module")
def sx() -> ModuleType:
    return _import("sample_executor")


@pytest.fixture(autouse=True)
def _no_network_no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any real HTTP connection or child process from a script under test fails the test at once."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a deploy script tried to open a connection or start a process")

    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


# --- check 10: measure_ttfb.py -----------------------------------------------------------------


def test_ttfb_p95_is_the_inclusive_95th_percentile(ttfb: ModuleType) -> None:
    """1..21 has its 95th percentile at position 19 (value 20); two points interpolate; ties stay put."""
    assert ttfb._p95([float(x) for x in range(1, 22)]) == 20.0
    assert ttfb._p95([float(x) for x in range(21, 0, -1)]) == 20.0
    assert ttfb._p95([10.0, 20.0]) == pytest.approx(19.5)
    assert ttfb._p95([5.0] * 10) == 5.0


class _Clock:
    """A settable perf_counter; the fake connection moves it the way the network would."""

    def __init__(self) -> None:
        self.now = 2.0

    def perf_counter(self) -> float:
        return self.now


class _Response:
    def __init__(self, status: int, first: bytes, clock: _Clock) -> None:
        self.status, self._first, self._clock = status, first, clock

    def read(self, amt: int | None = None) -> bytes:
        if amt == 1:
            self._clock.now += 0.125
            return self._first
        self._clock.now += 100.0
        return b""


def _fake_connection(
    monkeypatch: pytest.MonkeyPatch, ttfb: ModuleType, status: int, first: bytes
) -> list[tuple[Any, ...]]:
    """Replace the connection and the clock of ttfb_ms; the list collects its connect and request events.

    The headers arrive 0.125 s after the send, the first byte 0.125 s later, the rest of the stream 100 s later.
    """
    sent: list[tuple[Any, ...]] = []
    clock = _Clock()

    class Conn:
        def __init__(self, host: str, port: int | None, timeout: int) -> None:
            sent.append(("connect", host, port))

        def request(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> None:
            sent.append(("request", method, path, body))

        def getresponse(self) -> _Response:
            clock.now += 0.125
            return _Response(status, first, clock)

        def close(self) -> None:
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", Conn)
    monkeypatch.setattr(ttfb, "time", clock)
    return sent


def test_ttfb_ms_is_milliseconds_from_send_to_first_byte(
    monkeypatch: pytest.MonkeyPatch, ttfb: ModuleType
) -> None:
    """A quarter second from the send to the first byte reads 250 ms, whatever the drain takes after."""
    sent = _fake_connection(monkeypatch, ttfb, 200, b"{")
    assert ttfb.ttfb_ms("http://10.1.2.3:11435", b"BODY") == 250.0
    assert sent == [
        ("connect", "10.1.2.3", 11435),
        ("request", "POST", "/api/generate", b"BODY"),
    ]


@pytest.mark.parametrize(("status", "first"), [(503, b"x"), (200, b"")])
def test_ttfb_ms_refuses_a_bad_status_or_an_empty_stream(
    monkeypatch: pytest.MonkeyPatch, ttfb: ModuleType, status: int, first: bytes
) -> None:
    """A non-200 answer or a stream with no first byte stops the run instead of recording a time."""
    _fake_connection(monkeypatch, ttfb, status, first)
    with pytest.raises(SystemExit):
        ttfb.ttfb_ms("http://h:1", b"")


def test_ttfb_main_warms_up_then_alternates_and_reports_gateway_minus_direct(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], ttfb: ModuleType
) -> None:
    """Two unmeasured warm-ups, then direct and gateway in turn; p50 is the median, added p95 is gateway - direct."""
    script = {
        "http://d:1": iter([999.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        "http://g:2": iter([999.0, 12.0, 22.0, 32.0, 42.0, 59.0]),
    }
    calls: list[str] = []

    def fake_ttfb(base: str, body: bytes) -> float:
        calls.append(base)
        return next(script[base])

    monkeypatch.setattr(ttfb, "ttfb_ms", fake_ttfb)
    argv = ["measure_ttfb", "--direct", "http://d:1", "--gateway", "http://g:2", "--model", "m"]
    monkeypatch.setattr("sys.argv", [*argv, "-n", "5"])
    assert ttfb.main() == 0
    assert calls == ["http://d:1", "http://g:2"] * 6
    # direct p95: 40 + (50 - 40) * 0.8 = 48; gateway p95: 42 + (59 - 42) * 0.8 = 55.6; added 7.6
    assert capsys.readouterr().out.splitlines() == [
        "direct   n=5 p50=30.00ms p95=48.00ms",
        "gateway  n=5 p50=32.00ms p95=55.60ms",
        "added p95=7.60ms (check 10 passes under 5 ms)",
    ]


# --- check 11: measure_throughput.py -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("http://127.0.0.1:11435", True),
        ("http://second-host.lan:11436", True),
        ("http://127.0.0.1", False),
        ("https://127.0.0.1:11435", False),
        ("127.0.0.1:11435", False),
        ("http://h:notaport", False),
    ],
)
def test_throughput_url_needs_http_host_and_port(tp: ModuleType, text: str, ok: bool) -> None:
    """urlsplit would fall back to port 80, so a URL without an explicit port is refused."""
    if ok:
        assert tp._url(text) == text
    else:
        with pytest.raises(argparse.ArgumentTypeError):
            tp._url(text)


def _fake_clocks(monkeypatch: pytest.MonkeyPatch, tp: ModuleType) -> None:
    perf, wall = iter([5.0, 7.5]), iter([100.0, 102.5])
    fake = SimpleNamespace(perf_counter=lambda: next(perf), time=lambda: next(wall))
    monkeypatch.setattr(tp, "time", fake)


def test_generate_keeps_ollamas_fields_and_the_wall_clock_around_the_call(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """Ollama's eval and prompt fields are renamed with _ns; wall_s is the perf_counter gap, start and end the epochs."""
    sent: list[tuple[str, str, str, bytes | None]] = []
    answer = {
        "eval_count": 200,
        "eval_duration": 4_000_000_000,
        "prompt_eval_count": 12,
        "prompt_eval_duration": 1_000,
        "total_duration": 5_000,
    }

    def fake_request(
        base: str, method: str, path: str, body: bytes | None = None
    ) -> dict[str, Any]:
        sent.append((base, method, path, body))
        return answer

    monkeypatch.setattr(tp, "_request", fake_request)
    _fake_clocks(monkeypatch, tp)
    assert tp.generate("http://h:1", "m", "hello", 64) == {
        "eval_count": 200,
        "eval_duration_ns": 4_000_000_000,
        "prompt_eval_count": 12,
        "prompt_eval_duration_ns": 1_000,
        "total_duration_ns": 5_000,
        "wall_s": 2.5,
        "start": 100.0,
        "end": 102.5,
    }
    base, method, path, body = sent[0]
    assert (base, method, path) == ("http://h:1", "POST", "/api/generate")
    assert body is not None
    request = json.loads(body)
    assert (request["model"], request["prompt"], request["stream"]) == ("m", "hello", False)
    assert request["options"] == {"temperature": 0, "seed": 42, "num_predict": 64}


def test_generate_defaults_the_prompt_fields_ollama_omits_to_zero(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """A fully cached prompt has no prompt_eval fields; they read 0, the eval figures stay."""
    monkeypatch.setattr(
        tp, "_request", lambda *_a, **_k: {"eval_count": 7, "eval_duration": 1_000_000_000}
    )
    _fake_clocks(monkeypatch, tp)
    got = tp.generate("http://h:1", "m", "p", 8)
    assert (got["prompt_eval_count"], got["prompt_eval_duration_ns"], got["total_duration_ns"]) == (
        0,
        0,
        0,
    )
    assert (got["eval_count"], got["eval_duration_ns"]) == (7, 1_000_000_000)


@pytest.mark.parametrize(
    "answer",
    [
        {"eval_duration": 10},
        {"eval_count": 5},
        {"eval_count": 5, "eval_duration": 0},
        {"eval_count": 5, "eval_duration": -3},
        {"eval_count": "5", "eval_duration": 10},
    ],
)
def test_generate_refuses_a_missing_or_zero_eval_figure(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType, answer: dict[str, Any]
) -> None:
    """A throughput of tokens over a zero or absent duration is refused, not computed."""
    monkeypatch.setattr(tp, "_request", lambda *_a, **_k: answer)
    _fake_clocks(monkeypatch, tp)
    with pytest.raises(SystemExit):
        tp.generate("http://h:1", "m", "p", 8)


def test_gpu_fit_is_full_gpu_only_when_size_equals_size_vram(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """The entry named `m` or `m:latest` is read; another model's entry before it is skipped."""
    models = [
        {"name": "other:1", "model": "other:1", "size": 1, "size_vram": 1},
        {"name": "m:latest", "model": "m:latest", "size": 100, "size_vram": 100},
    ]
    monkeypatch.setattr(tp, "_request", lambda *_a, **_k: {"models": models})
    assert tp.gpu_fit("http://h:1", "m") == {"size": 100, "size_vram": 100, "full_gpu": True}
    models[1]["size_vram"] = 60
    assert tp.gpu_fit("http://h:1", "m") == {"size": 100, "size_vram": 60, "full_gpu": False}
    assert tp.gpu_fit("http://h:1", "m:latest")["full_gpu"] is False


@pytest.mark.parametrize(
    "answer",
    [{"models": []}, {"models": None}, {}, {"models": [{"name": "m", "model": "m", "size": 5}]}],
)
def test_gpu_fit_refuses_an_unloaded_model_or_an_entry_without_sizes(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType, answer: dict[str, Any]
) -> None:
    """No loaded entry for the model, or one without size and size_vram, stops the run."""
    monkeypatch.setattr(tp, "_request", lambda *_a, **_k: answer)
    with pytest.raises(SystemExit):
        tp.gpu_fit("http://h:1", "m")


def _gen(count: int, dur_ns: int, start: float = 0.0, end: float = 1.0) -> dict[str, Any]:
    """A fake generate() result with the fields host_stats reads."""
    return {"eval_count": count, "eval_duration_ns": dur_ns, "start": start, "end": end}


def _req(phase: str, host: str, repeat: int, extra: bool, **result: Any) -> dict[str, Any]:
    return {"phase": phase, "host": host, "repeat_index": repeat, "extra": extra, **result}


def test_tok_s_is_total_tokens_over_total_seconds_not_a_mean_of_ratios(tp: ModuleType) -> None:
    """100 tok in 1 s plus 100 tok in 3 s is 200/4 = 50 tok/s; the mean of 100 and 33.3 would be 66.7."""
    assert tp._tok_s([_gen(100, 1_000_000_000), _gen(100, 3_000_000_000)]) == 50.0
    assert tp._tok_s([_gen(150, 1_000_000_000)]) == 150.0


def test_host_stats_repeats_exclude_extras_but_decode_and_wall_include_them(
    tp: ModuleType,
) -> None:
    """Three repeats (50, 200, 150 tok/s) and one extra request; the extra counts only in decode and wall."""
    s = 1_000_000_000
    records = [
        _req("both", "w", 0, False, **_gen(100, s, 100.0, 101.0)),
        _req("both", "w", 0, False, **_gen(100, 3 * s, 101.0, 104.0)),
        _req("both", "w", 1, False, **_gen(200, s, 104.0, 105.0)),
        _req("both", "w", 1, False, **_gen(200, s, 105.0, 106.0)),
        _req("both", "w", 2, False, **_gen(150, s, 106.0, 108.0)),
        _req("both", "w", 3, True, **_gen(50, s, 109.0, 110.0)),
    ]
    got = tp.host_stats(records)
    # tokens 800 over 8 s of eval; wall span 100.0 to 110.0 is 10 s
    assert got["decode_tok_s"] == 100.0
    assert got["wall_tok_s"] == 80.0
    assert got["repeat_min"] == 50.0
    assert got["repeat_max"] == 200.0
    assert got["repeat_mean"] == pytest.approx((50.0 + 200.0 + 150.0) / 3)


def test_host_stats_with_one_request_has_equal_repeat_figures(tp: ModuleType) -> None:
    """A single sample is its own mean, min and max; the wall span is that request's own start to end."""
    got = tp.host_stats([_req("workstation", "w", 0, False, **_gen(90, 2_000_000_000, 5.0, 8.0))])
    assert got == {
        "decode_tok_s": 45.0,
        "repeat_mean": 45.0,
        "repeat_min": 45.0,
        "repeat_max": 45.0,
        "wall_tok_s": 30.0,
    }


def _summary_requests() -> list[dict[str, Any]]:
    """One request per (phase, host), each over a 2 s span; decode tok/s 100, 40, 80 and 25."""
    s = 1_000_000_000
    return [
        _req("workstation", "workstation", 0, False, **_gen(100, s, 0.0, 2.0)),
        _req("executor", "executor", 0, False, **_gen(40, s, 0.0, 2.0)),
        _req("both", "workstation", 0, False, **_gen(80, s, 0.0, 2.0)),
        _req("both", "executor", 0, False, **_gen(25, s, 0.0, 2.0)),
    ]


def test_summarize_compares_both_against_each_host_alone(tp: ModuleType) -> None:
    """Wall tok/s 50 and 20 alone, 40 and 12.5 together: total 52.5 is +5% over the workstation's 50."""
    got = tp.summarize(_summary_requests())
    assert got["both_total_wall"] == 52.5
    assert got["added_pct"] == pytest.approx(5.0)
    assert got["executor_over_workstation"] == 0.4
    assert got["both_over_alone"] == {"workstation": 0.8, "executor": 0.625}
    assert got["phases"]["workstation"]["workstation"]["wall_tok_s"] == 50.0
    assert got["phases"]["executor"]["executor"]["decode_tok_s"] == 40.0
    assert got["phases"]["both"]["executor"]["wall_tok_s"] == 12.5
    assert set(got["phases"]) == {"workstation", "executor", "both"}


def test_print_table_rows_and_comparison_lines(
    capsys: pytest.CaptureFixture[str], tp: ModuleType
) -> None:
    """One row per phase and host with one-decimal figures, then the three comparison lines."""
    tp.print_table(tp.summarize(_summary_requests()))
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["workstation"] * 2 + ["100.0"] * 4 + ["50.0"]
    assert lines[2].split() == ["executor", "executor", "40.0", "40.0", "40.0", "40.0", "20.0"]
    assert lines[4].split() == ["both", "executor", "25.0", "25.0", "25.0", "25.0", "12.5"]
    assert lines[-3:] == [
        "both total wall 52.5 tok/s, added +5.0%",
        "executor / workstation decode 0.400",
        "both vs alone (decode): workstation 0.800, executor 0.625",
    ]
    assert len(lines) == 1 + 4 + 3


def _args(repeats: int) -> argparse.Namespace:
    return argparse.Namespace(repeats=repeats, model="m", num_predict=8)


def _scripted_generate(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType, hook: Callable[[int], None] | None = None
) -> list[str]:
    """Replace generate(); it returns a clocked result per call, runs `hook(call_number)`, lists the prompts."""
    prompts: list[str] = []

    def fake(base: str, model: str, prompt: str, num_predict: int) -> dict[str, Any]:
        prompts.append(prompt)
        if len(prompts) > 200:
            raise RuntimeError("runaway loop: run_sequence never stopped")
        if hook is not None:
            hook(len(prompts))
        return _gen(10, 1_000_000_000, float(len(prompts)), len(prompts) + 0.5)

    monkeypatch.setattr(tp, "generate", fake)
    return prompts


def test_run_sequence_alone_runs_prompts_times_repeats_and_no_extras(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """Two repeats of the five prompts is ten requests in prompt order; own_done is set by the tenth, not before."""
    prompts = _scripted_generate(monkeypatch, tp)
    own_done = threading.Event()
    got = tp.run_sequence(
        _args(2), "workstation", "workstation", "http://h:1", threading.Event(), own_done, None
    )
    assert [(r["repeat_index"], r["prompt_index"], r["extra"]) for r in got] == [
        (r, p, False) for r in range(2) for p in range(5)
    ]
    assert prompts == list(tp.PROMPTS) * 2
    assert own_done.is_set()
    assert {r["phase"] for r in got} == {"workstation"}


def test_run_sequence_in_both_keeps_cycling_as_extras_until_the_other_host_is_done(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """One repeat is five requests; the other host finishing during call 8 leaves calls 6 to 8 as extras."""
    other_done = threading.Event()
    own_done = threading.Event()
    seen_done_at: list[int] = []

    def hook(call: int) -> None:
        if own_done.is_set() and not seen_done_at:
            seen_done_at.append(call - 1)
        if call == 8:
            other_done.set()

    _scripted_generate(monkeypatch, tp, hook)
    got = tp.run_sequence(
        _args(1), "both", "executor", "http://h:1", threading.Event(), own_done, other_done
    )
    assert seen_done_at == [5]
    assert [(r["repeat_index"], r["prompt_index"], r["extra"]) for r in got] == [
        (0, 0, False),
        (0, 1, False),
        (0, 2, False),
        (0, 3, False),
        (0, 4, False),
        (1, 0, True),
        (1, 1, True),
        (1, 2, True),
    ]


def test_run_sequence_with_the_other_host_already_done_stops_at_the_sequence(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """If the other host finished first, this one still runs its full sequence and then stops with no extras."""
    _scripted_generate(monkeypatch, tp)
    other_done = threading.Event()
    other_done.set()
    got = tp.run_sequence(
        _args(1), "both", "w", "http://h:1", threading.Event(), threading.Event(), other_done
    )
    assert len(got) == 5
    assert not any(r["extra"] for r in got)


def test_run_sequence_stops_at_once_when_the_stop_event_is_set(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """A stop set during the second request ends the loop after it, short of the sequence and of own_done."""
    stop, own_done = threading.Event(), threading.Event()
    _scripted_generate(monkeypatch, tp, lambda call: stop.set() if call == 2 else None)
    got = tp.run_sequence(_args(3), "workstation", "w", "http://h:1", stop, own_done, None)
    assert len(got) == 2
    assert not own_done.is_set()


def test_run_phase_both_runs_hosts_together_and_extras_are_only_the_early_finisher(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """The executor waits until the workstation has cycled past its sequence; only the workstation has extras."""
    total = 5
    count_by_base: Counter[str] = Counter()
    lock = threading.Lock()
    ahead = threading.Event()

    def fake(base: str, model: str, prompt: str, num_predict: int) -> dict[str, Any]:
        with lock:
            count_by_base[base] += 1
            n = count_by_base[base]
            if n > 1_000_000:
                raise RuntimeError("runaway loop: run_sequence never stopped")
            if base == "http://w:1" and n >= total + 2:
                ahead.set()
        if base == "http://e:2" and n == 1:
            assert ahead.wait(timeout=10)
        return _gen(10, 1_000_000_000, float(n), n + 0.5)

    monkeypatch.setattr(tp, "generate", fake)
    hosts = {"workstation": "http://w:1", "executor": "http://e:2"}
    got = tp.run_phase(_args(1), hosts, "both", threading.Event())
    by_host = {h: [r for r in got if r["host"] == h] for h in hosts}
    assert [r["extra"] for r in by_host["executor"]] == [False] * total
    assert [r["extra"] for r in by_host["workstation"]][:total] == [False] * total
    assert len(by_host["workstation"]) >= total + 2
    assert all(r["extra"] for r in by_host["workstation"][total:])
    assert [r["start"] for r in got] == sorted(r["start"] for r in got)
    assert {r["phase"] for r in got} == {"both"}


def test_run_phase_raises_the_first_error_and_stops_the_other_host(
    monkeypatch: pytest.MonkeyPatch, tp: ModuleType
) -> None:
    """One host failing re-raises its exception in the caller and sets stop so the other host's loop ends."""
    stop = threading.Event()

    def fake(base: str, model: str, prompt: str, num_predict: int) -> dict[str, Any]:
        if base == "http://e:2":
            raise RuntimeError("executor down")
        stop.wait(timeout=0.001)
        return _gen(10, 1_000_000_000)

    monkeypatch.setattr(tp, "generate", fake)
    hosts = {"workstation": "http://w:1", "executor": "http://e:2"}
    with pytest.raises(RuntimeError, match="executor down"):
        tp.run_phase(_args(1), hosts, "both", stop)
    assert stop.is_set()


# --- check 12: sample_executor.py --------------------------------------------------------------


def test_remote_script_validates_the_interface_and_formats_the_settings(sx: ModuleType) -> None:
    """Three setting lines (interval via %g, duration as given), then the constant body; odd interface names refuse."""
    text = sx.remote_script("wlan0", 5.0, 900)
    assert text.startswith("IFACE='wlan0'\nINTERVAL='5'\nDURATION=900\n")
    assert text.endswith(sx.REMOTE_BODY)
    assert sx.remote_script("en-p0s3.1:0", 2.5, 3).splitlines()[1] == "INTERVAL='2.5'"
    for bad in ("", "wl an", "x'; rm -rf /", "a$(b)", "a\nb"):
        with pytest.raises(SystemExit):
            sx.remote_script(bad, 5.0, 1)


def test_num_reads_a_float_or_none(sx: ModuleType) -> None:
    assert sx._num("1.5") == 1.5
    assert sx._num("-40") == -40.0
    assert sx._num(None) is None
    assert sx._num("[N/A]") is None
    assert sx._num("") is None


def test_cpu_times_total_and_idle_use_the_first_eight_fields(sx: ModuleType) -> None:
    """Total is user..steal (eight fields, guest columns left out); idle is idle plus iowait."""
    assert sx._cpu_times(["cpu  10 0 20 300 40 0 0 0 0 0"]) == (370.0, 340.0)
    assert sx._cpu_times(["cpu  1 1 1 1 1 1 1 1 100 100"]) == (8.0, 2.0)
    assert sx._cpu_times(["cpu  1 2 3 4 5"]) == (15.0, 9.0)  # the shortest line with an iowait
    assert sx._cpu_times(["cpu  1 2 3 4"]) is None  # no iowait field
    assert sx._cpu_times(["cpu0 1 2 3 4 5 6 7 8"]) is None
    assert sx._cpu_times(["cpu  1 2 x 4 5 6"]) is None
    assert sx._cpu_times(["cpu  1 2 3 4 5 6 7 x"]) is None  # all eight fields must be digits
    assert sx._cpu_times([]) is None


def test_gpu_needs_exactly_three_numeric_cells(sx: ModuleType) -> None:
    assert sx._gpu(["55, 10240, 61"]) == [55.0, 10240.0, 61.0]
    assert sx._gpu(["[N/A], 10, 61"]) == [None, 10.0, 61.0]
    assert sx._gpu(["55, 10240"]) == [None, None, None]
    assert sx._gpu(["1, 2, 3, 4"]) == [None, None, None]
    assert sx._gpu([]) == [None, None, None]


def test_wifi_quality_ratio_and_signal_from_iwconfig_and_iw(sx: ModuleType) -> None:
    """iwconfig gives a ratio rounded to 3 places and dBm; iw gives dBm only; a zero denominator gives no ratio."""
    assert sx._wifi('wlan0  ESSID:"x"  Link Quality=45/70  Signal level=-65 dBm') == (0.643, -65.0)
    assert sx._wifi("Link Quality:35/70  Signal level:-50 dBm") == (0.5, -50.0)
    assert sx._wifi("Connected to aa:bb\n\tsignal: -52 dBm\n") == (None, -52.0)
    assert sx._wifi("Link Quality=0/0  Signal level=-70 dBm") == (None, -70.0)
    assert sx._wifi("missing") == (None, None)
    assert sx._wifi("") == (None, None)


def test_mb_rounds_to_two_places_and_passes_none(sx: ModuleType) -> None:
    assert sx._mb(None, 1024) is None
    assert sx._mb(1536, 1024) == 1.5
    assert sx._mb(1000, 1024) == 0.98
    assert sx._mb(0.0, 1024) == 0.0


def test_grab_returns_the_first_numeric_group_or_none(sx: ModuleType) -> None:
    lines = ["MemTotal: 100 kB", "MemAvailable: 42 kB"]
    assert sx._grab(lines, r"^MemAvailable:\s+(\d+)") == 42.0
    assert sx._grab(lines, r"^SwapTotal:\s+(\d+)") is None
    assert sx._grab([], r"^(\S+)") is None


def test_parse_raw_converts_every_section_to_its_field(sx: ModuleType) -> None:
    """Each section's text becomes its named number: kB to MB, bytes to MB, swap used as total minus free."""
    sections = {
        "psi_cpu": ["some avg10=1.50 avg60=0.00 avg300=0.00 total=123"],
        "psi_mem": [
            "some avg10=0.10 avg60=0 avg300=0 total=1",
            "full avg10=0.05 avg60=0 avg300=0 total=1",
        ],
        "psi_io": [
            "some avg10=2.00 avg60=0 avg300=0 total=1",
            "full avg10=1.00 avg60=0 avg300=0 total=1",
        ],
        "stat": ["cpu  100 0 100 800 0 0 0 0 0 0", "cpu0 1 1 1 1 1 1 1 1 1 1"],
        "load": ["0.52 0.40 0.30 1/200 1234"],
        "mem": [
            "MemTotal:       16000000 kB",
            "MemAvailable:    2097152 kB",
            "SwapTotal:       4194304 kB",
            "SwapFree:        3145728 kB",
        ],
        "net": ["rx_bytes 6000", "tx_bytes 2500", "rx_dropped 4", "tx_errors missing"],
        "wifi": ["wlan0  Link Quality=70/70  Signal level=-40 dBm"],
        "gpu": ["55, 10240, 61"],
        "ollama_mem": ["2147483648"],
        "ollama_cpu": ["usage_usec 1", "nr_periods 10", "nr_throttled 3", "throttled_usec 4"],
    }
    assert sx.parse_raw(sections) == {
        "load1": 0.52,
        "mem_avail_mb": 2048.0,
        "swap_used_mb": 1024.0,
        "wifi_quality": 1.0,
        "wifi_signal_dbm": -40.0,
        "gpu_util_pct": 55.0,
        "gpu_mem_mb": 10240.0,
        "gpu_temp_c": 61.0,
        "ollama_mem_mb": 2048.0,
        "cpu_total": 1000.0,
        "cpu_idle": 800.0,
        "nr_throttled": 3.0,
        "psi_cpu_some10": 1.5,
        "psi_mem_some10": 0.1,
        "psi_mem_full10": 0.05,
        "psi_io_some10": 2.0,
        "psi_io_full10": 1.0,
        "rx_bytes": 6000.0,
        "tx_bytes": 2500.0,
        "rx_dropped": 4.0,
        "tx_errors": None,
    }


def test_parse_raw_missing_sections_are_none_and_zero_swap_is_zero_not_none(sx: ModuleType) -> None:
    """Sections that came back as the single word `missing`, or not at all, give None; swap 0 - 0 is 0.0."""
    empty = sx.parse_raw({})
    assert len(empty) == 21 and all(v is None for v in empty.values())
    gone = sx.parse_raw({k: ["missing"] for k in ("gpu", "stat", "mem", "net", "wifi", "load")})
    assert all(v is None for v in gone.values())
    zero_swap = sx.parse_raw({"mem": ["SwapTotal: 0 kB", "SwapFree: 0 kB"]})
    assert zero_swap["swap_used_mb"] == 0.0
    assert zero_swap["mem_avail_mb"] is None


def _nums(**override: float | None) -> dict[str, float | None]:
    base: dict[str, float | None] = dict.fromkeys(
        (
            "cpu_total",
            "cpu_idle",
            "rx_bytes",
            "tx_bytes",
            "rx_dropped",
            "tx_errors",
            "nr_throttled",
        ),
        0.0,
    )
    return {**base, **override}


def test_derive_busy_percent_rates_and_deltas_from_two_samples(sx: ModuleType) -> None:
    """CPU 800 jiffies elapsed with 600 idle is 25% busy; 5000 B and 500 B over 5 s are 1000 and 100 B/s."""
    prev = _nums(
        cpu_total=1000, cpu_idle=800, rx_bytes=1000, tx_bytes=2000, rx_dropped=0, nr_throttled=3
    )
    now = _nums(
        cpu_total=1800, cpu_idle=1400, rx_bytes=6000, tx_bytes=2500, rx_dropped=4, nr_throttled=10
    )
    assert sx._derive(now, prev, 5.0) == {
        "cpu_busy_pct": 25.0,
        "wifi_rx_Bps": 1000.0,
        "wifi_tx_Bps": 100.0,
        "wifi_rx_dropped_delta": 4.0,
        "wifi_tx_errors_delta": 0.0,
        "ollama_throttled_delta": 7.0,
    }


def test_derive_without_a_previous_sample_is_all_none(sx: ModuleType) -> None:
    got = sx._derive(_nums(cpu_total=5), None, 5.0)
    assert set(got.values()) == {None} and len(got) == 6


def test_derive_rounding_resets_and_unusable_intervals(sx: ModuleType) -> None:
    """Busy rounds to 2 places and rates to 1; a reset counter, a zero CPU total or a bad interval gives None."""
    prev = _nums(cpu_total=0, cpu_idle=0, rx_bytes=0, tx_bytes=100)
    now = _nums(cpu_total=3, cpu_idle=1, rx_bytes=1000, tx_bytes=50)
    got = sx._derive(now, prev, 3.0)
    assert got["cpu_busy_pct"] == 66.67  # 100 * (1 - 1/3)
    assert got["wifi_rx_Bps"] == 333.3  # 1000 / 3
    assert got["wifi_tx_Bps"] is None  # tx_bytes went backwards: counter reset
    for seconds in (None, 0.0, -2.0):
        no_rate = sx._derive(now, prev, seconds)
        assert no_rate["wifi_rx_Bps"] is None
        assert no_rate["cpu_busy_pct"] == 66.67
    still = sx._derive(_nums(cpu_total=7, cpu_idle=7), _nums(cpu_total=7, cpu_idle=7), 5.0)
    assert still["cpu_busy_pct"] is None  # no jiffies elapsed
    assert still["wifi_rx_dropped_delta"] == 0.0  # an unchanged counter is a delta of 0, not None


def _sample(stamp: str, stat: str, rx: int, tx: int, dropped: int, throttled: int) -> list[str]:
    """The marker lines of one remote sample, the way REMOTE_BODY prints them."""
    return [
        f"@@sample {stamp}",
        "@@psi_cpu",
        "some avg10=1.50 avg60=0.00 avg300=0.00 total=123",
        "@@psi_mem",
        "some avg10=0.10 avg60=0.00 avg300=0.00 total=1",
        "full avg10=0.05 avg60=0.00 avg300=0.00 total=1",
        "@@psi_io",
        "some avg10=2.00 avg60=0.00 avg300=0.00 total=1",
        "full avg10=1.00 avg60=0.00 avg300=0.00 total=1",
        "@@stat",
        stat,
        "@@load",
        "0.52 0.40 0.30 1/200 1234",
        "@@mem",
        "MemTotal:       16000000 kB",
        "MemAvailable:    2097152 kB",
        "SwapTotal:       4194304 kB",
        "SwapFree:        3145728 kB",
        "@@net",
        f"rx_bytes {rx}",
        f"tx_bytes {tx}",
        f"rx_dropped {dropped}",
        "tx_errors 0",
        "@@wifi",
        "wlan0  Link Quality=70/70  Signal level=-40 dBm",
        "@@gpu",
        "55, 10240, 61",
        "@@ollama_mem",
        "2147483648",
        "@@ollama_cpu",
        "usage_usec 1",
        f"nr_throttled {throttled}",
        "@@end",
    ]


LEVELS = {
    "load1": 0.52,
    "mem_avail_mb": 2048.0,
    "swap_used_mb": 1024.0,
    "wifi_quality": 1.0,
    "wifi_signal_dbm": -40.0,
    "gpu_util_pct": 55.0,
    "gpu_mem_mb": 10240.0,
    "gpu_temp_c": 61.0,
    "ollama_mem_mb": 2048.0,
    "psi_cpu_some10": 1.5,
    "psi_mem_some10": 0.1,
    "psi_mem_full10": 0.05,
    "psi_io_some10": 2.0,
    "psi_io_full10": 1.0,
}


def _clock(*values: float) -> Callable[[], float]:
    it: Iterator[float] = iter(values)
    return lambda: next(it)


def test_exec_records_stamp_with_the_injected_clock_and_derive_from_the_previous_sample(
    sx: ModuleType,
) -> None:
    """ts is now() at each sample marker; the executor's own stamps (5 s apart) turn byte deltas into rates."""
    lines = [
        "stray line before the first marker",
        *_sample("1000.5", "cpu  100 0 100 800 0 0 0 0 0 0", 1000, 2000, 0, 3),
        *_sample("1005.5", "cpu  200 0 200 1400 0 0 0 0 0 0", 6000, 2500, 4, 10),
    ]
    first, second = list(sx.exec_records(lines, _clock(50.0, 60.0)))
    assert first == {
        "kind": "exec",
        "ts": 50.0,
        **LEVELS,
        "cpu_busy_pct": None,
        "wifi_rx_Bps": None,
        "wifi_tx_Bps": None,
        "wifi_rx_dropped_delta": None,
        "wifi_tx_errors_delta": None,
        "ollama_throttled_delta": None,
    }
    assert second == {
        "kind": "exec",
        "ts": 60.0,
        **LEVELS,
        "cpu_busy_pct": 25.0,
        "wifi_rx_Bps": 1000.0,
        "wifi_tx_Bps": 100.0,
        "wifi_rx_dropped_delta": 4.0,
        "wifi_tx_errors_delta": 0.0,
        "ollama_throttled_delta": 7.0,
    }


def test_exec_records_without_usable_stamps_give_no_rates_and_a_cut_sample_gives_no_record(
    sx: ModuleType,
) -> None:
    """An unparseable stamp leaves the byte rates None (counter deltas still work); a sample with no @@end is dropped."""
    lines = [
        *_sample("not-a-number", "cpu  100 0 100 800 0 0 0 0 0 0", 1000, 2000, 0, 3),
        *_sample("also-bad", "cpu  200 0 200 1400 0 0 0 0 0 0", 6000, 2500, 4, 10),
        *_sample("3.0", "cpu  300 0 300 2000 0 0 0 0 0 0", 7000, 2600, 4, 10)[:-1],
    ]
    records = list(sx.exec_records(lines, _clock(1.0, 2.0, 3.0)))
    assert [r["ts"] for r in records] == [1.0, 2.0]
    assert records[1]["wifi_rx_Bps"] is None and records[1]["wifi_tx_Bps"] is None
    assert records[1]["wifi_rx_dropped_delta"] == 4.0
    assert records[1]["cpu_busy_pct"] == 25.0


@pytest.mark.parametrize("second_stamp", ["10.0", "9.0"])
def test_exec_records_equal_or_backwards_executor_stamps_give_no_rate(
    sx: ModuleType, second_stamp: str
) -> None:
    """The second sample 0 s or -1 s after the first has no byte rate (no division by zero)."""
    lines = [
        *_sample("10.0", "cpu  100 0 100 800 0 0 0 0 0 0", 1000, 2000, 0, 3),
        *_sample(second_stamp, "cpu  200 0 200 1400 0 0 0 0 0 0", 6000, 2500, 4, 10),
    ]
    second = list(sx.exec_records(lines, _clock(1.0, 2.0)))[1]
    assert second["wifi_rx_Bps"] is None and second["wifi_tx_Bps"] is None


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "[1700000000.123456] 64 bytes from 192.0.2.101: icmp_seq=5 ttl=64 time=3.21 ms",
            {"kind": "ping", "ts": 1700000000.123456, "seq": 5, "rtt_ms": 3.21},
        ),
        (
            "[1700000001.000000] no answer yet for icmp_seq=6",
            {"kind": "ping", "ts": 1700000001.0, "seq": 6, "rtt_ms": None},
        ),
        (
            "[1700000002] 64 bytes from 192.0.2.101: icmp_seq=7 ttl=64 time=120 ms",
            {"kind": "ping", "ts": 1700000002.0, "seq": 7, "rtt_ms": 120.0},
        ),
        ("[1700000003.5] 64 bytes from 192.0.2.101: icmp_seq=7 ttl=64 time=4.0 ms (DUP!)", None),
        ("PING 192.0.2.101 (192.0.2.101) 56(84) bytes of data.", None),
        ("--- 192.0.2.101 ping statistics ---", None),
        ("rtt min/avg/max/mdev = 1.0/2.0/3.0/0.5 ms", None),
        ("", None),
    ],
)
def test_parse_ping_reads_reply_loss_and_ignores_the_rest(
    sx: ModuleType, line: str, expected: dict[str, Any] | None
) -> None:
    """A reply carries its rtt, a loss line none; duplicates, headers and the summary are not probes."""
    assert sx.parse_ping(line) == expected


def _ping(seq: int, rtt: float | None, ts: float = 0.0) -> dict[str, Any]:
    return {"kind": "ping", "ts": ts, "seq": seq, "rtt_ms": rtt}


def test_resolve_pings_keeps_a_reply_over_a_loss_and_the_first_of_equals(sx: ModuleType) -> None:
    """Per icmp_seq: a later reply replaces an earlier loss; a loss never replaces a reply; the first of two replies stays."""
    got = sx._resolve_pings(
        [
            _ping(1, None, 1.0),
            _ping(1, 4.0, 1.5),
            _ping(2, 2.0, 2.0),
            _ping(2, None, 2.5),
            _ping(3, None, 3.0),
            _ping(3, None, 3.5),
            _ping(4, 1.0, 4.0),
            _ping(4, 9.0, 4.5),
        ]
    )
    assert got == [_ping(1, 4.0, 1.5), _ping(2, 2.0, 2.0), _ping(3, None, 3.0), _ping(4, 1.0, 4.0)]
    assert sx._resolve_pings([]) == []


def test_percentile_interpolates_between_ranks(sx: ModuleType) -> None:
    """Linear interpolation over the sorted list: rank (n - 1) * q."""
    five = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert sx._percentile(five, 0.5) == 3.0
    assert sx._percentile(five, 0.95) == pytest.approx(4.8)
    assert sx._percentile(five, 0.0) == 1.0
    assert sx._percentile(five, 1.0) == 5.0
    assert sx._percentile([10.0, 20.0], 0.95) == pytest.approx(19.5)
    assert sx._percentile([7.0], 0.95) == 7.0
    assert sx._percentile([5.0, 5.0, 5.0], 0.95) == 5.0


def test_ping_stats_loss_percent_percentiles_and_max(sx: ModuleType) -> None:
    """Three replies and one loss: 25% loss, p50 2.0, p95 2 + 1 * 0.9 = 2.9, max 3.0."""
    pings = [_ping(1, 3.0), _ping(2, 1.0), _ping(3, None), _ping(4, 2.0)]
    assert sx._ping_stats(pings) == {
        "count": 4,
        "loss_pct": 25.0,
        "p50": 2.0,
        "p95": 2.9,
        "max": 3.0,
    }
    third = sx._ping_stats([_ping(1, 1.0), _ping(2, None), _ping(3, None)])
    assert third["loss_pct"] == 66.67
    assert (third["p50"], third["p95"], third["max"]) == (1.0, 1.0, 1.0)


def test_ping_stats_for_no_probes_and_for_all_lost(sx: ModuleType) -> None:
    assert sx._ping_stats([]) == {
        "count": 0,
        "loss_pct": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
    assert sx._ping_stats([_ping(1, None), _ping(2, None)]) == {
        "count": 2,
        "loss_pct": 100.0,
        "p50": None,
        "p95": None,
        "max": None,
    }


def test_window_stats_mean_and_max_of_the_numeric_values_only(sx: ModuleType) -> None:
    """None and text are skipped; a key with no number in the window is absent; exec_count counts every record."""
    execs = [
        {"kind": "exec", "ts": 1.0, "load1": 1.0, "gpu": None, "note": "x"},
        {"kind": "exec", "ts": 2.0, "load1": 2.0, "gpu": 50, "note": "y"},
        {"kind": "exec", "ts": 3.0, "load1": 4.5, "gpu": 90, "note": "z"},
    ]
    got = sx._window_stats(execs, [_ping(1, 2.0)], ["load1", "gpu", "note", "absent"])
    assert got["exec_count"] == 3
    assert got["metrics"] == {
        "load1": {"mean": 2.5, "max": 4.5},
        "gpu": {"mean": 70.0, "max": 90.0},
    }
    assert got["ping"]["count"] == 1
    assert sx._window_stats([], [], ["load1"]) == {
        "exec_count": 0,
        "metrics": {},
        "ping": sx._ping_stats([]),
    }


def test_window_stats_rounds_mean_and_max_to_two_places(sx: ModuleType) -> None:
    execs = [{"v": 1.0}, {"v": 1.0}, {"v": 2.0}]
    assert sx._window_stats(execs, [], ["v"])["metrics"]["v"] == {"mean": 1.33, "max": 2.0}


def test_fmt_integers_whole_hundreds_and_two_places(sx: ModuleType) -> None:
    assert sx._fmt(None) == "-"
    assert sx._fmt(42) == "42"
    assert sx._fmt(0.0) == "0.00"
    assert sx._fmt(99.994) == "99.99"
    assert sx._fmt(100.0) == "100"
    assert sx._fmt(123.456) == "123"
    assert sx._fmt(-123.456) == "-123"
    assert sx._fmt(-5.0) == "-5.00"


def test_render_is_a_table_with_metrics_as_rows_and_windows_as_columns(sx: ModuleType) -> None:
    """Header, one row per key (`-` where a window has no value), then the five ping rows; columns aligned."""
    nothing = {"count": 0, "loss_pct": None, "p50": None, "p95": None, "max": None}
    stats = {
        "before": {
            "exec_count": 2,
            "metrics": {"load1": {"mean": 0.5, "max": 0.9}},
            "ping": {"count": 4, "loss_pct": 25.0, "p50": 2.0, "p95": 2.9, "max": 3.0},
        },
        "both": {
            "exec_count": 3,
            "metrics": {"load1": {"mean": 123.456, "max": 200}},
            "ping": nothing,
        },
    }
    lines = sx._render(stats, ["load1", "gpu_util_pct"]).split("\n")
    assert not any(line != line.rstrip() for line in lines)
    cells = [(line[0:17].rstrip(), line[19:29].rstrip(), line[31:].rstrip()) for line in lines]
    assert cells == [
        ("metric (mean/max)", "before n=2", "both n=3"),
        ("load1", "0.50/0.90", "123/200"),
        ("gpu_util_pct", "-", "-"),
        ("ping_n", "4", "0"),
        ("ping_p50_ms", "2.00", "-"),
        ("ping_p95_ms", "2.90", "-"),
        ("ping_max_ms", "3.00", "-"),
        ("ping_loss_pct", "25.00", "-"),
    ]
    assert lines[0].index("before") == 19 and lines[0].index("both") == 31


def test_reach_summary_counts_up_and_down_and_keeps_raw_lines_under_ten(sx: ModuleType) -> None:
    """Up/down are counted from the `health=` text; nine lines keep their raw text, ten do not."""
    lines = ["t1 health=ok", "t2 health=down", "t3 health=ok", "t4 something else"]
    assert sx._reach_summary(lines) == {"lines": 4, "up": 2, "down": 1, "raw": lines}
    nine = ["x health=ok"] * 9
    assert sx._reach_summary(nine) == {"lines": 9, "up": 9, "down": 0, "raw": nine}
    ten = ["x health=down"] * 10
    assert sx._reach_summary(ten) == {"lines": 10, "up": 0, "down": 10}
    assert sx._reach_summary([]) == {"lines": 0, "up": 0, "down": 0, "raw": []}


def _write(path: Path, text: str) -> str:
    path.write_text(text)
    return str(path)


def test_windows_are_before_each_phase_by_start_then_after_the_latest_end(
    tmp_path: Path, sx: ModuleType
) -> None:
    """Phases sort by start whatever their order in the file; `after` begins at the largest end, not the last."""
    phases = {
        "phases": {
            "b": {"start": 10, "end": 20},
            "a": {"start": 0.5, "end": 100},
            "c": {"start": 15, "end": 16},
        }
    }
    got = sx._windows(_write(tmp_path / "t.json", json.dumps(phases)))
    inf = float("inf")
    assert got == [
        ("before", -inf, 0.5),
        ("a", 0.5, 100.0),
        ("b", 10.0, 20.0),
        ("c", 15.0, 16.0),
        ("after", 100.0, inf),
    ]


@pytest.mark.parametrize(
    "text",
    [
        '{"phases": {}}',
        "{}",
        '{"phases": {"a": {"start": 1}}}',
        '{"phases": {"a": {"start": "soon", "end": 2}}}',
        '{"phases": {"a": 5}}',
        '{"phases": [1, 2]}',
    ],
)
def test_windows_refuse_a_phases_file_without_numeric_starts_and_ends(
    tmp_path: Path, sx: ModuleType, text: str
) -> None:
    with pytest.raises(SystemExit):
        sx._windows(_write(tmp_path / "t.json", text))


def test_load_groups_records_by_kind_and_names_the_bad_line(tmp_path: Path, sx: ModuleType) -> None:
    good = _write(
        tmp_path / "ok.jsonl",
        '{"kind": "exec", "ts": 1}\n\n{"kind": "ping", "ts": 2}\n{"kind": "exec", "ts": 3}\n',
    )
    got = sx._load(good)
    assert {k: [r["ts"] for r in v] for k, v in got.items()} == {"exec": [1, 3], "ping": [2]}
    with pytest.raises(SystemExit, match=r"bad.jsonl:2:"):
        sx._load(_write(tmp_path / "bad.jsonl", '{"kind": "exec"}\n{not json\n'))
    with pytest.raises(SystemExit, match=r"nokind.jsonl:1: not a record with a kind"):
        sx._load(_write(tmp_path / "nokind.jsonl", '{"ts": 1}\n'))
    with pytest.raises(SystemExit, match=r"list.jsonl:1: not a record"):
        sx._load(_write(tmp_path / "list.jsonl", "[1]\n"))


def _jsonl(*records: dict[str, Any]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def _exec(ts: float, **fields: float | None) -> dict[str, Any]:
    return {"kind": "exec", "ts": ts, **fields}


def test_summarize_assigns_records_to_half_open_windows_and_prints_table_reach_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], sx: ModuleType
) -> None:
    """A record exactly at a phase's start belongs to it, one exactly at its end to the next window."""
    phases = _write(
        tmp_path / "t.json",
        json.dumps(
            {
                "phases": {
                    "workstation": {"start": 10.0, "end": 20.0},
                    "both": {"start": 20.0, "end": 30.0},
                }
            }
        ),
    )
    samples = _write(
        tmp_path / "s.jsonl",
        _jsonl(
            _exec(5.0, load1=1.0),
            _exec(10.0, load1=2.0, gpu_util_pct=50),
            _exec(12.0, load1=3.0, gpu_util_pct=None),
            _exec(15.0, load1=4.0, gpu_util_pct=90),
            _exec(20.0, load1=6.0),
            _exec(29.9, load1=8.0),
            _exec(30.0, load1=10.0),
            _ping(1, 1.0, 4.0),
            _ping(2, None, 11.0),
            _ping(2, 3.0, 11.5),
            _ping(3, 1.0, 14.0),
            _ping(4, None, 10.0),
            _ping(5, 5.0, 31.0),
            _ping(6, 7.0, 20.0),
            {"kind": "reach", "lines": ["t1 health=ok", "t2 health=down"]},
            {"kind": "reach", "lines": ["t3 health=ok"]},
        ),
    )
    assert sx.summarize(argparse.Namespace(samples=samples, phases=phases, json=True)) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == [
        "metric",
        "(mean/max)",
        "before",
        "n=1",
        "workstation",
        "n=3",
        "both",
        "n=2",
        "after",
        "n=1",
    ]
    assert out[-5:-1] == [
        "reach: 3 lines, up=2 down=1",
        "  t1 health=ok",
        "  t2 health=down",
        "  t3 health=ok",
    ]
    stats = json.loads(out[-1])
    assert stats["reach"] == {
        "lines": 3,
        "up": 2,
        "down": 1,
        "raw": ["t1 health=ok", "t2 health=down", "t3 health=ok"],
    }
    windows = stats["windows"]
    assert list(windows) == ["before", "workstation", "both", "after"]
    assert {n: w["exec_count"] for n, w in windows.items()} == {
        "before": 1,
        "workstation": 3,
        "both": 2,
        "after": 1,
    }
    assert windows["before"]["metrics"] == {"load1": {"mean": 1.0, "max": 1.0}}
    assert windows["workstation"]["metrics"] == {
        "load1": {"mean": 3.0, "max": 4.0},
        "gpu_util_pct": {"mean": 70.0, "max": 90.0},
    }
    assert windows["both"]["metrics"] == {"load1": {"mean": 7.0, "max": 8.0}}
    assert windows["after"]["metrics"] == {"load1": {"mean": 10.0, "max": 10.0}}
    # workstation pings: seq 2 resolved to its reply (3.0), seq 3 (1.0), seq 4 lost at exactly 10.0
    assert windows["workstation"]["ping"] == {
        "count": 3,
        "loss_pct": 33.33,
        "p50": 2.0,
        "p95": 2.9,
        "max": 3.0,
    }
    assert windows["before"]["ping"] == {
        "count": 1,
        "loss_pct": 0.0,
        "p50": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }
    # the probe at exactly 20.0 is the first of `both`, not the last of `workstation`
    assert windows["both"]["ping"] == {
        "count": 1,
        "loss_pct": 0.0,
        "p50": 7.0,
        "p95": 7.0,
        "max": 7.0,
    }
    assert windows["after"]["ping"]["max"] == 5.0


def test_summarize_of_an_empty_samples_file_prints_empty_windows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], sx: ModuleType
) -> None:
    """No records at all still prints a table (every window n=0, ping rows `-`) and a zero reach line, no JSON."""
    phases = _write(
        tmp_path / "t.json", json.dumps({"phases": {"only": {"start": 1.0, "end": 2.0}}})
    )
    samples = _write(tmp_path / "s.jsonl", "")
    assert sx.summarize(argparse.Namespace(samples=samples, phases=phases, json=False)) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == [
        "metric",
        "(mean/max)",
        "before",
        "n=0",
        "only",
        "n=0",
        "after",
        "n=0",
    ]
    assert out[1].split() == ["ping_n", "0", "0", "0"]
    assert out[2].split() == ["ping_p50_ms", "-", "-", "-"]
    assert out[-1] == "reach: 0 lines, up=0 down=0"
    assert len(out) == 1 + 5 + 1
