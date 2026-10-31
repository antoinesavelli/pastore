"""Check 11: Ollama decode throughput of one model, workstation alone, executor alone, both at once.

Depends on: stdlib only (argparse, http.client, json, pathlib, queue, statistics, sys, threading, time, typing, urllib.parse). Usage, from the repo root:
`uv run python deploy/measure_throughput.py --model <m> --out <results.json> [--workstation URL] [--executor URL] [--repeats N] [--num-predict N]`
"""

import argparse
import http.client
import json
import queue
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

Record = dict[str, Any]

TIMEOUT_S = 300
HOSTS = ("workstation", "executor")
PHASES = {"workstation": ("workstation",), "executor": ("executor",), "both": HOSTS}
PROMPTS = (
    "Explain in one paragraph how a hash table handles collisions.",
    "Write a short Python function that returns the n-th Fibonacci number without recursion.",
    "Summarize in two sentences: A lighthouse keeper logs the weather every hour, signals passing "
    "ships with a lamp, and repairs the lens when salt crusts over it. In storms the keeper stays "
    "awake all night and radios the coast guard if a ship drifts toward the rocks.",
    "List the steps to brew a pour-over coffee, from grinding the beans to serving the cup.",
    "A train leaves at 9:00 going 60 km/h and a second leaves the same station at 10:00 going "
    "90 km/h. When does the second catch up? Show the reasoning.",
)


def _url(text: str) -> str:
    """Argparse type: an http://host:port base URL (urlsplit would fall back to port 80 without one)."""
    try:
        parts = urlsplit(text)
        ok = parts.scheme == "http" and bool(parts.hostname) and parts.port is not None
    except ValueError:
        ok = False
    if not ok:
        raise argparse.ArgumentTypeError(f"{text!r} is not an http://host:port URL")
    return text


def _request(base: str, method: str, path: str, body: bytes | None = None) -> Record:
    """One HTTP call; the JSON object it answers, or SystemExit naming the host and status."""
    conn = http.client.HTTPConnection(urlsplit(base).netloc, timeout=TIMEOUT_S)
    try:
        conn.request(method, path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        status, raw = response.status, response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise SystemExit(f"{base}: {path} failed: {exc}") from exc
    finally:
        conn.close()
    if status != 200:
        raise SystemExit(f"{base}: {path} status {status}")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise SystemExit(f"{base}: {path} status {status}, body is not JSON") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{base}: {path} status {status}, body is not a JSON object")
    return data


def generate(base: str, model: str, prompt: str, num_predict: int) -> Record:
    """One non-streaming POST /api/generate; Ollama's timing fields plus a wall clock around it."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": {"temperature": 0, "seed": 42, "num_predict": num_predict},
        }
    ).encode()
    start, began = time.time(), time.perf_counter()
    data = _request(base, "POST", "/api/generate", body)
    wall = time.perf_counter() - began
    end = time.time()
    eval_count, eval_duration = data.get("eval_count"), data.get("eval_duration")
    if not isinstance(eval_count, int) or not isinstance(eval_duration, int) or eval_duration <= 0:
        raise SystemExit(f"{base}: status 200, missing or invalid eval_count / eval_duration")
    return {
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration,
        # Ollama omits the prompt fields when the whole prompt came from its cache.
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
        "total_duration_ns": data.get("total_duration", 0),
        "wall_s": wall,
        "start": start,
        "end": end,
    }


def gpu_fit(base: str, model: str) -> Record:
    """GET /api/ps; size and size_vram of the loaded model, and whether it sits fully in VRAM."""
    wanted = {model, f"{model}:latest"}
    for entry in _request(base, "GET", "/api/ps").get("models") or []:
        if wanted & {entry.get("name"), entry.get("model")}:
            size, vram = entry.get("size"), entry.get("size_vram")
            if not isinstance(size, int) or not isinstance(vram, int):
                raise SystemExit(f"{base}: /api/ps entry for {model} has no size / size_vram")
            return {"size": size, "size_vram": vram, "full_gpu": size == vram}
    raise SystemExit(f"{base}: {model} is not loaded after the warm-up request (/api/ps)")


def run_sequence(
    args: argparse.Namespace,
    phase: str,
    host: str,
    base: str,
    stop: threading.Event,
    own_done: threading.Event,
    other_done: threading.Event | None,
) -> list[Record]:
    """The full prompts x repeats sequence on one host; in `both` it then keeps cycling until the other host is done."""
    total = args.repeats * len(PROMPTS)
    records: list[Record] = []
    while not stop.is_set():
        extra = len(records) >= total
        if extra and (other_done is None or other_done.is_set()):
            break
        repeat, index = divmod(len(records), len(PROMPTS))
        result = generate(base, args.model, PROMPTS[index], args.num_predict)
        records.append(
            {
                "phase": phase,
                "host": host,
                "prompt_index": index,
                "repeat_index": repeat,
                "extra": extra,
                **result,
            }
        )
        if len(records) == total:
            own_done.set()
    return records


def run_phase(
    args: argparse.Namespace, hosts: dict[str, str], phase: str, stop: threading.Event
) -> list[Record]:
    """Run the phase's hosts at the same time on daemon threads; the first error stops them all."""
    names = PHASES[phase]
    done = {name: threading.Event() for name in names}
    results: queue.Queue[tuple[list[Record], BaseException | None]] = queue.Queue()

    def work(name: str) -> None:
        others = [done[n] for n in names if n != name]
        try:
            records = run_sequence(
                args, phase, name, hosts[name], stop, done[name], others[0] if others else None
            )
            results.put((records, None))
        except BaseException as exc:
            stop.set()
            results.put(([], exc))

    for name in names:
        threading.Thread(target=work, args=(name,), daemon=True).start()
    records: list[Record] = []
    for _ in names:
        part, exc = results.get()
        if exc is not None:
            stop.set()
            raise exc
        records += part
    return sorted(records, key=lambda r: r["start"])


def _tok_s(records: list[Record]) -> float:
    """Decode tokens per second: total eval_count over total eval_duration."""
    seconds: float = sum(r["eval_duration_ns"] for r in records) / 1e9
    tokens: int = sum(r["eval_count"] for r in records)
    return tokens / seconds


def host_stats(records: list[Record]) -> Record:
    """Decode and wall throughput of one host in one phase; repeat figures use the sequence, not the extras."""
    sequence = [r for r in records if not r["extra"]]
    per_repeat = [
        _tok_s([r for r in sequence if r["repeat_index"] == i])
        for i in sorted({r["repeat_index"] for r in sequence})
    ]
    span = max(r["end"] for r in records) - min(r["start"] for r in records)
    tokens: int = sum(r["eval_count"] for r in records)
    return {
        "decode_tok_s": _tok_s(records),
        "repeat_mean": statistics.mean(per_repeat),
        "repeat_min": min(per_repeat),
        "repeat_max": max(per_repeat),
        "wall_tok_s": tokens / span,
    }


def summarize(requests: list[Record]) -> Record:
    """Per-phase per-host stats, then the comparisons check 11 asks for."""
    phases = {
        phase: {
            name: host_stats([r for r in requests if r["phase"] == phase and r["host"] == name])
            for name in names
        }
        for phase, names in PHASES.items()
    }
    alone = {name: phases[name][name] for name in HOSTS}
    both = phases["both"]
    total = sum(s["wall_tok_s"] for s in both.values())
    return {
        "phases": phases,
        "both_total_wall": total,
        "added_pct": (total / alone["workstation"]["wall_tok_s"] - 1) * 100,
        "executor_over_workstation": alone["executor"]["decode_tok_s"]
        / alone["workstation"]["decode_tok_s"],
        "both_over_alone": {
            name: both[name]["decode_tok_s"] / alone[name]["decode_tok_s"] for name in HOSTS
        },
    }


def print_table(summary: Record) -> None:
    """Print the per-phase per-host table and the check 11 comparisons."""
    print(
        f"{'phase':12} {'host':12} {'decode t/s':>11} {'rep mean':>9} "
        f"{'rep min':>8} {'rep max':>8} {'wall t/s':>9}"
    )
    for phase, hosts in summary["phases"].items():
        for name, s in hosts.items():
            print(
                f"{phase:12} {name:12} {s['decode_tok_s']:>11.1f} "
                f"{s['repeat_mean']:>9.1f} {s['repeat_min']:>8.1f} {s['repeat_max']:>8.1f} "
                f"{s['wall_tok_s']:>9.1f}"
            )
    over = summary["both_over_alone"]
    print(
        f"both total wall {summary['both_total_wall']:.1f} tok/s, added {summary['added_pct']:+.1f}%"
    )
    print(f"executor / workstation decode {summary['executor_over_workstation']:.3f}")
    print(
        f"both vs alone (decode): workstation {over['workstation']:.3f}, executor {over['executor']:.3f}"
    )


def measure(args: argparse.Namespace, out: Path, stop: threading.Event) -> None:
    """Warm up and check GPU fit on both hosts, run the three phases, print the table, write the JSON."""
    hosts = {"workstation": args.workstation, "executor": args.executor}
    fits: dict[str, Record] = {}
    for name, base in hosts.items():
        generate(base, args.model, PROMPTS[0], args.num_predict)
        fit = fits[name] = gpu_fit(base, args.model)
        print(
            f"{name} {base}: size {fit['size']} size_vram {fit['size_vram']} "
            f"full GPU {'yes' if fit['full_gpu'] else 'NO'}",
            flush=True,
        )
        if not fit["full_gpu"]:
            print(
                f"WARNING: {name} {args.model} is not fully on the GPU; its tok/s is not a GPU number",
                flush=True,
            )
    phases: dict[str, Record] = {}
    requests: list[Record] = []
    for phase in PHASES:
        start = time.time()
        part = run_phase(args, hosts, phase, stop)
        phases[phase] = {"start": start, "end": time.time()}
        requests += part
        print(
            f"phase {phase}: done in {phases[phase]['end'] - start:.1f} s, "
            f"{len(part)} requests, {sum(r['extra'] for r in part)} extra",
            flush=True,
        )
    summary = summarize(requests)
    result = {
        "model": args.model,
        "num_predict": args.num_predict,
        "repeats": args.repeats,
        "hosts": hosts,
        "gpu_fit": fits,
        "phases": phases,
        "requests": requests,
        "summary": summary,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print_table(summary)
    print(f"wrote {out}", flush=True)


def main() -> int:
    """Parse and validate the arguments, run the measurement, and stop everything at once on Ctrl-C (exit 130)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstation", type=_url, default="http://127.0.0.1:11435")
    parser.add_argument("--executor", type=_url, default="http://127.0.0.1:11436")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.num_predict < 1:
        parser.error("--repeats and --num-predict must be at least 1")
    out = Path(args.out)
    if not out.parent.is_dir():
        raise SystemExit(f"{out.parent}: not a directory")
    stop = threading.Event()
    try:
        measure(args, out, stop)
    except KeyboardInterrupt:
        stop.set()
        print("interrupted, stopped all requests", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
