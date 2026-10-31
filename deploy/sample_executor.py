"""Check 12: sample a second host's load over read-only ssh while a job runs on it, ping it side by side, then compare.

Depends on: stdlib only (argparse, json, subprocess, threading, signal, re) and `ssh`, `ping`, `journalctl` on
PATH. The ssh alias, the ping address, the network interface, the health unit whose journal is counted and the
text that marks its up and down lines are arguments with generic defaults. Usage, from the repo root:
`uv run python deploy/sample_executor.py record --out samples.jsonl --duration 900 [--ssh-host H] [--ping-host IP]`
`uv run python deploy/sample_executor.py summarize samples.jsonl --phases throughput.json [--json]`
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from typing import IO, Any

SSH_HOST = "second-host"
GRACE_S = 30.0
STOP_WAIT_S = 5.0
REACH_UNIT = "workload-health"
UP_TEXT = "health=ok"
DOWN_TEXT = "health=down"
REACH_CAP = 200

Record = dict[str, Any]
Nums = dict[str, float | None]
COUNTERS = {
    "cpu_total",
    "cpu_idle",
    "rx_bytes",
    "tx_bytes",
    "rx_dropped",
    "tx_errors",
    "nr_throttled",
}

# Read-only on purpose: cat, head, grep, date, sleep, iwconfig <iface>, iw dev <iface> link, nvidia-smi --query-gpu.
REMOTE_BODY = """
PATH="$PATH:/usr/sbin:/sbin"
mk() { printf '@@%s\\n' "$1"; }
run() {
  mk "$1"; shift
  if command -v "$1" >/dev/null 2>&1; then "$@" </dev/null 2>/dev/null || echo missing; else echo missing; fi
}
stop_at=$(( $(date +%s) + DURATION + 1 ))
while [ "$(date +%s)" -lt "$stop_at" ]; do
  printf '@@sample %s\\n' "$(date +%s.%N)"
  run psi_cpu cat /proc/pressure/cpu
  run psi_mem cat /proc/pressure/memory
  run psi_io cat /proc/pressure/io
  run stat head -1 /proc/stat
  run load cat /proc/loadavg
  run mem grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
  mk net
  for s in rx_bytes tx_bytes rx_dropped tx_errors; do
    printf '%s %s\\n' "$s" "$(cat "/sys/class/net/$IFACE/statistics/$s" 2>/dev/null || echo missing)"
  done
  mk wifi
  if command -v iwconfig >/dev/null 2>&1; then iwconfig "$IFACE" 2>&1
  elif command -v iw >/dev/null 2>&1; then iw dev "$IFACE" link 2>&1
  else echo missing; fi
  run gpu nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
  run ollama_mem cat /sys/fs/cgroup/system.slice/ollama.service/memory.current
  run ollama_cpu cat /sys/fs/cgroup/system.slice/ollama.service/cpu.stat
  mk end
  sleep "$INTERVAL"
done
exit 0
"""

PING_HEAD = re.compile(r"^\[(\d+(?:\.\d+)?)\].*icmp_seq=(\d+)")


def remote_script(iface: str, interval: float, duration: int) -> str:
    """The POSIX sh text fed to the second host: validated settings on top, then the constant read-only loop."""
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", iface):
        raise SystemExit(f"bad --iface {iface!r}")
    return f"IFACE='{iface}'\nINTERVAL='{interval:g}'\nDURATION={duration}\n{REMOTE_BODY}"


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _lines(sections: dict[str, list[str]], name: str) -> list[str]:
    got = sections.get(name, [])
    return [] if got == ["missing"] else got


def _grab(lines: list[str], pattern: str) -> float | None:
    """The first numeric group of `pattern` in the section text, or None."""
    found = re.search(pattern, "\n".join(lines), re.MULTILINE)
    return _num(found.group(1)) if found else None


def _cpu_times(lines: list[str]) -> tuple[float, float] | None:
    """(total, idle) jiffies from the first /proc/stat line."""
    parts = lines[0].split() if lines else []
    if len(parts) < 6 or parts[0] != "cpu" or not all(p.isdigit() for p in parts[1:9]):
        return None
    vals = [float(p) for p in parts[1:9]]
    return sum(vals), vals[3] + vals[4]


def _gpu(lines: list[str]) -> list[float | None]:
    """(util %, memory MB, temperature C) from the first nvidia-smi csv line."""
    cells = [_num(c.strip()) for c in lines[0].split(",")] if lines else []
    return cells if len(cells) == 3 else [None, None, None]


def _wifi(text: str) -> tuple[float | None, float | None]:
    """(iwconfig link quality as a 0..1 ratio, signal dBm); iw output gives the signal only."""
    link = re.search(r"Link Quality[=:]\s*(\d+)/(\d+)", text)
    dbm = re.search(r"(?:Signal level[=:]|signal:)\s*(-?\d+)\s*dBm", text)
    ratio = round(int(link[1]) / int(link[2]), 3) if link and int(link[2]) else None
    return ratio, _num(dbm[1]) if dbm else None


def _mb(value: float | None, divisor: float) -> float | None:
    return None if value is None else round(value / divisor, 2)


def parse_raw(sections: dict[str, list[str]]) -> Nums:
    """Every number of one sample: levels under the record's names, counters under COUNTERS' names."""
    mem, net = _lines(sections, "mem"), _lines(sections, "net")
    cpu = _cpu_times(_lines(sections, "stat"))
    gpu = _gpu(_lines(sections, "gpu"))
    swap_total, swap_free = _grab(mem, r"^SwapTotal:\s+(\d+)"), _grab(mem, r"^SwapFree:\s+(\d+)")
    quality, signal_dbm = _wifi("\n".join(_lines(sections, "wifi")))
    swap_used = None if swap_total is None or swap_free is None else swap_total - swap_free
    nums: Nums = {
        "load1": _grab(_lines(sections, "load"), r"^(\S+)"),
        "mem_avail_mb": _mb(_grab(mem, r"^MemAvailable:\s+(\d+)"), 1024),
        "swap_used_mb": _mb(swap_used, 1024),
        "wifi_quality": quality,
        "wifi_signal_dbm": signal_dbm,
        "gpu_util_pct": gpu[0],
        "gpu_mem_mb": gpu[1],
        "gpu_temp_c": gpu[2],
        "ollama_mem_mb": _mb(_grab(_lines(sections, "ollama_mem"), r"^(\d+)"), 1048576),
        "cpu_total": cpu[0] if cpu else None,
        "cpu_idle": cpu[1] if cpu else None,
        "nr_throttled": _grab(_lines(sections, "ollama_cpu"), r"^nr_throttled (\d+)"),
    }
    for name, kinds in (("cpu", ("some",)), ("mem", ("some", "full")), ("io", ("some", "full"))):
        for kind in kinds:
            pattern = rf"^{kind} .*?avg10=([\d.]+)"
            nums[f"psi_{name}_{kind}10"] = _grab(_lines(sections, f"psi_{name}"), pattern)
    for counter in ("rx_bytes", "tx_bytes", "rx_dropped", "tx_errors"):
        nums[counter] = _grab(net, rf"^{counter} (\d+)")
    return nums


def _delta(nums: Nums, prev: Nums | None, key: str) -> float | None:
    if prev is None:
        return None
    now, before = nums[key], prev[key]
    if now is None or before is None or now < before:
        return None
    return now - before


def _derive(nums: Nums, prev: Nums | None, seconds: float | None) -> Nums:
    """Fields that need the previous sample: CPU busy percent, byte rates, counter deltas."""
    total, idle = _delta(nums, prev, "cpu_total"), _delta(nums, prev, "cpu_idle")
    busy = None if not total or idle is None else round(100.0 * (1.0 - idle / total), 2)
    out: Nums = {"cpu_busy_pct": busy}
    for key, counter in (("wifi_rx_Bps", "rx_bytes"), ("wifi_tx_Bps", "tx_bytes")):
        diff = _delta(nums, prev, counter)
        out[key] = None if diff is None or not seconds or seconds <= 0 else round(diff / seconds, 1)
    out["wifi_rx_dropped_delta"] = _delta(nums, prev, "rx_dropped")
    out["wifi_tx_errors_delta"] = _delta(nums, prev, "tx_errors")
    out["ollama_throttled_delta"] = _delta(nums, prev, "nr_throttled")
    return out


def exec_records(lines: Iterable[str], now: Callable[[], float]) -> Iterator[Record]:
    """One `exec` record per sample in the remote marker stream.

    `ts` is `now()` when the sample marker arrives (the local clock, like the phases and the pings);
    the sampled host's own stamp only divides the byte deltas into per-second rates.
    """
    prev: Nums | None = None
    prev_stamp: float | None = None
    sections: dict[str, list[str]] | None = None
    name, ts = "", 0.0
    stamp: float | None = None
    for line in lines:
        if line.startswith("@@sample"):
            ts, stamp, sections, name = now(), _num(line[8:]), {}, ""
        elif sections is None:
            continue
        elif line == "@@end":
            nums = parse_raw(sections)
            seconds = None if stamp is None or prev_stamp is None else stamp - prev_stamp
            levels = {k: v for k, v in nums.items() if k not in COUNTERS}
            yield {"kind": "exec", "ts": ts, **levels, **_derive(nums, prev, seconds)}
            prev, prev_stamp, sections = nums, stamp, None
        elif line.startswith("@@"):
            name = line[2:]
            sections[name] = []
        elif name:
            sections[name].append(line)


def parse_ping(line: str) -> Record | None:
    """A `ping -D -O` line as {"kind": "ping", "ts", "seq", "rtt_ms"} (rtt null for a loss line); None if not a probe line."""
    head = PING_HEAD.match(line)
    if head is None or "(DUP!)" in line:
        return None
    rtt = re.search(r"time=([\d.]+) ms", line)
    return {
        "kind": "ping",
        "ts": float(head[1]),
        "seq": int(head[2]),
        "rtt_ms": float(rtt[1]) if rtt else None,
    }


def _writer(handle: IO[str]) -> Callable[[Record], None]:
    """A thread-safe appender: one flushed JSON line per record."""
    lock = threading.Lock()

    def write(record: Record) -> None:
        with lock:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    return write


def read_reach(since: float, unit: str, markers: tuple[str, str]) -> list[str]:
    """The health unit's journal lines since `since` that hold an up or a down marker, system then user scope, last REACH_CAP."""
    lines: list[str] = []
    for scope in ([], ["--user"]):
        argv = ["journalctl", *scope, "-u", unit, "--since", f"@{int(since)}"]
        try:
            done = subprocess.run(
                [*argv, "--no-pager", "-o", "short-iso"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"journalctl failed: {exc}", file=sys.stderr)
            continue
        lines += [x for x in done.stdout.splitlines() if any(m in x for m in markers)]
    return lines[-REACH_CAP:]


def _spawn(argv: list[str], stdin: IO[str] | int, stderr: IO[str]) -> subprocess.Popen[str]:
    """Start a child in its own session, so a terminal Ctrl-C reaches only us and we stop it in order."""
    try:
        return subprocess.Popen(
            argv,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise SystemExit(f"cannot start {argv[0]}: {exc}") from exc


def _stop(proc: subprocess.Popen[str]) -> None:
    """Terminate the child's whole process group, wait, and kill it after STOP_WAIT_S."""
    if proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=STOP_WAIT_S)
            return
        except ProcessLookupError:
            proc.wait()
            return
        except subprocess.TimeoutExpired:
            continue


def _stdout_lines(proc: subprocess.Popen[str]) -> Iterator[str]:
    for line in proc.stdout or ():
        yield line.rstrip("\n")


def _pump(records: Iterable[Record], write: Callable[[Record], None], counts: Counter[str]) -> None:
    for item in records:
        write(item)
        counts[item["kind"]] += 1


def _run_children(
    args: argparse.Namespace,
    files: tuple[IO[str], IO[str], IO[str]],
    write: Callable[[Record], None],
    counts: Counter[str],
    caught: list[int],
) -> tuple[int | None, int | None]:
    """Run ssh (script on stdin) and ping side by side; return their exit codes as seen when the wait ended.

    The wait ends when ssh exits, a signal is caught, or the duration plus one interval plus GRACE_S passes.
    """
    script, ssh_err, ping_err = files
    ssh_argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-T",
        args.ssh_host,
        "sh",
        "-s",
    ]
    count = str(math.ceil(args.duration + args.interval) + 2)
    ping_argv = ["ping", "-D", "-O", "-i", "1", "-c", count, args.ping_host]
    procs: list[subprocess.Popen[str]] = []
    threads: list[threading.Thread] = []
    codes: tuple[int | None, int | None] = (None, None)
    try:
        ssh = _spawn(ssh_argv, script, ssh_err)
        procs.append(ssh)
        ping = _spawn(ping_argv, subprocess.DEVNULL, ping_err)
        procs.append(ping)
        pings = (r for r in map(parse_ping, _stdout_lines(ping)) if r is not None)
        for records in (exec_records(_stdout_lines(ssh), time.time), pings):
            threads.append(
                threading.Thread(target=_pump, args=(records, write, counts), daemon=True)
            )
            threads[-1].start()
        deadline = time.monotonic() + args.duration + args.interval + GRACE_S
        while not caught and ssh.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        codes = (ssh.poll(), ping.poll())
    finally:
        for proc in procs:
            _stop(proc)
        for thread in threads:
            thread.join(timeout=STOP_WAIT_S)
    return codes


def record(args: argparse.Namespace) -> int:
    """Sample the second host over one SSH session and ping it side by side.

    Exit 0 for a full run, 130 or 143 after SIGINT or SIGTERM, 1 when ssh or ping failed, ssh hung past
    the deadline, or no exec sample was recorded.
    """
    if not all(math.isfinite(x) and x > 0 for x in (args.duration, args.interval)):
        raise SystemExit("--duration and --interval must be finite and positive")
    text = remote_script(args.iface, args.interval, math.ceil(args.duration))
    try:
        out = open(args.out, "x", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(
            f"cannot create {args.out}: {exc.strerror} (--out must be a new file)"
        ) from exc
    caught: list[int] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, _frame: caught.append(signum))
    start, counts = time.time(), Counter[str]()
    problems: list[str] = []
    with (
        out,
        tempfile.TemporaryFile("w+") as script,
        tempfile.TemporaryFile("w+") as ssh_err,
        tempfile.TemporaryFile("w+") as ping_err,
    ):
        script.write(text)
        script.seek(0)
        write = _writer(out)
        ssh_code, ping_code = _run_children(
            args, (script, ssh_err, ping_err), write, counts, caught
        )
        for name, code, handle in (("ssh", ssh_code, ssh_err), ("ping", ping_code, ping_err)):
            if code not in (None, 0):
                handle.seek(0)
                problems.append(f"{name} exited {code}: {handle.read().strip()}")
        if ssh_code is None and not caught:
            problems.append(f"ssh still running {GRACE_S:.0f} s past the end; stopped it")
        if not counts["exec"] and not caught and ssh_code in (None, 0):
            problems.append("no exec sample was recorded")
        write(
            {
                "kind": "reach",
                "lines": read_reach(start, args.reach_unit, (args.up_text, args.down_text)),
            }
        )
    for problem in problems:
        print(problem, file=sys.stderr)
    if caught:
        return 128 + caught[0]
    return 1 if problems else 0


def _load(path: str) -> dict[str, list[Record]]:
    """Group the JSONL records by kind."""
    groups: dict[str, list[Record]] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: {exc}") from exc
            if not isinstance(rec, dict) or "kind" not in rec:
                raise SystemExit(f"{path}:{number}: not a record with a kind")
            groups.setdefault(rec["kind"], []).append(rec)
    return groups


def _windows(path: str) -> list[tuple[str, float, float]]:
    """(name, from, until) half-open: before, each phase in start order, after."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        spans = sorted(
            (float(p["start"]), float(p["end"]), name) for name, p in data["phases"].items()
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise SystemExit(
            f"{path}: expected phases.<name>.start and .end as epoch numbers ({exc!r})"
        ) from exc
    if not spans:
        raise SystemExit(f"{path}: phases is empty")
    last_end = max(end for _, end, _ in spans)
    return [
        ("before", -math.inf, spans[0][0]),
        *((n, s, e) for s, e, n in spans),
        ("after", last_end, math.inf),
    ]


def _resolve_pings(pings: list[Record]) -> list[Record]:
    """One record per icmp_seq: its reply if it ever got one, else its first loss line."""
    best: dict[int, Record] = {}
    for item in pings:
        seen = best.get(item["seq"])
        if seen is None or (seen["rtt_ms"] is None and item["rtt_ms"] is not None):
            best[item["seq"]] = item
    return list(best.values())


def _percentile(ordered: list[float], q: float) -> float:
    spot = (len(ordered) - 1) * q
    low = math.floor(spot)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (spot - low)


def _ping_stats(pings: list[Record]) -> Record:
    rtts = sorted(r["rtt_ms"] for r in pings if r.get("rtt_ms") is not None)
    stats: Record = {"count": len(pings), "loss_pct": None, "p50": None, "p95": None, "max": None}
    if pings:
        stats["loss_pct"] = round(100.0 * (len(pings) - len(rtts)) / len(pings), 2)
    if rtts:
        stats.update(
            p50=round(_percentile(rtts, 0.5), 2),
            p95=round(_percentile(rtts, 0.95), 2),
            max=rtts[-1],
        )
    return stats


def _window_stats(execs: list[Record], pings: list[Record], keys: list[str]) -> Record:
    """Count, mean and max of every numeric exec metric, plus the ping figures, for one window."""
    metrics: Record = {}
    for key in keys:
        vals = [r[key] for r in execs if isinstance(r.get(key), int | float)]
        if vals:
            metrics[key] = {"mean": round(sum(vals) / len(vals), 2), "max": round(max(vals), 2)}
    return {"exec_count": len(execs), "metrics": metrics, "ping": _ping_stats(pings)}


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.2f}"


def _render(stats: dict[str, Record], keys: list[str]) -> str:
    """Metrics as rows, windows as columns; metric cells are mean/max."""
    header = ["metric (mean/max)", *(f"{w} n={s['exec_count']}" for w, s in stats.items())]
    rows = [header]
    for key in keys:
        cells = []
        for s in stats.values():
            m = s["metrics"].get(key)
            cells.append("-" if m is None else f"{_fmt(m['mean'])}/{_fmt(m['max'])}")
        rows.append([key, *cells])
    pings = ("count", "ping_n"), ("p50", "ping_p50_ms"), ("p95", "ping_p95_ms")
    for field, label in (*pings, ("max", "ping_max_ms"), ("loss_pct", "ping_loss_pct")):
        rows.append([label, *(_fmt(s["ping"][field]) for s in stats.values())])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows
    )


def _reach_summary(lines: list[str], up_text: str = UP_TEXT, down_text: str = DOWN_TEXT) -> Record:
    """Count the health lines holding the up text and those holding the down text; keep the raw lines when there are few."""
    summary: Record = {
        "lines": len(lines),
        "up": sum(up_text in x for x in lines),
        "down": sum(down_text in x for x in lines),
    }
    if len(lines) < 10:
        summary["raw"] = lines
    return summary


def summarize(args: argparse.Namespace) -> int:
    """Print the before, per-phase and after comparison of a samples file; with --json add a JSON line."""
    groups = _load(args.samples)
    execs, pings = groups.get("exec", []), _resolve_pings(groups.get("ping", []))
    keys = list(dict.fromkeys(k for r in execs for k in r if k not in ("kind", "ts")))
    stats: dict[str, Record] = {}
    for name, low, high in _windows(args.phases):
        stats[name] = _window_stats(
            [r for r in execs if low <= r["ts"] < high],
            [r for r in pings if low <= r["ts"] < high],
            keys,
        )
    reach = _reach_summary(
        [x for r in groups.get("reach", []) for x in r.get("lines", [])],
        args.up_text,
        args.down_text,
    )
    print(_render(stats, keys))
    print(f"reach: {reach['lines']} lines, up={reach['up']} down={reach['down']}")
    for raw in reach.get("raw", []):
        print(f"  {raw}")
    if args.json:
        print(json.dumps({"windows": stats, "reach": reach}))
    return 0


def main() -> int:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="sample the second host and ping it")
    rec.add_argument("--out", required=True, help="new file; an existing one is refused")
    rec.add_argument("--duration", type=float, required=True, help="seconds")
    rec.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    rec.add_argument("--ssh-host", default=SSH_HOST, help="ssh alias of the second host")
    rec.add_argument(
        "--iface", default="wlan0", help="network interface of the second host to sample"
    )
    rec.add_argument(
        "--ping-host", default="192.0.2.101", help="address of the second host to ping"
    )
    rec.add_argument(
        "--reach-unit",
        default=REACH_UNIT,
        help="user or system unit whose journal holds the health lines",
    )
    summ = sub.add_parser("summarize", help="compare before, each phase and after")
    summ.add_argument("samples")
    summ.add_argument("--phases", required=True, help="throughput.json from the benchmark script")
    summ.add_argument("--json", action="store_true", help="also print the figures as one JSON line")
    for sp in (rec, summ):
        sp.add_argument(
            "--up-text", default=UP_TEXT, help="text that marks an up line of the health unit"
        )
        sp.add_argument(
            "--down-text", default=DOWN_TEXT, help="text that marks a down line of the health unit"
        )
    args = parser.parse_args()
    return record(args) if args.cmd == "record" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
