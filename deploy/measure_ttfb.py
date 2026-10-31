"""Check 10: time to first byte of a streaming generate, gateway vs direct, interleaved, p50 and p95 in ms.

Depends on: stdlib only (http.client, json, statistics, time). Usage, from the repo root:
`uv run python deploy/measure_ttfb.py --direct http://127.0.0.1:11435 --gateway http://127.0.0.1:<port> --model <m>`
"""

import argparse
import http.client
import json
import statistics
import time
from urllib.parse import urlsplit


def ttfb_ms(base: str, body: bytes) -> float:
    """One POST /api/generate; milliseconds from sending it to the first body byte, then drain the stream."""
    parts = urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port, timeout=120)
    start = time.perf_counter()
    conn.request("POST", "/api/generate", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    first = response.read(1)
    elapsed = (time.perf_counter() - start) * 1000
    response.read()
    conn.close()
    if response.status != 200 or not first:
        raise SystemExit(f"{base}: status {response.status}")
    return elapsed


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def main() -> int:
    """Warm up both paths once, then alternate direct and gateway requests and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("-n", type=int, default=50)
    args = parser.parse_args()
    body = json.dumps(
        {"model": args.model, "prompt": "Say hi.", "stream": True, "options": {"num_predict": 8}}
    ).encode()
    for base in (args.direct, args.gateway):
        ttfb_ms(base, body)
    runs: dict[str, list[float]] = {"direct": [], "gateway": []}
    for _ in range(args.n):
        runs["direct"].append(ttfb_ms(args.direct, body))
        runs["gateway"].append(ttfb_ms(args.gateway, body))
    for name, values in runs.items():
        print(
            f"{name:8} n={len(values)} p50={statistics.median(values):.2f}ms p95={_p95(values):.2f}ms"
        )
    print(
        f"added p95={_p95(runs['gateway']) - _p95(runs['direct']):.2f}ms (check 10 passes under 5 ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
