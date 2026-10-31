# pastore

pastore manages two Ollama hosts from one machine, the workstation. Its gateway (`pastore-gateway.service`, port 11434)
sits in front of the workstation's Ollama and serves every request in order of who sent it: Reins > webui > Claude
coding > corsu tests and activity > routine scheduled > hearth. **Hearth** is the background job lane. Resumable,
model-heavy jobs that other repos declare in a `gpu-tasks.yaml` run 24 hours a day as task+model pairs, but only the
pairs the owner has listed in `policy.yaml` `jobs:` with evidence, each after a contract test against a fake Ollama (503,
refused connection, hang). Each run is placed on a host where the model fits, one job per loaded model within that
host's caps in `policy.yaml` `hosts:`. When nothing can start, pastore idles and logs one reason from a closed list. A
single-user tool, written by directing AI coding agents and accepted on its tests and checks.

It assumes Linux with systemd user services, Ollama, and an NVIDIA 580-series driver installed as the Debian/Ubuntu
package `libnvidia-compute-580` (the driver check reads it with `dpkg-query`). It is published as the owner uses it: no
packaging, no support. `policy.yaml` holds the real hosts, consumers and listed pairs, so it is private and gitignored;
`policy.example.yaml` is the placeholder the repo ships (copy it to `policy.yaml` and edit it).

The gateway listens on `0.0.0.0:11434` (`deploy/pastore-gateway.service`) with no authentication. It refuses ten routes
unless the peer is on this machine (the model-changing ones: pull, push, create, copy, delete; blobs; account sign-out
and keys; web fetch and search); every other route is served to any peer (inference queued and routed by R23 and R24,
the rest relayed to the workstation's Ollama), except its own `/pastore/status`, answered to loopback only. Which LAN
peers can reach it is decided by the host firewall, outside this repo and not read by it.

## Use

Run from the repository root; dependencies are pinned in `pyproject.toml` and locked in `uv.lock`.

```
git clone https://github.com/antoinesavelli/pastore
cd pastore
cp policy.example.yaml policy.yaml                 # then edit your hosts, consumers and pairs
uv sync
uv run pastore check <path-to-gpu-tasks.yaml>      # validate a consumer manifest
uv run pastore check --contract <consumer>/<task>  # contract test against the fake Ollama (dispatcher stopped, R16)
uv run pastore status                              # dispatcher, hosts, gateway, runs
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- timeout 900 uv run pytest -q
```

Adding a pair to `policy.yaml` is the owner's gate (`pastore add <consumer>/<task> --model <m> --evidence <path>`, which
refuses without a passing contract test). The spec is `docs/DESIGN.md`, the decisions are `docs/DECISIONS.md`, the status
is `docs/STATUS.md`, and agent instructions are `AGENTS.md`.

## Area map

| Path | What it is |
|---|---|
| `AGENTS.md` | Agent instructions: doctrine core, invariants, entry points, ask-first |
| `CLAUDE.md` | `@AGENTS.md` only |
| `docs/agents/` | Agent topic files: `LOOP.md` (the OBSERVE to CLOSE loop as this repo runs it, delegation rules, the full Definition of Done), `CONVENTIONS.md` (docstrings, CCN, typing, injected clock, fake Ollama, subprocess-only seams) |
| `docs/DESIGN.md` | The spec (what runs): rules R1 to R35, module split, acceptance checks |
| `docs/DECISIONS.md` | Decisions and parked work |
| `docs/STATUS.md` | Public status: phases, what is built, open items |
| `LICENSE` | MIT |
| `.githooks/` | `pre-commit` refuses credentials and `.githooks/refuse` paths (state databases, the dispatcher lock, the private `policy.yaml`); `pre-push` runs gitleaks over the whole history |
| `.github/workflows/ci.yml` | `uv sync --locked`, ruff, ruff format, mypy, pytest on every push to `main` and every pull request |
| `policy.example.yaml` | Placeholder policy and the human gate: hosts, the R21 source table, consumers, task+model pairs (`jobs:` holds the listed pairs), the optional `watch:` section of the foreign-workload watch. Copy it to `policy.yaml` (private, gitignored); the gateway unit reads that file too |
| `pyproject.toml` · `uv.lock` | Package, pinned runtime (PyYAML, aiohttp) and dev tools, ruff / mypy / pytest config · locked dependencies |
| `pastore/` | Package; one line per module below, the module split is in `docs/DESIGN.md` §7 |
| `pastore/cli.py` · `__main__.py` | Entry point: every subcommand (`status` marks a listed pair whose evidence file's sha256 changed as `evidence-changed`; `ack-workload` clears reviewed watch overlaps); the only place the real clock and host are bound |
| `pastore/dispatcher.py` | The loop (`serve`, which binds SIGTERM and the SIGALRM tick bound and restores both on exit) and one-tick decisions: listing gate, idle reasons, N runs per host, placement, attempts, crashes, readers, run-now (cleared only where acted on), the daily run-folder prune |
| `pastore/placement.py` · `hosts.py` | Where a run goes (R25 with D-041: need, free VRAM from the host's `vram_gib`, preference, `no-fit`) · per-host model facts and the executor probe (`executor-down`) |
| `pastore/policy.py` | `policy.yaml`: parse (no YAML aliases; a crafted file is a `PolicyError`, so the last good policy stays), last-good cache, add/remove rendering and diff |
| `pastore/manifest.py` | Consumer `gpu-tasks.yaml`: schema, path rule, per-task fingerprint → `TaskSpec` |
| `pastore/state.py` | SQLite WAL state, schema version, dispatcher lock, reset |
| `pastore/checks.py` | Workstation pre-start checks (driver, Ollama, foreign GPU process, memory) → closed-list idle reason; a global gate |
| `pastore/workload_watch.py` | The R35 foreign-workload watch: the journals of the units named in `policy.yaml` `watch:`, read through an injected reader, against executor run spans; records `workload-overlap` and alerts only; `off` without a `watch:` section |
| `pastore/tick_bound.py` | The 60 s bound on a dispatcher tick (D-050 (6)): the one-shot SIGALRM timer, `TickOverrun`, and the critical sections that defer an overrun |
| `pastore/run_end.py` | A running job (`Active`, `NoHandle`), its poll (`poll_run`) and how its run ended: `oom_killed`, `outage_reason` with `gateway_outage` (a run cut by an outage is `stopped`, free the first time and charged from a second in a row, D-052 (1)), and the attempt charges (`charge_failure`, `charge_crash`, `MAX_ATTEMPTS`) |
| `pastore/runner.py` · `contract.py` | Run dirs and `prune_run_dirs` (30-day retention; the dispatcher keeps each pair's latest and every run awaiting review, D-050 (7), D-052 (5)), env, `systemd-run --user` scope, stop, recovery · the three-scenario contract test |
| `pastore/spec.py` · `fake_ollama.py` | Shared value types and closed lists · the fake Ollama `check --contract` and the tests run against (model sizes, `/api/ps`, `/api/show`) |
| `pastore/gateway/` | The Ollama front door (R20-R24, R32): `attribution.py` (peer or `/proc` client to a source level), `queue.py` (per-host slots, level then FIFO, starvation), `server.py` (the aiohttp proxy with its 5 min upstream idle limit and the refusal of ten routes off this machine; `/pastore/status`, which reports `started_at`), `__main__.py` (`python -m pastore.gateway`; starts on the last good policy copy `~/.local/state/pastore/gateway-policy.yaml` when `policy.yaml` fails to load) |
| `deploy/` | Unit text to copy and edit, none installed by the repo: `pastore.service` (the dispatcher's user unit), `pastore-gateway.service` (the gateway), `pastore-executor-link.service` (the local forward to the second host, R26) · `measure_ttfb.py`, the gateway time-to-first-byte measurement · `measure_throughput.py` and `sample_executor.py`, the throughput and second-host health measurements |
| `tests/` | Tests and test doubles (`tests/README.md`) |
