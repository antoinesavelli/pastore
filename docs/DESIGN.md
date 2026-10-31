# pastore: Design

Status: **v5 is built and live since 2026-10-31.** This file is the single design. It merges the former v4 and v5
design documents and keeps their numbering: refinements R1 to R35, acceptance checks 1 to 15, kill lines 1 to 4 and the
D- references in `docs/DECISIONS.md` (D-053). R1 to R19 were approved on 2026-10-24 (D-016, D-017), R20 to R33 on
2026-10-25 (D-033), R34 and R35 came with D-042, and D-050 and D-052 (Phase 6b) amended R23, R25, R29, R31, R32 and
R34, and D-054 amended R6, R15 and R35. A new refinement needs its own decision row.

v4's §1 is section 1, its R1-R19 are in section 3, its module split is section 7; sections 2-6 keep v5's numbering.

Retired by number: no whole rule; R30 (D-024) removed the quiet-hours window and the idle reason `outside-window`, so
the window parts of R6, R8 and R12 and the local-time clause of R17 are gone.

pastore (formerly fucone, renamed by D-032) is the manager, scheduler and watchdog for two Ollama hosts. Roles used in
code and policy: the **workstation** is the machine pastore runs on; the **executor** is a second Ollama host. The
executor is reached only through a local SSH forward and its caps are set by the owner on that host.

## 1. Goals and base rules

### Goals

What the owner asked for, in short:

- One manager, scheduler and watchdog for the two hosts, so local models can be used from both. It extends the original
  job queue in place; there is no second repo (D-018).
- Background work, called **hearth**: resumable, model-heavy jobs that other repos declare and that run around the clock
  on whichever host fits, instead of being babysat by sessions.
- As many models loaded per host as fit, one job per loaded model, and a few requests at once per job.
- A front door on Ollama's port that orders every request by who is asking, so interactive use goes first and
  background work yields.
- The owner approves a job type together with a model, once. Only approved pairs run, and only after an evaluation
  (pruvera) has passed the model for that kind of work and pastore's own contract test passes.
- Every idle period has a logged reason; nothing happens that the owner did not list.
- A single-user tool, built in phases by AI coding agents (D-028) and accepted on its tests and checks.

### Base rules

These are the v4 rules that v5 kept, as v5 amended them. Detail is in R1 to R19 (section 3).

**Purpose.** Resumable, model-heavy jobs (for example a model benchmark campaign or an embedding-index rebuild) run
themselves on the hosts. Otherwise pastore idles, always with a logged reason.

**Non-goals.** Still parked (revival triggers in `docs/DECISIONS.md`): foreground detection and yield to the owner, a
model planner, a per-request "items" shape, and tiers with weights and aging. The v4 non-goals "no gate or proxy" and
"no use of the executor" were reversed by D-025 and D-018.

**Stack.** Python 3.12, pastore's own `.venv`, runtime dependencies PyYAML and aiohttp (both pinned; aiohttp is for the
streaming front door, D-027); stdlib otherwise (sqlite3, subprocess, json, argparse, urllib). pastore never imports
another repo's code; it calls other repos by subprocess only. Dev tools: pytest, ruff, mypy.

**State.** `~/.local/state/pastore/state.db`, SQLite WAL: jobs, runs, events, contracts, meta. One dispatcher, held by
`flock` on `dispatcher.lock` in the same directory. `status` opens the database read-only. The dispatcher writes a
heartbeat each tick, and `status` shows `dispatcher-down` after 60 s without one. The schema version is in `PRAGMA
user_version` (2); a mismatch refuses to start. `pastore reset` deletes the state files and never job outputs.

**Manifests.** Each consumer repo declares jobs in its own `gpu-tasks.yaml`: `id`, `kind`, `argv` (a list, never a
shell string), `cwd`, `env`, `model` or `models`, `timeout`, `ok_exit` (default `[0]`), `resumable: true` (required),
`ollama_env` and `model_env` (R1, R29), `reader`, and an optional `ready: {every, check: argv}`. A `reader` is either
`{run: argv}`, a machine reader whose non-zero exit is `reader-failed`, or `{validate: argv, review: owner}`: pastore
runs the validator, and then the run sits in `awaiting-review` until `pastore ack <run>`. A document path is not a
reader. With no `ready`, a task is one-shot: `done` after its first success, rerun only by `pastore rerun` or by a
change to that task's own manifest entry (the fingerprint is per task entry, not per file). pastore checks the schema
and the path rule (R5) and refuses a bad manifest; `pastore check <path>` runs those checks by hand.

**Outputs never land in a repo directly.** pastore gives every run a fresh directory and passes it as
`PASTORE_RUN_DIR`; the job writes there. Promoting results into a repo is a separate, owner-reviewed step (`pastore ack`
plus the consumer's own command), never automatic. Found: a benchmark overwrote a git-tracked results file in place.

**Contract test before listing.** A task is listed in `policy.yaml` only after `pastore check --contract <task>` passes.
pastore runs the job against a fake Ollama that answers 503, refuses connections, and hangs past the timeout. In each
scenario the job must exit with a code not in `ok_exit` (or be killed at the timeout) and leave no result in
`PASTORE_RUN_DIR`. Found: a benchmark recorded an Ollama error as a failed row and exited 0, and an index builder treated
any HTTP error as "halve and retry".

**Policy.** `policy.yaml`, hand-edited, is the single human gate. A pair runs only once listed; list order is priority.
`pastore add` checks the entry and shows the diff before writing. A parse error keeps the last good policy and logs it,
and a policy edit never kills a running job. A ready pair waiting over 24 h is flagged `starved`.

**Window.** Retired: there is no quiet-hours window. Both hosts take work 24 hours a day (D-024, R30). `pastore run-now`
means "start this pending pair next".

**Execution.** Each run is a transient user scope inside the existing user `heavy-jobs.slice` (a slice the host defines,
with its own memory limits), not a separate cap that would overcommit RAM. Before a start, MemAvailable must be at least
15 GiB. Detail in R14.

**Pre-start checks.** On the workstation, a global gate, each failing check an idle reason: the driver check
(`/sys/module/nvidia/version` equals the upstream version of the installed `libnvidia-compute-580` package and
`nvidia-smi` exits 0); Ollama reachable at `127.0.0.1:11434`, the job endpoint, which is the gateway; the STOP file
`~/.local/state/pastore/STOP` absent; no GPU compute process outside the allowlist (R11); the MemAvailable rule. They
also block executor runs, because job processes run on the workstation (D-045 (3)). The closed idle-reason list is
`no-ready-task`, `no-fit`, `driver-mismatch`, `ollama-down`, `executor-down`, `gpu-foreign:<proc>`, `low-memory`,
`stopped`. Per-task events are `refused`, `reader-failed`, `awaiting-review`, `starved`, `done`, `failed`.
`dispatcher-down` is status-only.

**Attempts.** An exit not in `ok_exit`, or a timeout, is counted; three counted failures make the pair `failed`
(`MAX_ATTEMPTS`). A dispatcher crash is not charged to a job unless the same job was running at two consecutive crashes
(D-012). A run cut short by an outage is free the first time and charged from a second in a row (R34, D-052 (1)).
Startup recovery stops leftover `pastore-*.scope` units by name and returns `running` rows to pending.

**CLI.** `pastore start | stop [--now] | status | add | remove | check [--contract] | run-now | rerun | ack |
ack-workload | reset`. `add`, `remove`, `rerun` and `run-now` take a task `<consumer>/<id>` and `--model <name>`.

**Doctrine package.** The invariants table in `AGENTS.md`, each row pinned by a named test: only policy-listed pairs run;
a job never writes into a repo directly; a contract test is required before listing; never sudo; every idle period has a
closed-list reason; a driver mismatch refuses dispatch; scheduling decisions take an injected clock (no `time.time()` in
decision code); the gateway imports no dispatcher write path; no module under `pastore/` builds an ssh command; a tick overrun never touches a running job (D-050 (6)); and the
gateway-route and interruption rules of R23 and R34. Code conventions: a module docstring with a `Depends on:` line,
one-line docstrings, cyclomatic complexity under 15 (aim for 10 or less), type hints, ruff and mypy clean. Tests use a
fake Ollama HTTP server; no test touches the real GPU or a real Ollama. A defensive-coding scorecard is run and its
floors are accepted (it has no baseline mechanism).

**First consumers.** Two repos were listed first, each only after its own contract test passed (D-015, D-049): a model
benchmark (`torra/model-bench`) and an embedding-index rebuild for a notes vault (`memory/index`).

**Parked.** See `docs/DECISIONS.md` (Parked) for the items and their triggers. Lessons carried: socket timing
(`ss -ti`) cannot see non-streamed requests or prefill; consumers swallow errors; releasing a job request while the
owner is active evicts the owner's prefix cache; Ollama `/v1/*` has no `num_ctx` field; a status command whose exit
code means active or inactive reads a crash as inactive, so read sentinel files directly and fail closed; an
unprivileged `ss -p` cannot attribute sshd sockets; `/api/ps` `expires_at` advances per real use and not on
`/api/tags` (verified on Ollama 0.33.1 only).

## 2. What was decided (owner-chosen)

Each item below is a decision in `docs/DECISIONS.md` (D-018 to D-032), chosen by the owner.

- **Role (D-018).** pastore is the manager, scheduler and watchdog for the workstation and the executor, extended in
  place. **Hearth** is the background job lane inside pastore (the v4 queue), the lowest priority level.
- **Build gate (D-019).** Design first; building started after v4 had run clean (the gate was passed by D-037).
- **Executor (D-020, D-021).** Usable at any time, within caps the owner sets on that host. Nothing stops a job
  automatically if it disturbs the other workload on that host; R35 records and alerts (D-042). The executor runs only
  its `ollama` service: no job code, files, venvs or run directories there. pastore never touches another workload's
  units or files on it, never uses sudo there, never pulls a model, and starts no work when the host's state cannot be
  read.
- **Listing (D-022).** The owner approves a **job type plus a model**, once. pastore refuses the pair unless an
  evaluation (pruvera) has passed that model for that kind of work and pastore's contract test passes. pruvera never
  starts anything.
- **Capacity (D-023).** Each host loads as many models as fit in VRAM, one job per loaded model, within per-host caps
  in `policy.yaml`. Each job may send 2 to 3 requests at once. The workstation part was applied and rolled back on
  2026-10-31 (D-046, D-047; R28): the workstation runs one slot and one loaded model (`max_models: 1, parallel: 1`),
  and how it reaches several loaded models is open (D-047).
- **Hours (D-024).** Both hosts take work 24 hours a day.
- **Front door (D-025, D-026).** pastore replaced the earlier two-level priority proxy on port 11434 and orders every
  Ollama request by source: **Reins > webui > Claude coding > corsu tests and activity > routine scheduled > hearth.**
  Reins is a phone chat app on the LAN; webui is a web chat UI in a local Docker container. A higher level jumps the
  queue; a running request is never cancelled.
- **Stack (D-027).** One added runtime dependency, `aiohttp`, pinned, for the streaming front door.
- **Execution (D-028).** One plan in phases. Each phase is planned by one agent session, built by several with disjoint
  files, then checked by fresh sessions that get the code and the phase's checks, never the builders' reports.

## 3. Refinements

### Base rules, R1 to R19 (owner-OK 2026-10-24)

| # | Refinement | Why |
|---|---|---|
| R1 | **Endpoint injection.** Manifest field `ollama_env` (required): the env var the job reads its Ollama base URL from. pastore sets it to the gateway, `http://127.0.0.1:11434`, and to the fake's URL under `--contract`. | The contract test can only reach a job that lets pastore pick its endpoint. Found: one consumer hard-coded its host and another read a differently named variable. |
| R2 | **Layout.** State: `~/.local/state/pastore/` (`state.db`, `dispatcher.lock`, `STOP`). Outputs: `~/.local/share/pastore/runs/<run_id>/` with `out/` = `PASTORE_RUN_DIR` and `stdout.log` / `stderr.log` beside it. `run_id` = `<UTC yyyymmddTHHMMSS>-<n>`, the first free `n` from 1. `reset` deletes `state.db*` only, refuses while the lock is held, and keeps `STOP`. | Outputs outside the state directory make "reset never touches outputs" structural; logs outside `out/` keep "no result" checkable; the timestamp keeps run directories unique across resets. |
| R3 | **"No result" is mechanical.** After a contract scenario, `out/` is empty or contains a file named `FAILED`. | The contract test's "or mark it failed" needs one checkable form. |
| R4 | **Contract scenarios are bounded.** Each of the three runs with timeout `min(task timeout, 300 s)`; being killed at that timeout counts as a non-`ok_exit` exit. | A hang scenario at a multi-hour task timeout makes `check --contract` unusable; live runs are killed at the timeout the same way. |
| R5 | **Path rule.** Consumer root = the directory holding `gpu-tasks.yaml` (a consumer need not be a git repo). `argv[0]` (and every `ready.check` and `reader` argv[0]): a bare name must `which`-resolve and be `torra` or `opencode`; a path must be under the root after `os.path.abspath` (not `realpath`, so a repo `.venv/bin/python` symlink counts as inside). `cwd` must be under the root after `realpath`. | A venv interpreter is a symlink to the system Python; `realpath` would refuse every Python consumer. The remaining gap (a hand-made in-repo symlink) is covered by the policy and contract gates. |
| R6 | **Policy shape.** `version: 2`, `hosts: {<name>: {url, vram_gib, max_models, parallel, context}}` (a host named `workstation` is required), `sources: [<rule>, ...]` (R21), `consumers: {<name>: <path to gpu-tasks.yaml>}`, `jobs: [{task, model, evidence, sha256}, ...]` (R29) and an optional `watch:` section (R35). Task refs are `<consumer>/<id>` everywhere in the CLI; a pair is a task plus `--model`. | One unambiguous name per task across repos, and one unit the owner approves. |
| R7 | **Contract pass is per fingerprint, enforced twice.** A pass is recorded in `state.db` against the task-entry fingerprint (sha256 of its canonical JSON). `add` refuses without one, and the dispatcher emits `refused` for a listed task without one. A manifest edit or `reset` therefore requires a fresh `check --contract`. | `policy.yaml` is hand-edited, so a check only in `add` is bypassable. |
| R8 | **Ready semantics.** `ready.check` runs at most once per `every` (for example `24h`, `6h`); exit 0 means ready (until it runs), 1 means not ready, and any other exit is a `refused` event `ready-check-exit:<n>`. Checks must not use the GPU. `starved` fires once per ready episode. | A crashing check must not read as "not ready" forever. |
| R9 | **Reader outcomes block, not retry.** `reader-failed` and `awaiting-review` hold the task until `rerun` or `ack`; neither counts as an attempt. | The job already ran; rerunning it blindly does not fix bad output. |
| R10 | **Stop vs STOP.** `pastore stop`: the dispatcher exits once the running jobs end, starting nothing new. `--now`: it also stops the running scopes; those runs return to pending, uncharged. The `STOP` file pauses: the dispatcher stays up, idles `stopped`, and running jobs continue. | Three distinct owner intents: quit, quit now, pause. |
| R11 | **GPU allowlist.** Exact process paths, held in `GPU_ALLOWLIST` in `checks.py`: Ollama's runner (`llama-server`), `ollama`, and the desktop's system monitor. Any other GPU compute process is the idle reason `gpu-foreign:<proc>`. | Ollama 0.33.1 runs models in `llama-server`, not `ollama`. Consequence (accepted with the v4 non-goals): the owner's foreground Ollama use is invisible to the dispatcher; the gateway orders it instead. |
| R12 | **Loop.** One tick every 30 s (`TICK_S`) with the injected clock: heartbeat, re-read the policy (last good on error; at startup an unparseable policy refuses to start), handle a stop request, poll the running jobs, place the ready ones, run the R35 watch, and prune old run folders (R31). Recovery runs once at start. With no run active, the idle reason is the first of `stopped`, `no-ready-task`, the global gate (`driver-mismatch`, `ollama-down`, `gpu-foreign:<proc>`, `low-memory`), the first unreadable host's reason in policy order (`ollama-down` or `executor-down`), and `no-fit` (D-045). It is logged when it changes, not every tick. `dispatcher-down` = lock free or heartbeat older than 60 s. | Bounded `events` table; one reason per idle period. |
| R13 | **Driver check detail.** Upstream version = `dpkg-query -W -f='${Version}'` for `libnvidia-compute-580`, up to the first `-`. The package name is a constant; a driver-series change is a code edit. | No speculative config. |
| R14 | **Execution detail.** The runner spawns `systemd-run --user --scope --quiet --slice=heavy-jobs.slice --unit=pastore-<run_id> -p CPUWeight=20 -p RuntimeMaxSec=<timeout> -- <argv>` as a child (systemd ends the scope at the task timeout even if pastore has died), with `cwd` and an explicit env: `PATH`, `HOME`, `LANG`, `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS`, the manifest `env`, `PASTORE_RUN_DIR`, the `ollama_env` variable (the gateway URL) and the `model_env` variable (R29). Nothing else from the dispatcher's environment is passed (a scope runs its command as systemd-run's own child, so it would otherwise inherit the caller's env). A timeout or `stop --now` calls `systemctl --user stop pastore-<run_id>.scope`. The MemAvailable floor is `15 * 1024**3` bytes from `/proc/meminfo` (`MIN_AVAILABLE_BYTES`). | A scope the host's slice already limits; the explicit env keeps a job from inheriting the dispatcher's. |
| R15 | **Schema.** `jobs` (keyed by task and model, R30), `runs` (with `host` and `model`), `events`, `contracts(ref, fingerprint, passed_at)` (R7) and `meta` (`heartbeat_at`, `idle_reason`, `stop_requested`, `run_now`, `workload_watch`, `workload_acked_id`); `PRAGMA user_version` = 2. | The v5 schema; v4 state is not migrated (R30). |
| R16 | **CLI to dispatcher requests go through `meta`, polled each tick**, with no signals. `run-now <ref> --model <m>` needs the pair listed, a contract pass for its current fingerprint, the job `pending` (a done, failed or blocked job needs `rerun` first) and a live dispatcher (else refused, exit 1). It skips readiness, never `STOP` or the host checks, and is placed before the other ready pairs; a run-now that is blocked or fits nowhere is recorded as `refused` and dropped, not lost. Requests left over from a previous dispatcher are dropped at start. `check --contract` holds the dispatcher lock, so it never runs beside a dispatcher job. A dispatcher that finds its lock held at start retries once after 1 s (a `status` probe can hold a shared lock for an instant). The one signal handled is SIGTERM from systemd: it sets a flag, the tick sleep (1 s steps) wakes within 1 s, and the tick logs `dispatcher-stop` `sigterm`, then acts as `stop --now` (D-035). | Owner intent reaches the single writer without a second process touching the GPU. |
| R17 | **Readers and ready checks** run with the job's base-env allowlist plus the manifest `env` (plus `PASTORE_RUN_DIR` for readers), `cwd` = the task `cwd`, and a 45 s timeout (exit 124; a missing binary is exit 127), short enough that a tick never outlives the 60 s `dispatcher-down` rule. (The local-time clause for the window is retired, R30.) | Bounded helper calls. |
| R18 | **A tick never ends the dispatcher.** Host commands never raise (timeout is 124, missing binary is 127); unexpected `nvidia-smi` output fails closed as `driver-mismatch`, unreadable meminfo as `low-memory`; a job that cannot start is a counted failure; anything else is caught per tick, recorded as a `tick-error` event and shown by `status`. Recovery also returns job rows stuck `running` with no running run to `pending`. | One bad entry or host hiccup must not end the dispatcher. |
| R19 | **Contract and fingerprint are harder to fool.** In the 503 and hang scenarios the job must actually reach the fake (proves it honours `ollama_env`, R1); the fingerprint hashes consumer name + resolved root + entry, so repointing a consumer to another checkout needs a fresh contract; `ack` of a run from an older fingerprint leaves the job's rerun pending. | Found in verification: a job dying at import passed the contract, and a moved consumer reused an old pass. |

Known limit: "a job never writes into a repo directly" is enforced for what pastore controls (`PASTORE_RUN_DIR` outside
every repo, no automatic promotion). A job whose `cwd` is its repo can still write there; only its consumer's own fix
and the owner's review stop that.

### Fleet rules, R20 to R35 (owner-OK 2026-10-25, D-033; R34 and R35 by D-042)

Facts behind them: the workstation's Ollama listens on `127.0.0.1:11435`; the gateway listens on `0.0.0.0:11434`, which
the workstation's tools use; Ollama has no authentication, and which LAN peers can reach the gateway's port rests on the
host's network configuration, outside this repo. The executor's Ollama is reached only through the link of R26.

| # | Refinement | Why |
|---|---|---|
| R20 | **Two processes, two user units.** `pastore-gateway.service` (the front door, `Restart=always`, `RestartSec=1`) and `pastore.service` (the dispatcher). The gateway reads `state.db` read-only and never imports dispatcher code paths that write. | A dispatcher bug or restart never takes the owner's tools down; check 6 is about the gateway alone. |
| R21 | **Source levels**, first match wins: (6) hearth = the client's cgroup is a `pastore-<run_id>.scope`; (2) webui = peer in `172.17.0.0/16`; (1) Reins = any other non-loopback peer except the executor; (3) coding = loopback and the client's process ancestry includes `claude` or `opencode`; (4) corsu = loopback and the client's exe or cwd is under a directory named in the policy (the owner's projects); (5) routine = loopback and the client runs in a systemd unit other than a session scope; unknown loopback = (5). The mapping table lives in `policy.yaml` `sources:`; code holds only the mechanism. | Hearth is matched first and exactly, so a corsu job pastore started is hearth, not corsu. Unknown never outranks coding and never sinks below hearth. "Any other LAN device is high" keeps the earlier proxy's owner rule. |
| R22 | **Attribution mechanism.** Loopback peer port to socket inode via `/proc/net/tcp{,6}`, inode to pid via `/proc/<pid>/fd`, then ancestry (`/proc/<pid>/status` PPid, `comm`, `exe`, `cwd`) and `/proc/<pid>/cgroup`. The gateway runs as the owner's user, so another user's process is unattributable and reads as unknown (5). Lookup is bounded at 20 ms; over budget reads as unknown. | Kernel socket ownership is exact; only the program-to-level mapping is a rule. A slow lookup must not add latency (check 10). |
| R23 | **Slots and queue.** Per host, slots = `parallel` x `max_models`, from `policy.yaml` `hosts:`. One priority queue per host: level first, FIFO within a level. A client hang-up removes its queue entry. Inference routes (`/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings`, `/v1/*`) are gated; read routes (`/api/tags`, `/api/ps`, `/api/show`, `/api/version`) pass ungated to the workstation. Model-management routes (`/api/pull`, `/api/push`, `/api/create`, `/api/copy`, `/api/delete`) pass to the workstation only and are never sent to the executor. Those five and five more (D-052 (4)): `/api/blobs` and anything under it, `/api/signout`, `/api/user/keys` and anything under it, `/api/experimental/web_fetch` and `/api/experimental/web_search`, are refused with 403 (`this route is refused from off this machine`), any method, unless the peer is on this machine (D-050 (9c)): loopback, the connection's own local address, any address the kernel lists for this host (`/proc/net/fib_trie`, `/proc/net/if_inet6`), or a local container in `172.17.0.0/16`. A request that is not a gated inference request is refused when either its path with repeated slashes collapsed, or that path with `.` and `..` resolved, matches, so it refuses more forms of the ten paths, never fewer. | Same gated set as the proxy it replaced. pastore never pulls models itself, and a LAN peer cannot change or delete a model, touch blobs or the Ollama account, or use its web fetch and search through these ten routes. |
| R24 | **Routing.** Levels 1 to 5 go to the workstation. Hearth requests go to the host the dispatcher assigned to that run (read from `state.db` by the scope's run id). | The owner's requests stay on the workstation and nothing interactive lands on the executor; the executor serves listed, scheduled work only. |
| R25 | **Placement.** The dispatcher assigns a run to a host where its model fits: need = `size` from `/api/tags` x 1.2 + 1 GiB + `parallel` x KV per slot (2 x layers x KV heads x head dim x context x 2 bytes, from `/api/show` `model_info` at the host's context length, the policy host field `context`; amended by D-041 and D-045). Once a model has loaded on a host, its `size` from `/api/ps` (GPU plus CPU part, so a partly offloaded model reads as not fitting; D-045) replaces the estimate for that host. A model whose KV fields are missing is placed only on an otherwise empty card. free = host `vram_gib` minus `size_vram` of resident models from `/api/ps` (models held by a pastore run count as resident); `vram_gib` is the usable VRAM the host's Ollama reports, in GiB, compared as `int(vram_gib * 2**30)` bytes, with no reserve field (D-050 (1)). Preference: the host where the model is already loaded, then the executor, then the workstation. A model that fits on neither waits with idle reason `no-fit`. A host is a candidate only if the model is in its `/api/tags` (pastore never pulls) and it has fewer pastore runs than `max_models` (D-045). Model names compare in Ollama's canonical form, `:latest` implied (D-049). | Leaves the workstation's GPU to the owner when the executor can do the job. The 1.2 x + 1 GiB rule is a starting guess that check 8 pins and check 11 refines. |
| R26 | **Executor link.** An SSH local forward from the workstation, `127.0.0.1:11436` to the second host's own `127.0.0.1:11434`, using the SSH alias `second-host`, as the user unit `pastore-executor-link.service` on the workstation (`deploy/pastore-executor-link.service` is the unit text). No network change on the second host. | The executor's Ollama never has to listen on the LAN (Ollama has no authentication). Needs no sudo on the workstation. Setup first verifies that the second host's Ollama listens on loopback only. |
| R27 | **Executor caps** (set by the owner on that host, outside this repo; this file gives no values): its Ollama service runs under CPU and memory limits, a parallel-requests setting and a loaded-models setting, and the `hosts:` entry for the host (`max_models`, `parallel`) matches them. Host health for the dispatcher is one check: `GET /api/version` through the link within 2 s, else idle reason `executor-down` and the executor takes no new run. No module under `pastore/` builds an ssh command; besides the link unit, only the hand-run check 12 sampler (`deploy/sample_executor.py`) uses ssh, read-only. | Static caps as chosen (D-020, D-041). "State unreadable means no new work" applies to the only state pastore reads there. |
| R28 | **Workstation Ollama settings.** The design asked the workstation's Ollama for several parallel slots and loaded models (D-023). The trial on 2026-10-31 was rolled back (D-046, D-047): at several slots a 32B-parameter model spilled off the card, so the workstation stays at one slot and one loaded model, and how it reaches several loaded models is open. | D-023 needs both; the setting also changes how Ollama behaves for the owner's own sessions (recorded in Known limits). |
| R29 | **Listing shape.** `policy.yaml` `jobs:` entries are `{task: <consumer>/<id>, model: <name>, evidence: <path>, sha256: <hex>}`. The model must be in the manifest's `models`; pastore passes it in the env var named by the required manifest field `model_env`. `pastore add` requires `--evidence <path>` to an existing file, records its sha256, and refuses without a current contract pass. A listed pair whose evidence file no longer matches the recorded sha256 (or cannot be read) shows `evidence-changed` on its line in `pastore status` and keeps running; `pastore remove` then `pastore add` records the new sha256 and clears it (D-050 (8)). `policy.yaml` may not use YAML aliases (refused as `invalid YAML: aliases are not allowed`, which closes a "billion laughs" file; anchors without aliases parse), and any crafted file (bad tags, huge ints, deep nesting) is a `PolicyError`. The dispatcher keeps its last good policy (one `policy-error` event per change of the file; the file is marked seen before it is parsed, so a failure that escapes the parse, a tick overrun included, costs one tick, not every tick) and a gateway starting on it falls back to its copy (R32). A multi-model sweep becomes one run per model. | The pair is the unit the owner approves and the unit pastore places, so a sweep spreads across both hosts. pruvera has no machine-readable "model X passes job Y" verdict yet (dependency noted in D-033); the evidence file is the stand-in. |
| R30 | **24 hours.** The `policy.yaml` `window` and the idle reason `outside-window` are removed; `run-now` stays as "start this pending pair next". Schema `user_version` 2: `runs` gains `host` and `model`; `jobs` is keyed by task plus model. v4 state is not migrated: v5 starts a fresh `state.db` and re-runs the contracts. | The window has no meaning with priority in front; a fresh state avoids a migration nobody needs. |
| R31 | **Rename.** Package and CLI `pastore`, state `~/.local/state/pastore/`, runs `~/.local/share/pastore/runs/`, scope prefix `pastore-`, units as in R20 and R26. pastore sets `PASTORE_RUN_DIR`; the old name `FUCONE_RUN_DIR` is no longer set, and a manifest that names it in `env` is refused (D-045). Run folders older than 30 days (by the UTC stamp in the run id) are deleted (D-050 (7)), except each pair's latest run; contract-test folders and folders with no run row go by age alone, symlinks are never followed, and the run rows in `state.db` stay. The dispatcher prunes at most once a day by its clock, first on the first tick after a start that reaches the end of the tick, and records a `runs-pruned` event (count, first and last folder name) when it deleted any. A folder without the `yyyymmddTHHMMSS-` stamp is never deleted. The folder of a run in `awaiting-review` is never pruned; once acked it falls under the 30-day rule (D-052 (5)). | Pruning keeps the runs folder from growing without bound; each pair's latest run and every run awaiting review stay whatever their age. |
| R32 | **Gateway logging, starvation, idle limit and policy copy.** One journal line per inference request: level, source, model, host, wait ms, run ms, status. A request queued over 10 min logs `starved` once. A refused request (R23) logs `refused route=<path> peer=<ip>`. A relayed reply that gets no bytes from Ollama for 5 min (`UPSTREAM_IDLE_S` = 300 s) is ended (D-050 (5); kept for every reply, D-052 (6)): 504 before the headers are sent, the client connection closed mid-body after, status `upstream-idle`; a connect failure stays 502 `upstream-error`. So a `stream: false` generation (or a pull or create stage) that sends nothing for 5 min is cut too. `pastore status` reads the gateway's `GET /pastore/status`, answered to loopback only, which also reports `started_at`, the gateway's start in UTC (read by the dispatcher for R34). The gateway keeps a copy of the last policy it started on at `~/.local/state/pastore/gateway-policy.yaml` (the exact bytes it parsed, written atomically, the directory never created); when `policy.yaml` fails to load at start it starts on that copy and logs a one-line journal warning (`<4>`), and with no copy or a bad copy it exits 2. | The evidence for checks 3, 4 and kill line 2; no second database. An Ollama that goes silent holds a caller for 5 min at most, and once a copy exists a bad edit to `policy.yaml` does not keep the front door down across a restart. |
| R33 | **Cutover.** An owner-run script (it needs sudo, which no agent uses) stops and disables the earlier system-level priority proxy; `pastore-gateway.service` then binds `0.0.0.0:11434`. Ollama stays on `127.0.0.1:11435`. The rollback script (re-enable the old proxy) is rehearsed once before cutover. | Check 6. Port 11434 is above 1024, so a user unit can hold it. |
| R34 | **Interruption and resume** (D-042). Every interruption ends in a known state and the work is picked up without a human. Dispatcher killed (SIGKILL, OOM, power loss, reboot): on start, recovery stops leftover `pastore-*` scopes, returns `running` rows to `pending` with host and model kept, and charges the job only under D-012's two-consecutive-crashes rule. A run whose host went `executor-down` during the run is released as an outage end (below) and re-placed. A job killed by the kernel OOM killer is charged (it is the job's fault) and logged `oom`. Gateway killed: in-flight callers get an error at once, never a hang; a restarted gateway starts with every slot free. A write cut mid-transaction leaves `state.db` passing `PRAGMA integrity_check`. Each case has a test that kills a real child process (never a real unit) and checks the state after restart. A run that ends not ok (a timeout included) and that a gateway outage cut short is `stopped` with detail `ollama-down`, as a lost executor's run is with `executor-down`, and its pair returns to pending, or to `failed` when a charged outage end brings it to 3 attempts (D-050 (4); D-052 (1) amending D-045 and D-012). The first such outage end charges nothing; a second consecutive outage end of the same pair charges an attempt, as D-012 does for crashes. The count is the job's crash streak: a normal end (ok, failed or timed out) resets it, a recovered crash counts in it, and a dispatcher start resets the streak of every job not running then, so a restart between two outage ends makes the second free again. At a non-ok end the dispatcher reads the gateway's loopback-only `GET /pastore/status` on the job endpoint; the run counts as cut by the gateway when that read fails (refused, no answer within 2 s, or a cut-short reply) or the page's `started_at` is later than the run's start. A gateway that started no later than the run, or an answer that is no gateway status page (an HTTP error, bad JSON, a missing or naive `started_at`), leaves the run charged. A gateway restart in the poll gap after a run's own failure can give that run one uncharged retry (accepted, D-052 (2)). For a non-ok end that did not time out the OOM check comes first (D-052 (3)), the kernel log read once: an OOM-killed run is charged and logged `oom` even when an outage was seen too. Without an OOM kill, and always after a timeout, the executor check comes next and keeps priority over the gateway read. | A host-wide out-of-memory event on 2026-10-30 killed every session's running command at once; pastore must come back from the same event with no lost or double-charged work. |
| R35 | **Foreign-workload watch** (D-042, D-054). Record and alert only; it never stops anything (D-020 stands). The optional `watch:` section of `policy.yaml` names the signals; without it the status is `off` and the journal is not read. Each dispatcher tick reads, on the workstation only (no SSH to the executor), `journalctl -u <unit>` for each distinct unit the signals name, in first-seen order. A signal has a `name`, a systemd `unit`, and either `good` and `bad` regexes (searched in each journal message; `bad` is tested first) or a `value` regex with one group and a `limit` (good when the captured digits, leading zeros stripped, are at most 9 long and at most `limit`), and an optional `first_bad: true`. A section has 1 to 8 signals with unique names, and a `status:` naming one of its good/bad signals, whose latest sample sets the status. A disruption is a good-to-bad transition of one signal; a first bad sample with no earlier sample opens a disruption only for a `first_bad` signal. A disruption that starts while a run is on the executor, or within 10 min after one ends, records a `workload-overlap` event with both time spans, logs a `<4>pastore: workload-overlap ...` journal warning, and shows in `pastore status` until `pastore ack-workload`, which records `workload-ack`. Runs are kept for the watch 30 min after they end, and the journal is read from the earlier of 30 min ago and 15 min before the earliest kept run. The status comes from the latest sample of the `status` signal in the last 30 min: `armed` (good), `down` (bad; the watch stays armed and still finds overlaps), `no-signal` (no sample), `unreadable: <reason>` (the journal read failed) or `off`; it is kept in meta `workload_watch` and printed as `workload watch: <status>`. A new overlap is one event, deduped by run, signal and disruption start. | The watch needs no baseline: it arms itself and catches the first overlap once the watched workload is up. It feeds kill line 1. The windows are fixed in code and have no measured baseline. |

**Known limits.** Priority orders requests, not VRAM: a hearth job that loads a model on the workstation can push the
owner's loaded model out, and the owner's next request waits for the reload (check 2 measures it). The workstation runs
one slot and one loaded model (`policy.yaml` `hosts:` `workstation`: `max_models: 1, parallel: 1`; the trial in R28 was
rolled back on 2026-10-31, D-046, D-047), so the owner's request does not share the card with a second one. If D-047's
open question raises the slots again, requests would share the card and each would run slower (stated for the owner's
review). A tool pointed at 11434 fails while the gateway is down; `Restart=always` bounds that to seconds, and only a
tool with its own fallback rule avoids it. The owner accepted the priority limit as it stands (D-050 (9)).

Three more limits, as the code does them (the owner accepted the first and the third as they stand and narrowed the
second, D-050 (9)):

- **Attribution trusts the owner's own uid** (R21, R22; `pastore/gateway/attribution.py`, `policy.yaml` `sources:`). A
  process of that uid reaches coding (3) when its own `comm` or an ancestor's is `claude` or `opencode`, and corsu (4)
  when its exe or cwd is under a directory the policy names. A hearth job is read as hearth only while its cgroup leaf
  matches `pastore-*.scope`; a process it starts outside that scope (for example with `systemd-run --user`) is read by
  the later rules, as coding (3), corsu (4) or else routine (5), all above hearth (6).
- **The gateway is open to whoever reaches its port, except for ten routes** (`deploy/pastore-gateway.service`,
  `pastore/gateway/server.py`). It binds `0.0.0.0:11434` with no authentication. It refuses `/api/pull`, `/api/delete`,
  `/api/create`, `/api/copy`, `/api/push`, `/api/blobs` and anything under it, `/api/signout`, `/api/user/keys` and
  anything under it, `/api/experimental/web_fetch` and `/api/experimental/web_search` with 403 unless the peer is on
  this machine (R23; D-050 (9c), D-052 (4)); every other route it neither gates nor answers itself (`/pastore/...`) is
  served to any peer. So the inference and read routes are open to LAN peers, and so are the Ollama routes outside those
  ten: in Ollama 0.33.1 `/api/me` (it shows the signed-in ollama.com account), `/api/status` and an experimental
  model-recommendations route. A LAN peer outside `172.17.0.0/16` other than the executor is read as Reins (1). A gated
  `/v1/` POST whose path resolves to a refused route is relayed as sent, not refused. Ollama does not run the refused
  route for it: gin does not resolve dot segments, and if Go's ServeMux in front redirects to the cleaned path, the
  client's follow-up comes back through the gateway and is refused. Which LAN peers can reach the port rests on the
  host's network configuration, outside this repo and not read by it.
- **A slow dispatcher tick delays SIGTERM and can read as `dispatcher-down`** (`pastore/dispatcher.py`,
  `pastore/tick_bound.py`, `pastore/checks.py`, `pastore/hosts.py`, `pastore/runner.py`, `pastore/cli.py`). Each tick is
  followed by a 30 s wait (`TICK_S`) that sleeps in 1 s steps, so a SIGTERM is handled within about 1 s only during that
  wait (D-035); the handler only sets a flag, so one that arrives mid-tick waits for the tick to end, and a shutdown on
  2026-10-26 SIGKILLed the user manager 5 s after its SIGTERM (D-035). A tick is bounded at 60 s in total
  (`TICK_BOUND_S`, D-050 (6)): a one-shot SIGALRM timer around each tick raises `TickOverrun`, which is recorded as a
  `tick-error` event, and the next tick starts after the usual wait; running jobs are not touched. An overrun that lands
  inside a job start, the handling of one ended run (stop on timeout, the outage and OOM reads, reader, state writes), a
  stop request or a run-now refusal waits for that step to finish, so the bound is exceeded by at most that step. A
  pending `run-now` request survives an overrun: it is cleared only where it is acted on. Most of a tick's calls have
  their own timeout: a host command (`dpkg-query`, `nvidia-smi`, `journalctl`) 30 s (`run_argv`), a ready check or
  reader 45 s (`SUBPROCESS_TIMEOUT_S`), the Ollama liveness check 5 s, the executor probe 2 s, the gateway status read
  at a non-ok end 2 s, and each `/api/tags`, `/api/ps` and `/api/show` read 5 s (socket timeouts). A job stop, which a
  run past its timeout (`poll_run` in `pastore/run_end.py`, called from the dispatcher's `_watch_all`) or a SIGTERM or
  `stop --now` (`_stop_all`) triggers in the tick, takes up to 30 s for `systemctl --user stop` plus 30 s for the
  process, then waits with no timeout after the kill (`RunHandle.stop`). The
  heartbeat is written at the start of a tick and is read as stale 60 s later (`HEARTBEAT_STALE`) until the next tick
  writes it, so for a tick that runs past about 30 s, `pastore status` can print `dispatcher-down` and `pastore run-now`
  refuse from 60 s after the tick starts until the next tick, while the dispatcher is alive and holds its lock.

## 4. Acceptance checks (owner-approved)

| # | Check | How it is shown |
|---|---|---|
| 1 | Answers pass through unchanged (streaming, non-streaming, `/v1`) | Test: byte-equal to a direct call against the fake Ollama |
| 2 | A request waits at most for the one in-flight request, never a whole job | Test with the fake; live: journal `wait ms` for level 1 to 3 while hearth runs |
| 3 | A higher level starts next | Test; live: start a long hearth job, send a Reins chat, journal order |
| 4 | Starvation is visible | Test: a request queued over the limit logs `starved` |
| 5 | Ollama down gives a fast error | Test: refused or 503 upstream returns an error within 5 s |
| 6 | Gateway down restarts in about 10 s; rollback rehearsed once | `systemctl --user kill pastore-gateway`, time to `active`; the rollback script run once before cutover |
| 7 | A hung-up caller frees its place | Test |
| 8 | Big models never go to the executor | Test: placement refuses a model whose need exceeds the executor's usable VRAM (`vram_gib`, D-050 (1)) |
| 9 | Executor offline: work stays or waits, with a logged reason | Test: link down gives `executor-down`, no failed run |
| 10 | Gateway adds under 5 ms before the first token | Measured: time to first byte through 11434 vs 11435, 50 requests, p95 (`deploy/measure_ttfb.py`) |
| 11 | Throughput measured properly | Same model and prompts, 3 repeats: workstation alone, executor alone, both; results are in the owner's operations log (`deploy/measure_throughput.py`) |
| 12 | Health of the workload that shares the executor, recorded during check 11 | Record only: the signals R35 reads, plus host counters and ping loss and latency, sampled read-only while the throughput run loads the executor (`deploy/sample_executor.py`); results are in the owner's operations log |
| 13 | 72 hours clean | No `tick-error`, no unexpected restart, every listed job reaches `done` |
| 14 | Interruption resumes cleanly (D-042, R34) | One test per R34 case; live: one drill, `systemctl --user kill -s KILL pastore.service` during a run, then the run resumes uncharged |
| 15 | A workload overlap is caught (D-042, R35) | Test with recorded journal lines: an overlap records `workload-overlap` and a warning; no overlap records nothing; a bad status signal reads `down`; no `watch:` section reads `off` |

## 5. Out of scope (owner-approved)

1. Auto-listing from pruvera. 2. Anything on the executor besides Ollama. 3. A tripwire that stops work on a
disturbance (check 12 only records; R35 only alerts). 4. Cancelling a request in progress. 5. More than two machines,
cloud or paid models. 6. Changing other repos' code from pastore (consumers change in their own repos). 7. A web
dashboard. 8. Publishing the repository, which is the owner's separate step (D-053).

## 6. Kill lines (owner-approved)

Any one stops the build or rolls back to the old proxy:

1. **The watched workload is disturbed.** Check 12 shows the health of the workload that shares the executor worse under
   executor load: executor work stops at once.
2. **The front door costs more than it gives.** After two weeks the owner's requests are not noticeably faster: roll
   back to the two-level proxy.
3. **The executor is not worth it.** Check 11 shows under about 25% added throughput: drop the executor half, keep the
   workstation front door.
4. **It keeps breaking.** More than 2 gateway outages that took the owner's tools down in the first month: back to
   direct Ollama while it is fixed.

## 7. Module map

One row per module under `pastore/`; the unit texts and measurement scripts are under `deploy/`, the tests under
`tests/` (see `README.md` for the full map).

| Module | Owns |
|---|---|
| `spec.py` | Shared value types (`TaskSpec`, `Host`, `Pair`, readers, `Ready`, source levels and rules) and the closed idle-reason and event lists |
| `fake_ollama.py` | The fake Ollama that `check --contract` and the tests run against |
| `policy.py` | Parse and check `policy.yaml` (including the `watch:` section), last good copy, `add` and `remove` diff and rendering (R6, R29) |
| `manifest.py` | Parse and check `gpu-tasks.yaml`, the path rule (R5), per-task fingerprints |
| `state.py` | SQLite WAL schema, `user_version`, the dispatcher lock, heartbeat, read-only open, reset (R15) |
| `cli.py`, `__main__.py` | The argparse surface and `pastore status`; the only place the real clock and host are bound for the dispatcher side (the gateway binds its own in `gateway/__main__.py`) |
| `checks.py` | Pre-start host checks mapped to idle reasons (R11 to R13) |
| `hosts.py` | Per-host model facts from `/api/tags`, `/api/ps`, `/api/show`, and the executor probe (R25, R27) |
| `placement.py` | Which host a run goes to: need, free VRAM, preference, `no-fit` (R25) |
| `runner.py` | Run directories and pruning, the `systemd-run` scope, job env, stop, leftover-scope recovery (R2, R14, R31) |
| `contract.py` | The three-scenario contract harness over the fake Ollama (R1, R3, R4, R19) |
| `run_end.py` | A running job and how its run ended: OOM kill, outage reads, attempt charges (R34) |
| `tick_bound.py` | The 60 s bound on a dispatcher tick (D-050 (6)) |
| `workload_watch.py` | The R35 foreign-workload watch: journal reader, samples, disruptions, overlaps, statuses |
| `dispatcher.py` | The loop: ordering, placement, attempts, crash accounting, events, `serve()` (R12, R16, R18) |
| `gateway/attribution.py` | Peer address or `/proc` client to a source level (R21, R22) |
| `gateway/queue.py` | Per-host slots, level then FIFO, starvation (R23, R32) |
| `gateway/server.py`, `gateway/__main__.py` | The aiohttp proxy, route refusal, `/pastore/status`, policy copy, `python -m pastore.gateway` (R20, R23, R24, R32) |
