# tests/ - Area Map

Tests for pastore. No test touches the real GPU, the real Ollama, real systemd or the real ~/.local/state/pastore: state and output directories come from pytest's tmp_path, Ollama is `pastore/fake_ollama.py` (the same fake `check --contract` uses), host commands go through tests/fakes.py's RecordingRunner, time through FakeClock.

| File | What it pins |
|---|---|
| `fakes.py` | `FakeClock` and `RecordingRunner` test doubles (a helper, not a test) |
| `test_fakes.py` | `pastore/fake_ollama.py` and `fakes.py` behave as documented |
| `test_policy.py` | `policy.yaml` parsing, the optional `watch:` section (validation, render round trip with backslashes), the tracked `policy.example.yaml`, last-good on parse error, `add`/`remove` diffs |
| `test_manifest.py` | `gpu-tasks.yaml` schema, the argv/cwd path rule, per-task fingerprints |
| `test_state.py` | SQLite schema, `user_version` mismatch refusal, rows of the retired watch names kept and ignored, the dispatcher lock, heartbeat, reset |
| `test_cli.py` | The CLI surface and its exit codes; `status` lines from state and from the gateway's `/pastore/status` |
| `test_checks.py` | Each pre-start check and its closed-list idle reason |
| `test_runner.py` | Run directories, the `systemd-run --user` scope argv, timeouts, stop, leftover-scope recovery |
| `test_contract.py` | The three contract scenarios against the fake Ollama |
| `test_dispatcher.py` | Ordering, N runs per host, placement, checks 8 and 9, run-now, stop, attempts, crash accounting, events |
| `test_dispatcher_gateway_outage.py` | D-050 (4): a non-ok end with the gateway unreachable, or restarted since the run began, is `stopped` with `ollama-down` and not charged; a 404, bad JSON or an old, naive or missing `started_at` is charged as before; a lost executor keeps priority; a clean end never reads the status page |
| `test_dispatcher_tick_bound.py` | D-050 (6): the 60 s SIGALRM tick bound over the real `serve`: an overrun is a `tick-error`, a start, an end or a stop in progress finishes first, a running job is never touched, the timer is disarmed before the nap, a pending run-now survives an overrun, the handlers are restored |
| `test_retention.py` | D-050 (7): run folders older than 30 days are deleted except each pair's latest; the run rows stay |
| `test_placement.py` | The R25 fit rule with D-041 (KV per slot, loaded size), preference order, check 8 |
| `test_hosts.py` | `/api/tags`, `/api/ps`, `/api/show` parsing, the executor probe, HTTP failure modes → idle reasons (check 9) |
| `interrupt_harness.py` | A real dispatcher process on a temp home with stubbed host and HTTP, for the R34 kill tests (a helper, not a test) |
| `test_interrupt_dispatcher_kill.py` | R34, check 14: a SIGKILLed dispatcher's run recovers uncharged, then charged on the second crash (D-012); an orphaned harness exits |
| `test_interrupt_reboot.py` | R34, check 14: a reboot with a running row and no scope; stale meta dropped; the crash streak across two losses and a `done` between them |
| `test_interrupt_executor_link.py` | R34, check 14: a run cut by a dropped executor link is `stopped`, uncharged and placed again (D-045) |
| `test_interrupt_oom.py` | R34, check 14: an OOM-killed job is charged and logged `oom` from the kernel log of its scope |
| `test_interrupt_gateway.py` | R34, check 14: a gateway killed mid-stream fails its callers at once; the restart has every slot free (run over the tracked `policy.example.yaml`) |
| `test_interrupt_state_write.py` | R34, check 14: a write cut mid-transaction leaves `PRAGMA integrity_check` ok; a hot journal is rolled back on open |
| `test_workload_watch.py` | R35, check 15: journal parsing, the configured signals, disruptions, overlap windows and watch statuses |
| `test_workload_wiring.py` | R35, check 15: the watch in the tick (dedupe, the `<4>` alert, never blocks work, `off` with no `watch:` section), `pastore status` lines, `ack-workload`, a state holding the retired watch rows |
| `test_invariants.py` | Source scans: never sudo, never ssh (`test_invariant_never_ssh`: no module under `pastore/` builds an ssh command; D-020, D-021, R26), no real clock outside `cli.py`; the closed idle-reason list |
| `test_gateway_invariants.py` | The sudo and clock scans over `pastore/gateway/`; only its `__main__.py` binds the real clock; `test_invariant_gateway_imports_no_dispatcher_write_path`: the gateway loads no dispatcher-only module (R20) |
| `test_gateway_attribution.py` | Source levels against a fake `/proc` (plus real-`/proc` cases), with the rules of the tracked `policy.example.yaml`: peer rules, own-host addresses, the 20 ms budget, the pid cache under threads |
| `test_gateway_queue.py` | Per-host slots, level then FIFO, hang-up removal, starvation reported once, forged or double release |
| `test_gateway_server.py` · `gateway_fakes.py` | Byte-equal pass-through, gating, routing by level and run host, journal lines, `/pastore/status`, hang-ups · an Ollama-like aiohttp fake and a scripted attribution (helpers, not tests) |
| `test_gateway_main.py` | The gateway entry point refuses a bad policy, falls back to the last good copy and serves with handler cancellation; `deploy/pastore-gateway.service` points `--policy` at `policy.yaml` under its `WorkingDirectory` (`%h/pastore`) and the tracked `policy.example.yaml` loads |
| `test_deploy_units.py` | `deploy/pastore.service` (the dispatcher's user unit): `ExecStart` runs the repo's `.venv` python with `WorkingDirectory` the repo, `Restart=on-failure`, `RestartSec`, the start limit, no `OnFailure=` line, `WantedBy=default.target`, no `User=`, no sudo |
| `test_deploy_scripts.py` | The arithmetic of the hand-run `deploy/` scripts with injected input, no network or subprocess: `measure_ttfb.py` (check 10: p95, ms to first byte, warm-up, gateway minus direct), `measure_throughput.py` (check 11: tok/s, repeats and extras, the both-hosts overlap, added %), `sample_executor.py` (check 12: sample parsing, derived rates, ping loss and percentiles, phase windows, the health-unit counts through the given up and down texts, and the `--ssh-host`, `--ping-host`, `--iface`, `--reach-unit`, `--up-text`, `--down-text` arguments with generic defaults) |
