# pastore: status

Public status. Edit in place; decisions are in `docs/DECISIONS.md`, the spec in `docs/DESIGN.md`. The owner keeps an
operations log (what is listed, what is live, dated runs) outside this repository.

**Where it stands.** v5 is built and runs on the owner's two machines: the dispatcher (`pastore.service`) and the Ollama
front door (`pastore-gateway.service`, `:11434`) are user services on the workstation, and a local forward
(`pastore-executor-link.service`) reaches the second Ollama host. It is a single-user tool. The real `policy.yaml` is
private and gitignored; `policy.example.yaml` is the placeholder the repo ships.

**Tests.** About 1,600 tests (1,626 at the last full run), all against fakes: no test touches a real GPU, a real Ollama
or real systemd. ruff, `ruff format --check` and `mypy --strict` are clean. v5 was built in units by AI coding agents,
each unit checked by a separate fresh reviewer against the tests and checks; the owner decides what is listed.

## Phases

| Phase | State |
|---|---|
| 1, 2 Rename to pastore (R31); shared types and state schema v2 | Done |
| 3a The gateway: priority by source, per-host slots, `/pastore/status` | Done |
| 3b The fleet dispatcher: placement, N runs per host, `no-fit`, `executor-down` | Done |
| 4a, 4b Second-host preparation and bring-up, the local forward (R26) | Done |
| 5a Interruption tests (R34) and the foreign-workload watch (R35) | Done |
| 5 Consumers, listing and activation (R29) | Done |
| 6, 6b Review and fix rounds with fresh reviewers; hardening from the owner's answers (D-050, D-052) | Done |
| 7 Audit | Pending by the owner's call (D-053) |
| 8 Soak and close | Remaining |

## Open items

- The live interruption drill (check 14) and the soak run in Phase 8.
- The foreign-workload watch windows and limits have no baseline and are the owner's to revisit (check 15).
- `dispatcher.py` has a weak maintainability index in the owner's scorecard.
- The job results of one consumer are written only at the end of a run, so its "resumable" is nominal.
- A few Ollama read routes (account and status information) stay open to LAN peers; the gateway does not filter them.
- Also open: check 3's top level (Reins) has not been exercised live; the workstation stays at 1 model and 1 slot
  through the soak (D-047, D-050 (2)); `pastore add` drops comments from `policy.yaml` (`render_policy` writes
  canonical text).
- Publication: the About text, README lead and topics are drafts the owner approves before the first push (D-053).
