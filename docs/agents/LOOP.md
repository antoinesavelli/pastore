# The loop here

The doctrine loop (AGENTS.md core block) with what is specific to pastore.

## OBSERVE

- Read `docs/STATUS.md` and the section you are touching in `docs/DESIGN.md` (the spec, what runs). Read the real host,
  not memory: `systemctl --user cat heavy-jobs.slice`, `cat /sys/module/nvidia/version`,
  `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`, `uv run pastore status`.
- `git status --short` and `git log --oneline -10` first; another session may be mid-edit.

## MODEL: what is ask-first here

The AGENTS.md "Ask first / never" list, plus: anything that would start a real job on a GPU from a session
(`run-now`, or `start` while the unit is stopped). `pastore.service` is the dispatcher, so that is the owner's call.

## ACT: delegation

- Each builder works in its own worktree on files no other builder touches (the module split is in `docs/DESIGN.md` §7);
  verifiers have edits forbidden. How many builders and verifiers a phase gets follows D-038.
- Builders never stage or commit; the delegator verifies and commits by explicit path.
- Mechanical units (README maps, `Depends on:` lines, fixture boilerplate) may go to a local model; say so openly.

## VERIFY: Definition of Done

Run each yourself; one that cannot run is reported as not run.

1. `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- timeout 900 uv run pytest -q` passes (every test or
   benchmark run is capped, `deploy/measure_ttfb.py` included: an uncapped 54 GB process was once OOM-killed and took
   other sessions' running commands with it); new behavior has a test that pins it and its failure mode; every
   invariant row in AGENTS.md names its guarding test.
2. `uv run ruff check .` reads `All checks passed!`; `uv run ruff format --check .` and `uv run mypy` are clean.
3. Scorecard: the owner's private defensive-coding report (`defensive_coding_report.py <repo> --area pastore`, kept in
   a private repository of the owner's). Floors held (it has no baseline; accept its floors).
4. Each module's docstring is one line + an accurate `Depends on:` line.
5. The shared doctrine check passes (`tools/check.sh agents` in the owner's system library): AGENTS.md carries the
   current doctrine core and `CLAUDE.md` is exactly `@AGENTS.md`.
6. Docs current: `README.md` map for added/moved files, `docs/STATUS.md` for state changes, a `docs/DECISIONS.md` row
   for a new decision or reversal.
7. Committed by explicit path (`git add <paths>`, `git diff --cached` read in full, then `git commit`). Never push
   without the owner's say-so.

## CLOSE

What changed, what was verified and how, what is next, and update `docs/STATUS.md` in the same commit.
