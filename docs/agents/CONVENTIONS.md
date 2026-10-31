# Conventions

The shared code conventions (the full doctrine linked in `AGENTS.md`, sections *Code conventions* and *Testing*), plus
what pastore adds.

## Code

- Module docstring: one line + a `Depends on:` line. Function docstrings: one line, no `Args:`/`Returns:`.
- No comments unless the WHY is non-obvious. CCN < 15, aim ≤ 10. Type hints everywhere; `mypy --strict` clean.
- Fail loud at boundaries (manifests, policy, `/proc`, `nvidia-smi`, `dpkg-query`, HTTP); trust internal calls.
  No bare `except`, no silent fallback.
- **Injected clock.** Decision code (ready, starved, attempts, heartbeat age, placement) takes a `now` value or a clock
  callable; only `cli.py` / the dispatcher entry binds the real clock.
- **Subprocess and host seams are injected.** Every external call (`systemd-run`, `systemctl --user`, `nvidia-smi`,
  `dpkg-query`, reading `/proc/meminfo` or `/sys/module/nvidia/version`) goes through a small callable the tests
  replace. argv lists only, never `shell=True`. Never `sudo`.
- Never import another repo's code.

## Tests

- No test touches the real GPU, real Ollama, real systemd or the real `~/.local/state/pastore`: state and output dirs
  come from `tmp_path`; Ollama is `pastore/fake_ollama.py` (503, refused connection, hang).
- Each invariant in AGENTS.md has one test named for it; the AGENTS.md row cites it.
- A test that passes whether or not the code works is worse than none.
