"""Three-scenario contract harness: a job must fail loud, leaving no result, when Ollama is 503, refused or hung.

Depends on: fucone.spec, fucone.runner, fucone.checks (run_argv), fucone.fake_ollama; records nothing in state.
"""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from fucone.checks import run_argv
from fucone.fake_ollama import FakeOllama, Mode, refused_url
from fucone.runner import Runner, fresh_run_dir, has_result, start_run
from fucone.spec import TaskSpec


class Started(Protocol):
    """What the harness needs from a started run (RunHandle satisfies it)."""

    def poll(self) -> int | None: ...
    def stop(self, run: Runner, wait_s: float = 30.0) -> int: ...


SCENARIOS = ("503", "refused", "hang")
CONTRACT_TIMEOUT_CAP_S = 300
STOP_WAIT_S = 2.0
_FAKE_MODES: dict[str, Mode] = {"503": "503", "hang": "hang"}


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario's outcome; `passed` means the job reached the fake, failed loud and left no result."""

    name: str
    run_id: str
    exit_code: int | None
    timed_out: bool
    result_left: bool
    contacted: bool
    passed: bool


@dataclass(frozen=True)
class ContractResult:
    """The three scenarios for one task fingerprint."""

    ref: str
    fingerprint: str
    scenarios: tuple[ScenarioResult, ...]

    @property
    def passed(self) -> bool:
        """True when every scenario passed."""
        return all(s.passed for s in self.scenarios)


def _scenario(
    name: str,
    spec: TaskSpec,
    runs_dir: Path,
    base_env: Mapping[str, str],
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    start: Callable[..., Started],
    run: Runner,
    fake: Callable[[Mode], FakeOllama],
    refused: Callable[[], str],
    poll_s: float,
) -> ScenarioResult:
    run_id, run_dir = fresh_run_dir(runs_dir, clock(), f"-contract-{name}")
    limit_s = min(spec.timeout_s, CONTRACT_TIMEOUT_CAP_S)
    with ExitStack() as stack:
        server = None if name == "refused" else stack.enter_context(fake(_FAKE_MODES[name]))
        url = refused() if server is None else server.url
        handle = start(spec, run_dir, run_id, url, base_env)
        began = clock()
        exit_code = handle.poll()
        timed_out = False
        while exit_code is None:
            if (clock() - began).total_seconds() >= limit_s:
                handle.stop(run, STOP_WAIT_S)
                timed_out = True
                break
            sleep(poll_s)
            exit_code = handle.poll()
        contacted = server is None or bool(server.requests)
    result_left = has_result(run_dir.out)
    failed_loud = timed_out or exit_code not in spec.ok_exit
    passed = failed_loud and not result_left and contacted
    return ScenarioResult(name, run_id, exit_code, timed_out, result_left, contacted, passed)


def run_contract(
    spec: TaskSpec,
    runs_dir: Path,
    base_env: Mapping[str, str],
    *,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    start: Callable[..., Started] = start_run,
    run: Runner = run_argv,
    fake: Callable[[Mode], FakeOllama] = FakeOllama,
    refused: Callable[[], str] = refused_url,
    poll_s: float = 0.5,
) -> ContractResult:
    """Run the three scenarios against the task and report each; records nothing."""
    results = tuple(
        _scenario(name, spec, runs_dir, base_env, clock, sleep, start, run, fake, refused, poll_s)
        for name in SCENARIOS
    )
    return ContractResult(spec.ref, spec.fingerprint, results)
