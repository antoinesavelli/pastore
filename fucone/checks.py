"""Pre-start host checks (driver, Ollama, foreign GPU processes, memory) mapped to closed-list idle reasons.

Depends on: fucone.spec (idle_reason); every host interaction goes through the injected `Host`.
"""

import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from fucone.spec import idle_reason

DRIVER_PACKAGE = "libnvidia-compute-580"
MODULE_VERSION_FILE = Path("/sys/module/nvidia/version")
MEMINFO = Path("/proc/meminfo")
MIN_AVAILABLE_BYTES = 15 * 1024**3
GPU_ALLOWLIST = frozenset(
    {"/usr/bin/cosmic-monitor", "/usr/local/lib/ollama/llama-server", "/usr/local/bin/ollama"}
)
OLLAMA_URL = "http://127.0.0.1:11434"
_APPS_QUERY = ("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader")


class CheckError(RuntimeError):
    """A host command or file gave output outside what the checks expect."""


@dataclass(frozen=True)
class Host:
    """The three host seams: run an argv, read a text file, get an HTTP status."""

    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
    read_text: Callable[[Path], str]
    http_status: Callable[[str, float], int]


def run_argv(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run an argv list with output captured and a 30 s timeout."""
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=30, check=False)


def _http_status(url: str, timeout_s: float) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return int(response.status)
    except urllib.error.HTTPError as err:
        return err.code


def real_host() -> Host:
    """The production Host: subprocess, Path.read_text and urllib."""
    return Host(run=run_argv, read_text=Path.read_text, http_status=_http_status)


def driver_ok(host: Host) -> bool:
    """True when the loaded module equals the installed package's upstream version and nvidia-smi works."""
    try:
        loaded = host.read_text(MODULE_VERSION_FILE).strip()
        pkg = host.run(["dpkg-query", "-W", "-f=${Version}", DRIVER_PACKAGE])
        if pkg.returncode != 0:
            return False
        installed = pkg.stdout.strip().split("-", 1)[0]
        if not loaded or not installed or loaded != installed:
            return False
        return host.run(["nvidia-smi"]).returncode == 0
    except OSError:
        return False


def ollama_ok(host: Host, url: str) -> bool:
    """True when Ollama answers /api/tags with 200."""
    try:
        return host.http_status(url + "/api/tags", 5.0) == 200
    except OSError:
        return False


def foreign_gpu_process(host: Host) -> str | None:
    """The first GPU compute process not on the allowlist, else None."""
    result = host.run(_APPS_QUERY)
    if result.returncode != 0:
        raise CheckError(f"nvidia-smi compute-apps query exited {result.returncode}")
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        _, sep, name = line.partition(",")
        if not sep:
            raise CheckError(f"unparseable compute-apps line: {line!r}")
        if name.strip() not in GPU_ALLOWLIST:
            return name.strip()
    return None


def mem_available_bytes(host: Host) -> int:
    """MemAvailable from /proc/meminfo, in bytes."""
    for line in host.read_text(MEMINFO).splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                raise CheckError(f"unparseable MemAvailable line: {line!r}")
            return int(fields[1]) * 1024
    raise CheckError("no MemAvailable line in meminfo")


def stopped(stop_file: Path) -> bool:
    """True when the STOP file exists."""
    return stop_file.exists()


def host_idle_reason(host: Host, ollama_url: str = OLLAMA_URL) -> str | None:
    """The first failing host check as a closed-list idle reason, or None when all pass."""
    if not driver_ok(host):
        return idle_reason("driver-mismatch")
    if not ollama_ok(host, ollama_url):
        return idle_reason("ollama-down")
    foreign = foreign_gpu_process(host)
    if foreign is not None:
        return idle_reason("gpu-foreign", foreign)
    if mem_available_bytes(host) < MIN_AVAILABLE_BYTES:
        return idle_reason("low-memory")
    return None
