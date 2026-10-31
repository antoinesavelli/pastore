"""Source level of a request (R21, R22): the peer address, or for a local peer the client process found through /proc.

Depends on: pastore.spec (Level, SourceRule, UNKNOWN_LEVEL) and stdlib only; reads /proc under an injected root.
"""

import fnmatch
import ipaddress
import os
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pastore.spec import UNKNOWN_LEVEL, Level, SourceRule

BUDGET_S = 0.020
Address = tuple[str, int]

_Ip = ipaddress.IPv4Address | ipaddress.IPv6Address
_Net = ipaddress.IPv4Network | ipaddress.IPv6Network
_TABLES = (("net/tcp", False), ("net/tcp6", True))
_RECENT_PIDS = 32
_FD_BATCH = 256
_MAX_DEPTH = 64
_DELETED = " (deleted)"
_SPACE = re.compile(r"\s")


@dataclass(frozen=True)
class Attribution:
    """Who sent a request: its level, a short space-free label for the journal, and the hearth run id, if any."""

    level: Level
    source: str
    run_id: str | None = None


@dataclass(frozen=True)
class _Facts:
    pid: int
    leaf: str | None
    comms: tuple[str, ...]
    paths: tuple[str, ...]


class _Unknown(Exception):
    """Attribution gave up for a named reason; `label` is the journal label."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def _normalize(address: Address) -> tuple[_Ip, int]:
    """Parse the address and unwrap an IPv4-mapped IPv6 one to plain IPv4."""
    ip = ipaddress.ip_address(address[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip, address[1]


def _kernel_addr(ip: _Ip, port: int, v6_table: bool) -> str:
    """The `ip:port` text /proc/net/tcp{,6} prints: each 32-bit word of the address in little-endian order."""
    packed = ip.packed
    if v6_table and ip.version == 4:
        packed = bytes(10) + b"\xff\xff" + packed
    words = "".join(packed[i : i + 4][::-1].hex() for i in range(0, len(packed), 4))
    return f"{words.upper()}:{port:04X}"


def _host_address(proc: Path, ip: _Ip) -> bool:
    """Whether the kernel lists `ip` as an address of this machine; an unreadable or odd file means no."""
    name = "net/fib_trie" if ip.version == 4 else "net/if_inet6"
    try:
        lines = (proc / name).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    if ip.version == 6:
        return any(line.split()[:1] == [ip.packed.hex()] for line in lines)
    leaf = ""
    for line in lines:
        words = line.split()
        if words[:1] == ["|--"] and len(words) == 2:
            leaf = words[1]
        elif words[:3] == ["/32", "host", "LOCAL"] and leaf == str(ip):
            return True
    return False


def _nets(values: Sequence[str]) -> tuple[_Net, ...]:
    return tuple(ipaddress.ip_network(v) for v in values)


def _in(ip: _Ip, nets: Sequence[_Net]) -> bool:
    return any(ip in n for n in nets)


def _glob(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _under(path: str, directory: str) -> bool:
    return PurePosixPath(path).is_relative_to(PurePosixPath(directory))


def _leaf(cgroup: str) -> str | None:
    """The last path component of the cgroup-v2 line (`0::/...`), None when absent or the root."""
    path = next((line[3:] for line in cgroup.splitlines() if line.startswith("0::")), "")
    return path.rstrip("/").rpartition("/")[2] or None


def _ppid(status: str) -> int:
    """The `PPid:` value of a status file, 0 when absent or invalid."""
    for line in status.splitlines():
        if line.startswith("PPid:"):
            value = line.partition(":")[2].strip()
            return int(value) if value.isdecimal() else 0
    return 0


def _ids(fields: list[str], want: list[str]) -> tuple[int, int] | None:
    """The (uid, inode) of a table row between the wanted addresses, None for any other or a malformed row."""
    if (
        fields[1:3] != want
        or len(fields) < 10
        or not (fields[7].isdecimal() and fields[9].isdecimal())
    ):
        return None
    return int(fields[7]), int(fields[9])


def _match(rule: SourceRule, facts: _Facts) -> str | None:
    """The journal label when the client's facts satisfy this rule, else None; a peer rule never matches here."""
    if rule.kind == "cgroup" and facts.leaf is not None:
        if _glob(facts.leaf, rule.values) and not _glob(facts.leaf, rule.exclude):
            return f"cgroup:{facts.leaf}"
    elif rule.kind == "ancestor":
        comm = next((c for c in facts.comms if c in rule.values), None)
        if comm is not None:
            return f"ancestor:{comm}:{facts.pid}"
    elif rule.kind == "under" and any(_under(p, d) for p in facts.paths for d in rule.values):
        return f"under:{facts.pid}"
    return None


def _run_id(level: Level, leaf: str | None) -> str | None:
    """The run id of a hearth client whose cgroup leaf is `pastore-<run_id>.scope`."""
    if level is Level.HEARTH and leaf and leaf.startswith("pastore-") and leaf.endswith(".scope"):
        return leaf.removeprefix("pastore-").removesuffix(".scope") or None
    return None


class Attributor:
    """Maps a connection to its source level by the policy's rules, first match wins, within a time budget.

    One instance serves concurrent calls from several threads.
    """

    def __init__(
        self,
        rules: Sequence[SourceRule],
        clock: Callable[[], float],
        *,
        proc: Path = Path("/proc"),
        uid: int | None = None,
        budget_s: float = BUDGET_S,
    ) -> None:
        self._rules = tuple(rules)
        self._peer_rules = tuple(
            (r.level, _nets(r.values), _nets(r.exclude)) for r in rules if r.kind == "peer"
        )
        self._clock = clock
        self._proc = proc
        self._uid = os.getuid() if uid is None else uid
        self._budget_s = budget_s
        self._recent: tuple[int, ...] = ()
        self._remember_lock = threading.Lock()

    def __call__(self, peer: Address, local: Address) -> Attribution:
        """Attribute the connection whose client end is `peer` and whose server end is `local`."""
        deadline = self._clock() + self._budget_s
        try:
            client, server = _normalize(peer), _normalize(local)
            if self._is_local(client[0], server[0]):
                return self._local(client, server, deadline)
            return self._remote(client[0])
        except _Unknown as exc:
            return Attribution(UNKNOWN_LEVEL, exc.label)
        except (OSError, ValueError) as exc:
            return Attribution(UNKNOWN_LEVEL, f"unknown:{type(exc).__name__}")

    def _is_local(self, client: _Ip, server: _Ip) -> bool:
        """A peer is local when it is loopback, the server's own address, or any address of this host (NAT)."""
        return client.is_loopback or client == server or _host_address(self._proc, client)

    def _remote(self, ip: _Ip) -> Attribution:
        label = f"peer:{ip}"
        for level, nets, exclude in self._peer_rules:
            if _in(ip, nets) and not _in(ip, exclude):
                return Attribution(level, label)
        return Attribution(UNKNOWN_LEVEL, label)

    def _local(
        self, client: tuple[_Ip, int], server: tuple[_Ip, int], deadline: float
    ) -> Attribution:
        uid, inode = self._find_row(client, server, deadline)
        if uid != self._uid:
            return Attribution(UNKNOWN_LEVEL, f"uid:{uid}")
        facts = self._facts(self._find_pid(inode, deadline), deadline)
        for rule in self._rules:
            label = _match(rule, facts)
            if label is not None:
                return Attribution(
                    rule.level, _SPACE.sub("_", label), _run_id(rule.level, facts.leaf)
                )
        return Attribution(UNKNOWN_LEVEL, f"local:{facts.pid}")

    def _check(self, deadline: float) -> None:
        if self._clock() > deadline:
            raise _Unknown("unknown:timeout")

    def _find_row(
        self, client: tuple[_Ip, int], server: tuple[_Ip, int], deadline: float
    ) -> tuple[int, int]:
        """The (uid, inode) of the client socket: the tcp row whose local end is `client` and remote end `server`."""
        for name, v6_table in _TABLES:
            if not v6_table and 6 in (client[0].version, server[0].version):
                continue
            want = [_kernel_addr(*client, v6_table), _kernel_addr(*server, v6_table)]
            with (self._proc / name).open(encoding="utf-8", errors="replace") as table:
                for line in table:
                    self._check(deadline)
                    ids = _ids(line.split(), want)
                    if ids is not None:
                        return ids
        raise _Unknown("unknown:no-socket")

    def _candidates(self, recent: tuple[int, ...]) -> Iterator[int]:
        """The recently matched pids first, then every other numeric pid newest first."""
        yield from recent
        with os.scandir(self._proc) as entries:
            pids = sorted((int(e.name) for e in entries if e.name.isdecimal()), reverse=True)
        yield from (pid for pid in pids if pid not in recent)

    def _remember(self, pid: int) -> None:
        with self._remember_lock:
            self._recent = (pid, *(p for p in self._recent if p != pid))[:_RECENT_PIDS]

    def _find_pid(self, inode: int, deadline: float) -> int:
        if inode == 0:
            raise _Unknown("unknown:no-pid")
        target = f"socket:[{inode}]"
        for pid in self._candidates(self._recent):
            self._check(deadline)
            if self._holds(pid, target, deadline):
                self._remember(pid)
                return pid
        raise _Unknown("unknown:no-pid")

    def _holds(self, pid: int, target: str, deadline: float) -> bool:
        """Whether our own process `pid` has an fd that is the socket `target`."""
        base = f"{self._proc}/{pid}"
        try:
            if os.stat(base).st_uid != self._uid:
                return False
            names = os.listdir(f"{base}/fd")
        except OSError:
            return False
        for i, name in enumerate(names):
            if i and i % _FD_BATCH == 0:
                self._check(deadline)
            try:
                if os.readlink(f"{base}/fd/{name}") == target:
                    return True
            except OSError:
                continue
        return False

    def _read(self, pid: int, name: str) -> str:
        return (self._proc / str(pid) / name).read_text(encoding="utf-8", errors="replace")

    def _link(self, pid: int, name: str) -> str | None:
        try:
            return os.readlink(self._proc / str(pid) / name).removesuffix(_DELETED)
        except OSError:
            return None

    def _comms(self, pid: int, deadline: float) -> tuple[str, ...]:
        """The `comm` of the pid and its ancestors within the budget, ending at pid 0 or 1, a vanished process or a repeat."""
        chain: list[str] = []
        seen: set[int] = set()
        while pid > 1 and pid not in seen and len(seen) < _MAX_DEPTH:
            seen.add(pid)
            self._check(deadline)
            try:
                chain.append(self._read(pid, "comm").rstrip("\n"))
                pid = _ppid(self._read(pid, "status"))
            except OSError:
                break
        return tuple(chain)

    def _facts(self, pid: int, deadline: float) -> _Facts:
        try:
            leaf = _leaf(self._read(pid, "cgroup"))
        except OSError as exc:
            raise _Unknown("unknown:pid-gone") from exc
        paths = tuple(p for p in (self._link(pid, "exe"), self._link(pid, "cwd")) if p is not None)
        return _Facts(pid, leaf, self._comms(pid, deadline), paths)
