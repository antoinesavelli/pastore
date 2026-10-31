"""Gateway attribution (R21, R22): peer rules, the /proc lookup in a fake /proc, the budget, the pid cache, real /proc.

Depends on: pastore.gateway.attribution, pastore.policy (the real policy.yaml rules), pastore.spec.
"""

import ipaddress
import os
import socket
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pastore.gateway import attribution
from pastore.gateway.attribution import Address, Attribution, Attributor
from pastore.policy import load_policy
from pastore.spec import UNKNOWN_LEVEL, Level, SourceRule

UID = os.getuid()
REPO_POLICY = Path(__file__).resolve().parent.parent / "policy.yaml"
RULES = load_policy(REPO_POLICY).sources
LOCAL: Address = ("127.0.0.1", 11434)
SESSION = "0::/user.slice/user-1000.slice/session-3.scope"
HEARTH = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/pastore-r123.scope"
SERVICE = "0::/system.slice/foo.service"
# The kernel's real /proc/net/fib_trie and /proc/net/if_inet6 layout (from this workstation), with its public
# and VPN addresses replaced by documentation (2001:db8::/32) and made-up 100.64.x.y values.
FIB_TRIE = """\
Id 52:
  +-- 100.64.0.0/10 2 0 2
     |-- 100.64.56.78
        /32 universe UNICAST
     +-- 100.96.0.0/15 2 0 2
        |-- 100.96.100.100
           /32 universe UNICAST
        |-- 100.97.1.2
           /32 universe UNICAST
Main:
  +-- 0.0.0.0/0 2 0 2
     +-- 0.0.0.0/4 2 0 2
        |-- 0.0.0.0
           /0 universe UNICAST
        |-- 192.0.2.0
           /24 link UNICAST
     |-- 172.17.0.0
        /16 link UNICAST
Local:
  +-- 0.0.0.0/0 3 1 5
     +-- 192.0.2.128/25 2 0 2
        |-- 192.0.2.167
           /32 host LOCAL
        |-- 192.0.2.255
           /32 link BROADCAST
     +-- 96.0.0.0/3 2 0 2
        |-- 100.64.12.34
           /32 host LOCAL
        +-- 127.0.0.0/8 2 0 2
           +-- 127.0.0.0/31 1 0 0
              |-- 127.0.0.0
                 /8 host LOCAL
              |-- 127.0.0.1
                 /32 host LOCAL
           |-- 127.255.255.255
              /32 link BROADCAST
     +-- 172.17.0.0/16 2 0 2
        |-- 172.17.0.1
           /32 host LOCAL
        |-- 172.17.255.255
           /32 link BROADCAST
"""
IF_INET6 = """\
fe800000000000000200aafffe000001 05 40 20 80 veth1234567
fe800000000000000200aafffe000002 04 40 20 80  docker0
fe800000000000000001000200030004 02 40 20 80   enp7s0
20010db8000000015555666677778888 02 40 00 00   enp7s0
20010db8000000010000000000003d51 02 80 00 00   enp7s0
fe800000000000000005000600070008 03 40 20 80 tun0
20010db8000000011111222233334444 02 40 00 01   enp7s0
00000000000000000000000000000001 01 80 10 80       lo
fd000000000000000000000012345678 03 80 00 80 tun0
"""


def _word(ip: str) -> str:
    """The /proc/net/tcp{,6} text of an address, built independently of the module: native-endian 32-bit words."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    raw = socket.inet_pton(family, ip)
    return "".join(f"{w:08X}" for w in struct.unpack(f"={len(raw) // 4}I", raw))


def _addr(ip: str, port: int) -> str:
    return f"{_word(ip)}:{port:04X}"


class FakeProc:
    """A /proc tree under tmp_path: tcp tables and per-pid fd, comm, status, cgroup, exe and cwd files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "net").mkdir(parents=True)
        self.rows: dict[str, list[str]] = {"tcp": [], "tcp6": []}
        self._write_tables()

    def _write_tables(self) -> None:
        for name, rows in self.rows.items():
            header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"
            (self.root / "net" / name).write_text("\n".join([header, *rows]) + "\n")

    def row(
        self, local: str, remote: str, inode: int, *, uid: int = UID, table: str = "tcp"
    ) -> None:
        """Add a socket row; `local` and `remote` are already kernel-formatted `HEX:PORT`."""
        n = len(self.rows[table])
        line = f"{n:4d}: {local} {remote} 01 00000000:00000000 00:00000000 00000000 {uid:5d}        0 {inode} 1 0000"
        self.rows[table].append(line)
        self._write_tables()

    def addresses(self) -> None:
        """Write the kernel's lists of this machine's own addresses."""
        (self.root / "net" / "fib_trie").write_text(FIB_TRIE)
        (self.root / "net" / "if_inet6").write_text(IF_INET6)

    def raw_row(self, line: str, table: str = "tcp") -> None:
        self.rows[table].append(line)
        self._write_tables()

    def client(
        self,
        port: int,
        inode: int,
        *,
        ip: str = "127.0.0.1",
        server: str | None = None,
        uid: int = UID,
    ) -> Address:
        """Add the client-side row of a connection from `ip:port` to the gateway at `server` (default `ip`:11434)."""
        table = "tcp6" if ":" in ip else "tcp"
        self.row(_addr(ip, port), _addr(server or ip, 11434), inode, uid=uid, table=table)
        return (ip, port)

    def process(
        self,
        pid: int,
        *,
        comm: str | bytes = "python3",
        ppid: int = 1,
        cgroup: str = SESSION,
        inodes: tuple[int, ...] = (),
        exe: str | None = "/usr/bin/python3",
        cwd: str | None = "/home/user",
    ) -> Path:
        d = self.root / str(pid)
        (d / "fd").mkdir(parents=True)
        for i, inode in enumerate(inodes):
            (d / "fd" / str(i + 3)).symlink_to(f"socket:[{inode}]")
        (d / "comm").write_bytes(comm if isinstance(comm, bytes) else f"{comm}\n".encode())
        (d / "status").write_text(f"Name:\t{comm!s}\nPid:\t{pid}\nPPid:\t{ppid}\nUid:\t{UID}\n")
        (d / "cgroup").write_text(cgroup + "\n")
        if exe is not None:
            (d / "exe").symlink_to(exe)
        if cwd is not None:
            (d / "cwd").symlink_to(cwd)
        return d


class CountingClock:
    """A clock that reads 0.0 for its first `ok_calls` calls and 1000.0 after, counting every call."""

    def __init__(self, ok_calls: int = 10**9) -> None:
        self.ok_calls = ok_calls
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return 0.0 if self.calls <= self.ok_calls else 1000.0


@pytest.fixture
def proc(tmp_path: Path) -> FakeProc:
    return FakeProc(tmp_path / "proc")


def _attributor(
    proc: FakeProc,
    rules: tuple[SourceRule, ...] = RULES,
    *,
    clock: Callable[[], float] = lambda: 0.0,
    budget_s: float = attribution.BUDGET_S,
) -> Attributor:
    return Attributor(rules, clock, proc=proc.root, uid=UID, budget_s=budget_s)


def _remote(tmp_path: Path, rules: tuple[SourceRule, ...] = RULES) -> Attributor:
    """An attributor whose /proc does not exist: any /proc read would surface as a different label."""
    return Attributor(rules, lambda: 0.0, proc=tmp_path / "missing", uid=UID)


def _loopback(proc: FakeProc, **process: Any) -> Attribution:
    """Attribute a loopback client served by pid 300 with the given process facts."""
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), **process)
    return _attributor(proc)(peer, LOCAL)


@pytest.fixture
def spy_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every os.readlink path, in order."""
    seen: list[str] = []
    real = os.readlink

    def spy(path: str | os.PathLike[str], *a: object, **k: object) -> str:
        seen.append(str(path))
        return real(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "readlink", spy)
    return seen


def _fd_pids(reads: list[str]) -> list[int]:
    """The pids whose fd entries were read, in first-read order."""
    pids: list[int] = []
    for path in reads:
        parts = Path(path).parts
        if "fd" in parts and int(parts[-3]) not in pids:
            pids.append(int(parts[-3]))
    return pids


# --- the kernel's text format -------------------------------------------------------------------


def test_address_encoding_matches_the_kernel_text() -> None:
    """Pinned against literal strings the kernel prints (little-endian words), so the helper cannot mirror a bug."""

    def kernel(ip: str, port: int, v6_table: bool) -> str:
        return attribution._kernel_addr(attribution._normalize((ip, port))[0], port, v6_table)

    assert kernel("127.0.0.1", 11434, False) == "0100007F:2CAA"
    assert kernel("127.0.0.53", 53, False) == "3500007F:0035"
    assert kernel("172.17.0.2", 1, False) == "020011AC:0001"
    assert kernel("::1", 80, True) == "00000000000000000000000001000000:0050"
    assert kernel("127.0.0.1", 80, True) == "0000000000000000FFFF00000100007F:0050"
    assert kernel("fe80::1", 80, True) == "000080FE000000000000000001000000:0050"
    assert _addr("192.0.2.5", 80) == "050200C0:0050"
    assert _addr("::ffff:127.0.0.1", 80) == "0000000000000000FFFF00000100007F:0050"


# --- every level by its rule ---------------------------------------------------------------------


def test_hearth_by_pastore_scope_carries_the_run_id(proc: FakeProc) -> None:
    got = _loopback(proc, cgroup=HEARTH)
    assert got == Attribution(Level.HEARTH, "cgroup:pastore-r123.scope", "r123")


def test_webui_by_docker_bridge_peer_needs_no_proc(tmp_path: Path) -> None:
    got = _remote(tmp_path)(("172.17.0.2", 5555), ("172.17.0.1", 11434))
    assert got == Attribution(Level.WEBUI, "peer:172.17.0.2")


def test_reins_by_lan_peer(tmp_path: Path) -> None:
    got = _remote(tmp_path)(("192.0.2.74", 5555), ("192.0.2.5", 11434))
    assert got == Attribution(Level.REINS, "peer:192.0.2.74")


def test_executor_peer_is_excluded_from_reins(tmp_path: Path) -> None:
    attribute = _remote(tmp_path)
    expected = Attribution(UNKNOWN_LEVEL, "peer:192.0.2.101")
    assert attribute(("192.0.2.101", 5555), ("192.0.2.5", 11434)) == expected
    assert attribute(("::ffff:192.0.2.101", 5555), ("192.0.2.5", 11434)) == expected


def test_ipv4_mapped_peer_is_matched_as_ipv4(tmp_path: Path) -> None:
    got = _remote(tmp_path)(("::ffff:192.0.2.74", 1), ("::ffff:192.0.2.5", 11434))
    assert got == Attribution(Level.REINS, "peer:192.0.2.74")


def test_ipv6_peer_matches_only_ipv6_networks(tmp_path: Path) -> None:
    rules = (
        SourceRule(Level.WEBUI, "peer", ("172.17.0.0/16",)),
        SourceRule(Level.REINS, "peer", ("::/0",)),
    )
    assert _remote(tmp_path, rules)(("2001:db8::7", 1), ("192.0.2.5", 11434)).level is Level.REINS
    got = _remote(tmp_path, rules[:1])(("2001:db8::7", 1), ("192.0.2.5", 11434))
    assert got == Attribution(UNKNOWN_LEVEL, "peer:2001:db8::7")


def test_coding_through_claude_two_ancestors_up(proc: FakeProc) -> None:
    proc.process(250, comm="bash", ppid=200)
    proc.process(200, comm="claude", ppid=1)
    got = _loopback(proc, comm="python3", ppid=250)
    assert got == Attribution(Level.CODING, "ancestor:claude:300")


def test_corsu_by_cwd_under_the_corsu_tree(proc: FakeProc) -> None:
    got = _loopback(proc, cwd="/srv/projects/torra")
    assert got == Attribution(Level.CORSU, "under:300")


def test_corsu_by_exe_under_the_corsu_tree(proc: FakeProc) -> None:
    got = _loopback(proc, exe="/srv/projects/torra/.venv/bin/python")
    assert got == Attribution(Level.CORSU, "under:300")


def test_corsu_sibling_directory_is_not_corsu(proc: FakeProc) -> None:
    got = _loopback(proc, cwd="/srv/projects2/x", exe="/srv/projects2/bin/y")
    assert got == Attribution(UNKNOWN_LEVEL, "local:300")


def test_a_deleted_suffix_is_stripped_from_exe_and_cwd(proc: FakeProc) -> None:
    got = _loopback(proc, cwd="/srv/projects (deleted)")
    assert got.level is Level.CORSU


def test_routine_by_service_unit(proc: FakeProc) -> None:
    got = _loopback(proc, cgroup=SERVICE)
    assert got == Attribution(Level.ROUTINE, "cgroup:foo.service")


def test_session_scope_is_excluded_from_routine_and_nothing_else_matching_is_unknown(
    proc: FakeProc,
) -> None:
    got = _loopback(proc, cgroup=SESSION)
    assert got == Attribution(UNKNOWN_LEVEL, "local:300")


def test_first_match_wins_hearth_over_a_claude_ancestor(proc: FakeProc) -> None:
    proc.process(200, comm="claude", ppid=1)
    got = _loopback(proc, ppid=200, cgroup="0::/app.slice/pastore-x.scope")
    assert got == Attribution(Level.HEARTH, "cgroup:pastore-x.scope", "x")


def test_first_match_wins_coding_over_a_service_unit(proc: FakeProc) -> None:
    proc.process(200, comm="opencode", ppid=1)
    got = _loopback(proc, ppid=200, cgroup=SERVICE)
    assert got == Attribution(Level.CODING, "ancestor:opencode:300")


def test_hearth_scope_without_a_run_id_has_none(proc: FakeProc) -> None:
    got = _loopback(proc, cgroup="0::/app.slice/pastore-.scope")
    assert got == Attribution(Level.HEARTH, "cgroup:pastore-.scope", None)


def test_run_id_is_only_for_a_hearth_level_match(proc: FakeProc) -> None:
    rules = (SourceRule(Level.CODING, "cgroup", ("pastore-*.scope",)),)
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), cgroup=HEARTH)
    assert _attributor(proc, rules)(peer, LOCAL) == Attribution(
        Level.CODING, "cgroup:pastore-r123.scope", None
    )


def test_peer_rules_never_match_a_local_peer(proc: FakeProc) -> None:
    proc.process(300, inodes=(9001,))
    peer = proc.client(40001, 9001)
    rules = (SourceRule(Level.REINS, "peer", ("0.0.0.0/0",)),)
    assert _attributor(proc, rules)(peer, LOCAL) == Attribution(UNKNOWN_LEVEL, "local:300")


def test_labels_are_space_free(proc: FakeProc) -> None:
    proc.process(200, comm="my tool\tx", ppid=1)
    rules = (SourceRule(Level.CODING, "ancestor", ("my tool\tx",)),)
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), ppid=200)
    got = _attributor(proc, rules)(peer, LOCAL)
    assert got == Attribution(Level.CODING, "ancestor:my_tool_x:300")


def test_a_comm_that_is_not_utf8_does_not_break_the_chain(proc: FakeProc) -> None:
    proc.process(250, comm=b"bad\xffname", ppid=200)
    proc.process(200, comm="claude", ppid=1)
    assert _loopback(proc, ppid=250).level is Level.CODING


def test_ancestor_walk_does_not_include_pid_1(proc: FakeProc) -> None:
    rules = (SourceRule(Level.CODING, "ancestor", ("systemd",)),)
    proc.process(1, comm="systemd", ppid=0)
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), ppid=1)
    assert _attributor(proc, rules)(peer, LOCAL) == Attribution(UNKNOWN_LEVEL, "local:300")


def test_ancestor_walk_ends_on_a_parent_cycle(proc: FakeProc) -> None:
    rules = (SourceRule(Level.CODING, "ancestor", ("never",)),)
    proc.process(400, comm="a", ppid=401)
    proc.process(401, comm="b", ppid=400)
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), ppid=400)
    assert _attributor(proc, rules)(peer, LOCAL) == Attribution(UNKNOWN_LEVEL, "local:300")


def test_ancestor_walk_ends_at_a_missing_parent(proc: FakeProc) -> None:
    rules = (SourceRule(Level.CODING, "ancestor", ("python3",)),)
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), ppid=777)
    assert _attributor(proc, rules)(peer, LOCAL) == Attribution(
        Level.CODING, "ancestor:python3:300"
    )


def test_ancestor_walk_is_at_most_64_deep(proc: FakeProc) -> None:
    for pid in range(500, 600):
        proc.process(pid, comm=f"p{pid}", ppid=pid + 1)
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), ppid=500)
    deepest = (SourceRule(Level.CODING, "ancestor", ("p562",)),)
    beyond = (SourceRule(Level.CODING, "ancestor", ("p563",)),)
    assert _attributor(proc, deepest)(peer, LOCAL).level is Level.CODING
    assert _attributor(proc, beyond)(peer, LOCAL).level is UNKNOWN_LEVEL


# --- local addresses -----------------------------------------------------------------------------


def test_a_peer_equal_to_the_local_address_goes_through_proc(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001, ip="192.0.2.5")
    proc.process(200, comm="claude", ppid=1)
    proc.process(300, inodes=(9001,), ppid=200)
    assert _attributor(proc)(peer, ("192.0.2.5", 11434)) == Attribution(
        Level.CODING, "ancestor:claude:300"
    )


def test_a_lan_peer_that_is_not_the_local_address_never_reads_proc(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001, ip="192.0.2.5")
    proc.process(300, inodes=(9001,), cgroup=HEARTH)
    assert _attributor(proc)(peer, ("192.0.2.9", 11434)) == Attribution(Level.REINS, "peer:192.0.2.5")


def test_loopback_net_127_8_and_an_ipv4_mapped_local_peer(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001, ip="127.0.0.2", server="127.0.0.2")
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    assert _attributor(proc)(peer, ("127.0.0.2", 11434)).source == "cgroup:foo.service"
    assert (
        _attributor(proc)(("::ffff:127.0.0.2", 40001), ("::ffff:127.0.0.2", 11434)).source
        == "cgroup:foo.service"
    )


def test_a_loopback_client_on_another_127_address_than_the_server_is_local(proc: FakeProc) -> None:
    proc.row(_addr("127.0.0.2", 40001), _addr("127.0.0.1", 11434), 9001)
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    assert _attributor(proc)(("127.0.0.2", 40001), LOCAL).source == "cgroup:foo.service"


def _is_host_address(proc: FakeProc | Path, ip: str) -> bool:
    return attribution._host_address(
        proc.root if isinstance(proc, FakeProc) else proc, ipaddress.ip_address(ip)
    )


@pytest.mark.parametrize(
    ("ip", "listed"),
    [
        ("192.0.2.167", True),
        ("100.64.12.34", True),
        ("172.17.0.1", True),
        ("127.0.0.1", True),
        ("192.0.2.255", False),
        ("172.17.255.255", False),
        ("127.255.255.255", False),
        ("127.0.0.0", False),
        ("192.0.2.0", False),
        ("100.64.56.78", False),
        ("192.0.2.74", False),
        ("192.0.2.16", False),
    ],
)
def test_fib_trie_lists_only_host_local_32_entries(proc: FakeProc, ip: str, listed: bool) -> None:
    proc.addresses()
    assert _is_host_address(proc, ip) is listed


@pytest.mark.parametrize(
    ("ip", "listed"),
    [
        ("::1", True),
        ("2001:db8:0:1:1111:2222:3333:4444", True),
        ("fe80::1:2:3:4", True),
        ("fd00::1234:5678", True),
        ("2001:db8:0:1::", False),
        ("2001:db8:0:1::1", False),
        ("2001:db8::7", False),
    ],
)
def test_if_inet6_lists_the_hosts_ipv6_addresses(proc: FakeProc, ip: str, listed: bool) -> None:
    proc.addresses()
    assert _is_host_address(proc, ip) is listed


def test_missing_or_malformed_address_files_mean_not_ours(proc: FakeProc) -> None:
    def neither() -> bool:
        return not _is_host_address(proc, "192.0.2.167") and not _is_host_address(proc, "::1")

    assert neither()
    (proc.root / "net" / "fib_trie").write_text(
        "garbage\n|--\n/32 host LOCAL\n|-- 192.0.2.167 x\n/32 host LOCAL\n"
    )
    (proc.root / "net" / "if_inet6").write_text("zz\n\n   \n0001 01\n")
    assert neither()
    (proc.root / "net" / "fib_trie").write_bytes(b"\xff\xfe |-- \x80\n")
    (proc.root / "net" / "if_inet6").write_bytes(b"\xff\xfe\n")
    assert neither()
    for name in ("fib_trie", "if_inet6"):
        (proc.root / "net" / name).unlink()
        (proc.root / "net" / name).mkdir()
    assert neither()


@pytest.mark.parametrize(
    ("client", "server", "expected"),
    [
        ("127.0.0.2", "192.0.2.5", True),
        ("192.0.2.5", "192.0.2.5", True),
        ("192.0.2.167", "172.17.0.1", True),
        ("192.0.2.5", "172.17.0.1", False),
        ("192.0.2.167", None, True),
        ("192.0.2.5", None, False),
    ],
)
def test_is_local_is_loopback_the_server_address_or_an_address_of_this_host(
    proc: FakeProc, client: str, server: str | None, expected: bool
) -> None:
    proc.addresses()
    other = ipaddress.ip_address(server) if server else None
    assert attribution.is_local(ipaddress.ip_address(client), other, proc.root) is expected


def test_a_natted_own_host_client_is_local_and_unknown_without_a_socket_row(proc: FakeProc) -> None:
    """A host process arriving as the host's LAN address (Docker MASQUERADE) must not be taken for a Reins device."""
    proc.addresses()
    got = _attributor(proc)(("192.0.2.167", 40001), ("172.17.0.1", 11434))
    assert got == Attribution(UNKNOWN_LEVEL, "unknown:no-socket")


def test_an_own_host_address_peer_with_a_socket_row_is_attributed_through_proc(
    proc: FakeProc,
) -> None:
    proc.addresses()
    peer = proc.client(40001, 9001, ip="192.0.2.167", server="172.17.0.1")
    proc.process(200, comm="claude", ppid=1)
    proc.process(300, inodes=(9001,), ppid=200)
    assert _attributor(proc)(peer, ("172.17.0.1", 11434)) == Attribution(
        Level.CODING, "ancestor:claude:300"
    )


def test_a_lan_phone_that_is_not_an_address_of_this_host_stays_reins(proc: FakeProc) -> None:
    proc.addresses()
    got = _attributor(proc)(("192.0.2.74", 40001), ("192.0.2.167", 11434))
    assert got == Attribution(Level.REINS, "peer:192.0.2.74")


def test_the_docker_webui_container_stays_webui_beside_the_host_addresses(proc: FakeProc) -> None:
    proc.addresses()
    got = _attributor(proc)(("172.17.0.2", 40001), ("172.17.0.1", 11434))
    assert got == Attribution(Level.WEBUI, "peer:172.17.0.2")


def test_an_own_global_ipv6_peer_is_local_and_a_foreign_one_is_not(proc: FakeProc) -> None:
    proc.addresses()
    server = ("2001:db8:0:1::3d51", 11434)
    own = _attributor(proc)(("2001:db8:0:1:1111:2222:3333:4444", 40001), server)
    assert own == Attribution(UNKNOWN_LEVEL, "unknown:no-socket")
    foreign = _attributor(proc)(("2001:db8::7", 40001), server)
    assert foreign == Attribution(Level.REINS, "peer:2001:db8::7")


def test_an_ipv4_mapped_own_host_peer_is_local(proc: FakeProc) -> None:
    proc.addresses()
    got = _attributor(proc)(("::ffff:192.0.2.167", 40001), ("::ffff:172.17.0.1", 11434))
    assert got == Attribution(UNKNOWN_LEVEL, "unknown:no-socket")


def test_the_address_files_are_read_only_for_peers_that_are_neither_loopback_nor_the_server(
    proc: FakeProc, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc.addresses()
    read: list[str] = []
    real = Path.read_text

    def spy(self: Path, *args: Any, **kwargs: Any) -> str:
        read.append(self.name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    attribute = _attributor(proc)
    attribute(("127.0.0.1", 1), LOCAL)
    attribute(("192.0.2.5", 1), ("192.0.2.5", 11434))
    assert read == []
    attribute(("192.0.2.74", 1), ("192.0.2.5", 11434))
    assert read == ["fib_trie"]
    attribute(("2001:db8::7", 1), ("192.0.2.5", 11434))
    assert read == ["fib_trie", "if_inet6"]


def test_real_proc_lists_loopback_and_this_hosts_lan_address() -> None:
    proc = Path("/proc")
    assert _is_host_address(proc, "127.0.0.1")
    assert not _is_host_address(proc, "203.0.113.50")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))
        except OSError:
            pytest.skip("no route out, so no LAN address to look for")
        lan = probe.getsockname()[0]
    assert _is_host_address(proc, lan)


def test_ipv6_loopback_row_in_tcp6(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001, ip="::1")
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    assert _attributor(proc)(peer, ("::1", 11434)) == Attribution(
        Level.ROUTINE, "cgroup:foo.service"
    )


def test_ipv4_mapped_row_in_tcp6(proc: FakeProc) -> None:
    proc.row(_addr("::ffff:127.0.0.1", 40001), _addr("::ffff:127.0.0.1", 11434), 9001, table="tcp6")
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    assert _attributor(proc)(("127.0.0.1", 40001), LOCAL).source == "cgroup:foo.service"


# --- unattributable clients ----------------------------------------------------------------------


def test_another_users_socket_is_unknown_with_its_uid(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001, uid=UID + 7)
    proc.process(300, inodes=(9001,), cgroup=HEARTH)
    assert _attributor(proc)(peer, LOCAL) == Attribution(UNKNOWN_LEVEL, f"uid:{UID + 7}")


def test_no_socket_row_is_unknown(proc: FakeProc) -> None:
    proc.client(40001, 9001)
    proc.process(300, inodes=(9001,))
    assert _attributor(proc)(("127.0.0.1", 40002), LOCAL) == Attribution(
        UNKNOWN_LEVEL, "unknown:no-socket"
    )


def test_a_row_for_another_server_end_is_not_the_client_socket(proc: FakeProc) -> None:
    proc.row(_addr("127.0.0.1", 40001), _addr("127.0.0.1", 8080), 9001)
    proc.process(300, inodes=(9001,))
    assert _attributor(proc)(("127.0.0.1", 40001), LOCAL).source == "unknown:no-socket"


def test_no_process_holding_the_inode_is_unknown(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(1234,))
    assert _attributor(proc)(peer, LOCAL) == Attribution(UNKNOWN_LEVEL, "unknown:no-pid")


def test_an_orphaned_socket_with_inode_0_is_no_pid_without_scanning(
    proc: FakeProc, spy_reads: list[str]
) -> None:
    peer = proc.client(40001, 0)
    proc.process(300, inodes=(0,), cgroup=SERVICE)
    assert _attributor(proc)(peer, LOCAL) == Attribution(UNKNOWN_LEVEL, "unknown:no-pid")
    assert _fd_pids(spy_reads) == []


def test_pids_not_owned_by_our_uid_are_skipped(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001, uid=UID + 1)
    proc.process(300, inodes=(9001,))
    got = Attributor(RULES, lambda: 0.0, proc=proc.root, uid=UID + 1)(peer, LOCAL)
    assert got == Attribution(UNKNOWN_LEVEL, "unknown:no-pid")


def test_a_vanished_pid_is_unknown_not_an_error(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001)
    d = proc.process(300, inodes=(9001,))
    (d / "cgroup").unlink()
    assert _attributor(proc)(peer, LOCAL) == Attribution(UNKNOWN_LEVEL, "unknown:pid-gone")


@pytest.mark.skipif(os.getuid() == 0, reason="root reads every directory")
def test_an_unreadable_fd_dir_is_skipped(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    locked = proc.process(400, inodes=(1,))
    (locked / "fd").chmod(0)
    try:
        got = _attributor(proc)(peer, LOCAL)
    finally:
        (locked / "fd").chmod(0o700)
    assert got == Attribution(Level.ROUTINE, "cgroup:foo.service")


def test_a_pid_dir_without_fds_and_non_numeric_entries_are_skipped(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    (proc.root / "500").mkdir()
    (proc.root / "self").mkdir()
    (proc.root / "cpuinfo").write_text("x")
    assert _attributor(proc)(peer, LOCAL).source == "cgroup:foo.service"


def test_malformed_rows_are_skipped_and_the_real_row_is_still_found(proc: FakeProc) -> None:
    peer = ("127.0.0.1", 40001)
    proc.raw_row("garbage")
    proc.raw_row("")
    proc.raw_row("   0: ZZ:00 QQ:11 01")
    proc.raw_row(
        f"   1: {_addr('127.0.0.1', 40001)} {_addr('127.0.0.1', 11434)} 01 0:0 0:0 0 notauid 0 9001"
    )
    proc.raw_row(f"   2: {_addr('127.0.0.1', 40001)} {_addr('127.0.0.1', 11434)} 01")
    proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    assert _attributor(proc)(peer, LOCAL).source == "cgroup:foo.service"


def test_only_malformed_rows_give_no_socket(proc: FakeProc) -> None:
    proc.raw_row(
        f"   1: {_addr('127.0.0.1', 40001)} {_addr('127.0.0.1', 11434)} 01 0:0 0:0 0 notauid 0 9001"
    )
    assert _attributor(proc)(("127.0.0.1", 40001), LOCAL).source == "unknown:no-socket"


def test_a_missing_table_is_unknown_with_the_reason_not_a_raise(proc: FakeProc) -> None:
    (proc.root / "net" / "tcp").unlink()
    got = _attributor(proc)(("127.0.0.1", 40001), LOCAL)
    assert got == Attribution(UNKNOWN_LEVEL, "unknown:FileNotFoundError")


def test_an_unparsable_address_is_unknown_not_a_raise(proc: FakeProc) -> None:
    assert _attributor(proc)(("not-an-ip", 1), LOCAL) == Attribution(
        UNKNOWN_LEVEL, "unknown:ValueError"
    )


def test_ancestor_matching_is_exact_not_a_substring(proc: FakeProc) -> None:
    proc.process(200, comm="claude-desktop", ppid=1)
    assert _loopback(proc, ppid=200) == Attribution(UNKNOWN_LEVEL, "local:300")


def test_the_leaf_comes_from_the_0_line_on_a_hybrid_v1_v2_host(proc: FakeProc) -> None:
    got = _loopback(
        proc,
        cgroup="12:cpu,cpuacct:/user.slice\n1:name=systemd:/x.scope\n0::/system.slice/foo.service",
    )
    assert got == Attribution(Level.ROUTINE, "cgroup:foo.service")


def test_a_cgroup_file_without_a_v2_line_has_no_leaf(proc: FakeProc) -> None:
    assert _loopback(proc, cgroup="1:name=systemd:/foo.service") == Attribution(
        UNKNOWN_LEVEL, "local:300"
    )


def test_an_fd_whose_readlink_fails_does_not_end_the_pid(
    proc: FakeProc, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = proc.client(40001, 9001)
    d = proc.process(300, cgroup=SERVICE)
    (d / "fd" / "3").mkdir()
    (d / "fd" / "4").symlink_to("socket:[9001]")
    real = os.listdir
    monkeypatch.setattr(
        os, "listdir", lambda path: sorted(real(path), key=lambda n: (not n.isdecimal(), n))
    )
    assert _attributor(proc)(peer, LOCAL).source == "cgroup:foo.service"


def test_the_socket_inode_is_matched_exactly(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001)
    proc.process(400, inodes=(90011, 19001), cgroup="0::/x/pastore-r1.scope")
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    assert _attributor(proc)(peer, LOCAL).source == "cgroup:foo.service"


def test_a_garbage_ppid_ends_the_chain_without_aborting_the_call(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001)
    d = proc.process(300, inodes=(9001,), comm="claude")
    (d / "status").write_text("PPid:\t-\n")
    assert _attributor(proc)(peer, LOCAL) == Attribution(Level.CODING, "ancestor:claude:300")


# --- budget --------------------------------------------------------------------------------------


def test_the_clock_is_consulted_between_table_rows(proc: FakeProc) -> None:
    for i in range(200):
        proc.client(41000 + i, 8000 + i)
    clock = CountingClock()
    _attributor(proc, clock=clock)(("127.0.0.1", 41199), LOCAL)
    assert clock.calls >= 200


def test_over_budget_between_rows_returns_unknown_at_once(proc: FakeProc) -> None:
    for i in range(200):
        proc.client(41000 + i, 8000 + i)
    clock = CountingClock(ok_calls=2)
    got = _attributor(proc, clock=clock)(("127.0.0.1", 41199), LOCAL)
    assert got == Attribution(UNKNOWN_LEVEL, "unknown:timeout")
    assert clock.calls <= 4


def test_over_budget_between_pids_stops_the_scan(proc: FakeProc, spy_reads: list[str]) -> None:
    peer = proc.client(40001, 9001)
    for pid, inodes in ((301, (9001,)), (302, (1,)), (303, (2,))):
        proc.process(pid, inodes=inodes)

    def clock() -> float:
        return 1000.0 if any("/303/fd/" in p for p in spy_reads) else 0.0

    got = _attributor(proc, clock=clock)(peer, LOCAL)
    assert got == Attribution(UNKNOWN_LEVEL, "unknown:timeout")
    assert _fd_pids(spy_reads) == [303]


def test_over_budget_inside_a_large_fd_table_stops_the_scan(
    proc: FakeProc, spy_reads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(*range(100, 1100), 9001))
    real = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: sorted(real(path), key=int))

    def clock() -> float:
        return 1000.0 if len(spy_reads) > 10 else 0.0

    assert _attributor(proc, clock=clock)(peer, LOCAL) == Attribution(
        UNKNOWN_LEVEL, "unknown:timeout"
    )
    assert len(spy_reads) < 1001


def test_over_budget_right_after_the_pid_is_found_returns_timeout(
    proc: FakeProc, spy_reads: list[str]
) -> None:
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), cgroup=SERVICE)

    def clock() -> float:
        return 1000.0 if any(p.endswith("/300/fd/3") for p in spy_reads) else 0.0

    assert _attributor(proc, clock=clock)(peer, LOCAL) == Attribution(
        UNKNOWN_LEVEL, "unknown:timeout"
    )


def test_over_budget_inside_the_ancestor_walk_stops_it(
    proc: FakeProc, monkeypatch: pytest.MonkeyPatch
) -> None:
    for pid in range(500, 550):
        proc.process(pid, comm=f"p{pid}", ppid=pid + 1)
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), ppid=500)
    comm_reads: list[str] = []
    real = Path.read_text

    def spy(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "comm":
            comm_reads.append(str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)

    def clock() -> float:
        return 1000.0 if len(comm_reads) >= 3 else 0.0

    assert _attributor(proc, clock=clock)(peer, LOCAL) == Attribution(
        UNKNOWN_LEVEL, "unknown:timeout"
    )
    assert len(comm_reads) == 3


def _jump_after_first_call(to: float) -> Callable[[], float]:
    calls: list[None] = []

    def clock() -> float:
        calls.append(None)
        return 0.0 if len(calls) == 1 else to

    return clock


def test_the_default_budget_is_20_ms_and_the_default_uid_is_ours(proc: FakeProc) -> None:
    peer = proc.client(40001, 9001)
    proc.process(300, inodes=(9001,), cgroup=SERVICE)
    inside = Attributor(RULES, _jump_after_first_call(0.019), proc=proc.root)(peer, LOCAL)
    over = Attributor(RULES, _jump_after_first_call(0.021), proc=proc.root)(peer, LOCAL)
    assert inside == Attribution(Level.ROUTINE, "cgroup:foo.service")
    assert over == Attribution(UNKNOWN_LEVEL, "unknown:timeout")


# --- pid cache -----------------------------------------------------------------------------------


def _many_pids(proc: FakeProc, pids: range) -> None:
    for pid in pids:
        proc.process(pid, inodes=(pid * 10,), cgroup=SERVICE)


def test_a_repeat_client_is_found_without_the_full_scan(
    proc: FakeProc, spy_reads: list[str]
) -> None:
    _many_pids(proc, range(301, 351))
    attribute = _attributor(proc)
    first = proc.client(40001, 3200)
    assert attribute(first, LOCAL).source == "cgroup:foo.service"
    assert len(_fd_pids(spy_reads)) == 31
    spy_reads.clear()
    second = proc.client(40002, 3201)
    (proc.root / "320" / "fd" / "98").symlink_to("socket:[3201]")
    assert attribute(second, LOCAL).source == "cgroup:foo.service"
    assert _fd_pids(spy_reads) == [320]


def test_a_cached_pid_that_no_longer_holds_the_socket_falls_back_to_the_scan(
    proc: FakeProc,
) -> None:
    _many_pids(proc, range(301, 306))
    attribute = _attributor(proc)
    attribute(proc.client(40001, 3030), LOCAL)
    second = proc.client(40002, 3041)
    (proc.root / "304" / "fd" / "99").symlink_to("socket:[3041]")
    assert attribute(second, LOCAL).source == "cgroup:foo.service"


def test_the_cache_holds_only_the_last_32_matched_pids(
    proc: FakeProc, spy_reads: list[str]
) -> None:
    _many_pids(proc, range(301, 341))
    attribute = _attributor(proc)
    for pid in range(340, 300, -1):
        peer = proc.client(50000 + pid, pid * 10)
        attribute(peer, LOCAL)
    spy_reads.clear()
    peer = proc.client(40001, 9999)
    (proc.root / "340" / "fd" / "99").symlink_to("socket:[9999]")
    attribute(peer, LOCAL)
    assert len(_fd_pids(spy_reads)) == 33


def test_a_call_that_caches_a_pid_mid_scan_does_not_hide_it_from_another_call(
    proc: FakeProc, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call A scans past pid 305 while call B matches and caches it; A must still find its own socket in 305."""
    _many_pids(proc, range(301, 311))
    attribute = _attributor(proc)
    (proc.root / "305" / "fd" / "98").symlink_to("socket:[3051]")
    first, second = proc.client(40001, 3050), proc.client(40002, 3051)
    real = os.readlink
    fired: list[bool] = []
    seen: list[Attribution] = []

    def spy(path: str | os.PathLike[str], *a: Any, **k: Any) -> str:
        if "/308/fd/" in str(path) and not fired:
            fired.append(True)
            seen.append(attribute(second, LOCAL))
        return real(path, *a, **k)

    monkeypatch.setattr(os, "readlink", spy)
    assert attribute(first, LOCAL) == Attribution(Level.ROUTINE, "cgroup:foo.service")
    assert seen == [Attribution(Level.ROUTINE, "cgroup:foo.service")]


def _concurrent_sources(attribute: Attributor, peers: list[Address]) -> list[str]:
    """Attribute every peer at once from its own thread, released together by a barrier."""
    start = threading.Barrier(len(peers))
    sources: list[str] = []

    def run(peer: Address) -> None:
        start.wait()
        sources.append(attribute(peer, LOCAL).source)

    threads = [threading.Thread(target=run, args=(peer,)) for peer in peers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sources


def test_two_threads_sharing_one_attributor_both_find_their_client(proc: FakeProc) -> None:
    for pid in range(400, 700):
        proc.process(pid, inodes=tuple(range(pid * 100, pid * 100 + 20)), cgroup=SERVICE)
    holder = proc.process(300, cgroup=SERVICE)
    wrong: list[str] = []
    for r in range(60):
        inodes = (900000 + 2 * r, 900001 + 2 * r)
        peers = [proc.client(20000 + 2 * r + k, inode) for k, inode in enumerate(inodes)]
        for k, inode in enumerate(inodes):
            (holder / "fd" / str(1000 + 2 * r + k)).symlink_to(f"socket:[{inode}]")
        sources = _concurrent_sources(_attributor(proc), peers)
        wrong += [s for s in sources if s != "cgroup:foo.service"]
    assert wrong == []


# --- real /proc ----------------------------------------------------------------------------------


def test_real_proc_attributes_this_process_through_its_own_loopback_socket(
    capsys: pytest.CaptureFixture[str],
) -> None:
    comm = Path(f"/proc/{os.getpid()}/comm").read_text().strip()
    rules = (SourceRule(Level.CODING, "ancestor", (comm,)),)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with (
            socket.create_connection(listener.getsockname()) as client,
            listener.accept()[0] as accepted,
        ):
            peer = accepted.getpeername()[:2]
            local = accepted.getsockname()[:2]
            assert peer == client.getsockname()[:2]
            attribute = Attributor(rules, time.monotonic, proc=Path("/proc"), budget_s=1.0)
            start = time.perf_counter()
            got = attribute(peer, local)
            elapsed_ms = (time.perf_counter() - start) * 1000
    with capsys.disabled():
        print(f"\nreal /proc attribution took {elapsed_ms:.1f} ms")
    assert got.level is Level.CODING
    assert got.source.endswith(f":{os.getpid()}")
