"""Per-host model facts (/api/tags, /api/ps, /api/show) and the executor health probe, read each tick by the dispatcher.

Depends on: pastore.spec (Host, WORKSTATION, canonical_model, idle_reason); stdlib urllib and json; the HTTP call is injected.
"""

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from pastore.spec import WORKSTATION, Host, canonical_model, idle_reason

HttpJson = Callable[[str, Mapping[str, object] | None, float], object]
PROBE_TIMEOUT_S = 2.0
READ_TIMEOUT_S = 5.0
MAX_BODY_BYTES = 8 * 1024 * 1024


class HostError(RuntimeError):
    """A host answered /api/tags, /api/ps or /api/show with a shape pastore does not expect."""


@dataclass(frozen=True)
class KvShape:
    """The three model_info numbers the KV cache size depends on (D-041)."""

    layers: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class HostState:
    """One policy host as the dispatcher sees it this tick; every mapping is empty when `reason` is set."""

    host: Host
    reason: str | None
    sizes: Mapping[str, int]
    resident: Mapping[str, int]
    loaded: Mapping[str, int] = field(default_factory=dict)
    kv: Mapping[str, KvShape | None] = field(default_factory=dict)


def _fetch(url: str, body: Mapping[str, object] | None, timeout_s: float) -> bytes:
    if body is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw: bytes = response.read(MAX_BODY_BYTES + 1)
            missing: int = response.length or 0
    except urllib.error.HTTPError as err:
        err.close()
        raise
    except http.client.HTTPException as exc:
        raise OSError(f"{url}: {exc!r}") from exc
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError(f"{url}: answer over {MAX_BODY_BYTES} bytes")
    if missing:
        raise OSError(f"{url}: answer cut short, {missing} bytes missing")
    return raw


def http_json(url: str, body: Mapping[str, object] | None, timeout_s: float) -> object:
    """The production HttpJson: urllib GET, or POST of the JSON body; non-2xx, a cut-short or oversized answer raises."""
    raw = _fetch(url, body, timeout_s)
    try:
        return json.loads(raw)
    except RecursionError as exc:
        raise ValueError(f"{url}: JSON nested too deeply") from exc


def executor_reason(host: Host, get: HttpJson) -> str | None:
    """executor-down unless GET {url}/api/version answers a JSON object with a string version; the probe uses a 2 s socket timeout."""
    try:
        answer = get(host.url + "/api/version", None, PROBE_TIMEOUT_S)
    except (OSError, ValueError):
        return idle_reason("executor-down")
    if isinstance(answer, dict) and isinstance(answer.get("version"), str):
        return None
    return idle_reason("executor-down")


def _plain_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _entry(
    host: Host, path: str, keys: tuple[str, ...], entry: object
) -> tuple[str, dict[str, int]]:
    if not isinstance(entry, dict):
        raise HostError(f"{host.name}: {path} lists a model entry that is not an object")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise HostError(f"{host.name}: {path} lists a model without a non-empty string name")
    values: dict[str, int] = {}
    for key in keys:
        value = _plain_int(entry.get(key))
        if value is None or value < 0:
            raise HostError(
                f"{host.name}: {path} lists {name!r} without a non-negative integer {key}"
            )
        values[key] = value
    return canonical_model(name), values


def _listing(
    host: Host, get: HttpJson, path: str, keys: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    answer = get(host.url + path, None, READ_TIMEOUT_S)
    if not isinstance(answer, dict) or not isinstance(answer.get("models"), list):
        raise HostError(f"{host.name}: {path} did not answer an object with a models list")
    rows: dict[str, dict[str, int]] = {}
    for item in answer["models"]:
        name, values = _entry(host, path, keys, item)
        if name in rows:
            raise HostError(f"{host.name}: {path} lists {name!r} twice")
        rows[name] = values
    return rows


def model_sizes(host: Host, get: HttpJson) -> dict[str, int]:
    """/api/tags as canonical model name -> size bytes."""
    rows = _listing(host, get, "/api/tags", ("size",))
    return {name: values["size"] for name, values in rows.items()}


def ps_sizes(host: Host, get: HttpJson) -> tuple[dict[str, int], dict[str, int]]:
    """/api/ps as (canonical name -> size_vram bytes, canonical name -> size bytes), from one request."""
    rows = _listing(host, get, "/api/ps", ("size_vram", "size"))
    return (
        {name: values["size_vram"] for name, values in rows.items()},
        {name: values["size"] for name, values in rows.items()},
    )


def _positive(value: object) -> int | None:
    number = _plain_int(value)
    return number if number is not None and number > 0 else None


def _head_dim(info: Mapping[str, object], arch: str) -> int | None:
    key = f"{arch}.attention.key_length"
    if key in info:
        return _positive(info[key])
    width = _positive(info.get(f"{arch}.embedding_length"))
    heads = _positive(info.get(f"{arch}.attention.head_count"))
    if width is None or heads is None:
        return None
    return width // heads or None


def kv_shape(host: Host, model: str, get: HttpJson) -> KvShape | None:
    """/api/show model_info as layers, KV heads and head dim; None when any of them is missing or not a positive int."""
    answer = get(host.url + "/api/show", {"model": model}, READ_TIMEOUT_S)
    if not isinstance(answer, dict):
        raise HostError(f"{host.name}: /api/show for {model!r} did not answer an object")
    info = answer.get("model_info")
    if not isinstance(info, dict) or not isinstance(info.get("general.architecture"), str):
        return None
    arch: str = info["general.architecture"]
    layers = _positive(info.get(f"{arch}.block_count"))
    kv_heads = _positive(info.get(f"{arch}.attention.head_count_kv"))
    head_dim = _head_dim(info, arch)
    if layers is None or kv_heads is None or head_dim is None:
        return None
    return KvShape(layers, kv_heads, head_dim)


def read_host(host: Host, get: HttpJson, models: Collection[str]) -> HostState:
    """The host's health and model facts by canonical name, KV shapes for the `models` it has; an unreadable host takes no new run."""
    local = host.name == WORKSTATION
    if not local:
        down = executor_reason(host, get)
        if down is not None:
            return HostState(host, down, {}, {})
    try:
        sizes = model_sizes(host, get)
        resident, loaded = ps_sizes(host, get)
        names = dict.fromkeys(canonical_model(model) for model in models)
        kv = {name: kv_shape(host, name, get) for name in names if name in sizes}
    except (OSError, ValueError, HostError):
        return HostState(host, idle_reason("ollama-down" if local else "executor-down"), {}, {})
    return HostState(host, None, sizes, resident, loaded, kv)
