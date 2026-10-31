"""Chooses the Ollama host for one run of a model from each host's models, loaded models and active runs (R25, D-041).

Depends on: pastore.spec (Host, WORKSTATION); pure code, no I/O and no clock.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pastore.spec import WORKSTATION, Host

GIB = 1024**3


@dataclass(frozen=True)
class ModelFacts:
    """One model on one host: tag size, KV bytes per slot (None when unknown), last loaded size seen."""

    size: int
    kv_per_slot: int | None
    loaded: int | None = None


@dataclass(frozen=True)
class HostLoad:
    """One healthy host as placement sees it: its models, loaded models (size_vram) and one `held` entry per run."""

    host: Host
    models: Mapping[str, ModelFacts]
    resident: Mapping[str, int]
    held: tuple[str, ...]


def kv_bytes_per_slot(layers: int, kv_heads: int, head_dim: int, context: int) -> int:
    """KV cache bytes of one slot: 2 x layers x KV heads x head dim x context x 2 bytes (fp16)."""
    return 2 * layers * kv_heads * head_dim * context * 2


def base_need(size: int) -> int:
    """VRAM a model needs before its KV cache: size x 1.2 + 1 GiB, rounded up."""
    return -(-size * 6 // 5) + GIB


def need_bytes(facts: ModelFacts, parallel: int) -> int | None:
    """VRAM a model needs on a host: the loaded figure once seen, else the estimate, None when KV is unknown."""
    if facts.loaded is not None:
        return facts.loaded
    if facts.kv_per_slot is None:
        return None
    return base_need(facts.size) + parallel * facts.kv_per_slot


def free_bytes(load: HostLoad) -> int:
    """Host VRAM minus the resident models' VRAM minus the need of held models not yet resident."""
    free = load.host.vram_gb * GIB - sum(load.resident.values())
    for model in set(load.held).difference(load.resident):
        facts = load.models.get(model)
        need = None if facts is None else need_bytes(facts, load.host.parallel)
        if need is None:
            return 0
        free -= need
    return free


def _required(facts: ModelFacts, load: HostLoad) -> int:
    """VRAM to compare with the card: the need, or the base need when the KV size is unknown."""
    need = need_bytes(facts, load.host.parallel)
    return base_need(facts.size) if need is None else need


def _eligible(model: str, load: HostLoad) -> bool:
    """Whether the host has the model, a free slot, no run of it, and could ever fit it."""
    facts = load.models.get(model)
    if facts is None or len(load.held) >= load.host.max_models or model in load.held:
        return False
    if need_bytes(facts, load.host.parallel) is None and (load.resident or load.held):
        return False
    return _required(facts, load) <= load.host.vram_gb * GIB


def _loaded_host(model: str, eligible: Sequence[HostLoad]) -> str | None:
    """The first eligible host where the model is already loaded: no free-VRAM check."""
    for load in eligible:
        if model in load.resident:
            return load.host.name
    return None


def _first_fit(model: str, eligible: Sequence[HostLoad]) -> str | None:
    """The first eligible host with free VRAM for the model, the workstation last."""
    ordered = [ld for ld in eligible if ld.host.name != WORKSTATION]
    ordered += [ld for ld in eligible if ld.host.name == WORKSTATION]
    for load in ordered:
        if _required(load.models[model], load) <= free_bytes(load):
            return load.host.name
    return None


def place(model: str, loads: Sequence[HostLoad]) -> str | None:
    """The host name for a run of `model`, or None when it fits nowhere (idle reason no-fit)."""
    eligible = [load for load in loads if _eligible(model, load)]
    loaded = _loaded_host(model, eligible)
    return _first_fit(model, eligible) if loaded is None else loaded
