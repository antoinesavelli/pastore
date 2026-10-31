"""Parse and validate a consumer's gpu-tasks.yaml into TaskSpecs: schema, path rule (R5), per-task fingerprints.

Depends on: pastore.spec, PyYAML and stdlib only.
"""

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import yaml

from pastore.spec import BASE_ENV_KEYS, RUN_DIR_ENV, Ready, RunReader, TaskSpec, ValidateReader

ALLOWED_BARE = frozenset({"torra", "opencode"})
NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
DURATION = re.compile(r"([1-9][0-9]*)([smh])")
UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}
TASK_KEYS = frozenset(
    "id kind argv cwd env model models timeout ok_exit resumable ollama_env model_env reader ready".split()
)
RETIRED_RUN_DIR_ENV = "FUCONE_RUN_DIR"  # the retired R31 name: no longer set, still refused (D-045)
RESERVED_ENV = frozenset({RUN_DIR_ENV, RETIRED_RUN_DIR_ENV})
PASTORE_SETS = frozenset(BASE_ENV_KEYS) | RESERVED_ENV
Which = Callable[[str], str | None]
_NO_ARGV: tuple[str, ...] = ()
_NO_READER: RunReader | ValidateReader = RunReader(())


class ManifestError(Exception):
    """The manifest is unreadable or invalid; the message lists every problem."""


def parse_duration(value: object) -> int:
    """Seconds from a positive int or a `<n>s|m|h` string."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    match = DURATION.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ManifestError(f"duration must be a positive int or <n>s|m|h, got {value!r}")
    return int(match[1]) * UNIT_SECONDS[match[2]]


def _argv(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ManifestError(f"{name} must be a list, never a shell string")
    if not isinstance(value, list) or not value or not all(isinstance(a, str) for a in value):
        raise ManifestError(f"{name} must be a non-empty list of strings")
    return tuple(value)


def _inside(root: Path, path: str) -> bool:
    return os.path.commonpath([str(root), path]) == str(root)


def _check_exe(argv: tuple[str, ...], cwd: Path, root: Path, which: Which, name: str) -> None:
    exe = argv[0] if argv else ""
    if "/" not in exe:
        if exe not in ALLOWED_BARE:
            raise ManifestError(f"{name}[0] {exe!r} is a bare name; only {sorted(ALLOWED_BARE)}")
        if which(exe) is None:
            raise ManifestError(f"{name}[0] {exe!r} is not on PATH")
    elif not _inside(root, os.path.abspath(os.path.join(cwd, exe))):
        raise ManifestError(f"{name}[0] {exe!r} resolves outside the consumer root {root}")


def _cwd(value: object, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError("cwd must be a non-empty string")
    real = os.path.realpath(os.path.join(root, value))
    if not _inside(root, real):
        raise ManifestError(f"cwd {value!r} resolves outside the consumer root {root}")
    return Path(real)


def _ollama_env(value: object) -> str:
    if not isinstance(value, str) or not ENV_NAME.fullmatch(value):
        raise ManifestError(f"ollama_env must match {ENV_NAME.pattern}")
    if value in PASTORE_SETS:
        raise ManifestError(f"ollama_env {value!r} may not be any of {sorted(PASTORE_SETS)}")
    return value


def _model_env(value: object, ollama_env: str) -> str:
    if not isinstance(value, str) or not ENV_NAME.fullmatch(value):
        raise ManifestError(f"model_env must match {ENV_NAME.pattern}")
    if value == ollama_env or value in PASTORE_SETS:
        raise ManifestError(
            f"model_env {value!r} may not be ollama_env or any of {sorted(PASTORE_SETS)}"
        )
    return value


def _env(value: object, ollama_env: str, model_env: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ManifestError("env must be a mapping of string to string")
    reserved = (RESERVED_ENV | ({ollama_env, model_env} - {""})) & set(value)
    if reserved:
        raise ManifestError(f"env may not set {sorted(reserved)}; pastore owns them")
    return dict(value)


def _models(raw: dict[str, object]) -> tuple[str, ...]:
    if ("model" in raw) == ("models" in raw):
        raise ManifestError("exactly one of model or models is required")
    value = raw["model"] if "model" in raw else raw["models"]
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(m, str) and m for m in items):
        raise ManifestError("model must be a non-empty string, models a non-empty list of them")
    return tuple(items)


def _ok_exit(value: object) -> frozenset[int]:
    if value is None:
        return frozenset({0})
    if not isinstance(value, list) or not value:
        raise ManifestError("ok_exit must be a non-empty list of ints")
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in value):
        raise ManifestError("ok_exit must be a non-empty list of ints")
    return frozenset(value)


def _keyed(value: object, allowed: set[str], name: str) -> dict[str, object]:
    if isinstance(value, str):
        raise ManifestError(f"{name} must be a mapping; a doc path is not a reader")
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise ManifestError(f"{name} must be a mapping with keys from {sorted(allowed)}")
    return value


def _reader(value: object, cwd: Path, root: Path, which: Which) -> RunReader | ValidateReader:
    raw = _keyed(value, {"run", "validate", "review"}, "reader")
    if set(raw) == {"run"}:
        argv = _argv(raw["run"], "reader.run")
        _check_exe(argv, cwd, root, which, "reader.run")
        return RunReader(argv)
    if set(raw) == {"validate", "review"} and raw["review"] == "owner":
        argv = _argv(raw["validate"], "reader.validate")
        _check_exe(argv, cwd, root, which, "reader.validate")
        return ValidateReader(argv, "owner")
    raise ManifestError("reader must be exactly {run: argv} or {validate: argv, review: owner}")


def _ready(value: object, cwd: Path, root: Path, which: Which) -> Ready | None:
    if value is None:
        return None
    raw = _keyed(value, {"every", "check"}, "ready")
    if set(raw) != {"every", "check"}:
        raise ManifestError("ready must have exactly every and check")
    check = _argv(raw["check"], "ready.check")
    _check_exe(check, cwd, root, which, "ready.check")
    return Ready(parse_duration(raw["every"]), check)


def _fingerprint(raw: dict[str, object], consumer: str, root: Path) -> str:
    try:
        text = json.dumps(
            {"consumer": consumer, "root": str(root), "entry": raw},
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError as exc:
        raise ManifestError(f"entry is not plain JSON-able data: {exc}") from exc
    return hashlib.sha256(text.encode()).hexdigest()


def _guard[T](errors: list[str], tid: str, fn: Callable[[], T], default: T) -> T:
    try:
        return fn()
    except ManifestError as exc:
        errors.append(f"{tid}: {exc}")
        return default


def _parse_task(
    raw: dict[str, object], tid: str, consumer: str, root: Path, which: Which, errors: list[str]
) -> TaskSpec:
    """Build a TaskSpec from one entry, appending every problem to errors (default values on error)."""

    def guard[T](fn: Callable[[], T], default: T) -> T:
        return _guard(errors, tid, fn, default)

    errors += [f"{tid}: unknown key {k!r}" for k in raw if k not in TASK_KEYS]
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        errors.append(f"{tid}: kind must be a non-empty string")
    if raw.get("resumable") is not True:
        errors.append(f"{tid}: resumable must be literally true")
    ollama_env = guard(lambda: _ollama_env(raw.get("ollama_env")), "")
    model_env = guard(lambda: _model_env(raw.get("model_env"), ollama_env), "")
    cwd = guard(lambda: _cwd(raw.get("cwd"), root), root)
    argv = guard(lambda: _argv(raw.get("argv"), "argv"), _NO_ARGV)
    if argv:
        guard(lambda: _check_exe(argv, cwd, root, which, "argv"), None)
    return TaskSpec(
        consumer=consumer,
        id=tid,
        kind=kind if isinstance(kind, str) else "",
        argv=argv,
        cwd=cwd,
        env=guard(lambda: _env(raw.get("env"), ollama_env, model_env), {}),
        models=guard(lambda: _models(raw), ()),
        timeout_s=guard(lambda: parse_duration(raw.get("timeout")), 0),
        ok_exit=guard(lambda: _ok_exit(raw.get("ok_exit")), frozenset({0})),
        ollama_env=ollama_env,
        model_env=model_env,
        reader=guard(lambda: _reader(raw.get("reader"), cwd, root, which), _NO_READER),
        ready=guard(lambda: _ready(raw.get("ready"), cwd, root, which), None),
        root=root,
        fingerprint=guard(lambda: _fingerprint(raw, consumer, root), ""),
    )


def _entries(data: object) -> list[object]:
    if not isinstance(data, dict) or set(data) != {"version", "tasks"}:
        raise ManifestError("manifest must be a mapping with exactly version and tasks")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ManifestError(f"version must be 1, got {data['version']!r}")
    if not isinstance(data["tasks"], list):
        raise ManifestError("tasks must be a list")
    return data["tasks"]


def load_manifest(path: Path, consumer: str, which: Which = shutil.which) -> dict[str, TaskSpec]:
    """Validate a gpu-tasks.yaml; return its tasks by id or raise ManifestError listing every problem."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    root = path.parent.resolve()
    tasks: dict[str, TaskSpec] = {}
    errors: list[str] = []
    for index, raw in enumerate(_entries(data)):
        tid = raw.get("id") if isinstance(raw, dict) else None
        if not isinstance(raw, dict) or not isinstance(tid, str) or not NAME.fullmatch(tid):
            errors.append(f"tasks[{index}]: needs a mapping with id matching {NAME.pattern}")
        elif tid in tasks:
            errors.append(f"{tid}: duplicate id")
        else:
            tasks[tid] = _parse_task(raw, tid, consumer, root, which, errors)
    if errors:
        raise ManifestError("\n".join(errors))
    return tasks
