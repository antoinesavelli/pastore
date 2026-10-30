"""Pins gpu-tasks.yaml validation: schema, the argv/cwd path rule (R5) and per-task fingerprints.

Depends on: pastore.manifest, pastore.spec and pytest only.
"""

from pathlib import Path

import pytest

from pastore.manifest import ManifestError, load_manifest, parse_duration
from pastore.spec import Ready, RunReader, ValidateReader

TASK = """  - id: bench
    kind: eval
    argv: [.venv/bin/python, -m, bench]
    cwd: .
    env: {MODE: fast}
    model: qwen3:8b
    timeout: 2h
    resumable: true
    ollama_env: OLLAMA_HOST
    model_env: PASTORE_MODEL
    reader: {run: [.venv/bin/python, read.py]}
"""
OTHER = TASK.replace("id: bench", "id: second")


def fake_which(name: str) -> str | None:
    return f"/usr/bin/{name}" if name in ("torra", "opencode") else None


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").symlink_to("/usr/bin/python3")
    return root


def write(root: Path, tasks: str, header: str = "version: 1\ntasks:\n") -> Path:
    path = root / "gpu-tasks.yaml"
    path.write_text(header + tasks)
    return path


def load(root: Path, tasks: str) -> dict[str, object]:
    return dict(load_manifest(write(root, tasks), "torra", fake_which))


def test_valid_manifest_builds_task_specs(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    spec = load_manifest(write(root, TASK), "torra", fake_which)["bench"]
    assert spec.ref == "torra/bench" and spec.root == root.resolve() and spec.cwd == root.resolve()
    assert spec.timeout_s == 7200 and spec.ok_exit == frozenset({0})
    assert spec.models == ("qwen3:8b",) and spec.env == {"MODE": "fast"}
    assert spec.reader == RunReader((".venv/bin/python", "read.py")) and spec.ready is None
    assert len(spec.fingerprint) == 64


def test_model_env_lands_in_the_task_spec(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    spec = load_manifest(
        write(root, TASK.replace("PASTORE_MODEL", "BENCH_MODEL")), "torra", fake_which
    )["bench"]
    assert spec.model_env == "BENCH_MODEL" and spec.ollama_env == "OLLAMA_HOST"


def test_validate_reader_and_ready_rule(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    entry = TASK.replace(
        "reader: {run: [.venv/bin/python, read.py]}",
        "reader: {validate: [torra, check], review: owner}\n"
        "    ready: {every: 6h, check: [.venv/bin/python, ready.py]}\n    ok_exit: [0, 3]",
    )
    spec = load_manifest(write(root, entry), "torra", fake_which)["bench"]
    assert spec.reader == ValidateReader(("torra", "check"), "owner")
    assert spec.ready == Ready(21600, (".venv/bin/python", "ready.py"))
    assert spec.ok_exit == frozenset({0, 3})


def test_venv_symlink_counts_as_inside_but_escape_is_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert "bench" in load_manifest(write(root, TASK), "torra", fake_which)
    for bad in ("/usr/bin/python3", "../outside", "../repo/../outside/x"):
        with pytest.raises(ManifestError, match=r"bench: argv\[0\].*outside the consumer root"):
            load(root, TASK.replace(".venv/bin/python, -m, bench", f"{bad}, -m, bench"))
    with pytest.raises(ManifestError, match=r"reader.run\[0\].*outside"):
        load(root, TASK.replace("{run: [.venv/bin/python,", "{run: [/usr/bin/python3,"))
    with pytest.raises(ManifestError, match=r"cwd '\.\./x' resolves outside"):
        load(root, TASK.replace("cwd: .", "cwd: ../x"))
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(ManifestError, match="cwd 'link' resolves outside"):
        load(root, TASK.replace("cwd: .", "cwd: link"))


def test_bare_names_must_be_allowed_and_found(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    with pytest.raises(ManifestError, match="bare name"):
        load(root, TASK.replace(".venv/bin/python, -m", "python, -m"))
    assert "bench" in load(root, TASK.replace(".venv/bin/python, -m", "opencode, -m"))
    with pytest.raises(ManifestError, match="not on PATH"):
        load_manifest(
            write(root, TASK.replace(".venv/bin/python, -m", "torra, -m")), "t", lambda _: None
        )


def test_model_env_changes_the_fingerprint(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    before = load_manifest(write(root, TASK), "torra", fake_which)["bench"]
    after = load_manifest(
        write(root, TASK.replace("PASTORE_MODEL", "BENCH_MODEL")), "torra", fake_which
    )["bench"]
    assert before.fingerprint != after.fingerprint


def test_fingerprint_is_per_task_entry(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    before = load_manifest(write(root, TASK + OTHER), "torra", fake_which)
    edited = OTHER.replace("timeout: 2h", "timeout: 3h")
    after = load_manifest(write(root, TASK + edited), "torra", fake_which)
    assert after["bench"].fingerprint == before["bench"].fingerprint
    assert after["second"].fingerprint != before["second"].fingerprint
    assert before["bench"].fingerprint != before["second"].fingerprint


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("argv: [.venv/bin/python, -m, bench]", "argv: .venv/bin/python -m bench", "never a shell"),
        ("resumable: true", "resumable: false", "resumable must be literally true"),
        ("    resumable: true\n", "", "resumable must be literally true"),
        ("    ollama_env: OLLAMA_HOST\n", "", "ollama_env must match"),
        ("ollama_env: OLLAMA_HOST", "ollama_env: ollama-host", "ollama_env must match"),
        ("    model_env: PASTORE_MODEL\n", "", "model_env must match"),
        ("model_env: PASTORE_MODEL", "model_env: pastore-model", "model_env must match"),
        ("model_env: PASTORE_MODEL", "model_env: PASTORE-MODEL", "model_env must match"),
        ("model_env: PASTORE_MODEL", "model_env: 7", "model_env must match"),
        ("model_env: PASTORE_MODEL", "model_env: OLLAMA_HOST", "model_env 'OLLAMA_HOST' may not"),
        (
            "ollama_env: OLLAMA_HOST\n    model_env: PASTORE_MODEL",
            "ollama_env: MY_OLLAMA\n    model_env: MY_OLLAMA",
            "model_env 'MY_OLLAMA' may not",
        ),
        ("model_env: PASTORE_MODEL", "model_env: PASTORE_RUN_DIR", "model_env 'PASTORE_RUN_DIR'"),
        ("model_env: PASTORE_MODEL", "model_env: FUCONE_RUN_DIR", "model_env 'FUCONE_RUN_DIR'"),
        ("env: {MODE: fast}", "env: {PASTORE_MODEL: x}", r"env may not set \['PASTORE_MODEL'\]"),
        ("    model: qwen3:8b\n", "", "exactly one of model or models"),
        ("model: qwen3:8b", "model: a\n    models: [b]", "exactly one of model or models"),
        ("timeout: 2h", "timeout: 0", "duration"),
        ("timeout: 2h", "timeout: 2d", "duration"),
        ("kind: eval", "kind: ''", "kind"),
        ("env: {MODE: fast}", "env: {PASTORE_RUN_DIR: /x}", "PASTORE_RUN_DIR"),
        ("env: {MODE: fast}", "env: {FUCONE_RUN_DIR: /x}", "FUCONE_RUN_DIR"),
        ("env: {MODE: fast}", "env: {OLLAMA_HOST: x}", "OLLAMA_HOST"),
        ("env: {MODE: fast}", "env: {MODE: 1}", "string to string"),
        ("kind: eval", "kind: eval\n    surprise: 1", "unknown key 'surprise'"),
        ("reader: {run: [.venv/bin/python, read.py]}", "reader: docs/READER.md", "not a reader"),
        ("reader: {run: [.venv/bin/python, read.py]}", "reader: {validate: [torra, x]}", "exactly"),
        (
            "reader: {run: [.venv/bin/python, read.py]}",
            "reader: {validate: [torra, x], review: me}",
            "exactly",
        ),
        ("cwd: .\n", "cwd: .\n    ok_exit: []\n", "ok_exit"),
        ("cwd: .\n", "cwd: .\n    ready: {every: 1h}\n", "every and check"),
    ],
)
def test_invalid_entries_are_refused_with_task_prefix(
    tmp_path: Path, old: str, new: str, fragment: str
) -> None:
    root = make_root(tmp_path)
    with pytest.raises(ManifestError, match=fragment) as info:
        load(root, TASK.replace(old, new))
    assert str(info.value).startswith("bench: ")


def test_invalid_env_names_do_not_reserve_the_empty_key(tmp_path: Path) -> None:
    entry = (
        TASK.replace("env: {MODE: fast}", 'env: {"": x}')
        .replace("ollama_env: OLLAMA_HOST", "ollama_env: bad")
        .replace("PASTORE_MODEL", "bad")
    )
    with pytest.raises(ManifestError) as info:
        load(make_root(tmp_path), entry)
    assert str(info.value).splitlines() == [
        "bench: ollama_env must match [A-Z_][A-Z0-9_]*",
        "bench: model_env must match [A-Z_][A-Z0-9_]*",
    ]


def test_file_level_problems_and_all_errors_listed(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    with pytest.raises(ManifestError, match="duplicate id"):
        load(root, TASK + TASK)
    with pytest.raises(ManifestError, match="version must be 1"):
        write(root, TASK, header="version: 2\ntasks:\n")
        load_manifest(root / "gpu-tasks.yaml", "t", fake_which)
    with pytest.raises(ManifestError, match="exactly version and tasks"):
        write(root, TASK, header="version: 1\nextra: 1\ntasks:\n")
        load_manifest(root / "gpu-tasks.yaml", "t", fake_which)
    with pytest.raises(ManifestError, match="cannot read"):
        load_manifest(root / "missing.yaml", "t", fake_which)
    broken = TASK.replace("resumable: true", "resumable: no").replace(
        "PASTORE_MODEL", "bad"
    ) + OTHER.replace("2h", "xx")
    with pytest.raises(ManifestError) as info:
        load(root, broken)
    lines = str(info.value).splitlines()
    assert any(line.startswith("bench: resumable") for line in lines)
    assert any(line.startswith("bench: model_env must match") for line in lines)
    assert any(line.startswith("second: duration") for line in lines)


@pytest.mark.parametrize(("value", "seconds"), [(90, 90), ("45s", 45), ("10m", 600), ("2h", 7200)])
def test_parse_duration(value: object, seconds: int) -> None:
    assert parse_duration(value) == seconds


@pytest.mark.parametrize("value", [0, -5, True, 1.5, "0s", "5", "5d", "-1h", "h", None, [1]])
def test_parse_duration_refuses(value: object) -> None:
    with pytest.raises(ManifestError):
        parse_duration(value)


def test_fingerprint_changes_with_the_consumer_root(tmp_path: Path) -> None:
    one = load_manifest(write(make_root(tmp_path / "a"), TASK), "torra", fake_which)
    two = load_manifest(write(make_root(tmp_path / "b"), TASK), "torra", fake_which)
    assert one["bench"].fingerprint != two["bench"].fingerprint
