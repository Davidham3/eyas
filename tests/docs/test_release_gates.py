"""Release-gate scripts: the tag/version alignment gate before publish.

Covers ``scripts/check_tag_version.py`` — exact-string alignment after
stripping ``refs/tags/`` and one leading ``v`` (no PEP 440 normalization:
the release tag must match the pyproject version character for character),
the script's read of the real pyproject, and the CLI contract the release
workflow calls (exit 0 on match / 1 on mismatch, both values on stderr).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_tag_version.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_tag_version", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("tag", "version", "expected"),
    [
        ("v0.1.0rc1", "0.1.0rc1", True),
        ("0.1.0", "0.1.0", True),  # a missing v prefix is tolerated
        ("refs/tags/v0.1.0", "0.1.0", True),  # the ref shape GitHub Actions passes
        ("v0.1.0", "0.1.0rc1", False),
        # dirty inputs are NOT normalized: case and whitespace must match exactly
        ("v0.1.0", " 0.1.0", False),
        ("v0.1.0", "0.1.0 ", False),
        ("V0.1.0rc1", "0.1.0rc1", False),
    ],
)
def test_version_matches(tag: str, version: str, expected: bool) -> None:
    assert _load().version_matches(tag, version) is expected


def test_real_pyproject_version_round_trips() -> None:
    """The script reads the shipped pyproject: version 0.1.0rc1, and its
    tag form aligns with itself."""
    mod = _load()
    version = mod.pyproject_version()
    assert version == "0.1.0rc1"
    assert mod.version_matches("v" + version, version) is True


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_exit_zero_on_match() -> None:
    proc = _cli("v0.1.0rc1")
    assert proc.returncode == 0, proc.stderr


def test_cli_exit_one_and_both_values_on_stderr_when_mismatched() -> None:
    proc = _cli("v0.1.0")
    assert proc.returncode == 1
    assert "v0.1.0" in proc.stderr
    assert "0.1.0rc1" in proc.stderr


def test_cli_requires_exactly_one_tag_argument() -> None:
    for args in ((), ("v0.1.0rc1", "extra")):
        proc = _cli(*args)
        assert proc.returncode == 2, proc.stdout + proc.stderr
