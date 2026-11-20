"""Repository-hygiene gate: gitignore must never swallow repository content.

Regression post-mortem: an unanchored ``runs/`` rule in ``.gitignore`` (meant
for the root-level runtime output directory) also matched ``tests/runs/``, so
a directory-style ``git add tests/runs/`` silently skipped all ten test files
— git does not warn when an add hits an ignore rule. The suite stayed green
everywhere because pytest reads the filesystem, not the index; only
cloning the published repository from a fresh checkout exposed the loss.

These checks run wherever the checkout is a git working tree (local and CI)
and skip otherwise (sdist unpacks):

1. no tracked file may match an ignore rule (``git ls-files -i -c``);
2. no on-disk test file under ``tests/`` or ``examples/`` may be ignored,
   untracked or not — the exact failure mode above, caught at add time
   instead of clone time;
3. the runtime ``runs/`` output directory stays ignored via the root-anchored
   ``/runs/`` rule, so re-anchoring mistakes in either direction are caught.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), "-c", "core.quotePath=false", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _inside_git_worktree() -> bool:
    proc = _git("rev-parse", "--is-inside-work-tree")
    return proc.returncode == 0 and proc.stdout.strip() == "true"


pytestmark = pytest.mark.skipif(
    not _inside_git_worktree(), reason="not a git working tree (e.g. sdist unpack)"
)


def test_no_tracked_file_is_ignored() -> None:
    """A file that is tracked yet ignored means a rule was added after the
    fact, or ``git add -f`` hid a conflict — either way it must be resolved."""
    proc = _git("ls-files", "-i", "-c", "--exclude-standard")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == []


@pytest.mark.parametrize("root", ["tests", "examples"])
def test_no_test_file_is_ignored(root: str) -> None:
    """Every ``test_*.py`` on disk under ``root`` must be addable: the
    directory-add silent-skip that lost ``tests/runs/`` fails here instead."""
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--others", "--cached", root],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    candidates = [
        rel for rel in listing if Path(rel).name.startswith("test_") and rel.endswith(".py")
    ]
    if not candidates:
        pytest.skip(f"no test files listed under {root}/")
    probe = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--stdin"],
        input="\n".join(candidates) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.stdout.splitlines() == []


def test_runtime_runs_directory_stays_ignored(tmp_path: Path) -> None:
    """``/runs/`` (root-anchored) must still ignore the root-level runtime
    output directory while leaving ``tests/runs/`` addable."""
    probe = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", "--stdin"],
        input="runs/record.sqlite\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, "runs/ at the repo root must stay ignored"
