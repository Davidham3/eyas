"""Docs reference integrity: every relative markdown link resolves on disk, and
every documentation page is wired into a myst toctree.

Split deliberately from the sphinx build itself (CI docs job): sphinx only sees
files under ``docs/``, so links reaching into ``examples/`` — legitimate on
GitHub's web renderer — are unresolvable to it; the invariant that actually
matters to readers is that **the target file exists in the repo**, checked here.
And a page missing from every toctree silently disappears from the site;
strict-mode builds would catch it, strict-mode also dies on the examples/ links,
so toctree coverage is asserted directly here instead.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# never link/nav-scan: disposable scratch (drafts with placeholder URLs, local
# copies), hidden tool/agent directories (dot dirs, except `.github`, which is
# ordinary public repo metadata), and the raw scraped provenance archive (its
# links are site-root paths captured from the source pages, not repo paths)
SKIP_DIRS = {"outputs", "node_modules", "tmp", "reference/apo-series/research"}

_LINK = re.compile(r"\]\(([^)\s]+)\)")
_TOCTREE = re.compile(r"^```\{toctree\}\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _tracked_markdown() -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        rel_dir = Path(dirpath).relative_to(REPO).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir

        # default-arg binds the current rel_dir (B023: the filter runs per-iteration)
        def kept(d: str, rel_dir: str = rel_dir) -> bool:
            if d in {".git", "__pycache__", ".venv", "tmp", "_build"} or (
                d.startswith(".") and d != ".github"
            ):
                return False
            full = f"{rel_dir}/{d}" if rel_dir else d
            return not any(full == skip or full.startswith(skip + "/") for skip in SKIP_DIRS)

        dirnames[:] = [d for d in dirnames if kept(d)]
        out.extend(Path(dirpath) / n for n in filenames if n.endswith(".md"))
    return out


def test_relative_markdown_links_exist_on_disk():
    broken: list[str] = []
    for md in _tracked_markdown():
        for target in _LINK.findall(md.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = (md.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(REPO)} -> {target}")
    assert not broken, "relative markdown links must point at files that exist:\n" + "\n".join(
        broken
    )


def _toctree_pages() -> set[str]:
    pages: set[str] = set()
    for md in (REPO / "docs").rglob("*.md"):
        for block in _TOCTREE.findall(md.read_text(encoding="utf-8")):
            for line in block.splitlines():
                entry = line.strip()
                if not entry or entry.startswith(":"):
                    continue
                pages.add(entry.removesuffix(".md"))
    return pages


def test_toctree_entries_point_at_existing_pages():
    missing = [p for p in sorted(_toctree_pages()) if not (REPO / "docs" / f"{p}.md").is_file()]
    assert not missing, f"toctree references non-existent pages: {missing}"


def test_every_docs_page_is_in_a_toctree():
    pages = {
        md.relative_to(REPO / "docs").with_suffix("").as_posix()
        for md in (REPO / "docs").rglob("*.md")
        if "_build" not in md.parts
    }
    pages.discard("index")
    unlisted = sorted(pages - _toctree_pages())
    assert not unlisted, (
        f"docs pages missing from every toctree (they would vanish from the site): {unlisted}"
    )
