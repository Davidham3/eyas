"""Documentation-discipline gates over the shipped tree (zero GPU, zero deps).

Two invariants that keep in-code prose reviewable as the tree evolves:

1. every path token a module docstring names (``src/``, ``configs/``, …) must
   still resolve on disk — the cheap mechanical half of guarding long docstrings
   against silent rot; prose claims beyond paths stay a review responsibility.
2. every ``arXiv:`` citation carries a venue and year on the citing line (or its
   immediate wrap) — a bare identifier is not reviewer-verifiable offline, and a
   wrong venue is worse than none.

Both gates are intentionally dumb and total: no allowlists, no exemptions.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_PATH_TOKEN = re.compile(
    r"(?<![\w/.-])(?:src|tests|configs|scripts|docs|examples)/[A-Za-z0-9_./-]*[A-Za-z0-9_]"
)
_VENUE = re.compile(r"(?:ICLR|NeurIPS|ICML|EMNLP|ACL|AAAI|NAACL|COLM|TMLR)\D{0,12}\d{4}")


def _module_docstrings() -> list[tuple[Path, str]]:
    rows = []
    for root in (REPO / "src", REPO / "scripts"):
        for py in sorted(root.rglob("*.py")):
            doc = ast.get_docstring(ast.parse(py.read_text(encoding="utf-8")), clean=True)
            rows.append((py, doc or ""))
    return rows


def test_module_docstring_paths_resolve() -> None:
    broken = []
    for py, doc in _module_docstrings():
        for token in _PATH_TOKEN.findall(doc):
            token = token.rstrip(".,;:")
            if "*" in token:
                continue  # glob shapes are prose, not claims about one path
            if not (REPO / token).exists():
                broken.append(f"{py.relative_to(REPO)}: {token}")
    assert not broken, "module docstrings name paths that do not exist:\n" + "\n".join(broken)


def test_arxiv_citations_carry_venue() -> None:
    offenders = []
    for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for py in sorted(root.rglob("*.py")):
            if py == Path(__file__):
                continue  # this file quotes the literal token in its own contract
            lines = py.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if "arXiv:" not in line:
                    continue
                window = line + (lines[i + 1] if i + 1 < len(lines) else "")
                if not _VENUE.search(window):
                    offenders.append(f"{py.relative_to(REPO)}:{i + 1}: {line.strip()}")
    assert not offenders, (
        "arXiv citations without venue+year on the citing line/wrap:\n" + "\n".join(offenders)
    )
