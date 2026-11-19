#!/usr/bin/env python3
"""Gate the release workflow on tag/version alignment.

why: a tag-driven publish must never ship a wheel whose version differs from
the tag users will cite. ``uv build`` derives the version from
``pyproject.toml``; this gate refuses the release when the pushed tag and that
declared version disagree, so a stale or mistyped tag fails before upload.

what: compares the tag with ``[project].version`` by exact string equality
after stripping a ``refs/tags/`` prefix and one leading ``v`` — the release tag
is required to match the version character for character, so no PEP 440
normalization is applied. Exit 0 on match; exit 1 on mismatch (both values on
stderr); exit 2 when argv is not exactly one tag.

Usage (repo root):
    uv run python scripts/check_tag_version.py <tag>
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"


def version_matches(tag: str, pyproject_version: str) -> bool:
    """True when the tag names exactly this version.

    Accepts a full ref (``refs/tags/v1.2.3``) and tolerates a missing ``v``
    prefix (``1.2.3``); the remaining text must equal ``pyproject_version``
    verbatim — no case folding, whitespace stripping or version parsing.
    """
    ref = tag.removeprefix("refs/tags/").removeprefix("v")
    return ref == pyproject_version


def pyproject_version(path: Path = PYPROJECT) -> str:
    """Read ``[project].version`` from a pyproject file (stdlib tomllib)."""
    with path.open("rb") as fh:
        project = tomllib.load(fh)["project"]
    return str(project["version"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a release tag against pyproject version.")
    parser.add_argument("tag", help="release tag, e.g. v0.1.0rc1 or refs/tags/v0.1.0rc1")
    args = parser.parse_args(argv)

    version = pyproject_version()
    if version_matches(args.tag, version):
        return 0
    print(
        f"tag {args.tag!r} does not match pyproject version {version!r}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
