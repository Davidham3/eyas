"""Write the GitHub Pages root redirect and the mutable ``stable`` alias.

The versioned build produces ``site/<ref>/`` directories (``main`` plus one per
final release tag). GitHub Pages needs an ``index.html`` at the site root, and
users want a stable URL that always points at the newest release.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

_RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

_ROOT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=stable/index.html">
  <link rel="canonical" href="stable/index.html">
  <title>Eyas documentation</title>
</head>
<body><a href="stable/index.html">Eyas documentation</a></body>
</html>
"""


def latest_release_tag(tags: Iterable[str]) -> str | None:
    """The highest ``vX.Y.Z`` tag by numeric components, ignoring rc tags."""
    best: tuple[tuple[int, int, int], str] | None = None
    for tag in tags:
        match = _RELEASE_TAG.match(tag)
        if match is None:
            continue
        key = tuple(int(part) for part in match.groups())
        if best is None or key > best[0]:
            best = (key, tag)
    return None if best is None else best[1]


def write_redirects(site: Path, latest: str | None) -> None:
    """Root -> ``stable/``; ``stable/`` -> ``../<latest>/`` or ``../main/``."""
    site.mkdir(parents=True, exist_ok=True)
    (site / ".nojekyll").touch()
    (site / "index.html").write_text(_ROOT, encoding="utf-8")
    target = f"../{latest}/" if latest is not None else "../main/"
    stable = site / "stable"
    stable.mkdir(parents=True, exist_ok=True)
    (stable / "index.html").write_text(
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f'  <meta http-equiv="refresh" content="0; url={target}">\n'
        f'  <link rel="canonical" href="{target}">\n'
        "  <title>Eyas documentation</title>\n"
        "</head>\n"
        f'<body><a href="{target}">Eyas documentation</a></body>\n'
        "</html>\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    site = Path(args[0] if args else "site")
    tags = subprocess.run(
        ["git", "tag", "-l"], check=True, capture_output=True, text=True
    ).stdout.split()
    latest = latest_release_tag(tags)
    write_redirects(site, latest)
    print(f"docs redirects: / -> stable/ -> {latest or 'main'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
