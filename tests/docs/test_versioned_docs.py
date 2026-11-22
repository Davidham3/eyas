"""Versioned-docs contract: final-release whitelist, wired selector, pinned tool."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONF = REPO / "docs" / "conf.py"
TEMPLATE = REPO / "docs" / "_templates" / "versioning.html"


def _conf_assignments() -> dict[str, object]:
    tree = ast.parse(CONF.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return values


def test_multiversion_extension_and_whitelists() -> None:
    conf = _conf_assignments()
    assert "sphinx_multiversion" in conf["extensions"]
    tag_pattern = conf["smv_tag_whitelist"]
    assert re.match(tag_pattern, "v0.1.0")
    assert not re.match(tag_pattern, "v0.1.0rc1")
    assert conf["smv_branch_whitelist"] == r"^main$"
    released = conf["smv_released_pattern"]
    assert re.match(released, "refs/tags/v0.1.0")
    assert not re.match(released, "refs/heads/main")


def test_version_selector_template_is_wired() -> None:
    conf = _conf_assignments()
    assert TEMPLATE.is_file()
    sidebars = conf["html_sidebars"]["**"]
    assert "versioning.html" in sidebars
    assert "sbt-sidebar-nav.html" in sidebars
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "versions.releases" in text
    assert "window.location" in text


def test_ci_docs_job_installs_the_pinned_versioned_docs_tool() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    runs = "\n".join(step.get("run", "") for step in doc["jobs"]["docs"]["steps"])
    assert "sphinx-multiversion-scylla==0.3.9" in runs
