"""Versioned-docs contract: final-release whitelist, wired selector, pinned tool."""

from __future__ import annotations

import ast
import importlib.util
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


def _redirect_mod():
    path = REPO / "scripts" / "render_docs_redirects.py"
    spec = importlib.util.spec_from_file_location("render_docs_redirects", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_release_tag_sorts_numerically_and_skips_rc() -> None:
    mod = _redirect_mod()
    assert mod.latest_release_tag(["v0.9.0", "v0.10.0", "v0.1.0rc1", "main"]) == "v0.10.0"
    assert mod.latest_release_tag(["v0.1.0rc1", "main"]) is None


def test_write_redirects_points_root_at_stable(tmp_path: Path) -> None:
    mod = _redirect_mod()
    mod.write_redirects(tmp_path, "v1.2.3")
    root = (tmp_path / "index.html").read_text(encoding="utf-8")
    stable = (tmp_path / "stable" / "index.html").read_text(encoding="utf-8")
    assert "stable/index.html" in root
    assert "../v1.2.3/" in stable
    assert (tmp_path / ".nojekyll").is_file()


def test_write_redirects_falls_back_to_main(tmp_path: Path) -> None:
    mod = _redirect_mod()
    mod.write_redirects(tmp_path, None)
    stable = (tmp_path / "stable" / "index.html").read_text(encoding="utf-8")
    assert "../main/" in stable


def test_docs_workflow_builds_every_version() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/docs.yml").read_text(encoding="utf-8"))
    steps = [step for job in doc["jobs"].values() for step in job.get("steps", [])]
    checkout = next(step for step in steps if "actions/checkout@" in step.get("uses", ""))
    assert checkout["with"]["fetch-depth"] == 0
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "sphinx-multiversion docs site" in runs
    assert "scripts/render_docs_redirects.py site" in runs
    triggers = doc.get("on", doc.get(True))
    assert triggers["release"]["types"] == ["published"]
