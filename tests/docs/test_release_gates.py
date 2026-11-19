"""Release-gate scripts: the tag/version alignment gate before publish.

Covers ``scripts/check_tag_version.py`` — exact-string alignment after
stripping ``refs/tags/`` and one leading ``v`` (no PEP 440 normalization:
the release tag must match the pyproject version character for character),
the script's read of the real pyproject, and the CLI contract the release
workflow calls (exit 0 on match / 1 on mismatch, both values on stderr).

The appended section pins the ``.github/workflows/release.yml`` contract:
one build feeding a TestPyPI rehearsal, a three-OS smoke matrix, the
approval-gated PyPI publish, and the draft GitHub Release.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

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
    """The script reads the shipped pyproject, and its tag form aligns with
    itself — version-agnostic so the documented 0.1.0 bump cannot rot it."""
    mod = _load()
    version = mod.pyproject_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", version), version
    assert mod.version_matches("v" + version, version) is True


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_exit_zero_on_match() -> None:
    version = _load().pyproject_version()
    proc = _cli("v" + version)
    assert proc.returncode == 0, proc.stderr


def test_cli_exit_one_and_both_values_on_stderr_when_mismatched() -> None:
    version = _load().pyproject_version()
    tag = "v" + version + ".post999"  # guaranteed mismatch at any current version
    proc = _cli(tag)
    assert proc.returncode == 1
    # both values quoted and reported, not just the tag
    assert f"'{tag}'" in proc.stderr
    assert f"'{version}'" in proc.stderr


def test_cli_requires_exactly_one_tag_argument() -> None:
    for args in ((), ("v0.1.0rc1", "extra")):
        proc = _cli(*args)
        assert proc.returncode == 2, proc.stdout + proc.stderr


# --- release workflow contract -------------------------------------------------

WORKFLOW = REPO / ".github" / "workflows" / "release.yml"
JOB_ORDER = ["build", "testpypi", "smoke", "pypi", "github-release"]
PRERELEASE_PATTERN = r"(a|b|rc)[0-9]+"
GATE_COMMAND = 'uv run python scripts/check_tag_version.py "${GITHUB_REF_NAME}"'

# The one moving ref the release pipeline is allowed: the trusted-publishing
# action only publishes branches/tags of its own repo, so release/v1 is stable.
ALLOWED_MOVING_ACTION = "pypa/gh-action-pypi-publish@release/v1"
MOVING_REF = re.compile(r"@(main|master|v1|release/v1)$")


def _is_moving_ref(ref: str) -> bool:
    return MOVING_REF.search(ref) is not None


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(doc: dict, job: str) -> list[dict]:
    return doc["jobs"][job]["steps"]


def _job_runs(doc: dict, job: str) -> str:
    return "\n".join(step["run"] for step in _steps(doc, job) if "run" in step)


def _has_uses(step: dict, prefix: str) -> bool:
    return step.get("uses", "").startswith(prefix)


def _true(value: object) -> bool:
    """YAML 1.1 loads unquoted true/false as booleans; tolerate string spellings."""
    return value is True or value == "true"


def test_workflow_is_triggered_by_version_tags_only() -> None:
    doc = _workflow()
    # YAML 1.1 resolves the bare key `on` to boolean True.
    triggers = doc["on"] if "on" in doc else doc[True]
    assert triggers == {"push": {"tags": ["v*"]}}


def test_job_graph_is_the_linear_publish_chain() -> None:
    jobs = _workflow()["jobs"]
    assert list(jobs) == JOB_ORDER
    assert jobs["build"].get("needs") is None
    edges = {job: jobs[job].get("needs") for job in JOB_ORDER[1:]}
    assert edges == dict(zip(JOB_ORDER[1:], JOB_ORDER[:-1], strict=True))


def test_release_runs_are_never_cancelled_mid_flight() -> None:
    concurrency = _workflow()["concurrency"]
    assert concurrency
    if isinstance(concurrency, dict):
        assert concurrency["group"] == "release-${{ github.ref }}"
        assert concurrency.get("cancel-in-progress") is not True


@pytest.mark.parametrize(
    ("job", "name", "url"),
    [
        ("testpypi", "testpypi", "https://test.pypi.org/p/eyas"),
        ("pypi", "release", "https://pypi.org/p/eyas"),
    ],
)
def test_publish_environments_name_their_target_index(job: str, name: str, url: str) -> None:
    environment = _workflow()["jobs"][job]["environment"]
    assert environment["name"] == name
    assert environment["url"] == url


def test_version_gate_step_calls_the_alignment_script() -> None:
    gate = [step for step in _steps(_workflow(), "build") if step.get("name") == "Version gate"]
    assert len(gate) == 1
    assert gate[0]["run"].strip() == GATE_COMMAND


def test_build_job_syncs_then_gates_then_builds_and_checks() -> None:
    runs = [step.get("run", "") for step in _steps(_workflow(), "build")]
    needles = ("uv sync", "check_tag_version", "uv build", "twine check")
    positions = [next(i for i, run in enumerate(runs) if needle in run) for needle in needles]
    assert positions == sorted(set(positions))


def test_distributions_are_built_and_checked_exactly_once() -> None:
    doc = _workflow()
    all_runs = "\n".join(_job_runs(doc, job) for job in JOB_ORDER)
    assert all_runs.count("uv build") == 1
    assert all_runs.count("twine check") == 1


def test_build_uploads_the_single_python_dist_artifact() -> None:
    uploads = [
        step for step in _steps(_workflow(), "build") if _has_uses(step, "actions/upload-artifact")
    ]
    assert [step["with"]["name"] for step in uploads] == ["python-dist"]
    assert [step["with"]["path"] for step in uploads] == ["dist/"]


def test_publish_consumers_download_the_same_artifact() -> None:
    doc = _workflow()
    for job in ("testpypi", "pypi", "github-release"):
        downloads = [
            step for step in _steps(doc, job) if _has_uses(step, "actions/download-artifact")
        ]
        assert [step["with"]["name"] for step in downloads] == ["python-dist"], job
        # the same dist/ layout the publishers read
        assert [step["with"]["path"] for step in downloads] == ["dist/"], job
    # The smoke matrix installs from TestPyPI, so it must not smuggle a local dist.
    assert not any(_has_uses(step, "actions/download-artifact") for step in _steps(doc, "smoke"))


def test_smoke_matrix_covers_every_supported_os_and_python() -> None:
    job = _workflow()["jobs"]["smoke"]
    assert job["runs-on"] == "${{ matrix.os }}"
    strategy = job["strategy"]
    assert strategy["fail-fast"] is False
    assert strategy["matrix"]["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert strategy["matrix"]["python-version"] == ["3.12", "3.13", "3.14"]
    setup = next(step for step in job["steps"] if _has_uses(step, "actions/setup-python"))
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"


def test_smoke_installs_the_rehearsed_extra_from_testpypi() -> None:
    runs = _job_runs(_workflow(), "smoke")
    assert "--index-url https://test.pypi.org/simple/" in runs
    assert "--extra-index-url https://pypi.org/simple/" in runs
    assert "eyas[api]==" in runs
    assert 'python -c "import eyas"' in runs
    assert "eyas --help" in runs
    install = next(
        step
        for step in _steps(_workflow(), "smoke")
        if step.get("name") == "Install the published wheel from TestPyPI"
    )
    # one shell across the three OSes: the multi-line pip invocation needs bash
    assert install["shell"] == "bash"


def test_only_the_github_release_job_may_write_contents() -> None:
    doc = _workflow()
    writers = [
        job
        for job in JOB_ORDER
        if doc["jobs"][job].get("permissions", {}).get("contents") == "write"
    ]
    assert writers == ["github-release"]
    # A workflow-wide default must not grant contents: write either.
    assert doc.get("permissions", {}).get("contents") != "write"


def test_publish_jobs_use_trusted_publishing_oidc() -> None:
    doc = _workflow()
    publishers = [
        job
        for job in JOB_ORDER
        if any(_has_uses(step, "pypa/gh-action-pypi-publish") for step in _steps(doc, job))
    ]
    assert publishers == ["testpypi", "pypi"]
    oidc_writers = [
        job
        for job in JOB_ORDER
        if doc["jobs"][job].get("permissions", {}).get("id-token") == "write"
    ]
    assert oidc_writers == publishers


def _publisher(doc: dict, job: str) -> dict:
    return next(step for step in _steps(doc, job) if _has_uses(step, "pypa/gh-action-pypi-publish"))


def test_testpypi_publishes_to_the_rehearsal_index_with_skip_existing() -> None:
    """The highest-consequence line in the workflow: the rehearsal must never
    touch the real index, and a re-run must tolerate an already-published file."""
    with_ = _publisher(_workflow(), "testpypi")["with"]
    assert with_["repository-url"] == "https://test.pypi.org/legacy/"
    assert _true(with_["skip-existing"])


def test_pypi_publisher_uses_the_real_index() -> None:
    with_ = _publisher(_workflow(), "pypi").get("with", {})
    assert "test.pypi.org" not in str(with_)
    assert with_.get("repository-url") in (None, "https://upload.pypi.org/legacy/")


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.1.0rc1", True),
        ("v0.1.0a1", True),
        ("v0.1.0b1", True),
        ("v0.1.0-rc1", True),
        ("v0.1.0", False),
    ],
)
def test_prerelease_pattern_classifies_tags(tag: str, expected: bool) -> None:
    assert (re.search(PRERELEASE_PATTERN, tag) is not None) is expected


def test_workflow_derives_prerelease_from_the_tested_pattern() -> None:
    doc = _workflow()
    assert PRERELEASE_PATTERN in _job_runs(doc, "github-release")
    release = next(
        step
        for step in _steps(doc, "github-release")
        if _has_uses(step, "softprops/action-gh-release")
    )
    assert "steps.meta.outputs.prerelease" in release["with"]["prerelease"]


def test_github_release_is_a_draft_with_matched_files_and_install_hint() -> None:
    release = next(
        step
        for step in _steps(_workflow(), "github-release")
        if _has_uses(step, "softprops/action-gh-release")
    )
    assert release["uses"] == "softprops/action-gh-release@v2"
    with_ = release["with"]
    assert _true(with_["draft"])
    assert _true(with_["fail_on_unmatched_files"])
    assert with_["files"].split() == ["dist/*.whl", "dist/*.tar.gz"]
    assert 'uv pip install "eyas[api]==' in with_["body"]
    assert "CHANGELOG" in with_["body"]


def test_action_references_are_immutably_pinned() -> None:
    doc = _workflow()
    uses = [
        step["uses"]
        for job in doc["jobs"].values()
        for step in job.get("steps", [])
        if "uses" in step
    ]
    moving = {ref for ref in uses if _is_moving_ref(ref) and ref != ALLOWED_MOVING_ACTION}
    assert moving == set()
    # the allowlist branch must be reachable: the detector classifies every
    # moving shape (the exemption, not a regex miss, is why release/v1 passes)
    assert _is_moving_ref(ALLOWED_MOVING_ACTION)
    for ref in ("main", "master", "v1", "release/v1"):
        assert _is_moving_ref(f"owner/action@{ref}")
    assert not _is_moving_ref("owner/action@v10")


def test_setup_uv_is_pinned_to_the_lockfile_interpreter_tool() -> None:
    doc = _workflow()
    pins = {
        step["with"]["version"]
        for job in doc["jobs"].values()
        for step in job.get("steps", [])
        if _has_uses(step, "astral-sh/setup-uv@")
    }
    assert pins == {"0.11.21"}
