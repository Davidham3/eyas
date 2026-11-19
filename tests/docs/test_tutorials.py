"""Tutorial-command gate (the d2l discipline, sized for a niche framework).

Every `````bash`` block in ``docs/tutorials/*.md`` and ``README.md`` must be
classified here: either **RUN** (executed verbatim by this test, so the docs
can never rot), **RUN_STUB** (executed against an in-process stdlib stub of
the OpenAI API — the only edit is rewriting the canonical docs URL literal to
the stub's ephemeral port), **FAIL** (executed too, asserting the documented
error), or **SKIP** (with a recorded reason — placeholder paths, GPU
specs, heavy installs). Adding a command block to the tutorials without
classifying it fails this test: the classification map is a completeness
contract, not a suggestion.

Execution rules:
- blocks run with ``cwd=REPO`` (the docs assume repo-relative config paths);
- ``uv run …`` is rewritten to this interpreter's console script — same entry
  point, but the gate never re-locks (a local uv.toml mirror would otherwise
  rewrite uv.lock origins as a side effect);
- blocks that create repo-relative ``outputs/`` dirs: only the run dirs this
  execution newly created are removed afterwards (gitignored anyway, but the
  suite must not litter).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.support.stub_openai_server import StubOpenAIServer

REPO = Path(__file__).resolve().parents[2]
_DOC_FILES = sorted((REPO / "docs" / "tutorials").glob("*.md")) + [REPO / "README.md"]

# key = "<file-stem>#<block-index-in-order>". Keep the maps disjoint and
# together exhaustive over every ```bash block in the docs (asserted below).
RUN: dict[str, str] = {
    # guided-tour §8's export half: gen-configs assembles two published cells into
    # the gitignored tmp/ — zero GPU, zero data, executed verbatim (the GENERATED
    # header + why-notes rendering is the live contract)
    "guided-tour#7": "gen-configs export of pets raw+cot cells to gitignored tmp/ + GENERATED header check",
}
# blocks documented to FAIL: executed too, asserting non-zero exit AND the
# documented error text — fail-fast promises in the docs are as live as the
# success paths
FAIL: dict[str, str] = {
    "guided-tour#1": "documents --set refusing to invent structure; error text pinned verbatim",
}
SKIP: dict[str, str] = {
    "getting-started#0": "git clone of the public repo https://github.com/Davidham3/eyas.git: needs network, skipped (install walkthrough)",
    "getting-started#1": "downloads TextVQA images (5001-question labeled val pool, ~3 parquet shards of ~310 MB)",
    "getting-started#2": "downloads TextVQA images + user-supplied OpenAI endpoint; config shape pinned by tests/configs/test_configs.py, story numbers from the recorded fresh OpenAI-endpoint run on the page",
    "getting-started#3": "downloads TextVQA images + user-supplied OpenAI endpoint; config shape pinned by tests/configs/test_configs.py, story numbers from the recorded fresh OpenAI-endpoint run on the page",
    "first-optimization#0": "assembled story cells need TextVQA data + user-supplied OpenAI endpoint; the same two commands are classified at getting-started#2/#3 and the config shapes are pinned by tests/configs/test_configs.py",
    "first-optimization#1": "gen-configs export of the story cell, cp'd to a reader-authored your_hill_climb.yaml (walkthrough placeholder, not a tree artifact)",
    "first-optimization#2": "your_hill_climb.yaml: placeholder authored during the walkthrough (needs endpoint + TextVQA data)",
    "first-optimization#3": "DEBUG/status example over the placeholder your_hill_climb.yaml",
    "custom-method#0": "rs_demo.yaml is reader-authored in the custom-method tutorial; the shipped equivalent (examples/strategy_skeleton) is pytest-gated there",
    "custom-task#0": "my_config.yaml: placeholder authored during the walkthrough",
    "understanding-results#0": "exp.yaml: placeholder; --resume fidelity is pinned by tests/runs/test_resume.py",
    "understanding-results#1": "exp.yaml: placeholder; --set semantics are pinned by tests/configs/test_runtime_plugins.py",
    "understanding-results#2": "illustrative run-dir path from a user's own outputs/",
    "benchmark#0": "installs the vllm extra and downloads datasets — CI jobs do this on the GPU tier only",
    "benchmark#1": "quickstart cell needs model weights + dataset (GPU tier)",
    "benchmark#2": "same GPU-tier config as benchmark#1",
    "benchmark#3": "gen-configs export + sweep over GPU-tier cells",
    "benchmark#4": "benchmark-spec sweep: GPU-tier assembled cells + real weights path via --set",
    "benchmark#5": "reports over outputs/ produced by benchmark#1..4 (GPU tier)",
    "guided-tour#4": "reads the run dirs produced by guided-tour#0/#3 (cross-block state); the multi-run report shape is pinned by tests/engine/test_runtime.py and the pasted table comes from a real stub-server execution",
    "guided-tour#6": "the published-cell address runs real data + real weights (GPU tier); the assembled pets/raw cell is golden-pinned by tests/configs/test_configspec.py",
    "README#0": "uv pip install from PyPI: needs network and mutates the environment; the wheel-install contract is exercised by the release preflight, not the docs gate",
    "README#2": "downloads TextVQA images (5001-question labeled val pool, ~3 parquet shards of ~310 MB) and needs the prepare extra",
    "README#3": "needs the TextVQA splits built by README#2 plus a user-supplied OpenAI-compatible endpoint; the inline config mirrors the shipped textvqa/opro openai variant, whose backend is golden-pinned by tests/configs/test_configspec.py",
}

# documented stderr signatures for the FAIL blocks
FAIL_SIGNATURES: dict[str, str] = {
    "guided-tour#1": "is not a mapping",
}

# blocks executed against the in-repo stdlib stub of the OpenAI API (Task 1's
# StubOpenAIServer): the ONLY permitted edit is rewriting STUB_URL_LITERAL to
# the stub's ephemeral base URL — everything else runs verbatim. The rewrite
# reaches inside referenced config files too: a block without an in-block
# literal is repointed at tmp-dir copies of its ``--config`` files with the
# literal rewritten inside the copies (the configs in the tree are never
# touched).
RUN_STUB: dict[str, str] = {
    "README#1": "install smoke test: a heredoc-written zero-data demo_shapes config (URL literal in the block) + one raw evaluation, all against the stub",
    "guided-tour#0": "the tour's opening run: URL literal lives inside configs/tutorial_shapes.yaml, rewritten inside the config copy the block is repointed at",
    "guided-tour#2": "reader-edited config written in-block (literal in the heredoc) + final_eval run + md report, all against the stub",
    "guided-tour#3": "two-seed replication via --set on an existing key (URL literal via the referenced config's copy)",
    "guided-tour#5": "fresh run then --resume of its finished dir: the resume replays the record store at zero new requests, all against the stub",
}
# canonical OpenAI-compatible base URL as printed in the docs; a reader with a
# real local server on :8000 can run a RUN_STUB block unchanged
STUB_URL_LITERAL = "http://localhost:8000/v1"


def _blocks(path: Path) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def _all_keys() -> set[str]:
    keys: set[str] = set()
    for path in _DOC_FILES:
        for i in range(len(_blocks(path))):
            keys.add(f"{path.stem}#{i}")
    return keys


def test_classification_is_complete_and_disjoint():
    keys = _all_keys()
    classified = set(RUN) | set(SKIP) | set(FAIL) | set(RUN_STUB)
    assert keys == classified, (
        "every ```bash block in docs/tutorials/*.md and README.md must be classified "
        f"in tests/docs/test_tutorials.py — unclassified: {sorted(keys - classified)}, "
        f"stale entries: {sorted(classified - keys)}"
    )
    maps = [RUN, SKIP, FAIL, RUN_STUB]
    for i, a in enumerate(maps):
        for b in maps[i + 1 :]:
            assert not (set(a) & set(b)), f"classification maps overlap: {sorted(set(a) & set(b))}"
    assert set(FAIL) == set(FAIL_SIGNATURES)


def test_stub_url_rewrite_requires_literal_hit():
    with pytest.raises(AssertionError, match="no stub URL literal"):
        _rewrite_stub_url("eyas run --config x.yaml", "http://127.0.0.1:9/v1")


def test_stub_url_rewrite_replaces_every_literal():
    block = f'BASE="{STUB_URL_LITERAL}"\ncurl {STUB_URL_LITERAL}/chat/completions'
    rewritten = _rewrite_stub_url(block, "http://127.0.0.1:9/v1")
    assert STUB_URL_LITERAL not in rewritten
    assert rewritten.count("http://127.0.0.1:9/v1") == 2


def test_run_stub_block_rewrites_url_and_reaches_stub_server():
    """End-to-end exercise of the RUN_STUB machinery with a synthetic block
    (no doc block is registered yet): the stub URL literal must be rewritten
    to the ephemeral port and actually served, and EYAS_DOC_STUB_KEY must be
    visible to the block."""
    block = f"""\
python - <<'PY'
import json, os, urllib.request
url = "{STUB_URL_LITERAL}/chat/completions"
body = json.dumps({{"messages": [{{"role": "user", "content": "A) x\\nB) y"}}]}}).encode()
resp = urllib.request.urlopen(url, data=body)
print(resp.read().decode(), os.environ["EYAS_DOC_STUB_KEY"])
PY
"""
    rc, stdout, stderr, served = _run_stub_block(block)
    assert rc == 0, f"block failed (rc={rc}):\n--- stderr ---\n{stderr[-2000:]}"
    assert served == 1, "the rewritten URL never reached the stub server"
    assert "Answer: A" in stdout and "stub-key" in stdout


def test_stub_url_rewrite_repoints_referenced_config(tmp_path):
    """Literal lives in the referenced config, not the block: the config is
    copied into the tmp dir with the literal rewritten inside the copy, the
    block is repointed at the copy, and the file in the tree stays untouched."""
    src = tmp_path / "cfg.yaml"
    src.write_text(f"backend:\n  params:\n    base_url: {STUB_URL_LITERAL}\n", encoding="utf-8")
    block = f"uv run eyas run --config {src} --output-dir /tmp/x"
    copy_dir = tmp_path / "stub-copies"
    rewritten = _rewrite_stub_url(block, "http://127.0.0.1:9/v1", copy_dir)
    assert STUB_URL_LITERAL not in rewritten
    assert str(copy_dir / "cfg.yaml") in rewritten
    assert STUB_URL_LITERAL in src.read_text(encoding="utf-8")  # tree config untouched
    copy = (copy_dir / "cfg.yaml").read_text(encoding="utf-8")
    assert copy.count("http://127.0.0.1:9/v1") == 1


def test_stub_url_rewrite_raises_when_neither_block_nor_config_has_literal(tmp_path):
    """No literal in the block AND none in the referenced config (the mock-backed
    demo fixture): a misclassified RUN_STUB block must fail, not silently pass."""
    src = tmp_path / "demo_cot.yaml"
    shutil.copy(REPO / "tests" / "fixtures" / "demo_cot.yaml", src)
    with pytest.raises(AssertionError, match="no stub URL literal"):
        _rewrite_stub_url(
            f"uv run eyas run --config {src}", "http://127.0.0.1:9/v1", tmp_path / "stub-copies"
        )


def test_run_stub_block_executes_url_from_referenced_config(tmp_path):
    """End-to-end: a real eyas run whose ONLY URL occurrence lives inside the
    public tutorial config — the repointed copy must actually reach the stub."""
    block = (
        f'uv run eyas run --config configs/tutorial_shapes.yaml --output-dir "{tmp_path / "run"}"'
    )
    rc, stdout, stderr, served = _run_stub_block(block, tmp_path / "stub-copies")
    assert rc == 0, f"block failed (rc={rc}):\n--- stderr ---\n{stderr[-2000:]}"
    assert served > 0, "the stub server served no request — the copy rewrite never reached it"


def _rewrite_stub_url(block: str, actual_base: str, tmp_dir: Path | None = None) -> str:
    """Swap the canonical docs URL literal for the stub server's actual base.

    An in-block literal is rewritten in place. A block with no literal may
    carry the URL inside the config it references: every ``--config <path>``
    (or ``--config=<path>``) file is copied into ``tmp_dir``, the copy gets
    the literal rewritten inside it, and the block is repointed at the copy —
    the config in the tree is never touched. Zero replacements anywhere means
    the block was misclassified (or the docs drifted): a RUN_STUB block must
    talk to the documented URL."""
    if STUB_URL_LITERAL in block:
        return block.replace(STUB_URL_LITERAL, actual_base)
    copies: dict[str, Path | None] = {}
    replaced = 0

    def _repoint(match: re.Match[str]) -> str:
        nonlocal replaced
        ref = match.group("path")
        if ref not in copies:
            src = Path(ref)
            src = src if src.is_absolute() else REPO / ref
            copy: Path | None = None
            if src.is_file() and tmp_dir is not None:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                copy = tmp_dir / src.name
                text = src.read_text(encoding="utf-8")
                replaced += text.count(STUB_URL_LITERAL)
                copy.write_text(text.replace(STUB_URL_LITERAL, actual_base), encoding="utf-8")
            copies[ref] = copy
        copy = copies[ref]
        return match.group(0) if copy is None else f"{match.group('lead')}{copy}"

    rewritten = re.sub(r"(?P<lead>--config(?:=|\s+))(?P<path>\S+)", _repoint, block)
    assert replaced > 0, (
        f"RUN_STUB block has no stub URL literal {STUB_URL_LITERAL!r} — neither in the "
        f"block nor in any config it references (misclassified or the docs drifted):\n{block}"
    )
    return rewritten


def _env(**extra: str) -> dict[str, str]:
    env = dict(
        os.environ,
        PATH=f"{Path(sys.executable).parent}{os.pathsep}" + os.environ.get("PATH", ""),
    )
    env.update(extra)
    return env


def _run_stub_block(block: str, tmp_dir: Path | None = None) -> tuple[int, str, str, int]:
    """Execute a RUN_STUB block verbatim against a freshly started stub
    server. Returns (rc, stdout, stderr, served-request-count). ``tmp_dir``
    receives the rewritten copies of any referenced config files."""
    server = StubOpenAIServer()
    base = server.start()
    try:
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="eyas-stub-block-"))
        lines = _executables(_rewrite_stub_url(block, base, tmp_dir))
        assert lines, "block has no executable line"
        # explicit env: both documented key variables are pinned to the stub
        # key, so an inherited real API key can never be sent to the stub
        env = _env(EYAS_DOC_STUB_KEY="stub-key", EYAS_API_KEY="stub-key")
        proc = subprocess.run(
            "\n".join(lines),
            shell=True,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return proc.returncode, proc.stdout, proc.stderr, server.request_count
    finally:
        server.stop()


def test_executables_preserves_heredoc_bodies():
    """Doc blocks may create a file via heredoc (the tour writes its
    reader-edited config that way); heredoc bodies are YAML/Python and must
    keep their indentation, while the uv-rewrite outside keeps working."""
    eyas = str(Path(sys.executable).parent / "eyas")
    block = "cat > /tmp/x.yaml <<'YAML'\nrun:\n  name: x\nYAML\nuv run eyas status /tmp/x\n"
    lines = _executables(block)
    assert lines[1:4] == ["run:", "  name: x", "YAML"]
    assert lines[4].startswith(eyas)


def _executables(block: str) -> list[str]:
    """Rewrite doc-canonical `uv run eyas` to the venv console script (the
    identical entry point without ever invoking uv). Heredoc bodies pass
    through verbatim — stripping them would flatten YAML/Python indentation."""
    eyas = str(Path(sys.executable).parent / "eyas")
    out: list[str] = []
    heredoc_tag: str | None = None
    for line in block.splitlines():
        if heredoc_tag is not None:
            out.append(line)
            if line.strip() == heredoc_tag:
                heredoc_tag = None
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        marker = re.search(r"<<-?\s*'(\w+)'|<<-?\s*(\w+)", stripped)
        if marker:
            heredoc_tag = marker.group(1) or marker.group(2)
        if stripped.startswith("uv run eyas"):
            stripped = eyas + stripped[len("uv run eyas") :]
        out.append(stripped)
    return out


@pytest.mark.parametrize("key", sorted([*RUN, *FAIL, *RUN_STUB]))
def test_tutorial_command_runs(key: str, tmp_path):
    stem, idx = key.split("#")
    path = next(p for p in _DOC_FILES if p.stem == stem)
    block = _blocks(path)[int(idx)]

    outputs_dir = REPO / "outputs"
    before = {p.name for p in outputs_dir.iterdir()} if outputs_dir.is_dir() else set()

    try:
        if key in RUN_STUB:
            rc, stdout, stderr, served = _run_stub_block(block, tmp_path / "stub-copies")
            assert rc == 0, (
                f"{key} failed (rc={rc}):\n$ {block}\n"
                f"--- stdout ---\n{stdout[-2000:]}\n--- stderr ---\n{stderr[-2000:]}"
            )
            assert served > 0, (
                f"{key} is classified RUN_STUB but the stub server served no "
                f"request — the rewritten URL was never reached:\n$ {block}"
            )
            return
        lines = _executables(block)
        assert lines, f"{key}: block has no executable line"
        env = _env()
        proc = subprocess.run(
            "\n".join(lines),
            shell=True,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if key in FAIL:
            sig = FAIL_SIGNATURES[key]
            assert proc.returncode != 0, (
                f"{key} is documented to fail (and the docs teach that failure), "
                f"but it succeeded:\n$ {block}"
            )
            assert sig in proc.stderr, (
                f"{key}: documented error signature {sig!r} missing from stderr:\n"
                f"{proc.stderr[-2000:]}"
            )
        else:
            assert proc.returncode == 0, (
                f"{key} failed (rc={proc.returncode}):\n$ {block}\n"
                f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
            )
    finally:
        # remove only the run dirs THIS execution created under outputs/
        if outputs_dir.is_dir():
            for created in {p.name for p in outputs_dir.iterdir()} - before:
                shutil.rmtree(outputs_dir / created, ignore_errors=True)
