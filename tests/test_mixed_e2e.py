"""Zero-GPU end-to-end: a mixed-modality split must evaluate every combo.

The mock backend (``ScriptedMockBackend``) records each request in full, so the
strong assertions are on what the engine would have received -- one test-split
request per combo, in ``MIXED_COMBOS`` order, carrying exactly that combo's
media -- with ``result.json`` only confirming the scored outcome. A run that
silently skipped a combo, or rendered one combo's payload into another's
request, fails here even though our own bookkeeping never ran.

The mock never decodes media, so the dataset is synthesized by
``prepare_mixed_data.build_records`` (pure, no av) with 1-byte placeholder
files. Layout matters: ``data_dir`` must be ``<root>/data`` so the loader
resolves media against ``<root>/images`` / ``<root>/videos``.

Method coverage, and the two deviations from a stock run:

- ``raw`` runs the shipped quickstart config untouched (non-raw rendering:
  passage parts, then images, then clips, then the query prose).
- ``cot`` drops its ``judge`` section. The mock has no judge-side distinction
  -- ``responses=["red"]`` would be replayed as the verdict and add judge
  requests after each evaluation batch, so the final-eval request block the
  ledger assertion walks would no longer be the sample requests. The rule-based
  score is identical, and the assert here is about what media the render layer
  put on the wire.
- ``spo`` runs the shipped config as-is (``raw_template: true``), which is the
  point: its seeds name ``{text}``/``{image}``/``{video}`` and every combo must
  render. The mock's flat ``"red"`` reply is not a renderable template, so SPO
  legitimately finishes on its seed after the single generation request -- no
  pairwise vote ever needs to select, and the run still evaluates all seven
  test combos through the in-place marker render path (the text-only request
  also pins the passage-to-prose newline).

Accuracy: ``build_records`` rotates the color by ``(idx + combo_idx) % 4`` over
PALETTE (blue, green, red, yellow). With ``per_combo=1`` the seven test combos
get blue, green, red, yellow, blue, green, red -- two reds -- and the mock
always answers "red", so exactly 2 of 7 samples score: ``accuracy == 2/7``.
This was verified against a real run's ``result.json`` (no invented field
names: ``final_eval`` has ``split``/``prompt_hash``/``metrics``/``cost``, and
``cost.requests`` is its sample-count record).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from _spec_cells import cell

from eyas.backends.mock import ScriptedMockBackend
from eyas.cli import main
from eyas.tasks.mixed_mcq import MIXED_COMBOS

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "prepare_mixed_data.py"
_METHODS = ("raw", "cot", "spo")

_MOCK_ANSWER = "red"
# the seven test combos' colors rotate blue/green/red/yellow from combo index 0
_EXPECTED_HITS = 2
_PER_COMBO = 1

# media descriptors in ScriptedMockBackend.render_request_text
_IMAGE_RE = re.compile(r"<image:[^>]*/images/([^>]+)\.png>")
_VIDEO_RE = re.compile(r"<video:[^>]*/videos/([^>]+)\.mp4>")
_PASSAGE_MARK = "The object is "


def _load_prepare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_mixed_data", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prepare(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Write the mixed dataset with 1-byte media placeholders; return the root
    plus the test split's records (the expected payloads come from them)."""
    mod = _load_prepare()
    root = tmp_path / "mixed"
    (root / "images").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "data").mkdir()
    test_records: list[dict] = []
    for split in ("train", "test"):
        records = mod.build_records(split, _PER_COMBO)
        for rec in records:
            for rel in rec.get("images", []) + rec.get("videos", []):
                (root / rel).write_bytes(b"x")
        (root / "data" / f"{split}.json").write_text(json.dumps(records), encoding="utf-8")
        if split == "test":
            test_records = records
    return root, test_records


def _config(tmp_path: Path, root: Path, method: str) -> Path:
    """The shipped spec cell for the method, pointed at the synthesized dataset
    and at the mock backend (the vllm block is dropped wholesale: zero GPU)."""
    base = cell("mixed", method)
    base["task"]["params"]["data_dir"] = str(root / "data")
    base["backend"] = {
        "name": "mock",
        "params": {"responses": [_MOCK_ANSWER], "cycle": True},
    }
    if method == "cot":
        # the mock has no judge role: keep the final-eval request block = sample
        # requests (see module docstring)
        base.pop("judge", None)
    path = tmp_path / f"mixed_{method}.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


@pytest.fixture
def mock_backends(monkeypatch: pytest.MonkeyPatch) -> list[ScriptedMockBackend]:
    """Every ScriptedMockBackend the run assembles, in creation order (the
    runtime wraps it in traced/normalizing decorators, which is why the
    instance is captured here rather than read off the run)."""
    captured: list[ScriptedMockBackend] = []
    original = ScriptedMockBackend.__init__

    @functools.wraps(original)
    def spy(self: ScriptedMockBackend, *args, **kwargs) -> None:
        original(self, *args, **kwargs)
        captured.append(self)

    monkeypatch.setattr(ScriptedMockBackend, "__init__", spy)
    return captured


def _combo_payload(combo: str) -> tuple[bool, list[str], list[str]]:
    """What a combo's test request must carry: passage presence + media stems.
    ``per_combo=1`` means the stems are exactly ``<combo>_test_00``."""
    modalities = set(combo.split("_"))
    stem = f"{combo}_test_00"
    images = [stem] if "image" in modalities else []
    videos = [stem] if "video" in modalities else []
    return "text" in modalities, images, videos


@pytest.mark.parametrize("method", list(_METHODS))
def test_mock_run_evaluates_all_seven_combos(
    tmp_path: Path, method: str, mock_backends: list[ScriptedMockBackend], capsys
) -> None:
    root, test_records = _prepare(tmp_path)
    cfg = _config(tmp_path, root, method)
    out = tmp_path / "run"
    rc = main(["run", "--config", str(cfg), "--output-dir", str(out)])
    assert rc == 0, capsys.readouterr().err

    # the seven test combos each carry exactly one red-answer sample
    assert sum(rec["answer"] == _MOCK_ANSWER for rec in test_records) == _EXPECTED_HITS

    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    final = result["final_eval"]
    assert final["split"] == "test"
    assert final["metrics"]["accuracy"] == pytest.approx(_EXPECTED_HITS / len(MIXED_COMBOS))
    assert final["cost"]["requests"] == len(MIXED_COMBOS)

    # what the engine actually sent: the run ends with one final-eval batch of
    # seven requests, one per test combo in MIXED_COMBOS order
    assert len(mock_backends) == 1
    backend = mock_backends[0]
    assert backend.call_sizes[-1] == len(MIXED_COMBOS), backend.call_sizes
    texts = [text for text, _ in backend.recorded_requests]
    assert len(texts) >= len(MIXED_COMBOS)
    tail = texts[-len(MIXED_COMBOS) :]
    # no test-split request was issued before the final-eval block, so the tail
    # is exactly the reporting pass
    early = [text for text in texts[: -len(MIXED_COMBOS)] if "_test_" in text]
    assert not early, early

    for combo, text in zip(MIXED_COMBOS, tail, strict=True):
        has_passage, images, videos = _combo_payload(combo)
        assert (_PASSAGE_MARK in text) is has_passage, f"{method}/{combo}: passage presence"
        assert _IMAGE_RE.findall(text) == images, f"{method}/{combo}: images"
        assert _VIDEO_RE.findall(text) == videos, f"{method}/{combo}: videos"
        # the rendering order the docs promise: passage, then images, then clips
        order: list[tuple[int, str]] = []
        if has_passage:
            order.append((text.index(_PASSAGE_MARK), "passage"))
        if images:
            image_at = _IMAGE_RE.search(text)
            assert image_at is not None
            order.append((image_at.start(), "image"))
        if videos:
            video_at = _VIDEO_RE.search(text)
            assert video_at is not None
            order.append((video_at.start(), "video"))
        assert order == sorted(order), f"{method}/{combo}: payload order {order}"

    if method == "spo":
        # raw_template mode: {text} is replaced in place and the passage stays
        # off the prose that follows it (one newline, even though the sample
        # carries no payload for the markers in between)
        text_only = tail[MIXED_COMBOS.index("text")]
        assert f"{_PASSAGE_MARK}{test_records[0]['answer']}.\nAnswer " in text_only
