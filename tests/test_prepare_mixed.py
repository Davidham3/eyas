"""prepare_mixed_data contract tests (zero download, zero network).

Covers: the pure record builder (combo-major coverage, payload/combo agreement,
unique ids, palette arithmetic, dataset-root-relative media paths) and the full
generator round-trip through the real ``load_mixed_mcq_samples`` loader (split
sizes, media counts, idempotent media writes).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mixed_data.py"

# combos that declare a clip: video, text_video, image_video, text_image_video
VIDEO_COMBOS = 4
# combos that declare an image: image, text_image, image_video, text_image_video
IMAGE_COMBOS = 4
SPLITS = ("train", "test")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_mixed_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_records_cover_every_combo_and_payload_matches():
    mod = _load()
    records = mod.build_records("train", per_combo=2)
    assert [r["combo"] for r in records] == [c for c in mod.MIXED_COMBOS for _ in range(2)]
    assert len({r["id"] for r in records}) == len(records)
    for rec in records:
        parts = rec["combo"].split("_")
        assert bool(rec.get("passage")) == ("text" in parts)
        assert len(rec.get("images", [])) == (1 if "image" in parts else 0)
        assert len(rec.get("videos", [])) == (1 if "video" in parts else 0)
        assert rec["answer"] in rec["choices"]
        assert rec["choices"] == list(mod.PALETTE)
        assert rec["question"] == "What color is the object?"


def test_records_are_deterministic_and_id_media_paths_agree():
    mod = _load()
    first = mod.build_records("test", per_combo=3)
    assert first == mod.build_records("test", per_combo=3)
    by_combo = {}
    for rec in first:
        by_combo.setdefault(rec["combo"], []).append(rec)
    for combo_idx, combo in enumerate(mod.MIXED_COMBOS):
        for idx, rec in enumerate(by_combo[combo]):
            color = mod.PALETTE[(idx + combo_idx) % len(mod.PALETTE)]
            stem = f"{combo}_test_{idx:02d}"
            assert rec["id"] == f"mixed_{stem}"
            assert rec["answer"] == color
            if "text" in combo.split("_"):
                assert rec["passage"] == f"The object is {color}."
            for key, pattern in (
                ("images", f"images/{stem}.png"),
                ("videos", f"videos/{stem}.mp4"),
            ):
                if key in rec:
                    assert rec[key] == [pattern]
                    assert not Path(rec[key][0]).is_absolute()
    # ids embed the split, so two splits of one dataset never collide
    train_ids = {rec["id"] for rec in mod.build_records("train", per_combo=3)}
    assert train_ids.isdisjoint({rec["id"] for rec in first})


def test_generate_writes_loadable_dataset(tmp_path):
    pytest.importorskip("av")
    mod = _load()
    per_combo = 1
    mod.generate(tmp_path, train_per_combo=per_combo, test_per_combo=per_combo)

    for split in SPLITS:
        records = json.loads((tmp_path / "data" / f"{split}.json").read_text(encoding="utf-8"))
        assert len(records) == len(mod.MIXED_COMBOS)
        assert [rec["combo"] for rec in records] == list(mod.MIXED_COMBOS)
    expected_clips = VIDEO_COMBOS * len(SPLITS) * per_combo
    assert len(sorted((tmp_path / "videos").glob("*.mp4"))) == expected_clips
    expected_images = IMAGE_COMBOS * len(SPLITS) * per_combo
    assert len(sorted((tmp_path / "images").glob("*.png"))) == expected_images

    # round-trip through the real loader (media paths resolve off the dataset root)
    from eyas.tasks.datasets import load_mixed_mcq_samples

    samples = load_mixed_mcq_samples(tmp_path / "data", "test")
    assert len(samples) == len(mod.MIXED_COMBOS)
    assert [sample.inputs["combo"] for sample in samples] == list(mod.MIXED_COMBOS)
    assert all(sample.references["label"] in sample.inputs["choices"] for sample in samples)


def test_generate_is_idempotent_for_media(tmp_path):
    pytest.importorskip("av")
    mod = _load()
    mod.generate(tmp_path, train_per_combo=1, test_per_combo=1)
    media = sorted((*tmp_path.glob("images/*.png"), *tmp_path.glob("videos/*.mp4")))
    assert media
    stamps = {path: path.stat().st_mtime_ns for path in media}
    mod.generate(tmp_path, train_per_combo=1, test_per_combo=1)
    assert {path: path.stat().st_mtime_ns for path in media} == stamps


def test_encoded_clip_decodes_to_the_same_frame_count(tmp_path):
    """The generator self-checks with PyAV; the runtime decodes with vLLM's
    OpenCV backend. Pin the two to the same frame count so a clip is never
    verified by one decoder and silently truncated by the other."""
    pytest.importorskip("av")
    pytest.importorskip("vllm")
    mod = _load()
    path = tmp_path / "clip.mp4"
    mod.encode_clip(path, (255, 0, 0))

    from eyas.backends.prefetch import decode_video

    loaded = decode_video(str(path))
    frames, meta = loaded.media
    assert len(frames) == mod.CLIP_FRAMES
    assert meta["total_num_frames"] == mod.CLIP_FRAMES
