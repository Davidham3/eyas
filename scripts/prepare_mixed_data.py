#!/usr/bin/env python3
"""Synthesize the mixed-modality smoke dataset (zero downloads, deterministic).

Writes the layout the ``mixed_mcq`` task consumes (default ``--out
datasets/bench/mixed``, already gitignored)::

    datasets/bench/mixed/
    ├── data/train.json       [{"id", "combo", "passage"?, "images"?, "videos"?,
    ├── data/test.json         "question", "choices", "answer"}]
    ├── images/*.png          palette solid squares (pure-stdlib PNGs)
    └── videos/*.mp4          8-frame solid-color H.264 clips

Every combo of ``MIXED_COMBOS`` gets ``per_combo`` samples per split; each
sample asks one question over the shared palette, and its payload is exactly
what the combo declares (a passage, one PNG, one clip). Media paths are
relative to the dataset root -- the sibling of ``data`` -- which is the
loader's default ``media_root``.

Deterministic: no RNG. The color is palette arithmetic over (index, combo
index), so a rerun reproduces identical records and file names. Existing media
files are left untouched, so regenerating over a populated directory is cheap.

Only the clips need PyAV, and ``av`` is imported lazily inside the encode path,
so no runtime dependency is added::

    uv run --with av python scripts/prepare_mixed_data.py --out datasets/bench/mixed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eyas.tasks.demo_shapes import PALETTE, PALETTE_RGB, png_solid
from eyas.tasks.mixed_mcq import MIXED_COMBOS

__all__ = [
    "CLIP_FRAMES",
    "MIXED_COMBOS",
    "build_records",
    "encode_clip",
    "generate",
    "main",
]

IMAGE_SIZE = 32  # demo_shapes' default PNG side
CLIP_FRAMES = 8
CLIP_SIZE = 64
CLIP_RATE = 2
QUESTION = "What color is the object?"


def build_records(split: str, per_combo: int) -> list[dict[str, Any]]:
    """One split's JSON records: combo-major order, ``per_combo`` samples each.

    Sample color = ``PALETTE[(idx + combo_index) % len(PALETTE)]`` and the
    answer is that color (the answer never depends on the medium); ids are
    ``mixed_<combo>_<split>_<idx:02d>`` and media paths are dataset-root
    relative (``images/...`` / ``videos/...``). Only the fields the combo
    declares are emitted, so the records satisfy the loader's payload contract
    by construction.
    """
    if per_combo < 0:
        raise ValueError(f"per_combo must be >= 0, got {per_combo}")
    records: list[dict[str, Any]] = []
    for combo_idx, combo in enumerate(MIXED_COMBOS):
        parts = combo.split("_")
        for idx in range(per_combo):
            color = PALETTE[(idx + combo_idx) % len(PALETTE)]
            stem = f"{combo}_{split}_{idx:02d}"
            record: dict[str, Any] = {
                "id": f"mixed_{stem}",
                "combo": combo,
                "question": QUESTION,
                "choices": list(PALETTE),
                "answer": color,
            }
            if "text" in parts:
                record["passage"] = f"The object is {color}."
            if "image" in parts:
                record["images"] = [f"images/{stem}.png"]
            if "video" in parts:
                record["videos"] = [f"videos/{stem}.mp4"]
            records.append(record)
    return records


def encode_clip(
    path: Path,
    rgb: tuple[int, int, int],
    *,
    frames: int = CLIP_FRAMES,
    size: int = CLIP_SIZE,
    rate: int = CLIP_RATE,
) -> None:
    """Encode a solid-color H.264 clip (``av`` imported here: clips are the only
    part of this script that needs it).

    Frames are written as raw ``rgb24`` plane buffers (no numpy); the muxer is
    flushed, then the file is decoded back and its frame count verified, so a
    truncated clip fails generation instead of surfacing mid-run.
    """
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=rate)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "30", "preset": "veryfast"}
        for _ in range(frames):
            frame = av.VideoFrame(size, size, "rgb24")
            frame.planes[0].update(bytes(rgb) * (size * size))
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():  # flush the encoder into the muxer
            out.mux(packet)
    with av.open(str(path)) as decoded:
        written = sum(1 for _ in decoded.decode(video=0))
    if written != frames:
        raise RuntimeError(f"clip {path} decodes to {written} frames, expected {frames}")


def generate(out: Path, *, train_per_combo: int = 4, test_per_combo: int = 4) -> None:
    """Write the whole dataset under ``out``.

    ``out/data/{train,test}.json`` plus ``out/images`` and ``out/videos`` (media
    is a sibling of ``data`` = the loader's default media root). Media files
    that already exist are kept untouched (idempotent); the JSON is rewritten
    every time (deterministic bytes, negligible cost).
    """
    out = Path(out)
    data_dir = out / "data"
    for directory in (data_dir, out / "images", out / "videos"):
        directory.mkdir(parents=True, exist_ok=True)
    for split, per_combo in (("train", train_per_combo), ("test", test_per_combo)):
        records = build_records(split, per_combo)
        for record in records:
            rgb = PALETTE_RGB[record["answer"]]
            for rel in record.get("images", []):
                path = out / rel
                if not path.is_file():
                    path.write_bytes(png_solid(rgb, IMAGE_SIZE))
            for rel in record.get("videos", []):
                path = out / rel
                if not path.is_file():
                    encode_clip(path, rgb)
        (data_dir / f"{split}.json").write_text(
            json.dumps(records, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        counts: dict[str, int] = {}
        for record in records:
            counts[record["combo"]] = counts.get(record["combo"], 0) + 1
        print(
            f"{split}: {len(records)} samples, "
            + ", ".join(f"{combo}={n}" for combo, n in counts.items())
        )
    print(
        f"mixed dataset ready at {out} "
        f"(train={len(build_records('train', train_per_combo))} "
        f"test={len(build_records('test', test_per_combo))})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("datasets/bench/mixed"))
    parser.add_argument("--train-per-combo", type=int, default=4)
    parser.add_argument("--test-per-combo", type=int, default=4)
    args = parser.parse_args(argv)
    generate(
        args.out,
        train_per_combo=args.train_per_combo,
        test_per_combo=args.test_per_combo,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
