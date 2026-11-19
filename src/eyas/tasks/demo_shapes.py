"""demo_shapes: built-in demo task with zero external data (pure-stdlib PNGs).

Purpose: let external users verify right after cloning that the eyas pipeline
runs end to end, with no GPU and no data download
(``uv run eyas run --config tests/fixtures/demo_cot.yaml``). It synthesizes
"solid color square" images — hand-written PNGs via zlib + struct (8bit
truecolor, no PIL dependency), persisted deterministically to
``~/.cache/eyas/demo_shapes/`` (skipped when already present; content is
byte-for-byte reproducible). The ``shapes`` param is additive: requesting
``("square", "circle")`` appends circle samples (palette-major order, a
color-disc circle on a white background) while the default stays byte-for-byte
the original 12 squares — every existing demo path is untouched.

The same module hosts the zero-data VQA demo (``demo_vqa``): a deterministic
question/answer corpus over the same PNG renderer (color and shape questions),
so VQA is demoable with no dataset either (``tests/fixtures/demo_vqa.yaml``).

Honest disclaimer: this is a pipeline demo, not a research benchmark — the
mock backend's responses do not "know" the correct answers, so the metrics
carry no algorithmic signal; real signal requires a real engine backend and real
datasets.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    TextPart,
    system_message,
    user_message,
)
from eyas.core.objective import Objective
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.vqa import VqaTask

__all__ = [
    "PALETTE",
    "PALETTE_RGB",
    "SHAPES",
    "DemoShapesTask",
    "DemoVqaTask",
    "default_image_dir",
    "generate_demo_samples",
    "png_circle",
    "png_disk",
    "png_solid",
]

PALETTE: tuple[str, ...] = ("blue", "green", "red", "yellow")
PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    "blue": (30, 100, 220),
    "green": (40, 170, 70),
    "red": (215, 50, 45),
    "yellow": (240, 200, 40),
}
SHAPES: tuple[str, ...] = ("square", "circle")
_WHITE = (255, 255, 255)
_SIZE = 32
_PER_COLOR = 3
_TRAIN_SIZE = 8


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _png(rows: list[bytes], size: int) -> bytes:
    """Assemble pre-rendered pixel rows (``size * 3`` bytes each) into a valid
    8bit truecolor PNG (color type 2) with filter byte 0 on every row."""
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def png_solid(rgb: tuple[int, int, int], size: int = _SIZE) -> bytes:
    """Minimal valid PNG: 8bit truecolor (color type 2), one solid color
    for the whole image."""
    return _png([bytes(rgb) * size] * size, size)


def png_circle(rgb: tuple[int, int, int], size: int = _SIZE) -> bytes:
    """Color disc on a white background: pixels within ``size // 2 - 1`` of
    the center pixel (``size // 2``, ``size // 2``) get ``rgb``, the rest is
    white (same hand-written truecolor writer as :func:`png_solid`)."""
    center = size // 2
    radius = size // 2 - 1
    rows: list[bytes] = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            dx, dy = x - center, y - center
            px = rgb if dx * dx + dy * dy <= radius * radius else _WHITE
            row += bytes(px)
        rows.append(bytes(row))
    return _png(rows, size)


def png_disk(ctx: dict[str, Any], size: int = _SIZE) -> bytes:
    """Renderer dispatch over a shape context ``{"kind": "square"|"circle",
    "rgb": (r, g, b)}`` — the single entry point demo generation goes through."""
    kind = ctx["kind"]
    rgb: tuple[int, int, int] = ctx["rgb"]
    if kind == "square":
        return png_solid(rgb, size)
    if kind == "circle":
        return png_circle(rgb, size)
    raise ValueError(f"Unknown demo_shapes kind: {kind!r} (accepted: square, circle)")


def default_image_dir() -> Path:
    return Path.home() / ".cache" / "eyas" / "demo_shapes"


def _validate_shapes(shapes: Sequence[str]) -> None:
    unknown = [s for s in shapes if s not in SHAPES]
    if unknown:
        raise ValueError(f"Unknown demo_shapes shapes: {unknown} (accepted: {', '.join(SHAPES)})")


def materialize(
    shapes: Sequence[str] = ("square",), image_dir: Path | None = None
) -> list[tuple[str, str, Path]]:
    """Generate palette × shapes × ``_PER_COLOR`` PNGs (deterministic paths and
    content; skipped when already present). Returns ``(color, shape, path)``
    in emit order — the default single-shape run produces exactly 12 solid
    squares named ``{color}_{k}.png``."""
    _validate_shapes(shapes)
    image_dir = image_dir if image_dir is not None else default_image_dir()
    image_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, str, Path]] = []
    for color in PALETTE:
        for shape in shapes:
            # squares keep the bare color stem; other shapes add a suffix
            stem = color if shape == "square" else f"{color}_{shape}"
            for k in range(_PER_COLOR):
                path = image_dir / f"{stem}_{k}.png"
                if not path.exists():
                    path.write_bytes(png_disk({"kind": shape, "rgb": PALETTE_RGB[color]}))
                out.append((color, shape, path))
    return out


@register_task("demo_shapes")
class DemoShapesTask(Task):
    """Four-way color choice for a solid shape (default 12 samples: first 8
    for train, last 4 for test/val; labels cycle over the palette). ``shapes``
    is additive — extra kinds append samples in palette-major order (all of a
    color's variants first), so the default square-only run is unchanged."""

    name = "demo_shapes"
    objectives: ClassVar[tuple[Objective, ...]] = (
        Objective(key="accuracy", maximize=True, display_name="accuracy"),
    )
    default_prompt = "What color is the square?"

    def __init__(
        self, *, image_dir: str | None = None, shapes: Sequence[str] = ("square",)
    ) -> None:
        entries = materialize(shapes, Path(image_dir) if image_dir is not None else None)
        self._samples = [
            Sample(
                id=f"demo_{i:02d}",
                inputs={"image": str(path), "choices": list(PALETTE)},
                # label cycles over the palette per sample index — deliberately independent
                # of the image's own color (the demo's metric carries no signal; this is
                # also what fixes the mock backend's 2/8 score on the demo train split —
                # label order pinned by tests/tasks/test_demo_task.py — and must not shift)
                references={"label": PALETTE[i % len(PALETTE)], "shape": shape},
            )
            for i, (_color, shape, path) in enumerate(entries)
        ]
        train_size = _TRAIN_SIZE if len(shapes) == 1 else len(shapes) * _PER_COLOR * 2
        self._splits = {
            "train": self._samples[:train_size],
            "test": self._samples[train_size:],
            "val": self._samples[train_size:],
        }

    def load_split(self, split: str) -> list[Sample]:
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split!r} (demo_shapes supports train/test/val)")
        return list(self._splits[split])

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        choices = ", ".join(sample.inputs["choices"])
        # the question names the sample's actual shape (a circle asked "this
        # square" feeds the model a false premise); square-only runs keep the
        # original bytes
        shape = sample.references.get("shape", "square")
        return ChatRequest(
            messages=(
                system_message(prompt),
                user_message(
                    ImagePart(image=sample.inputs["image"]),
                    TextPart(
                        f"\nWhat color is this {shape}? Choices: [{choices}]\n"
                        "Answer with one color name from the choices."
                    ),
                ),
            ),
            request_id=f"{sample.id}:{variant.id}",
        )

    def demo_media_parts(self, sample: Sample) -> list[ImagePart]:
        """A few-shot demo carries the sample image, like evaluation."""
        return [ImagePart(image=sample.inputs["image"])]

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        text = response.text.lower()
        matched = [c for c in PALETTE if c in text]
        value = max(matched, key=lambda c: text.rfind(c)) if matched else None
        return Prediction(value=value, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": float(prediction.value == sample.references["label"])}


_COLOR_QUESTION = "What color is this shape?"
_SHAPE_QUESTION = "What shape is this?"


def generate_demo_samples(count: int = 96, image_dir: Path | None = None) -> list[Sample]:
    """Deterministic zero-data VQA corpus (no RNG): samples alternate the two
    questions, colors cycle over the palette and shapes cycle over ``SHAPES``
    every two samples, so the ground truth is pure index arithmetic. For the
    color question ``references={"answers": [color]}``, for the shape question
    ``references={"answers": [shape]}`` (the VqaTask sample convention). Same
    count + dir → identical ids, questions, ground truth, and image bytes."""
    image_dir = image_dir if image_dir is not None else default_image_dir().parent / "demo_vqa"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for i in range(count):
        color = PALETTE[(i // 2) % len(PALETTE)]
        shape = SHAPES[(i // 2) % len(SHAPES)]
        question = _COLOR_QUESTION if i % 2 == 0 else _SHAPE_QUESTION
        answer = color if i % 2 == 0 else shape
        path = image_dir / f"vqa_{i:03d}_{shape}_{color}.png"
        if not path.exists():
            path.write_bytes(png_disk({"kind": shape, "rgb": PALETTE_RGB[color]}))
        samples.append(
            Sample(
                id=f"demo_vqa_{i:02d}",
                inputs={"image": str(path), "question": question},
                references={"answers": [answer]},
            )
        )
    return samples


@register_task("demo_vqa")
class DemoVqaTask(VqaTask):
    """Zero-data VQA demo: scores through the full VqaTask path (normalize +
    VQA accuracy) over :func:`generate_demo_samples` — no data files needed."""

    name = "demo_vqa"

    def __init__(self, *, image_dir: str | None = None, count: int = 96) -> None:
        target = (
            Path(image_dir) if image_dir is not None else default_image_dir().parent / "demo_vqa"
        )
        super().__init__(images_dir=str(target), samples=generate_demo_samples(count, target))
