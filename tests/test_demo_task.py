"""Contract tests for the demo_shapes builtin task (outward usability).

Covers: pure-stdlib PNG structural validity (chunk CRC / IHDR / IDAT),
deterministic materialization, the build_request / parse_response / score loop,
mock cycle semantics (loop vs fail fast), eyas run end to end (mock + cot, predictable metrics),
the additive shapes extension (circle renderer), the deterministic VQA demo corpus, and the
demo_vqa zero-data task.
"""

from __future__ import annotations

import asyncio
import struct
import zlib
from pathlib import Path

import pytest

from eyas.backends.base import ChatRequest, ChatResponse, ImagePart, SamplingConfig, TextPart
from eyas.backends.mock import ScriptedMockBackend
from eyas.engine.runtime import load_config, run_experiment
from eyas.registry import get
from eyas.tasks.base import Sample
from eyas.tasks.demo_shapes import (
    PALETTE,
    PALETTE_RGB,
    SHAPES,
    DemoShapesTask,
    default_image_dir,
    generate_demo_samples,
    png_solid,
)
from eyas.tasks.protocol import IdentityProtocol
from eyas.tasks.vqa import normalize_vqa_answer

_REPO = Path(__file__).resolve().parents[1]

_COLOR_Q = "What color is this shape?"
_SHAPE_Q = "What shape is this?"


def _chunks(blob: bytes) -> list[tuple[bytes, bytes]]:
    out: list[tuple[bytes, bytes]] = []
    pos = 8
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    while pos < len(blob):
        (length,) = struct.unpack(">I", blob[pos : pos + 4])
        tag = blob[pos + 4 : pos + 8]
        data = blob[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", blob[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + data) & 0xFFFFFFFF, f"CRC mismatch: {tag!r}"
        out.append((tag, data))
        pos += 12 + length
    assert out[-1][0] == b"IEND" and out[-1][1] == b""
    return out


class TestPngSolid:
    def test_valid_structure(self):
        blob = png_solid((1, 2, 3), size=8)
        tags = [tag for tag, _ in _chunks(blob)]
        assert tags == [b"IHDR", b"IDAT", b"IEND"]

        ihdr = dict(_chunks(blob))[b"IHDR"]
        width, height, depth, ctype = struct.unpack(">IIBB", ihdr[:10])
        assert (width, height, depth, ctype) == (8, 8, 8, 2)  # truecolor

        idat = dict(_chunks(blob))[b"IDAT"]
        row = b"\x00" + bytes((1, 2, 3)) * 8
        assert zlib.decompress(idat) == row * 8

    def test_deterministic(self):
        assert png_solid((1, 2, 3)) == png_solid((1, 2, 3))
        assert png_solid((1, 2, 3)) != png_solid((4, 5, 6))


class TestDemoShapesTask:
    def test_materialize_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        task_a = DemoShapesTask()
        task_b = DemoShapesTask()

        img_dir = default_image_dir()
        assert img_dir == tmp_path / ".cache" / "eyas" / "demo_shapes"
        pngs = sorted(img_dir.glob("*.png"))
        assert len(pngs) == len(PALETTE) * 3
        # re-instantiating without rewriting stays consistent: deterministic paths and bytes
        assert [s.inputs["image"] for s in task_a.load_split("train")] == [
            s.inputs["image"] for s in task_b.load_split("train")
        ]

        train = task_a.load_split("train")
        test = task_a.load_split("test")
        assert len(train) == 8 and len(test) == 4
        assert [s.references["label"] for s in train] == [
            PALETTE[i % len(PALETTE)] for i in range(8)
        ]
        for s in [*train, *test]:
            assert Path(s.inputs["image"]).exists()

    def test_unknown_split_fails_fast(self, tmp_path):
        task = DemoShapesTask(image_dir=str(tmp_path / "imgs"))
        with pytest.raises(ValueError, match="split"):
            task.load_split("dev")

    def test_request_parse_score(self, tmp_path):
        task = DemoShapesTask(image_dir=str(tmp_path / "imgs"))
        sample = task.load_split("train")[0]
        req = task.build_request("What color?", sample, IdentityProtocol().variants(sample)[0])

        images = [p for m in req.messages for p in m.content if isinstance(p, ImagePart)]
        assert len(images) == 1 and Path(images[0].image).is_file()

        assert task.parse_response(ChatResponse(text="It is Red."), sample).value == "red"
        assert task.parse_response(ChatResponse(text="Cannot determine"), sample).value is None

        label = sample.references["label"]
        hit = task.parse_response(ChatResponse(text=label), sample)
        assert task.score(hit, sample) == {"accuracy": 1.0}
        miss = task.parse_response(ChatResponse(text="???"), sample)
        assert task.score(miss, sample) == {"accuracy": 0.0}

    def test_demo_media_parts_carry_the_sample_image(self, tmp_path):
        task = DemoShapesTask(image_dir=str(tmp_path / "imgs"))
        sample = task.load_split("train")[0]
        parts = task.demo_media_parts(sample)
        assert len(parts) == 1
        assert isinstance(parts[0], ImagePart)
        assert parts[0].image == sample.inputs["image"]

    def test_registered(self):
        import eyas.builtin  # noqa: F401

        assert get("task", "demo_shapes") is DemoShapesTask


class TestMockCycle:
    def test_cycle_restarts_script(self):
        backend = ScriptedMockBackend(responses=["a", "b"], cycle=True)
        req = ChatRequest(messages=(), request_id="r")
        out1 = asyncio.run(backend.agenerate([req], SamplingConfig()))
        out2 = asyncio.run(backend.agenerate([req, req], SamplingConfig()))
        out3 = asyncio.run(backend.agenerate([req], SamplingConfig()))
        assert [r.text for r in out1] == ["a"]
        assert [r.text for r in out2] == ["b", "a"]
        assert [r.text for r in out3] == ["b"]

    def test_exhaustion_fails_without_cycle(self):
        backend = ScriptedMockBackend(responses=["a"])
        req = ChatRequest(messages=(), request_id="r")
        asyncio.run(backend.agenerate([req], SamplingConfig()))
        with pytest.raises(AssertionError, match="script exhausted"):
            asyncio.run(backend.agenerate([req], SamplingConfig()))


class TestDemoRunEntry:
    def test_end_to_end_cot(self, tmp_path):
        # mock always answers red -> accuracy = share of red samples in train = 2/8
        config = load_config(_REPO / "tests" / "fixtures" / "demo_cot.yaml")
        config["task"]["params"]["image_dir"] = str(tmp_path / "imgs")
        run_dir = tmp_path / "run"
        result = run_experiment(config, output_dir=run_dir)

        assert result.best is not None
        assert result.best.scores["accuracy"] == pytest.approx(0.25)
        assert (run_dir / "result.json").is_file()
        assert (run_dir / "checkpoint.json").is_file()


def _decode_png(blob: bytes) -> tuple[int, int, list[list[bytes]]]:
    """Decode a hand-written truecolor PNG (filter byte 0) into rows of RGB byte strings."""
    chunks = dict(_chunks(blob))
    width, height, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, ctype) == (8, 2), "expected 8bit truecolor"
    raw = zlib.decompress(chunks[b"IDAT"])
    stride = 1 + width * 3
    assert len(raw) == stride * height
    rows = []
    for y in range(height):
        line = raw[y * stride : (y + 1) * stride]
        assert line[0] == 0, "the hand-written writer never filters rows"
        rows.append([line[1 + 3 * x : 4 + 3 * x] for x in range(width)])
    return width, height, rows


class TestShapesExtension:
    """Additive contract: the default stays the 12 squares; shapes=(...) only adds samples."""

    def test_default_construct_unchanged(self, tmp_path):
        task = DemoShapesTask(image_dir=str(tmp_path / "imgs"))
        samples = [*task.load_split("train"), *task.load_split("test")]
        assert len(samples) == 12
        assert [s.id for s in samples] == [f"demo_{i:02d}" for i in range(12)]
        assert all(s.references["shape"] == "square" for s in samples)
        for i, s in enumerate(samples):
            if i % 4 == 0:  # ids divisible by 4 are squares (trivially all-12 by default)
                assert s.references["shape"] == "square"
            # labels cycle over the palette per file index — deliberately independent
            # of the image's own color (that's what makes the mock-red demo score 2/8)
            assert s.references["label"] == PALETTE[i % len(PALETTE)]
            # "the same 12 solid-color squares": byte-for-byte what png_solid always produced
            blob = Path(s.inputs["image"]).read_bytes()
            assert blob == png_solid(PALETTE_RGB[PALETTE[i // 3]])

    def test_square_and_circle_extension(self, tmp_path):
        assert SHAPES == ("square", "circle")
        task = DemoShapesTask(image_dir=str(tmp_path / "imgs"), shapes=("square", "circle"))
        samples = [*task.load_split("train"), *task.load_split("test")]
        assert len(samples) == 24
        assert [s.id for s in samples] == [f"demo_{i:02d}" for i in range(24)]
        # palette-major, then shape: 3 squares + 3 circles per color; colors blue×6,
        # green×6, red×6, yellow×6; labels keep cycling per index (as in the default run)
        per_color = len(SHAPES) * 3
        assert [s.references["shape"] for s in samples] == (["square"] * 3 + ["circle"] * 3) * 4
        assert [s.references["label"] for s in samples] == [
            PALETTE[i % len(PALETTE)] for i in range(24)
        ]
        assert len(task.load_split("train")) == 12 and len(task.load_split("test")) == 12
        for s in samples:
            assert Path(s.inputs["image"]).exists()
        # the full shape pattern pinned above is the statement of the same
        # palette-major, then shape order
        # circles decode truthfully: center painted with the IMAGE's color,
        # corner on the white background (not any palette color)
        all_rgbs = set(PALETTE_RGB.values())
        for i, s in enumerate(samples):
            blob = Path(s.inputs["image"]).read_bytes()
            image_color = PALETTE[i // per_color]
            if s.references["shape"] == "circle":
                width, height, rows = _decode_png(blob)
                center = rows[height // 2][width // 2]
                corner = rows[0][0]
                assert center == bytes(PALETTE_RGB[image_color])
                assert corner == b"\xff\xff\xff"
                assert corner not in all_rgbs
            else:
                # squares keep the full-face solid fill bytes
                assert blob == png_solid(PALETTE_RGB[image_color])
        # default square-only run keeps the bare color names; circles append beside them
        names = sorted(p.name for p in Path(str(tmp_path / "imgs")).glob("*.png"))
        assert f"{PALETTE[0]}_circle_0.png" in names and f"{PALETTE[0]}_0.png" in names

    def test_unknown_shape_fails_fast(self, tmp_path):
        with pytest.raises(ValueError, match=r"square.*circle|circle.*square"):
            DemoShapesTask(image_dir=str(tmp_path / "imgs"), shapes=("triangle",))

    def test_vqa_demo_config_binds_and_runs(self, tmp_path):
        # the auto-parametrized tests/test_configs.py family validates binding for
        # every configs/*.yaml; this pins the VQA demo's end-to-end numbers on tmp dirs
        config = load_config(_REPO / "tests" / "fixtures" / "demo_vqa.yaml")
        config["task"]["params"]["image_dir"] = str(tmp_path / "imgs")
        mock_responses = config["backend"]["params"]["responses"]
        assert any(r in PALETTE for r in mock_responses), "at least one color-answer hit"
        assert any(r in SHAPES for r in mock_responses), "at least one shape-answer hit"
        run_dir = tmp_path / "run"
        result = run_experiment(config, output_dir=run_dir)
        assert result.best is not None
        # mock cycles ["blue", "square"]: shape questions hit when the pair's shape is
        # "square" (24 of 48 pairs), color questions hit when the pair's color is "blue"
        # (12 of 48 pairs) -> 36 of 96 samples score 1/3 (VQA's min(1, eq/3) with a
        # single annotated ground-truth answer)
        assert result.best.scores["accuracy"] == pytest.approx((24 + 12) / 96 / 3)
        assert (run_dir / "result.json").is_file()


class TestGenerateDemoSamples:
    def test_deterministic_vqa_corpus(self, tmp_path):
        samples = generate_demo_samples(24, tmp_path)
        assert len(samples) == 24
        for i, s in enumerate(samples):
            if i % 2 == 0:
                assert s.inputs["question"] == _COLOR_Q
                assert s.references["answers"] == [PALETTE[(i // 2) % len(PALETTE)]]
            else:
                assert s.inputs["question"] == _SHAPE_Q
                assert s.references["answers"] == [SHAPES[(i // 2) % len(SHAPES)]]
            assert Path(s.inputs["image"]).is_file()
        # determinism: same count + dir → identical ids / questions / ground truth
        again = generate_demo_samples(24, tmp_path)
        assert [(s.id, s.inputs, s.references) for s in again] == [
            (s.id, s.inputs, s.references) for s in samples
        ]


class TestDemoVqaTask:
    def test_registered_and_scores_through_vqa_path(self, tmp_path):
        import eyas.builtin  # noqa: F401

        task_cls = get("task", "demo_vqa")
        task = task_cls(image_dir=str(tmp_path / "vqa"), count=24)
        train = task.load_split("train")
        assert train, "demo_vqa train split must be non-empty"

        sample = train[0]  # a color question
        answer = sample.references["answers"][0]
        prediction = task.parse_response(ChatResponse(text=answer), sample)
        assert prediction.value == normalize_vqa_answer(answer)
        # VqaTask.score is VQA-standard min(1, eq/3): with the single annotated ground-truth
        # answer of the demo corpus a correct prediction scores 1/3, not 1.0
        # (>=3 duplicates would hit 1.0)
        assert task.score(prediction, sample) == {"accuracy": pytest.approx(1 / 3)}
        duplicated = Sample(
            id=sample.id,
            inputs=sample.inputs,
            references={"answers": [answer, answer, answer]},
        )
        assert task.score(prediction, duplicated) == {"accuracy": 1.0}

        miss = task.parse_response(ChatResponse(text="chartreuse hexagon"), sample)
        assert task.score(miss, sample) == {"accuracy": 0.0}


def test_circle_prompt_text_names_the_circle(tmp_path):
    # the user-turn question must name the sample's actual shape: a circle
    # sample asked "this square" feeds the model a false premise
    task = DemoShapesTask(image_dir=str(tmp_path / "imgs"), shapes=("square", "circle"))
    samples = task.load_split("train")
    circle = next(s for s in samples if s.references["shape"] == "circle")
    square = next(s for s in samples if s.references["shape"] == "square")

    def _text(sample: Sample) -> str:
        req = task.build_request("What color?", sample, IdentityProtocol().variants(sample)[0])
        return " ".join(p.text for m in req.messages for p in m.content if isinstance(p, TextPart))

    assert "circle" in _text(circle) and "square" not in _text(circle)
    assert "square" in _text(square)  # default square rendering stays byte-identical
