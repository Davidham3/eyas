from __future__ import annotations

import json
from pathlib import Path

import pytest

from eyas.backends.base import ChatRequest, ImagePart, TextPart, VideoPart, user_message
from eyas.tasks.base import Variant
from eyas.tasks.datasets import load_mixed_mcq_samples
from eyas.tasks.mixed_mcq import MIXED_COMBOS, MixedMcqTask


def _write_dataset(root: Path, records: list[dict]) -> Path:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)
    for rec in records:
        for rel in rec.get("images", []) + rec.get("videos", []):
            (root / rel).write_bytes(b"x")
    for split in ("train", "test"):
        (root / "data" / f"{split}.json").write_text(json.dumps(records), encoding="utf-8")
    return root / "data"


def _record(combo: str, idx: int = 0, color: str = "red") -> dict:
    rec = {
        "id": f"mix_{combo}_{idx}",
        "combo": combo,
        "question": "What color is the object?",
        "choices": ["red", "blue", "green", "yellow"],
        "answer": color,
    }
    if "text" in combo:
        rec["passage"] = f"The object described is {color}."
    if "image" in combo:
        rec["images"] = [f"images/{combo}_{idx}.png"]
    if "video" in combo:
        rec["videos"] = [f"videos/{combo}_{idx}.mp4"]
    return rec


def test_loader_accepts_all_seven_combos(tmp_path):
    data_dir = _write_dataset(tmp_path, [_record(c, i) for i, c in enumerate(MIXED_COMBOS)])
    samples = load_mixed_mcq_samples(data_dir, "test")
    assert [s.inputs["combo"] for s in samples] == list(MIXED_COMBOS)
    assert all(s.references["label"] == "red" for s in samples)


def test_loader_rejects_combo_payload_mismatch(tmp_path):
    rec = _record("image")
    rec["combo"] = "video"
    data_dir = _write_dataset(tmp_path, [rec])
    with pytest.raises(ValueError, match="mix_image_0"):
        load_mixed_mcq_samples(data_dir, "test")


def test_loader_rejects_answer_outside_choices(tmp_path):
    rec = _record("text")
    rec["answer"] = "purple"
    data_dir = _write_dataset(tmp_path, [rec])
    with pytest.raises(ValueError, match="not one of the choices"):
        load_mixed_mcq_samples(data_dir, "test")


def test_loader_rejects_missing_media_file(tmp_path):
    rec = _record("image")
    data_dir = _write_dataset(tmp_path, [rec])
    (tmp_path / rec["images"][0]).unlink()
    with pytest.raises(FileNotFoundError, match="image file not found"):
        load_mixed_mcq_samples(data_dir, "test")


def test_loader_rejects_unknown_combo(tmp_path):
    rec = _record("text")
    rec["combo"] = "audio"
    data_dir = _write_dataset(tmp_path, [rec])
    with pytest.raises(ValueError, match="unknown combo"):
        load_mixed_mcq_samples(data_dir, "test")


def test_loader_rejects_single_choice(tmp_path):
    rec = _record("text")
    rec["choices"] = ["red"]
    data_dir = _write_dataset(tmp_path, [rec])
    with pytest.raises(ValueError, match="at least 2"):
        load_mixed_mcq_samples(data_dir, "test")


def test_loader_rejects_non_array_json(tmp_path):
    data_dir = _write_dataset(tmp_path, [])
    (data_dir / "test.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_mixed_mcq_samples(data_dir, "test")


@pytest.mark.parametrize("key", ["images_dir", "videos_dir"])
def test_task_rejects_an_explicit_media_dir(tmp_path, key):
    # the loader resolves media paths against data_dir's parent; forwarding a
    # separate root would double-prefix every path and fail at first load
    data_dir = _write_dataset(tmp_path, [_record("image")])
    with pytest.raises(ValueError, match="media root"):
        MixedMcqTask(data_dir=str(data_dir), **{key: str(tmp_path)})


def _task(tmp_path, records, **kwargs) -> MixedMcqTask:
    data_dir = _write_dataset(tmp_path, records)
    return MixedMcqTask(data_dir=str(data_dir), **kwargs)


def test_text_combo_renders_text_only(tmp_path):
    task = _task(tmp_path, [_record("text")])
    req = task.build_request(
        "Classify the sample.", task.load_split("test")[0], Variant("identity")
    )
    assert req.messages[0].content == (
        TextPart("The object described is red."),
        TextPart(
            "Classify the sample.\nQuestion: What color is the object?\n"
            "Choices: ['red', 'blue', 'green', 'yellow']"
        ),
    )


def test_image_video_combo_orders_passage_then_image_then_clip(tmp_path):
    task = _task(tmp_path, [_record("text_image_video")])
    req = task.build_request("Classify.", task.load_split("test")[0], Variant("identity"))
    parts = req.messages[0].content
    kinds = [type(p) for p in parts]
    # the clip is pinned to a line start in the non-raw path too: InternVL's
    # template emits <video> as plain text and the re-encoded block is only
    # findable when it opens a line
    assert kinds == [TextPart, ImagePart, TextPart, VideoPart, TextPart]
    assert parts[1] == ImagePart(f"{tmp_path}/images/text_image_video_0.png")
    assert parts[2] == TextPart("\n")
    assert parts[3] == VideoPart(f"{tmp_path}/videos/text_image_video_0.mp4")


def test_non_raw_video_only_puts_the_clip_on_its_own_line(tmp_path):
    task = _task(tmp_path, [_record("text_video")])
    parts = (
        task.build_request("Classify.", task.load_split("test")[0], Variant("identity"))
        .messages[0]
        .content
    )
    assert [type(p) for p in parts] == [TextPart, TextPart, VideoPart, TextPart]
    assert parts[1] == TextPart("\n")


def test_a_repeated_marker_renders_its_payload_once(tmp_path):
    # an optimizer can mutate a genotype into naming the same marker twice; a
    # second clip would push the request past the 32k window and score 0
    task = _task(tmp_path, [_record("video")], raw_template=True)
    parts = (
        task.build_request(
            "{video}again {video}\nQuestion: {question}\nChoices: [{choices}]",
            task.load_split("test")[0],
            Variant("identity"),
        )
        .messages[0]
        .content
    )
    assert sum(isinstance(p, VideoPart) for p in parts) == 1


def test_planner_accepts_one_clip_plus_one_image_and_rejects_two_clips(tmp_path):
    from eyas.engine.capability import ResourceProfile
    from eyas.engine.planner import PlannedRequest, Rejected, RequestPlanner

    profile = ResourceProfile(max_images=None, ctx_reserve=0)
    planner = RequestPlanner(profile, context_window=32_768)
    task = _task(tmp_path, [_record("text_image_video"), _record("video", idx=1)])

    combined = (
        task.build_request("Classify.", task.load_split("test")[0], Variant("identity"))
        .messages[0]
        .content
    )
    assert [type(p) for p in combined].count(VideoPart) == 1
    planned = planner.plan(ChatRequest(messages=(user_message(*combined),)))
    assert isinstance(planned, PlannedRequest), planned

    # the same sample with a second clip cannot fit a 32k window: the planner
    # rejects it here because vLLM would raise at add_request and abort the run
    two_clips = (*combined[:-1], VideoPart(f"{tmp_path}/videos/second.mp4"), combined[-1])
    rejected = planner.plan(ChatRequest(messages=(user_message(*two_clips),)))
    assert isinstance(rejected, Rejected)
    assert "exceeds usable window" in rejected.reason


def test_loader_rejects_more_than_one_clip(tmp_path):
    rec = _record("video")
    rec["videos"] = [rec["videos"][0], "videos/second.mp4"]
    (tmp_path / "videos").mkdir(parents=True, exist_ok=True)
    (tmp_path / "videos" / "second.mp4").write_bytes(b"x")
    data_dir = _write_dataset(tmp_path, [rec])
    with pytest.raises(ValueError, match="mix_video_0.*at most one clip"):
        load_mixed_mcq_samples(data_dir, "test")


def test_raw_template_puts_a_mid_prompt_clip_on_its_own_line(tmp_path):
    task = _task(tmp_path, [_record("video")], raw_template=True)
    req = task.build_request(
        "Watch {video} carefully.\nQuestion: {question}\nChoices: [{choices}]",
        task.load_split("test")[0],
        Variant("identity"),
    )
    parts = req.messages[0].content
    assert [type(p) for p in parts] == [TextPart, TextPart, VideoPart, TextPart]
    assert parts[1] == TextPart("\n")


@pytest.mark.parametrize(
    ("combo", "n_image", "n_video"),
    [
        ("text", 0, 0),
        ("image", 1, 0),
        ("video", 0, 1),
        ("text_image", 1, 0),
        ("text_video", 0, 1),
        ("image_video", 1, 1),
        ("text_image_video", 1, 1),
    ],
)
def test_each_combo_emits_exactly_its_declared_media(tmp_path, combo, n_image, n_video):
    task = _task(tmp_path, [_record(combo)])
    parts = (
        task.build_request("Q.", task.load_split("test")[0], Variant("identity"))
        .messages[0]
        .content
    )
    assert sum(isinstance(p, ImagePart) for p in parts) == n_image
    assert sum(isinstance(p, VideoPart) for p in parts) == n_video
    # no empty placeholder ever ships (the HF templates emit a block per item)
    assert any(isinstance(p, TextPart) for p in parts)
    assert all(p.text for p in parts if isinstance(p, TextPart))


def test_raw_template_drops_absent_modality_markers(tmp_path):
    task = _task(
        tmp_path,
        [_record("image")],
        raw_template=True,
    )
    req = task.build_request(
        "{text}{video}Look at the sample.\nQuestion: {question}\nChoices: [{choices}]",
        task.load_split("test")[0],
        Variant("identity"),
    )
    parts = req.messages[0].content
    # {text}/{video} vanish (the sample has neither); the image's payload has no
    # marker left to land on, so it is prepended (base-class rule c) instead of
    # being dropped, and the remaining text stays one merged segment.
    assert [type(p) for p in parts] == [ImagePart, TextPart]
    assert parts[0] == ImagePart(f"{tmp_path}/images/image_0.png")
    assert parts[1] == TextPart(
        "Look at the sample.\nQuestion: What color is the object?\n"
        "Choices: ['red', 'blue', 'green', 'yellow']"
    )


def test_raw_template_keeps_a_passage_the_template_never_names(tmp_path):
    task = _task(tmp_path, [_record("text")], raw_template=True)
    req = task.build_request(
        "Look at the sample.\nQuestion: {question}\nChoices: [{choices}]",
        task.load_split("test")[0],
        Variant("identity"),
    )
    parts = req.messages[0].content
    # a genotype that drops {text} must not silently lose the sample's passage
    # (same rule as media: the payload is prepended, never dropped)
    assert [type(p) for p in parts] == [TextPart, TextPart]
    assert parts[0] == TextPart("The object described is red.")


def test_raw_template_separates_a_passage_from_surrounding_prose(tmp_path):
    task = _task(tmp_path, [_record("text")], raw_template=True)
    req = task.build_request(
        "Read this.{text}Answer the question.\nQuestion: {question}\nChoices: [{choices}]",
        task.load_split("test")[0],
        Variant("identity"),
    )
    parts = req.messages[0].content
    # a seed writes {text} between prose without spaces; the passage must not
    # fuse into it ("this.The object"), and it stays one merged text segment
    assert [type(p) for p in parts] == [TextPart]
    assert parts[0] == TextPart(
        "Read this.\nThe object described is red.\nAnswer the question.\n"
        "Question: What color is the object?\nChoices: ['red', 'blue', 'green', 'yellow']"
    )


def test_raw_template_separates_a_passage_past_markers_that_render_nothing(tmp_path):
    # the shipped SPO seed shape: {text}{image}{video} on a text-only sample
    task = _task(tmp_path, [_record("text")], raw_template=True)
    req = task.build_request(
        "{text}{image}{video}Answer the question.\nQuestion: {question}\nChoices: [{choices}]",
        task.load_split("test")[0],
        Variant("identity"),
    )
    parts = req.messages[0].content
    assert [type(p) for p in parts] == [TextPart]
    assert parts[0] == TextPart(
        "The object described is red.\nAnswer the question.\n"
        "Question: What color is the object?\nChoices: ['red', 'blue', 'green', 'yellow']"
    )


def test_raw_template_requires_question_and_choices(tmp_path):
    task = _task(tmp_path, [_record("text")], raw_template=True)
    with pytest.raises(ValueError, match="question"):
        task.build_request("Answer {choices}", task.load_split("test")[0], Variant("identity"))


def test_demos_never_carry_clips(tmp_path):
    task = _task(tmp_path, [_record("text_image_video")])
    parts = task.demo_media_parts(task.load_split("test")[0])
    assert [type(p) for p in parts] == [ImagePart]


def test_demo_carries_one_image_per_demo(tmp_path):
    # APE writes one "[Image k]" line per demo and its ResourceProfile caps
    # images at demos_per_prompt, so a demo must contribute exactly one part
    rec = _record("image")
    rec["images"] = ["images/a.png", "images/b.png"]
    task = _task(tmp_path, [rec])
    parts = task.demo_media_parts(task.load_split("test")[0])
    assert parts == [ImagePart(f"{tmp_path}/images/a.png")]


def test_text_only_demo_carries_no_media(tmp_path):
    task = _task(tmp_path, [_record("text")])
    assert task.demo_media_parts(task.load_split("test")[0]) == []


def test_loader_rejects_non_dict_item_with_index(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text('["oops"]')
    with pytest.raises(ValueError, match=r"\[0\] must be a JSON object, got str"):
        load_mixed_mcq_samples(data_dir, "test")
