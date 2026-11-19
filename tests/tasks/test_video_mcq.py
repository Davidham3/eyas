"""Video MCQ task contract tests.

Pins the two things the media hook generalizes: the image assembly stays
byte-identical, and the video variant renders a clip part plus one Question
line (the base MCQ skeleton otherwise verbatim).
"""

import json

import pytest

from eyas.backends.base import ImagePart, TextPart, VideoPart
from eyas.tasks.base import Sample, Variant
from eyas.tasks.datasets import load_video_mcq_samples
from eyas.tasks.mcq import McqTask
from eyas.tasks.video_mcq import VideoMcqTask


def _samples():
    return [
        Sample(
            id="v1",
            inputs={
                "video": "videos/v1.mp4",
                "question": "What did he do?",
                "choices": ["run", "jump", "sit", "eat", "sing"],
            },
            references={"label": "run"},
        )
    ]


def test_image_path_is_byte_identical_to_todays_assembly():
    sample = Sample(
        id="i1", inputs={"image": "a.jpg", "choices": ["cat", "dog"]}, references={"label": "cat"}
    )
    req = McqTask(samples=[sample]).build_request("Look at {image}", sample, Variant("identity"))
    assert req.messages[0].content == (
        ImagePart("a.jpg"),
        TextPart("Look at \nChoices: ['cat', 'dog']"),
    )


def test_video_classify_assembly_has_question_line_and_target_video():
    assert VideoMcqTask.default_prompt == "Classify the content of the target video."
    task = VideoMcqTask(samples=_samples())
    req = task.build_request("Answer.", task.load_split("test")[0], Variant("identity"))
    assert req.messages[0].content == (
        VideoPart("videos/v1.mp4"),
        TextPart(
            "Answer.\nQuestion: What did he do?\nChoices: ['run', 'jump', 'sit', 'eat', 'sing']"
        ),
    )


def test_video_raw_template_places_clip_and_fills_question():
    task = VideoMcqTask(samples=_samples(), raw_template=True)
    req = task.build_request(
        "{video}Q: {question}\nOptions: {choices}", task.load_split("test")[0], Variant("identity")
    )
    assert req.messages[0].content == (
        VideoPart("videos/v1.mp4"),
        TextPart("Q: What did he do?\nOptions: 'run', 'jump', 'sit', 'eat', 'sing'"),
    )


def test_video_raw_template_puts_mid_prompt_clip_on_its_own_line():
    task = VideoMcqTask(samples=_samples(), raw_template=True)
    req = task.build_request(
        "Watch {video} carefully.\n{question}\n{choices}",
        task.load_split("test")[0],
        Variant("identity"),
    )
    assert req.messages[0].content == (
        TextPart("Watch "),
        TextPart("\n"),
        VideoPart("videos/v1.mp4"),
        TextPart(" carefully.\nWhat did he do?\n'run', 'jump', 'sit', 'eat', 'sing'"),
    )


def test_video_raw_template_trailing_clip_keeps_the_last_choices_line():
    task = VideoMcqTask(samples=_samples(), raw_template=True)
    req = task.build_request(
        "Answer.\n{choices}\n{question}\n{video}",
        task.load_split("test")[0],
        Variant("identity"),
    )
    assert req.messages[0].content == (
        TextPart("Answer.\n'run', 'jump', 'sit', 'eat', 'sing'\nWhat did he do?\n"),
        TextPart("\n"),
        VideoPart("videos/v1.mp4"),
    )


def test_video_raw_template_prepends_missing_clip_marker():
    task = VideoMcqTask(samples=_samples(), raw_template=True)
    req = task.build_request(
        "Q: {question}\n{choices}", task.load_split("test")[0], Variant("identity")
    )
    assert isinstance(req.messages[0].content[0], VideoPart)


def test_video_raw_template_repeats_clip_when_marker_appears_twice():
    task = VideoMcqTask(samples=_samples(), raw_template=True)
    req = task.build_request(
        "{video}Q: {question} {video}\n{choices}", task.load_split("test")[0], Variant("identity")
    )
    assert [type(p) for p in req.messages[0].content] == [
        VideoPart,
        TextPart,
        TextPart,
        VideoPart,
        TextPart,
    ]


@pytest.mark.parametrize(
    "prompt, missing", [("Q: {question}", "choices"), ("{choices}", "question")]
)
def test_video_raw_template_requires_both_placeholders(prompt, missing):
    task = VideoMcqTask(samples=_samples(), raw_template=True)
    with pytest.raises(ValueError, match=missing):
        task.build_request(prompt, task.load_split("test")[0], Variant("identity"))


def test_video_opro_assembly_uses_the_real_question():
    task = VideoMcqTask(samples=_samples(), instruction_pos="Q_begin")
    text = (
        task.build_request("Answer carefully.", task.load_split("test")[0], Variant("identity"))
        .messages[0]
        .content[1]
        .text
    )
    assert text.startswith("Answer carefully.\nQuestion: What did he do?\nChoices: ")
    assert text.endswith("Target Video:")


def test_video_loader_validates_choices_membership(tmp_path):
    (tmp_path / "test.json").write_text(
        json.dumps([{"video": "v1.mp4", "question": "q?", "choices": ["a", "b"], "answer": "c"}])
    )
    with pytest.raises(ValueError, match="answer"):
        load_video_mcq_samples(tmp_path, "test", tmp_path / "videos")


def test_video_loader_builds_samples_with_default_id(tmp_path):
    (tmp_path / "test.json").write_text(
        json.dumps(
            [{"video": "clips/v1.mp4", "question": "q?", "choices": ["a", "b"], "answer": "b"}]
        )
    )
    samples = load_video_mcq_samples(tmp_path, "test", tmp_path / "videos")
    assert len(samples) == 1
    sample = samples[0]
    assert sample.id == "v1#0"
    assert sample.inputs["video"] == str(tmp_path / "videos" / "clips/v1.mp4")
    assert sample.inputs["question"] == "q?"
    assert sample.inputs["choices"] == ["a", "b"]
    assert sample.references["label"] == "b"


def test_loader_rejects_non_dict_item_with_index(tmp_path):
    (tmp_path / "test.json").write_text("[null]")
    with pytest.raises(ValueError, match=r"\[0\] must be a JSON object, got NoneType"):
        load_video_mcq_samples(tmp_path, "test", tmp_path / "videos")
