"""VqaTask fast tests: request assembly, answer-normalizing parse, VQA accuracy
scoring, split loading, Evaluator end-to-end contract on ScriptedMockBackend. All zero GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eyas.backends.base import ChatResponse
from eyas.backends.mock import ScriptedMockBackend
from eyas.tasks.datasets import load_vqa_samples
from eyas.tasks.protocol import IdentityProtocol
from eyas.tasks.vqa import VqaTask, normalize_vqa_answer


def _sample(answers: list[str], question: str = "What animal is this?") -> Any:
    from eyas.tasks.base import Sample

    return Sample(
        id="img1#0",
        inputs={"image": "/data/images/img1.jpg", "question": question},
        references={"answers": answers},
    )


# ---------------------------------------------------------------------------
# normalize / parse / score
# ---------------------------------------------------------------------------
def test_normalize_vqa_answer_strips_punct_and_case():
    assert normalize_vqa_answer("  A Dog.  ") == "a dog"
    assert normalize_vqa_answer("Yes, it's a cat!") == "yes its a cat"
    assert normalize_vqa_answer("RUNNING\tFAST\n") == "running fast"


def test_parse_response_uses_first_nonempty_line():
    task = VqaTask(samples=[])
    resp = ChatResponse(text="\n  A Dog. \nBecause it barks.")
    assert task.parse_response(resp, _sample(["a dog"])).value == "a dog"
    assert task.parse_response(resp, _sample(["a dog"])).raw == resp.text


def test_parse_response_empty_output_is_none():
    task = VqaTask(samples=[])
    assert task.parse_response(ChatResponse(text="   \n"), _sample(["a dog"])).value is None


def test_score_vqa_metric_min_one_eq_over_three():
    task = VqaTask(samples=[])
    ten = ["yes"] * 4 + ["no"] * 6
    pred = task.parse_response(ChatResponse(text="Yes"), _sample(ten))
    assert task.score(pred, _sample(ten)) == {"accuracy": 1.0}

    two = ["yes"] * 2 + ["no"] * 8
    pred = task.parse_response(ChatResponse(text="Yes"), _sample(two))
    assert task.score(pred, _sample(two)) == {"accuracy": pytest.approx(2 / 3)}

    pred_none = task.parse_response(ChatResponse(text="   "), _sample(ten))
    assert task.score(pred_none, _sample(ten)) == {"accuracy": 0.0}


def test_score_normalizes_both_sides():
    task = VqaTask(samples=[])
    # ground truth "A Dog." and prediction "a dog" are equal after normalization (3/3 -> 1.0)
    answers = ["A Dog.", "A dog.", "a DOG!"]
    pred = task.parse_response(ChatResponse(text="a dog"), _sample(answers))
    assert task.score(pred, _sample(answers)) == {"accuracy": pytest.approx(1.0)}
    # 2/4 annotations equal -> min(1, 2/3) = 2/3
    mixed = ["A Dog.", "A dog.", "cat", "cat"]
    pred = task.parse_response(ChatResponse(text="a dog"), _sample(mixed))
    assert task.score(pred, _sample(mixed)) == {"accuracy": pytest.approx(2 / 3)}


# ---------------------------------------------------------------------------
# build_request
# ---------------------------------------------------------------------------
def test_build_request_renders_scaffold_with_image_first():
    task = VqaTask(samples=[])
    from eyas.tasks.base import Variant

    sample = _sample(["a dog"])
    req = task.build_request("Describe briefly.", sample, Variant(id="identity", index=0))
    assert req.request_id == "img1#0:identity"
    (msg,) = req.messages
    assert msg.role == "user"
    image_part, text_part = msg.content
    assert image_part.image == "/data/images/img1.jpg"
    assert text_part.text == "Describe briefly.\nQuestion: What animal is this?\nAnswer:"


def test_build_request_strips_image_markers_and_rstrips_prompt():
    task = VqaTask(samples=[])
    from eyas.tasks.base import Variant

    req = task.build_request(
        "  Look at {image} carefully.<image> ", _sample(["a dog"]), Variant(id="identity")
    )
    (msg,) = req.messages
    # only rstrip trailing whitespace (same rule as McqTask); leading whitespace kept
    assert msg.content[1].text == "  Look at  carefully.\nQuestion: What animal is this?\nAnswer:"


# ---------------------------------------------------------------------------
# load_split (JSON files)
# ---------------------------------------------------------------------------
def _write_split(tmp_path: Path, items: list[dict[str, Any]], split: str = "train") -> Path:
    data_dir = tmp_path / "vqa"
    data_dir.mkdir(exist_ok=True)
    (data_dir / f"{split}.json").write_text(json.dumps(items), encoding="utf-8")
    return data_dir


def test_load_vqa_samples_resolves_paths_and_default_id(tmp_path: Path):
    data_dir = _write_split(
        tmp_path,
        [
            {"image": "a.jpg", "question": "q1", "answers": ["yes", "no"]},
            {"id": "custom-id", "image": "b.png", "question": "q2", "answers": ["blue"]},
        ],
    )
    samples = load_vqa_samples(data_dir, "train", tmp_path / "images")
    assert [s.id for s in samples] == ["a#0", "custom-id"]
    assert samples[0].inputs == {"image": str(tmp_path / "images" / "a.jpg"), "question": "q1"}
    assert samples[0].references == {"answers": ["yes", "no"]}


def test_load_vqa_samples_missing_file_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_vqa_samples(tmp_path, "train", tmp_path)


def test_load_vqa_samples_rejects_missing_fields(tmp_path: Path):
    data_dir = _write_split(tmp_path, [{"image": "a.jpg", "question": "q1"}])
    with pytest.raises(ValueError, match="answers"):
        load_vqa_samples(data_dir, "train", tmp_path / "images")

    data_dir2 = _write_split(tmp_path, [{"image": "a.jpg", "question": "q1", "answers": []}])
    with pytest.raises(ValueError, match="answers"):
        load_vqa_samples(data_dir2, "train", tmp_path / "images")


def test_load_vqa_samples_rejects_bad_split_and_bad_top_level(tmp_path: Path):
    # the same split whitelist the group loader enforces…
    with pytest.raises(ValueError, match=r"split must be one of \('train', 'test', 'val'\)"):
        load_vqa_samples(tmp_path, "dev", tmp_path)
    # …and the JSON top level must be an array, not a stray object (which
    # would otherwise iterate as keys and fail with a nonsense message)
    data_dir = tmp_path / "TopLevel"
    data_dir.mkdir()
    (data_dir / "train.json").write_text('{"image": "a.jpg"}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"top level must be a JSON array, got dict"):
        load_vqa_samples(data_dir, "train", tmp_path / "images")


def test_vqa_task_load_split_caches_and_rejects_incomplete_config():
    with pytest.raises(ValueError, match="data_dir"):
        VqaTask()
    task = VqaTask(data_dir="missing-dir", images_dir="x")
    with pytest.raises(FileNotFoundError):
        task.load_split("train")


# ---------------------------------------------------------------------------
# Evaluator end-to-end contract (engine assembly, mock backend, zero GPU)
# ---------------------------------------------------------------------------
def test_vqa_task_end_to_end_with_evaluator(tmp_path: Path):
    from eyas.backends.scheduling import run_sync
    from eyas.engine.domain import Prompt
    from tests.support.engine import full_split_plan, make_evaluator

    data_dir = _write_split(
        tmp_path,
        [
            {"id": "q0", "image": "a.jpg", "question": "Is there a dog?", "answers": ["yes"] * 10},
            {"id": "q1", "image": "b.jpg", "question": "Color?", "answers": ["blue"] * 10},
        ],
    )
    task = VqaTask(data_dir=data_dir, images_dir=tmp_path)
    backend = ScriptedMockBackend(responses=["yes", "I see a red car."])

    prompt = Prompt("Answer briefly.")
    evaluator = make_evaluator(
        task,
        backend,
        protocol=IdentityProtocol(),
        produces=frozenset({"accuracy", "parse_fail"}),
    )
    evals = run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))
    ev = evals[0]
    # q0 hit (10/10 equal answers -> 1.0), q1 miss (0 -> 0.0), cross-sample mean 0.5
    assert ev.scores["accuracy"] == pytest.approx(0.5)
    by_id = {o.sample_id: o for o in ev.per_sample}
    assert by_id["q0"].prediction == "yes"
    assert by_id["q1"].prediction == "i see a red car"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_vqa_task_registered():
    import eyas.builtin  # noqa: F401
    from eyas.registry import get

    assert get("task", "vqa") is VqaTask


def test_load_vqa_samples_rejects_non_dict_item_with_index(tmp_path: Path):
    data_dir = tmp_path / "vqa"
    data_dir.mkdir()
    (data_dir / "train.json").write_text(
        '[{"image": "a.jpg", "question": "q", "answers": ["yes"]}, "oops"]', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"\[1\] must be a JSON object, got str"):
        load_vqa_samples(data_dir, "train", tmp_path / "images")
