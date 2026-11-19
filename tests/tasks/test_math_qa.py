"""MathQaTask fast tests: numeric answer parsing, request assembly (plain /
assistant-prefix / raw_template), numeric-exact scoring and the text loader.
All zero GPU, zero network, tmp_path-only data."""

from __future__ import annotations

import json

import pytest

from eyas.backends.base import ChatResponse, TextPart
from eyas.tasks.base import Prediction, Sample, Variant
from eyas.tasks.datasets import load_math_qa_samples
from eyas.tasks.math_qa import MathQaTask, parse_numeric_answer

_SAMPLE = Sample(id="s1", inputs={"question": "2+2?"}, references={"answer": "4"})


def _task(**kw) -> MathQaTask:
    return MathQaTask(samples=[_SAMPLE], **kw)


def test_answer_line_wins_over_earlier_numbers():
    assert parse_numeric_answer("We have 3 apples and 4 pears.\nAnswer: 7") == 7


def test_falls_back_to_last_number_when_no_answer_line():
    assert parse_numeric_answer("3 plus 4 is 7. So 7") == 7


def test_normalizes_currency_and_commas():
    assert parse_numeric_answer("Answer: $1,250") == 1250


def test_no_number_is_unparseable():
    assert parse_numeric_answer("I cannot tell.") is None


def test_build_request_renders_instruction_question_and_answer_cue():
    req = _task().build_request("Solve it.", _SAMPLE, Variant("identity"))
    assert req.messages[0].content == (TextPart("Solve it.\nQuestion: 2+2?\nAnswer:"),)


def test_assistant_prefix_appends_a_continuation_turn():
    req = _task(assistant_prefix="Let's think step by step.").build_request(
        "Solve.", _SAMPLE, Variant("identity")
    )
    assert req.messages[1].role == "assistant"
    assert req.messages[1].content == (TextPart("Let's think step by step."),)


def test_raw_template_requires_question_placeholder():
    with pytest.raises(ValueError, match=r"\{question\}"):
        _task(raw_template=True).build_request("Solve it.", _SAMPLE, Variant("identity"))
    ok = _task(raw_template=True).build_request("Problem: {question}", _SAMPLE, Variant("identity"))
    assert ok.messages[0].content == (TextPart("Problem: 2+2?"),)


def test_score_is_numeric_exact():
    task = _task()
    assert task.score(Prediction(value=4, raw="Answer: 4"), _SAMPLE) == {"accuracy": 1.0}
    assert task.score(Prediction(value=5, raw="Answer: 5"), _SAMPLE) == {"accuracy": 0.0}
    assert task.score(Prediction(value=None, raw="?"), _SAMPLE) == {"accuracy": 0.0}


def test_loader_validates_and_defaults_ids(tmp_path):
    (tmp_path / "test.json").write_text('[{"question": "1+1?", "answer": "2"}]')
    samples = load_math_qa_samples(tmp_path, "test")
    assert samples[0].id == "test#0" and samples[0].references == {"answer": "2"}
    (tmp_path / "val.json").write_text('[{"answer": "2"}]')
    with pytest.raises(ValueError, match="missing required field 'question'"):
        load_math_qa_samples(tmp_path, "val")


# ---------------------------------------------------------------------------
# Additional contract checks (parse_response, edge cases, loader, registry)
# ---------------------------------------------------------------------------
def test_parse_response_parses_the_text():
    pred = _task().parse_response(ChatResponse(text="So 2+2 is 4.\nAnswer: 4"), _SAMPLE)
    assert pred.value == 4
    assert pred.raw == "So 2+2 is 4.\nAnswer: 4"
    assert _task().parse_response(ChatResponse(text="no idea"), _SAMPLE).value is None


def test_media_markers_are_stripped_from_the_instruction():
    req = _task().build_request("Look {image} at <image> {video} it.", _SAMPLE, Variant("identity"))
    assert req.messages[0].content == (TextPart("Look  at   it.\nQuestion: 2+2?\nAnswer:"),)


def test_empty_instruction_starts_at_the_question_line():
    req = _task().build_request("  ", _SAMPLE, Variant("identity"))
    assert req.messages[0].content == (TextPart("Question: 2+2?\nAnswer:"),)
    assert req.request_id == "s1:identity"


def test_no_assistant_turn_without_prefix():
    assert len(_task().build_request("Solve.", _SAMPLE, Variant("identity")).messages) == 1


def test_raw_template_rejects_choices_placeholder():
    with pytest.raises(ValueError, match=r"\{choices\}"):
        _task(raw_template=True).build_request(
            "Problem: {question}\nChoices: {choices}", _SAMPLE, Variant("identity")
        )


def test_score_tolerates_float_vs_int_gold():
    sample = Sample(id="s2", inputs={"question": "1/2?"}, references={"answer": "0.5"})
    assert _task().score(Prediction(value=0.5, raw=""), sample) == {"accuracy": 1.0}
    assert _task().score(Prediction(value=0.5001, raw=""), sample) == {"accuracy": 0.0}


def test_loader_fails_fast_on_missing_file_and_bad_top_level(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_math_qa_samples(tmp_path, "test")
    (tmp_path / "train.json").write_text('{"question": "1+1?", "answer": "2"}')
    with pytest.raises(ValueError, match="top level must be a JSON array"):
        load_math_qa_samples(tmp_path, "train")
    with pytest.raises(ValueError, match="split must be one of"):
        load_math_qa_samples(tmp_path, "dev")


def test_task_requires_data_source_and_is_registered():
    with pytest.raises(ValueError, match="data_dir"):
        MathQaTask()
    import eyas.builtin  # noqa: F401
    from eyas.registry import get

    assert get("task", "math_qa") is MathQaTask


def test_math_qa_end_to_end_with_evaluator(tmp_path):
    """Text-only requests survive the whole engine assembly (planner + mock backend)."""
    from eyas.backends.mock import ScriptedMockBackend
    from eyas.backends.scheduling import run_sync
    from eyas.engine.domain import Prompt
    from eyas.tasks.protocol import IdentityProtocol
    from tests.support.engine import full_split_plan, make_evaluator

    items = [
        {"id": "m0", "question": "1+1?", "answer": "2"},
        {"id": "m1", "question": "3*3?", "answer": "9"},
    ]
    (tmp_path / "train.json").write_text(json.dumps(items), encoding="utf-8")
    task = MathQaTask(data_dir=tmp_path)
    backend = ScriptedMockBackend(responses=["Answer: 2", "I think 8."])

    prompt = Prompt("Solve it.")
    evaluator = make_evaluator(
        task,
        backend,
        protocol=IdentityProtocol(),
        produces=frozenset({"accuracy", "parse_fail"}),
    )
    evals = run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))
    ev = evals[0]
    # m0 hit, m1 miss (fallback parses 8 from the reasoning) -> cross-sample mean 0.5
    assert ev.scores["accuracy"] == pytest.approx(0.5)
    by_id = {o.sample_id: o for o in ev.per_sample}
    assert by_id["m1"].prediction == 8


def test_loader_rejects_non_dict_item_with_index(tmp_path):
    (tmp_path / "train.json").write_text('[{"question": "1+1?", "answer": "2"}, 42]')
    with pytest.raises(ValueError, match=r"\[1\] must be a JSON object, got int"):
        load_math_qa_samples(tmp_path, "train")
