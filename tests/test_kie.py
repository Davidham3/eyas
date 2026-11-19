"""KieTask fast tests: normalization, tolerant parsing, greedy-aligned metrics, request
rendering, split loading, Evaluator end-to-end contract on ScriptedMockBackend. All zero GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eyas.backends.base import ChatResponse, ImagePart
from eyas.backends.mock import ScriptedMockBackend
from eyas.tasks.base import Sample, Variant
from eyas.tasks.kie import (
    KieTask,
    load_kie_samples,
    normalize_field_value,
    parse_kie_entities,
    score_entities,
)
from eyas.tasks.protocol import IdentityProtocol

_DIGITS = {"phonenum": "digits"}


def _sample(
    entities: list[dict[str, str]],
    fields: list[str] | None = None,
    image: str = "/data/images/img1.jpg",
) -> Any:
    union = fields if fields is not None else sorted({f for e in entities for f in e})
    return Sample(
        id="train_0000",
        inputs={"image": image, "fields": union, "group": "n1"},
        references={"entities": entities, "label": json.dumps(entities, ensure_ascii=False)},
    )


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
def test_normalize_digits_keeps_only_digits():
    assert normalize_field_value("138-1234 5678", "digits") == "13812345678"
    assert normalize_field_value("no phone", "digits") == ""


def test_normalize_text_strips_cjk_and_ascii_punct():
    assert normalize_field_value("  A Dog.  ", "text") == "a dog"
    assert normalize_field_value("Lake Bay (Phase 1) Shop!", "text") == "lake bay phase 1 shop"
    assert normalize_field_value("East Street\nWest Mall", "text") == "east street west mall"


def test_normalize_unknown_kind_fails_fast():
    with pytest.raises(ValueError, match="normalizer kind"):
        normalize_field_value("x", "phonenum")


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def test_parse_plain_array_and_value_coercion():
    assert parse_kie_entities('[{"name": "a", "n": 3}]') == [{"name": "a", "n": "3"}]


def test_parse_tolerates_fence_and_noise():
    text = '```json\n[{"name": "a"}]\n```\nThe above is the result.'
    assert parse_kie_entities(text) == [{"name": "a"}]


def test_parse_bare_object_wraps_single_record():
    assert parse_kie_entities('{"name": "a"}') == [{"name": "a"}]


def test_parse_empty_array_is_valid_zero_entities():
    assert parse_kie_entities("[]") == []


def test_parse_invalid_structures_return_none():
    assert parse_kie_entities("not JSON") is None
    assert parse_kie_entities('[{"name": {"x": 1}}]') is None
    assert parse_kie_entities("[{}]") is None
    assert parse_kie_entities('["scalar"]') is None


def test_parse_repairs_mismatched_closer():
    # `]` closing an object inside a JSON array (a shape models actually emit)
    assert parse_kie_entities(
        '```json\n[\n\t{"address": "No. 10-5 Taixing South Road, Hedong, Tianjin"]\n]\n```'
    ) == [{"address": "No. 10-5 Taixing South Road, Hedong, Tianjin"}]
    # brackets inside string literals are untouched by balancing
    assert parse_kie_entities('[{"address": "in-[room] {gear} shop"}]') == [
        {"address": "in-[room] {gear} shop"}
    ]
    # truncated trailing opener is auto-closed
    assert parse_kie_entities('[{"address": "abc"') == [{"address": "abc"}]
    # genuine breakage balancing cannot fix (missing colon) still lands as parse_fail
    assert parse_kie_entities('[{"address" "a"}]') is None


def test_parse_single_field_wraps_bare_strings():
    # real CoT shape: with a single-field spec the model emits a bare-string array
    text = 'Reasoning... final result:\n```json\n["Baogongmiao Store"]\n```'
    assert parse_kie_entities(text, single_field="address") == [{"address": "Baogongmiao Store"}]
    # a bare top-level string is wrapped the same way; all elements are kept
    assert parse_kie_entities('"Store A, Changzhou"', single_field="address") == [
        {"address": "Store A, Changzhou"}
    ]
    assert parse_kie_entities('["a", "b"]', single_field="address") == [
        {"address": "a"},
        {"address": "b"},
    ]
    # without single_field (multi-field spec) a bare string cannot be attributed, still rejected
    assert parse_kie_entities('["a"]') is None


def test_parse_rejects_scalar_documents():
    # top-level scalar documents cannot be records; a bare string without a
    # single_field spec cannot be attributed either
    assert parse_kie_entities('"Store A"') is None
    assert parse_kie_entities("42") is None
    assert parse_kie_entities("null") is None


def test_parse_empty_text_is_parse_fail():
    assert parse_kie_entities("") is None
    assert parse_kie_entities("   \n ") is None


def test_parse_scalar_value_kinds():
    # numeric scalars coerce to str (the phone-number-as-number shape);
    # bool/null/nested values are structurally invalid -> parse_fail
    assert parse_kie_entities('[{"phonenum": 13812345678}]') == [{"phonenum": "13812345678"}]
    assert parse_kie_entities('[{"n": 1.5}]') == [{"n": "1.5"}]
    assert parse_kie_entities('[{"n": true}]') is None
    assert parse_kie_entities('[{"n": null}]') is None
    assert parse_kie_entities('[{"n": [1]}]') is None


def test_parse_single_field_bare_string_keeps_value_verbatim():
    # surviving coercion pin: a bare string under single_field is already a
    # scalar — the value passes through unchanged (no re-coercion step)
    assert parse_kie_entities('["13800138000"]', single_field="phonenum") == [
        {"phonenum": "13800138000"}
    ]


def test_parse_repairs_escaped_quote_inside_string_literal():
    # the balancer's escape tracking: the quoted ] must not be seen as a
    # closer, the object's missing } is appended
    assert parse_kie_entities('[{"address": "x\\"}"]') == [{"address": 'x"}'}]
    # an unterminated literal swallows the appended closers into the string:
    # unrepairable, stays parse_fail (the repair never invents a quote)
    assert parse_kie_entities('[{"address": "abc') is None


def test_parse_response_derives_single_field_from_sample():
    task = KieTask(samples=[])
    resp = ChatResponse(text='["Baogongmiao Store"]', finish_reason="stop")
    single = task.parse_response(resp, _sample([{"address": "x"}]))
    assert single.value == [{"address": "Baogongmiao Store"}]
    multi = task.parse_response(resp, _sample([{"name": "a", "phonenum": "138"}]))
    assert multi.value is None


# ---------------------------------------------------------------------------
# score: alignment and the three metrics
# ---------------------------------------------------------------------------
def test_score_perfect_match_with_phone_digits_normalizer():
    ref = [{"name": "Zhangsan Store", "phonenum": "13812345678"}]
    pred = [{"phonenum": "138-1234-5678", "name": "Zhangsan Store"}]
    assert score_entities(pred, ref, _DIGITS) == {
        "field_acc": 1.0,
        "strict_acc": 1.0,
        "char_sim": 1.0,
    }


def test_score_phone_without_normalizer_misses_but_aligns():
    ref = [{"phonenum": "13812345678"}]
    # text normalization strips '-' but keeps spaces - only the space-separated form is a true miss
    pred = [{"phonenum": "138 1234 5678"}]
    scores = score_entities(pred, ref, {})
    assert scores["field_acc"] == 0.0
    assert scores["strict_acc"] == 0.0
    assert 0.0 < scores["char_sim"] < 1.0


def test_score_extra_pred_record_penalizes():
    scores = score_entities([{"name": "a"}, {"name": "b"}], [{"name": "a"}], {})
    assert scores["field_acc"] == pytest.approx(0.5)
    assert scores["strict_acc"] == 0.0
    assert scores["char_sim"] == pytest.approx(0.5)


def test_score_missing_pred_record_penalizes():
    scores = score_entities([{"name": "a"}], [{"name": "a"}, {"name": "b"}], {})
    assert scores["field_acc"] == pytest.approx(0.5)
    assert scores["strict_acc"] == 0.0


def test_score_duplicate_values_align_one_to_one():
    scores = score_entities([{"name": "x"}], [{"name": "x"}, {"name": "x"}], {})
    assert scores["field_acc"] == pytest.approx(0.5)
    assert scores["strict_acc"] == 0.0


def test_score_near_miss_gets_char_sim_partial_credit():
    scores = score_entities([{"name": "Pharmac"}], [{"name": "Pharmacy"}], {})
    assert scores["field_acc"] == 0.0
    assert scores["strict_acc"] == 0.0
    # 8 chars vs 7-char prefix: ratio = 2*7/15
    assert scores["char_sim"] == pytest.approx(14 / 15)


def test_score_totally_different_value_zero_everything():
    scores = score_entities([{"name": "bbbb"}], [{"name": "aaaa"}], {})
    assert scores == {"field_acc": 0.0, "strict_acc": 0.0, "char_sim": 0.0}


def test_score_extra_pred_field_breaks_strict():
    scores = score_entities([{"name": "a", "note": "z"}], [{"name": "a"}], {})
    assert scores["field_acc"] == pytest.approx(0.5)
    assert scores["strict_acc"] == 0.0
    assert scores["char_sim"] == pytest.approx(0.5)


def test_score_empty_vs_empty_full_score_and_one_sided_zero():
    assert score_entities([], [], {}) == {
        "field_acc": 1.0,
        "strict_acc": 1.0,
        "char_sim": 1.0,
    }
    assert score_entities([], [{"name": "a"}], {})["field_acc"] == 0.0
    assert score_entities([{"name": "a"}], [], {})["field_acc"] == 0.0


def test_score_pred_record_missing_ref_field_contributes_nothing():
    # alignment hits a ref field the prediction never mentions: no hit, no
    # character similarity credit — it is simply absent
    scores = score_entities([{"name": "a"}], [{"name": "a", "note": "z"}], {})
    assert scores == {"field_acc": 0.5, "strict_acc": 0.0, "char_sim": 0.5}


def test_score_none_pred_all_zero():
    assert score_entities(None, [{"name": "a"}], {}) == {
        "field_acc": 0.0,
        "strict_acc": 0.0,
        "char_sim": 0.0,
    }


def test_score_empty_entity_record_fails_fast_with_index() -> None:
    # an empty record carries no scorable fields: it must raise naming the side
    # and the record index, never leak a ZeroDivisionError from the alignment
    with pytest.raises(ValueError, match=r"ref record \[0\] is empty"):
        score_entities([{}], [{}], {})
    with pytest.raises(ValueError, match=r"pred record \[1\] is empty"):
        score_entities([{"name": "a"}, {}], [{"name": "a"}], {})
    with pytest.raises(ValueError, match=r"ref record \[1\] is empty"):
        score_entities([{"name": "a"}], [{"name": "a"}, {}], {})


# ---------------------------------------------------------------------------
# build_request
# ---------------------------------------------------------------------------
def test_build_request_renders_spec_and_image_marker():
    task = KieTask(samples=[])
    sample = _sample([{"name": "a"}], fields=["name", "phonenum"])
    req = task.build_request("Extract.{image}Field spec: {choices}", sample, Variant(id="identity"))
    assert req.request_id == "train_0000:identity"
    (msg,) = req.messages
    # {image} splits in place: TextPart("Extract.") + ImagePart + TextPart("Field spec: ...")
    text_part_1, image_part, text_part_2 = msg.content
    assert image_part.image == "/data/images/img1.jpg"
    assert text_part_1.text == "Extract."
    assert text_part_2.text.startswith("Field spec:")
    assert '"name": "..."' in text_part_2.text
    assert '"phonenum": "..."' in text_part_2.text


def test_build_request_appends_spec_tail_when_placeholder_lost():
    task = KieTask(samples=[])
    sample = _sample([{"name": "a"}], fields=["name"])
    req = task.build_request("Instruction missing its placeholders", sample, Variant(id="identity"))
    (msg,) = req.messages
    image_part, text_part = msg.content
    assert image_part.image == "/data/images/img1.jpg"
    assert text_part.text.startswith("Instruction missing its placeholders\nOutput format:")
    assert '"name": "..."' in text_part.text


def test_build_request_strips_image_tag_and_defaults_image_first():
    task = KieTask(samples=[])
    sample = _sample([{"name": "a"}], fields=["name"])
    # no {image} marker -> image prepended; <image> stripped; {choices} inline (no tail append)
    req = task.build_request("<image>Answer. Field spec: {choices}", sample, Variant(id="identity"))
    (msg,) = req.messages
    image_part, text_part = msg.content
    assert image_part.image == "/data/images/img1.jpg"
    assert text_part.text == 'Answer. Field spec: [{"name": "..."}]'


def test_demo_media_parts_returns_the_sample_image():
    task = KieTask(samples=[])
    sample = _sample([{"name": "a"}], fields=["name"])
    parts = task.demo_media_parts(sample)
    assert len(parts) == 1
    assert isinstance(parts[0], ImagePart)
    assert parts[0].image == "/data/images/img1.jpg"


# ---------------------------------------------------------------------------
# load_split (JSON files + fail fast)
# ---------------------------------------------------------------------------
def _write_split(tmp_path: Path, items: list[dict[str, Any]], split: str = "train") -> Path:
    data_dir = tmp_path / "kie"
    data_dir.mkdir(exist_ok=True)
    for item in items:
        Path(item["image"]).touch()
    (data_dir / f"{split}.json").write_text(json.dumps(items), encoding="utf-8")
    return data_dir


def test_load_kie_samples_derives_fields_group_label(tmp_path: Path):
    img_a = str(tmp_path / "a.jpg")
    img_b = str(tmp_path / "b.jpg")
    data_dir = _write_split(
        tmp_path,
        [
            {
                "image": img_a,
                "entities": [
                    {"phonenum": "138", "name": "Store A"},
                    {"phonenum": "139", "name": "Store B", "address": "1 Some Road"},
                ],
            },
            {"image": img_b, "entities": []},
        ],
    )
    samples = load_kie_samples(data_dir, "train")
    assert [s.id for s in samples] == ["train_0000", "train_0001"]
    assert samples[0].inputs["fields"] == ["address", "name", "phonenum"]
    assert samples[0].inputs["group"] == "n2+"
    assert samples[1].inputs["group"] == "n0"
    assert samples[1].references["label"] == "[]"
    assert json.loads(samples[0].references["label"])[0] == {"phonenum": "138", "name": "Store A"}


def test_load_kie_samples_missing_file_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_kie_samples(tmp_path, "train")


def test_load_kie_samples_rejects_bad_structures(tmp_path: Path):
    img = str(tmp_path / "a.jpg")
    Path(img).touch()
    cases: list[tuple[Any, str]] = [
        ({"entities": []}, "image"),
        ({"image": img}, "entities"),
        ({"image": img, "entities": "no"}, "entities"),
        ({"image": img, "entities": [None]}, "non-empty object"),
        ({"image": img, "entities": [{"name": {"x": 1}}]}, "scalar"),
        ({"image": str(tmp_path / "missing.jpg"), "entities": []}, "image does not exist"),
    ]
    for record, match in cases:
        data_dir = tmp_path / "kie_bad"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "train.json").write_text(json.dumps([record]), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_kie_samples(data_dir, "train")


def test_load_kie_samples_rejects_non_list_and_empty(tmp_path: Path):
    data_dir = tmp_path / "kie_obj"
    data_dir.mkdir()
    (data_dir / "train.json").write_text('{"image": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_kie_samples(data_dir, "train")

    data_dir2 = tmp_path / "kie_empty"
    data_dir2.mkdir()
    (data_dir2 / "train.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="loaded no samples"):
        load_kie_samples(data_dir2, "train")


def test_load_kie_samples_rejects_bad_split_name(tmp_path: Path):
    with pytest.raises(ValueError, match="split must be one of"):
        load_kie_samples(tmp_path, "traen")


def test_load_kie_samples_rejects_non_object_record(tmp_path: Path):
    data_dir = tmp_path / "kie_rec"
    data_dir.mkdir()
    (data_dir / "train.json").write_text("[42]", encoding="utf-8")
    with pytest.raises(ValueError, match=r"train\.json\[0\] must be an object"):
        load_kie_samples(data_dir, "train")


def test_load_kie_samples_scalar_field_error_names_field_and_type(tmp_path: Path):
    img = str(tmp_path / "a.jpg")
    Path(img).touch()
    data_dir = tmp_path / "kie_scalar"
    data_dir.mkdir()
    record = {"image": img, "entities": [{"name": {"x": 1}, "phonenum": True}]}
    (data_dir / "train.json").write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"entities\[0\] field 'name' value must be a "
        r"string/numeric scalar, got dict",
    ):
        load_kie_samples(data_dir, "train")


def test_kie_task_load_split_with_explicit_samples_returns_copy():
    sample = _sample([{"name": "a"}])
    task = KieTask(samples=[sample])
    first = task.load_split("train")
    assert first == [sample]
    first.append(sample)
    assert task.load_split("train") == [sample]


def test_kie_task_config_validation_and_split_cache(tmp_path: Path):
    with pytest.raises(ValueError, match="data_dir"):
        KieTask()
    with pytest.raises(ValueError, match="field_normalizers"):
        KieTask(data_dir="x", field_normalizers={"phonenum": "sha256"})
    task = KieTask(data_dir=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        task.load_split("train")


# ---------------------------------------------------------------------------
# Evaluator end-to-end contract (engine assembly)
# ---------------------------------------------------------------------------
def test_kie_task_end_to_end_with_evaluator(tmp_path: Path):
    from eyas.backends.scheduling import run_sync
    from eyas.engine.domain import Prompt
    from tests.support.engine import full_split_plan, make_evaluator

    img_a = str(tmp_path / "a.jpg")
    img_b = str(tmp_path / "b.jpg")
    data_dir = _write_split(
        tmp_path,
        [
            {"image": img_a, "entities": [{"name": "Foo Store", "phonenum": "13800"}]},
            {"image": img_b, "entities": []},
        ],
    )
    task = KieTask(data_dir=data_dir, field_normalizers=_DIGITS)
    backend = ScriptedMockBackend(responses=['[{"name": "Foo Store", "phonenum": "13800"}]', "[]"])

    prompt = Prompt("Extract.{image}Field spec: {choices}")
    evaluator = make_evaluator(
        task,
        backend,
        protocol=IdentityProtocol(),
        produces=frozenset({"field_acc", "strict_acc", "char_sim", "parse_fail"}),
    )
    evals = run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))
    ev = evals[0]
    assert ev.scores["field_acc"] == pytest.approx(1.0)
    assert ev.scores["strict_acc"] == pytest.approx(1.0)
    assert ev.scores["char_sim"] == pytest.approx(1.0)
    assert ev.scores["parse_fail"] == 0.0
    by_id = {o.sample_id: o for o in ev.per_sample}
    assert by_id["train_0000"].prediction == [{"name": "Foo Store", "phonenum": "13800"}]
    assert by_id["train_0001"].prediction == []


def test_kie_task_end_to_end_parse_fail_scores_zero(tmp_path: Path):
    from eyas.backends.scheduling import run_sync
    from eyas.engine.domain import Prompt
    from tests.support.engine import full_split_plan, make_evaluator

    img_a = str(tmp_path / "a.jpg")
    img_b = str(tmp_path / "b.jpg")
    data_dir = _write_split(
        tmp_path,
        [
            {"image": img_a, "entities": [{"name": "Foo Store"}]},
            {"image": img_b, "entities": []},
        ],
    )
    task = KieTask(data_dir=data_dir)
    backend = ScriptedMockBackend(responses=["The model emitted unparseable output", "[]"])

    prompt = Prompt("Extract")
    evaluator = make_evaluator(
        task,
        backend,
        protocol=IdentityProtocol(),
        produces=frozenset({"field_acc", "strict_acc", "char_sim", "parse_fail"}),
    )
    evals = run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))
    assert evals[0].scores["field_acc"] == pytest.approx(0.5)  # empty-vs-empty 1.0 lifts the mean
    assert evals[0].scores["parse_fail"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Registry and objectives
# ---------------------------------------------------------------------------
def test_kie_task_registered():
    import eyas.builtin  # noqa: F401
    from eyas.registry import get

    assert get("task", "kie") is KieTask
    assert [obj.key for obj in KieTask.objectives] == ["field_acc", "strict_acc", "char_sim"]


def test_kie_default_normalizer_digits_for_phonenum():
    # default policy: phonenum -> digits (separator spelling does not affect equality), rest text
    task = KieTask(samples=[])
    sample = _sample([{"name": "Zhangsan Store", "phonenum": "13812345678"}])
    pred = task.parse_response(
        ChatResponse(text='[{"phonenum": "138-1234-5678", "name": "Zhangsan Store"}]'), sample
    )
    assert task.score(pred, sample) == {
        "field_acc": 1.0,
        "strict_acc": 1.0,
        "char_sim": 1.0,
    }
