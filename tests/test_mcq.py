"""MCQ task / identity protocol / group-style dataset loading contract tests.

The end-to-end case replays the evaluation flow verbatim on a dual mock backend
(classify + judge): classify (with image) -> raw output -> text-only LLM judge
-> match_prediction, asserting hand-computed metrics (accuracy is per-sample 0/1, averaged).
"""

import ast
import re
from pathlib import Path

import pytest

from eyas.backends.base import ImagePart, TextPart
from eyas.backends.mock import ScriptedMockBackend
from eyas.judges.mcq_judge import (
    McqLLMJudge,
    build_judge_prompt,
    match_judge_result_cot,
)
from eyas.registry import get
from eyas.tasks.base import Sample
from eyas.tasks.datasets import (
    build_mcq_samples,
    derive_sample_seed,
    load_group_split,
    load_mcq_samples,
    resolve_dataset_paths,
    stratified_sample_train,
)
from eyas.tasks.mcq import McqTask, gen_scorer_prompt, match_prediction
from eyas.tasks.protocol import IdentityProtocol

CHOICES = ["boeing", "airbus", "embraer"]
CHOICES_STR = ", ".join(f"'{c}'" for c in CHOICES)  # the rendering build_request must emit
INSTRUCTION = "Classify the aircraft in the image."


def make_mcq_samples() -> list[Sample]:
    return [
        Sample(
            id="s1",
            inputs={"image": "/data/images/001.g/1.jpg", "choices": CHOICES, "group": "g"},
            references={"label": "boeing"},
        ),
        Sample(
            id="s2",
            inputs={"image": "/data/images/001.g/2.jpg", "choices": CHOICES, "group": "g"},
            references={"label": "airbus"},
        ),
    ]


# ---------------------------------------------------------------------------
# match_prediction (last-position-wins rule)
# ---------------------------------------------------------------------------
def test_match_prediction_rules():
    assert match_prediction("boeing", CHOICES) == "boeing"
    assert match_prediction('  "boeing."  ', CHOICES) == "boeing"  # strips quotes/periods
    assert match_prediction("A Boeing 747 aircraft", CHOICES) == "boeing"  # substring + case
    # both choices are substrings: mutual-containment cluster -> standalone last occurrence
    assert (
        match_prediction("it is an airbus a320 plane", ["airbus", "airbus a320"]) == "airbus a320"
    )
    assert match_prediction("completely unrelated text", CHOICES) is None


def test_match_prediction_containment_cluster_resolves_by_standalone_last():
    # real CoT trace shape: the choices nest ("visible" ⊂ "invisible"); the response
    # enumerates both definitions and answers only at the end - inherited hits are not evidence.
    choices = ["visible", "invisible"]
    cot_artifact = (
        '1. Definitions:\n- "invisible": the image does not show a screen.\n'
        '- "visible": the image shows screen content.\n'
        "2. Analysis: moire patterns are present.\n"
        'In conclusion, this image belongs to the "visible" category.\n'
        "---\nThe final answer is:\nvisible"
    )
    assert match_prediction(cot_artifact, choices) == "visible"
    assert match_prediction("In summary. The final answer is: ['invisible']", choices) == (
        "invisible"
    )
    # a contained choice with no standalone occurrence (pure suffix spillover) never wins
    assert match_prediction("this is invisible", choices) == "invisible"
    # cluster resolution with chained containment (a ⊂ b ⊂ c)
    assert match_prediction("talk about b first, then ab, final answer: ab", ["a", "b", "ab"]) == (
        "ab"
    )
    assert match_prediction("final answer: a", ["a", "ab"]) == "a"
    # single hit: only one choice appears in the text, the length rule does not apply
    assert (
        match_prediction(
            "not a warbler but a black footed albatross",
            ["sooty albatross", "black footed albatross"],
        )
        == "black footed albatross"
    )


def test_containment_capitalized_options():
    assert match_prediction("It is invisible.", ["Visible", "Invisible"]) == "Invisible"


def test_containment_mixed_case_span_count():
    # "Visible" occurs twice as substring of the answer text; the longer
    # containment cluster must still win regardless of casing
    assert match_prediction("VISIBLE and invisible", ["Visible", "Invisible"]) == "Invisible"


def test_match_prediction_last_position_wins_over_length():
    # real quickstart trace shape: the response enumerates candidate breed definitions
    # first (longer choice names appear there) and answers only at the end - last position wins.
    choices = ["Abyssinian", "British Shorthair", "Sphynx"]
    cot = (
        "The cat has a short, fine, ticked coat.\n"
        "- **British Shorthair** – round face, dense plush coat.\n"
        "- **Sphynx** – hairless; obviously not.\n\n"
        "✅ The best match is **Abyssinian**.\n\n---\n\n"
        "**Final Answer: Abyssinian**"
    )
    assert match_prediction(cot, choices) == "Abyssinian"

    # length tie-break anchor: a longer choice appearing later still wins (length as backup)
    assert (
        match_prediction(
            "an abyssinian, or more precisely an egyptian mau", ["abyssinian", "egyptian mau"]
        )
        == "egyptian mau"
    )


# ---------------------------------------------------------------------------
# McqTask request construction ({image} marker stripping + Choices rendering)
# ---------------------------------------------------------------------------
def test_mcq_build_request_strips_image_marker_and_keeps_choices():
    task = McqTask(samples=make_mcq_samples())
    sample = make_mcq_samples()[0]
    variant = IdentityProtocol().variants(sample)[0]
    req = task.build_request("{image}" + INSTRUCTION, sample, variant)
    assert len(req.messages) == 1 and req.messages[0].role == "user"
    parts = req.messages[0].content
    assert isinstance(parts[0], ImagePart) and parts[0].image == "/data/images/001.g/1.jpg"
    text = parts[1].text
    assert isinstance(parts[1], TextPart)
    assert "{image}" not in text and INSTRUCTION in text
    # identity single variant: choice order is the original order
    assert text.endswith(f"Choices: {CHOICES}")
    assert req.request_id == "s1:identity"


# ---------------------------------------------------------------------------
# gen_scorer_prompt (OPRO scorer assembly) — one test per decision branch.
# Caller-level inputs: McqTask.build_request passes (instruction, choices,
# instruction_pos, include_qa) through verbatim (instruction_pos validated to
# the 4 positions by __init__); instruction can be empty once the image
# markers are stripped. QUESTION is the fixed question line the module
# docstring pins: question + Choices (", ".join(f"'{c}'")) + 'Target Image:'.
# ---------------------------------------------------------------------------
QUESTION = (
    "Classify the content of the target image.\n"
    "Choices: ['boeing', 'airbus', 'embraer']\n"
    "Target Image:"
)


def test_scorer_prompt_no_qa_instruction_before_question():
    # before_Q and Q_begin share the no-QA rendering (instruction leads)
    for pos in ("before_Q", "Q_begin"):
        assert gen_scorer_prompt(INSTRUCTION, CHOICES, pos, include_qa=False) == (
            INSTRUCTION + "\n" + QUESTION
        )


def test_scorer_prompt_no_qa_instruction_after_question():
    # Q_end and A_begin share the no-QA rendering (instruction trails the
    # 'Target Image:' line) — verbatim-port semantics
    for pos in ("Q_end", "A_begin"):
        assert gen_scorer_prompt(INSTRUCTION, CHOICES, pos, include_qa=False) == (
            QUESTION + "\n" + INSTRUCTION
        )


def test_scorer_prompt_no_qa_empty_instruction_is_question_only():
    for pos in ("before_Q", "Q_begin", "Q_end", "A_begin"):
        assert gen_scorer_prompt("", CHOICES, pos, include_qa=False) == QUESTION


def test_scorer_prompt_qa_before_Q():
    assert gen_scorer_prompt(INSTRUCTION, CHOICES, "before_Q", include_qa=True) == (
        INSTRUCTION + "\n" + "Q: " + QUESTION + "\n\nA:"
    )


def test_scorer_prompt_qa_Q_begin():
    # instruction sits inside the Q: shell, before the question
    assert gen_scorer_prompt(INSTRUCTION, CHOICES, "Q_begin", include_qa=True) == (
        "Q: " + INSTRUCTION + "\n" + QUESTION + "\n\nA:"
    )


def test_scorer_prompt_qa_Q_end():
    assert gen_scorer_prompt(INSTRUCTION, CHOICES, "Q_end", include_qa=True) == (
        "Q: " + QUESTION + "\n" + INSTRUCTION + "\n\nA:"
    )


def test_scorer_prompt_qa_A_begin():
    # instruction continues the answer line after 'A: '
    assert gen_scorer_prompt(INSTRUCTION, CHOICES, "A_begin", include_qa=True) == (
        f"Q: {QUESTION}\n\nA: {INSTRUCTION}"
    )


def test_scorer_prompt_qa_empty_instruction_is_bare_shell():
    # every position collapses to the same bare Q/A shell with no instruction
    for pos in ("before_Q", "Q_begin", "Q_end", "A_begin"):
        assert gen_scorer_prompt("", CHOICES, pos, include_qa=True) == ("Q: " + QUESTION + "\n\nA:")


def test_scorer_prompt_rejects_unknown_position():
    # unknown instruction_pos raises ValueError, and every valid position is
    # enumerated in the message — the split function refuses loudly, never coercing
    for include_qa in (False, True):
        with pytest.raises(
            ValueError, match=r"gen_scorer_prompt\.instruction_pos must be one of .* got 'middle'"
        ):
            gen_scorer_prompt(INSTRUCTION, CHOICES, "middle", include_qa=include_qa)


def test_mcq_build_request_renders_opro_scorer_assembly():
    # the only caller path: instruction_pos switches build_request from the
    # compact assembly to gen_scorer_prompt, with include_qa passed through
    sample = make_mcq_samples()[0]
    variant = IdentityProtocol().variants(sample)[0]
    task = McqTask(samples=make_mcq_samples(), instruction_pos="Q_end")
    req = task.build_request("{image}" + INSTRUCTION, sample, variant)
    parts = req.messages[0].content
    assert isinstance(parts[0], ImagePart)
    assert isinstance(parts[1], TextPart)
    assert parts[1].text == QUESTION + "\n" + INSTRUCTION
    task_qa = McqTask(samples=make_mcq_samples(), instruction_pos="A_begin", include_qa=True)
    req_qa = task_qa.build_request(INSTRUCTION, sample, variant)
    assert req_qa.messages[0].content[1].text == f"Q: {QUESTION}\n\nA: {INSTRUCTION}"


def test_mcq_parse_and_score():
    task = McqTask(samples=make_mcq_samples())
    sample = make_mcq_samples()[0]

    class _Resp:
        text = "The answer is Boeing"

    pred = task.parse_response(_Resp(), sample)
    assert pred.value == "boeing"
    assert task.score(pred, sample) == {"accuracy": 1.0}

    class _Bad:
        text = "no idea"

    pred_bad = task.parse_response(_Bad(), sample)
    assert pred_bad.value is None
    assert task.score(pred_bad, sample) == {"accuracy": 0.0}


def test_mcq_task_requires_dataset_or_samples():
    with pytest.raises(ValueError, match="dataset"):
        McqTask()


# ---------------------------------------------------------------------------
# End to end: dual-channel mock, classify + LLM judge (judge requests routed by the "judge:" prefix)
# ---------------------------------------------------------------------------
def test_mcq_end_to_end_with_llm_judge():
    from eyas.backends.scheduling import run_sync
    from eyas.engine.domain import Prompt

    samples = make_mcq_samples()
    task = McqTask(samples=samples)

    classify_calls: list[dict] = []
    judge_calls: list[dict] = []

    def responder(request, sampling, index):
        if request.request_id.startswith("judge:"):
            payload = request.messages[0].content[0].text
            m = re.search(r"Choices: (\[[^\n]*\])", payload)
            assert m is not None, "judge prompt is missing the Choices list (list repr format)"
            assert ast.literal_eval(m.group(1)) == CHOICES  # identity variant: original order
            judge_calls.append({"request_id": request.request_id, "prompt": payload})
            # assert the remaining parts of the default judge prompt
            assert payload.startswith(f"Original Task: {INSTRUCTION}\n")
            assert "Based on the prediction above" in payload
            assert payload.rstrip().endswith("output 'None'.")
            sample_id, variant_id = request.request_id.removeprefix("judge:").split(":")
            assert variant_id == "identity"
            return _resp("boeing" if sample_id == "s1" else "airbus")
        text_part = request.messages[0].content[1].text
        classify_calls.append({"request_id": request.request_id, "text": text_part})
        # assert the classify rendering format
        assert text_part.startswith(INSTRUCTION + "\nChoices: [")
        sample_id = request.request_id.split(":")[0]
        # s1 answers perfectly; s2 emits unparseable free text to force the judge fallback
        return _resp("boeing" if sample_id == "s1" else "a large white aircraft")

    from tests.support.engine import full_split_plan, make_evaluator

    prompt = Prompt(INSTRUCTION)
    evaluator = make_evaluator(
        task,
        ScriptedMockBackend(responder=responder),
        protocol=IdentityProtocol(),
        produces=frozenset({"accuracy", "parse_fail"}),
        judge=McqLLMJudge(),
    )
    evals = run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))

    # identity single variant: 2 classify + 2 judge requests (the judge channel
    # is billed into the run's cost, so totals include both)
    assert len(classify_calls) == 2
    assert len(judge_calls) == 2
    assert evals[0].cost.requests == 4
    assert evals[0].cost.judge_requests == 2
    # _resp carries no usage -> every billed request reports 0 tokens
    assert evals[0].cost.input_tokens == 0
    assert evals[0].cost.judge_input_tokens == 0

    # hand check: s1 judge -> boeing (correct), s2 judge -> airbus (correct) -> mean 1.0
    assert evals[0].scores["accuracy"] == pytest.approx(1.0)


def test_build_judge_prompt_uses_list_repr_choices():
    prompt = build_judge_prompt("task text", "prediction text", ["a", "b"])
    assert "Original Task: task text\n" in prompt
    assert 'Model\'s Prediction: "prediction text"' in prompt
    assert "Choices: ['a', 'b']" in prompt  # list repr, not ", ".join format


# ---------------------------------------------------------------------------
# match_judge_result_cot (CoT-baseline judging normalization) — verdict shapes
# as McqLLMJudge(variant="cot") feeds them: raw judge text + shifted choices.
# ---------------------------------------------------------------------------
def test_match_judge_result_cot_exact_hit_after_strip():
    assert match_judge_result_cot("boeing", CHOICES) == "boeing"
    assert (
        match_judge_result_cot("  boeing\n", CHOICES) == "boeing"
    )  # strip precedes the exact check


def test_match_judge_result_cot_none_verdict_maps_to_none():
    # the judge prompt instructs 'None' for invalid predictions: it must not
    # surface as the string "None" (that would score against a real label)
    assert match_judge_result_cot("None", CHOICES) is None
    assert match_judge_result_cot("  None ", CHOICES) is None


def test_match_judge_result_cot_first_substring_hit_in_choice_order():
    # the ported rule is choice-list order, NOT match_prediction's
    # longest-first: the same text resolves differently per list order
    assert (
        match_judge_result_cot("The answer is airbus a320", ["airbus", "airbus a320"]) == "airbus"
    )
    assert (
        match_judge_result_cot("The answer is airbus a320", ["airbus a320", "airbus"])
        == "airbus a320"
    )


def test_match_judge_result_cot_malformed_output_falls_to_none():
    # unparseable judge output -> None -> verified != label -> 0 (documented default)
    assert match_judge_result_cot("a large white aircraft", CHOICES) is None
    # the ported CoT rule has no case folding (match_prediction does)
    assert match_judge_result_cot("The answer is Boeing", CHOICES) is None


def test_mcq_llm_judge_fails_fast_without_context():
    import asyncio

    judge = McqLLMJudge()
    sample = make_mcq_samples()[0]
    ctx = get_judge_context_missing()
    with pytest.raises(RuntimeError, match="variant"):
        asyncio.run(judge.ajudge(_resp("x"), _pred("x"), sample, ctx))


class _Resp:
    def __init__(self, text: str):
        self.text = text
        self.request_id = "req"


class _Pred:
    def __init__(self, raw: str):
        self.value = None
        self.raw = raw


def get_judge_context_missing():
    from eyas.backends.mock import ScriptedMockBackend as B
    from eyas.judges.base import JudgeContext

    return JudgeContext(backend=B(responses=["ok"]))


# ---------------------------------------------------------------------------
# ajudge decisions (context fail-fast / Original-Task source / variant prompt
# convention / normalization rule). Inputs are what the real caller passes:
# Evaluator._score_one fills backend + variant + instruction + user_text per
# verdict, then calls ``ajudge``; assertions are on the verdict and on the
# prompt the judging backend actually received.
# ---------------------------------------------------------------------------
def _judge_ctx(**overrides):
    """JudgeContext as the Evaluator fills it, with per-test overrides (a None
    value stands in for a slot the caller forgot to fill)."""
    from eyas.judges.base import JudgeContext
    from eyas.tasks.base import Variant

    slots: dict = {
        "backend": ScriptedMockBackend(responses=["boeing"]),
        "variant": Variant(id="identity"),
        "instruction": INSTRUCTION,
        "user_text": None,
    }
    slots.update(overrides)
    return JudgeContext(**slots)


def test_mcq_llm_judge_fails_fast_without_backend():
    import asyncio

    with pytest.raises(RuntimeError, match="backend"):
        asyncio.run(
            McqLLMJudge().ajudge(
                _resp("x"), _pred("x"), make_mcq_samples()[0], _judge_ctx(backend=None)
            )
        )


def test_mcq_llm_judge_fails_fast_without_instruction():
    import asyncio

    backend = ScriptedMockBackend(responses=["boeing"])
    with pytest.raises(RuntimeError, match="instruction"):
        asyncio.run(
            McqLLMJudge().ajudge(
                _resp("x"), _pred("x"), make_mcq_samples()[0], _judge_ctx(instruction=None)
            )
        )
    assert backend.call_count == 0  # fails fast before spending a judging call


def test_mcq_llm_judge_fails_fast_when_classify_original_task_lacks_user_text():
    import asyncio

    # classify_original_task=True asks for the rendered classify text; a
    # hand-built JudgeContext (no auto-fill) must not silently judge on nothing
    with pytest.raises(RuntimeError, match="user_text"):
        asyncio.run(
            McqLLMJudge(classify_original_task=True).ajudge(
                _resp("x"), _pred("x"), make_mcq_samples()[0], _judge_ctx()
            )
        )


def _nested_choice_sample() -> Sample:
    """Choices that nest ("airbus" ⊂ "airbus a320"), so the two normalization
    rules observable in the verdict: choice-list order vs longest-first."""
    return Sample(
        id="s1",
        inputs={"image": "/data/images/1.jpg", "choices": ["airbus", "airbus a320"]},
        references={"label": "airbus a320"},
    )


def _judge_reply(judge: McqLLMJudge, seen: list[str]) -> object:
    """Run ``judge`` over the nested-choice sample with a free-text reply that
    mentions both choices, recording the judging prompt."""
    import asyncio

    def responder(request, sampling, index):
        seen.append(request.messages[0].content[0].text)
        return _resp("The answer is airbus a320")

    ctx = _judge_ctx(backend=ScriptedMockBackend(responder=responder))
    raw = "The answer is airbus a320"
    return asyncio.run(judge.ajudge(_resp("x"), _pred(raw), _nested_choice_sample(), ctx))


def test_mcq_cot_variant_uses_blank_line_prompt_and_choice_order_matching():
    seen: list[str] = []
    verdict = _judge_reply(McqLLMJudge(variant="cot"), seen)
    prompt = seen[0]
    # the cot/OPRO prompt convention keeps a blank line after Prediction and Choices
    assert 'Model\'s Prediction: "The answer is airbus a320"\n\n' in prompt
    assert "Choices: ['airbus', 'airbus a320']\n\n" in prompt
    # first substring hit in choice-list order, not the longest-first rule
    assert verdict.note == "airbus"
    assert verdict.scores == {"accuracy": 0.0}
    # the judging call was billed; the mock server omitted usage -> empty dict
    assert verdict.usage == {}


def test_mcq_opro_variant_keeps_cot_prompt_but_longest_first_matching():
    seen: list[str] = []
    verdict = _judge_reply(McqLLMJudge(variant="opro"), seen)
    prompt = seen[0]
    assert 'Model\'s Prediction: "The answer is airbus a320"\n\n' in prompt
    assert "Choices: ['airbus', 'airbus a320']\n\n" in prompt
    # same reply, opposite resolution: match_prediction picks the longer choice
    assert verdict.note == "airbus a320"
    assert verdict.scores == {"accuracy": 1.0}


def test_mcq_classify_original_task_judges_on_the_rendered_classify_text():
    """The OPRO pairing documented in eyas/tasks/mcq.py: the "Original Task"
    section is the text the classify request actually rendered, not the bare
    candidate prompt."""
    from eyas.backends.base import user_text
    from eyas.backends.scheduling import run_sync
    from eyas.engine.domain import Prompt
    from tests.support.engine import full_split_plan, make_evaluator

    classify_texts: dict[str, str] = {}
    judge_prompts: dict[str, str] = {}

    def responder(request, sampling, index):
        if request.request_id.startswith("judge:"):
            sample_id = request.request_id.removeprefix("judge:").split(":")[0]
            judge_prompts[sample_id] = request.messages[0].content[0].text
            return _resp("boeing" if sample_id == "s1" else "airbus")
        # free text: the deterministic parse fails, so judging takes over
        sample_id = request.request_id.split(":")[0]
        classify_texts[sample_id] = user_text(request)
        return _resp("a large white aircraft")

    prompt = Prompt(INSTRUCTION)
    evaluator = make_evaluator(
        McqTask(samples=make_mcq_samples()),
        ScriptedMockBackend(responder=responder),
        protocol=IdentityProtocol(),
        produces=frozenset({"accuracy", "parse_fail"}),
        judge=McqLLMJudge(variant="opro", classify_original_task=True),
    )
    evals = run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))

    assert set(judge_prompts) == {"s1", "s2"}
    for sample_id, judge_prompt in judge_prompts.items():
        rendered = classify_texts[sample_id]
        assert rendered != INSTRUCTION  # the classify text carries the choices too
        assert judge_prompt.startswith(f"Original Task: {rendered}\n")
    assert evals[0].scores["accuracy"] == pytest.approx(1.0)


def test_mcq_llm_judge_sync_entry_points_at_the_async_api():
    from eyas.judges.base import JudgeContext

    with pytest.raises(NotImplementedError, match="ajudge"):
        McqLLMJudge().judge(_resp("x"), _pred("x"), make_mcq_samples()[0], JudgeContext())


# ---------------------------------------------------------------------------
# Dataset loading + stratified sampling
# ---------------------------------------------------------------------------
@pytest.fixture
def group_dataset(tmp_path):
    data_dir = tmp_path / "GroupTest"
    data_dir.mkdir()
    (data_dir / "g1_train.json").write_text(
        '[{"filename": "a/2.jpg", "label": "x"},'
        ' {"filename": "a/1.jpg", "label": "y"},'
        ' {"filename": "a/3.jpg", "label": "x"}]',
        encoding="utf-8",
    )
    (data_dir / "g2_train.json").write_text(
        '[{"filename": "b/1.jpg", "label": "z"}]', encoding="utf-8"
    )
    return data_dir


def test_load_mcq_samples_ordering_and_choices(group_dataset, tmp_path):
    images_dir = tmp_path / "images"
    samples = load_mcq_samples("Test", "train", data_dir=group_dataset, images_dir=images_dir)
    # groups sorted by filename; JSON order preserved within a group
    assert [s.id for s in samples] == ["a/2.jpg", "a/1.jpg", "a/3.jpg", "b/1.jpg"]
    # choices = in-group labels by descending frequency (x:2, y:1 -> [x, y])
    assert samples[0].inputs["choices"] == ["x", "y"]
    assert samples[3].inputs["choices"] == ["z"]
    # image path = images_dir / filename; label kept as-is
    assert samples[0].inputs["image"] == str(images_dir / "a/2.jpg")
    assert samples[0].references["label"] == "x"
    # group name goes into inputs
    assert samples[0].inputs["group"] == "g1"
    assert samples[3].inputs["group"] == "g2"


# ---------------------------------------------------------------------------
# Train-split stratified sampling (per_group + two-level seed derivation)
# ---------------------------------------------------------------------------
@pytest.fixture
def group_dataset_big(tmp_path):
    data_dir = tmp_path / "GroupTest"
    data_dir.mkdir()

    def _write(name: str, items: list[tuple[str, str]]):
        payload = ", ".join(f'{{"filename": "{fn}", "label": "{lb}"}}' for fn, lb in items)
        (data_dir / name).write_text(f"[{payload}]", encoding="utf-8")

    g1_train = [(f"a/{i}.jpg", "x") for i in range(4)] + [(f"a/{i}.jpg", "y") for i in range(4, 8)]
    g2_train = [(f"b/{i}.jpg", lb) for i, lb in enumerate(["p", "p", "q", "q", "r", "r"])]
    _write("g1_train.json", g1_train)
    _write("g2_train.json", g2_train)
    # test split has separate content (verifies test is not sampled)
    _write("g1_test.json", [(f"a/{i}.jpg", "x") for i in range(100, 103)])
    _write("g2_test.json", [("b/200.jpg", "p")])
    return data_dir, tmp_path / "images"


def test_train_split_stratified_sampling_matches_oracle(group_dataset_big):
    data_dir, images_dir = group_dataset_big
    sampled = load_mcq_samples(
        "Test", "train", data_dir=data_dir, images_dir=images_dir, per_group=5, sample_seed=42
    )
    # oracle: sample on the group structure first, then build (same path as load_mcq_samples)
    datasets, _ = load_group_split("Test", "train", data_dir=data_dir, images_dir=images_dir)
    oracle = build_mcq_samples(
        stratified_sample_train(datasets, per_group=5, seed=derive_sample_seed(42)), images_dir
    )
    assert [s.id for s in sampled] == [s.id for s in oracle]
    assert len(sampled) == 10  # 5 per group
    # stratified: each of g2's three labels gets at least 1 (quota path)
    g2_labels = {s.references["label"] for s in sampled if s.inputs["group"] == "g2"}
    assert g2_labels == {"p", "q", "r"}


def test_train_sampling_deterministic_and_test_full(group_dataset_big):
    data_dir, images_dir = group_dataset_big
    task = McqTask(
        "Test", data_dir=data_dir, images_dir=images_dir, train_per_group=5, train_sample_seed=42
    )
    first = [s.id for s in task.load_split("train")]
    second = [s.id for s in task.load_split("train")]
    assert first == second  # fixed seed -> stable across instances and calls (comparable fitness)
    # test split stays full, unaffected by train_per_group
    test_ids = [s.id for s in task.load_split("test")]
    assert test_ids == ["a/100.jpg", "a/101.jpg", "a/102.jpg", "b/200.jpg"]
    # per_group=0 -> train is also full
    task_full = McqTask("Test", data_dir=data_dir, images_dir=images_dir, train_per_group=0)
    assert len(task_full.load_split("train")) == 14


def test_build_mcq_samples_is_pure_function(group_dataset, tmp_path):
    from eyas.tasks.datasets import load_group_split

    datasets, images_dir = load_group_split(
        "Test", "train", data_dir=group_dataset, images_dir=tmp_path / "images"
    )
    samples = build_mcq_samples(datasets, images_dir)
    assert len(samples) == 4


def test_load_group_split_rejects_bad_split(group_dataset):
    with pytest.raises(ValueError, match="split"):
        load_mcq_samples("Test", "dev", data_dir=group_dataset)


def test_resolve_dataset_paths_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_dataset_paths("Nope")


def test_stratified_sample_train_deterministic_and_stratified():
    datasets = {
        "g": [
            {"filename": f"{label}-{i}.jpg", "label": label}
            for label, n in (("a", 3), ("b", 2))
            for i in range(n)
        ]
    }
    seed = derive_sample_seed(42)
    r1 = stratified_sample_train(datasets, per_group=3, seed=seed)
    r2 = stratified_sample_train(datasets, per_group=3, seed=seed)
    assert r1 == r2  # identical under the same seed (parity depends on it)
    labels = sorted(s["label"] for s in r1["g"])
    assert labels == ["a", "a", "b"]  # quota stratification: at least 1 per label
    assert len(r1["g"]) == 3

    # per_group >= group size returns the group as-is
    whole = stratified_sample_train(datasets, per_group=10, seed=seed)
    assert len(whole["g"]) == 5


# ---------------------------------------------------------------------------
# resolve_dataset_paths — every way the two directories can be supplied
# (explicit pair / registry under the env root / the fail-fast combinations).
# The env root is read at call time, so the tests set the environment variable.
# ---------------------------------------------------------------------------
def test_resolve_dataset_paths_explicit_pair_wins_without_registry_lookup():
    # a name the registry has never heard of: both directories explicit means the
    # registry is not consulted at all, so this must not raise
    data_dir, images_dir = resolve_dataset_paths(
        "Nope", data_dir="relative/data", images_dir="/absolute/images"
    )
    assert (data_dir, images_dir) == (Path("relative/data"), Path("/absolute/images"))


def test_resolve_dataset_paths_one_explicit_dir_still_uses_the_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("EYAS_DATASET_ROOT", str(tmp_path))
    # only data_dir explicit -> images_dir comes from the registered config
    data_dir, images_dir = resolve_dataset_paths("Aircraft", data_dir=str(tmp_path / "custom"))
    assert data_dir == tmp_path / "custom"
    assert images_dir == tmp_path / "Aircraft" / "images"
    # only images_dir explicit -> data_dir comes from the registered config
    data_dir, images_dir = resolve_dataset_paths("Aircraft", images_dir=str(tmp_path / "imgs"))
    assert data_dir == tmp_path / "Aircraft"
    assert images_dir == tmp_path / "imgs"


def test_resolve_dataset_paths_registry_root_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("EYAS_DATASET_ROOT", str(tmp_path))
    # the registered relative layout is resolved under the root verbatim (CUB's
    # images dir is doubly nested — a careless single join would flatten it)
    data_dir, images_dir = resolve_dataset_paths("CUB")
    assert data_dir == tmp_path / "CUB"
    assert images_dir == tmp_path / "CUB-200-2011" / "CUB-200-2011" / "images"


def test_resolve_dataset_paths_without_root_names_the_env_vars(monkeypatch):
    monkeypatch.delenv("EYAS_DATASET_ROOT", raising=False)
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    with pytest.raises(ValueError, match="EYAS_DATASET_ROOT/DATASET_ROOT env var not set"):
        resolve_dataset_paths("Aircraft")
    # one explicit directory is not enough: the still-missing half reports the
    # env var AND the argument the caller could have passed instead
    with pytest.raises(ValueError, match="env var not set and images_dir not passed explicitly"):
        resolve_dataset_paths("Aircraft", data_dir=str(Path("/tmp/data")))


def test_resolve_dataset_paths_unknown_dataset_lists_the_options():
    with pytest.raises(ValueError, match=r"Unknown dataset 'Nope', options: \['Aircraft'"):
        resolve_dataset_paths("Nope")


def test_load_group_split_no_split_files_names_the_directory(tmp_path):
    empty = tmp_path / "Empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=rf"no \*_test\.json found under {empty}"):
        load_group_split("Test", "test", data_dir=empty, images_dir=tmp_path / "images")


# ---------------------------------------------------------------------------
# Malformed dataset records: a group JSON element missing a
# required field must fail with the group + record position, not leak a bare
# KeyError from deep inside the loader (same contract load_vqa_samples pins).
# ---------------------------------------------------------------------------
def test_load_mcq_samples_malformed_record_names_group_and_index(tmp_path):
    data_dir = tmp_path / "Malformed"
    data_dir.mkdir()
    images_dir = tmp_path / "images"
    (data_dir / "g1_train.json").write_text(
        '[{"filename": "a/1.jpg", "label": "x"}, {"filename": "a/2.jpg"}]', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"group 'g1'\[1\] missing required field 'label'"):
        load_mcq_samples("Test", "train", data_dir=data_dir, images_dir=images_dir)
    # filename is the other required field (it becomes Sample.id, so a missing
    # one cannot be defaulted without breaking cross-run stability)
    (data_dir / "g1_train.json").write_text('[{"label": "x"}]', encoding="utf-8")
    with pytest.raises(ValueError, match=r"group 'g1'\[0\] missing required field 'filename'"):
        load_mcq_samples("Test", "train", data_dir=data_dir, images_dir=images_dir)


# ---------------------------------------------------------------------------
# Duplicate sample ids: id = filename is the cross-run cache key, so the same
# filename in two groups (or twice in one group) must fail fast naming the id
# and both groups — never silently collapse into one cached evaluation.
# ---------------------------------------------------------------------------
def test_load_mcq_samples_duplicate_id_across_groups_fails_fast(tmp_path: Path) -> None:
    data_dir = tmp_path / "DupTest"
    data_dir.mkdir()
    (data_dir / "g1_train.json").write_text(
        '[{"filename": "001.jpg", "label": "x"}]', encoding="utf-8"
    )
    (data_dir / "g2_train.json").write_text(
        '[{"filename": "001.jpg", "label": "y"}]', encoding="utf-8"
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "001.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="001") as exc:
        load_mcq_samples("Test", "train", data_dir=data_dir, images_dir=images_dir)
    msg = str(exc.value)
    assert "g1" in msg and "g2" in msg  # both colliding groups are named


def test_load_mcq_samples_duplicate_id_within_one_group_fails_fast(tmp_path: Path) -> None:
    data_dir = tmp_path / "DupTest"
    data_dir.mkdir()
    (data_dir / "g1_train.json").write_text(
        '[{"filename": "001.jpg", "label": "x"}, {"filename": "001.jpg", "label": "y"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="001") as exc:
        load_mcq_samples("Test", "train", data_dir=data_dir, images_dir=tmp_path / "images")
    assert "g1" in str(exc.value)


# ---------------------------------------------------------------------------
# stratified_sample_train direct-boundary contracts (the per_group <= 0
# passthrough, unequal class sizes, and the labels-outnumber-the-quota branch).
# ---------------------------------------------------------------------------
def _group(n_by_label: dict[str, int]) -> list[dict]:
    return [
        {"filename": f"{label}-{i}.jpg", "label": label}
        for label, n in n_by_label.items()
        for i in range(n)
    ]


def test_stratified_sample_train_non_positive_per_group_is_identity():
    datasets = {"g1": _group({"x": 2}), "g2": _group({"y": 1})}
    for per_group in (0, -1):
        out = stratified_sample_train(datasets, per_group=per_group, seed=42)
        assert out is datasets  # sampling disabled -> the input structure is returned untouched


def test_stratified_sample_train_unequal_class_sizes_stays_at_quota():
    # "a" could supply every slot; stratification must still cap it and give the
    # scarce classes their guaranteed one, at exactly per_group total
    datasets = {"g": _group({"a": 6, "b": 2, "c": 1})}
    seed = derive_sample_seed(42)
    out = stratified_sample_train(datasets, per_group=6, seed=seed)
    picked = out["g"]
    assert len(picked) == 6
    assert {s["label"] for s in picked} == {"a", "b", "c"}
    counts = {label: sum(1 for s in picked if s["label"] == label) for label in ("a", "b", "c")}
    assert counts == {"a": 4, "b": 1, "c": 1}  # proportional share of the 3 leftover slots
    # every picked record is one of the inputs, no duplicates
    assert len({s["filename"] for s in picked}) == 6
    assert all(any(s is p for p in datasets["g"]) for s in picked)
    # same seed -> identical subset AND order; a different seed is allowed to differ
    assert out == stratified_sample_train(datasets, per_group=6, seed=seed)
    other = stratified_sample_train(datasets, per_group=6, seed=seed + 1)
    assert [s["filename"] for s in other["g"]] != [s["filename"] for s in picked]


def test_stratified_sample_train_labels_exactly_equal_quota_take_one_each():
    # num_labels == per_group: every label gets its guaranteed single slot and
    # there is no remainder to distribute (quota == 1 fast path in _label_quotas)
    datasets = {"g": _group({"a": 3, "b": 3})}
    out = stratified_sample_train(datasets, per_group=2, seed=derive_sample_seed(42))
    assert len(out["g"]) == 2
    assert {s["label"] for s in out["g"]} == {"a", "b"}


def test_stratified_sample_train_labels_exceeding_quota_pick_distinct_labels():
    # 5 labels, 3 slots: one record per chosen label (never 3 records of one label)
    datasets = {"g": _group({label: 2 for label in "abcde"})}
    seed = derive_sample_seed(42)
    out = stratified_sample_train(datasets, per_group=3, seed=seed)
    picked = out["g"]
    assert len(picked) == 3
    assert len({s["label"] for s in picked}) == 3
    assert out == stratified_sample_train(datasets, per_group=3, seed=seed)
    # pinned draw for this seed: pins the label subset AND the within-label pick
    # and the emission order (parity depends on all three)
    assert [s["filename"] for s in picked] == ["d-0.jpg", "b-1.jpg", "e-1.jpg"]


def test_stratified_sample_train_unequal_groups_share_the_same_seed_stream():
    # a group that passes through unsampled must not shift the seed stream of
    # the next group (each group draws from the one rng in registry order)
    datasets = {
        "small": _group({"x": 2, "y": 2}),
        "big": _group({"p": 4, "q": 4}),
    }
    seed = derive_sample_seed(42)
    out = stratified_sample_train(datasets, per_group=5, seed=seed)
    assert out["small"] == datasets["small"]  # len <= per_group -> copied as-is
    assert out["small"] is not datasets["small"]
    assert len(out["big"]) == 5
    # oracle: the same draw computed for the big group alone consumes the identical
    # rng steps, since the passthrough group draws nothing
    solo = stratified_sample_train({"big": datasets["big"]}, per_group=5, seed=seed)
    assert [s["filename"] for s in out["big"]] == [s["filename"] for s in solo["big"]]


# ---------------------------------------------------------------------------
# McqTask.build_request assembly modes not reached by the identity path:
# SPO full template (in-place image / image prepended / missing placeholder),
# the target-image suffix, the assistant continuation turn, and the malformed
# sample boundary.
# ---------------------------------------------------------------------------
def _variant_for(sample: Sample):
    return IdentityProtocol().variants(sample)[0]


def test_build_request_raw_template_replaces_image_in_place():
    sample = make_mcq_samples()[0]
    task = McqTask(samples=make_mcq_samples(), raw_template=True)
    req = task.build_request("look {image} then {choices} done", sample, _variant_for(sample))
    parts = req.messages[0].content
    assert [type(p) for p in parts] == [TextPart, ImagePart, TextPart]
    assert parts[0].text == "look "
    assert parts[1].image == sample.inputs["image"]
    assert parts[2].text == f" then {CHOICES_STR} done"


def test_build_request_raw_template_repeats_every_image_marker():
    # two markers -> two image parts, empty text segments dropped
    sample = make_mcq_samples()[0]
    task = McqTask(samples=make_mcq_samples(), raw_template=True)
    req = task.build_request("{image}a{choices}b{image}", sample, _variant_for(sample))
    parts = req.messages[0].content
    assert [type(p) for p in parts] == [ImagePart, TextPart, ImagePart]
    assert [parts[0].image, parts[2].image] == [sample.inputs["image"]] * 2
    assert parts[1].text == f"a{CHOICES_STR}b"


def test_build_request_raw_template_prepends_image_when_marker_missing():
    sample = make_mcq_samples()[0]
    task = McqTask(samples=make_mcq_samples(), raw_template=True)
    req = task.build_request("Classify these: {choices}", sample, _variant_for(sample))
    parts = req.messages[0].content
    assert [type(p) for p in parts] == [ImagePart, TextPart]
    assert parts[0].image == sample.inputs["image"]
    assert parts[1].text == f"Classify these: {CHOICES_STR}"


def test_build_request_raw_template_requires_choices_placeholder():
    sample = make_mcq_samples()[0]
    task = McqTask(samples=make_mcq_samples(), raw_template=True)
    with pytest.raises(ValueError, match=r"raw_template mode requires the prompt to contain a "):
        task.build_request(INSTRUCTION, sample, _variant_for(sample))


def test_build_request_target_image_suffix_rstrips_instruction():
    sample = make_mcq_samples()[0]
    task = McqTask(samples=make_mcq_samples(), target_image_suffix=True)
    req = task.build_request(f"  {INSTRUCTION}  ", sample, _variant_for(sample))
    parts = req.messages[0].content
    # only the trailing side is rstripped (verbatim-port semantics); the newline
    # before Choices closes the instruction line
    assert parts[1].text == f"  {INSTRUCTION}\nChoices: [{CHOICES_STR}]\nTarget Image:"


def test_build_request_appends_assistant_prefix_turn():
    sample = make_mcq_samples()[0]
    task = McqTask(samples=make_mcq_samples(), assistant_prefix="Let me think step by step.")
    req = task.build_request(INSTRUCTION, sample, _variant_for(sample))
    assert [m.role for m in req.messages] == ["user", "assistant"]
    assert req.messages[1].content[0].text == "Let me think step by step."


def test_mcq_task_ctor_validates_position_combinations_and_exposes_them():
    # unknown position and the raw_template/OPRO clash must fail at the ctor,
    # so build_request never has to defend against them; the properties feed
    # the optimizer's consistency check
    samples = make_mcq_samples()
    with pytest.raises(
        ValueError, match=r"instruction_pos must be one of .* or None, got 'middle'"
    ):
        McqTask(samples=samples, instruction_pos="middle")
    with pytest.raises(ValueError, match="raw_template and instruction_pos are mutually exclusive"):
        McqTask(samples=samples, raw_template=True, instruction_pos="Q_begin")
    task = McqTask(samples=samples, instruction_pos="A_begin", include_qa=True)
    assert task.instruction_pos == "A_begin"
    assert task.include_qa is True


def test_build_request_malformed_sample_names_the_sample_and_field():
    # A sample without its choices/image must raise with the
    # sample id and the missing key (a bare KeyError would leave the optimizer
    # with no way to find the bad record in a 10k-sample split)
    task = McqTask(samples=[])
    no_choices = Sample(
        id="bad1", inputs={"image": "/i/1.jpg", "group": "g"}, references={"label": "x"}
    )
    with pytest.raises(ValueError, match=r"sample 'bad1' is missing required input 'choices'"):
        task.build_request(INSTRUCTION, no_choices, _variant_for(no_choices))
    no_image = Sample(
        id="bad2", inputs={"choices": CHOICES, "group": "g"}, references={"label": "x"}
    )
    with pytest.raises(ValueError, match=r"sample 'bad2' is missing required input 'image'"):
        task.build_request(INSTRUCTION, no_image, _variant_for(no_image))


def test_registry_has_mcq_components():
    import eyas.builtin  # noqa: F401

    assert get("task", "mcq") is McqTask
    assert get("protocol", "identity") is IdentityProtocol
    assert get("judge", "mcq_llm") is McqLLMJudge


def _resp(text: str):
    from eyas.backends.base import ChatResponse

    return ChatResponse(text=text)


def _pred(raw: str):
    from eyas.tasks.base import Prediction

    return Prediction(value=None, raw=raw)


def test_mcq_data_dir_alone_rejected_at_construction(tmp_path):
    # data_dir without dataset/samples/loader can never load a split: the
    # config error must surface at construction as a ValueError, not as an
    # assert deep inside load_split (asserts are stripped under -O)
    with pytest.raises(ValueError, match="dataset name"):
        McqTask(data_dir=str(tmp_path))


def test_parse_response_and_score_name_missing_fields():
    from eyas.tasks.base import Prediction

    task = McqTask(samples=[])
    no_choices = Sample(
        id="bad3", inputs={"image": "/i/3.jpg", "group": "g"}, references={"label": "x"}
    )
    with pytest.raises(ValueError, match=r"sample 'bad3' is missing required input 'choices'"):
        task.parse_response(_resp("boeing"), no_choices)
    no_label = Sample(
        id="bad4", inputs={"image": "/i/4.jpg", "choices": CHOICES, "group": "g"}, references={}
    )
    with pytest.raises(ValueError, match=r"sample 'bad4' is missing required reference 'label'"):
        task.score(Prediction(value="boeing", raw="boeing"), no_label)


def test_dataset_root_env_set_after_import_takes_effect(tmp_path, monkeypatch):
    # the env root is read at CALL time: an os.environ set after the module
    # import (plugin / notebook / test setup) must still take effect
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    monkeypatch.setenv("EYAS_DATASET_ROOT", str(tmp_path))
    data_dir, images_dir = resolve_dataset_paths("Aircraft")
    assert data_dir == tmp_path / "Aircraft"
    assert images_dir == tmp_path / "Aircraft" / "images"


def test_group_name_containing_split_suffix_survives(tmp_path):
    # removesuffix, not replace: a group name that itself contains "_train"
    # (e.g. foo_train_bar) must survive stem parsing intact
    data_dir = tmp_path / "G"
    data_dir.mkdir()
    (data_dir / "foo_train_bar_train.json").write_text(
        '[{"filename": "a.jpg", "label": "x"}]', encoding="utf-8"
    )
    groups, _ = load_group_split("Test", "train", data_dir=data_dir, images_dir=tmp_path / "images")
    assert list(groups) == ["foo_train_bar"]


def test_load_group_split_rejects_non_dict_item_with_index(tmp_path):
    data_dir = tmp_path / "G"
    data_dir.mkdir()
    (data_dir / "g1_train.json").write_text(
        '[{"filename": "a.jpg", "label": "x"}, "oops"]', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"g1.*\[1\].*must be a JSON object, got str"):
        load_group_split("Test", "train", data_dir=data_dir, images_dir=tmp_path / "images")
