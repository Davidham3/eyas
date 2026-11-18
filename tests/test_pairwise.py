"""PairwiseJudge extension point: default implementation + registration + end-to-end judge swap flips verdicts."""

from __future__ import annotations

import pytest
from test_contract import CompareOnlyConfig, CompareOnlyStrategy, ToyCompare

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    TextPart,
    system_message,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.engine.budget import LoadPlan, WorkItem
from eyas.engine.domain import Cost, Evaluation, Prompt
from eyas.engine.evaluation import CompareEvalModel, Evaluator
from eyas.engine.judges import AnswerPairwiseJudge, DefaultPairwiseJudge, PairwiseJudge
from eyas.engine.registry import get as engine_get
from eyas.engine.registry import registered as engine_registered
from eyas.engine.strategy import build_run_context
from eyas.tasks.protocol import IdentityProtocol


def test_default_parse_ab_tie() -> None:
    j = DefaultPairwiseJudge()
    assert j.parse("A") == "A"
    assert j.parse("B better") == "B"
    # lowercase tokens are NOT verdicts: uppercasing first would turn the
    # article "a" in prose ("it is a tie") into a phantom A verdict
    assert j.parse(" b better") == "tie"
    assert j.parse("too close to call") == "tie"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: B", "B"),
        ("A", "A"),
        ("B is better", "B"),
        ("**B**", "B"),
        ("Analysis: both are similar. A edges out.", "A"),
        ("Both answers are equally good", "tie"),  # no standalone token -> no-verdict sentinel
    ],
)
def test_parse_first_standalone_token(text: str, expected: str) -> None:
    """The verdict is the FIRST standalone uppercase A/B token anywhere in the
    reply, not the first letter: 'Answer: B' must not be misread as A (the old
    startswith rule biased every prefixed reply toward position A)."""
    assert DefaultPairwiseJudge().parse(text) == expected


def test_judge_requests_carry_a_user_message():
    """Structural gate mirroring the gen-request guard (test_contract): a
    system-only ChatRequest raises in the real Qwen3.5 HF chat template
    ('No user query found in messages') — BOTH pairwise
    judges ride real engines too, so each must carry a non-empty user message."""
    for judge in (DefaultPairwiseJudge(), AnswerPairwiseJudge()):
        request = judge.build_request("what animal is this?", "careful", "brief")
        assert any(
            m.role == "user" and any(isinstance(p, TextPart) and p.text.strip() for p in m.content)
            for m in request.messages
        ), f"{judge.name}: system-only judge request"


def test_registered_in_engine_registry():
    assert "default" in engine_registered("pairwise_judge")
    assert engine_get("pairwise_judge", "default") is DefaultPairwiseJudge


def test_compare_model_uses_injected_judge():
    class AlwaysB(DefaultPairwiseJudge):
        name = "always_b"

        def parse(self, text: str) -> str:
            return "B"

    model = CompareEvalModel(pairwise_judge=AlwaysB())
    assert isinstance(model.judge, AlwaysB)


async def test_swapped_judge_flips_winner():
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        # the (A)/(B) payload rides the user message (system-only requests crash the
        # real chat templates); read the full judge text, not just the directive
        s = user_text(request)
        a_side = s.split("(B)")[0]
        return ChatResponse(text="A" if "step by step" in a_side.lower() else "B")

    class InvertJudge(DefaultPairwiseJudge):
        """Invert A/B by content: default-A verdicts flip to B (under round-robin the positions are
        constant-symmetric, so only content-dependence breaks the symmetry)."""

        name = "invert"

        def parse(self, text: str) -> str:
            base = super().parse(text)
            return {"A": "B", "B": "A"}.get(base, "tie")

    task = ToyCompare()
    good, bad = Prompt("Answer step by step."), Prompt("Answer instantly.")

    async def run(model: CompareEvalModel) -> dict[str, float]:
        backend = ScriptedMockBackend(responder=responder)
        ctx = build_run_context(
            CompareOnlyStrategy,
            CompareOnlyConfig(),
            task,
            IdentityProtocol(),
            backend,
            model,
        )
        evaluator = Evaluator(
            task=task,
            protocol=IdentityProtocol(),
            backend=backend,
            planner=ctx.evaluator.planner,
            model=model,
        )
        plan = LoadPlan(items=(WorkItem(good, ("q1", "q2")), WorkItem(bad, ("q1", "q2"))))
        evals = await evaluator.aevaluate([good, bad], plan, "train")
        return {e.prompt_id: e.scores["compare_score"] for e in evals}

    # default judge: step-by-step wins
    by_default = await run(CompareEvalModel(pairwise_judge=DefaultPairwiseJudge()))
    assert by_default[good.id] == 1.0 and by_default[bad.id] == 0.0

    # swap in the inverting judge: the winner flips, proving the judging strategy is pluggable
    by_invert = await run(CompareEvalModel(pairwise_judge=InvertJudge()))
    assert by_invert[bad.id] == 1.0 and by_invert[good.id] == 0.0


async def test_compare_accounts_response_tokens():
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        return ChatResponse(text="A", usage={"prompt_tokens": 7, "completion_tokens": 3})

    task = ToyCompare()
    good, bad = Prompt("Answer step by step."), Prompt("Answer instantly.")
    backend = ScriptedMockBackend(responder=responder)
    model = CompareEvalModel(pairwise_judge=DefaultPairwiseJudge())
    ctx = build_run_context(
        CompareOnlyStrategy, CompareOnlyConfig(), task, IdentityProtocol(), backend, model
    )
    evaluator = Evaluator(
        task=task,
        protocol=IdentityProtocol(),
        backend=backend,
        planner=ctx.evaluator.planner,
        model=model,
    )
    plan = LoadPlan(items=(WorkItem(good, ("q1", "q2")), WorkItem(bad, ("q1", "q2"))))
    evals = await evaluator.aevaluate([good, bad], plan, "train")
    n_requests = sum(e.cost.requests for e in evals)
    assert n_requests > 0
    # compare convention: the batch cost is carried ONCE (first Evaluation only),
    # so summing across evaluations counts every game exactly once
    assert sum(e.cost.input_tokens for e in evals) == 7 * n_requests
    assert sum(e.cost.output_tokens for e in evals) == 3 * n_requests


# --- compare cost single-count + real per-sample view (3 prompts x 2 samples) ---

# Scripted verdict pattern, keyed by (A-side prompt, B-side prompt, question).
# Position-consistent (swapping A/B flips the verdict) and NON-degenerate: on q1
# the prompts rank alpha > beta > gamma, on q2 gamma beats beta and ties alpha.
_PATTERN_VERDICTS: dict[tuple[str, str, str], str] = {
    ("alpha", "beta", "q1"): "A",
    ("alpha", "beta", "q2"): "B",
    ("alpha", "gamma", "q1"): "A",
    ("alpha", "gamma", "q2"): "tie",
    ("beta", "alpha", "q1"): "B",
    ("beta", "alpha", "q2"): "A",
    ("beta", "gamma", "q1"): "A",
    ("beta", "gamma", "q2"): "B",
    ("gamma", "alpha", "q1"): "B",
    ("gamma", "alpha", "q2"): "tie",
    ("gamma", "beta", "q1"): "B",
    ("gamma", "beta", "q2"): "A",
}


def _pattern_responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
    lines = user_text(request).splitlines()
    key = (
        lines[1].removeprefix("(A) "),
        lines[2].removeprefix("(B) "),
        lines[0].removeprefix("Question: "),
    )
    # unknown game -> KeyError: the pattern must cover every scheduled pair
    return ChatResponse(
        text=_PATTERN_VERDICTS[key], usage={"prompt_tokens": 7, "completion_tokens": 3}
    )


async def _run_pattern_compare() -> list[Evaluation]:
    task = ToyCompare()
    prompts = [Prompt("alpha"), Prompt("beta"), Prompt("gamma")]
    backend = ScriptedMockBackend(responder=_pattern_responder)
    model = CompareEvalModel(pairwise_judge=DefaultPairwiseJudge())
    ctx = build_run_context(
        CompareOnlyStrategy, CompareOnlyConfig(), task, IdentityProtocol(), backend, model
    )
    evaluator = Evaluator(
        task=task,
        protocol=IdentityProtocol(),
        backend=backend,
        planner=ctx.evaluator.planner,
        model=model,
    )
    plan = LoadPlan(items=tuple(WorkItem(p, ("q1", "q2")) for p in prompts))
    return await evaluator.aevaluate(prompts, plan, "train")


async def test_compare_cost_counted_once() -> None:
    """The whole-batch bill rides ONLY the first Evaluation: summing across
    evaluations (what _round_cost does) must count each game exactly once —
    n*(n-1)*len(samples) requests, tokens counted once."""
    evals = await _run_pattern_compare()
    assert sum(e.cost.requests for e in evals) == 3 * 2 * 2  # n(n-1)*samples
    assert sum(e.cost.input_tokens for e in evals) == 7 * 12  # single-batch total
    assert sum(e.cost.output_tokens for e in evals) == 3 * 12
    assert evals[0].cost.requests == 12
    assert all(e.cost == Cost() for e in evals[1:])


async def test_compare_per_sample_is_real_and_bounded() -> None:
    """per_sample is the prompt's REAL mean game outcome on that sample
    (win 1.0 / tie 0.5 / loss 0.0 over its 2*(n-1)=4 games), never a fabricated
    uniform value. Hand-computed from _PATTERN_VERDICTS (verdict names the
    A/B SIDE, so a reversed fixture swaps who wins):
      alpha: q1 four wins -> 1.0;  q2 two losses + two ties -> 0.25
      beta:  q1 two wins + two losses -> 0.5; q2 two wins + two losses -> 0.5
      gamma: q1 four losses -> 0.0; q2 two wins + two ties -> 0.75
    """
    evals = await _run_pattern_compare()
    assert all(0.0 <= o.scores["compare_score"] <= 1.0 for e in evals for o in e.per_sample)
    expected = [(1.0, 0.25), (0.5, 0.5), (0.0, 0.75)]
    for e, (q1, q2) in zip(evals, expected, strict=True):
        assert [o.sample_id for o in e.per_sample] == ["q1", "q2"]
        assert e.per_sample[0].scores["compare_score"] == q1
        assert e.per_sample[1].scores["compare_score"] == q2
    # the aggregate stays the cross-sample mean of the real per-sample outcomes
    assert [e.scores["compare_score"] for e in evals] == [0.625, 0.5, 0.375]


def test_pairwise_judge_is_abstract():
    assert issubclass(DefaultPairwiseJudge, PairwiseJudge)
    with pytest.raises(TypeError):
        PairwiseJudge()  # type: ignore[abstract]


def test_old_style_subclass_without_system_prompt_cannot_instantiate():
    """The abstract seam after build_request/parse went concrete: the ONE
    remaining abstract member is ``system_prompt``, so a pre-refactor subclass
    (its own build_request + parse, no system_prompt) must fail loudly at
    instantiation, not later inside the default build_request."""

    class PreRefactorJudge(PairwiseJudge):
        name = "pre_refactor"

        def build_request(self, question: str, a_text: str, b_text: str) -> ChatRequest:
            return ChatRequest(messages=(user_message(f"{question} {a_text} {b_text}"),))

        def parse(self, text: str) -> str:
            return "tie"

    with pytest.raises(TypeError):
        PreRefactorJudge()  # type: ignore[abstract]


def test_subclass_overriding_build_request_fully_is_the_escape_hatch():
    """The documented escape hatch still opens: providing system_prompt and
    overriding build_request end-to-end (custom request shape, own parse)
    instantiates and works — the shared request template is the default, not a
    trap."""

    class CustomJudge(PairwiseJudge):
        name = "custom"
        system_prompt = "Custom directive."

        def build_request(self, question: str, a_text: str, b_text: str) -> ChatRequest:
            return ChatRequest(
                messages=(system_message(self.system_prompt), user_message(f"{b_text}|{a_text}"))
            )

        def parse(self, text: str) -> str:
            return "B"

    judge = CustomJudge()
    request = judge.build_request("q?", "alpha", "beta")
    assert request.messages == (system_message("Custom directive."), user_message("beta|alpha"))
    assert judge.parse("A") == "B"


# --- AnswerPairwiseJudge (SPO's reference-free OvO evaluator) ---


def _answer_judge_request_text() -> str:
    judge = AnswerPairwiseJudge()
    request = judge.build_request("what animal is this?", "A careful answer.", "A brief answer.")
    return ScriptedMockBackend.render_request_text(request)


def test_answer_judge_registered_in_engine_registry():
    assert "answer" in engine_registered("pairwise_judge")
    assert engine_get("pairwise_judge", "answer") is AnswerPairwiseJudge
    assert issubclass(AnswerPairwiseJudge, PairwiseJudge)


def test_answer_judge_request_has_no_gold_field():
    """The reference-free contract: the serialized request carries the question +
    the two ANSWERS and nothing else — no gold field, no reference text, and the
    word "gold" never appears (spo wires this judge for in-loop selection; the gold
    labels must be structurally invisible to it)."""
    text = _answer_judge_request_text()
    assert "what animal is this?" in text
    assert "(A) A careful answer." in text and "(B) A brief answer." in text
    assert "gold" not in text.lower()
    assert "ground truth" not in text.lower()
    # the request is answer-vs-answer: it asks WHICH answer is better, never whether
    # either answer MATCHES a label
    assert "two answers to the same question" in text


def test_answer_judge_parses_ab_tie():
    judge = AnswerPairwiseJudge()
    assert judge.parse("A") == "A"
    assert judge.parse(" B is better ") == "B"
    assert judge.parse("too close to call") == "tie"


# --- judge-family contract: shared request task-line + A/B/tie parse (both classes) ---


def test_pairwise_judges_system_prompts_differ():
    assert (
        DefaultPairwiseJudge().build_request("q", "a", "b").messages[0].content
        != AnswerPairwiseJudge().build_request("q", "a", "b").messages[0].content
    )


@pytest.mark.parametrize("cls", [DefaultPairwiseJudge, AnswerPairwiseJudge])
class TestPairwiseJudgeContract:
    def test_task_line_shared(self, cls):
        req = cls().build_request("Q?", "alpha", "beta")
        assert user_text(req) == "Question: Q?\n(A) alpha\n(B) beta\nAnswer with A or B."

    @pytest.mark.parametrize(
        "raw,want",
        [
            ("  A.", "A"),
            ("  a.", "tie"),  # lowercase is not a verdict token
            ("B is better", "B"),
            ("TIE", "tie"),
            ("", "tie"),
            ("perhaps A", "A"),  # first standalone token wins, wherever it sits
        ],
    )
    def test_parse_verdicts(self, cls: type[PairwiseJudge], raw: str, want: str) -> None:
        assert cls().parse(raw) == want
