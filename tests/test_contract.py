"""Contract tests (zero GPU): prove the new abstractions carry the algorithms + a compare-only method rejected at assembly time.

Propositions (the root causes being treated):
1. hill_climb runs end to end on ScriptedMockBackend (new Strategy/Budget/EvalModel/Loop interplay).
2. A compare-only method installed on an accuracy board -> assembly-time ValueError, not a ``—`` reported after the run (probed by the in-file CompareOnlyStrategy fixture, so the guarantee survives any single method's retirement).
3. RequestPlanner as the single gate: over the image limit -> typed Rejected.
4. durable SampleOutcome: per-sample prediction preserved beyond the cache.
5. Cost attributed per prompt: sum(per-candidate cost.requests) == requests actually issued, no double counting within a batch.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, ClassVar

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    SamplingConfig,
    TextPart,
    system_message,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.core.objective import Objective
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.budget import BanditBudget, LoadPlan, WorkItem
from eyas.engine.capability import COMPARE_EVAL, BackendLimits, CapSpec, ResourceProfile
from eyas.engine.domain import (
    Archive,
    Cost,
    Evaluation,
    OptimizerState,
    Prompt,
    PromptListState,
    SampleOutcome,
    rank_by_score,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
    CompareEvalModel,
    Evaluator,
    SqliteRecordStore,
)
from eyas.engine.genio import (
    GEN_MARKER,
    MAX_BODY_LEN,
    OUT_END,
    OUT_START,
    render_sample_reference,
)
from eyas.engine.planner import PlannedRequest, Rejected, RequestPlanner
from eyas.engine.registry import get as engine_get
from eyas.engine.registry import register_strategy
from eyas.engine.registry import registered as engine_registered
from eyas.engine.strategies.ape import ApeConfig, ApeStrategy
from eyas.engine.strategies.cot import CotConfig, CotStrategy
from eyas.engine.strategies.evoprompt import EvopromptConfig, EvopromptStrategy
from eyas.engine.strategies.hill_climb import HillClimbConfig, HillClimbStrategy
from eyas.engine.strategies.opro import OproConfig, OproStrategy, serialize_meta_prompt
from eyas.engine.strategies.protegi import ProtegiConfig, ProtegiState, ProtegiStrategy
from eyas.engine.strategies.raw import RawConfig, RawStrategy
from eyas.engine.strategies.spo import (
    SpoConfig,
    SpoState,
    SpoStrategy,
    _ensure_choices_placeholder,
)
from eyas.engine.strategy import Decision, EvalView, OptimizerLoop, Strategy, build_run_context
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.math_qa import MathQaTask
from eyas.tasks.protocol import IdentityProtocol

_LABELS = {"q1": "cat", "q2": "dog"}


def _toy_samples() -> list[Sample]:
    return [
        Sample(id="q1", inputs={"question": "q1", "image": "q1.png"}, references={"label": "cat"}),
        Sample(id="q2", inputs={"question": "q2", "image": "q2.png"}, references={"label": "dog"}),
    ]


class ToyMCQ(Task):
    name = "toy_mcq"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return _toy_samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


class ToyCompare(Task):
    name = "toy_compare"
    objectives = (Objective(key="compare_score"),)
    default_prompt = "Compare."

    def load_split(self, split: str) -> Sequence[Sample]:
        return _toy_samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(messages=(user_message(sample.inputs["question"]),))

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"compare_score": 1.0}


def _mock() -> ScriptedMockBackend:
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        question = user_text(request)
        sys_msgs = [m for m in request.messages if m.role == "system"]
        prompt_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        label = _LABELS.get(question, "")
        answer = label if "step by step" in prompt_text.lower() else "wrong"
        return ChatResponse(text=answer)

    return ScriptedMockBackend(responder=responder)


def _gen_mock() -> ScriptedMockBackend:
    """Classification scores correct on step-by-step; generation requests carrying GEN_MARKER return an instruction containing step by step.

    The instruction is bracketed — the official OPRO output format — so the
    ``single_bracket`` yield channel produces a candidate; the OUT tags stay for the
    line/tag parsers of the other generative methods."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if GEN_MARKER in sys_text:
            return ChatResponse(
                text=f"{OUT_START}[Classify the animal carefully and think step by step.]{OUT_END}"
            )
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    return ScriptedMockBackend(responder=responder)


def _run(strategy_cls, config, task, eval_model, seeds, backend=None, **kw):
    ctx = build_run_context(
        strategy_cls,
        config,
        task,
        IdentityProtocol(),
        backend or _mock(),
        eval_model,
        **kw,
    )
    return OptimizerLoop(ctx).run(strategy_cls(config, ctx), list(seeds))


def test_hill_climb_end_to_end_finds_step_by_step():
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        HillClimbStrategy,
        HillClimbConfig(rounds=2, keep_size=2, per_parent=3),
        ToyMCQ(),
        eval_model,
        ["Classify the animal."],
    )
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0
    assert result.total_cost.requests > 0
    assert any(o.correct for o in result.best.per_sample)  # durable per-sample


@dataclass
class CompareOnlyConfig:
    """No parameter fields: candidate prompts are always injected via ``seeds``."""


class CompareOnlyStrategy(Strategy):
    """Synthetic compare-only fixture (test-local name ``_test_compare_only``).

    Declares ONLY compare_eval semantics and produces no absolute metric: mounted
    on an accuracy board it must fail at assembly, mounted on a compare task it
    runs one evaluate round over the seed pool and finishes with the top-ranked
    candidate. The assembly-time guarantee is a framework property and must not
    key on any single method, so the probe lives here."""

    config_cls: ClassVar[type] = CompareOnlyConfig
    state_cls: ClassVar[type] = PromptListState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="_test_compare_only", requires_eval=frozenset({COMPARE_EVAL})
    )

    def expected_rounds(self) -> int | None:
        return 1  # one evaluate round over the pool, then finish

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        return PromptListState(candidates=list(seeds), round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, PromptListState)
        unevaluated = [p for p in state.candidates if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))
        ranked = rank_by_score(state.candidates, view.score)
        return Decision("finish", best_id=ranked[0].id)


def test_compare_only_fixture_rejected_from_accuracy_board():
    # Registered inside the test body: a module-level registration would leak the
    # fixture into the resume harness's registry-derived parametrisation, which is
    # computed at collection time (before any test runs).
    register_strategy("_test_compare_only")(CompareOnlyStrategy)
    assert engine_get("strategy", "_test_compare_only") is CompareOnlyStrategy
    with pytest.raises(ValueError, match="accuracy"):
        _run(CompareOnlyStrategy, CompareOnlyConfig(), ToyMCQ(), CompareEvalModel(), ["seed"])


def test_registry_has_exactly_the_published_strategies() -> None:
    # Guard for the shipped method set: only the published eight are registered
    # (duplicate/unexpected names fail this test). Fixtures registered by other
    # test modules are excluded by module origin.
    published = [
        name
        for name in engine_registered("strategy")
        if engine_get("strategy", name).__module__.startswith("eyas.engine.strategies")
    ]
    assert published == [
        "ape",
        "cot",
        "evoprompt",
        "hill_climb",
        "opro",
        "protegi",
        "raw",
        "spo",
    ]


def test_compare_only_fixture_assembles_on_compare_task():
    ctx = build_run_context(
        CompareOnlyStrategy,
        CompareOnlyConfig(),
        ToyCompare(),
        IdentityProtocol(),
        _mock(),
        CompareEvalModel(),
    )
    assert ctx.primary_objective.key == "compare_score"


def test_planner_rejects_over_image_limit():
    planner = RequestPlanner(ResourceProfile(max_images=1))
    request = ChatRequest(messages=(user_message(ImagePart("a.png"), ImagePart("b.png"), "two"),))
    planned = planner.plan(request)
    assert isinstance(planned, Rejected)
    assert "image" in planned.reason


async def test_compare_channel_picks_step_by_step_winner():
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        # the (A)/(B) payload rides the user message (see test_judge_requests_carry_a_user_message)
        s = user_text(request)
        a_side = s.split("(B)")[0]
        return ChatResponse(text="A" if "step by step" in a_side.lower() else "B")

    backend = ScriptedMockBackend(responder=responder)
    task = ToyCompare()
    ctx = build_run_context(
        CompareOnlyStrategy,
        CompareOnlyConfig(),
        task,
        IdentityProtocol(),
        backend,
        CompareEvalModel(),
    )
    evaluator = Evaluator(
        task=task,
        protocol=IdentityProtocol(),
        backend=backend,
        planner=ctx.evaluator.planner,
        model=CompareEvalModel(),
    )
    good, bad = Prompt("Answer step by step."), Prompt("Answer instantly.")
    plan = LoadPlan(items=(WorkItem(good, ("q1", "q2")), WorkItem(bad, ("q1", "q2"))))
    evals = await evaluator.aevaluate([good, bad], plan, "train")
    by = {e.prompt_id: e.scores["compare_score"] for e in evals}
    assert by[good.id] == 1.0 and by[bad.id] == 0.0


async def test_absolute_cost_attribution_per_prompt():
    # Batch evaluation cost is attributed per prompt: sum(per candidate) == requests actually issued (not double-counted per batch)
    task = ToyMCQ()
    ctx = build_run_context(
        RawStrategy,
        RawConfig(),
        task,
        IdentityProtocol(),
        _mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    a, b = Prompt("Answer now."), Prompt("Answer step by step.")
    plan = LoadPlan(items=(WorkItem(a, ("q1", "q2")), WorkItem(b, ("q1", "q2"))))
    evals = await ctx.evaluator.aevaluate([a, b], plan, "train")
    by = {e.prompt_id: e.cost.requests for e in evals}
    assert by[a.id] == 2 and by[b.id] == 2  # 2 samples x 1 variant each
    assert sum(e.cost.requests for e in evals) == 4


def test_raw_degenerate_path():
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        RawStrategy,
        RawConfig(),
        ToyMCQ(),
        eval_model,
        ["Classify the animal."],
    )
    assert result.best is not None
    assert (
        result.best.scores["accuracy"] == 0.0
    )  # no step by step -> all wrong; the degenerate path still yields metrics
    assert len(result.ranked) == 1
    assert len(result.history) == 1


def test_cot_identity_in_rendering_variant():
    # CoT's algorithmic identity = the rendered "step by step"; the mock scores correct on it (identity lives in rendering, not propose)
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        CotStrategy, CotConfig(base_prompt="Classify the animal."), ToyMCQ(), eval_model, ["unused"]
    )
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0


def _cot_strategy(config: CotConfig) -> CotStrategy:
    ctx = build_run_context(
        CotStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        _mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    return CotStrategy(config, ctx)


def test_cot_preserves_user_seeds_with_the_suffix_appended():
    """User-provided seeds are never silently discarded: each seed renders as
    itself + the CoT suffix (the append mechanism), one candidate per seed."""
    strat = _cot_strategy(CotConfig(base_prompt="BASE", cot_suffix="SUF"))
    state = strat.init([Prompt("user seed one"), Prompt("user seed two")])
    assert [p.text for p in state.candidates] == ["user seed one SUF", "user seed two SUF"]


def test_cot_default_fallback_seed_renders_through_base_prompt():
    """The runtime fallback shape (strategy.seeds unset -> [task.default_prompt])
    keeps the existing base_prompt mechanism: the shipped cot cells carry their
    CoT guidance in base_prompt, which may differ from the task default."""
    strat = _cot_strategy(CotConfig(base_prompt="BASE", cot_suffix="SUF"))
    state = strat.init([Prompt(ToyMCQ.default_prompt)])
    assert [p.text for p in state.candidates] == ["BASE SUF"]
    # an empty suffix leaves the base text verbatim (the shipped quickstart shape)
    strat = _cot_strategy(CotConfig(base_prompt="BASE", cot_suffix=""))
    state = strat.init([Prompt(ToyMCQ.default_prompt)])
    assert [p.text for p in state.candidates] == ["BASE"]


def test_cot_empty_seeds_raise_with_guidance():
    strat = _cot_strategy(CotConfig())
    with pytest.raises(ValueError, match="strategy.seeds"):
        strat.init([])


def test_raw_empty_seeds_raise_with_guidance():
    ctx = build_run_context(
        RawStrategy,
        RawConfig(),
        ToyMCQ(),
        IdentityProtocol(),
        _mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    strat = RawStrategy(RawConfig(), ctx)
    with pytest.raises(ValueError, match="strategy.seeds"):
        strat.init([])


def test_compare_only_fixture_runs_on_compare_task(tmp_path):
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        # the (A)/(B) payload rides the user message (see test_judge_requests_carry_a_user_message)
        s = user_text(request)
        a_side = s.split("(B)")[0]
        return ChatResponse(text="A" if "step by step" in a_side.lower() else "B")

    backend = ScriptedMockBackend(responder=responder)
    ctx = build_run_context(
        CompareOnlyStrategy,
        CompareOnlyConfig(),
        ToyCompare(),
        IdentityProtocol(),
        backend,
        CompareEvalModel(),
        final_eval_split="test",
    )
    strategy = CompareOnlyStrategy(CompareOnlyConfig(), ctx)
    result = OptimizerLoop(ctx).run(strategy, ["Answer step by step.", "Answer instantly."])
    assert result.best is not None
    assert result.best.scores["compare_score"] == 1.0
    # compare semantics skip the closing evaluation (single-candidate compare
    # degenerates to the 0.5 shortcut, re-evaluation is meaningless)
    assert result.final_eval is None


def test_ape_generates_clean_candidate():
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        ApeStrategy,
        ApeConfig(num_candidates=3, demos_per_prompt=2),
        ToyMCQ(),
        eval_model,
        ["seed"],
        backend=_gen_mock(),
    )
    assert result.best is not None
    assert (
        result.best.scores["accuracy"] == 1.0
    )  # the cleaned generated candidate contains step by step


def test_ape_floor_keeps_default_when_generation_echoes_dataset_text():
    """Label-free guard: on a label-free task the generation channel may echo
    dataset text (questions/annotations) instead of instructions. APE's published
    candidate set includes the original instruction; that is also the monotonicity
    floor -- selection must never end up worse than raw."""
    garbage = "what is the name of this magazine?"

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if GEN_MARKER in sys_text:
            return ChatResponse(text=f"{OUT_START}{garbage}{OUT_END}")
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if garbage not in sys_text else "")

    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        ApeStrategy,
        ApeConfig(num_candidates=3, demos_per_prompt=2),
        ToyMCQ(),
        eval_model,
        ["seed"],
        backend=ScriptedMockBackend(responder=responder),
    )
    assert result.best is not None
    assert result.best.prompt_id == Prompt(ToyMCQ.default_prompt).id  # floor wins, not echo
    assert result.best.scores["accuracy"] == 1.0


# --- APE forward-induction fingerprints ---
#
# Paper (arXiv:2211.01910 §3.1, forward mode) + author repo template memo: the model
# induces the instruction from INPUT→OUTPUT demonstrations -- "I gave a friend an
# instruction. Based on the instruction they produced the following input-output
# pairs: [full_DEMO] The instruction was to [APE]", demos "Input: [INPUT]\nOutput:
# [OUTPUT]". Pinned at the RECORDED full request texts (the demo block carries
# Input/Output pairs and the meta-prompt the induction framing).


def _ape_ctx(config: ApeConfig, backend: ScriptedMockBackend, seed: int = 0):
    ctx = build_run_context(
        ApeStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=seed,
    )
    return ctx, ApeStrategy(config, ctx)


def test_ape_induction_conditions_on_input_output_pairs():
    """Every induction request's demo block pairs each ``Input:`` line with a
    nonempty ``Output:`` line (the paper's demos_template), and the completions
    are the selection-split gold answers (the toy split: cat/dog)."""
    backend = _gen_mock()
    ctx, strategy = _ape_ctx(ApeConfig(num_candidates=3, demos_per_prompt=2), backend)
    state = strategy.init([Prompt("seed")])
    assert state.candidates
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(gen) == 3  # one induction request per demo group
    for text in gen:
        lines = text.splitlines()
        input_idx = [i for i, line in enumerate(lines) if line.startswith("Input: ")]
        assert len(input_idx) == 2, "demo block must carry one Input line per demo"
        for i in input_idx:
            assert lines[i + 1].startswith("Output: ")
            assert lines[i + 1][len("Output: ") :].strip(), "gold completion must be nonempty"
        assert "Output: cat" in text and "Output: dog" in text
        # the question text rides the same user_text extraction path the
        # evaluation renders (opro's exemplar ruling), never a raw-input rebuild
        assert "q1" in text and "q2" in text
        # ToyMCQ renders text only, so the base Task hook attaches no media: demo
        # lines are plain "Input: <question>" (the [Image k] form is pinned in
        # tests/test_strategy_ape.py against a real McqTask).
        assert "[Image" not in text


def test_ape_induction_framing_is_the_paper_template():
    """The framing constants are the author-repo forward template verbatim: the
    induction ask ("I gave a friend an instruction...") before the demos, the
    completion cue ("The instruction was to") after them, and the [APE] slot is
    completed by the model itself."""
    from eyas.engine.strategies import ape as ape_mod

    assert (
        ape_mod.INDUCTION_INTRO
        == "I gave a friend an instruction. Based on the instruction they produced "
        "the following input-output pairs:"
    )
    assert ape_mod.INDUCTION_SUFFIX == "The instruction was to"
    backend = _gen_mock()
    ctx, strategy = _ape_ctx(ApeConfig(num_candidates=3, demos_per_prompt=2), backend)
    strategy.init([Prompt("seed")])
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert gen
    for text in gen:
        assert ape_mod.INDUCTION_INTRO in text
        assert ape_mod.INDUCTION_SUFFIX in text
        assert text.index(ape_mod.INDUCTION_INTRO) < text.index("Input: ")
        assert text.index("Output: ") < text.index(ape_mod.INDUCTION_SUFFIX)


def test_ape_demo_renders_list_reference_as_text():
    """Fidelity doc ape D6: a list-valued reference (KIE-style
    multi-value answers) renders its demo ``Output:`` line as natural text
    (``Output: a, b``) — the Python repr (``['a', 'b']``) must never enter
    the induction demos as pseudo-syntax."""
    backend = _gen_mock()
    ctx, strategy = _ape_ctx(ApeConfig(num_candidates=1, demos_per_prompt=1), backend)
    sample = Sample(
        id="k1",
        inputs={"question": "extract the fields", "image": "k1.png"},
        references={"fields": ["a", "b"]},
    )
    lines = strategy._demo_lines(ToyMCQ.default_prompt, [sample])
    # ToyMCQ attaches no demo media, so the line carries no [Image k] marker
    assert lines == ["Input: extract the fields\nOutput: a, b"]
    assert "['a', 'b']" not in lines[0]


def test_opro_uses_history_meta_prompt():
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        OproStrategy,
        OproConfig(rounds=2, num_generated=3),
        ToyMCQ(),
        eval_model,
        ["Classify the animal."],
        backend=_gen_mock(),
    )
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0


# --- OPRO meta-prompt fingerprints ---


def _opro_ctx(config: OproConfig, backend: ScriptedMockBackend):
    ctx = build_run_context(
        OproStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    return ctx, OproStrategy(config, ctx)


def test_opro_trajectory_keeps_top_entries_ascending():
    """Official ``[-max_num_instructions:]`` on the ascending list: as history grows
    beyond ``keep`` the meta-prompt keeps the HIGHEST-scoring entries (the paper:
    "We only keep instructions with the highest scores") and still displays them
    ascending, highest last before the generate-a-better-one directive."""
    history = [(f"instruction {i}", i / 100.0) for i in range(10)]
    block = serialize_meta_prompt(history, keep=3)
    rows = block.splitlines()
    assert rows[0].startswith("[7] instruction 7")
    assert rows[-1] == "[9] instruction 9"
    scores = [int(line.split("]")[0][1:]) for line in rows]
    assert scores == sorted(scores)
    # a bottom-slice would keep the lowest rows; the worst entry must never be shown
    assert "instruction 0" not in block


def test_opro_score_prefix_is_integer_percent():
    """Score display = the official 100-bucket integer (``round(score * 100)``),
    rendered as ``[<int>]``, never a 0–1 float."""
    block = serialize_meta_prompt([("low", 0.0123), ("mid", 0.5), ("hi", 0.999)], keep=8)
    rows = block.splitlines()
    assert [line.split(" ", 1)[0] for line in rows] == ["[1]", "[50]", "[100]"]
    for line in rows:
        assert re.fullmatch(r"\[\d{1,3}\] .+", line)


def test_opro_meta_request_carries_text_only_exemplars():
    """Paper §5.3: exemplars are "critical". Text-only ruling: the block carries
    the question text the evaluation path renders (via the same ``user_text``
    extraction) plus the first reference answer — never image tokens/paths."""
    backend = _gen_mock()
    ctx, strategy = _opro_ctx(OproConfig(rounds=1, num_generated=1), backend)
    result = OptimizerLoop(ctx).run(strategy, ["Classify the animal."])
    assert result.best is not None
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert gen, "no generation request was served"
    meta = gen[0]
    assert "Examples:" in meta
    lines = meta.splitlines()
    ex_idx = next(i for i, line in enumerate(lines) if "Examples:" in line)
    ex_lines = []
    for line in lines[ex_idx + 1 :]:
        if not line.strip():
            break
        ex_lines.append(line)
    assert 0 < len(ex_lines) <= 3
    for line in ex_lines:
        assert line.startswith("Question: ") and " Answer: " in line
    assert "Question: q1 Answer: cat" in meta
    assert "Question: q2 Answer: dog" in meta
    assert "<image" not in meta and ".png" not in meta


def test_opro_exemplar_renders_list_reference_as_text():
    """Same defect class as fidelity doc ape D6: an exemplar whose first
    reference is a list (KIE-style multi-value answers) shows ``Answer: a, b``,
    never the Python repr ``['a', 'b']``."""

    class ToyListRef(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id="k1",
                    inputs={"question": "k1", "image": "k1.png"},
                    references={"fields": ["a", "b"]},
                )
            ]

    backend = _gen_mock()
    cfg = OproConfig(rounds=1, num_generated=1)
    ctx = build_run_context(
        OproStrategy,
        cfg,
        ToyListRef(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    lines = OproStrategy(cfg, ctx)._exemplar_lines("Classify the animal.")
    assert lines == ["Question: k1 Answer: a, b"]


def test_opro_exemplar_answer_uses_the_shared_reference_convention():
    """KIE stores ``entities`` BEFORE ``label``: the exemplar answer must come
    from ``render_sample_reference`` (label wins whenever the key exists — the
    same helper the evaluators use), never the positional first reference."""

    class ToyKieRef(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id="k1",
                    inputs={"question": "k1", "image": "k1.png"},
                    references={"entities": ["a", "b"], "label": "gold"},
                )
            ]

    cfg = OproConfig(rounds=1, num_generated=1)
    ctx = build_run_context(
        OproStrategy,
        cfg,
        ToyKieRef(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    lines = OproStrategy(cfg, ctx)._exemplar_lines("Classify the animal.")
    assert lines == ["Question: k1 Answer: gold"]


# --- OPRO empty-round retry (paper stop clause must not fire on one lazy decode) ---


def _lazy_gen_mock(*, always_lazy: bool) -> ScriptedMockBackend:
    """First generation response (or every one, with always_lazy) is unparsable —
    the textvqa failure shape: a plain sentence without square brackets, so the
    single_bracket channel yields nothing for that completion."""

    gen_calls = {"n": 0}

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if GEN_MARKER in sys_text:
            gen_calls["n"] += 1
            if always_lazy or gen_calls["n"] == 1:
                return ChatResponse(text="Answer the question about the image carefully.")
            return ChatResponse(
                text=f"{OUT_START}[Classify the animal carefully and think step by step.]{OUT_END}"
            )
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    return ScriptedMockBackend(responder=responder), gen_calls


def test_opro_empty_round_retries_and_recovers():
    """Round-1 generation yields nothing (unparsable reply) → the same meta-prompt
    is retried once, the retry's candidate enters the pool, and the run keeps its
    remaining rounds instead of finishing early."""
    backend, gen_calls = _lazy_gen_mock(always_lazy=False)
    result = _run(
        OproStrategy,
        OproConfig(rounds=2, num_generated=1),
        ToyMCQ(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        ["Classify the animal."],
        backend=backend,
    )
    assert gen_calls["n"] >= 2, "empty round was not retried"
    # the retried candidate scored 1.0 (step-by-step wins on the toy task)
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0


def test_opro_empty_retry_exhausted_finishes_with_seed():
    """Every generation stays unparsable: after the single retry the run finishes
    (paper stop clause), issuing exactly 1 + retry_on_empty generation calls — no
    unbounded retry loop."""
    backend, gen_calls = _lazy_gen_mock(always_lazy=True)
    result = _run(
        OproStrategy,
        OproConfig(rounds=5, num_generated=2, retry_on_empty=1),
        ToyMCQ(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        ["Classify the animal."],
        backend=backend,
    )
    # rounds=5 budget is NOT burned: (1 initial + 1 retry) x K=2 requests, then finish
    assert gen_calls["n"] == 2 * 2
    assert result.best is not None
    assert result.best.prompt_id == Prompt("Classify the animal.").id


def test_opro_retry_disabled_finishes_on_first_empty_round():
    """retry_on_empty=0 restores the pre-retry semantics: one empty generation
    round ends the run immediately (config escape hatch)."""
    backend, gen_calls = _lazy_gen_mock(always_lazy=True)
    _run(
        OproStrategy,
        OproConfig(rounds=5, num_generated=2, retry_on_empty=0),
        ToyMCQ(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        ["Classify the animal."],
        backend=backend,
    )
    assert gen_calls["n"] == 2  # single batch of K=2, no retry


def test_opro_meta_request_states_regularizer():
    """Paper §2.2 informal regularizer: the meta-instructions must steer the
    optimizer toward "concise" and "generally applicable" new instructions."""
    backend = _gen_mock()
    ctx, strategy = _opro_ctx(OproConfig(rounds=1, num_generated=1), backend)
    OptimizerLoop(ctx).run(strategy, ["Classify the animal."])
    meta = next(t for t, _ in backend.recorded_requests if GEN_MARKER in t)
    assert "concise" in meta
    assert "generally applicable" in meta


def test_opro_live_run_keeps_top_trajectory_entry():
    """Ascending + top-keep on a live mock run: with ``max_num_instructions=1`` the
    second propose (history = seed 0.0 + offspring 1.0) must show the highest-scoring
    entry and drop the lowest — an inverted slice would show the seed."""
    backend = _gen_mock()
    ctx, strategy = _opro_ctx(
        OproConfig(rounds=2, num_generated=1, max_num_instructions=1), backend
    )
    result = OptimizerLoop(ctx).run(strategy, ["Classify the animal."])
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    # round 2 proposes the identical instruction (mock always answers the same) →
    # dedup empties the round → one same-meta-prompt retry batch before finishing
    assert len(gen) == 3
    # first propose: only the seed (score 0) exists yet
    assert "[0] Classify the animal." in gen[0]
    # second propose: keep=1 shows the top entry, not the worst one
    assert "[100] Classify the animal carefully and think step by step." in gen[1]
    assert "[0] Classify the animal." not in gen[1]
    assert gen[2] == gen[1]  # the retry re-sends the identical meta-prompt


def test_opro_one_candidate_per_completion():
    """Official extraction yield: ONE raw completion contributes at most one new
    candidate (``single_bracket``: the first ``[...]`` slice; echo dropped)."""
    completions = [
        "Here is my suggestion: [Think A.] and also [Think B.] plus [Think C.]",
        "My instruction is: [Think D.]",
    ]
    idx = {"n": 0}

    def responder(request: ChatRequest, sampling: SamplingConfig, i: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if GEN_MARKER in sys_text:
            text = completions[min(idx["n"], len(completions) - 1)]
            idx["n"] += 1
            return ChatResponse(text=text)
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    ctx, strategy = _opro_ctx(
        OproConfig(rounds=1, num_generated=2), ScriptedMockBackend(responder=responder)
    )
    result = OptimizerLoop(ctx).run(strategy, ["Classify the animal."])
    generated = [t for t in result.texts.values() if t.startswith("Think ")]
    # official yield: only Think A. + Think D. — the extra brackets inside one
    # completion never become candidates
    assert generated == ["Think A.", "Think D."]


def test_evoprompt_de_positional_replacement():
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        EvopromptStrategy,
        EvopromptConfig(popsize=2, rounds=1, mode="de"),
        ToyMCQ(),
        eval_model,
        ["Classify A.", "Classify B."],
        backend=_gen_mock(),
    )
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0  # DE greedily adopts the better trial positionally


# --- EvoPrompt operator fingerprints ---
#
# The paper's core contribution is the DESIGNED operators (arXiv:2309.08532 §3.2/§3.3):
# roulette-wheel GA parents (Table 4: wheel beats tournament/random), the two-step
# merge→adjust GA template, and the four-step DE text adaptation with the champion in
# the "Prompt 3" slot. The mock never reads request text, so these must be pinned here.


def _evo_ctx(config: EvopromptConfig, backend: ScriptedMockBackend, seed: int = 42):
    ctx = build_run_context(
        EvopromptStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=seed,
    )
    return ctx, EvopromptStrategy(config, ctx)


def _evo_view(ctx, scores: Mapping[str, float]) -> EvalView:
    """A real EvalView over a fabricated archive: ``scores`` maps prompt text → accuracy.

    An Evaluation without per-sample outcomes evaluates as its aggregate scores
    (evaluation.py backfill), so injecting one per prompt gives the strategy genuine
    ``view.score`` / ``view.evaluated`` answers for arbitrary prompts."""
    archive = Archive(ctx.score_of)
    for text, accuracy in scores.items():
        prompt = Prompt(text)
        archive.add(
            Evaluation(prompt.id, ctx.eval_split, "absolute", {"accuracy": accuracy}, (), Cost())
        )
    return EvalView(archive, ctx.score_of)


def _ga_parent_draws(mode: str, pop: list[Prompt], fitness: list[float], draws: int):
    """Seeded stream of GA parent pairs for one selection mode (fresh strategy/RNG each call)."""
    _, strat = _evo_ctx(
        EvopromptConfig(popsize=len(pop), rounds=1, mode="ga", sel_mode=mode),
        _gen_mock(),
        seed=7,
    )
    rng = strat.derived_rng()
    return [strat._ga_parents(pop, fitness, rng) for _ in range(draws)]


def test_evoprompt_ga_parents_are_fitness_proportional():
    """Paper §3.2 p_i = s_i / Σ_j s_j: under the roulette DEFAULT the first draw is
    the fitness share (1/10 / 1/10 / 8/10, statistical bands over 10 000 seeded
    draws) and the second parent is the next distinct member the wheel lands on —
    a tournament (exact shares 1/6 / 1/6 / 2/3) or a renormalized second draw
    lands far outside the bands. The `sel_mode="tournament"` knob switches the
    regime (self-paired parents prove the two independent tours); the seeded
    stream is reproducible (the derived_rng resume-determinism mechanism)."""
    pop = [Prompt(f"parent {i}") for i in range(3)]
    fitness = [1.0, 1.0, 8.0]
    draws_n = 10_000
    roulette = _ga_parent_draws("roulette", pop, fitness, draws_n)
    assert _ga_parent_draws("roulette", pop, fitness, 1000) == roulette[:1000]  # seeded
    tournament = _ga_parent_draws("tournament", pop, fitness, draws_n)

    # roulette: two distinct wheel draws; marginals within ±0.02 of the paper's
    # proportions (several σ at n=10 000) — a tournament (first-draw shares
    # 1/6 / 1/6 / 2/3) lands far outside these bands
    assert all(a.id != b.id for a, b in roulette)
    counts = Counter(next(i for i, p in enumerate(pop) if p.id == a.id) for a, _ in roulette)
    assert all(abs(counts[i] / draws_n - f / 10.0) < 0.02 for i, f in enumerate(fitness))
    # fitness-8 member appears as SOME parent far more often than a fitness-1 one
    assert sum(1 for a, b in roulette if {a.id, b.id} & {pop[2].id}) > sum(
        1 for a, b in roulette if {a.id, b.id} & {pop[0].id}
    )
    # tournament knob: two INDEPENDENT 2-way tours — a member can pair with itself
    # (~41% expected under the exact shares 1/6 / 1/6 / 2/3), impossible for the wheel
    assert sum(1 for a, b in tournament if a.id == b.id) / draws_n > 0.25
    tour_counts = Counter(next(i for i, p in enumerate(pop) if p.id == a.id) for a, _ in tournament)
    assert tour_counts[2] < 7500  # ≈ 6670, clearly below the wheel's 8000
    # zero fitness degenerates to a uniform wheel (Σ s_j = 0 has no proportions)
    flat = _ga_parent_draws("roulette", pop, [0.0] * 3, 1000)
    assert all(a.id != b.id for a, b in flat)
    flat_counts = Counter(a.id for a, _ in flat)
    assert all(abs(count / 1000 - 1 / 3) < 0.06 for count in flat_counts.values())


def test_evoprompt_ga_init_raises_on_single_distinct_seed():
    """Structural infeasibility is an INIT-time fact: an evoprompt GA pool with
    fewer than 2 distinct prompts (one seed, or duplicates collapsing under
    prompt_hash) raises before any request is issued, naming the arithmetic and
    the fix — never at the first GA round after engine startup. A popsize < 2
    config is rejected even earlier, at construction, in both modes."""
    config = EvopromptConfig(popsize=10, rounds=3, mode="ga")
    # the exact loop shape: OptimizerLoop calls init() before its first evaluate
    with pytest.raises(ValueError, match="population >= 2"):
        _run(
            EvopromptStrategy,
            config,
            ToyMCQ(),
            AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
            ["Classify the animal."],
            backend=_gen_mock(),
        )
    # duplicates dedup by prompt id to one distinct member → the same raise
    _, strat = _evo_ctx(config, _gen_mock())
    with pytest.raises(ValueError, match="strategy.seeds"):
        strat.init([Prompt("same"), Prompt("same")])
    # the guard's message carries the arithmetic and the fix
    with pytest.raises(ValueError) as excinfo:
        strat.init([Prompt("alone")])
    message = str(excinfo.value)
    assert "1 distinct prompt(s)" in message and "popsize=10" in message
    # a two-distinct pool crosses the guard (init returns, seed_eval pending)
    state = strat.init([Prompt("first"), Prompt("second")])
    assert state.phase == "seed_eval"
    # popsize=1 is structurally infeasible in BOTH modes and is rejected at
    # strategy construction (the earliest point the config is knowable) —
    # the degenerate-pool pad in _de_donors stays reachable only as an
    # internal shape guard, never through a shipped popsize=1 config
    for mode in ("ga", "de"):
        bad = EvopromptConfig(popsize=1, rounds=1, mode=mode)
        with pytest.raises(ValueError, match="popsize"):
            _evo_ctx(bad, _gen_mock())


def test_evoprompt_rejects_popsize_below_two_at_construction():
    """popsize < 2 cannot cross (GA) or difference (DE) — the guard fires at
    construction for both modes, naming the knob, never at the first round
    after engine startup."""
    for mode in ("ga", "de"):
        for bad_popsize in (0, 1, -3):
            config = EvopromptConfig(popsize=bad_popsize, rounds=1, mode=mode)
            with pytest.raises(ValueError, match="popsize"):
                _evo_ctx(config, _gen_mock())
    # popsize=2 constructs cleanly in both modes (the minimum viable population)
    for mode in ("ga", "de"):
        _, strat = _evo_ctx(EvopromptConfig(popsize=2, rounds=1, mode=mode), _gen_mock())
        assert strat.config.popsize == 2


def test_evoprompt_negative_fitness_is_clamped_before_the_wheel():
    """Oriented scores can be negative (minimize objectives): a negative share
    must be clamped to 0 BEFORE rng.choices — the raw negative weight otherwise
    distorts the cumulative wheel (a non-monotonic cum_weights list silently
    starves the positive members). fitness [-0.5, 0.2, 0.4] clamps to
    [0, 0.2, 0.4]: member 0 never drawn, member 1 takes a 1/3 share."""
    pop = [Prompt(f"member {i}") for i in range(3)]
    fitness = [-0.5, 0.2, 0.4]
    draws = _ga_parent_draws("roulette", pop, fitness, 3000)
    first_counts = Counter(next(i for i, p in enumerate(pop) if p.id == a.id) for a, _ in draws)
    assert first_counts[0] == 0  # clamped to zero: no share
    assert abs(first_counts[1] / 3000 - 1 / 3) < 0.05  # the unclamped wheel starves it to 0
    assert all(a.id != b.id for a, b in draws)


def test_evoprompt_fitness_none_and_zero_both_carry_zero_share():
    """_fitness distinguishes None (never evaluated) from a real 0.0 score with
    an explicit check — both carry zero wheel share, but the None path must not
    ride truthiness (a future score type must not be silently coerced)."""
    ctx, strat = _evo_ctx(EvopromptConfig(popsize=2, rounds=1, mode="ga"), _gen_mock())
    scored_zero = Prompt("scored zero")
    never_evaluated = Prompt("never evaluated")
    view = _evo_view(ctx, {scored_zero.text: 0.0})
    assert strat._fitness(view, [scored_zero, never_evaluated]) == [0.0, 0.0]


def test_evoprompt_ga_ten_seed_pool_flows_to_population_of_ten():
    """The shipped cell's official N_P=10 pool end to end: the seed round evaluates
    ALL 10 pool members (the pool is evaluated before truncating, and popsize=10 truncates
    nothing), so the GA round that follows crosses among 10 distinct parents — with
    popsize=10 and a 10-seed pool the crossing population is never degenerate."""
    seeds = [f"Classify the animal with cue {i}." for i in range(10)]
    backend = _gen_mock()
    ctx = build_run_context(
        EvopromptStrategy,
        EvopromptConfig(popsize=10, rounds=1, mode="ga"),
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    rounds: list[dict] = []
    result = OptimizerLoop(ctx).run(
        EvopromptStrategy(EvopromptConfig(popsize=10, rounds=1, mode="ga"), ctx),
        list(seeds),
        on_round=rounds.append,
    )
    # the seed round (round 0) carried all 10 pool members as candidates
    seed_round = rounds[0]["history"][0]
    assert seed_round["round"] == 0 and seed_round["n_candidates"] == 10
    # the seed round evaluated ALL 10 pool members (the pool is evaluated in full
    # before truncating; popsize=10 then truncates nothing) — 10 members × 2
    # toy samples. "cue" isolates the seed-pool evals from the later trial eval;
    # the GEN_MARKER exclusion drops the GA calls (they cite pool members in
    # their Prompt 1/2 parent slots).
    seed_eval_requests = [
        t for t, _ in backend.recorded_requests if "cue" in t and GEN_MARKER not in t
    ]
    assert len(seed_eval_requests) == 10 * 2
    assert len({next(s for s in seeds if s in t) for t in seed_eval_requests}) == 10
    # the GA round crossed distinct parents: every child request pairs two DIFFERENT
    # pool members (population = 10 → distinctness never needs padding)
    gen_requests = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(gen_requests) == 10  # N children per generation (Algorithm 2)
    for text in gen_requests:
        slot1 = [s for s in seeds if f"Prompt 1: {s}" in text]
        slot2 = [s for s in seeds if f"Prompt 2: {s}" in text]
        assert len(slot1) == len(slot2) == 1 and slot1[0] != slot2[0]
    # GA update = elitist top-10 over pool ∪ trials: the run completes with a best
    assert result.best is not None
    assert result.history


def test_evoprompt_ga_request_carries_two_step_merge_template():
    """Official GA semantics: a one-shot example, then Step 1 merge/combine-the-core-ideas
    + Step 2 adjust-randomly, one call per population member (Algorithm 2: N children)."""
    from eyas.engine.strategies import evoprompt as evo

    assert "Merge" in evo.GA_STEP1_INSTRUCTION
    assert "combine their core idea" in evo.GA_STEP1_INSTRUCTION
    assert "Adjust" in evo.GA_STEP2_INSTRUCTION and "randomly" in evo.GA_STEP2_INSTRUCTION
    assert evo.OUT_START in evo.GA_STEP2_INSTRUCTION and evo.OUT_END in evo.GA_STEP2_INSTRUCTION

    backend = _gen_mock()
    ctx, strat = _evo_ctx(EvopromptConfig(popsize=2, rounds=1, mode="ga"), backend)
    seeds = ["Classify the animal one-shot.", "Classify the animal two-shot."]
    state = strat.init([Prompt(s) for s in seeds])
    state.phase = "gen"
    view = _evo_view(ctx, dict.fromkeys(seeds, 0.5))
    state.cur_best, state.cur_best_score = state.population[0], 0.5
    trials = strat._variation(state, view)
    assert len(trials) == len(state.population)
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(gen) == 2  # one LLM call per child
    for text in gen:
        assert evo.GA_EXAMPLE in text  # the one-shot example precedes the parents
        assert evo.GA_STEP1_INSTRUCTION in text
        assert evo.GA_STEP2_INSTRUCTION in text
    assert f"Prompt 1: {state.population[0].text}" in gen[0]
    assert "Prompt 2: " in gen[0]


def test_evoprompt_de_request_is_the_four_step_operator():
    """Official DE template slots: Step 1 identify different parts (Prompt 1/2 = the two
    donors), Step 2 mutate them, Step 3 combine with Prompt 3 = the CHAMPION (cur_best,
    not the per-round population max), Step 4 crossover with the Basic Prompt = the base."""
    from eyas.engine.strategies import evoprompt as evo

    for marker in (
        evo.DE_STEP1_INSTRUCTION,
        evo.DE_STEP2_INSTRUCTION,
        evo.DE_STEP3_INSTRUCTION,
        evo.DE_STEP4_INSTRUCTION,
    ):
        assert marker in evo.DE_TEMPLATE
    assert "different parts" in evo.DE_STEP1_INSTRUCTION
    assert "Randomly mutate" in evo.DE_STEP2_INSTRUCTION
    assert "Prompt 3" in evo.DE_STEP3_INSTRUCTION
    assert "Crossover" in evo.DE_STEP4_INSTRUCTION

    backend = _gen_mock()
    ctx, strat = _evo_ctx(EvopromptConfig(popsize=3, rounds=1, mode="de"), backend)
    seeds = [
        "Classify the animal one-shot.",
        "Classify the animal two-shot.",
        "Classify the animal three-shot.",
    ]
    state = strat.init([Prompt(s) for s in seeds])
    state.phase = "gen"
    champion = state.population[0]
    view = _evo_view(ctx, {seeds[0]: 1.0, seeds[1]: 0.5, seeds[2]: 0.25})
    state.cur_best, state.cur_best_score = champion, 1.0
    trials = strat._variation(state, view)
    assert len(trials) == 3
    de_requests = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(de_requests) == 3  # one DE call per population member (Algorithm 3)
    for base, text in zip(state.population, de_requests, strict=True):
        assert evo.DE_EXAMPLE in text  # the one-shot example precedes the slots
        for marker in (
            evo.DE_STEP1_INSTRUCTION,
            evo.DE_STEP2_INSTRUCTION,
            evo.DE_STEP3_INSTRUCTION,
            evo.DE_STEP4_INSTRUCTION,
        ):
            assert marker in text
        assert f"Prompt 3: {champion.text}" in text  # champion wired into the combine slot
        assert f"Basic Prompt: {base.text}" in text
        base_pos = text.index(f"Basic Prompt: {base.text}")
        for other in seeds:
            if other == base.text:
                continue
            donor_positions = [
                text.index(f"Prompt {slot}: {other}")
                for slot in (1, 2)
                if f"Prompt {slot}: {other}" in text
            ]
            # the non-base members are the differencing donors, in the Prompt 1/2 slots,
            # and the basic-prompt slot is the LAST prompt filled in the template
            assert donor_positions and min(donor_positions) < base_pos


def test_evoprompt_de_donors_exclude_the_base():
    """Algorithm 3: r1 ≠ r2 ≠ i — both donors drawn from population \\ {base}."""
    pop = [Prompt(f"member {i}") for i in range(5)]
    _, strat = _evo_ctx(EvopromptConfig(popsize=5, rounds=1, mode="de"), _gen_mock(), seed=3)
    rng = strat.derived_rng()
    for j in range(len(pop)):
        for _ in range(50):
            donor_a, donor_b = strat._de_donors(pop, j, rng)
            assert donor_a.id != donor_b.id
            assert donor_a.id != pop[j].id and donor_b.id != pop[j].id
    # population \ {base} smaller than 2 pads the twin slot with the base itself — the
    # minimum deviation that keeps the per-slot trial shape at degenerate popsize 1
    lone = [Prompt("only one")]
    donor_a, donor_b = strat._de_donors(lone, 0, rng)
    assert donor_a.id == donor_b.id == lone[0].id


class MarkerSeedsToy(Task):
    """Sample s{i} is correct exactly for prompts listing marker s{i} (the backend echoes
    the marker back when the prompt names it). Seed i lists s0..s{i}, so the eight seeds
    get EIGHT DIFFERENT scores and the top-popsize truncation has a unique right answer."""

    name = "toy_marker_seeds"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(
                id=f"s{i}",
                inputs={"question": f"s{i}", "image": f"s{i}.png"},
                references={"marker": f"s{i}"},
            )
            for i in range(8)
        ]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        hit = 1.0 if sample.references["marker"] in response.text else 0.0
        return Prediction(value=hit, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": float(prediction.value)}


def _marker_seed_mock() -> ScriptedMockBackend:
    """Echo backend: answers a sample with its question when the prompt names that
    marker, otherwise with text that matches nothing."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_text = "".join(
            p.text for m in request.messages if m.role == "system" for p in m.content
        )
        question = user_text(request)
        return ChatResponse(text=question if question in sys_text else "no-marker")

    return ScriptedMockBackend(responder=responder)


def test_evoprompt_init_evaluates_pool_then_truncates_top_popsize():
    """The initial population is the TOP-popsize of the EVALUATED pool — all 8 seeds
    are evaluated first; truncation happens only once the scores exist (cutting the pool
    to popsize BEFORE evaluation would make the rank/truncate path dead)."""
    backend = _marker_seed_mock()
    seeds = [f"Classify. markers: {', '.join(f's{k}' for k in range(i + 1))}" for i in range(8)]
    cfg = EvopromptConfig(popsize=4, rounds=1, mode="de")
    ctx = build_run_context(
        EvopromptStrategy,
        cfg,
        MarkerSeedsToy(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    snapshots: list[dict] = []
    eval_requests_at_round: list[int] = []

    def on_round(snapshot: dict) -> None:
        snapshots.append(snapshot)
        eval_requests_at_round.append(
            sum(1 for t, _ in backend.recorded_requests if GEN_MARKER not in t)
        )

    result = OptimizerLoop(ctx).run(EvopromptStrategy(cfg, ctx), list(seeds), on_round=on_round)
    # the FIRST round evaluated ALL EIGHT seeds (8 prompts × 8 samples); truncating
    # to popsize=4 before evaluating anything would be wrong
    assert result.history[0].n_candidates == 8
    assert eval_requests_at_round[0] == 8 * 8
    # truncation happened with the scores in hand: the survivors are the top-4 (seeds 7..4,
    # scores 1.0 / 0.875 / 0.75 / 0.625) — keeping the first four would always be
    # wrong. The last checkpoint is the round-end snapshot of the trial round (phase
    # eval_trials: the flip to done happens in the NEXT step, which emits no checkpoint).
    assert snapshots[-1]["state"]["phase"] == "eval_trials"
    assert snapshots[-1]["state"]["population"] == [seeds[i] for i in (7, 6, 5, 4)]


def test_evoprompt_state_serializes_trials_and_cur_best():
    """A mid-round resume restores trials/cur_best from the checkpoint instead of
    crashing on empty trials and losing the champion. Full round-trip (the
    checkpoint.json JSON path)."""
    from eyas.engine.strategies.evoprompt import EvoState

    pop = [Prompt("p one"), Prompt("p two")]
    state = EvoState(
        population=list(pop),
        trials=[Prompt("t one"), Prompt("t two")],
        cur_best=pop[1],
        cur_best_score=0.75,
        round_idx=2,
        phase="eval_trials",
    )
    restored = EvoState.from_json(json.loads(json.dumps(state.to_json())))
    assert [p.text for p in restored.population] == [p.text for p in pop]
    assert [p.text for p in restored.trials] == ["t one", "t two"]
    assert restored.cur_best is not None and restored.cur_best.id == pop[1].id
    assert restored.cur_best_score == 0.75
    assert restored.round_idx == 2
    assert restored.phase == "eval_trials"


def test_evoprompt_mid_round_resume_survives_trial_state():
    """Resuming from the post-generation checkpoint (phase=eval_trials, trials only in
    memory) restores the serialized trials/cur_best, so the resumed round neither
    crashes on mismatched lengths nor loses the champion identity."""
    from eyas.engine.strategies.evoprompt import EvoState

    cfg = EvopromptConfig(popsize=2, rounds=2, mode="de")
    seeds = ["Classify the animal one-shot.", "Classify the animal two-shot."]

    snapshots: list[dict] = []
    ctx, interrupted = _evo_ctx(cfg, _gen_mock())
    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(interrupted, list(seeds), on_round=_capture_eval_trials(snapshots))
    assert snapshots, "no eval_trials checkpoint was produced"
    checkpoint = json.loads(json.dumps(snapshots[-1]))
    # trials must BE in the checkpoint, not just survive one
    assert checkpoint["state"].get("trials"), "trials missing from the eval_trials checkpoint"
    assert checkpoint["state"].get("cur_best") is not None

    ctx, resumed_strategy = _evo_ctx(cfg, _gen_mock())
    resumed = OptimizerLoop(ctx).run(resumed_strategy, list(seeds), resume=checkpoint)
    assert resumed.best is not None
    restored = EvoState.from_json(checkpoint["state"])
    assert restored.cur_best is not None and restored.cur_best_score is not None
    assert len(restored.trials) == 2


def _capture_eval_trials(snapshots: list[dict]):
    def on_round(snapshot: dict) -> None:
        snapshots.append(snapshot)
        state = snapshot["state"]
        if state.get("phase") == "eval_trials" and state.get("trials"):
            raise KeyboardInterrupt("synthetic crash at the trial-eval boundary")

    return on_round


def test_protegi_gradient_and_bandit_budget():
    config = ProtegiConfig(beam_size=2, rounds=1, n_gradients=2)
    ctx = build_run_context(
        ProtegiStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        budget=BanditBudget(eval_rounds=2, samples_per_eval=1),
    )
    result = OptimizerLoop(ctx).run(ProtegiStrategy(config, ctx), ["Classify the animal."])
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0  # the gradient rewrite adopts step by step


# --- ProTeGi mechanism fingerprints ---
#
# Paper (arXiv:2305.03495) Algorithm 2: a TWO-CALL dialogue per parent (∇ critique →
# δ edit fed the parsed gradient), a paraphrase (mc) expansion pass capped at
# max_expansion_factor, and error groups RANDOMLY sampled. Selection ranks by the
# bandit's pooled Q̂ (SelectTop_b(Q_T)), never by archive best-subset scores.
# The mock never reads request text, so these must be pinned here.


def _protegi_ctx(
    config: ProtegiConfig,
    backend: ScriptedMockBackend,
    task: Task | None = None,
    seed: int = 42,
    **kw,
):
    ctx = build_run_context(
        ProtegiStrategy,
        config,
        task or ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=seed,
        **kw,
    )
    return ctx, ProtegiStrategy(config, ctx)


def test_protegi_gradient_is_a_critique_then_edit_dialogue():
    """Algorithm 2 ∇→δ: the critique request asks for tagged reasons why the prompt got
    the sampled errors wrong; the edit request embeds the PARSED gradient text after
    "using this feedback:" — never the raw tagged completion (which would echo the
    <START>/<END> markers back)."""
    from eyas.engine.strategies import protegi as pg

    assert "reasons why" in pg.CRITIQUE_TEMPLATE
    assert "using this feedback" in pg.EDIT_TEMPLATE
    assert "Paraphrase" in pg.SYNONYM_TEMPLATE
    assert "keeping the semantic meaning" in pg.SYNONYM_TEMPLATE

    backend = _gen_mock()
    ctx, strat = _protegi_ctx(ProtegiConfig(rounds=1, beam_size=2, n_gradients=1), backend)
    result = OptimizerLoop(ctx).run(strat, ["Classify the animal."])
    assert result.best is not None
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    critiques = [t for t in gen if "reasons why" in t]
    edits = [t for t in gen if "using this feedback" in t]
    assert critiques, "no critique request was issued"
    assert edits, "no edit request was issued"
    assert gen.index(critiques[0]) < gen.index(edits[0])  # critique precedes the edit
    critique = critiques[0]
    assert f'My current prompt is: "{ToyMCQ.default_prompt}"' in critique  # official A.1 slot
    assert "Text: q1 / Label: cat / Prediction: wrong" in critique  # _sample_error_str blocks
    assert "Give 1 reasons why" in critique  # official literal, num_feedbacks rendered
    assert "Wrap each reason with <START> and <END>" in critique
    gradient = "[Classify the animal carefully and think step by step.]"
    for text in edits:
        assert f"using this feedback: {gradient}," in text  # PARSED gradient injected
        # the raw tagged completion never reaches the edit prompt: critique markers
        # are parsed away (the edit template itself only ever speaks the OUT tags)
        assert "<START>" not in text and "<END>" not in text


def test_protegi_critique_renders_list_reference_as_text():
    """Fidelity doc ape D6 invariant: a wrong sample whose
    reference is a list (KIE-style multi-value answers) must render its
    ``Label:`` slot in the critique error string as natural text, never the
    Python repr (which also made the gold permanently unmatchable in the
    error-pool filter)."""

    class _ProtegiListRefToy(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id=f"q{i}",
                    inputs={"question": f"q{i}", "image": f"q{i}.png"},
                    references={"label": [f"a{i}", f"b{i}"]},
                )
                for i in (1, 2)
            ]

    backend = _gen_mock()
    ctx, strat = _protegi_ctx(
        ProtegiConfig(rounds=1, beam_size=2, n_gradients=1),
        backend,
        task=_ProtegiListRefToy(),
    )
    labels = strat._labels_by_sample()
    assert labels["q1"] == ("q1", "a1, b1")  # rendered gold, not the Python repr
    OptimizerLoop(ctx).run(strat, ["Classify the animal."])
    critiques = [t for t, _ in backend.recorded_requests if GEN_MARKER in t and "reasons why" in t]
    assert critiques, "no critique request was issued"
    assert "Label: a1, b1" in critiques[0] or "Label: a2, b2" in critiques[0]
    assert "['a1', 'b1']" not in critiques[0] and "['a2', 'b2']" not in critiques[0]


def test_protegi_labels_read_the_task_reference_key():
    """GSM8K cells: math_qa samples carry references={"answer": ...}; the critique
    label must render that value, not an empty string from a fixed "label" key."""
    samples = [
        Sample(id=f"m{k}", inputs={"question": f"{k}+1?"}, references={"answer": str(k + 1)})
        for k in range(2)
    ]
    ctx, strat = _protegi_ctx(
        ProtegiConfig(rounds=1),
        ScriptedMockBackend(responder=lambda request, sampling, idx: ChatResponse(text="x")),
        task=MathQaTask(samples=samples),
    )
    assert strat._labels_by_sample() == {"m0": ("0+1?", "1"), "m1": ("1+1?", "2")}


def test_render_sample_reference_prefers_label_over_position():
    """KIE stores its entities list BEFORE the label; a positional read would
    render dict reprs as the gold, so the label key must win."""
    label = '[{"name": "ACME", "address": "1 Main St"}]'
    assert (
        render_sample_reference(
            {"entities": [{"name": "ACME", "address": "1 Main St"}], "label": label}
        )
        == label
    )
    assert render_sample_reference({"answer": "7"}) == "7"
    assert render_sample_reference({}) == ""


def test_protegi_labels_prefer_the_label_key_when_present():
    """The sroie boards' samples carry entities first; the critique Label slot
    must render the serialized label, never the entities list."""
    label = '[{"name": "ACME", "address": "1 Main St"}]'

    class _ProtegiKieRefToy(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id="k1",
                    inputs={"question": "extract"},
                    references={
                        "entities": [{"name": "ACME", "address": "1 Main St"}],
                        "label": label,
                    },
                )
            ]

    ctx, strat = _protegi_ctx(
        ProtegiConfig(rounds=1),
        ScriptedMockBackend(responder=lambda request, sampling, idx: ChatResponse(text="x")),
        task=_ProtegiKieRefToy(),
    )
    assert strat._labels_by_sample() == {"k1": ("extract", label)}


def test_protegi_expansion_pools_paraphrases_under_the_cap():
    """MC expansion: synonym requests carry the official paraphrase wording; the
    offspring pool is UNIQUE (paraphrase-of-paraphrase duplicates dropped), parents
    included (official `new_task_sections + [task_section]`), capped at
    max_expansion_factor per parent over the shared pool before selection."""
    backend = _gen_mock()
    cfg = ProtegiConfig(rounds=1, beam_size=2, n_gradients=1, mc_samples=2)
    ctx, strat = _protegi_ctx(cfg, backend)
    pools: list[list[str]] = []

    def crash(snapshot: dict) -> None:
        state = snapshot["state"]
        if state.get("offspring"):
            pools.append(list(state["offspring"]))
            raise KeyboardInterrupt("capture the post-expansion offspring pool")

    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(strat, ["Classify the animal."], on_round=crash)
    paraphrases = [t for t, _ in backend.recorded_requests if "paraphras" in t.lower()]
    assert paraphrases, "no paraphrase (mc) request was issued"
    assert "keeping the semantic meaning" in paraphrases[0]
    # the paraphrase phase runs on the dedicated synonym tier
    assert ctx.tier_requests.get("synonym") == len(paraphrases)
    assert pools
    offspring = pools[-1]
    assert len(offspring) <= cfg.max_expansion_factor
    assert len(offspring) == len(set(offspring))  # unique pool (dedup before truncation)
    assert any("step by step" in t for t in offspring)  # edit + mc fanout produced successors
    # offspring holds SUCCESSORS only — parents re-enter selection from state.beam
    # (the pool = beam + offspring mirrors official ``new_prompts += prompts``)
    assert ToyMCQ.default_prompt not in offspring


def test_protegi_error_groups_are_seeded_samples():
    """`random.sample` semantics: 8 distinct errors, group size 4 → every seed's first
    group is a 4-sample mixture, and two seeds draw DIFFERENT groups (a head slice is
    seed-invariant and fails the cross-seed inequality)."""
    n_errors = 8

    class EightErrorToy(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id=f"q{i}",
                    inputs={"question": f"q{i}", "image": f"q{i}.png"},
                    references={"label": "cat"},
                )
                for i in range(n_errors)
            ]

    def first_group(seed: int) -> str:
        cfg = ProtegiConfig(rounds=1, beam_size=1, n_gradients=1, n_errors_per_grad=4)
        ctx, strat = _protegi_ctx(cfg, _gen_mock(), EightErrorToy(), seed=seed)
        prompt = Prompt(ToyMCQ.default_prompt)
        outcomes = tuple(
            SampleOutcome(f"q{i}", "wrong", {"accuracy": 0.0}) for i in range(n_errors)
        )
        archive = Archive(ctx.score_of)
        archive.add(Evaluation(prompt.id, "train", "absolute", {"accuracy": 0.0}, outcomes, Cost()))
        view = EvalView(archive, ctx.score_of)
        errors = strat._error_group(view, prompt, strat._labels_by_sample())
        assert len(errors) == n_errors  # the pool is the FULL error set (no head slice)
        group = strat._sample_group(errors, strat.derived_rng())
        return ",".join(q for q, _, _ in group)

    groups = {seed: first_group(seed) for seed in (1, 2)}
    for text in groups.values():
        assert len(text.split(",")) == 4
    assert groups[1] != groups[2]


def test_protegi_beam_ranks_by_bandit_q_hat_not_archive_scores():
    """SelectTop_b(Q_T) on the DEGENERATE fixture shape (beam == arms, offspring
    empty — the arm basis coincides with the pool, documented as degenerate);
    the in-flight shape (arms == offspring, distinct from a carryover beam) is
    pinned by ``test_protegi_beam_ranks_arms_by_identity_not_position``.

    Inverted fixture: arm0 pulled high (0.9), arm1 pulled low (0.1), but arm1's
    ARCHIVE score higher (1.0) — the bandit beam keeps arm0 FIRST (pooled Q̂
    descending) while the archive-score ranking (the paper's Unif comparator)
    would put arm1 first."""
    cfg = ProtegiConfig(rounds=1, beam_size=2)
    arms = [Prompt("arm zero"), Prompt("arm one")]
    budget = BanditBudget(eval_rounds=1, samples_per_eval=1)
    ctx, _ = _protegi_ctx(cfg, _gen_mock(), budget=budget)
    budget.propose_load(0, None, arms, ["q1", "q2"])  # one pull round -> exhausted
    evals = [
        Evaluation(a.id, "train", "absolute", {"accuracy": s}, (), Cost())
        for a, s in zip(arms, [0.9, 0.1], strict=True)
    ]
    budget.observe(evals, [0.9, 0.1])
    archive = Archive(ctx.score_of)
    for a, s in zip(arms, [0.5, 1.0], strict=True):  # archive scores deliberately inverted
        archive.add(Evaluation(a.id, "train", "absolute", {"accuracy": s}, (), Cost()))
    view = EvalView(archive, ctx.score_of)
    # beam == arm basis (the degenerate coincidence): positions map 1:1 here
    state = ProtegiState(
        beam=list(arms), offspring=[], arm_order=[a.id for a in arms], round_idx=0, phase="score"
    )
    decision = ProtegiStrategy(cfg, ctx).step(state, view)
    assert decision.action == "finish"
    assert state.beam == [arms[0], arms[1]]  # bandit Q̂ order, NOT the archive-score order
    assert decision.best_id == arms[0].id
    # and the Unif comparator: FlatBudget offers no ranking → archive-score order
    flat_ctx, _ = _protegi_ctx(cfg, _gen_mock())
    state2 = ProtegiState(beam=list(arms), offspring=[], round_idx=0, phase="score")
    ProtegiStrategy(cfg, flat_ctx).step(state2, view)
    assert state2.beam == [arms[1], arms[0]]


def test_protegi_beam_ranks_arms_by_identity_not_position():
    """The arm basis is ``state.offspring`` (what the score phase
    proposes), NOT the mixed ``beam + offspring`` selection pool. A stale
    ``beam_order`` position must never address the pool: with carryover parents
    occupying pool slots 0..1, every arm index applied to the pool would be
    off-by-len(beam) — a positional pick selects a different top-b set than the
    identity-keyed mapping."""
    cfg = ProtegiConfig(rounds=1, beam_size=3)
    budget = BanditBudget(eval_rounds=1, samples_per_eval=1)
    ctx, _ = _protegi_ctx(cfg, _gen_mock(), budget=budget)
    beam = [Prompt("carryover parent A"), Prompt("carryover parent B")]
    arms = [Prompt(f"arm {i}") for i in range(4)]
    budget.propose_load(0, None, arms, ["q1", "q2"])  # arms == the proposed offspring
    budget.observe(
        [
            Evaluation(a.id, "train", "absolute", {"accuracy": r}, (), Cost())
            for a, r in zip(arms[:2], [0.9, 0.1], strict=True)
        ],
        [0.9, 0.1],
    )  # only arm0/arm1 pulled: arm2/arm3 stay zero-pull (∞-first in Q̂ order)
    archive = Archive(ctx.score_of)
    for a in arms:  # arms scored in the archive; carryover parents unscored
        archive.add(Evaluation(a.id, "train", "absolute", {"accuracy": 0.5}, (), Cost()))
    view = EvalView(archive, ctx.score_of)
    state = ProtegiState(
        beam=list(beam),
        offspring=list(arms),
        arm_order=[a.id for a in arms],  # the committed arm basis (identity mapping)
        round_idx=0,
        phase="score",
    )
    decision = ProtegiStrategy(cfg, ctx).step(state, view)
    assert decision.action == "finish"
    # Q̂ order with zero-pull arms first: [arm2, arm3, arm0, arm1]
    assert state.beam == [arms[2], arms[3], arms[0]]
    # Positional application to the mixed pool [A, B, arm0..arm3] would select
    # {pool[2]=arm0, pool[3]=arm1, pool[0]=A} — a DIFFERENT set (A in, arm2/arm3
    # out): a positional off-by-len(beam) pick fails this assertion.
    assert decision.best_id == arms[2].id  # beam head, never the pool-offset pick


def test_protegi_score_phase_skips_the_empty_successor_pool():
    """A fully unparsable/deduped expansion must not burn empty bandit rounds: with no
    successors the score phase goes straight to selection (official: no new prompts →
    nothing for the bandit to pull)."""
    cfg = ProtegiConfig(rounds=1, beam_size=2)
    budget = BanditBudget(eval_rounds=4, samples_per_eval=1)
    ctx, _ = _protegi_ctx(cfg, _gen_mock(), budget=budget)
    arm = Prompt("only survivor")
    archive = Archive(ctx.score_of)
    archive.add(Evaluation(arm.id, "train", "absolute", {"accuracy": 0.5}, (), Cost()))
    view = EvalView(archive, ctx.score_of)
    state = ProtegiState(beam=[arm], offspring=[], round_idx=0, phase="score")
    decision = ProtegiStrategy(cfg, ctx).step(state, view)
    assert decision.action == "finish"  # select + final over the surviving beam
    # the empty pull was never proposed to the bandit
    assert budget._round == 0  # noqa: SLF001


def test_bandit_rank_orders_by_q_hat_with_unvisited_first():
    from eyas.engine.budget import EvenSplitBudget, FlatBudget

    arms = [Prompt(f"arm {i}") for i in range(4)]
    ids = ["s0", "s1", "s2", "s3"]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1, keep_top=0.5)
    assert budget.rank() is None  # before the first round there is nothing to rank
    budget.propose_load(0, None, arms, ids)
    evals = [
        Evaluation(a.id, "train", "absolute", {"accuracy": r}, (), Cost())
        for a, r in zip(arms, [0.1, 0.4, 0.9, 0.7], strict=True)
    ]
    budget.observe(evals, [0.1, 0.4, 0.9, 0.7])
    assert budget.rank() == [2, 3, 1, 0]  # pooled Q̂ descending (SelectTop_b(Q_T))
    # a second pull updates the means; the ranking follows the pooled mean, not UCB
    plan = budget.propose_load(1, None, arms, ids)
    pulled = [i for i, a in enumerate(arms) if a.id in {w.prompt.id for w in plan.items}]
    assert len(pulled) == 2  # narrowed set (keep_top 0.5)
    budget.observe(
        [
            Evaluation(arms[i].id, "train", "absolute", {"accuracy": 0.0}, (), Cost())
            for i in pulled
        ],
        [0.0] * len(pulled),
    )
    ranked = budget.rank()
    qhat = [
        v / c if c else float("inf")
        for v, c in zip(budget._values, budget._counts, strict=True)  # noqa: SLF001
    ]
    assert ranked == sorted(range(4), key=lambda i: (-qhat[i], i))
    assert ranked is not None and ranked[0] == 2  # arm2 (0.45) still leads arm3 (0.35)
    # zero-pull arms rank FIRST (UCB's unvisited optimism), behind nobody but each other
    fresh = BanditBudget(eval_rounds=3, samples_per_eval=1, keep_top=0.5)
    fresh.propose_load(0, None, arms, ids)
    fresh.observe(
        [Evaluation(arms[0].id, "train", "absolute", {"accuracy": 0.9}, (), Cost())], [0.9]
    )
    assert fresh.rank() == [1, 2, 3, 0]
    # allocators without a bandit ranking return None (the Unif-comparator contract)
    assert FlatBudget().rank() is None
    assert EvenSplitBudget(4).rank() is None


def test_protegi_state_serializes_offspring_and_seen():
    """A resumed run restores offspring texts + seen hashes from the checkpoint, so
    the in-flight offspring is never silently discarded."""
    beam = [Prompt("seed prompt")]
    offspring = [Prompt("child one"), Prompt("child two")]
    state = ProtegiState(
        beam=list(beam),
        offspring=list(offspring),
        arm_order=["h1", "h2"],
        seen={"s", "a", "m"},
        round_idx=2,
        phase="score",
    )
    data = json.loads(json.dumps(state.to_json()))
    assert set(data) == {"beam", "offspring", "arm_order", "seen", "round_idx", "phase"}
    restored = ProtegiState.from_json(data)
    assert [p.text for p in restored.beam] == [p.text for p in beam]
    assert [p.text for p in restored.offspring] == ["child one", "child two"]
    assert restored.arm_order == ["h1", "h2"]
    assert restored.seen == {"s", "a", "m"}
    assert restored.round_idx == 2
    assert restored.phase == "score"
    # round-trip through a Prompt rebuild keeps identity (hash, not text equality)
    assert restored.offspring[0].id == offspring[0].id


def test_spo_challenger_loop_end_to_end():
    # SPO form: the OvO vote drives in-loop selection (the gold-absence / vote-count /
    # accept-rule fingerprints below); the ARCHIVE metric the leaderboard ranks on is real
    # task accuracy. The compare EvalModel stays off the board (accuracy tasks need absolute
    # scoring). The judge always votes A (position bias): under the ABBA
    # group that is 2 wins / 2 losses per pair → a TIE → the incumbent does NOT advance;
    # the step-by-step challenger still enters the leaderboard at accuracy 1.0 via its
    # items-round record (the toy window covers the 2-sample split, and the seed's own
    # records answer "wrong" everywhere).
    def responder(request, sampling, idx):
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith("You compare two answers"):
            return ChatResponse(text="A")
        if "###GEN###" in sys_text:
            return ChatResponse(
                text=f"{OUT_START}[Classify carefully and think step by step.]{OUT_END}"
            )
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    backend = ScriptedMockBackend(responder=responder)
    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    ctx = build_run_context(
        SpoStrategy,
        cfg,
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    result = OptimizerLoop(ctx).run(SpoStrategy(cfg, ctx), ["Answer now.", "Answer later."])
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0
    # the best's Evaluation carries FULL-split evidence (here the num_pairs=2
    # window happens to cover the 2-sample toy split, so the closing pass is
    # correctly skipped — test_spo_judge_tie_votes_count_half_not_lost pins
    # the closing pass on a split the window does NOT cover)
    assert len(result.best.per_sample) == len(_LABELS)
    # every archive entry the report ranks on carries the full split
    assert all(len(e.per_sample) == len(_LABELS) for e in result.ranked)


class RawTemplateToy(Task):
    """Minimal emulation of the McqTask raw_template guard: raise when the prompt lacks {choices}."""

    name = "toy_raw_template"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "raw"

    def load_split(self, split: str) -> Sequence[Sample]:
        return _toy_samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        if "{choices}" not in prompt:
            raise ValueError(
                "raw_template mode requires the prompt to contain a {choices} placeholder"
            )
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def test_spo_offspring_keeps_choices_placeholder(tmp_path):
    # Regression anchor: under the raw_template guard, offspring losing {choices}
    # must be backfilled by the spo generation-side fallback, or the whole sweep
    # aborts.
    ctx = build_run_context(
        SpoStrategy,
        SpoConfig(rounds=1),
        RawTemplateToy(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    result = OptimizerLoop(ctx).run(
        SpoStrategy(SpoConfig(rounds=1), ctx),
        ["{choices} Answer now.", "{choices} Answer later."],
    )
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0
    assert _ensure_choices_placeholder("Do it.") == "Do it.\nChoices: [{choices}]"
    assert _ensure_choices_placeholder("Do it. {choices}") == "Do it. {choices}"


def test_spo_over_window_offspring_finishes_on_the_incumbent():
    # A reply missing the <prompt> contract falls back to the raw body; when that
    # body cannot fit the context window, the items round planner-rejects every
    # window sample and the vote tripwire used to abort the whole run. An
    # offspring no sample can render is unusable exactly like a placeholder-less
    # rewrite: keep the incumbent instead of crashing.
    body = "x" * MAX_BODY_LEN  # the extraction cap: ~500 estimated tokens vs the 400 window

    def responder(request, sampling, idx):
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if GEN_MARKER in sys_text:
            return ChatResponse(text=body)
        return ChatResponse(text="wrong")

    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    ctx = build_run_context(
        SpoStrategy,
        cfg,
        ToyMCQ(),
        IdentityProtocol(),
        ScriptedMockBackend(responder=responder),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        backend_limits=BackendLimits(context_window=400),
    )
    result = OptimizerLoop(ctx).run(SpoStrategy(cfg, ctx), ["Answer now.", "Answer later."])
    assert result.best_text() == "Answer now."


class _SpoMidWindowRejectPlanner(RequestPlanner):
    """Mid-window rejection mock: the offspring renders fine on the window's
    FIRST sample but is planner-rejected on a later one (a per-sample
    over-window shape). Every other request passes the wrapped limits."""

    def __init__(self, inner: RequestPlanner, offspring_mark: str, rejected_id: str) -> None:
        super().__init__(inner.profile, inner.context_window)
        self._offspring_mark = offspring_mark
        self._rejected_id = rejected_id

    def plan(self, request: ChatRequest) -> PlannedRequest | Rejected:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if self._offspring_mark in sys_text and user_text(request) == self._rejected_id:
            return self._reject(f"offspring over window on sample {self._rejected_id}")
        return super().plan(request)


def test_spo_mid_window_rejection_finishes_on_the_incumbent():
    # The renderability preflight must probe EVERY window sample, not just the
    # first: a challenger the planner accepts on the window's first sample but
    # rejects on a later one leaves that sample unrecorded (planner-rejected
    # samples never persist records), and the vote tripwire used to abort the
    # whole run mid-flight. The full-window preflight must route it to the
    # existing graceful finish-on-the-incumbent path instead.
    labels = {"q1": "c1", "q2": "c2", "q3": "c3"}

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            return ChatResponse(text="A")
        if _SPO_GEN_SYSTEM in sys_text:
            return ChatResponse(
                text=f"{OUT_START}Classify carefully and think step by step.{OUT_END}"
            )
        if "step by step" in sys_text.lower():
            return ChatResponse(text=labels.get(user_text(request), ""))
        return ChatResponse(text="wrong")

    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=responder)
    ctx, strategy = _spo_ctx(cfg, backend, seed=7, task=_SpoQ3Toy())
    # seed 7 shuffles the 3-sample split to q1,q2,q3: the first window is
    # [q1, q2] — q1 (also the split's first sample, the legacy single probe)
    # is accepted, the LATER window sample q2 is rejected.
    window = _spo_pair_schedule(cfg.num_pairs, seed=7, ids=list(labels), num_pairs=cfg.num_pairs)
    assert window == ["q1", "q2"]
    planner = _SpoMidWindowRejectPlanner(ctx.evaluator.planner, "step by step", window[1])
    ctx.evaluator.planner = planner
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best_text() == "Answer now."
    # rejection observability: the planner saw the over-window offspring sample
    assert any("q2" in reason for reason in planner.reject_counts)
    # the graceful path never issues a vote batch nor evaluates the offspring
    assert _spo_votes(backend) == []
    assert not any("step by step" in t for t, _ in backend.recorded_requests)


def test_spo_without_choices_mode_keeps_offspring_verbatim():
    """prompt_includes_choices=False gates BOTH the C3 rule and the generation-side
    backfill: the offspring body reaches evaluation byte-identical."""
    cfg = SpoConfig(rounds=1, prompt_includes_choices=False)
    backend = ScriptedMockBackend(responder=_spo_judge_responder)
    ctx, strategy = _spo_ctx(cfg, backend)
    OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    texts = [t for t, _ in backend.recorded_requests]
    assert any("Think step by step.\nBe concise." in t for t in texts)
    assert not any("Choices: [{choices}]" in t for t in texts)


def test_spo_without_choices_mode_runs_a_question_only_raw_template():
    """GSM8K-cell regression: MathQaTask's raw_template guard raises on {choices},
    so with the backfill gated off a one-round SPO run must complete without the
    appended Choices line ever entering a request."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            return ChatResponse(
                text=_spo_prefer(lambda t: "step by step" in t)(sys_text + user_text(request))
            )
        if _SPO_GEN_SYSTEM in sys_text:
            return ChatResponse(
                text="<prompt>Work step by step.\nQuestion: {question}\nAnswer:</prompt>"
            )
        return ChatResponse(text="4")

    cfg = SpoConfig(rounds=1, prompt_includes_choices=False)
    samples = [
        Sample(id=f"m{k}", inputs={"question": f"{k}+1?"}, references={"answer": str(k + 1)})
        for k in range(2)
    ]
    backend = ScriptedMockBackend(responder=responder)
    ctx = build_run_context(
        SpoStrategy,
        cfg,
        MathQaTask(samples=samples, raw_template=True),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=42,
    )
    result = OptimizerLoop(ctx).run(
        SpoStrategy(cfg, ctx), ["{question}\nAnswer:", "Be concise.\n{question}\nAnswer:"]
    )
    assert result.best is not None
    assert any("Answer: 1" in t for t in _spo_optimizes(backend))
    assert not any("Choices: [{choices}]" in t for t, _ in backend.recorded_requests)


def test_spo_offspring_missing_required_placeholder_ends_the_run_cleanly():
    """A wholesale rewrite that drops {question} on a raw_template board must not
    reach the evaluator: the run stops on its incumbent instead of raising from
    build_request mid-round."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            return ChatResponse(text=_spo_prefer(lambda t: True)(sys_text + user_text(request)))
        if _SPO_GEN_SYSTEM in sys_text:
            return ChatResponse(text="<prompt>Answer the question.</prompt>")
        return ChatResponse(text="4")

    cfg = SpoConfig(rounds=2)
    samples = [
        Sample(id=f"m{k}", inputs={"question": f"{k}+1?"}, references={"answer": str(k + 1)})
        for k in range(2)
    ]
    backend = ScriptedMockBackend(responder=responder)
    ctx = build_run_context(
        SpoStrategy,
        cfg,
        MathQaTask(samples=samples, raw_template=True),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=42,
    )
    result = OptimizerLoop(ctx).run(
        SpoStrategy(cfg, ctx), ["{question}\nAnswer:", "Be concise.\n{question}\nAnswer:"]
    )
    assert result.best is not None
    assert result.best.prompt_id in {
        Prompt(t).id for t in ("{question}\nAnswer:", "Be concise.\n{question}\nAnswer:")
    }
    assert not any("Answer the question." in t for t, _ in backend.recorded_requests)


def test_record_store_roundtrips_non_str_prediction(tmp_path):
    # KIE predictions are list[dict]: JSON-encoded round-trip through the sqlite column; str predictions pass through as-is.
    store = SqliteRecordStore(path=tmp_path / "s.sqlite")
    store.put(
        SampleOutcome("s1", [{"name": "x", "address": "y"}], {"field_acc": 1.0}), "p1", "train"
    )
    store.put(SampleOutcome("s2", "B", {"accuracy": 1.0}), "p1", "train")
    first = store.get("p1", "s1", "train")
    second = store.get("p1", "s2", "train")
    assert first is not None and first.prediction == [{"name": "x", "address": "y"}]
    assert second is not None and second.prediction == "B"


def test_record_store_rejects_positional_path(tmp_path):
    # Both params are keyword-only by contract: positionally, a path would
    # silently bind to ``namespace`` (store silently in-memory, namespace = a
    # str() of the path) — TypeError at the call site beats that.
    with pytest.raises(TypeError):
        SqliteRecordStore(tmp_path / "s.sqlite")


def test_duplicate_candidates_in_one_evaluate_decision_bill_once():
    # An in-batch twin is never a record-store hit (records fill after generation),
    # so without dedup it would be sent and billed twice; the loop dedups by
    # prompt.id before budget + evaluator — contract 5's "no double counting within a
    # batch" extended from accounting honesty to not sending the duplicate work at all.

    class DupSeedStrategy(RawStrategy):
        def step(self, state: OptimizerState, view: EvalView) -> Decision:
            assert isinstance(state, PromptListState)
            seed = state.candidates[0]
            if not view.evaluated(seed):
                twin = Prompt(seed.text)
                assert twin.id == seed.id  # identity is the normalized text
                return Decision("evaluate", prompts=(seed, twin))
            return Decision("finish", best_id=seed.id)

    result = _run(
        DupSeedStrategy,
        RawConfig(),
        ToyMCQ(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        ["seed"],
    )
    assert result.history[0].n_candidates == 1
    assert result.total_cost.requests == len(_LABELS)  # each sample sent once, not twice


def test_bandit_budget_narrows_adaptively():
    arms = [Prompt(f"arm {i}") for i in range(4)]
    ids = [f"s{i}" for i in range(4)]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1, keep_top=0.5)
    plan0 = budget.propose_load(0, _noop_state(), arms, ids)
    assert len(plan0.items) == 4  # round 0 evaluates every arm
    rewards = [0.0, 0.0, 0.0, 1.0]
    evals = [
        Evaluation(a.id, "t", "absolute", {"accuracy": r}, (), Cost())
        for a, r in zip(arms, rewards, strict=True)
    ]
    budget.observe(evals, rewards)
    plan1 = budget.propose_load(1, _noop_state(), arms, ids)
    assert len(plan1.items) == 2  # adaptively narrowed to the top 50%
    chosen_ids = {w.prompt.id for w in plan1.items}
    assert arms[3].id in chosen_ids  # UCB prefers the high-mean arm


def _noop_state():
    return None  # BanditBudget does not read state


def test_judge_ctx_filled_with_variant_and_user_text():
    # The Evaluator judging context must fill in
    # variant / instruction / user_text -- protocol-aware judges like McqLLMJudge hard-depend on them.
    from eyas.backends.scheduling import run_sync
    from eyas.judges.base import Judge, Verdict

    captured: list[object] = []

    class CaptureJudge(Judge):
        name = "capture"

        def judge(self, response, prediction, sample, ctx):
            captured.append(ctx)
            return Verdict(scores={"accuracy": 1.0}, decided=True)

    planner = RequestPlanner(ResourceProfile(), context_window=None)
    evaluator = Evaluator(
        task=ToyMCQ(),
        protocol=IdentityProtocol(),
        backend=_mock(),
        planner=planner,
        model=AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        judge=CaptureJudge(),
    )
    prompt = Prompt("Classify the animal.")
    plan = LoadPlan(items=(WorkItem(prompt, ("q1", "q2")),))
    evals = run_sync(evaluator.aevaluate([prompt], plan, "train", 0))
    assert len(evals) == 1 and len(captured) == 2
    for ctx in captured:  # type: ignore[union-attr]
        assert ctx.variant is not None
        assert ctx.instruction == "Classify the animal."
        assert ctx.user_text in {"q1", "q2"}  # the user plain text of the actually rendered request


async def test_judge_scoring_fans_out_concurrently():
    # Judge calls must fan out concurrently: a serial judging layer drowns evaluation
    # wall time, so judge_concurrency must take real effect.
    import asyncio

    from eyas.judges.base import Judge, Verdict

    active = 0
    max_active = 0

    class SlowJudgeBackend(ScriptedMockBackend):
        async def agenerate(
            self, requests: Sequence[ChatRequest], sampling: SamplingConfig
        ) -> list[ChatResponse]:
            nonlocal active, max_active
            if len(requests) == 1:  # judge fan-out shape: one request at a time
                active += 1
                max_active = max(max_active, active)
                try:
                    await asyncio.sleep(0.02)
                finally:
                    active -= 1
            return await super().agenerate(requests, sampling)

    class SlowAsyncJudge(Judge):
        name = "slow_async"

        def judge(self, response, prediction, sample, ctx):
            raise NotImplementedError("async path only")

        async def ajudge(self, response, prediction, sample, ctx):
            assert ctx.backend is not None
            await ctx.backend.agenerate(
                [ChatRequest(messages=(user_message("judge"),))], SamplingConfig()
            )
            return Verdict(scores={"accuracy": 1.0}, decided=True)

    class WideToyMCQ(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(id=f"q{i}", inputs={"question": f"q{i}"}, references={"label": ""})
                for i in range(8)
            ]

    prompt = Prompt("Classify the animal.")
    plan = LoadPlan(items=(WorkItem(prompt),))

    def _evaluator(judge_concurrency: int) -> Evaluator:
        return Evaluator(
            task=WideToyMCQ(),
            protocol=IdentityProtocol(),
            backend=SlowJudgeBackend(
                responder=lambda request, sampling, idx: ChatResponse(text="wrong")
            ),
            planner=RequestPlanner(ResourceProfile(), context_window=None),
            model=AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
            judge=SlowAsyncJudge(),
            judge_concurrency=judge_concurrency,
        )

    evals = await _evaluator(64).aevaluate([prompt], plan, "train")
    assert max_active == 8  # all samples judged in flight (a serial implementation is always 1)
    assert len(evals[0].per_sample) == 8
    assert all(o.correct for o in evals[0].per_sample)

    active = max_active = 0
    evals1 = await _evaluator(1).aevaluate([prompt], plan, "train")
    assert max_active == 1  # the narrowed window takes effect
    assert len(evals1[0].per_sample) == 8


def test_ape_stratified_demos_labelless_single_bucket() -> None:
    """Label-free vqa families: the coverage check must use the
    ``"?"`` default that bucketing uses, not bare ``references.get("label")`` —
    label-free samples (open-ended vqa families) made the chosen-label set
    ``{None}`` vs bucket set ``{"?"}`` -- a guaranteed RuntimeError. Label-free
    samples must form one degenerate bucket and build demo groups."""
    from random import Random

    from eyas.engine.strategies.ape import stratified_demo_groups

    samples = [
        Sample(
            id=f"q{i}",
            inputs={"image": f"i{i}.jpg", "question": f"what {i}?"},
            references={"answers": [f"a{i}"]},
        )
        for i in range(6)
    ]
    groups = stratified_demo_groups(samples, 4, 2, Random(0))
    assert len(groups) == 4 and all(len(g) == 2 for g in groups)


# --- SPO self-supervised OvO fingerprints ---
#
# Paper (arXiv:2502.06855 / official FoundationAgents/SPO,
# https://github.com/FoundationAgents/SPO/commit/e8381f07): selection is a
# reference-free OUTPUT-vs-OUTPUT pairwise vote — the evaluator compares the two prompts'
# ANSWERS, never a gold label; φopt consumes the incumbent prompt AND its sample outputs;
# EVALUATION_REPETITION=4 votes per round with a randomized A/B swap; accept iff wins >
# losses. The generic mock responders never read gold, so "gold structurally invisible"
# is pinned at the RECORDED full request texts of a live mocked run.

_SPO_SYSTEM = "You compare two answers to the same question"
_SPO_GEN_SYSTEM = "###GEN### Optimize"


def _spo_judge_responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
    """Judge channel CONTENT-preferring (the label-correct answer over the
    seeds' ``wrong`` answer); execution answers correct only on the step-by-step
    switch; the optimize channel returns a tagged multi-line body."""
    sys_msgs = [m for m in request.messages if m.role == "system"]
    sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
    if sys_text.startswith(_SPO_SYSTEM):
        return ChatResponse(text=_spo_prefer(lambda t: t != "wrong")(sys_text + user_text(request)))
    if _SPO_GEN_SYSTEM in sys_text:
        return ChatResponse(
            text=(
                "<analyse>The incumbent answers briefly.</analyse>"
                "<modification>Add explicit reasoning.</modification>"
                "<prompt>Think step by step.\nBe concise.</prompt>"
            )
        )
    label = _LABELS.get(user_text(request), "")
    return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")


def _spo_voter(win: bool):
    """Fresh-per-run responder rigging the VOTE OUTCOME by CONTENT: when
    ``win`` the judge prefers the challenger's label-correct answer over the
    seeds' ``wrong`` answer, else the reverse — content preference is the only
    way to win under the ABBA swap. The optimize channel returns a FRESH
    step-by-step body per call so every round proposes a distinct challenger
    (an identical body would dedup against the incumbent and end the loop
    early). Under the reject path the incumbent never advances: every optimize
    request cites the original seeds' answer lines only (the seeds' ``wrong``
    answers), never a challenger offspring. The execution channel answers
    correctly only on the step-by-step switch (the challenger's answers are
    then distinguishable from the seeds' ``wrong`` answers)."""
    k = {"n": 0}
    prefer = _spo_prefer(lambda t: t != "wrong" if win else t == "wrong")

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            return ChatResponse(text=prefer(sys_text + user_text(request)))
        if _SPO_GEN_SYSTEM in sys_text:
            k["n"] += 1
            return ChatResponse(text=f"{OUT_START}Variant {k['n']}: think step by step.{OUT_END}")
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    return responder


def _spo_ctx(
    config: SpoConfig,
    backend: ScriptedMockBackend,
    seed: int = 42,
    task: Task | None = None,
    store: Any | None = None,
):
    ctx = build_run_context(
        SpoStrategy,
        config,
        task or ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=seed,
        store=store,
    )
    return ctx, SpoStrategy(config, ctx)


def _spo_votes(backend: ScriptedMockBackend) -> list[str]:
    return [t for t, _ in backend.recorded_requests if _SPO_SYSTEM in t]


def _spo_optimizes(backend: ScriptedMockBackend) -> list[str]:
    return [t for t, _ in backend.recorded_requests if _SPO_GEN_SYSTEM in t]


def _ab_sides(text: str) -> tuple[str, str]:
    m = re.search(r"\(A\) (.*?)\n\(B\) (.*?)\n", text, re.S)
    assert m is not None, f"no (A)/(B) answer blocks in:\n{text}"
    return m.group(1), m.group(2)


def _spo_prefer(pred: Callable[[str], bool]) -> Callable[[str], str]:
    """Content-preferring judge: pick the side whose ANSWER matches ``pred``.
    Position bias can never win a vote under ABBA (that is what the always-A
    fixture proves), so accept/reject scenarios rig CONTENT preference."""

    def vote(text: str) -> str:
        a, b = _ab_sides(text)
        if pred(a) and not pred(b):
            return "A"
        if pred(b) and not pred(a):
            return "B"
        return "tie"  # both/neither preferred: no signal

    return vote


def _spo_pair_schedule(
    n_pairs: int, seed: int = 42, ids: Sequence[str] | None = None, num_pairs: int = 0
) -> list[str]:
    """Replay of the strategy's ONE-SHUFFLE schedule: per round a fresh
    ``derived_rng()`` draw (``Random(seed)`` advances one getrandbits per round),
    ids consumed without replacement, an exhausted schedule appends a reshuffled
    coverage cycle. ``num_pairs`` (when given) sets the per-round window so the
    wrap point lands where the run puts it; otherwise one round of ``n_pairs``.
    """
    rng = Random(seed)
    base = list(ids) if ids is not None else [s.id for s in ToyMCQ().load_split("train")]
    window = num_pairs or n_pairs
    order: list[str] = []
    cursor = 0
    while len(order) < n_pairs:
        draw = Random(rng.getrandbits(64))
        if not order:
            pool = list(base)
            draw.shuffle(pool)
            order = pool
        while cursor + window > len(order):
            pool = list(base)
            draw.shuffle(pool)
            order.extend(pool)
        cursor += window
    return order[:n_pairs]


class _SpoWideToy(ToyMCQ):
    """8-sample split with answers that ECHO the question when the prompt carries the
    step-by-step switch (schedule membership is then visible in the vote answers)."""

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(
                id=f"r{i}",
                inputs={"question": f"r{i}", "image": f"r{i}.png"},
                references={"label": "x"},
            )
            for i in range(8)
        ]


def _spo_wide_responder():
    """Fresh per-run responder: the optimize channel yields a UNIQUE body per
    call (a fixed body would dedup round 2's challenger and end the loop
    early), and the execution channel echoes the question."""
    n = {"k": 0}

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        question = user_text(request)
        if sys_text.startswith(_SPO_SYSTEM):
            return ChatResponse(text="A")
        if _SPO_GEN_SYSTEM in sys_text:
            n["k"] += 1
            return ChatResponse(text=f"{OUT_START}Variant {n['k']}: think step by step.{OUT_END}")
        return ChatResponse(text=question if "step by step" in sys_text.lower() else "wrong")

    return responder


class _SpoQ3Toy(ToyMCQ):
    """3-sample split (q1/q2/q3) — a num_pairs of 2 leaves a remainder, forcing the
    schedule wrap mid-round-group."""

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(
                id=f"q{i}",
                inputs={"question": f"q{i}", "image": f"q{i}.png"},
                references={"label": f"c{i}"},
            )
            for i in range(1, 4)
        ]


def _spo_q3_responder():
    """Fresh per-run responder (unique optimize bodies — see _spo_wide_responder);
    the execution channel answers ans-<question> on the step-by-step switch."""
    n = {"k": 0}

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        question = user_text(request)
        if sys_text.startswith(_SPO_SYSTEM):
            return ChatResponse(text="A")
        if _SPO_GEN_SYSTEM in sys_text:
            n["k"] += 1
            return ChatResponse(text=f"{OUT_START}Variant {n['k']}: think step by step.{OUT_END}")
        return ChatResponse(
            text=f"ans-{question}" if "step by step" in sys_text.lower() else "wrong"
        )

    return responder


def test_spo_ovo_votes_carry_no_gold_and_swap_ab_within_four():
    """Gold discipline: on a live mocked run the recorded FULL request
    texts of every OvO judge call contain neither the gold reference text nor the word
    "gold" — selection inputs are model ANSWERS only. The optimize channel DOES carry
    the real references (paper C3 allows gold there), which makes the absence check in
    the judge channel meaningful rather than an artifact of a label-free fixture.
    Additionally every 4-vote group alternates the A/B positions 2/2 (the ABBA swap,
    official 50% randomization)."""
    gen = {"n": 0}

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            # content-blind position bias: ALWAYS (A). Under the ABBA group the
            # challenger is A on exactly 2 of 4 ballots, so this fixture can
            # never win a vote — it exercises the tie accounting end to end
            return ChatResponse(text="A")
        if _SPO_GEN_SYSTEM in sys_text:
            gen["n"] += 1
            if gen["n"] == 1:  # multi-line <prompt> body — the DOTALL extractor
                return ChatResponse(
                    text="<analyse>a</analyse><modification>m</modification>"
                    "<prompt>Think step by step.\nBe concise.</prompt>"
                )
            # round 2 challenger: fresh id (distinct text → no dedup) but the
            # SAME step-by-step switch → its answer still differs from the
            # never-advancing seed's "wrong" on every pair (always-A voting ties)
            return ChatResponse(
                text="<analyse>a</analyse><modification>m</modification>"
                "<prompt>Plan step by step.\nList evidence.</prompt>"
            )
        # the challenger's execution answer NEVER equals the gold label (it echoes
        # the question), so a gold string inside a judge request can only come
        # from a reference leak — never from the model's own answer
        return ChatResponse(
            text=f"ans-{user_text(request)}" if "step by step" in sys_text.lower() else "wrong"
        )

    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=responder)
    ctx, strategy = _spo_ctx(cfg, backend)
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best is not None
    # the optimize channel cites the real references — gold IS in this run's
    # recorded requests, just never in a judge request
    gens = _spo_optimizes(backend)
    assert gens and "Answer: cat" in gens[0]
    votes = _spo_votes(backend)
    # ONE voting round per challenger (the loop retires a challenger each round)
    assert len(votes) == cfg.num_pairs * cfg.votes_per_pair * cfg.rounds
    for text in votes:
        assert "gold" not in text.lower()
        assert "reference" not in text.lower()
        # the toy gold labels never appear in a judge request (the same sample's
        # execution ANSWER is what the request carries instead)
        assert "cat" not in text and "dog" not in text
    for g in range(0, len(votes), 4):
        sides = [_ab_sides(text) for text in votes[g : g + 4]]
        # ABBA (F,F,T,T): ballots 1/2 incumbent-first, 3/4 challenger-first.
        # The fixture's two sides answer differently on every pair, so the
        # text pins below are unambiguous.
        assert sides[0] == sides[1]  # the two F ballots are identical layouts
        assert sides[2] == sides[3]  # the two T ballots are identical layouts
        assert sides[0] == (sides[3][1], sides[3][0])  # mirrored ABBA
        challenger_a = sum(1 for s in sides if s[0] != "wrong")  # F,F,T,T → 2
        assert challenger_a == 2  # the 2/2 split of A-side occupancy
        assert all(s[0] != s[1] for s in sides)  # the two sides answer differently


def test_spo_votes_bill_the_compare_channel_only():
    """The OvO generations ride the record store (evaluate path); the judge calls bill
    the compare tier via generate — and nothing spo generates rides the 512-token
    generation tier except the optimize call (the resume harness's stream split)."""
    cfg = SpoConfig(rounds=2, num_pairs=16, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_voter(True))
    ctx, strategy = _spo_ctx(cfg, backend)
    OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    votes = _spo_votes(backend)
    # ONE voting round per challenger: rounds=2 → two voting rounds × 16 pairs × 4 votes
    assert len(votes) == cfg.num_pairs * cfg.votes_per_pair * cfg.rounds
    assert ctx.tier_requests.get("compare") == len(votes)


def test_spo_vote_group_of_four_swaps_abba_under_single_pair():
    """One pair, one round → exactly 4 judge requests; the challenger's ANSWER
    (the label-correct one, the incumbent's being ``wrong``) occupies the (A)
    slot in exactly 2 of them (the ABBA/BAAB 2/2 alternation)."""
    cfg = SpoConfig(rounds=1, num_pairs=1, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_judge_responder)
    ctx, strategy = _spo_ctx(cfg, backend)
    OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    votes = _spo_votes(backend)
    assert len(votes) == 4
    challenger_first = sum(1 for text in votes if _ab_sides(text)[0] != "wrong")
    assert challenger_first == 2  # ABBA: the challenger holds (A) in exactly 2 of 4


def test_spo_accept_rule_challenger_wins_advance_incumbent():
    """Accept iff wins > losses: the content-preferring voter picks the
    challenger's answer on every ballot (4-0 per pair). The fresh-offspring
    generator makes the optimize request of round 2 CITE round 1's challenger —
    only an advanced challenger is ever cited."""
    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_voter(True))
    ctx, strategy = _spo_ctx(cfg, backend)
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best is not None
    gens = _spo_optimizes(backend)
    assert len(gens) == 2  # one optimize per voting round (r1, r2)
    # round 2 cites the round-1 challenger (the voter advanced it)
    assert "Variant 1: think step by step." in gens[1]
    assert "Variant 2" not in gens[0]


def test_spo_accept_rule_challenger_loses_keep_incumbent():
    """The content-preferring voter picks the SEED's answer on every ballot
    (0-4 per pair) → the incumbent NEVER advances: every optimize request cites
    the original seeds' answer lines only (the seeds' wrong answers), never a
    challenger offspring."""
    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_voter(False))
    ctx, strategy = _spo_ctx(cfg, backend)
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best is not None
    gens = _spo_optimizes(backend)
    assert len(gens) == 2
    for text in gens:
        assert "Variant" not in text
        assert "Answer (model): wrong" in text


def test_spo_accept_rule_boundary_three_quarter_wins_advance_one_quarter_stays():
    """The exact accept-rule boundary (position-blind vote rigging, ABBA
    side layout [B,B,A,A] per group): 3 clean challenger wins + 1 loss per pair
    → wins 3 > losses 1 → the incumbent ADVANCES at round 1 (round 2's optimize
    cites the challenger); 1 win + 3 losses → wins < losses → the incumbent
    STAYS (every optimize cites only the seed's wrong-answer lines)."""

    def rig(wins: tuple[bool, ...]):
        # ABBA group layout [F,F,T,T] → the challenger sits at B,B,A,A
        champ = ("B", "B", "A", "A")
        anti = ("A", "A", "B", "B")
        n = {"k": 0}

        def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
            sys_msgs = [m for m in request.messages if m.role == "system"]
            sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
            if sys_text.startswith(_SPO_SYSTEM):
                j = n["k"] % 4
                n["k"] += 1
                return ChatResponse(text=champ[j] if wins[j] else anti[j])
            if _SPO_GEN_SYSTEM in sys_text:
                return ChatResponse(text=f"{OUT_START}Think step by step.{OUT_END}")
            label = _LABELS.get(user_text(request), "")
            return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

        return responder

    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)

    # 3 clean challenger wins + 1 loss per pair → wins 3 > losses 1 → advance
    backend = ScriptedMockBackend(responder=rig((True, True, True, False)))
    ctx, strategy = _spo_ctx(cfg, backend)
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best is not None
    gens = _spo_optimizes(backend)
    assert len(gens) == 2
    assert "Think step by step." in gens[1]  # round 2 cites the advanced challenger

    # 1 win + 3 losses per pair → wins < losses → the incumbent stays throughout
    backend = ScriptedMockBackend(responder=rig((True, False, False, False)))
    ctx, strategy = _spo_ctx(cfg, backend)
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best is not None
    gens = _spo_optimizes(backend)
    assert len(gens) == 2
    for text in gens:
        assert "Answer now." in text  # every round cites the seed
        assert "Think step by step." not in text  # the seed stayed incumbent throughout


def test_spo_tie_counts_half_needs_strict_win_sum():
    """2-2 per pair → 0.5 + 0.5 + 0.5 + 0.5 == 2.0 == losses 2.0: NOT wins > losses,
    so the incumbent stays despite the vote splitting evenly. A per-vote 0.5 accounting
    (2 + 4×0.5 > 2) would accept and round 2's optimize would cite the challenger."""
    seen = {"n": 0}

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            seen["n"] += 1
            return ChatResponse(text="A" if seen["n"] % 2 else "B")
        if _SPO_GEN_SYSTEM in sys_text:
            return ChatResponse(text=f"{OUT_START}Think step by step.{OUT_END}")
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=responder)
    ctx, strategy = _spo_ctx(cfg, backend)
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best is not None
    assert seen["n"] == 16  # two voting rounds (2 pairs × 4 votes each)
    gens = _spo_optimizes(backend)
    assert len(gens) == 2
    # the tie kept the seed incumbent: round 2's optimize request cites the
    # SEED (had the 2-2 vote advanced the challenger, the reference prompt
    # slot would carry its "Think step by step." text)
    assert "Think step by step." not in gens[1]
    # the seed holds a full-split archive record (the closing pass runs on it)
    seed_records = [e for e in result.ranked if e.prompt_id == Prompt("Answer now.").id]
    assert seed_records and len(seed_records[0].per_sample) == len(_LABELS)


def test_spo_optimize_request_carries_incumbent_outputs_and_contract():
    """φopt consumes (P*, A*): the optimize request embeds the incumbent text, its
    sampled samples' Question / Answer (reference) / Answer (model output) lines,
    the task requirements, and demands the <analyse>/<modification>/<prompt> contract.
    Images never enter the request."""
    cfg = SpoConfig(rounds=1, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_judge_responder)
    ctx, strategy = _spo_ctx(cfg, backend)
    OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    gens = _spo_optimizes(backend)
    assert gens, "no optimize request was served"
    text = gens[0]
    assert "Answer now." in text  # the reference prompt P*
    assert "Q: q1" in text and "Q: q2" in text
    assert "Answer: cat" in text and "Answer: dog" in text  # the references (C3 slot)
    assert "Answer (model): wrong" in text  # the outputs A* the optimization works from
    for tag in ("<analyse>", "<modification>", "<prompt>"):
        assert tag in text
    assert "{choices}" in text  # the mcq raw_template placeholder rule
    assert "<image" not in text and ".png" not in text


def test_spo_optimize_prompt_renders_list_reference_as_text():
    """Fidelity doc ape D6 invariant: a wrong sample whose
    reference is a list (KIE-style multi-value answers) must render its
    ``Answer:`` line as natural text, never the Python repr."""

    class _SpoListRefToy(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id=f"q{i}",
                    inputs={"question": f"q{i}", "image": f"q{i}.png"},
                    references={"label": [f"a{i}", f"b{i}"]},
                )
                for i in (1, 2)
            ]

    cfg = SpoConfig(rounds=1, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_judge_responder)
    ctx, strategy = _spo_ctx(cfg, backend, task=_SpoListRefToy())
    OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    gens = _spo_optimizes(backend)
    assert gens, "no optimize request was served"
    text = gens[0]
    assert "Answer: a1, b1" in text and "Answer: a2, b2" in text
    assert "['a1', 'b1']" not in text and "['a2', 'b2']" not in text


def test_spo_init_warns_when_extra_seeds_are_dropped(caplog):
    """Single-incumbent chaining is paper-correct, but a SILENT
    drop of provided seeds violates the fail-fast spirit — with more than one
    seed, ``init`` must name the drop on a WARNING while keeping the exact
    single-incumbent state."""
    eyas_root = logging.getLogger("eyas")
    old_propagate = eyas_root.propagate
    eyas_root.propagate = True
    caplog.set_level(logging.WARNING, logger="eyas.engine.strategies.spo")
    try:
        cfg = SpoConfig(rounds=1, num_pairs=2, votes_per_pair=4)
        backend = ScriptedMockBackend(responder=_spo_judge_responder)
        ctx, strategy = _spo_ctx(cfg, backend)
        state = strategy.init([Prompt("Answer now."), Prompt("Answer later.")])
        assert state.candidates == [Prompt("Answer now.")]  # behavior unchanged
        assert state.phase == "seed"
        warnings = [r for r in caplog.records if "ignored" in r.getMessage()]
        assert warnings, "no WARNING named the dropped seed"
        msg = warnings[0].getMessage()
        assert "single-incumbent" in msg  # the rationale rides the message
    finally:
        eyas_root.propagate = old_propagate


def test_spo_init_stays_silent_on_single_seed(caplog):
    eyas_root = logging.getLogger("eyas")
    old_propagate = eyas_root.propagate
    eyas_root.propagate = True
    caplog.set_level(logging.WARNING, logger="eyas.engine.strategies.spo")
    try:
        cfg = SpoConfig(rounds=1, num_pairs=2, votes_per_pair=4)
        backend = ScriptedMockBackend(responder=_spo_judge_responder)
        ctx, strategy = _spo_ctx(cfg, backend)
        strategy.init([Prompt("Answer now.")])
        assert not [r for r in caplog.records if "ignored" in r.getMessage()]
    finally:
        eyas_root.propagate = old_propagate


def test_spo_pair_schedule_shuffled_once_without_replacement():
    """The round slices replay the ONE-SHUFFLE schedule exactly (ids shuffled once by
    derived_rng(), consumed without replacement); with an 8-sample split the first
    8 pairs contain no repeat, and the vote questions follow the replay."""
    cfg = SpoConfig(rounds=3, num_pairs=4, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_wide_responder())
    ctx, strategy = _spo_ctx(cfg, backend, task=_SpoWideToy())
    OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    expected = _spo_pair_schedule(
        cfg.num_pairs * cfg.rounds, seed=42, ids=[f"r{i}" for i in range(8)], num_pairs=4
    )
    assert len(set(expected[:8])) == 8  # without replacement until full coverage
    votes = _spo_votes(backend)
    groups = [votes[g : g + 4] for g in range(0, len(votes), 4)]
    for g_idx, group in enumerate(groups):
        for text in group:
            assert f"Question: {expected[g_idx]}" in text


def test_spo_pair_schedule_wraps_with_reshuffle_after_full_coverage():
    """Wrap proof on a 3-sample split with num_pairs=2: rounds consume the
    one-shuffle ids without replacement, and the second round crosses the
    coverage boundary by APPENDING a reshuffled cycle — the window at the
    boundary is provably not a frozen rotation of the first round's pair. An
    identical second run (fresh backend, same seed) replays the whole wrapped
    schedule byte-identically; the reshuffled sequence is pinned against a
    local one-shuffle replay over the same ids."""
    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    runs = []
    for _ in range(2):
        backend = ScriptedMockBackend(responder=_spo_q3_responder())
        ctx, strategy = _spo_ctx(cfg, backend, task=_SpoQ3Toy())
        OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
        runs.append(_spo_votes(backend))
    assert runs[0] == runs[1]  # deterministic from the rng across runs
    votes = runs[0]
    ids = [f"q{i}" for i in range(1, 4)]
    expected = _spo_pair_schedule(cfg.num_pairs * cfg.rounds, seed=42, ids=ids, num_pairs=2)
    # without replacement over the FIRST coverage cycle…
    assert len(set(expected[:3])) == 3
    assert [s.id for s in _SpoQ3Toy().load_split("train")] == ids  # schedule base order
    # …and the wrap NEVER degrades into a frozen rotation of the last pair
    # (a num_pairs=2 rotation would replay the window [e0,e1] at [2:4])
    rotation = [expected[0], expected[1], expected[0], expected[1]]
    assert expected[2:] != rotation[2:]
    groups = [votes[g : g + 4] for g in range(0, len(votes), 4)]
    assert len(groups) == cfg.num_pairs * cfg.rounds
    for g_idx, group in enumerate(groups):
        for text in group:
            assert f"Question: {expected[g_idx]}" in text


def test_spo_judge_tie_votes_count_half_not_lost():
    """Unparsable replies count 0.5 for BOTH sums (the neutral-tie convention),
    never as a challenger loss: 2 tie + 2 clean challenger wins
    give wins=3.0 vs losses=1.0 → the challenger ADVANCES. Under
    tie-as-loss accounting it would be 2.0 vs 2.0 → no strict win, the seed
    stays incumbent, and the closing full-split pass would run on the SEED.
    The judge is content-driven: it picks the side whose answer is not the
    seed's constant 'wrong' — the ABBA layout decides which letter that is."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            a_line = next(
                line for line in user_text(request).splitlines() if line.startswith("(A)")
            )
            b_line = next(
                line for line in user_text(request).splitlines() if line.startswith("(B)")
            )
            if a_line == "(A) wrong" and b_line != "(B) wrong":
                return ChatResponse(text="B")
            if b_line == "(B) wrong" and a_line != "(A) wrong":
                return ChatResponse(text="A")
            return ChatResponse(text="maybe")  # both sides wrong -> neutral half-tie
        if _SPO_GEN_SYSTEM in sys_text:
            return ChatResponse(text=f"{OUT_START}Think step by step.{OUT_END}")
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    cfg = SpoConfig(rounds=1, num_pairs=1, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=responder)
    ctx, strategy = _spo_ctx(cfg, backend)
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    assert result.best is not None
    # the advanced challenger ran the closing full-split pass (and leads the
    # leaderboard at accuracy 1.0, vs the seed's 0.0)
    assert result.best.prompt_id != Prompt("Answer now.").id
    assert result.best.scores["accuracy"] == 1.0
    assert len(result.best.per_sample) == len(_LABELS)


def test_spo_state_serializes_phase_incumbent_cursor_and_schedule():
    """The phase machine's durable contract: phase, incumbent text, answered-cursor
    index, the per-round pair schedule and the optimize-call counter all survive
    the checkpoint.json JSON path."""
    state = SpoState(
        candidates=[Prompt("Answer now."), Prompt("Answer later."), Prompt("child")],
        round_idx=2,
        phase="propose",
        incumbent=Prompt("Answer later."),
        challenger=Prompt("child"),
        cursor=16,
        schedule=["q1", "q2", "q2", "q1"],
        gens_served=2,
        eval_cursor=14,
    )
    data = json.loads(json.dumps(state.to_json()))
    assert set(data) == {
        "candidates",
        "round_idx",
        "phase",
        "incumbent",
        "challenger",
        "cursor",
        "schedule",
        "gens_served",
        "eval_cursor",
    }
    restored = SpoState.from_json(data)
    assert [p.text for p in restored.candidates] == ["Answer now.", "Answer later.", "child"]
    assert restored.round_idx == 2
    assert restored.phase == "propose"
    assert restored.incumbent is not None and restored.incumbent.id == Prompt("Answer later.").id
    assert restored.challenger is not None and restored.challenger.id == Prompt("child").id
    assert restored.cursor == 16
    assert restored.schedule == ["q1", "q2", "q2", "q1"]
    assert restored.gens_served == 2
    assert restored.eval_cursor == 14


def test_spo_midphase_resume_from_voting_checkpoint_continues_exact_trajectory():
    """Mid-phase resume (after the items-evaluate round, BEFORE the vote): the fault
    lands at the first judge request after the round-1 items-eval checkpoint
    (phase=voting persisted, no vote served); the resumed run re-enters the voting
    phase and its request stream continues the pre-crash stream EXACTLY (identical
    tail — continuation of the same trajectory, zero completed work re-issued), and
    the final result matches the uninterrupted run.

    The resumed run resumes with the SAME record store the interrupted run used —
    the in-process mirror of a run-dir resume, where the run dir's
    ``cache/eyas.sqlite`` persists and archive-covered evaluate requests replay
    with zero engine traffic. A fresh store would re-issue those evaluate requests
    (correct, just not byte-exact), so exactness is asserted over the spo
    channels' streams (vote/optimize), which never consult the store.
    """
    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    seeds = ["Answer now.", "Answer later."]
    # durable before the fault: seed-eval (2) + round-1 optimize gen (1) +
    # items-eval (2 challenger requests; the incumbent's subset re-evaluation
    # is a record-store hit) = 5 served requests. Request 6 is the first ballot
    # of the 8-request vote batch — the fault kills the WHOLE batch (never
    # inside a served batch), strictly after the items checkpoint, at the head
    # of the voting phase.
    fault_at = 6

    class _FaultedMockBackend(ScriptedMockBackend):
        """Raises when the batch being asked WOULD serve request ``fault_at`` (the
        kill shape of test_resume._FaultedBackend, batch-granular by design)."""

        def __init__(self, responder, fault_at: int) -> None:
            super().__init__(responder=responder)
            self._fault_at = fault_at

        async def agenerate(
            self, requests: Sequence[ChatRequest], sampling: SamplingConfig
        ) -> list[ChatResponse]:
            if self.call_count + len(requests) >= self._fault_at:
                raise KeyboardInterrupt(f"synthetic crash before request {self._fault_at}")
            return await super().agenerate(requests, sampling)

    # uninterrupted reference (no shared store: a real run writes the
    # run-dir sqlite too, where a re-evaluation of a stored prompt costs zero
    # engine requests — the mock's store is irrelevant to the stream contract)
    ref_backend = ScriptedMockBackend(responder=_spo_judge_responder)
    ctx, strategy = _spo_ctx(cfg, ref_backend)
    full = OptimizerLoop(ctx).run(strategy, list(seeds))
    full_reqs = list(ref_backend.recorded_requests)

    # interrupted run: crashes before serving ANY vote of the round-1 vote batch
    backend = _FaultedMockBackend(_spo_judge_responder, fault_at)
    ctx, strategy = _spo_ctx(cfg, backend)
    snapshots: list[dict] = []
    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(strategy, list(seeds), on_round=snapshots.append)
    checkpoint = json.loads(json.dumps(snapshots[-1]))
    assert snapshots[-1]["round_idx"] == 2  # seed round + items-eval round checkpointed
    assert checkpoint["state"]["phase"] == "eval"  # vote pending — the mid-phase point
    assert checkpoint["state"]["cursor"] == 2  # round-1 consumed the one-shuffle order
    assert (
        checkpoint["state"]["challenger"]
        == "Think step by step.\nBe concise.\nChoices: [{choices}]"
    )
    order = checkpoint["state"]["schedule"]  # the seeded one-shuffle order
    assert sorted(order) == ["q1", "q2"]
    assert _spo_votes(backend) == []  # no vote was ever served before the crash
    pre_crash = list(backend.recorded_requests)
    assert pre_crash == full_reqs[: len(pre_crash)]  # prefix fidelity

    resumed_backend = ScriptedMockBackend(responder=_spo_judge_responder)
    ctx, strategy = _spo_ctx(cfg, resumed_backend)
    resumed = OptimizerLoop(ctx).run(strategy, list(seeds), resume=checkpoint)
    assert resumed.best is not None
    assert len(resumed.best.per_sample) == len(_LABELS)
    assert [r.n_requests for r in resumed.history] == [r.n_requests for r in full.history]
    assert resumed.total_cost.requests == full.total_cost.requests
    # EXACT re-derivation at the fault: the re-vote ballot texts and the optimize
    # requests of the resumed tail are BYTE-IDENTICAL to the full run's (the vote
    # window replays from the persisted schedule/cursor). The resumed run starts
    # AT the fault: every pre-crash request is archive-covered (evaluate) or
    # state-covered (the optimize reply whose output is the persisted challenger),
    # so the resumed stream issues nothing before the vote batch — the tail IS
    # the whole resumed stream, not a post-prefix slice of it.
    tail = resumed_backend.recorded_requests
    votes_resumed = [t for t, _ in tail if _SPO_SYSTEM in t]
    votes_full = [t for t, _ in full_reqs if _SPO_SYSTEM in t]
    assert votes_resumed == votes_full
    gens_resumed = [t for t, _ in tail if _SPO_GEN_SYSTEM in t]
    # the round-1 optimize reply's output (the challenger) is checkpointed in
    # state; only generations AFTER the fault re-appear in the resumed stream
    gens_full = [t for t, _ in full_reqs if _SPO_GEN_SYSTEM in t]
    assert gens_resumed == gens_full[1:]
    # every in-flight interrupted request replays (supersequence, harness rule)
    it = iter(tail)
    missing = [fp for fp in full_reqs[len(pre_crash) :] if not any(fp == seen for seen in it)]
    assert not missing


def test_spo_vote_fails_fast_when_record_store_does_not_cover_the_window():
    """Fail-fast discipline (the tripwire sits on the
    DURABLE record store):
    voting must NEVER substitute a fabricated empty answer for an id the
    durable store does not cover. The honest dip — the incumbent scoring
    BELOW its historical peak on the live window — is legal
    (test_spo_incumbent_dip_below_archive_peak_still_votes pins the vote
    still builds), so only a genuine store bug can now starve the vote. The
    probe subclass simulates one: once the round's items round has evaluated
    the window (phase=eval with eval_cursor == cursor — the vote step), it
    pops the incumbent's stored answers for that window — the lost-write
    shape. The next vote must raise, not fabricate."""

    class _StoreHoleSpo(SpoStrategy):
        """Holes the incumbent's store entries for the live window exactly
        once (the public RecordStore API has no delete — a lost write is
        exactly the store bug this probe simulates)."""

        def __init__(self, config: Any, ctx: Any) -> None:
            super().__init__(config, ctx)
            self.holed = False

        def step(self, state: OptimizerState, view: EvalView) -> Decision:
            if (
                not self.holed
                and state.phase == "eval"
                and state.eval_cursor == state.cursor
                and state.incumbent is not None
            ):
                self.holed = True
                data = self.ctx.evaluator.store._data  # noqa: SLF001 — deliberate probe corruption
                window = state.schedule[state.cursor - self.config.num_pairs : state.cursor]
                for sample_id in window:
                    data.pop((state.incumbent.id, sample_id, self.ctx.eval_split), None)
            return super().step(state, view)

    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=_spo_voter(False))
    ctx = build_run_context(
        _StoreHoleSpo,
        cfg,
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=42,
    )
    with pytest.raises(RuntimeError, match="vote window not covered"):
        OptimizerLoop(ctx).run(_StoreHoleSpo(cfg, ctx), ["Answer now.", "Answer later."])
    assert [r for r in backend.recorded_requests if _SPO_SYSTEM in r[0]] == []


def test_spo_incumbent_dip_below_archive_peak_still_votes():
    """Regression: the round's VOTE ANSWERS must come from the durable record
    store, never from the archive-best record. Scenario — descending incumbent
    scores across two windows: round 1's window holds the incumbent's easy samples (subset
    record 1.0 DISPLACES the seed's 0.5 full-split record → archive best =
    round-1's window), round 2's window holds its hard samples (0.0 dip → the
    best record never moves and still shows round-1's ids). The round-2 vote
    must still read round-2's ids: the items round just answered BOTH prompts
    on that window into the store, so the vote must build cleanly."""
    base = [f"s{i}" for i in range(4)]
    order = _spo_pair_schedule(len(base), seed=42, ids=base, num_pairs=2)
    window1, window2 = set(order[:2]), set(order[2:4])
    assert window1.isdisjoint(window2)  # without-replacement consumption

    class _DipToy(ToyMCQ):
        """4-sample split; the seed prompt answers the window-1 ids correctly
        (that window is the incumbent's archive peak) and misses the rest."""

        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id=sid,
                    inputs={"question": sid, "image": f"{sid}.png"},
                    references={"label": f"label-{sid}"},
                )
                for sid in base
            ]

    labels = {sid: f"label-{sid}" for sid in base}
    k = {"n": 0}

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if sys_text.startswith(_SPO_SYSTEM):
            return ChatResponse(text="A")  # constant tie: the seed stays incumbent
        if _SPO_GEN_SYSTEM in sys_text:
            k["n"] += 1
            return ChatResponse(text=f"{OUT_START}Variant {k['n']}: think step by step.{OUT_END}")
        question = user_text(request)
        # the seed answers its peak-window ids correctly; the step-by-step
        # challenger echoes the question (scores 0.0 everywhere — a challenger
        # the tie-vote rejects, which keeps the dipping seed incumbent)
        return ChatResponse(
            text=labels[question]
            if question in window1
            else (f"echo-{question}" if "step by step" in sys_text.lower() else "wrong")
        )

    cfg = SpoConfig(rounds=2, num_pairs=2, votes_per_pair=4)
    backend = ScriptedMockBackend(responder=responder)
    ctx, strategy = _spo_ctx(cfg, backend, task=_DipToy())
    result = OptimizerLoop(ctx).run(strategy, ["Answer now.", "Answer later."])
    votes = _spo_votes(backend)
    assert len(votes) == len(base) * cfg.votes_per_pair  # round 2 voted cleanly
    # the DIP actually happened: the incumbent's archive best is the round-1
    # record over window-1 ids only (accuracy 1.0 displaced the seed's 0.5
    # full-split record) — precisely the record the round-2 vote window
    # (window-2 ids) is absent from
    seed_best = next(e for e in result.ranked if e.prompt_id == Prompt("Answer now.").id)
    assert {o.sample_id for o in seed_best.per_sample} == window1
    assert seed_best.scores["accuracy"] == 1.0
    # the round-2 ballots carry the durable answers of THIS window: the seed's
    # stored "wrong" on its hard samples vs the challenger's echoes
    groups = [votes[len(window1) * 4 + i * 4 : len(window1) * 4 + i * 4 + 4] for i in range(2)]
    for g_idx, group in enumerate(groups):
        sid = order[2 + g_idx]
        for text in group:
            assert f"Question: {sid}" in text
        sides = [_ab_sides(text) for text in group]
        assert "wrong" in sides[0][0] + sides[0][1]  # the seed's stored answer voted
        assert any(f"echo-{sid}" in a + b for a, b in sides)  # the challenger's too


def test_spo_items_decision_evaluates_exactly_the_subsets():
    """Loop primitive: Decision(items=...) evaluates EXACTLY those (prompt, sample)
    work items — the archive's per-sample outcomes are exactly the requested subsets,
    and the RoundRecord bills the subset sizes, not the full split."""

    class ItemsProbeStrategy(RawStrategy):
        def step(self, state: OptimizerState, view: EvalView) -> Decision:
            assert isinstance(state, PromptListState)
            seed = state.candidates[0]
            if not view.evaluated(seed):
                return Decision("evaluate", items=(WorkItem(seed, ("q1",)),))
            return Decision("finish", best_id=seed.id)

    result = _run(
        ItemsProbeStrategy,
        RawConfig(),
        ToyMCQ(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        ["Classify the animal."],
    )
    assert result.history[0].n_requests == 1  # the subset size, not the 2-sample split
    assert [o.sample_id for e in result.ranked for o in e.per_sample] == ["q1"]


def test_archive_score_tie_prefers_larger_evidence_set():
    """Archive.add contract (SPO closing-pass rule): strictly higher score wins; on a score tie
    the LARGER per-sample evidence set wins — a subset round must not lock a
    prompt's reported entry below its full-split record (spo's closing pass
    overwrites its vote-subset record exactly this way; without the rule the
    final Evaluation would report subset evidence at the full-split score).
    Equal score AND equal size keep the first writer (deterministic, no
    insertion-order drift under resume)."""

    def _eval(prompt: Prompt, ids: tuple[str, ...], acc: float, round_idx: int) -> Evaluation:
        return Evaluation(
            prompt_id=prompt.id,
            split="train",
            eval_model="absolute",
            scores={"accuracy": acc},
            per_sample=tuple(
                SampleOutcome(sample_id=s, prediction="p", scores={"accuracy": acc}) for s in ids
            ),
            cost=Cost(requests=len(ids), wall_seconds=0.0),
            round_idx=round_idx,
        )

    def _score_of(e: Evaluation) -> float:
        return float(e.scores["accuracy"])

    archive = Archive(_score_of)
    p = Prompt("tied prompt")
    archive.add(_eval(p, ("q1",), 1.0, 0))
    archive.add(_eval(p, ("q1", "q2"), 1.0, 1))  # tie, larger evidence -> wins
    assert len(archive.best(p.id).per_sample) == 2  # type: ignore[union-attr]
    archive.add(_eval(p, ("q1",), 1.0, 2))  # tie, smaller evidence -> keeps best
    assert len(archive.best(p.id).per_sample) == 2  # type: ignore[union-attr]
    archive.add(_eval(p, ("q1",), 0.5, 3))  # lower score -> never displaces
    assert len(archive.best(p.id).per_sample) == 2  # type: ignore[union-attr]
    best = archive.best(p.id)
    q = Prompt("first writer")
    archive.add(_eval(q, ("q1", "q2"), 1.0, 4))
    archive.add(_eval(q, ("q1", "q2"), 1.0, 5))  # equal score + size -> first writer stays
    assert archive.best(q.id) is not None
    assert archive.best(q.id).round_idx == 4
    assert best is not None and best.round_idx == 1


def test_takeover_protocol_all_allocators_accept_and_bandit_invalidates():
    """Budget protocol: Flat/EvenSplit accept ``takeover`` as a no-op; BanditBudget
    invalidates its arm state (a strategy-owned round bypasses propose_load, so the
    allocator must not carry arms/counts over a takeover)."""
    from eyas.engine.budget import EvenSplitBudget, FlatBudget

    FlatBudget().takeover(0)  # no-op contract
    EvenSplitBudget(4).takeover(0)

    arms = [Prompt(f"arm {i}") for i in range(3)]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1)
    budget.propose_load(0, None, arms, ["s0", "s1", "s2"])
    assert budget.get_state() is not None
    budget.takeover(1)
    assert budget.get_state() is None  # arms/counts/round cleared
    # the next propose_load re-baselines on the freshly proposed arm set
    plan = budget.propose_load(2, None, arms, ["s0", "s1", "s2"])
    assert len(plan.items) == 3


def test_bandit_observe_after_takeover_is_a_documented_noop():
    """A documented no-op: the OptimizerLoop calls ``observe`` unconditionally
    after EVERY evaluate round — including a strategy-owned ``Decision(items=...)``
    round whose ``takeover`` already unbound the arms (the spo×bandit API shape).
    After such a takeover ``observe`` is a documented no-op: no raise, zero state
    corruption (arms stay unbound, counts untouched), and the next ``propose_load``
    re-baselines UCB on the fresh arm set."""
    arms = [Prompt(f"arm {i}") for i in range(2)]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1)
    budget.propose_load(0, None, arms, ["s0", "s1"])
    evals = [Evaluation(a.id, "train", "absolute", {"accuracy": 0.5}, (), Cost()) for a in arms]
    budget.observe(evals, [0.5, 0.5])
    assert budget._counts == [1, 1]  # noqa: SLF001
    budget.takeover(1)
    # the loop's unconditional observe after the strategy-owned round: no raise
    assert budget.observe(evals, [0.9, 0.9]) == [e.prompt_id for e in evals]
    assert budget._arms is None  # noqa: SLF001  # still unbound — no silent re-binding
    assert budget.get_state() is None  # and nothing was credited
    assert budget._round == 0  # noqa: SLF001  # the takeover cursor never advanced
    # the next propose_load re-baselines: fresh arms, zero counts, window at 0
    plan = budget.propose_load(2, None, arms, ["s0", "s1"])
    assert budget._counts == [0, 0]  # noqa: SLF001
    assert all(item.sample_ids == ("s0",) for item in plan.items)


def test_bandit_window_exhaustion_raises_with_remediation():
    """Config is law: "sample
    quantities belong to the configurator — runtime auto-shrinking IS a silent
    downgrade." With the official ``8×32`` caps over the 248-sample bench splits
    the 8th pull is a legitimate PARTIAL window (``248 − 7×32 = 24`` — data-end);
    a NINTH pull starts at or
    beyond the split end and raises ``ValueError``. The message carries the
    arithmetic (pull, start, n, planned) AND both remediation values computed
    for the configurator — ``samples_per_eval ≤ n // eval_rounds`` (248 // 8 =
    31, the value the shipped 248-split cells now pin) and ``eval_rounds ≤
    n // samples_per_eval`` (248 // 32 = 7). The framework refuses to guess."""
    n, spe, rounds = 248, 32, 8
    arms = [Prompt(f"arm {i}") for i in range(4)]
    ids = [f"s{i:03d}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
    budget.propose_load(0, None, arms, ids)  # a first arm set takes pull 1 ...
    budget.takeover(0)  # ... a strategy-owned round re-baselines before the scoring pulls
    served: list[tuple[str, ...]] = []
    for pull in range(1, rounds + 1):  # the 8 scoring pulls; the 8th is PARTIAL at 224
        plan = budget.propose_load(pull, None, arms, ids)
        window = tuple(ids[(pull - 1) * spe : pull * spe])
        served.append(window)
        for item in plan.items:
            assert item.sample_ids == window
    assert len(served[-1]) == n - 7 * spe == 24  # last-partial-window serves (legal tail)
    assert len(set(served)) == rounds  # one window per pull — the silent repeat stays dead
    with pytest.raises(ValueError, match=r"eval_rounds × samples_per_eval") as exc:
        budget.propose_load(rounds + 1, None, arms, ids)
    msg = str(exc.value)
    assert "samples_per_eval ≤ 31" in msg  # fix A: n // eval_rounds = 248 // 8
    assert "eval_rounds ≤ 7" in msg  # fix B: n // samples_per_eval = 248 // 32
    # the raise is state-clean: the round cursor never advanced past the 8 pulls
    assert budget._round == rounds  # noqa: SLF001


def test_bandit_exact_split_boundary_raises_only_beyond_the_last_window():
    """n = eval_rounds × samples_per_eval exactly: every pull carries a full
    window; the pull BEYOND the horizon raises with both remediation values
    (at the exact boundary both fixes name the current values — the plan is
    already maximal)."""
    arms = [Prompt("a"), Prompt("b")]
    ids = ["s0", "s1", "s2", "s3"]
    budget = BanditBudget(eval_rounds=2, samples_per_eval=2)
    for pull, start in ((1, 0), (2, 2)):  # two full windows cover the split exactly
        plan = budget.propose_load(pull, None, arms, ids)
        assert all(item.sample_ids == tuple(ids[start : start + 2]) for item in plan.items)
    with pytest.raises(ValueError, match=r"2 × 2") as exc:
        budget.propose_load(3, None, arms, ids)
    assert "samples_per_eval ≤ 2" in str(exc.value)  # 4 // 2
    assert "eval_rounds ≤ 2" in str(exc.value)  # 4 // 2


def test_bandit_degenerate_split_raise_names_impossible_fixes():
    """n < samples_per_eval (7 samples, 8×32): pull 1 serves the 7-sample head
    (a partial window — the split end is reached inside the first pull), pull 2
    starts at 32 ≥ 7 and raises. Both remediation values compute to 0 — neither
    cap can be repaired into a fit — and the message says so explicitly: the
    only fix is more selection data (or a different split)."""
    n, spe, rounds = 7, 32, 8
    arms = [Prompt("a"), Prompt("b")]
    ids = [f"s{i}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe)
    plan = budget.propose_load(1, None, arms, ids)
    for item in plan.items:
        assert item.sample_ids == tuple(ids)  # partial head serve
    with pytest.raises(ValueError, match=r"window exhaustion") as exc:
        budget.propose_load(2, None, arms, ids)
    msg = str(exc.value)
    assert "samples_per_eval ≤ 0" in msg  # 7 // 8 — impossible
    assert "eval_rounds ≤ 0" in msg  # 7 // 32 — impossible
    assert "cannot host the plan at any cap" in msg  # the impossible-fix hint


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"eval_rounds": 0}, r"eval_rounds must be >= 1, got 0"),
        ({"samples_per_eval": 0}, r"samples_per_eval must be >= 1, got 0"),
        ({"samples_per_eval": -1}, r"samples_per_eval must be >= 1, got -1"),
        ({"c": -0.1}, r"c must be >= 0, got -0.1"),
        ({"keep_top": 0.0}, r"keep_top must be in \(0, 1\], got 0.0"),
        ({"keep_top": -0.5}, r"keep_top must be in \(0, 1\], got -0.5"),
        ({"keep_top": 1.5}, r"keep_top must be in \(0, 1\], got 1.5"),
    ],
)
def test_bandit_ctor_rejects_invalid_params(kwargs: dict, match: str) -> None:
    """Every numeric ctor param is range-validated at construction (fail fast,
    naming the offending value): a zero/negative window or round count, a
    negative UCB exploration coefficient, or a keep_top outside the (0, 1]
    arm-fraction range must die at assembly, not silently distort the plan."""
    with pytest.raises(ValueError, match=match):
        BanditBudget(**kwargs)


def test_bandit_ctor_accepts_boundary_values() -> None:
    budget = BanditBudget(eval_rounds=1, samples_per_eval=1, c=0.0, keep_top=1.0)
    assert (budget.eval_rounds, budget.samples_per_eval, budget.c, budget.keep_top) == (
        1,
        1,
        0.0,
        1.0,
    )


@pytest.mark.parametrize("n", [1, 2, 7, 24, 64, 74, 110, 248, 256, 296])
@pytest.mark.parametrize("rounds", [1, 3, 8])
@pytest.mark.parametrize("spe", [1, 8, 32])
def test_bandit_window_plan_grid_serves_or_raises(n, rounds, spe):
    """Config is law: the configured plan (``rounds`` pulls of
    ``spe``) executes exactly — a fitting plan serves the consecutive windows
    verbatim (the last one partial at the data end is legal — consecutive
    windows may end on a partial final window); a plan
    whose pull starts at or beyond the split end raises AT that pull, carrying
    both computed remediation values, and the raise is final (later pulls raise
    too — no adaptation ever kicks in)."""
    arms = [Prompt("a"), Prompt("b")]
    ids = [f"s{i}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
    raised = False
    for k in range(rounds):
        if k * spe >= n:
            with pytest.raises(ValueError, match="window exhaustion") as exc:
                budget.propose_load(k, None, arms, ids)
            msg = str(exc.value)
            assert f"samples_per_eval ≤ {n // rounds}" in msg
            assert f"eval_rounds ≤ {n // spe}" in msg
            assert budget._round == k  # noqa: SLF001  # state-clean at the raise
            # the raise is final: exhaustion is a config fact, not a transient
            with pytest.raises(ValueError, match="window exhaustion"):
                budget.propose_load(k + 1, None, arms, ids)
            assert budget._round == k  # noqa: SLF001
            raised = True
            break
        plan = budget.propose_load(k, None, arms, ids)
        window = tuple(ids[k * spe : min((k + 1) * spe, n)])
        assert window  # a served window is never empty
        for item in plan.items:
            assert item.sample_ids == window
    if not raised:
        assert budget._round == rounds  # noqa: SLF001  # the plan completed exactly


def test_bandit_normal_path_serves_the_expected_windows():
    """Normal-path invariance (guard kept by the shipped contract): the served windows are
    byte-identical to the documented shape — consecutive windows of ``spe`` from
    the plain ``samples[start : start + spe]`` slice. The cub 261 / pets 296
    bench cells (8×32 = 256 ≤ n) serve window-for-window unchanged; the
    hypothetical 8×32 over the 248 splits completes too, ending on the legal
    24-sample partial tail (pull 8 starts at 224 < 248, a legal data-end tail)."""
    for n in (248, 256, 261, 296):
        rounds, spe = 8, 32
        assert (rounds - 1) * spe < n  # no pull of the plan starts at/beyond the end
        arms = [Prompt(f"arm {i}") for i in range(3)]
        ids = [f"s{i:04d}" for i in range(n)]
        budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
        for pull in range(1, rounds + 1):
            plan = budget.propose_load(pull, None, arms, ids)
            window = tuple(ids[(pull - 1) * spe : pull * spe])  # window arithmetic, verbatim
            assert window  # every served window is nonempty
            for item in plan.items:
                assert item.sample_ids == window
        assert budget._round == rounds  # noqa: SLF001


def test_bandit_exact_fit_serves_full_windows():
    """8×31 = 248 exactly (the shape the 12 sroie/textvqa bench cells now ship):
    every pull carries a FULL 31-sample window, the eighth ends exactly at the
    split end, and no pull arms exhaustion."""
    n, spe, rounds = 248, 31, 8
    arms = [Prompt("a"), Prompt("b")]
    ids = [f"s{i:03d}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
    for pull in range(1, rounds + 1):
        plan = budget.propose_load(pull, None, arms, ids)
        window = tuple(ids[(pull - 1) * spe : pull * spe])
        assert len(window) == spe
        for item in plan.items:
            assert item.sample_ids == window


# --- Structural gen-request guard ---
#
# The real Qwen3.5 HF chat template RAISES `No user query found in messages` on a
# SYSTEM-ONLY ChatRequest — a system-only generation request crashes the very
# first real-engine run because ScriptedMockBackend never renders templates, so
# fast-tier mock tests cannot see it. House contract:
# system carries the GEN_MARKER plumbing (+ optional directive), the model-facing
# task payload rides a USER message — all four peer generative strategies (ape /
# evoprompt / opro / spo) follow it. This is the standing structure-level
# gate: every generative strategy runs its full mock loop and
# EVERY generation request (GEN_MARKER in its system text) must carry a non-empty
# user message. Cheap, structure-level, zero template rendering. (The real-template
# byte smoke exists only as the SLOW-tier tests/test_renderer_goldens.py, whose
# case matrix never covers a strategy gen request — that gap is why this gate is
# structure-level and lives in the fast tier.)


def _role_capture(captured: list[ChatRequest]) -> ScriptedMockBackend:
    """Mock twin of _gen_mock that ALSO records the ChatRequest objects, so the
    guard can assert message roles (recorded_requests flattens them away)."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        captured.append(request)
        sys_text = "".join(
            p.text for m in request.messages if m.role == "system" for p in m.content
        )
        if GEN_MARKER in sys_text:
            return ChatResponse(
                text=f"{OUT_START}[Classify the animal carefully and think step by step.]{OUT_END}"
            )
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    return ScriptedMockBackend(responder=responder)


def test_every_generation_request_carries_a_user_message():
    generative: list[tuple[str, type, Any, BanditBudget | None, list[str]]] = [
        (
            "ape",
            ApeStrategy,
            ApeConfig(num_candidates=2, demos_per_prompt=1),
            None,
            ["Classify the animal."],
        ),
        (
            "evoprompt",
            EvopromptStrategy,
            EvopromptConfig(popsize=2, rounds=1, mode="de"),
            None,
            ["Classify the animal one-shot.", "Classify the animal two-shot."],
        ),
        (
            "opro",
            OproStrategy,
            OproConfig(rounds=1, num_generated=2),
            None,
            ["Classify the animal."],
        ),
        (
            "protegi",
            ProtegiStrategy,
            ProtegiConfig(rounds=1, beam_size=2, n_gradients=1),
            BanditBudget(eval_rounds=2, samples_per_eval=1),
            ["Classify the animal."],
        ),
        (
            "spo",
            SpoStrategy,
            SpoConfig(rounds=1, num_pairs=2, votes_per_pair=4),
            None,
            ["Answer now.", "Answer later."],
        ),
    ]
    for name, cls, config, budget, seeds in generative:
        captured: list[ChatRequest] = []
        ctx = build_run_context(
            cls,
            config,
            ToyMCQ(),
            IdentityProtocol(),
            _role_capture(captured),
            AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
            budget=budget,
            seed=42,
        )
        OptimizerLoop(ctx).run(cls(config, ctx), seeds)
        gen_requests = [
            req
            for req in captured
            if any(
                m.role == "system" and GEN_MARKER in "".join(p.text for p in m.content)
                for m in req.messages
            )
        ]
        assert gen_requests, f"{name} issued no generation request through the mock loop"
        for req in gen_requests:
            has_user = any(
                m.role == "user"
                and any(isinstance(p, TextPart) and p.text.strip() for p in m.content)
                for m in req.messages
            )
            assert has_user, (
                f"{name}: system-only generation request — the real HF chat templates "
                f"raise 'No user query found in messages'"
            )


def test_eval_model_config_name_resolves_through_registry_fail_fast() -> None:
    """Rule 3: `_build_eval_model` resolves the config name via the eval_model
    registry — an unknown name fails fast at assembly, never silently degrading
    to the absolute model; a registered non-built-in name reports the explicit
    construction gap instead of impersonating a built-in."""
    from eyas.engine.evaluation import AbsoluteEvalModel, CompareEvalModel
    from eyas.engine.runtime import _build_eval_model

    toy = ToyMCQ()
    assert isinstance(_build_eval_model({}, toy), AbsoluteEvalModel)
    assert isinstance(_build_eval_model({"name": "compare"}, toy), CompareEvalModel)
    with pytest.raises(KeyError, match="unregistered eval_model"):
        _build_eval_model({"name": "quantum"}, toy)

    class CustomEvalModel:  # registered third-party, heterogeneous constructor
        pass

    with monkeypatched_registry(CustomEvalModel):
        with pytest.raises(ValueError, match="cannot construct it from config"):
            _build_eval_model({"name": "custom_em"}, toy)


def monkeypatched_registry(cls: type):  # tiny ctx manager to keep the test isolated
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        from eyas.engine import registry as engine_registry

        bucket = engine_registry._REGISTRY["eval_model"]
        bucket["custom_em"] = cls
        try:
            yield
        finally:
            del bucket["custom_em"]

    return _ctx()


def test_even_split_budget_divides_the_total_budget_evenly() -> None:
    """EvenSplitBudget contract (protegi bf semantics): the
    configured total sample budget is split across candidates as
    ``per = max(1, min(|split|, ceil(total / n)))``, every candidate gets the
    same subset, the split caps the per-candidate window, and the allocator is
    stateless (rank None / takeover no-op / get_state None). Pins the registered
    built-in's core arithmetic — the shape no shipped config selects today."""
    from eyas.engine.budget import EvenSplitBudget

    prompts = tuple(Prompt(f"p{i}") for i in range(3))
    sample_ids = tuple(f"s{i}" for i in range(20))

    plan = EvenSplitBudget(total_sample_budget=9).propose_load(0, None, prompts, sample_ids)  # type: ignore[arg-type]
    assert len(plan.items) == 3
    assert all(
        item.sample_ids == ("s0", "s1", "s2") for item in plan.items
    )  # ceil(9/3)=3, shared head slice

    capped = EvenSplitBudget(total_sample_budget=100).propose_load(0, None, prompts, sample_ids)  # type: ignore[arg-type]
    assert len(capped.items[0].sample_ids) == 20  # capped by |split|

    floor = EvenSplitBudget(total_sample_budget=1).propose_load(0, None, prompts, sample_ids)  # type: ignore[arg-type]
    assert all(len(item.sample_ids) == 1 for item in floor.items)

    budget = EvenSplitBudget(total_sample_budget=9)
    assert budget.rank() is None
    assert budget.takeover(0) is None
    assert budget.get_state() is None
    budget.set_state(None)

    with pytest.raises(ValueError, match="must be >= 1"):
        EvenSplitBudget(total_sample_budget=0)  # fail fast: the budget must be >= 1


# --- shared evaluate-then-finish step skeleton (cot/raw) ---


def _accuracy_of(evaluation: Evaluation) -> float:
    return evaluation.scores.get("accuracy", 0.0)


@pytest.fixture
def make_view():
    """A real EvalView over an Archive: ``evaluated=True`` folds in one Evaluation
    for the prompt, ``evaluated=False`` leaves the archive empty — so
    ``view.evaluated`` answers from the genuine archive keying, not a stub."""

    def _make(prompt: Prompt, *, evaluated: bool) -> EvalView:
        archive = Archive(_accuracy_of)
        if evaluated:
            archive.add(Evaluation(prompt.id, "train", "absolute", {"accuracy": 1.0}, (), Cost()))
        return EvalView(archive, _accuracy_of)

    return _make


@pytest.mark.parametrize(
    "strategy_cls,config",
    [(CotStrategy, CotConfig()), (RawStrategy, RawConfig())],
    ids=["cot", "raw"],
)
def test_step_evaluates_then_finishes(strategy_cls, config, make_view) -> None:
    """Degenerate single-candidate contract shared by cot/raw: the first step asks
    to evaluate the pool's only candidate, the next step finishes on it."""
    ctx = build_run_context(
        strategy_cls,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        _mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    strategy = strategy_cls(config, ctx)
    state = strategy.init([Prompt("p0")])
    prompt = state.candidates[0]
    first = strategy.step(state, make_view(prompt, evaluated=False))
    assert first.action == "evaluate"
    assert first.prompts == (prompt,)
    second = strategy.step(state, make_view(prompt, evaluated=True))
    assert second.action == "finish"
    assert second.best_id == prompt.id
