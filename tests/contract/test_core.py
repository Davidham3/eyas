"""Contract tests (core): hill_climb end to end, compare-only assembly rejection,
registry surface, planner gate, raw/cot seed rules, durable record store,
per-prompt billing, judge context, gen-request guard, step skeleton.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    SamplingConfig,
    TextPart,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.budget import BanditBudget, LoadPlan, WorkItem
from eyas.engine.capability import ResourceProfile
from eyas.engine.domain import (
    Archive,
    Cost,
    Evaluation,
    OptimizerState,
    Prompt,
    PromptListState,
    SampleOutcome,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
    CompareEvalModel,
    Evaluator,
    SqliteRecordStore,
)
from eyas.engine.genio import (
    GEN_MARKER,
    OUT_END,
    OUT_START,
)
from eyas.engine.planner import Rejected, RequestPlanner
from eyas.engine.registry import get as engine_get
from eyas.engine.registry import register_strategy
from eyas.engine.registry import registered as engine_registered
from eyas.engine.strategies.ape import ApeConfig, ApeStrategy
from eyas.engine.strategies.cot import CotConfig, CotStrategy
from eyas.engine.strategies.evoprompt import EvopromptConfig, EvopromptStrategy
from eyas.engine.strategies.hill_climb import HillClimbConfig, HillClimbStrategy
from eyas.engine.strategies.opro import OproConfig, OproStrategy
from eyas.engine.strategies.protegi import ProtegiConfig, ProtegiStrategy
from eyas.engine.strategies.raw import RawConfig, RawStrategy
from eyas.engine.strategies.spo import (
    SpoConfig,
    SpoStrategy,
)
from eyas.engine.strategy import Decision, EvalView, OptimizerLoop, build_run_context
from eyas.tasks.base import Sample
from eyas.tasks.protocol import IdentityProtocol
from tests.contract.helpers import (
    _LABELS,
    CompareOnlyConfig,
    CompareOnlyStrategy,
    ToyCompare,
    ToyMCQ,
    _cot_strategy,
    _mock,
    _run,
)


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
