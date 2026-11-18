"""Planner chokepoint made structural + judging doctrine + best_id divergence.

Pins the planner/judging contract rulings:

1. ``PlannedBackend`` (planner.py): paths that hold a *backend slot* (a judge
   via JudgeContext, pairwise games via CompareEvalModel) physically cannot
   bypass the planner. Accepted requests forward UNCHANGED; a rejected request
   gets an empty response with ``OVER_WINDOW_FINISH`` and the rest of the
   batch is unaffected (the chokepoint is structure, not prose);
2. ``build_run_context`` arms the over-window guard from the window the
   backend declares;
3. judging-channel doctrine: candidate generation and evaluation
   rendering take the deployment layer (seed/thinking ride through); judging
   is deliberately PINNED greedy — two runs whose deployment sampling differs
   produce byte-identical judging requests;
4. a finish decision declaring a champion different from the archive-score
   leader emits a WARNING (the ledger rule is unchanged: archive ranking wins;
   the declared champion is observability only).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest
from conftest import full_split_plan
from test_contract import ToyMCQ
from test_judge_cost import CostToyMCQ

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ChatResponse,
    ImagePart,
    InferenceBackend,
    SamplingConfig,
    VideoPart,
    system_message,
    user_message,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.backends.scheduling import run_sync
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.capability import BackendLimits, ResourceProfile
from eyas.engine.domain import OptimizerState, Prompt, PromptListState, rank_by_score
from eyas.engine.evaluation import AbsoluteEvalModel, Evaluator
from eyas.engine.planner import PlannedBackend, PlannedRequest, Rejected, RequestPlanner
from eyas.engine.registry import get as engine_get
from eyas.engine.registry import register_strategy
from eyas.engine.strategies.raw import RawConfig, RawStrategy
from eyas.engine.strategy import Decision, EvalView, OptimizerLoop, build_run_context
from eyas.judges.mcq_judge import McqLLMJudge
from eyas.tasks.protocol import IdentityProtocol


class _Recorder(InferenceBackend):
    """Inner-backend stand-in capturing every (request, sampling) pair it sees."""

    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[tuple[ChatRequest, SamplingConfig]] = []

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        self.seen.extend((r, sampling) for r in requests)
        return [ChatResponse(text=f"r{len(self.seen)}") for _ in requests]


# ---------------------------------------------------------------------------
# 1. PlannedBackend: the chokepoint as structure
# ---------------------------------------------------------------------------


async def test_planned_backend_forwards_accepted_requests_unchanged() -> None:
    inner = _Recorder()
    inner.context_window = 1234  # declared window must pass THROUGH the adapter
    wrapped = PlannedBackend(inner, RequestPlanner(ResourceProfile(), context_window=10_000))
    req = ChatRequest(messages=(system_message("short"), user_message("hello")))
    responses = await wrapped.agenerate([req], SamplingConfig())
    assert responses[0].text == "r1"
    # the accepted request reaches the inner backend as the SAME object —
    # wrapping never copies or rewrites a request
    assert inner.seen[0][0] is req
    assert wrapped.context_window == 1234


async def test_planned_backend_rejected_slot_is_typed_and_rest_flows() -> None:
    inner = _Recorder()
    wrapped = PlannedBackend(inner, RequestPlanner(ResourceProfile(), context_window=100))
    fit = ChatRequest(messages=(user_message("small"),))
    over = ChatRequest(messages=(user_message("x" * 4000),))  # est ~1000 tok > window
    responses = await wrapped.agenerate([over, fit], SamplingConfig())
    assert responses[0].text == "" and responses[0].finish_reason == OVER_WINDOW_FINISH
    assert responses[1].text == "r1"  # only the accepted request hit the backend
    assert [r.request_id for (r, _) in inner.seen] == [fit.request_id]
    assert inner.seen[0][0] is fit


def test_planned_backend_passes_inner_slot_through() -> None:
    """The optimizer releases the engine via
    ``getattr(backend, "slot", None)``; a wrapper that swallows ``slot`` turns
    that release into a silent no-op — the exact OOM pattern BackendWrapper
    pins for the traced/normalized chain. PlannedBackend must reach through."""
    inner = _Recorder()
    inner.slot = object()  # type: ignore[attr-defined]
    wrapped = PlannedBackend(inner, RequestPlanner(ResourceProfile()))
    assert getattr(wrapped, "slot", None) is inner.slot


def test_planned_backend_rejects_response_count_mismatch() -> None:
    class _ShortBackend(InferenceBackend):
        name = "short"

        async def agenerate(
            self, requests: Sequence[ChatRequest], sampling: SamplingConfig
        ) -> list[ChatResponse]:
            return []

    wrapped = PlannedBackend(_ShortBackend(), RequestPlanner(ResourceProfile()))
    with pytest.raises(RuntimeError, match="response count mismatch"):
        run_sync(wrapped.agenerate([ChatRequest(messages=(user_message("a"),))], SamplingConfig()))


# ---------------------------------------------------------------------------
# 1b. video: a fixed coarse budget, never the per-image charge
# ---------------------------------------------------------------------------


def test_planner_charges_video_at_the_fixed_budget():
    planner = RequestPlanner(ResourceProfile(), context_window=100_000)
    req = ChatRequest(messages=(user_message(VideoPart("clip.mp4"), "q"),))
    assert isinstance(planner.plan(req), PlannedRequest)


def test_planner_rejects_video_over_window():
    # 65536 chars // 4 = 16384 tok. A reserve is required for the rejection to be
    # real: with the default reserve (0) the usable window is the full 10_000
    # tok and 16384 > 10_000 is already rejected — the reserve makes the case
    # explicit instead of incidental.
    planner = RequestPlanner(ResourceProfile(ctx_reserve=4_096), context_window=10_000)
    out = planner.plan(ChatRequest(messages=(user_message(VideoPart("clip.mp4")),)))
    assert isinstance(out, Rejected)
    assert "exceeds usable window" in out.reason
    # the same planner still hosts a single image: the video charge is the video
    # charge, not the image charge (~256 tok)
    assert isinstance(
        planner.plan(ChatRequest(messages=(user_message(ImagePart("a.jpg")),))), PlannedRequest
    )


def test_planner_rejects_two_videos_at_the_real_window():
    # Measured on Qwen3.5/32k: one NExT-QA clip (32 frames) costs ~11.6k tokens,
    # so two clips plus a prompt cannot fit the window. The coarse charge must
    # be large enough that a duplicated-placeholder offspring is planner-rejected
    # instead of reaching vLLM, whose add_request raises VLLMValidationError and
    # aborts the whole run (matrix_q2b_nextqa_s42: 3 clips, 35483 > 32768).
    planner = RequestPlanner(ResourceProfile(), context_window=32_768)
    req = ChatRequest(messages=(user_message(VideoPart("a.mp4"), VideoPart("b.mp4"), "q" * 2_000),))
    out = planner.plan(req)
    assert isinstance(out, Rejected)
    assert "exceeds usable window" in out.reason


def test_planner_admits_one_video_at_the_real_window():
    # The coarse charge is a guard, not a substitute for the tokenizer: a normal
    # single-clip request must keep flowing through the same 32k window.
    planner = RequestPlanner(ResourceProfile(), context_window=32_768)
    req = ChatRequest(messages=(user_message(VideoPart("a.mp4"), "q" * 2_000),))
    assert isinstance(planner.plan(req), PlannedRequest)


def test_video_does_not_consume_the_image_cap():
    """``max_images`` tracks visual *images* only; a clip is not an image slot."""
    planner = RequestPlanner(ResourceProfile(max_images=0), context_window=100_000)
    req = ChatRequest(messages=(user_message(VideoPart("clip.mp4")),))
    assert isinstance(planner.plan(req), PlannedRequest)


# ---------------------------------------------------------------------------
# 2. build_run_context arms the guard from the declared window
# ---------------------------------------------------------------------------


def _toy_run_kwargs() -> dict[str, Any]:
    cfg_cls = engine_get("strategy", "raw").config_cls
    return {
        "strategy_cls": engine_get("strategy", "raw"),
        "config": cfg_cls(),
        "task": ToyMCQ(),
        "protocol": IdentityProtocol(),
        "eval_model": AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    }


def test_declared_backend_window_arms_the_planner_guard() -> None:
    backend = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    backend.context_window = 8192
    kwargs = _toy_run_kwargs()
    ctx = build_run_context(**kwargs, backend=backend, seed=42)
    assert ctx.evaluator.planner.context_window == 8192


def test_explicit_backend_limits_override_the_declared_window() -> None:
    backend = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    backend.context_window = 8192
    kwargs = _toy_run_kwargs()
    ctx = build_run_context(
        **kwargs, backend=backend, backend_limits=BackendLimits(context_window=4096), seed=42
    )
    assert ctx.evaluator.planner.context_window == 4096


def test_unknown_window_keeps_the_guard_honestly_disarmed() -> None:
    backend = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    kwargs = _toy_run_kwargs()
    ctx = build_run_context(**kwargs, backend=backend, seed=42)
    assert ctx.evaluator.planner.context_window is None


def test_max_steps_zero_fails_fast_at_assembly() -> None:
    """A loop that never steps leaves the finish decision unbound; assembly
    refuses the shape outright."""
    kwargs = _toy_run_kwargs()
    backend = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    with pytest.raises(ValueError, match="max_steps"):
        build_run_context(**kwargs, backend=backend, seed=42, max_steps=0)


def test_wrapper_chain_forwards_the_declared_window(tmp_path) -> None:
    """The runtime hands build_run_context the OUTERMOST backend of the
    raw→traced→normalized chain (runtime.py wiring). If any wrapper swallowed
    the declared window the guard would silently never arm in config-driven
    runs, so the wrapped path must be tested, not only the bare backend."""
    from eyas.backends.normalize import NormalizingBackend
    from eyas.runtime.trace import TracedBackend, TraceRecorder

    raw = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    raw.context_window = 8192
    wired = NormalizingBackend(
        TracedBackend(raw, TraceRecorder(tmp_path / "trace.jsonl", labels={"t": "x"}))
    )
    assert wired.context_window == 8192  # every link forwards
    kwargs = _toy_run_kwargs()
    ctx = build_run_context(**kwargs, backend=wired, seed=42)
    assert ctx.evaluator.planner.context_window == 8192  # guard really armed


# ---------------------------------------------------------------------------
# 2b. capabilities_for(config): declared resource profile tracks the config
# ---------------------------------------------------------------------------


def test_capabilities_for_tracks_config_not_class_defaults() -> None:
    """The class CapSpec freezes
    ResourceProfile(max_images=ApeConfig.demos_per_prompt) at the constructor
    DEFAULT (2); a config raising demos_per_prompt to 4 must not have every
    generation request planner-rejected (4 images > 2) — that would be silent
    seed-only degradation while cost_by_tier still echoes generate traffic."""
    ape = engine_get("strategy", "ape")
    assert ape.capabilities.resource_profile.max_images == 2  # class default view
    cfg = ape.config_cls(demos_per_prompt=4)
    assert ape.capabilities_for(cfg).resource_profile.max_images == 4
    assert ape.capabilities.resource_profile.max_images == 2  # class view untouched

    backend = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    ctx = build_run_context(ape, cfg, ToyMCQ(), IdentityProtocol(), backend, eval_model, seed=42)
    # the armed planner enforces the CONFIGured limit, not the frozen default
    assert ctx.evaluator.planner.profile.max_images == 4


# ---------------------------------------------------------------------------
# 3. judging channel: pinned greedy across deployment sampling changes
# ---------------------------------------------------------------------------


def _recording_mock(responder: Any) -> tuple[ScriptedMockBackend, _Recorder]:
    """ScriptedMockBackend whose agenerate is wrapped to record (request,
    sampling) pairs. The wrapper must capture the ORIGINAL bound method —
    re-reading backend.agenerate after the assignment self-recurses."""
    backend = ScriptedMockBackend(responder=responder)
    recorder = _Recorder()
    orig = backend.agenerate

    async def via_recorder(
        requests: Sequence[ChatRequest], s: SamplingConfig
    ) -> list[ChatResponse]:
        recorder.seen.extend((r, s) for r in requests)
        return await orig(requests, s)

    backend.agenerate = via_recorder  # type: ignore[method-assign]
    return backend, recorder


def _judge_run(sampling: SamplingConfig) -> list[tuple[ChatRequest, SamplingConfig]]:
    """One absolute+judge evaluation; returns the JUDGE-channel (request,
    sampling) pairs exactly as they reached the backend."""

    def responder(request: ChatRequest, s: SamplingConfig, idx: int) -> ChatResponse:
        if (request.request_id or "").startswith("judge:"):
            return ChatResponse(text="cat")
        return ChatResponse(text="hmm")  # unparseable -> every sample reaches the judge

    backend, recorder = _recording_mock(responder)
    evaluator = Evaluator(
        task=CostToyMCQ(),
        protocol=IdentityProtocol(),
        backend=backend,
        planner=RequestPlanner(ResourceProfile()),
        model=AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        judge=McqLLMJudge(),
        sampling=sampling,
    )
    prompt = Prompt("Classify the animal.")
    run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))
    return [(r, s) for (r, s) in recorder.seen if (r.request_id or "").startswith("judge:")]


def test_judge_channel_is_byte_identical_across_deployment_seeds() -> None:
    run_a = _judge_run(SamplingConfig(temperature=0.0, seed=7, thinking=None))
    run_b = _judge_run(SamplingConfig(temperature=0.0, seed=99, thinking=True))
    assert run_a and run_b  # the judge channel really fired
    # doctrine: a verdict that rides the run's seed/thinking would
    # make relative scoring irreproducible — judging is pinned greedy
    assert [(dataclasses.asdict(r), dataclasses.asdict(s)) for r, s in run_a] == [
        (dataclasses.asdict(r), dataclasses.asdict(s)) for r, s in run_b
    ]
    for _, sampling in run_a:
        assert sampling.temperature == 0.0
        assert sampling.seed is None
        assert sampling.thinking is None


def test_over_window_judge_request_is_rejected_and_unbilled() -> None:
    # window armed between the eval estimate (~20 tok) and the judge estimate
    # (hundreds of chars): the verdict channel is rejected, evaluation flows
    def responder(request: ChatRequest, s: SamplingConfig, idx: int) -> ChatResponse:
        if (request.request_id or "").startswith("judge:"):
            return ChatResponse(text="cat")
        return ChatResponse(text="hmm")

    mock, recorder = _recording_mock(responder)
    evaluator = Evaluator(
        task=CostToyMCQ(),
        protocol=IdentityProtocol(),
        backend=mock,
        planner=RequestPlanner(ResourceProfile(), context_window=30),
        model=AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        judge=McqLLMJudge(),
    )
    prompt = Prompt("Classify the animal.")
    evals = run_sync(evaluator.aevaluate([prompt], full_split_plan(prompt), "train", 0))
    judged = [r for (r, _) in recorder.seen if (r.request_id or "").startswith("judge:")]
    assert judged == []  # over-window judge requests never left the chokepoint
    assert evals[0].cost.requests == 2 and evals[0].cost.judge_requests == 0


# ---------------------------------------------------------------------------
# 4. best_id: declared champion vs archive-score leader
# ---------------------------------------------------------------------------


@register_strategy("audit_liar")
class _LiarStrategy(RawStrategy):
    """Toy method that always declares the archive-score LOSER as champion —
    models the by-construction divergence (bandit-protegi's arm winner vs the
    ledger leader) with certainty, instead of relying on a fixture's luck."""

    config_cls: ClassVar[type] = RawConfig

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        return PromptListState(candidates=list(seeds), round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, PromptListState)
        missing = [p for p in state.candidates if not view.evaluated(p)]
        if missing:
            return Decision("evaluate", prompts=tuple(missing))
        ranked = rank_by_score(state.candidates, view.score)
        return Decision("finish", best_id=ranked[-1].id)


def test_declared_best_divergence_warns_without_changing_the_ledger(caplog) -> None:
    cfg = _LiarStrategy.config_cls()
    backend = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    kwargs = _toy_run_kwargs()
    ctx = build_run_context(
        _LiarStrategy,
        cfg,
        kwargs["task"],
        kwargs["protocol"],
        backend,
        kwargs["eval_model"],
        seed=42,
    )
    good, bad = Prompt("Answer step by step."), Prompt("Answer instantly.")
    with caplog.at_level(logging.WARNING, logger="eyas.engine.strategy"):
        result = OptimizerLoop(ctx).run(_LiarStrategy(cfg, ctx), [good.text, bad.text])
    assert result.best is not None
    warns = [r.message for r in caplog.records if "archive-score leader" in r.message]
    assert len(warns) == 1
    # the WARNING names both sides: the ledger winner and the declared loser
    declared = bad.id if result.best.prompt_id == good.id else good.id
    assert result.best.prompt_id in warns[0] and declared in warns[0]
    # and the ledger still ranks by archive score: the DECLARED loser wins
    assert result.best.prompt_id == result.ranked[0].prompt_id


def test_declared_best_matching_leader_stays_silent(caplog) -> None:
    raw = engine_get("strategy", "raw")
    cfg = raw.config_cls()
    backend = ScriptedMockBackend(responder=lambda r, s, i: ChatResponse(text="cat"))
    kwargs = _toy_run_kwargs()
    ctx = build_run_context(
        raw, cfg, kwargs["task"], kwargs["protocol"], backend, kwargs["eval_model"], seed=42
    )
    with caplog.at_level(logging.WARNING, logger="eyas.engine.strategy"):
        OptimizerLoop(ctx).run(raw(cfg, ctx), ["Classify the animal."])
    assert not [r for r in caplog.records if "archive-score leader" in r.message]
