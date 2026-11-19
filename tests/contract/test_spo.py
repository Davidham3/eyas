"""Contract tests (SPO): self-supervised OvO voting fingerprints (SPO,
Findings of EMNLP 2025, arXiv:2502.06855), accept rules, schedules, resume.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from random import Random
from typing import Any

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    system_message,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.core.objective import Objective
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.budget import WorkItem
from eyas.engine.capability import BackendLimits
from eyas.engine.domain import (
    OptimizerState,
    Prompt,
    PromptListState,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
)
from eyas.engine.genio import (
    GEN_MARKER,
    MAX_BODY_LEN,
    OUT_END,
    OUT_START,
)
from eyas.engine.planner import PlannedRequest, Rejected, RequestPlanner
from eyas.engine.strategies.raw import RawConfig, RawStrategy
from eyas.engine.strategies.spo import (
    SpoConfig,
    SpoState,
    SpoStrategy,
    _ensure_choices_placeholder,
)
from eyas.engine.strategy import Decision, EvalView, OptimizerLoop, build_run_context
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.math_qa import MathQaTask
from eyas.tasks.protocol import IdentityProtocol
from tests.contract.helpers import _LABELS, ToyMCQ, _gen_mock, _run, _toy_samples


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


# --- SPO self-supervised OvO fingerprints ---
#
# Paper (SPO, Findings of EMNLP 2025, arXiv:2502.06855 / official FoundationAgents/SPO,
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
