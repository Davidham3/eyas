"""Contract tests (OPRO): history meta-prompt fingerprints (ICLR 2024, arXiv:2309.03409),
empty-round retry, trajectory rules.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.domain import (
    Prompt,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
)
from eyas.engine.genio import (
    GEN_MARKER,
    OUT_END,
    OUT_START,
)
from eyas.engine.strategies.opro import OproConfig, OproStrategy, serialize_meta_prompt
from eyas.engine.strategy import OptimizerLoop, build_run_context
from eyas.tasks.base import Sample
from eyas.tasks.protocol import IdentityProtocol
from tests.contract.helpers import _LABELS, ToyMCQ, _gen_mock, _run


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
