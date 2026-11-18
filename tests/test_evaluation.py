"""Absolute-evaluation batch mechanics: within-batch (prompt, sample) dedup +
per-WorkItem billing.

Contract under test:
1. one evaluate batch dedups (prompt_id, sample_id) pairs BEFORE dispatch —
   first occurrence wins, candidate-first (prompt-major) request order intact
   (radix-cache prefix reuse depends on it);
2. each WorkItem is billed only for the pairs it owns; a pair shared across
   items is billed once total, attributed to the first owning item (eval and
   judge channels alike);
3. single-item / disjoint-prompt batches keep the existing behavior (the
   per-method contract + resume tests are the wider regression net).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from conftest import make_evaluator

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    system_message,
    user_message,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.backends.scheduling import run_sync
from eyas.core.objective import Objective
from eyas.engine.budget import LoadPlan, WorkItem
from eyas.engine.domain import Prompt, SampleOutcome
from eyas.engine.evaluation import Evaluator, InMemoryRecordStore, SqliteRecordStore
from eyas.judges.mcq_judge import McqLLMJudge
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.mcq import match_prediction
from eyas.tasks.protocol import IdentityProtocol

_CHOICES = ["cat", "dog"]
_EVAL_USAGE = {"prompt_tokens": 10, "completion_tokens": 4}
_JUDGE_USAGE = {"prompt_tokens": 6, "completion_tokens": 2}
_LABELS = {"s1": "cat", "s2": "dog", "s3": "cat"}


class DedupToyMCQ(Task):
    """Minimal text-only MCQ shape (direct Evaluator assembly — no registry)."""

    name = "dedup_toy_mcq"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(
                id=sid,
                inputs={"question": f"animal {sid}", "choices": _CHOICES},
                references={"label": label},
            )
            for sid, label in _LABELS.items()
        ]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"])),
            request_id=sample.id,
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(
            value=match_prediction(response.text, sample.inputs["choices"]), raw=response.text
        )

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _backend(
    eval_text: str | None = None, *, dispatched: list[tuple[str, str]] | None = None
) -> ScriptedMockBackend:
    """Responder answering each eval request with its sample's label (or a fixed
    unparseable text to force the judge channel); judge requests answer "cat".
    ``dispatched`` records (prompt text, sample id) in dispatch order."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        if request.request_id.startswith("judge:"):
            return ChatResponse(text="cat", usage=dict(_JUDGE_USAGE))
        if dispatched is not None:
            prompt_part = request.messages[0].content[0]
            assert hasattr(prompt_part, "text")
            dispatched.append((prompt_part.text, request.request_id))
        text = eval_text if eval_text is not None else _LABELS[request.request_id]
        return ChatResponse(text=text, usage=dict(_EVAL_USAGE))

    return ScriptedMockBackend(responder=responder)


def _evaluator(backend: ScriptedMockBackend, judge: McqLLMJudge | None = None) -> Evaluator:
    return make_evaluator(
        DedupToyMCQ(),
        backend,
        protocol=IdentityProtocol(),
        produces=frozenset({"accuracy", "parse_fail"}),
        judge=judge,
    )


def test_shared_prompt_overlapping_items_dispatch_unique_pairs_once() -> None:
    prompt = Prompt("Classify the animal.")
    dispatched: list[tuple[str, str]] = []
    backend = _backend(dispatched=dispatched)
    plan = LoadPlan(items=(WorkItem(prompt, ("s1", "s2")), WorkItem(prompt, ("s2", "s3"))))
    evals = run_sync(_evaluator(backend).aevaluate([prompt], plan, "train", 0))
    # 3 unique pairs, not 4: the shared (prompt, s2) is dispatched exactly once
    assert backend.call_count == 3
    assert [sid for _, sid in dispatched] == ["s1", "s2", "s3"]
    # both items still see their OWN full subset, scored from the shared outcome
    assert [e.prompt_id for e in evals] == [prompt.id, prompt.id]
    assert [o.sample_id for o in evals[0].per_sample] == ["s1", "s2"]
    assert [o.sample_id for o in evals[1].per_sample] == ["s2", "s3"]
    assert evals[0].scores["accuracy"] == 1.0
    assert evals[1].scores["accuracy"] == 1.0


def test_shared_pairs_billed_once_first_owner_wins() -> None:
    prompt = Prompt("Classify the animal.")
    backend = _backend()
    plan = LoadPlan(items=(WorkItem(prompt, ("s1", "s2")), WorkItem(prompt, ("s2", "s3"))))
    evals = run_sync(_evaluator(backend).aevaluate([prompt], plan, "train", 0))
    # ownership: s1, s2 -> item 0; s3 -> item 1 (first occurrence wins)
    assert evals[0].cost.requests == 2
    assert evals[1].cost.requests == 1
    assert evals[0].cost.input_tokens == 2 * _EVAL_USAGE["prompt_tokens"]
    assert evals[0].cost.output_tokens == 2 * _EVAL_USAGE["completion_tokens"]
    assert evals[1].cost.input_tokens == _EVAL_USAGE["prompt_tokens"]
    assert evals[1].cost.output_tokens == _EVAL_USAGE["completion_tokens"]
    # the batch bill is counted exactly once across items
    assert sum(e.cost.requests for e in evals) == backend.call_count == 3
    assert sum(e.cost.input_tokens for e in evals) == 3 * _EVAL_USAGE["prompt_tokens"]
    assert sum(e.cost.output_tokens for e in evals) == 3 * _EVAL_USAGE["completion_tokens"]


def test_candidate_first_order_preserved_with_dedup() -> None:
    a = Prompt("Prompt A: classify.")
    b = Prompt("Prompt B: classify.")
    dispatched: list[tuple[str, str]] = []
    backend = _backend(dispatched=dispatched)
    plan = LoadPlan(
        items=(WorkItem(a, ("s1", "s2")), WorkItem(b, ("s2",)), WorkItem(a, ("s2", "s3")))
    )
    evals = run_sync(_evaluator(backend).aevaluate([a, b], plan, "train", 0))
    # prompt-major first-occurrence order: A's pairs stay contiguous ahead of B's
    assert dispatched == [(a.text, "s1"), (a.text, "s2"), (b.text, "s2"), (a.text, "s3")]
    assert backend.call_count == 4
    # (a, s2) is shared between items 0 and 2: billed to item 0 only
    assert [e.cost.requests for e in evals] == [2, 1, 1]
    assert sum(e.cost.requests for e in evals) == 4


def test_within_item_duplicate_ids_dispatch_once() -> None:
    prompt = Prompt("Classify the animal.")
    backend = _backend()
    plan = LoadPlan(items=(WorkItem(prompt, ("s1", "s1", "s2")),))
    evals = run_sync(_evaluator(backend).aevaluate([prompt], plan, "train", 0))
    assert backend.call_count == 2
    assert evals[0].cost.requests == 2


def test_cache_hit_shared_pair_skips_dispatch_for_both_items() -> None:
    prompt = Prompt("Classify the animal.")
    backend = _backend()
    evaluator = _evaluator(backend)
    store = InMemoryRecordStore()
    store.put(SampleOutcome("s2", "dog", {"accuracy": 1.0}, correct=True), prompt.id, "train")
    evaluator.store = store
    plan = LoadPlan(items=(WorkItem(prompt, ("s1", "s2")), WorkItem(prompt, ("s2", "s3"))))
    evals = run_sync(evaluator.aevaluate([prompt], plan, "train", 0))
    # the cached pair is dispatched by neither item
    assert backend.call_count == 2
    assert [e.cost.requests for e in evals] == [1, 1]
    for e in evals:
        s2 = next(o for o in e.per_sample if o.sample_id == "s2")
        assert s2.prediction == "dog"
        assert s2.scores["accuracy"] == 1.0


def test_judge_channel_shared_pairs_billed_once() -> None:
    prompt = Prompt("Classify the animal.")
    backend = _backend("hmm")  # unparseable eval answers force every pair through the judge
    plan = LoadPlan(items=(WorkItem(prompt, ("s1", "s2")), WorkItem(prompt, ("s2", "s3"))))
    evals = run_sync(_evaluator(backend, judge=McqLLMJudge()).aevaluate([prompt], plan, "train", 0))
    # 3 eval + 3 judge calls: the shared pair is judged once too
    assert backend.call_count == 6
    assert [e.cost.judge_requests for e in evals] == [2, 1]
    assert [e.cost.requests for e in evals] == [4, 2]
    assert sum(e.cost.requests for e in evals) == 6
    assert sum(e.cost.judge_input_tokens for e in evals) == 3 * _JUDGE_USAGE["prompt_tokens"]
    assert sum(e.cost.judge_output_tokens for e in evals) == 3 * _JUDGE_USAGE["completion_tokens"]
    assert sum(e.cost.input_tokens for e in evals) == (
        3 * _EVAL_USAGE["prompt_tokens"] + 3 * _JUDGE_USAGE["prompt_tokens"]
    )
    assert sum(e.cost.output_tokens for e in evals) == (
        3 * _EVAL_USAGE["completion_tokens"] + 3 * _JUDGE_USAGE["completion_tokens"]
    )


def test_media_failure_sentinel_lands_durable_error_row(tmp_path: Path) -> None:
    """Y15 pin: a media failure rides the over-window sentinel shape
    (finish_reason=OVER_WINDOW_FINISH, empty text) — the evaluator turns it
    into a DURABLE record-store row with a null prediction and parse_fail=1.0
    (the same mechanism the over-window skip uses), while the healthy samples
    score normally and the batch survives."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        if request.request_id == "s2":  # the sample whose media file failed to load
            return ChatResponse(
                text="", request_id=request.request_id, finish_reason=OVER_WINDOW_FINISH
            )
        return ChatResponse(text=_LABELS[request.request_id], usage=dict(_EVAL_USAGE))

    prompt = Prompt("Classify the animal.")
    backend = ScriptedMockBackend(responder=responder)
    evaluator = _evaluator(backend)
    store = SqliteRecordStore(path=tmp_path / "cache" / "eyas.sqlite")
    evaluator.store = store
    plan = LoadPlan(items=(WorkItem(prompt, ("s1", "s2", "s3")),))
    evals = run_sync(evaluator.aevaluate([prompt], plan, "train", 0))
    # the error row is durable: null prediction + parse_fail=1.0, readable back
    row = store.get(prompt.id, "s2", "train")
    assert row is not None
    assert row.prediction is None
    assert row.scores == {"parse_fail": 1.0}
    # the healthy samples still scored; the failed one counts as 0 accuracy
    assert evals[0].scores["accuracy"] == pytest.approx(2 / 3)
    assert evals[0].scores["parse_fail"] == pytest.approx(1 / 3)
    assert backend.call_count == 3  # no exception, no batch re-run
    store.close()
