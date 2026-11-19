"""Mock contract test for ``my_method`` — fully self-contained.

Everything a method test needs is assembled in this file: a toy task registered
under a name nothing else uses, a scripted mock backend, an evaluator, and the
managed loop. Nothing here imports from the framework's own test suite, so this
file is also the template you copy for your own method.

Two ways to run it, both documented in the README. From the repo root:

    uv run pytest examples/strategy_skeleton -q

or, on a plain checkout with no editable install (no prepared environment —
`../../src` is the checkout's own source tree):

    cd examples/strategy_skeleton && PYTHONPATH=../../src uv run python -m pytest -q
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

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
from eyas.backends.scheduling import run_sync
from eyas.core.objective import Objective
from eyas.engine import (
    builtin,  # noqa: F401  registers the built-in pieces (strategies, budgets)
)
from eyas.engine.budget import FlatBudget, LoadPlan, WorkItem
from eyas.engine.capability import ResourceProfile
from eyas.engine.domain import Prompt
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
    CompareEvalModel,
    Evaluator,
    InMemoryRecordStore,
    SqliteRecordStore,
)
from eyas.engine.planner import RequestPlanner
from eyas.engine.strategy import OptimizerLoop, RunResult, build_run_context
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.protocol import IdentityProtocol

# The package import below is absolute, so it needs the repo root on sys.path.
# pytest puts it there when you run `pytest examples/strategy_skeleton` from the
# root; when you enter this directory instead (the README's no-install command),
# nothing else would resolve it — an installed `eyas` distribution does not make
# `examples` importable. Resolving it from this file's location works from
# anywhere and keeps the test runnable in both shapes.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.strategy_skeleton import my_method  # noqa: E402

# What the toy world looks like: the question ids the mock knows, and its one
# rule. A prompt containing "step by step" (case-insensitive) makes the mock
# answer correctly, so the score is decided by the prompt text alone — the
# smallest shape in which a search has something to find.
_LABELS = {"s1": "cat", "s2": "dog", "s3": "bird"}


@register_task("skeleton_toy_task")
class SkeletonToyTask(Task):
    """Three trivial samples, no images: the answer is the reference label."""

    name = "skeleton_toy_task"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(id=sid, inputs={"question": sid}, references={"label": label})
            for sid, label in _LABELS.items()
        ]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _answering_from_the_prompt(
    request: ChatRequest, sampling: SamplingConfig, idx: int
) -> ChatResponse:
    """The mock's whole model of the world: obey the system prompt, answer well.

    A request is a sequence of messages; each message's content is a list of
    parts (text and images), so reading the instruction means filtering to the
    text parts of the system message.
    """
    system = "".join(
        part.text
        for message in request.messages
        if message.role == "system"
        for part in message.content
        if isinstance(part, TextPart)
    )
    label = _LABELS.get(user_text(request), "")
    return ChatResponse(text=label if "step by step" in system.lower() else "nonsense")


def _registry_form(task: SkeletonToyTask, store: InMemoryRecordStore | SqliteRecordStore):
    """The pieces ``build_run_context`` assembles for you, wired by hand.

    Build this when you want to exercise the evaluator in isolation — a cache
    question, a cost question, a judge in the loop — without driving a whole
    optimization run. For a run, ``build_run_context`` below wires the same
    pieces from the strategy's capability declaration.
    """
    return Evaluator(
        task=task,
        protocol=IdentityProtocol(),
        backend=ScriptedMockBackend(responder=_answering_from_the_prompt),
        planner=RequestPlanner(ResourceProfile(), context_window=None),
        # produces = the metric keys this evaluation can emit; assembly validation
        # refuses a strategy whose optimization objective is not among them
        model=AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        store=store,
    )


def _run(
    config: my_method.RandomSearchConfig,
    seeds: list[str],
    *,
    eval_model: AbsoluteEvalModel | CompareEvalModel | None = None,
    store: InMemoryRecordStore | SqliteRecordStore | None = None,
    run_dir: Path | None = None,
    max_steps: int | None = None,
    resume: dict[str, object] | None = None,
    on_round: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    """Library-form run: assembly → managed loop → result. No YAML, no run dir.

    ``max_steps`` is the loop's hard ceiling, counted in decisions: this method
    uses one step to evaluate and one to propose per round, so the ceiling has to
    leave room for the method's own plan plus its final ``finish``. Pass it
    smaller than the plan to simulate an interrupted run (see the resume test).
    """
    ctx = build_run_context(
        my_method.RandomSearchStrategy,
        config,
        SkeletonToyTask(),
        IdentityProtocol(),
        ScriptedMockBackend(responder=_answering_from_the_prompt),
        eval_model or AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        budget=FlatBudget(),
        store=store or InMemoryRecordStore(),
        seed=config.seed,
        max_steps=max_steps if max_steps is not None else 2 * config.max_rounds + 4,
        run_dir=run_dir,
    )
    strategy = my_method.RandomSearchStrategy(config, ctx)
    return OptimizerLoop(ctx).run(strategy, seeds, resume=resume, on_round=on_round)


def test_registration_and_declared_surface():
    """What the framework reads off the class before running anything."""
    assert my_method.RandomSearchStrategy.name == "random_search"
    assert my_method.RandomSearchStrategy.capabilities.name == "random_search"
    # the strategy section's params are bound to this dataclass by the run entry
    assert my_method.RandomSearchStrategy.config_cls is my_method.RandomSearchConfig
    config = my_method.RandomSearchConfig(**{"neighborhood_size": 2, "max_rounds": 1})
    assert (config.neighborhood_size, config.edit_probability, config.seed) == (2, 0.3, 0)


def test_mock_run_finds_a_better_prompt_and_stops():
    """The contract test every method needs: zero GPU, ends on ``finish``."""
    config = my_method.RandomSearchConfig(
        neighborhood_size=3, edit_probability=0.5, seed=0, max_rounds=3
    )
    result = _run(config, ["Classify the animal."])

    assert result.best is not None
    # the mock answers correctly only for prompts carrying "step by step", so a
    # perfect score proves a mutation was proposed, scored and kept
    assert result.best.scores["accuracy"] == 1.0
    # the rounds the strategy drove: one for the seeds, then exactly its own max_rounds
    assert len(result.history) == config.max_rounds + 1
    assert result.total_cost.requests > 0
    # per-sample observations survive the evaluation, and the winner's text is kept
    assert all(outcome.correct is not None for outcome in result.best.per_sample)
    assert "step by step" in (result.best_text() or "").lower()


def test_cost_is_billed_per_round_and_candidate():
    """Requests = candidates × samples scored, so costs stay auditable."""
    config = my_method.RandomSearchConfig(neighborhood_size=2, max_rounds=1)
    result = _run(config, ["Classify the animal."])
    # every candidate in a round is scored on all 3 samples, and no request is
    # counted twice: the per-round bills sum to the run total
    assert sum(round_record.n_requests for round_record in result.history) == (
        result.total_cost.requests
    )
    assert all(r.n_requests == r.n_candidates * len(_LABELS) for r in result.history)


def test_dedup_holds_within_a_run(tmp_path):
    """Same normalized text = same identity = already-scored samples are not sent
    to the backend a second time (the record store lives with the run)."""
    store = SqliteRecordStore(path=tmp_path / "outcomes.sqlite")  # the run-dir store
    evaluator = _registry_form(SkeletonToyTask(), store)
    a = Prompt("Classify the animal.")
    b = Prompt("  Classify the animal.\n")  # normalization makes these the same prompt
    assert a.id == b.id

    first = run_sync(evaluator.aevaluate([a], LoadPlan(items=(WorkItem(a),)), "train", 0))
    assert (store.misses, store.hits) == (3, 0)
    assert first[0].cost.requests == 3  # three samples, three requests

    # a later round asking for the same prompt (however it is spelled) is free
    second = run_sync(evaluator.aevaluate([b], LoadPlan(items=(WorkItem(b),)), "train", 1))
    assert (store.misses, store.hits) == (3, 3)
    assert second[0].cost.requests == 0  # nothing went to the backend
    assert second[0].scores == first[0].scores
    # the per-sample observations come back complete on the zero-request round,
    # which is where a gradient-style method reads its signal
    assert [o.prediction for o in second[0].per_sample] == [
        o.prediction for o in first[0].per_sample
    ]
    # the loop never gets that far though: view.evaluated() already tells the
    # method the candidate has been scored, so it is not queued again


def test_run_with_no_rounds_still_reports_the_seed():
    """Termination is the method's call; the archive still carries the seed's metrics."""
    result = _run(my_method.RandomSearchConfig(max_rounds=0), ["Classify the animal."])
    assert result.best is not None
    assert result.best.scores["accuracy"] == 0.0  # the seed never says "step by step"
    assert len(result.ranked) == 1
    assert len(result.history) == 1  # the seed was evaluated, then the method finished


def test_assembly_refuses_an_evaluation_this_method_cannot_rank_with():
    """``capabilities`` is checked before a single request is sent: this method
    ranks by absolute scores, so a compare-only evaluation is rejected at startup."""
    with pytest.raises(ValueError, match="absolute_eval"):
        _run(
            my_method.RandomSearchConfig(max_rounds=1),
            ["Classify the animal."],
            eval_model=CompareEvalModel(),
        )


def test_resume_continues_the_same_run_from_the_checkpoint(tmp_path):
    """A resume restores your typed state, the run's RNG, the archive and the cost
    ledger — so it reaches the same candidates and never re-queues a scored one."""
    config = my_method.RandomSearchConfig(neighborhood_size=3, edit_probability=0.5, max_rounds=2)
    store_path = tmp_path / "eyas.sqlite"  # the run dir's record store
    seeds = ["Classify the animal."]

    snapshots: list[dict[str, Any]] = []
    interrupted = _run(
        config,
        seeds,
        store=SqliteRecordStore(path=store_path),
        max_steps=1,  # stop the loop after the seed round, as an interrupted run would
        on_round=snapshots.append,  # what runtime.py writes to checkpoint.json — fired
        # by evaluate decisions only; propose/finish never snapshot (see README run-dir table)
    )
    assert len(interrupted.history) == 1

    resumed = _run(
        config,
        seeds,
        store=SqliteRecordStore(path=store_path),
        resume=json.loads(json.dumps(snapshots[-1])),
    )
    uninterrupted = _run(config, seeds)

    assert resumed.best is not None and uninterrupted.best is not None
    assert resumed.best.prompt_id == uninterrupted.best.prompt_id
    assert set(resumed.texts) == set(uninterrupted.texts)  # same candidates explored
    # the ledger carries on rather than restarting: identical per-round bills, and
    # a total that matches the run that was never interrupted
    assert [(r.n_candidates, r.n_requests) for r in resumed.history] == [
        (r.n_candidates, r.n_requests) for r in uninterrupted.history
    ]
    assert resumed.total_cost.requests == uninterrupted.total_cost.requests
