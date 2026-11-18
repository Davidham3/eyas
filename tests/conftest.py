"""Shared test helpers: engine Evaluator assembly helper (reused by per-task end-to-end contracts)."""

from __future__ import annotations

import pytest

from eyas.backends.base import InferenceBackend
from eyas.engine.budget import LoadPlan, WorkItem
from eyas.engine.capability import ResourceProfile
from eyas.engine.domain import Prompt
from eyas.engine.evaluation import AbsoluteEvalModel, Evaluator
from eyas.engine.planner import RequestPlanner
from eyas.judges.base import Judge
from eyas.tasks.base import Task
from eyas.tasks.protocol import VariantProtocol


def make_evaluator(
    task: Task,
    backend: InferenceBackend,
    protocol: VariantProtocol,
    produces: frozenset[str],
    judge: Judge | None = None,
) -> Evaluator:
    """Standard zero-GPU engine Evaluator assembly for contract tests (planner + absolute scoring)."""
    return Evaluator(
        task=task,
        protocol=protocol,
        backend=backend,
        planner=RequestPlanner(ResourceProfile(), context_window=None),
        model=AbsoluteEvalModel(produces=frozenset(produces)),
        judge=judge,
    )


def full_split_plan(prompt: Prompt) -> LoadPlan:
    """Empty sample_ids = the full split."""
    return LoadPlan(items=(WorkItem(prompt),))


# Engine backend class for the sampling-contract pins: the module imports without
# the extra (all engine imports are deferred), so exposing the class here is safe
# for every test session.
@pytest.fixture
def vllm_cls() -> type:
    from eyas.backends.vllm import VllmBackend

    return VllmBackend
