"""Shared engine-assembly helpers and registered toy tasks for the test suite.

Split out of ``tests/conftest.py`` (plain functions never belonged there) and the
old flat test modules (cross-module ``from test_x import`` was a flat-layout
workaround): anything two or more topics assemble against lives here.
"""

from __future__ import annotations

from collections.abc import Sequence

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    system_message,
    user_message,
)
from eyas.core.objective import Objective
from eyas.engine.budget import LoadPlan, WorkItem
from eyas.engine.capability import ResourceProfile
from eyas.engine.domain import Prompt
from eyas.engine.evaluation import AbsoluteEvalModel, Evaluator
from eyas.engine.planner import RequestPlanner
from eyas.judges.base import Judge
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.mcq import match_prediction
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


_CHOICES = ["cat", "dog"]


def _samples() -> list[Sample]:
    return [
        Sample(
            id="s1",
            inputs={"image": "/data/1.jpg", "choices": _CHOICES, "group": "g"},
            references={"label": "cat"},
        ),
        Sample(
            id="s2",
            inputs={"image": "/data/2.jpg", "choices": _CHOICES, "group": "g"},
            references={"label": "dog"},
        ),
    ]


@register_task("cost_toy_mcq")
class CostToyMCQ(Task):
    """Minimal MCQ shape: unparseable text -> value=None (so exact_match abstains
    and the LLM judge takes over)."""

    name = "cost_toy_mcq"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return _samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["image"])),
            request_id=sample.id,
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        value = match_prediction(response.text, sample.inputs["choices"])
        return Prediction(value=value, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


@register_task("toy_final_eval")
class ToyFinalEval(Task):
    """The mock can answer train samples (q1 -> cat) but not test samples (the mock always answers cat) --
    final metric 0.0 vs train 1.0 forms a readable signal that "the final evaluation really ran on test"."""

    name = "toy_final_eval"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        if split == "test":
            return [Sample(id="q_test", inputs={"question": "q_test"}, references={"label": "dog"})]
        return [Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"})]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def toy_final_eval_cfg(**run_extra: object) -> dict[str, object]:
    return {
        "run": {"name": "toy_final_eval_run", "seed": 0, **run_extra},
        "task": {"name": "toy_final_eval"},
        "protocol": {"name": "identity"},
        "backend": {"name": "mock", "params": {"responses": ["cat"], "cycle": True}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {
            "name": "cot",
            "params": {"base_prompt": "Classify the animal.", "cot_suffix": "Think step by step."},
        },
    }
