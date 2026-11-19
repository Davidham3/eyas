"""Assembly-time resource intersection: backends declare ``max_images``
and the strategy's ResourceProfile is checked against that declaration when
the RunContext is built — an over-limit assembly is a startup ValueError, not
a planner rejection paid for request by request after the run starts.

Propositions:
(a) a synthetic strategy consuming 2 images per request + a backend declaring
    max_images=1 -> build_run_context raises at assembly;
(b) max_images=None (unknown / server-side limit) -> the bound is skipped;
(c) runtime wiring: a vllm config with limit_mm_per_prompt image=1 + ape
    demos_per_prompt=2 -> build_context_from_config raises before any engine
    launch (the declaration survives the raw->traced->normalized chain).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

import eyas.builtin  # noqa: F401  trigger task/backend registration
import eyas.engine.builtin  # noqa: F401  trigger strategy/eval-model registration
from eyas.backends.base import ChatRequest, ChatResponse, user_message
from eyas.backends.mock import ScriptedMockBackend
from eyas.core.objective import Objective
from eyas.engine.capability import (
    ABSOLUTE_EVAL,
    BackendLimits,
    CapSpec,
    EvalModelCap,
    ResourceProfile,
    TaskCap,
    validate_assembly,
)
from eyas.engine.domain import OptimizerState, Prompt, PromptListState
from eyas.engine.evaluation import AbsoluteEvalModel
from eyas.engine.runtime import build_context_from_config
from eyas.engine.strategy import Decision, EvalView, Strategy, build_run_context
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.protocol import IdentityProtocol


class _ToyTask(Task):
    name = "toy_images"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"})]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(messages=(user_message(sample.inputs["question"]),))

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0}


@dataclass
class _TwoImageConfig:
    """No parameter fields: the resource profile is fixed at 2 images."""


class _TwoImageStrategy(Strategy):
    """Synthetic fixture: consumes 2 images per request (the ape demo shape)."""

    config_cls: ClassVar[type] = _TwoImageConfig
    state_cls: ClassVar[type] = PromptListState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="_test_two_image",
        requires_eval=frozenset({ABSOLUTE_EVAL}),
        resource_profile=ResourceProfile(max_images=2),
    )

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        return PromptListState(candidates=list(seeds), round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, PromptListState)
        return Decision("finish", best_id=state.candidates[0].id)


class _OneImageBackend(ScriptedMockBackend):
    """Backend declaring a one-image engine limit (the vllm default shape)."""

    max_images = 1


def _assemble(backend: Any, **kw: Any) -> Any:
    return build_run_context(
        _TwoImageStrategy,
        _TwoImageConfig(),
        _ToyTask(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        **kw,
    )


def test_validate_assembly_reports_image_intersection():
    # The pure rule: an explicit BackendLimits bound produces the violation text.
    violations = validate_assembly(
        _TwoImageStrategy.capabilities,
        TaskCap(name="toy_images", metrics=frozenset({"accuracy"}), optimize_key="accuracy"),
        [
            EvalModelCap(
                name="absolute",
                semantics=frozenset({ABSOLUTE_EVAL}),
                produces=frozenset({"accuracy"}),
            )
        ],
        BackendLimits(max_images=1),
    )
    assert any("2 images per request" in v and "backend limit of 1" in v for v in violations)


def test_over_limit_strategy_rejected_at_assembly():
    # (a) The backend's declaration must reach validate_assembly through
    # build_run_context's fallback (no explicit backend_limits argument).
    with pytest.raises(ValueError, match="exceeding the backend limit of 1"):
        _assemble(_OneImageBackend(responses=["cat"], cycle=True))


def test_unknown_backend_limit_skips_intersection():
    # (b) None = unknown/server-side: the bound is honestly skipped.
    ctx = _assemble(ScriptedMockBackend(responses=["cat"], cycle=True))
    assert ctx.primary_objective.key == "accuracy"


def test_explicit_backend_limits_win_over_declaration():
    # The explicit argument keeps overriding what the backend declares.
    ctx = _assemble(
        _OneImageBackend(responses=["cat"], cycle=True),
        backend_limits=BackendLimits(max_images=2),
    )
    assert ctx.primary_objective.key == "accuracy"


def _vllm_cfg(tmp_path: Path, image_limit: int | None, demos: int) -> dict[str, Any]:
    """vllm backend config (dummy model path; the engine is never launched —
    assembly must verdict before slot.ensure) + ape consuming `demos` images."""
    params: dict[str, Any] = {"model_path": "dummy"}
    if image_limit is not None:
        params["engine_kwargs"] = {"limit_mm_per_prompt": {"image": image_limit}}
    return {
        "run": {"name": "y11", "seed": 42, "max_steps": 1},
        "task": {
            "name": "demo_shapes",
            "params": {"shapes": ["square"], "image_dir": str(tmp_path / "images")},
        },
        "protocol": {"name": "identity"},
        "backend": {"name": "vllm", "params": params},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {"name": "ape", "params": {"demos_per_prompt": demos}},
    }


@pytest.mark.parametrize("image_limit", [1, None], ids=["explicit-limit-1", "vllm-default-1"])
def test_runtime_vllm_image_limit_rejects_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_limit: int | None
):
    # (c) The full config path: vllm declaration -> wrapper chain -> assembly.
    # Any engine launch before the verdict is a test failure, not a side effect.
    def _no_launch(**kwargs: Any) -> Any:
        raise AssertionError("engine launched before assembly validation")

    monkeypatch.setattr("eyas.backends.vllm.launch_vllm_engine", _no_launch)
    with pytest.raises(ValueError, match="exceeding the backend limit of 1"):
        build_context_from_config(_vllm_cfg(tmp_path, image_limit, demos=2))


def test_runtime_vllm_config_within_limit_assembles(tmp_path: Path):
    # Positive control: the same config path with a sufficient image limit
    # assembles cleanly (the rejection above is the intersection, not a
    # broken vllm config).
    ctx = build_context_from_config(_vllm_cfg(tmp_path, image_limit=2, demos=2))
    try:
        assert ctx.primary_objective.key == "accuracy"
    finally:
        ctx.evaluator.backend.close()
