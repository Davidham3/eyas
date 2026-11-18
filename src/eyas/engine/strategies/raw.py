"""raw: degenerate method — a single candidate, one evaluation, no propose/select."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, PromptListState
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy, evaluate_then_finish


@dataclass
class RawConfig:
    pass


@register_strategy("raw")
class RawStrategy(Strategy):
    """Metrics follow the final_eval convention; the main loop only evaluates the seed
    once."""

    config_cls: ClassVar[type] = RawConfig
    state_cls: ClassVar[type] = PromptListState
    capabilities: ClassVar[CapSpec] = CapSpec(name="raw", requires_eval=frozenset({ABSOLUTE_EVAL}))

    def expected_rounds(self) -> int | None:
        return 1  # one evaluate round over the pool, then finish

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        if not seeds:
            raise ValueError(
                "raw needs at least one seed prompt: set strategy.seeds to a "
                "non-empty list, or leave it unset so the runtime falls back to "
                "task.default_prompt"
            )
        return PromptListState(candidates=[seeds[0]], round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        return evaluate_then_finish(state, view)
