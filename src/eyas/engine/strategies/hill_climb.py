"""hill_climb: pure string-mutation hill climbing — fits the new contract exactly, needs
no generation backend.

Lineage: the classic greedy local search over prompt space — the iterative-refinement
baseline family compared against in APE (arXiv:2211.01910, ICLR 2023) and OPRO
(arXiv:2309.03409, ICLR 2024); no single source paper of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, PromptListState, rank_by_score
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy


@dataclass
class HillClimbConfig:
    rounds: int = 2
    keep_size: int = 2
    per_parent: int = 2
    mutations: tuple[str, ...] = (
        "Think step by step.",
        "Answer with the option letter only.",
        "Be concise and direct.",
    )


@register_strategy("hill_climb")
class HillClimbStrategy(Strategy):
    config_cls: ClassVar[type] = HillClimbConfig
    state_cls: ClassVar[type] = PromptListState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="hill_climb", requires_eval=frozenset({ABSOLUTE_EVAL})
    )

    def expected_rounds(self) -> int | None:
        return self.config.rounds + 1  # seed round + per-round generate/eval

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        return PromptListState(candidates=list(seeds), round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, PromptListState)
        cfg = self.config
        unevaluated = [p for p in state.candidates if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))

        ranked = rank_by_score(state.candidates, view.score)
        if state.round_idx >= cfg.rounds:
            return Decision("finish", best_id=ranked[0].id)
        state.round_idx += 1

        offspring: list[Prompt] = []
        for parent in ranked[: cfg.keep_size]:
            for i in range(cfg.per_parent):
                suffix = cfg.mutations[i % len(cfg.mutations)]
                child = Prompt(
                    f"{parent.text} {suffix}",
                    metadata={"label": f"r{state.round_idx}:{suffix[:16]}"},
                )
                if not view.evaluated(child):
                    offspring.append(child)
        if not offspring:
            return Decision("finish", best_id=ranked[0].id)
        state.candidates = ranked[: cfg.keep_size] + offspring
        return Decision("propose", prompts=tuple(offspring))
