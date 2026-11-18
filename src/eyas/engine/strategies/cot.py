"""cot: the algorithm identity lives in a "rendered variant" (the CoT instruction) rather
than in propose — carried by degenerate evaluation.

Prompting trick source: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models" (arXiv:2201.11903, NeurIPS 2022).

The new abstraction explicitly admits "identity may live in rendering/decoding form":
cot's ``Let's think step by step.`` is carried by the prompt text (or the
task's assistant_prefix), and the strategy remains degenerate evaluate-then-finish.
User-provided seeds are preserved — the CoT suffix is appended to each seed; the
runtime's default-prompt fallback renders through ``base_prompt`` (the shipped cot
cells carry their CoT guidance there).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, PromptListState
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy, evaluate_then_finish


@dataclass
class CotConfig:
    base_prompt: str = "Answer the question about the image."
    cot_suffix: str = "Let's think step by step."


@register_strategy("cot")
class CotStrategy(Strategy):
    config_cls: ClassVar[type] = CotConfig
    state_cls: ClassVar[type] = PromptListState
    capabilities: ClassVar[CapSpec] = CapSpec(name="cot", requires_eval=frozenset({ABSOLUTE_EVAL}))

    def expected_rounds(self) -> int | None:
        return 1  # one evaluate round over the pool, then finish

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        """User-provided seeds are preserved: each renders as itself with the
        CoT suffix appended (the append mechanism), one candidate per seed.
        The runtime fallback shape — ``strategy.seeds`` unset, so the loop
        passes exactly ``[task.default_prompt]`` — renders through
        ``base_prompt`` instead: the shipped cot cells carry their CoT guidance
        there (it may differ from the task default, e.g. sroie/textvqa). Empty
        seeds fail fast with guidance (the runtime fallback normally prevents
        this; a direct library caller must pass a non-empty pool)."""
        if not seeds:
            raise ValueError(
                "cot needs at least one seed prompt: set strategy.seeds to a "
                "non-empty list, or leave it unset so the runtime falls back to "
                "task.default_prompt"
            )
        cfg = self.config
        texts = [s.text for s in seeds]
        if texts == [self.ctx.task.default_prompt]:
            texts = [cfg.base_prompt]
        candidates: list[Prompt] = []
        seen: set[str] = set()
        for text in texts:
            rendered = Prompt(f"{text} {cfg.cot_suffix}".strip())
            if rendered.id not in seen:
                seen.add(rendered.id)
                candidates.append(rendered)
        return PromptListState(candidates=candidates, round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        return evaluate_then_finish(state, view)
