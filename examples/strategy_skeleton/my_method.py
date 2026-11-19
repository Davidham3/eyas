"""A complete optimization method in one file.

The framework's job is the invariant-shaped part of a run (budget, evaluation,
cost accounting, checkpointing, the loop itself). Your method's algorithmic
identity is the couple of hundred bytes of policy inside :meth:`step` — what to
look at, what to try next, when to stop.

REPLACE ME — this file is a template. To start a new method, copy the directory
and rename:

* :class:`RandomSearchStrategy` + its ``@register_strategy("random_search")``
  name (that name is what ``strategy.name`` in a YAML config selects)
* :class:`RandomSearchConfig` — your knobs; the config's ``params:`` mapping is
  bound to it by keyword, so names must match your YAML
* :meth:`RandomSearchStrategy.step` — the search policy (the algorithm)
* the registration name inside ``demo_config.yaml`` / ``run_config.yaml``, and
  the toy task + assertions in ``my_method_test.py``

Nothing else is framework-facing: the framework only ever reads the four class
attributes below and calls :meth:`init` / :meth:`step`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf
from random import Random
from typing import ClassVar

from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, PromptListState
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy

# The vocabulary this method edits with (replace it with your own mutations).
_SUFFIXES = ("Think step by step.", "Answer with one word.", "Double-check the image.")
_PREFIXES = ("You are a careful classifier.", "Answer from the image alone.")


@dataclass(kw_only=True)
class RandomSearchConfig:
    """Typed knobs; the run entry builds this from the strategy's ``params:``
    (by keyword — the contract's ``RandomSearchConfig(*, ...)``)."""

    #: candidates proposed per round
    neighborhood_size: int = 3
    #: chance of also prepending a lead-in sentence to a candidate
    edit_probability: float = 0.3
    #: offsets the run's randomness stream (the loop seeds it from ``run.seed``)
    seed: int = 0
    #: mutation rounds after the seeds are scored, then the method finishes
    max_rounds: int = 3


@register_strategy("random_search")
class RandomSearchStrategy(Strategy):
    """Score what is unevaluated, then propose mutations of the best, stop early."""

    # The declaration set the framework reads before running anything. The
    # registration decorator overwrites `name` with the registry key.
    name: ClassVar[str] = "random_search"
    config_cls: ClassVar[type] = RandomSearchConfig
    # state the loop checkpoints every round: PromptListState = candidates +
    # round counter, which is enough for search-style methods
    state_cls: ClassVar[type[OptimizerState]] = PromptListState
    # The evaluation semantics this method needs. ABSOLUTE_EVAL ("I rank by
    # absolute scores") is also CapSpec's default; a compare-based method declares
    # COMPARE_EVAL instead, and an incompatible task/eval-model pairing then fails
    # at startup, before any GPU time is spent.
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="random_search", requires_eval=frozenset({ABSOLUTE_EVAL})
    )

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        """First thing the loop asks: turn the seed prompts into typed state."""
        return PromptListState(candidates=list(seeds), round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        """One decision. The loop runs whatever ``evaluate`` asks for and feeds
        the results back into ``view`` on the next call."""
        cfg = self.config
        assert isinstance(state, PromptListState)  # narrows state_cls for the type checker

        # 1. Score everything the archive has never seen. Nothing is re-scored:
        #    the record store answers cached samples, and dedup is free because a
        #    Prompt's identity is its normalized text (see the README's facts).
        unevaluated = [p for p in state.candidates if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))

        # 2. Rank what we have. Unscored prompts read as None — treat them worst.
        scored = [(p, view.score(p.id)) for p in state.candidates]
        best = max(scored, key=lambda item: -inf if item[1] is None else item[1])[0]
        if state.round_idx >= cfg.max_rounds:
            return Decision("finish", best_id=best.id)  # the run's last decision
        state.round_idx += 1

        # 3. Propose a neighborhood around the best and keep it for the next
        #    round's evaluate. ``propose`` is informational: the loop only
        #    evaluates what a later ``evaluate`` decision names.
        rng = self._round_rng()
        offspring: list[Prompt] = []
        for i in range(cfg.neighborhood_size):
            child = self._mutate(best, i, state.round_idx, rng)
            if not view.evaluated(child):  # never queue a prompt the archive has
                offspring.append(child)
        if not offspring:
            return Decision("finish", best_id=best.id)  # the neighborhood is exhausted
        state.candidates = [best, *offspring]
        return Decision("propose", prompts=tuple(offspring))

    def _round_rng(self) -> Random:
        """A child RNG from the run's stream — resume-reproducible by contract
        (``derived_rng`` advances with ``ctx.rng``), offset by ``config.seed``."""
        rng = self.derived_rng()
        return Random(rng.randrange(2**64) + self.config.seed)

    def _mutate(self, parent: Prompt, index: int, round_idx: int, rng: Random) -> Prompt:
        """One candidate: always a suffix, sometimes a lead-in on top of it.

        The suffix enumeration is deterministic (candidate ``i`` takes suffix
        ``i % len``) and only the lead-in is random, so any neighborhood covers the
        whole suffix vocabulary before it repeats. That is what lets the contract
        test assert an *improvement* instead of a lucky draw — keep your own
        mutation rule equally explorable, or your test can only assert that
        something ran.
        """
        text = f"{parent.text} {_SUFFIXES[index % len(_SUFFIXES)]}"
        if rng.random() < self.config.edit_probability:
            text = f"{_PREFIXES[index % len(_PREFIXES)]} {text}"
        # metadata is display-only (it labels the prompt in reports); it never
        # takes part in identity — two prompts of the same text are the same prompt
        return Prompt(text, metadata={"label": f"r{round_idx}:{index}"})
