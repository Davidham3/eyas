"""evoprompt: population evolution (GA roulette + DE position-aligned greedy).

Faithful to EvoPrompt (arXiv:2309.08532, ICLR 2024) and the official templates:

* **GA** (§3.2, Algorithm 2): parents come from fitness-proportional **roulette
  wheel** selection by default (paper Table 4: wheel beats tournament/random;
  ``sel_mode="tournament"`` exposes the ablation alternative), N children per
  generation, one ``GA_TEMPLATE`` call per child (Step 1 merge/combine-core-ideas
  = crossover, Step 2 adjust-randomly = mutation); update merges population +
  trials and keeps the top-N (elitist).
* **DE** (§3.3, Algorithm 3): per population slot the operator sees FOUR prompts
  — the base, two donors drawn r1≠r2≠base, and the cross-round strict-improvement
  champion ``cur_best`` in the template's "Prompt 3" combine slot — instantiated
  into ``DE_TEMPLATE``'s four steps (identify the different parts → mutate them →
  combine with Prompt 3 → crossover with the basic prompt); update is per-position
  greedy (trial ≥ old → replace).
* **Initial population** (§3.1): the full seed/artifact pool is evaluated
  FIRST, and only then truncated to the top-``popsize`` by score. The shipped
  GA cells pass the official N_P=10 pool through ``strategy.seeds`` (10 curated
  dataset prompts; bench inherits the quickstart pool verbatim); GA with an
  assembled pool of fewer than 2 distinct prompts is structurally unable to
  cross and raises at ``init`` — never at the first GA round after engine
  startup.

Wheel mechanics: the first parent is drawn exact p_i = s_i / Σ_j s_j; the second
parent is drawn from the remaining members under the same fitness proportions (the
wheel minus the drawn share). The official ``rng.choices`` with
replacement additionally emits same-parent twins — a registered deviation.
Negative fitness is clamped to 0 before the wheel; Σ(clamped fitness) = 0 has no
proportions → a uniform wheel. Tournament (the ablation knob)
runs two independent 2-way tournaments and CAN pair a member with itself — the
observable difference from the wheel.

Template semantics follow the published operator templates (carried verbatim
from the official repo); the wording here is the framework's own
English rendering of that semantics as module constants, and the final-answer
tag follows the framework output protocol (``###OUT###``/``###END###``, extracted
through ``genio.extract_candidates(..., SINGLE_TAG)``) instead of the official
``<prompt>`` tags — one candidate per call, the whole body kept, newlines intact.

Sampling tiers stay the framework decision (the paper's 0.5/0.95 is overridable
via ``sampling_profiles``); parent/donor draws go through ``derived_rng()`` so an
interrupted run replays them identically after ``--resume`` — which is load-bearing
together with the full state serialization (``trials``/``cur_best`` included): the
loop checkpoints after each completed round, and a mid-phase resume re-enters the
``eval_trials`` phase from the serialized trials + champion instead of crashing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any, ClassVar, Literal

from eyas.backends.base import ChatRequest, system_message, user_message
from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, rank_by_score
from eyas.engine.genio import GEN_MARKER, OUT_END, OUT_START, SINGLE_TAG, extract_candidates
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, RunContext, Strategy

Phase = Literal["seed_eval", "gen", "eval_trials", "done"]


@dataclass
class EvopromptConfig:
    popsize: int = 4
    rounds: int = 2
    mode: Literal["ga", "de"] = "de"
    #: GA parent selection: "roulette" is the paper default (§3.2, Table 4);
    #: "tournament" is the ablation alternative kept as a knob.
    sel_mode: Literal["roulette", "tournament"] = "roulette"


@dataclass
class EvoState(OptimizerState):
    population: list[Prompt] = field(default_factory=list)
    trials: list[Prompt] = field(default_factory=list)
    cur_best: Prompt | None = None
    cur_best_score: float | None = None
    round_idx: int = 0
    phase: Phase = "seed_eval"

    def to_json(self) -> dict[str, Any]:
        return {
            "population": [p.text for p in self.population],
            "trials": [p.text for p in self.trials],
            "cur_best": None if self.cur_best is None else self.cur_best.text,
            "cur_best_score": self.cur_best_score,
            "round_idx": self.round_idx,
            "phase": self.phase,
        }

    @classmethod
    def from_json(cls, data: Any) -> EvoState:
        cur_best_text = data.get("cur_best")
        return cls(
            population=[Prompt(t) for t in data.get("population", [])],
            trials=[Prompt(t) for t in data.get("trials", [])],
            cur_best=None if cur_best_text is None else Prompt(cur_best_text),
            cur_best_score=data.get("cur_best_score"),
            round_idx=int(data.get("round_idx", 0)),
            phase=str(data.get("phase", "seed_eval")),  # type: ignore[arg-type]
        )


# --- GA variation operator (§3.2: crossover THEN mutation, one call, one-shot) ---

#: GA Step 1 — the crossover half: merge both parents into one prompt by combining
#: their core ideas (the published template's one-example two-step merge).
GA_STEP1_INSTRUCTION = (
    "Step 1 (crossover): Merge the two prompts above and combine their core "
    "ideas into a new prompt:"
)

#: GA Step 2 — the mutation half: randomly adjust the Step-1 prompt, answer in
#: the framework output tags.
GA_STEP2_INSTRUCTION = (
    f"Step 2 (mutation): Adjust the prompt generated in Step 1 randomly and "
    f"generate a final instruction between {OUT_START} and {OUT_END}:"
)

#: One-shot example of the GA execution (paper Appendix B.2: "we prepend a
#: one-shot example of the algorithm execution, guiding LLMs to operate precisely").
GA_EXAMPLE = (
    "Here is an example:\n"
    "Prompt 1: Rewrite the input text into simpler text.\n"
    "Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.\n"
    f"{GA_STEP1_INSTRUCTION}\n"
    "Merged prompt: Rewrite the complex sentence in simpler terms, while keeping "
    "the meaning.\n"
    f"{GA_STEP2_INSTRUCTION}\n"
    f"{OUT_START}Reformulate the given sentence into simpler language while "
    f"preserving its core meaning.{OUT_END}\n"
)

#: GA variation operator: one call, ``<prompt1>``/``<prompt2>`` = the two parents.
#: The step lines are stated twice (official structure): once right after the
#: parents (before the worked example, where "above" refers to them) and once
#: after the example, before the final answer tags.
GA_TEMPLATE = (
    "Please follow the instruction step-by-step to generate a better prompt.\n"
    "\n"
    f"{GA_EXAMPLE}\n"
    "Now follow the same two steps for the following two prompts:\n"
    "Prompt 1: <prompt1>\n"
    "Prompt 2: <prompt2>\n"
    f"{GA_STEP1_INSTRUCTION}\n"
    f"{GA_STEP2_INSTRUCTION}\n"
    f"{OUT_START}\n"
)

# --- DE variation operator (§3.3: y = a + F(b−c) then x ⊗ y, as four text steps) ---

#: DE Step 1 — the difference vector F(b−c) in text: locate what differs between
#: the two donor prompts.
DE_STEP1_INSTRUCTION = "Step 1: Identify the different parts between the Prompt 1 and Prompt 2:"

#: DE Step 2 — the mutation F applied to the difference parts.
DE_STEP2_INSTRUCTION = "Step 2: Randomly mutate the different parts."

#: DE Step 3 — y = a + F(b−c): combine the mutated parts with "Prompt 3", which
#: the official default fills with the current BEST prompt.
DE_STEP3_INSTRUCTION = (
    "Step 3: Combine the different parts with Prompt 3, selectively replace it "
    "with the different parts in Step 2 and generate a new prompt."
)

#: DE Step 4 — the crossover x ⊗ y against the base (basic) prompt.
DE_STEP4_INSTRUCTION = (
    f"Step 4: Crossover the prompt in Step 3 with the following basic prompt and "
    f"generate a final instruction between {OUT_START} and {OUT_END}:"
)

#: One-shot example of the DE execution, reusing the Step constants.
DE_EXAMPLE = (
    "Here is an example:\n"
    "Prompt 1: Rewrite the input text into simpler text.\n"
    "Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.\n"
    f"{DE_STEP1_INSTRUCTION}\n"
    'Different parts: "the input text" vs "my complex sentence"; "into simpler '
    'text" vs "in simpler terms, but keep the meaning".\n'
    f"{DE_STEP2_INSTRUCTION}\n"
    '"the input text" -> "the given sentence"; "in simpler terms, but keep the '
    'meaning" -> "in plainer words, preserving the meaning".\n'
    f"{DE_STEP3_INSTRUCTION}\n"
    "Prompt 3: Simplify the wording of a sentence while preserving its meaning.\n"
    "New prompt: Rephrase the given sentence in plainer words, preserving the "
    "meaning.\n"
    f"{DE_STEP4_INSTRUCTION}\n"
    "Basic Prompt: Make sentences easier to read.\n"
    f"{OUT_START}Rewrite the sentence in plainer words so it reads more easily, "
    f"while keeping its meaning.{OUT_END}\n"
)

#: DE variation operator: one call; ``<prompt1>``/``<prompt2>`` = the donors,
#: ``<prompt3>`` = the champion, ``<prompt0>`` = the base (basic) prompt.
DE_TEMPLATE = (
    "Please follow the instruction step-by-step to generate a better prompt.\n"
    "\n"
    f"{DE_EXAMPLE}\n"
    "Now follow the same four steps for the following prompts:\n"
    f"{DE_STEP1_INSTRUCTION}\n"
    "Prompt 1: <prompt1>\n"
    "Prompt 2: <prompt2>\n"
    f"{DE_STEP2_INSTRUCTION}\n"
    f"{DE_STEP3_INSTRUCTION}\n"
    "Prompt 3: <prompt3>\n"
    f"{DE_STEP4_INSTRUCTION}\n"
    "Basic Prompt: <prompt0>\n"
    f"{OUT_START}\n"
)


@register_strategy("evoprompt")
class EvopromptStrategy(Strategy):
    config_cls: ClassVar[type] = EvopromptConfig
    state_cls: ClassVar[type] = EvoState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="evoprompt", requires_eval=frozenset({ABSOLUTE_EVAL})
    )

    def __init__(self, config: Any, ctx: RunContext) -> None:
        # popsize < 2 is structurally infeasible in BOTH modes (GA needs two
        # distinct parents to cross; DE needs donors to difference) — reject
        # at construction, the earliest point the config is knowable, never
        # at the first round after engine startup.
        if config.popsize < 2:
            raise ValueError(
                f"evoprompt popsize must be >= 2, got {config.popsize}; "
                "set strategy.params.popsize to at least 2 (parent crossing / "
                "DE differencing need two population members)"
            )
        super().__init__(config, ctx)

    def _rank(self, view: EvalView, prompts: Sequence[Prompt]) -> list[Prompt]:
        return rank_by_score(prompts, view.score)

    @staticmethod
    def _at_least(view: EvalView, a: Prompt, b: Prompt) -> bool:
        """``a`` scores no worse than ``b`` — the pairwise form of rank_by_score's
        ordering (a never-evaluated prompt scores worse than any real number)."""
        score_a, score_b = view.score(a.id), view.score(b.id)
        if score_a is None or score_b is None:
            return score_b is None
        return score_a >= score_b

    def _beats_cur_best(self, view: EvalView, prompt: Prompt, state: EvoState) -> bool:
        """Strict-improvement champion update (the recorded DE/GA semantics: full
        ties keep the incumbent champion); a never-evaluated prompt never wins."""
        score = view.score(prompt.id)
        cur = state.cur_best_score
        return score is not None and (cur is None or score > cur)

    @staticmethod
    def _fitness(view: EvalView, population: Sequence[Prompt]) -> list[float]:
        """Dev-split scores; a never-evaluated prompt carries zero fitness (the
        wheel's p_i = s_i / Σ_j s_j has no share for a prompt without a score).
        The None check is explicit — a real 0.0 score and a missing score both
        carry zero share, but truthiness must not decide (0.0 is a score)."""
        out: list[float] = []
        for p in population:
            score = view.score(p.id)
            out.append(0.0 if score is None else score)
        return out

    def _ga_parents(
        self, population: Sequence[Prompt], fitness: Sequence[float], rng: Random
    ) -> tuple[Prompt, Prompt]:
        """Two parents per child (Algorithm 2). Roulette (paper default): the first
        parent is exact p_i = s_i / Σ_j s_j; the second is drawn from the remaining
        members under the same fitness proportions (the wheel minus the drawn share)
        — two DISTINCT parents, every marginal fitness-proportional to first order.
        The official ``rng.choices`` with replacement
        additionally emits same-parent twins — a registered deviation. Negative
        fitness (oriented minimize-objective scores) is clamped to 0 BEFORE the
        wheel — a raw negative weight makes cum_weights non-monotonic and
        silently starves the positive members. Σ(clamped fitness) = 0 has no
        proportions → a uniform wheel. Tournament (ablation knob): two
        independent 2-way tournaments (higher fitness wins, ties take the first
        draw) — a member CAN pair with itself, the observable difference from the
        wheel."""
        n = len(population)
        if n < 2:
            raise ValueError("evoprompt GA needs population >= 2 for parent crossing")
        pool = list(range(n))
        if self.config.sel_mode == "roulette":
            clamped = [max(f, 0.0) for f in fitness]
            total = sum(clamped)
            weights = [f / total for f in clamped] if total > 0 else None
            first = rng.choices(pool, weights=weights, k=1)[0]
            rest = [i for i in pool if i != first]
            rest_weights = [clamped[i] for i in rest]
            if sum(rest_weights) > 0:
                second = rng.choices(rest, weights=rest_weights, k=1)[0]
            else:
                second = rng.choice(rest)
            return population[first], population[second]
        if self.config.sel_mode != "tournament":
            raise ValueError(f"unknown evoprompt sel_mode: {self.config.sel_mode!r}")

        def tour() -> int:
            draw_a, draw_b = rng.sample(pool, 2)
            return draw_a if fitness[draw_a] >= fitness[draw_b] else draw_b

        return population[tour()], population[tour()]

    def _de_donors(
        self, population: Sequence[Prompt], base_idx: int, rng: Random
    ) -> tuple[Prompt, Prompt]:
        """The two differencing donors for slot ``base_idx``: distinct from each
        other and from the base (Algorithm 3: r1 ≠ r2 ≠ i). A population whose
        non-base members are fewer than two pads the twin slots with what exists
        (documented deviation — the official code hard-requires popsize ≥ 4; the
        pad keeps the per-slot trial shape when the assembled pool is degenerate;
        popsize itself is guarded >= 2 at construction)."""
        base = population[base_idx]
        others = [p for i, p in enumerate(population) if i != base_idx]
        if len(others) >= 2:
            first, second = rng.sample(others, 2)
            return first, second
        pad = others[0] if others else base
        return pad, pad

    @staticmethod
    def _render(template: str, slots: Mapping[str, str]) -> ChatRequest:
        content = template
        for name, value in slots.items():
            content = content.replace(f"<{name}>", value)
        system = (
            f"{GEN_MARKER} Follow the template step by step and finish with the "
            f"final instruction between {OUT_START} and {OUT_END}."
        )
        return ChatRequest(messages=(system_message(system), user_message(content)))

    def _ga_request(self, parent_a: Prompt, parent_b: Prompt) -> ChatRequest:
        return self._render(GA_TEMPLATE, {"prompt1": parent_a.text, "prompt2": parent_b.text})

    def _de_request(
        self, donor_a: Prompt, donor_b: Prompt, base: Prompt, best: Prompt
    ) -> ChatRequest:
        # <prompt1>/<prompt2> = differencing donors, <prompt3> = the champion
        # (the official donor_random=False default), <prompt0> = the base
        return self._render(
            DE_TEMPLATE,
            {
                "prompt1": donor_a.text,
                "prompt2": donor_b.text,
                "prompt3": best.text,
                "prompt0": base.text,
            },
        )

    def _variation(self, state: EvoState, view: EvalView) -> list[Prompt]:
        """One generation of offspring: GA draws N roulette parent pairs, DE builds
        one four-slot request per position (the champion fills Prompt 3, falling
        back to the population's best before any cross-round champion exists)."""
        cfg = self.config
        population = state.population
        if not population:
            return []
        rng = self.derived_rng()
        requests: list[ChatRequest] = []
        if cfg.mode == "ga":
            fitness = self._fitness(view, population)
            for _ in population:
                parent_a, parent_b = self._ga_parents(population, fitness, rng)
                requests.append(self._ga_request(parent_a, parent_b))
        else:
            champion = state.cur_best or self._rank(view, population)[0]
            for i, base in enumerate(population):
                donor_a, donor_b = self._de_donors(population, i, rng)
                requests.append(self._de_request(donor_a, donor_b, base, champion))
        raws = self.generate(requests, "evolve")
        trials: list[Prompt] = []
        for raw, member in zip(raws, population, strict=True):
            candidates = extract_candidates(raw, SINGLE_TAG)
            instruction = candidates[0] if candidates else ""
            trials.append(Prompt(instruction, metadata={"label": "evo"}) if instruction else member)
        return trials

    def expected_rounds(self) -> int | None:
        return self.config.rounds + 1  # seed round + per-round generate/eval

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        pool = list(seeds)
        # Soft-consume ape.candidate_prompts (§9): merge upstream artifacts into the
        # initial pool when present, replacing the direct json-read path
        consumed = self.ctx.artifacts.get("ape.candidate_prompts")
        if consumed is not None:
            seen = {p.id for p in pool}
            for text in consumed.payload:
                child = Prompt(text, metadata={"label": "ape"})
                if child.id not in seen:
                    seen.add(child.id)
                    pool.append(child)
        # The pool is evaluated IN FULL first; truncation to the top-popsize
        # happens in seed_eval once the scores exist. Truncating before
        # evaluation would make the rank/truncate step dead code.
        if self.config.mode == "ga":
            # Assembly-time fail-fast: GA parent crossing needs two DISTINCT
            # parents, so an initial pool of fewer than two distinct prompts is
            # structurally infeasible — it must raise at INIT, not after engine
            # startup at the first GA round. DE is exempt: _de_donors documents
            # the pad for degenerate pops.
            distinct = len({p.id for p in pool})
            if distinct < 2:
                raise ValueError(
                    f"evoprompt GA needs population >= 2 for parent crossing, "
                    f"but the initial pool assembled {distinct} distinct prompt(s) "
                    f"(popsize={self.config.popsize} cannot be crossed); configure "
                    "strategy.seeds with >= 2 distinct prompts — the single "
                    "task.default_prompt fallback cannot reach a population of 2"
                )
        return EvoState(population=pool, phase="seed_eval")

    def _start_round(self, state: EvoState, view: EvalView) -> Decision:
        """Produce trials and emit the evaluate decision (shared by gen and the next
        round after eval_trials). When generation re-manufactured twins of prompts
        the archive already scored, the round completes in place — the scores exist,
        and re-emitting evaluate would only re-issue work the store answers from cache."""
        state.trials = self._variation(state, view)
        state.phase = "eval_trials"
        unevaluated = [t for t in state.trials if not view.evaluated(t)]
        if not unevaluated:
            return self._complete_round(state, view)
        return Decision("evaluate", prompts=tuple(unevaluated))

    def _complete_round(self, state: EvoState, view: EvalView) -> Decision:
        """Apply the round's update (DE per-position greedy / GA elitist top-N),
        advance the round counter, and start the next round or finish."""
        cfg = self.config
        if cfg.mode == "de":
            for i, (old, trial) in enumerate(zip(state.population, state.trials, strict=True)):
                if self._at_least(view, trial, old):
                    state.population[i] = trial
                    if self._beats_cur_best(view, trial, state):
                        state.cur_best, state.cur_best_score = trial, view.score(trial.id)
        else:  # ga top-k: pool = pop ∪ trials, dedup, sort desc, truncate
            pool = {p.id: p for p in state.population + state.trials}
            ranked = self._rank(view, list(pool.values()))
            state.population = ranked[: cfg.popsize]
            top = ranked[0]
            if self._beats_cur_best(view, top, state):
                state.cur_best, state.cur_best_score = top, view.score(top.id)
        state.round_idx += 1
        if state.round_idx >= cfg.rounds:
            state.phase = "done"
            return self._finish(state)
        return self._start_round(state, view)

    @staticmethod
    def _finish(state: EvoState) -> Decision:
        best = state.cur_best or state.population[0]
        return Decision("finish", best_id=best.id)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, EvoState)

        if state.phase == "seed_eval":
            unevaluated = [p for p in state.population if not view.evaluated(p)]
            if unevaluated:
                return Decision("evaluate", prompts=tuple(unevaluated))
            ranked = self._rank(view, state.population)
            state.population = ranked[: self.config.popsize]
            state.cur_best = state.population[0]
            state.cur_best_score = view.score(state.cur_best.id)
            return self._start_round(state, view)

        if state.phase == "eval_trials":
            return self._complete_round(state, view)

        return self._finish(state)
