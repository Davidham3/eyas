"""protegi: beam search + textual gradients + paraphrase expansion + bandit selection.

Paper contract (arXiv:2305.03495, EMNLP 2023, Algorithm 2, official microsoft/LMOps
prompt_optimization): each parent's expansion is a **two-call dialogue** — a
critique request (∇) asks for ``n_gradients`` tagged "reasons why" a randomly
sampled group of ``n_errors_per_grad`` errors was mispredicted, and every parsed
gradient feeds one edit request (δ) that revises the prompt "using this
feedback"; each new candidate then fans out ``mc_samples`` paraphrases (LLM_mc)
on the dedicated ``synonym`` tier. The successor pool holds children only,
capped at ``max_expansion_factor`` per parent; the parents re-enter the
selection pool from the beam (mirroring the official ``new_prompts +=
prompts``). Selection is bandit best-arm identification: with an allocator
that exposes a ranking (``BanditBudget.rank()`` — SelectTop_b(Q̂)) the beam
reorders by the bandit's pooled means — arm positions are mapped back to
prompts through the committed arm identity (:attr:`ProtegiState.arm_order`,
the insertion order last proposed to the allocator, never a positional index
into the mixed ``beam + offspring`` pool), and carryover beam members rank
behind every ranked arm. Without a ranking allocator (flat/even-split) it falls
back to ``rank_by_score`` over archive scores — the paper's "Unif" uniform
comparator, kept only as an opt-in.
Engineering deviations: failing samples are read directly from durable
``SampleOutcome`` records instead of a fresh 64-sample minibatch (documented),
and offspring carry a Lineage foreign key (parent attribution).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import ceil
from random import Random
from typing import Any, ClassVar, Literal

from eyas.backends.base import ChatRequest, system_message, user_message
from eyas.engine.budget import BanditBudget, BudgetAllocator
from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, SampleOutcome, rank_by_score
from eyas.engine.genio import (
    FEEDBACK_END,
    FEEDBACK_START,
    GEN_MARKER,
    OUT_END,
    OUT_START,
    SINGLE_TAG,
    clean_instruction,
    extract_candidates,
    parse_gradients,
    render_sample_reference,
)
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy

Phase = Literal["baseline", "expand", "score", "final", "done"]

# Official template semantics (paper appendix A.1 / optimizers.py), framework tags.
# The template IS the full task payload: it rides the USER message (peers' role split:
# system carries the GEN_MARKER plumbing, user the model-facing task text — chat
# templates that require a user turn reject a system-only request).
CRITIQUE_TEMPLATE = (
    'I am writing a classifier prompt. My current prompt is: "{prompt}" '
    "But this prompt gets the following examples wrong: {error_string} "
    "Give {num_feedbacks} reasons why the prompt could have gotten these examples wrong. "
    "Wrap each reason with {start} and {end}."
)
EDIT_TEMPLATE = (
    'I am writing a classifier prompt. My current prompt is: "{prompt}" '
    "Based on these examples the problem with this prompt is that {feedback} "
    "Please revise the prompt using this feedback: {feedback}, "
    "in the direction OPPOSITE to the flaw above. "
    "Output the improved prompt between {out_start} and {out_end}."
)
SYNONYM_TEMPLATE = (
    "Paraphrase the instruction below: generate a variation of the following "
    "instruction while keeping the semantic meaning. Output the variation between "
    "{out_start} and {out_end}. Instruction: {instruction}"
)


@dataclass
class ProtegiConfig:
    rounds: int = 1
    #: the paper's single beam parameter b: expansion breadth AND SelectTop_b
    #: truncation share it (no separate keep_size knob —
    #: official ProTeGi has exactly one b, carried here by beam_size)
    beam_size: int = 2
    #: critique reasons requested per error group (paper n_gradients)
    n_gradients: int = 1
    #: errors per randomly-sampled critique group (paper --errors_per_gradient, §3.2: 4)
    n_errors_per_grad: int = 4
    #: δ edits per gradient (paper --steps_per_gradient)
    mc_samples: int = 2
    #: max successor candidates per parent before selection (paper --max_expansion_factor)
    max_expansion_factor: int = 8


@dataclass
class ProtegiState(OptimizerState):
    beam: list[Prompt] = field(default_factory=list)
    offspring: list[Prompt] = field(default_factory=list)
    #: arm identities committed to the allocator: insertion order of the
    #: prompts last proposed to a scoring evaluation round — the sole
    #: coordinate system ``BudgetAllocator.rank()`` positions address
    #: (identity-keyed, so a rebound arm basis can never misaddress prompts)
    arm_order: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    round_idx: int = 0
    phase: Phase = "baseline"

    def to_json(self) -> dict[str, Any]:
        return {
            "beam": [p.text for p in self.beam],
            "offspring": [p.text for p in self.offspring],
            "arm_order": list(self.arm_order),
            "seen": sorted(self.seen),
            "round_idx": self.round_idx,
            "phase": self.phase,
        }

    @classmethod
    def from_json(cls, data: Any) -> ProtegiState:
        return cls(
            beam=[Prompt(t) for t in data.get("beam", [])],
            offspring=[Prompt(t) for t in data.get("offspring", [])],
            arm_order=list(data.get("arm_order", [])),
            seen=set(data.get("seen", [])),
            round_idx=int(data.get("round_idx", 0)),
            phase=str(data.get("phase", "baseline")),  # type: ignore[arg-type]
        )


@register_strategy("protegi")
class ProtegiStrategy(Strategy):
    config_cls: ClassVar[type] = ProtegiConfig
    state_cls: ClassVar[type] = ProtegiState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="protegi", requires_eval=frozenset({ABSOLUTE_EVAL})
    )

    def _labels_by_sample(self) -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        for sample in self.ctx.task.load_split(self.ctx.eval_split):
            out[sample.id] = (
                str(sample.inputs.get("question", "")),
                render_sample_reference(sample.references),
            )
        return out

    def _error_group(
        self, view: EvalView, parent: Prompt, labels: dict[str, tuple[str, str]]
    ) -> list[tuple[str, str, str]]:
        """The parent's full misprediction pool (paper ``_sample_error_str`` reads the
        minibatch errors; the engine reads the durable record). Head slicing would replay the
        same error window every round, so the critique group is SAMPLED downstream."""
        errors: list[tuple[str, str, str]] = []
        for outcome in view.per_sample(parent.id):
            if not isinstance(outcome, SampleOutcome):
                continue
            if outcome.prediction is None or outcome.correct:
                continue
            q, gold = labels.get(outcome.sample_id, ("", ""))
            if str(outcome.prediction) != gold:
                errors.append((q, str(outcome.prediction), gold))
        return errors

    def _sample_group(
        self, errors: Sequence[tuple[str, str, str]], rng: Random
    ) -> list[tuple[str, str, str]]:
        """Official ``random.sample(error_idxs, min(len(error_idxs), n))`` — the
        per-group draw is seeded through ``derived_rng()``, never a head slice."""
        if not errors:
            return []
        return rng.sample(list(errors), min(self.config.n_errors_per_grad, len(errors)))

    def _critique_request(
        self, parent: Prompt, group: Sequence[tuple[str, str, str]], n_gradients: int
    ) -> ChatRequest:
        error_string = "\n".join(
            f"- Text: {q} / Label: {gold} / Prediction: {pred}" for q, pred, gold in group
        )
        task = CRITIQUE_TEMPLATE.format(
            prompt=parent.text,
            error_string=error_string,
            num_feedbacks=n_gradients,
            start=FEEDBACK_START,
            end=FEEDBACK_END,
        )
        return ChatRequest(messages=(system_message(GEN_MARKER), user_message(task)))

    def _edit_request(self, parent: Prompt, gradient: str) -> ChatRequest:
        task = EDIT_TEMPLATE.format(
            prompt=parent.text, feedback=gradient, out_start=OUT_START, out_end=OUT_END
        )
        return ChatRequest(messages=(system_message(GEN_MARKER), user_message(task)))

    def _synonym_request(self, instruction: str) -> ChatRequest:
        task = SYNONYM_TEMPLATE.format(
            instruction=instruction, out_start=OUT_START, out_end=OUT_END
        )
        return ChatRequest(messages=(system_message(GEN_MARKER), user_message(task)))

    def _add_candidate(self, state: ProtegiState, pool: list[Prompt], text: str) -> None:
        """Dedup insert into the successor pool: the ``clean_instruction``'d text
        enters as a fresh Prompt unless it cleaned away empty, was already seen,
        or is already in the pool (the pool is mutated in place)."""
        candidate = clean_instruction(text)
        if not candidate:
            return
        child = Prompt(candidate, metadata={"label": "protegi"})
        if child.id not in state.seen and all(child.id != p.id for p in pool):
            state.seen.add(child.id)
            pool.append(child)

    def _n_groups(self, n_errors: int) -> int:
        """``n_groups = min(cfg.n_gradients, ceil(errors / group))`` per parent."""
        return min(
            self.config.n_gradients,
            max(1, ceil(n_errors / max(1, self.config.n_errors_per_grad))),
        )

    def _critique_phase(
        self, view: EvalView, state: ProtegiState, labels: dict[str, tuple[str, str]], rng: Random
    ) -> tuple[list[int], list[str]]:
        """Phase ∇ — one critique dialogue call per parent (n_groups tagged reasons)."""
        n_groups: list[int] = []
        requests: list[ChatRequest] = []
        for parent in state.beam:
            errors = self._error_group(view, parent, labels)
            g = self._n_groups(len(errors))
            n_groups.append(g)
            requests.append(self._critique_request(parent, self._sample_group(errors, rng), g))
        return n_groups, self.generate(requests, "gradient")

    def _edit_phase(
        self,
        state: ProtegiState,
        critiques: Sequence[str],
        n_groups: Sequence[int],
        pool: list[Prompt],
    ) -> list[list[str]]:
        """Phase δ — parse the tagged gradients; one edit per gradient × mc_samples.
        A completion with no parsable <START> reason yields NO candidate (official
        parse_gradients semantics: unparsable feedback contributes nothing).
        The new child texts are inserted into ``pool`` and also returned per parent
        (the mc phase paraphrases them)."""
        children_per_parent: list[list[str]] = []
        for parent, raw, g in zip(state.beam, critiques, n_groups, strict=True):
            gradients = parse_gradients(raw)[:g]
            edit_requests = [
                self._edit_request(parent, gradient)
                for gradient in gradients
                for _ in range(max(1, self.config.mc_samples))
            ]
            children: list[str] = []
            if edit_requests:
                for raw_edit in self.generate(edit_requests, "gradient"):
                    children.extend(extract_candidates(raw_edit, SINGLE_TAG))
            children_per_parent.append(children)
            for text in children:
                self._add_candidate(state, pool, text)
        return children_per_parent

    def _mc_phase(
        self, state: ProtegiState, children_per_parent: Sequence[Sequence[str]], pool: list[Prompt]
    ) -> None:
        """Phase mc (LLM_mc) — paraphrase every new candidate AND the parent sections
        (official loop over ``new_task_sections + [task_section]``). Under the
        deterministic echo mock every paraphrase duplicates its source and dedups
        away, so the pool and the generation stream are stable across attempts —
        the property the resume harness replays on."""
        sources = [p.text for p in state.beam] + [
            text for children in children_per_parent for text in children
        ]
        if sources:
            synonyms = self.generate([self._synonym_request(s) for s in sources], "synonym")
            for raw_synonym in synonyms:
                for body in extract_candidates(raw_synonym, SINGLE_TAG):
                    self._add_candidate(state, pool, body)

    def _cap_pool(self, state: ProtegiState, pool: list[Prompt], rng: Random) -> list[Prompt]:
        """Cap: "randomly sampled max_expansion_factor successor candidates per parent
        prior to bandit selection" (§3.2)."""
        cap = self.config.max_expansion_factor * max(1, len(state.beam))
        if self.config.max_expansion_factor > 0 and len(pool) > cap:
            return rng.sample(pool, cap)
        return pool

    def _expand(self, view: EvalView, state: ProtegiState, rng: Random) -> list[Prompt]:
        """Algorithm 2 + §3.2 fanout: per parent ``n_groups = min(cfg.n_gradients,
        ceil(errors / group))`` sampled critique groups → one gradient per group →
        ``mc_samples`` edits per gradient → paraphrases (LLM_mc) per new candidate and
        per parent. The unique successor pool (children only) is capped at
        ``max_expansion_factor`` per parent before selection; the parents re-enter the
        selection pool from the beam (official ``new_prompts += prompts``). A parent
        without recorded errors still issues the official no-error critique call.

        Orchestrates the paper's three fanout phases (∇ critique → δ edit → mc
        paraphrase) over one shared successor pool, then caps it."""
        labels = self._labels_by_sample()
        pool: list[Prompt] = []
        n_groups, critiques = self._critique_phase(view, state, labels, rng)
        children_per_parent = self._edit_phase(state, critiques, n_groups, pool)
        self._mc_phase(state, children_per_parent, pool)
        return self._cap_pool(state, pool, rng)

    def expected_rounds(self) -> int | None:
        return self.config.rounds + 1  # seed round + per-round generate/eval

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        seed = seeds[0]
        return ProtegiState(beam=[seed], seen={seed.id}, phase="baseline")

    def _baseline_phase(self, state: ProtegiState, view: EvalView) -> Decision | None:
        """Decision: evaluate the seed beam once, then advance to expansion."""
        if not view.evaluated(state.beam[0]):
            return Decision("evaluate", prompts=tuple(state.beam))
        state.phase = "expand"
        return None

    def _select_beam(self, state: ProtegiState, view: EvalView) -> list[Prompt]:
        """SelectTop_b(beam_size) over the mixed offspring + beam pool."""
        ranking = _budget_rank(self.ctx.budget)
        pool_by_id = {p.id: p for p in state.offspring + state.beam}
        if ranking is not None:
            # Q̂ order addresses the committed arm basis (arm_order), never a
            # positional index into the mixed beam + offspring pool. Carryover
            # rule: beam members that are not arms rank behind every ranked
            # arm (SelectTop_b(Q̂) covers the scored pool only); arms that are
            # also in the beam lose their beam slot to their Q̂ position.
            arm_ids = state.arm_order if state.arm_order else [p.id for p in state.offspring]
            arms = [pool_by_id[pid] for pid in arm_ids if pid in pool_by_id]
            ranked = _reorder(arms, ranking)
            ranked.extend(p for p in pool_by_id.values() if all(p.id != q.id for q in ranked))
        else:
            ranked = rank_by_score(list(pool_by_id.values()), view.score)
        return ranked[: self.config.beam_size]

    def _score_phase(self, state: ProtegiState, view: EvalView) -> Decision | None:
        """Decisions: bandit evaluation round on the successor pool, else the
        full-evaluation fallback, else select the beam and advance the round."""
        budget = self.ctx.budget
        # An empty successor pool gives the bandit nothing to pull (official: no new
        # prompts): fall straight through to selection instead of issuing an empty
        # evaluation round per remaining budget step.
        if state.offspring and isinstance(budget, BanditBudget) and not budget.exhausted:
            state.arm_order = [p.id for p in state.offspring]
            return Decision("evaluate", prompts=tuple(state.offspring))
        # Full-evaluation fallback (evaluate all offspring in one go when the budget
        # is not a bandit)
        unevaluated = [p for p in state.offspring if not view.evaluated(p)]
        if unevaluated:
            state.arm_order = [p.id for p in unevaluated]
            return Decision("evaluate", prompts=tuple(unevaluated))
        state.beam = self._select_beam(state, view)
        state.seen.update(p.id for p in state.beam)
        state.round_idx += 1
        state.phase = "final" if state.round_idx >= self.config.rounds else "expand"
        return None

    def _final_phase(self, state: ProtegiState, view: EvalView) -> Decision | None:
        """Decision: one closing evaluate for a carryover beam member that has
        no record yet, then done."""
        unevaluated = [p for p in state.beam if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))
        state.phase = "done"
        return None

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        """Phase machine: walk the baseline → expand → score → final chain from the
        state's current phase. A handler either returns this round's decision or
        advances the phase and lets the chain fall through to the next phase —
        exactly the original if-chain, with each phase's decisions in its own
        handler; once the chain runs out, finish."""
        assert isinstance(state, ProtegiState)
        if state.phase == "baseline":
            decision = self._baseline_phase(state, view)
            if decision is not None:
                return decision
        if state.phase == "expand":
            state.offspring = self._expand(view, state, self.derived_rng())
            state.phase = "score"
        if state.phase == "score":
            decision = self._score_phase(state, view)
            if decision is not None:
                return decision
        if state.phase == "final":
            decision = self._final_phase(state, view)
            if decision is not None:
                return decision
        best = _final_best(state, self.ctx.budget, view)
        return Decision("finish", best_id=best.id)


def _budget_rank(budget: BudgetAllocator) -> list[int] | None:
    """Positions into the allocator's CURRENT arm list (the prompts of its most
    recent ``propose_load``), ordered by the allocator's bandit ranking
    (SelectTop_b(Q̂)); None when the allocator does not rank (the Unif
    comparator takes over). Positions are NOT pool indices: map them through
    ``ProtegiState.arm_order`` — the identities committed at the scoring
    decisions — so a rebound arm basis (e.g. a final-phase carryover eval
    rebinding the arms) can never misaddress prompts."""
    rank: Callable[[], list[int] | None] | None = getattr(budget, "rank", None)
    return rank() if callable(rank) else None


def _reorder(arms: Sequence[Prompt], ranking: list[int]) -> list[Prompt]:
    """Ranking positions over ``arms`` → prompt order (out-of-range positions
    skipped, unreferenced arms keep their relative tail)."""
    items = list(arms)
    ordered = [items[i] for i in ranking if 0 <= i < len(items)]
    ordered.extend(p for p in items if all(p.id != q.id for q in ordered))
    return ordered


def _final_best(state: ProtegiState, budget: BudgetAllocator, view: EvalView) -> Prompt:
    """Finish pick. With an allocator ranking (bandit) the score-phase beam is
    already in Q̂ order — identity-mapped through ``arm_order`` at truncation —
    so the head is the pick; the allocator's arm basis may have been rebound
    since (a final-phase carryover eval re-``_reset``s it to a subset), and
    stale positions must never address the beam. Without a ranking (the Unif
    comparator) the beam is unordered: the archive-score pick, unchanged."""
    if _budget_rank(budget) is not None:
        return state.beam[0]
    return rank_by_score(state.beam, view.score)[0]
