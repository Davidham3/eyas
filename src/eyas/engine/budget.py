"""Budget allocation layer: budget is a first-class citizen with sovereignty.

The budget is an explicit, stateful ``BudgetAllocator`` (propose_load /
observe lifecycle), decoupled from the pure-function selection; because it is
an explicit object, cost becomes predictable, auditable, and profileable, and
methods such as protegi's UCB adaptation receive it instead of hardcoding it.

``LoadPlan`` is expressed as per-candidate ``WorkItem`` entries (which samples
each candidate is evaluated on), natively supporting "which candidate deserves
more budget".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Any, Protocol, runtime_checkable

from eyas.engine.domain import Evaluation, OptimizerState, Prompt

__all__ = [
    "BudgetAllocator",
    "EvenSplitBudget",
    "FlatBudget",
    "LoadPlan",
    "WorkItem",
]


@dataclass(frozen=True)
class WorkItem:
    """The subset of samples one candidate is evaluated on this round. Empty
    ``sample_ids`` = full split."""

    prompt: Prompt
    sample_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadPlan:
    """One round's evaluation load = a tuple of WorkItems."""

    items: tuple[WorkItem, ...]


@runtime_checkable
class BudgetAllocator(Protocol):
    """Stateful budget allocator. propose_load decides what to evaluate;
    observe consumes the results and updates internal state."""

    def propose_load(
        self,
        round_idx: int,
        state: OptimizerState,
        prompts: Sequence[Prompt],
        all_sample_ids: Sequence[str],
    ) -> LoadPlan: ...

    def takeover(self, round_idx: int) -> None:
        """The strategy owns this round's load (``Decision(items=...)`` bypassed
        ``propose_load``): allocators owning arm/counter state discard it
        WHOLESALE (BanditBudget unbinds its arms and zeroes counts/round — the
        next ``propose_load`` re-baselines on the freshly proposed arm set;
        arms are NOT carried across a takeover round), so stale counters never
        survive a strategy-owned round; stateless allocators accept the notice
        as a no-op."""
        ...

    def observe(self, evals: Sequence[Evaluation], rewards: Sequence[float]) -> list[str]:
        """Consume evaluation results + a scalar reward per entry (oriented
        score); return the surviving prompt_ids. Allocators holding arm state
        credit only rounds they scheduled (``BanditBudget`` no-ops once a
        ``takeover`` unbound the arms — the loop observes unconditionally)."""
        ...

    def rank(self) -> list[int] | None:
        """Final selection order over the current arm list (paper: SelectTop_b(Q_T)).
        Bandit allocators return arm positions — positions into the arm list of
        the most recent ``propose_load``, so callers must map them through the
        arm identities they committed (prompt id), never apply them as indices
        into a different prompt list — ordered by pooled Q̂ descending with
        zero-pull arms first; allocators without a bandit ranking return None
        (the caller falls back to its own score ranking — the paper's "Unif"
        comparator)."""
        ...

    def get_state(self) -> dict[str, Any] | None:
        """Serializable allocator internal state (for checkpoint); stateless
        allocators return None."""
        ...

    def set_state(self, data: Any) -> None:
        """Restore internal state from a checkpoint."""
        ...


class FlatBudget:
    """Full flat budget: every candidate evaluates the whole split (used by
    hill_climb / evoprompt / ape)."""

    name = "flat"

    def propose_load(
        self,
        round_idx: int,
        state: OptimizerState,
        prompts: Sequence[Prompt],
        all_sample_ids: Sequence[str],
    ) -> LoadPlan:
        ids = tuple(all_sample_ids)
        return LoadPlan(items=tuple(WorkItem(p, ids) for p in prompts))

    def takeover(self, round_idx: int) -> None:
        return None

    def observe(self, evals: Sequence[Evaluation], rewards: Sequence[float]) -> list[str]:
        return [e.prompt_id for e in evals]

    def rank(self) -> list[int] | None:
        return None

    def get_state(self) -> dict[str, Any] | None:
        return None

    def set_state(self, data: Any) -> None:
        return None


class EvenSplitBudget:
    """Split one total sample budget evenly across candidates (protegi bf
    semantics: each candidate gets total/n samples)."""

    name = "even_split"

    def __init__(self, total_sample_budget: int) -> None:
        if total_sample_budget < 1:
            raise ValueError(f"total_sample_budget must be >= 1, got {total_sample_budget}")
        self.total = total_sample_budget

    def propose_load(
        self,
        round_idx: int,
        state: OptimizerState,
        prompts: Sequence[Prompt],
        all_sample_ids: Sequence[str],
    ) -> LoadPlan:
        n = max(1, len(prompts))
        per = max(1, min(len(all_sample_ids), ceil(self.total / n)))
        subset = tuple(all_sample_ids[:per])
        return LoadPlan(items=tuple(WorkItem(p, subset) for p in prompts))

    def takeover(self, round_idx: int) -> None:
        return None

    def observe(self, evals: Sequence[Evaluation], rewards: Sequence[float]) -> list[str]:
        return [e.prompt_id for e in evals]

    def rank(self) -> list[int] | None:
        return None

    def get_state(self) -> dict[str, Any] | None:
        return None

    def set_state(self, data: Any) -> None:
        return None


class BanditBudget:
    """UCB1 online adaptive budget (ProTeGi semantics): the first round
    evaluates every arm once; later rounds only evaluate the arms ranked top by
    UCB, and ``rank()`` exposes the final SelectTop_b(Q̂) order for the caller's
    selection step.

    It is a standalone stateful allocator, so "which candidate deserves more
    budget" is a first-class decision rather than a method-local constant.
    Reward convention: oriented score (nominally near [0,1], maximize objective).
    """

    name = "bandit"

    def __init__(
        self, eval_rounds: int = 2, samples_per_eval: int = 4, c: float = 1.0, keep_top: float = 0.5
    ) -> None:
        if eval_rounds < 1:
            raise ValueError(f"eval_rounds must be >= 1, got {eval_rounds}")
        if samples_per_eval < 1:
            raise ValueError(f"samples_per_eval must be >= 1, got {samples_per_eval}")
        if c < 0:
            raise ValueError(f"c must be >= 0, got {c} (UCB exploration coefficient)")
        if not 0 < keep_top <= 1:
            raise ValueError(
                f"keep_top must be in (0, 1], got {keep_top} (fraction of arms kept per round)"
            )
        self.eval_rounds = eval_rounds
        self.samples_per_eval = samples_per_eval
        self.c = c
        self.keep_top = keep_top
        self._arms: list[Prompt] | None = None
        self._counts: list[int] = []
        self._values: list[float] = []
        self._samples: list[str] = []
        self._round = 0

    def _reset(self, prompts: Sequence[Prompt], all_sample_ids: Sequence[str]) -> None:
        self._arms = list(prompts)
        self._counts = [0] * len(prompts)
        self._values = [0.0] * len(prompts)
        self._samples = list(all_sample_ids)
        self._round = 0

    def _is_current(self, prompts: Sequence[Prompt]) -> bool:
        return self._arms is not None and [p.id for p in prompts] == [a.id for a in self._arms]

    def propose_load(
        self,
        round_idx: int,
        state: OptimizerState,
        prompts: Sequence[Prompt],
        all_sample_ids: Sequence[str],
    ) -> LoadPlan:
        """Pull ``self._round``: the UCB-narrowed arm set evaluated on the pull's
        window ``samples[_round·spe : (_round+1)·spe]`` (spe = samples_per_eval).
        **Config is law: the window plan executes exactly as
        configured — this method never rewrites a configured value.** A window
        overlapping the split end is legitimate — the last partial window is
        served as-is (the official algorithm draws fixed-size batches reusing
        samples across rounds; our fixed sequential windows end on a short tail
        when the split is not a multiple — a data fact, not parameter
        rewriting) — so ``eval_rounds × samples_per_eval`` is not required to
        divide the split. A pull starting
        **at or beyond** the split end is a misconfiguration and raises
        ``ValueError`` carrying the arithmetic (rounds, spe, split size, the
        failing pull) plus both remediation values for the configurator:
        ``samples_per_eval ≤ n // eval_rounds`` and ``eval_rounds ≤ n //
        samples_per_eval`` — the framework refuses to guess; the human fixes the
        config. Both failure shapes are excluded: silently re-serving window 0
        for every remaining round, and auto-refitting the configured
        quantities (sample quantities belong to the configurator).

        This runtime guard is the static-check substitute for a config scan
        (tests/test_configs.py can validate allocator pinning but not window
        feasibility: split sizes are runtime facts of the task data, not config keys —
        so a knob edit that arms window exhaustion is caught only when the arithmetic
        is actually attempted)."""
        if not self._is_current(prompts):
            self._reset(prompts, all_sample_ids)
        assert self._arms is not None
        n = len(self._arms)
        if self._round == 0:
            chosen = list(range(n))  # first round: all arms (round-robin init)
        else:
            chosen = self._ucb_order()[: max(1, ceil(n * self.keep_top))]  # adaptive narrowing
        start = self._round * self.samples_per_eval
        n_samples = len(self._samples)
        if start >= n_samples:
            spe_fix = n_samples // self.eval_rounds
            rounds_fix = n_samples // self.samples_per_eval
            impossible = (
                " — both fixes compute below 1: this split cannot host the plan at "
                "any cap, provide more selection data"
                if spe_fix < 1 or rounds_fix < 1
                else ""
            )
            raise ValueError(
                f"BanditBudget window exhaustion: pull {self._round + 1} of "
                f"{self.eval_rounds} starts at sample {start}, at or beyond the end of the "
                f"{n_samples}-sample selection split — eval_rounds × samples_per_eval "
                f"= {self.eval_rounds} × {self.samples_per_eval} = "
                f"{self.eval_rounds * self.samples_per_eval}, but the split holds "
                f"{n_samples} samples. Config is law — the framework does not shrink "
                f"configured quantities at runtime; fix the config. Remediation "
                f"(either suffices for full windows): samples_per_eval ≤ "
                f"{spe_fix} (n // eval_rounds = {n_samples} // {self.eval_rounds}) "
                f"or eval_rounds ≤ {rounds_fix} (n // samples_per_eval = "
                f"{n_samples} // {self.samples_per_eval}){impossible}"
            )
        subset = tuple(self._samples[start : start + self.samples_per_eval])
        self._round += 1
        return LoadPlan(items=tuple(WorkItem(self._arms[i], subset) for i in chosen))

    def takeover(self, round_idx: int) -> None:
        """Strategy-owned round (``Decision(items=...)``): the ENTIRE arm state
        is discarded — arms unbound, counts/values/samples zeroed, round reset
        (no keep-arms semantics across a takeover round). The next
        ``propose_load`` re-baselines UCB on the freshly proposed arm set."""
        self._arms = None
        self._counts = []
        self._values = []
        self._samples = []
        self._round = 0

    def _qhat(self) -> list[float]:
        """Pooled mean reward per arm; ``inf`` for never-pulled arms (UCB's
        optimism for the unvisited)."""
        return [
            v / c if c else float("inf") for v, c in zip(self._values, self._counts, strict=True)
        ]

    def _ucb_order(self) -> list[int]:
        assert self._arms is not None
        total = sum(self._counts) or 1
        qhat = self._qhat()
        scored: list[tuple[float, int]] = []
        for i in range(len(self._arms)):
            if self._counts[i] == 0:
                scored.append((float("inf"), i))
            else:
                bonus = self.c * sqrt(log(total) / self._counts[i])
                scored.append((qhat[i] + bonus, i))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [i for _, i in scored]

    def observe(self, evals: Sequence[Evaluation], rewards: Sequence[float]) -> list[str]:
        """Consume evaluation results + a scalar reward per entry (oriented
        score); return the surviving prompt_ids. Post-``takeover`` (arms unbound
        until the next ``propose_load``) this is a documented NO-OP — the
        OptimizerLoop calls ``observe`` unconditionally after every evaluate
        round, including a strategy-owned (``Decision(items=...)``) round whose
        ``takeover`` already discarded the arms; there is nothing to credit, the
        rewards of that round are dropped, and the next ``propose_load``
        re-baselines UCB on the freshly proposed arm set."""
        if self._arms is None:
            return [e.prompt_id for e in evals]
        by_id = {a.id: i for i, a in enumerate(self._arms)}
        for evaluation, reward in zip(evals, rewards, strict=True):
            i = by_id.get(evaluation.prompt_id)
            if i is not None:
                self._counts[i] += 1
                self._values[i] += reward
        return [e.prompt_id for e in evals]

    @property
    def exhausted(self) -> bool:
        return self._round >= self.eval_rounds

    def rank(self) -> list[int] | None:
        """SelectTop_b(Q_T): positions into the current arm list — the prompts
        of the most recent ``propose_load`` — by pooled Q̂ descending, zero-pull
        arms first (∞), ties by index. None before the first round — there is
        nothing ranked yet. The paper's final beam order (Algorithm 3's
        "return SelectTop_b(Q_T)"); the caller maps positions through the arm
        identities it committed (protegi: ``ProtegiState.arm_order``), so a
        later ``_reset`` can never make stale positions misaddress prompts."""
        if self._arms is None:
            return None
        qhat = self._qhat()
        return sorted(range(len(self._arms)), key=lambda i: (-qhat[i], i))

    def get_state(self) -> dict[str, Any] | None:
        if self._arms is None:
            return None
        return {
            "arms": [a.text for a in self._arms],
            "counts": self._counts,
            "values": self._values,
            "samples": self._samples,
            "round": self._round,
        }

    def set_state(self, data: Any) -> None:
        if not data:
            return
        self._arms = [Prompt(text) for text in data["arms"]]
        self._counts = list(data["counts"])
        self._values = list(data["values"])
        self._samples = list(data["samples"])
        self._round = int(data["round"])
