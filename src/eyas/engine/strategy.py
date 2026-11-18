"""Strategy base + managed optimization loop.

A free-form ``run()`` makes every invariant (budget / accounting / history /
checkpoint) depend on each method's own discipline. This module splits the
concerns instead: ``Strategy`` only writes the **one-step decision**
(``step`` returns a ``Decision``), while ``OptimizerLoop`` manages budget
deduction, running evaluations, cost accounting, history, watchdog bump, and
state serialization. Methods interact only through the ``RunContext``
orchestration facilities and **never get a bare backend**.

Concrete algorithms live in ``eyas.engine.strategies``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, ClassVar, Literal

from eyas.backends.base import InferenceBackend, SamplingConfig
from eyas.backends.scheduling import run_sync
from eyas.core.objective import Objective
from eyas.engine.artifacts import ArtifactStore
from eyas.engine.budget import BudgetAllocator, FlatBudget, LoadPlan, WorkItem
from eyas.engine.capability import (
    COMPARE_EVAL,
    BackendLimits,
    CapSpec,
    EvalModelCap,
    TaskCap,
    validate_assembly,
)
from eyas.engine.domain import (
    Archive,
    Artifact,
    Cost,
    Evaluation,
    OptimizerState,
    Prompt,
    PromptListState,
)
from eyas.engine.evaluation import Evaluator
from eyas.engine.sampling import DEFAULT_TIERS, merge_tier
from eyas.engine.status import read_status, write_json_atomic
from eyas.logging import get_logger, key_event
from eyas.tasks.base import Task
from eyas.tasks.protocol import VariantProtocol
from eyas.watchdog import bump

__all__ = [
    "Decision",
    "EvalView",
    "OptimizerLoop",
    "RoundRecord",
    "RunContext",
    "RunResult",
    "Strategy",
    "build_run_context",
    "evaluate_then_finish",
]

Action = Literal["evaluate", "propose", "finish"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class Decision:
    """One strategy step decision: evaluate (evaluate these) / propose (new
    candidates already in state) / finish.

    ``items``: a strategy-owned per-round load — when
    set on an ``evaluate`` decision the loop evaluates EXACTLY these work items
    (planner + record-store path unchanged; ``WorkItem`` already models per-prompt
    sample subsets), bills the round like any other, and notifies the allocator
    (``BudgetAllocator.takeover``) that ``propose_load`` was bypassed for the round."""

    action: Action
    prompts: tuple[Prompt, ...] = ()
    best_id: str | None = None
    items: tuple[WorkItem, ...] | None = None


@dataclass
class EvalView:
    """Read-only view fed to Strategy.step: query historical evaluations;
    methods cannot mutate it."""

    _archive: Archive
    _score_of: Any

    def evaluated(self, prompt: Prompt) -> bool:
        return self._archive.contains(prompt.id)

    def record(self, prompt: Prompt) -> Evaluation:
        """The candidate's historical-best Evaluation (per-sample outcomes
        included). A subset round leaves a subset record: callers reading the
        outcomes must know the round's sample set (spo's vote/optimize prompts
        pair outcomes with their scheduled ids)."""
        best = self._archive.best(prompt.id)
        if best is None:
            raise KeyError(f"prompt {prompt.id} has no evaluation in the archive")
        return best

    def score(self, prompt_id: str) -> float | None:
        best = self._archive.best(prompt_id)
        return None if best is None else self._score_of(best)

    def per_sample(self, prompt_id: str) -> tuple[Any, ...]:
        """The archived-best Evaluation's per-sample rows — an alias of
        ``record``'s payload for one prompt id. This is the **archive view**,
        not the durable record store: a subset round yields subset rows (pair
        them with the round's scheduled ids), and an evaluated-but-archived-lower
        prompt reads through its own best record. (Named ``evaluated`` rather
        than ``outcomes``: these are per-round in-memory views, not the durable
        layer.)"""
        best = self._archive.best(prompt_id)
        return () if best is None else best.per_sample


@dataclass
class RunContext:
    """Orchestration facilities of one run (constructed after capability
    validation passes). Methods interact only through it."""

    task: Task
    protocol: VariantProtocol
    evaluator: Evaluator
    budget: BudgetAllocator
    primary_objective: Objective
    rng: Random
    eval_split: str = "train"
    final_eval_split: str | None = None
    sample_ids: tuple[str, ...] = ()
    max_steps: int = 400
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    sampling_profiles: dict[str, SamplingConfig] = field(
        default_factory=lambda: dict(DEFAULT_TIERS)
    )
    tier_requests: dict[str, int] = field(default_factory=dict)
    #: The run dir, set by the runtime (``None`` for library use). The loop reads
    #: the run's observation files from here — ``--resume`` refills the round clock
    #: below from this dir's ``status.json``.
    run_dir: Path | None = None
    #: Wall seconds per completed round, in round order, measured by the loop on
    #: its own clock — the same quantity status.json's history entries report,
    #: measured over near-identical windows (the loop's round start→end vs the
    #: runtime's between-refresh interval; the resume refill from status.json is
    #: an approximation). The terminal progress line reads its tail (the round
    #: that just ended) and its mean (the eta); a resume refills it from the run
    #: dir's status so the estimate survives the restart. Kept on the context
    #: because
    #: ``RoundRecord.cost.wall_seconds`` is the per-evaluation bill, not the
    #: round's elapsed time.
    round_seconds: list[float] = field(default_factory=list)
    artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    #: sha256 of the canonical run config (path-only keys ``run.output_dir`` /
    #: ``run.output_root`` / ``run.cache_dir`` excluded), set by the runtime at
    #: startup; ``None`` for
    #: library use (no run config exists). The checkpoint freezes it so
    #: ``--resume`` can reject a config that changed since the interrupted run.
    config_fingerprint: str | None = None
    #: Per-top-level-section sha256s of the same canonical config — the resume
    #: mismatch diagnostic names the changed sections from this companion map
    #: (a whole-config hash alone cannot say WHICH section changed). ``None``
    #: for library use, alongside ``config_fingerprint``.
    config_sections: dict[str, str] | None = None

    def score_of(self, evaluation: Evaluation) -> float:
        return self.primary_objective.oriented(
            evaluation.scores.get(self.primary_objective.key, 0.0)
        )


@dataclass
class RoundRecord:
    round: int
    n_candidates: int
    n_requests: int
    cost: Cost
    #: The round's wall seconds on the LOOP's clock (round start→end, the same
    #: quantity ``ctx.round_seconds`` carries and the eta is projected over).
    #: Distinct from ``cost.wall_seconds`` (the per-evaluation bill, max across
    #: the round's evaluations) and from status.json's ``elapsed_sec`` (the
    #: runtime's between-refresh interval over a near-identical window).
    #: Default 0.0 keeps pre-existing checkpoints replayable.
    elapsed_sec: float = 0.0
    #: The archive leader AFTER this round folded in — ``{"prompt_hash",
    #: "metrics"}`` under the ledger's ranking rule (the same tie-break as the
    #: final ranking). Consecutive leaders are the best-so-far learning curve;
    #: their deltas are the per-round marginal gains. Without this field the
    #: final artifact keeps only the end-state ranking (no round attribution),
    #: so an improvement curve is unrecoverable post hoc. ``None`` (pre-field
    #: checkpoints) means "curve point not recorded".
    best_so_far: dict[str, Any] | None = None


@dataclass
class RunResult:
    best: Evaluation | None
    ranked: list[Evaluation]
    history: list[RoundRecord]
    total_cost: Cost
    texts: dict[str, str] = field(default_factory=dict)
    cost_by_tier: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    final_eval: dict[str, Any] | None = None

    def best_text(self) -> str | None:
        return None if self.best is None else self.texts.get(self.best.prompt_id)

    def export(self, path: Any) -> None:
        payload = {
            "best": (
                None
                if self.best is None
                else {
                    "prompt_hash": self.best.prompt_id,
                    "text": self.texts.get(self.best.prompt_id),
                    "metrics": dict(self.best.scores),
                }
            ),
            "ranked": [
                {
                    "prompt_hash": e.prompt_id,
                    "text": self.texts.get(e.prompt_id),
                    "metrics": dict(e.scores),
                }
                for e in self.ranked
            ],
            # per-round cost + wall seconds ride into result.json: the
            # per-round spend curve (which round burned what, judge split
            # included) must be readable from the final artifact alone, not
            # only from the checkpoint / status side files
            "history": [
                {
                    "round": r.round,
                    "n_candidates": r.n_candidates,
                    "n_requests": r.n_requests,
                    "cost": r.cost.to_json(),
                    "elapsed_sec": r.elapsed_sec,
                    "best_so_far": r.best_so_far,
                }
                for r in self.history
            ],
            "total_cost": self.total_cost.to_json(),
            "cost_by_tier": self.cost_by_tier,
            "artifacts": self.artifacts,
            "final_eval": self.final_eval,
        }
        write_json_atomic(Path(path), json.dumps(payload, ensure_ascii=False, indent=2))


class Strategy(ABC):
    """Method contract: typed config + capability declaration + init + step.
    Termination is expressed by step returning finish."""

    name: ClassVar[str]
    capabilities: ClassVar[CapSpec]
    config_cls: ClassVar[type]
    state_cls: ClassVar[type[OptimizerState]]

    @classmethod
    def capabilities_for(cls, config: Any) -> CapSpec:
        """Assembly-time view of the CapSpec (build_run_context consumes THIS,
        not the class default). Strategies whose declared resource profile
        DEPENDS on config values must override: the class-level
        ``capabilities`` freezes constructor defaults, so a config-raised
        limit would otherwise be enforced against the stale default (e.g. a
        demos=4 config with a max_images=2 guard → every generation silently
        rejected, degrading to seed-only)."""
        return cls.capabilities

    def __init__(self, config: Any, ctx: RunContext) -> None:
        self.config = config
        self.ctx = ctx

    def derived_rng(self) -> Random:
        return Random(self.ctx.rng.getrandbits(64))

    def expected_rounds(self) -> int | None:
        """Self-declared evaluation-round horizon (seed round included).

        Observation-only: the loop projects the round-progress echo's eta over
        ``min(max_steps, expected_rounds())``, never over the raw ceiling — a
        method that stops after 4 rounds must not be projected as if it had 400
        left. The declaration is not a bound and not a floor: ``max_steps`` stays
        the enforced ceiling and the echo's denominator, and a declaration above
        it is ignored. ``None`` (default) keeps the ceiling projection."""
        return None

    def restore(self, data: Mapping[str, Any]) -> OptimizerState:
        """Restore typed state from a checkpoint (for --resume)."""
        return self.state_cls.from_json(data)

    def emit_artifact(self, kind: str, payload: Any) -> Artifact:
        """Write a cross-method artifact into ctx.artifacts (address =
        ``<strategy>.<kind>``, content-addressed)."""
        artifact = Artifact(kind=kind, payload=payload, producer=self.name)
        self.ctx.artifacts.emit(artifact)
        return artifact

    def generate(self, requests: Sequence[Any], tier: str = "generate") -> list[str]:
        """Synchronous generation entry for LLM-backed propose: fetches the
        algorithm convention by **tier name** and merges it with the deployment
        baseline.

        Tiers carry only temperature/top_p/max_new_tokens; thinking/seed always
        come from ``ctx.sampling`` (via merge_sampling). This root-cures magic
        SamplingConfig inside optimizers. The synchronous call is valid only
        under OptimizerLoop (no running event loop).
        """
        merged = merge_tier(self.ctx.sampling, self.ctx.sampling_profiles, tier)
        self.ctx.tier_requests[tier] = self.ctx.tier_requests.get(tier, 0) + len(requests)
        return run_sync(self.ctx.evaluator.agenerate_prompts(requests, merged))

    @abstractmethod
    def init(self, seeds: Sequence[Prompt]) -> OptimizerState: ...

    @abstractmethod
    def step(self, state: OptimizerState, view: EvalView) -> Decision: ...


def evaluate_then_finish(state: OptimizerState, view: EvalView) -> Decision:
    """Shared step skeleton for the degenerate single-candidate methods (cot/raw):
    evaluate the pool's only candidate once, then finish on it.

    One copy of the evaluate-then-finish shape; a strategy that needs
    its own selection or multi-round machine writes its own ``step``."""
    assert isinstance(state, PromptListState)
    prompt = state.candidates[0]
    if not view.evaluated(prompt):
        return Decision("evaluate", prompts=(prompt,))
    return Decision("finish", best_id=prompt.id)


class OptimizerLoop:
    """The loop that owns the invariants: budget / evaluation / accounting /
    history / bump / final state."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def run(
        self,
        strategy: Strategy,
        seeds: Sequence[str],
        *,
        resume: Mapping[str, Any] | None = None,
        on_round: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunResult:
        ctx = self.ctx
        archive = Archive(ctx.score_of)
        view = EvalView(archive, ctx.score_of)

        # Terminal progress contract: one INFO line per completed round, keyed to
        # the round whose checkpoint just persisted — acked after the caller's
        # on_round returns, so the console never reports a round the run dir has
        # not frozen (status.json / ``eyas status`` already show it by then).
        def _on_round(snapshot: dict[str, Any]) -> None:
            if on_round is not None:
                on_round(snapshot)
            self._echo_round_progress(snapshot["history"], strategy.expected_rounds())

        state, history, total, texts, round_idx = self._start_state(
            strategy, seeds, resume, archive
        )
        for _step in range(ctx.max_steps):
            decision = self._step_round(strategy, state, view, round_idx)
            if decision.action == "finish":
                break
            for p in decision.prompts:
                texts.setdefault(p.id, p.text)
            if decision.action == "evaluate":
                prompts, plan = self._plan_round_load(decision, state, texts, round_idx)
                total = self._run_round(archive, history, prompts, plan, round_idx, total)
                round_idx += 1
                _on_round(
                    self._snapshot(strategy, state, archive, history, total, texts, round_idx)
                )
            # propose: state was already updated by the method; informational
            # (the next step evaluates the new candidates)

        return self._finish(decision, archive, history, total, texts, round_idx)

    def _start_state(
        self,
        strategy: Strategy,
        seeds: Sequence[str],
        resume: Mapping[str, Any] | None,
        archive: Archive,
    ) -> tuple[OptimizerState, list[RoundRecord], Cost, dict[str, str], int]:
        """Fresh init or checkpoint restore — the loop's starting state ledger.

        Resume replays the checkpoint's durable state (typed state, archive,
        round history, cost, texts, tier ledger, budget, rng, round cursor)
        onto the fresh loop objects; nothing here writes durables, so a crash
        mid-restore is indistinguishable from a crash before the run started."""
        ctx = self.ctx
        if resume is not None:
            state = strategy.restore(resume["state"])
            for item in resume.get("archive", []):
                archive.add(Evaluation.from_json(item))
            self._prime_restart(resume)
            history = [
                RoundRecord(
                    int(r["round"]),
                    int(r["n_candidates"]),
                    int(r["n_requests"]),
                    Cost.from_json(r.get("cost", {})),
                    float(r.get("elapsed_sec", 0.0)),
                    r.get("best_so_far"),
                )
                for r in resume.get("history", [])
            ]
            total = Cost.from_json(resume.get("total_cost", {}))
            texts: dict[str, str] = dict(resume.get("texts", {}))
            texts.update({p.id: p.text for p in _state_prompts(state)})
            # per-tier accounting survives the restart (old checkpoints without
            # the key start from an empty ledger — tolerant read, same shape as
            # the other restore fields)
            ctx.tier_requests = dict(resume.get("tier_requests", {}))
            ctx.budget.set_state(resume.get("budget_state"))
            rng_state = resume.get("rng_state")
            if rng_state is not None:
                ctx.rng.setstate(_rng_from_json(rng_state))
            round_idx = int(resume.get("round_idx", 0))
            # the eta clock survives the resume: the run dir's status history
            # carries the same per-round seconds the interrupted attempt measured
            ctx.round_seconds = _resumed_round_seconds(ctx.run_dir)
        else:
            state = strategy.init([Prompt(s) for s in seeds])
            history = []
            total = Cost()
            texts = {p.id: p.text for p in _state_prompts(state)}
            round_idx = 0
        return state, history, total, texts, round_idx

    def _step_round(
        self, strategy: Strategy, state: OptimizerState, view: EvalView, round_idx: int
    ) -> Decision:
        # Round breadcrumbs bracket strategy.step: "start" as the last line
        # means inside step, "step done" means inside evaluate (each bump
        # carries the tag, so a silent console still proves liveness)
        _log.debug("[loop] round=%d start", round_idx)
        bump(f"[loop] round={round_idx} start")
        t_step = time.monotonic()
        decision = strategy.step(state, view)
        _log.debug(
            "[loop] round=%d step done action=%s elapsed=%.2fs",
            round_idx,
            decision.action,
            time.monotonic() - t_step,
        )
        return decision

    def _plan_round_load(
        self,
        decision: Decision,
        state: OptimizerState,
        texts: dict[str, str],
        round_idx: int,
    ) -> tuple[tuple[Prompt, ...], LoadPlan]:
        """Turn an ``evaluate`` decision into the round's (prompts, plan)."""
        ctx = self.ctx
        items = decision.items
        if items is not None:
            # Strategy-owned subset load (SPO OvO round): evaluate EXACTLY the
            # decision's work items — the planner + record-store path is
            # unchanged (WorkItem already models per-prompt sample subsets).
            # One prompt on two sample subsets is two legitimate items: the
            # evaluator dedups (prompt, sample) pairs across the whole batch
            # BEFORE dispatch (first occurrence wins, candidate-first order
            # preserved) and bills each pair once, to the first item owning it
            # — no double dispatch, no double billing. Budget's own propose_load
            # is bypassed for the round; takeover() is the allocator's notice.
            if not items:
                raise ValueError("Decision(items=()) carries no work items")
            prompts = tuple(item.prompt for item in items)
            for p in prompts:
                texts.setdefault(p.id, p.text)
            plan: LoadPlan = LoadPlan(items=items)
            ctx.budget.takeover(round_idx)
            return prompts, plan
        # duplicate candidates inside ONE decision must not reach budget or the
        # candidate count: dedup by prompt.id before budget + evaluator so
        # RoundRecord counts candidates actually evaluated (the evaluator also
        # dedups (prompt, sample) pairs within the batch, but that is pair-level
        # dispatch/billing hygiene, not candidate identity)
        prompts = decision.prompts
        deduped = _dedup_prompts(prompts)
        if len(deduped) != len(prompts):
            _log.debug(
                "[loop] round=%d dedup evaluate batch %d -> %d candidates",
                round_idx,
                len(prompts),
                len(deduped),
            )
            prompts = deduped
        plan = ctx.budget.propose_load(round_idx, state, prompts, ctx.sample_ids)
        return prompts, plan

    def _run_round(
        self,
        archive: Archive,
        history: list[RoundRecord],
        prompts: tuple[Prompt, ...],
        plan: LoadPlan,
        round_idx: int,
        total: Cost,
    ) -> Cost:
        """Evaluate one round's load, fold the outcome into archive/history/budget;
        returns the updated running total. The round-end snapshot write stays in
        ``run`` (right after this returns and ``round_idx`` advances) — the
        one-durable-event-per-completed-round order is unchanged."""
        ctx = self.ctx
        t_round = time.monotonic()
        evals = run_sync(ctx.evaluator.aevaluate(prompts, plan, ctx.eval_split, round_idx))
        round_cost = _round_cost(evals)
        for e in evals:
            archive.add(e)
        ctx.budget.observe(evals, [ctx.score_of(e) for e in evals])
        total = total + round_cost
        round_elapsed = time.monotonic() - t_round
        ranked_now = archive.ranked()
        leader = ranked_now[0] if ranked_now else None
        history.append(
            RoundRecord(
                round_idx,
                len(prompts),
                round_cost.requests,
                round_cost,
                round_elapsed,
                None
                if leader is None
                else {
                    "prompt_hash": leader.prompt_id,
                    "metrics": dict(leader.scores),
                },
            )
        )
        ctx.round_seconds.append(round_elapsed)
        _log.debug(
            "[loop] round=%d end candidates=%d requests=%d elapsed=%.2fs",
            round_idx,
            len(prompts),
            round_cost.requests,
            round_elapsed,
        )
        bump(f"[loop] round={round_idx} end n={len(prompts)} elapsed={round_elapsed:.2f}s")
        return total

    def _finish(
        self,
        decision: Decision,
        archive: Archive,
        history: list[RoundRecord],
        total: Cost,
        texts: dict[str, str],
        round_idx: int,
    ) -> RunResult:
        """Close the ledger on the loop's last decision.

        No terminal snapshot on the finish decision: status.json closes with
        its own `done` write (runtime), so a fourth durable shape here would
        duplicate the last round's history/echo without any new round (the
        one-durable-event-per-completed-round contract the breadcrumbs and
        status tests pin). The finish decision's state mutation persisting
        nowhere is the REDO semantics of a post-checkpoint crash: a resumed
        run must re-step the in-flight phase from the last round-end snapshot
        — which spo's phase machine guarantees idempotent (committed state
        advances only with evaluate-round checkpoints).
        """
        ctx = self.ctx
        ranked = archive.ranked()
        # best_id is a declared observability field: the ledger's
        # best is ALWAYS the archive-score leader — the accounting rule behind
        # every published number. A method declaring a champion that differs
        # from the ledger (legitimately possible: protegi under a bandit
        # allocator names its arm winner)
        # makes the intent/ledger divergence VISIBLE without changing the rule.
        if decision.best_id is not None and ranked and decision.best_id != ranked[0].prompt_id:
            _log.warning(
                "[loop] declared best %s != archive-score leader %s — ledger keeps"
                " the archive ranking (declared champion is observability only)",
                decision.best_id,
                ranked[0].prompt_id,
            )
        final_eval = self._final_eval(ranked, texts, round_idx)
        return RunResult(
            best=ranked[0] if ranked else None,
            ranked=ranked,
            history=history,
            total_cost=total,
            texts=texts,
            cost_by_tier=dict(ctx.tier_requests),
            artifacts=dict(ctx.artifacts.manifest()),
            final_eval=final_eval,
        )

    def _echo_round_progress(
        self, status_history: Sequence[Mapping[str, Any]], declared_rounds: int | None = None
    ) -> None:
        """The run's terminal contract: one INFO line per completed round.

        ``round <i>/<max>: <n_requests> reqs in <x>s eta≈<y>s``

        * ``i`` — count of rounds whose checkpoint persisted (the same number
          status.json's ``round`` cursor and ``eyas status`` report); ``max`` is
          the loop's enforced step ceiling (``run.max_steps``), which shares that
          counting basis. A method whose own target is smaller simply stops
          early, so the method's self-declared round count is not used as ``max``
          (the loop cannot see it and it is not an enforced bound);
        * ``n_requests`` / ``x`` — the round record's actually issued requests and
          the loop's own clock for the round that just ended (never payloads,
          never estimates);
        * ``eta≈y`` — **mean completed-round duration × remaining rounds**, the
          horizon being ``min(max_steps, declared_rounds)``: the method's own
          declared horizon when it is smaller than the ceiling (a single-round
          method would otherwise project over the whole step ceiling), falling
          back to the ceiling when undeclared or over-declared. Plain mean over
          the rounds this attempt has measured. ``eta≈n/a`` verbatim until at
          least one completed round exists — a projection with no measured round
          behind it is a fabricated number.

        INFO is the default console level, so this line is the honest progress
        signal a user gets without opening the run dir at DEBUG.
        """
        completed = len(status_history)
        n_requests = int(status_history[-1].get("n_requests", 0)) if completed else 0
        # the clock's tail is the round that just ended (appended right before this
        # ack), so nothing here depends on the two histories being the same length
        seconds = self.ctx.round_seconds
        elapsed = seconds[-1] if seconds else 0.0
        # eta = mean duration of the rounds measured so far × the rounds left under
        # the horizon — no other input. With no measured round it stays n/a: a
        # projection over zero measurements is a fabricated number.
        if seconds:
            horizon = (
                min(self.ctx.max_steps, declared_rounds)
                if declared_rounds is not None
                else self.ctx.max_steps
            )
            eta = f"{sum(seconds) / len(seconds) * max(horizon - completed, 0):.2f}s"
        else:
            eta = "n/a"
        key_event(
            _log,
            "round %d/%d: %d reqs in %.2fs eta≈%s",
            completed,
            self.ctx.max_steps,
            n_requests,
            elapsed,
            eta,
        )

    def _prime_restart(self, resume: Mapping[str, Any]) -> None:
        """Backfill the record store from the checkpoint's train-split outcomes.

        Resume re-runs evaluate rounds against the engine; with the cache-first
        evaluator this backfill is exactly what the run dir's sqlite store
        makes free (every archived outcome already stored → zero re-issued
        requests). A library resume with a fresh in-memory store would
        otherwise re-issue those requests and break byte-replay of the
        reference stream — filling the store from the durable archive restores
        the persisted-run shape. The fill NEVER overwrites: an outcome already in
        the store (resume against the persisted run-dir cache) is
        skipped, so priming is a no-op on a populated store (pinned by
        test_resume_priming_never_overwrites_a_populated_store). Real coverage
        limit: error-carrying outcomes are skipped and legitimately re-run on
        the resumed engine; the archive the checkpoint carries is
        ``Archive.history()`` — append-only, so every candidate ever evaluated
        (losers included) is backfillable.
        """
        store = self.ctx.evaluator.store
        sample_ids = set(self.ctx.sample_ids)
        for item in resume.get("archive", []):
            evaluation = Evaluation.from_json(item)
            if evaluation.split != self.ctx.eval_split:
                continue
            for outcome in evaluation.per_sample:
                if outcome.error is not None or outcome.sample_id not in sample_ids:
                    continue
                if store.get(evaluation.prompt_id, outcome.sample_id, evaluation.split) is not None:
                    continue
                store.put(outcome, evaluation.prompt_id, evaluation.split)

    def _final_eval(
        self, ranked: list[Evaluation], texts: dict[str, str], round_idx: int
    ) -> dict[str, Any] | None:
        """Closing evaluation: the archived best is re-evaluated on
        ``final_eval.split`` — the loop split selects the model, the report
        split produces the numbers.

        Same split is skipped (the archive already holds numbers for that
        split); compare semantics are skipped (single-candidate compare
        degenerates to the 0.5 shortcut and could not produce valid metrics).
        Cost is recorded separately as final_eval.cost and not counted into
        total_cost (total requests follow the loop's accounting; resume
        comparisons are unaffected).
        """
        ctx = self.ctx
        if (
            ctx.final_eval_split is None
            or ctx.final_eval_split == ctx.eval_split
            or not ranked
            or COMPARE_EVAL in ctx.evaluator.model.semantics()
        ):
            return None
        text = texts.get(ranked[0].prompt_id)
        if not text:
            return None
        prompt = Prompt(text)
        plan = LoadPlan(items=(WorkItem(prompt),))
        evals = run_sync(ctx.evaluator.aevaluate([prompt], plan, ctx.final_eval_split, round_idx))
        if not evals:
            return None
        return {
            "split": ctx.final_eval_split,
            "prompt_hash": evals[0].prompt_id,
            "metrics": dict(evals[0].scores),
            "cost": evals[0].cost.to_json(),
        }

    def _snapshot(
        self,
        strategy: Strategy,
        state: OptimizerState,
        archive: Archive,
        history: list[RoundRecord],
        total: Cost,
        texts: dict[str, str],
        round_idx: int,
    ) -> dict[str, Any]:
        return {
            "strategy": strategy.name,
            "round_idx": round_idx,
            "state": state.to_json(),
            "archive": [e.to_json() for e in archive.history()],
            "history": [
                {
                    "round": r.round,
                    "n_candidates": r.n_candidates,
                    "n_requests": r.n_requests,
                    "cost": r.cost.to_json(),
                    "elapsed_sec": r.elapsed_sec,
                    "best_so_far": r.best_so_far,
                }
                for r in history
            ],
            "total_cost": total.to_json(),
            "texts": texts,
            "budget_state": self.ctx.budget.get_state(),
            "rng_state": _rng_to_json(self.ctx.rng.getstate()),
            # the per-tier request ledger (cost_by_tier's source) and the
            # artifact manifest ride with the checkpoint: resume restores the
            # ledger, and the runtime cross-checks the manifest against the
            # artifacts it reloads from run_dir/artifacts/
            "tier_requests": dict(self.ctx.tier_requests),
            "artifacts": self.ctx.artifacts.manifest(),
            # the run-config identity rides with every checkpoint: the runtime's
            # resume guard compares the fingerprint (and names changed sections
            # via the companion map). Library loops carry None → null: such a
            # checkpoint is unverifiable and the runtime rejects resuming it.
            "config_fingerprint": self.ctx.config_fingerprint,
            "config_sections": self.ctx.config_sections,
        }


def _dedup_prompts(prompts: tuple[Prompt, ...]) -> tuple[Prompt, ...]:
    """Collapse duplicate prompt ids inside one evaluate batch, keeping order."""
    if len({p.id for p in prompts}) == len(prompts):
        return prompts
    deduped: list[Prompt] = []
    seen: set[str] = set()
    for p in prompts:
        if p.id not in seen:
            seen.add(p.id)
            deduped.append(p)
    return tuple(deduped)


def _round_cost(evals: Sequence[Evaluation]) -> Cost:
    """Fold a round's evaluations into its bill: sums across the round, with
    ``wall_seconds`` the max (the round's wall span, not a meaningless sum)."""
    return Cost(
        requests=sum(e.cost.requests for e in evals),
        wall_seconds=max((e.cost.wall_seconds for e in evals), default=0.0),
        input_tokens=sum(e.cost.input_tokens for e in evals),
        output_tokens=sum(e.cost.output_tokens for e in evals),
        judge_requests=sum(e.cost.judge_requests for e in evals),
        judge_input_tokens=sum(e.cost.judge_input_tokens for e in evals),
        judge_output_tokens=sum(e.cost.judge_output_tokens for e in evals),
    )


def _rng_to_json(state: Any) -> list[Any]:
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_from_json(data: list[Any]) -> Any:
    version, internal, gauss = data
    return (int(version), tuple(int(x) for x in internal), gauss)


def _resumed_round_seconds(run_dir: Path | None) -> list[float]:
    """Refill the terminal eta's clock on resume from the run dir's status history.

    ``status.json``'s ``optimizing`` entries carry exactly the per-round seconds
    the interrupted attempt measured, so the eta does not restart from nothing
    after ``--resume``. No run dir (library use), no status file, or a status
    whose history holds no round entry → an empty clock: the eta reads ``n/a``
    again until a round completes in this attempt, never a number the run did
    not measure.
    """
    doc = read_status(run_dir) if run_dir is not None else None
    if doc is None:
        return []
    return [
        float(entry.get("elapsed_sec", 0.0))
        for entry in doc.get("history") or []
        if isinstance(entry, Mapping) and entry.get("phase") == "optimizing"
    ]


def _state_prompts(state: OptimizerState) -> list[Prompt]:
    for attr in ("candidates", "population", "beam", "best"):
        value = getattr(state, attr, None)
        if isinstance(value, Prompt):
            return [value]
        if isinstance(value, list):
            return [p for p in value if isinstance(p, Prompt)]
    return []


def build_run_context(
    strategy_cls: type[Strategy],
    config: Any,
    task: Task,
    protocol: VariantProtocol,
    backend: InferenceBackend,
    eval_model: Any,
    *,
    backend_limits: BackendLimits | None = None,
    budget: BudgetAllocator | None = None,
    judge: Any | None = None,
    store: Any | None = None,
    artifacts: Any | None = None,
    sampling: SamplingConfig | None = None,
    sampling_profiles: dict[str, SamplingConfig] | None = None,
    eval_split: str = "train",
    final_eval_split: str | None = None,
    seed: int = 42,
    max_steps: int = 400,
    run_dir: Path | None = None,
) -> RunContext:
    """Assembly-time capability validation → build the RunContext. Invalid
    combinations (e.g. a compare-only method entering the accuracy leaderboard) fail at startup."""
    if max_steps < 1:
        # fail fast at assembly: a loop that never steps would leave the
        # finish decision unbound (the ledger would have nothing to close on)
        raise ValueError(f"run.max_steps must be >= 1, got {max_steps}")
    store_obj = artifacts if artifacts is not None else ArtifactStore()
    optimize_key = task.objectives[0].key if task.objectives else "score"
    cap, limits = _validate_assembly(
        strategy_cls,
        config,
        task,
        eval_model,
        backend,
        backend_limits,
        store_obj,
        optimize_key,
    )
    evaluator = _build_evaluator(
        task,
        protocol,
        backend,
        cap,
        limits,
        eval_model,
        judge=judge,
        store=store,
        sampling=sampling,
    )
    primary = task.objectives[0] if task.objectives else Objective(key=optimize_key)
    sample_ids = tuple(s.id for s in task.load_split(eval_split))
    return RunContext(
        task=task,
        protocol=protocol,
        evaluator=evaluator,
        budget=budget or FlatBudget(),
        primary_objective=primary,
        rng=Random(seed),
        eval_split=eval_split,
        final_eval_split=final_eval_split,
        sample_ids=sample_ids,
        sampling=sampling or SamplingConfig(),
        sampling_profiles={**DEFAULT_TIERS, **(sampling_profiles or {})},
        artifacts=store_obj,
        max_steps=max_steps,
        run_dir=run_dir,
    )


def _validate_assembly(
    strategy_cls: type[Strategy],
    config: Any,
    task: Task,
    eval_model: Any,
    backend: InferenceBackend,
    backend_limits: BackendLimits | None,
    store_obj: ArtifactStore,
    optimize_key: str,
) -> tuple[CapSpec, BackendLimits]:
    """Declare the task/eval-model capabilities and run assembly validation;
    returns the validated (strategy cap, backend limits) the evaluator needs.
    Invalid combinations raise before any run state exists."""
    cap = strategy_cls.capabilities_for(config)
    metrics = frozenset({obj.key for obj in task.objectives} | {"parse_fail"})
    task_cap = TaskCap(name=task.name, metrics=metrics, optimize_key=optimize_key)
    eval_cap = EvalModelCap(
        name=eval_model.name, semantics=eval_model.semantics(), produces=eval_model.produces()
    )
    # Backend limits: explicit argument wins; otherwise take what the backend
    # declares (vllm knows its engine window and image cap; a remote
    # OpenAI-compatible endpoint that was not told declares None — the guard
    # stays honestly off / the image intersection skips that bound).
    limits = backend_limits or BackendLimits(
        max_images=backend.max_images, context_window=backend.context_window
    )
    violations = validate_assembly(
        cap,
        task_cap,
        [eval_cap],
        limits,
        available_artifacts=frozenset(store_obj.available()),
    )
    if violations:
        raise ValueError("assembly validation failed:\n  - " + "\n  - ".join(violations))
    return cap, limits


def _build_evaluator(
    task: Task,
    protocol: VariantProtocol,
    backend: InferenceBackend,
    cap: CapSpec,
    limits: BackendLimits,
    eval_model: Any,
    *,
    judge: Any | None = None,
    store: Any | None = None,
    sampling: SamplingConfig | None = None,
) -> Evaluator:
    from eyas.engine.planner import RequestPlanner  # local import to avoid a cycle

    # The declared window arms the over-window guard (the field must be plumbed
    # in, or the guard is never armed).
    # Accepted requests pass through unchanged — arming never alters a byte of
    # a request that fits; only over-window requests change shape (engine crash
    # risk → typed Rejected + the 0-score/parse-fail convention).
    planner = RequestPlanner(cap.resource_profile, context_window=limits.context_window)
    return Evaluator(
        task=task,
        protocol=protocol,
        backend=backend,
        planner=planner,
        model=eval_model,
        judge=judge,
        store=store if store is not None else _default_store(),
        sampling=sampling or SamplingConfig(),
    )


def _default_store() -> Any:
    from eyas.engine.evaluation import InMemoryRecordStore

    return InMemoryRecordStore()
