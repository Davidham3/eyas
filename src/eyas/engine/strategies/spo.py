"""spo: self-supervised prompt optimization by OvO answer voting (arXiv:2502.06855,
Findings of EMNLP 2025, reference impl FoundationAgents/SPO) — the paper's loop,
phase by phase:

- **seed** — the initial prompt P0 runs on the full split once. Its per-sample
  records feed the optimize prompt (with references — the paper allows gold on
  the OPTIMIZE side) and the leaderboard anchor;
- **propose** — one LLM call cites the incumbent plus its wrong samples
  (``Q:`` / ``Answer:`` reference / ``Answer (model):`` prediction) and
  generates a challenger template;
- **evaluate (items round)** — incumbent and challenger run on the round's
  ``num_pairs`` samples through ``Decision(items=...)`` (strategy-owned work
  items; ``budget.takeover`` notices the allocator);
- **voting** — inside the next ``step`` (store-complete by construction: the
  items round answered both candidates on the window into the durable record
  store) the two candidates' OWN answers go to the reference-free
  ``AnswerPairwiseJudge``:
  ``votes_per_pair`` ballots per pair with an ABBA position swap. The selection
  channel never sees a gold label — that discipline is SPO's whole trick.
  Accept iff the challenger's win-sum strictly exceeds its loss-sum; a tie
  counts 0.5 for each side (unparsable ballots are ties).
- **done** — unless the incumbent already holds a full-split record, one
  closing full-split evaluate freezes the absolute leaderboard entry, then
  ``finish``.

Pair schedule: the split ids shuffled ONCE by ``derived_rng()``, consumed
without replacement across rounds; exhausting the ids appends a reshuffled
coverage cycle (from the constant split ids, so a crash-window re-append is
idempotent). The schedule + cursors live in ``SpoState``, so a resumed run never
re-samples already-voted pairs: the ITEMS round advances ``cursor`` and marks
``eval_cursor`` in the very step that requests it, and the vote re-reads the
committed window from the persisted schedule with ZERO rng draws — every
in-flight phase touches the shared rng at most once between two checkpoints,
which keeps crash-retry re-derivation byte-identical. The optimize channel
replays deterministically as well: the persisted ``gens_served`` counter is
stamped into every request (``[Round n]``), so a resume against a deployment
whose reply script restarts still re-serves the reference run's reply to the
same byte-identical prompt.

The absolute EvalModel stays only for the seed/closing passes: spo's identity
is that SELECTION is vote-driven, not accuracy-driven — accuracy enters the
archive as the closing record and the reported number.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from eyas.backends.base import ChatRequest, system_message, user_message
from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt
from eyas.engine.genio import (
    GEN_MARKER,
    MAX_BODY_LEN,
    SINGLE_TAG,
    extract_candidates,
    render_sample_reference,
)
from eyas.engine.judges import AnswerPairwiseJudge
from eyas.engine.planner import Rejected
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy, WorkItem
from eyas.logging import get_logger
from eyas.tasks.base import Variant

_log = get_logger(__name__)

Phase = Literal["seed", "propose", "eval", "done"]

#: None sentinels ride the JSON checkpoint path as this empty prompt text.
_NO_PROMPT = ""

#: eval_cursor sentinel: the live challenger has no evaluated window yet — the
#: next step MUST run the items round (a resumed challenger text that an earlier
#: round already archived cannot shortcut it).
_NO_WINDOW = -1

#: the {choices} placeholder rule, phrased for the optimize prompt (paper C3
#: template mentions the placeholder requirement literally)
_CHOICES_RULE = "The {choices} placeholder is required in the optimized prompt."

#: the official extract_content(response, "prompt") pattern — the FULL body,
#: newlines included (a full-template method must not lose lines to a
#: line-first extractor)
_PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL)


def extract_prompt_body(raw: str) -> str:
    """The paper's candidate contract: the whole ``<prompt>...</prompt>`` body
    (DOTALL — multi-line templates survive verbatim). A completion that
    dropped the tags falls back to the genio ``SINGLE_TAG`` profile (the
    ``###OUT###`` body verbatim, capped at ``MAX_BODY_LEN``)."""
    match = _PROMPT_TAG_RE.search(raw)
    body = match.group(1) if match else (extract_candidates(raw, SINGLE_TAG) or [""])[0]
    return body.strip()[:MAX_BODY_LEN]


def _abba_order(n: int) -> list[bool]:
    """Ballot side order (challenger_first per ballot): the canonical ABBA
    position swap — first half False (incumbent-first), second half mirrored
    True (challenger-first); at n=4 exactly F,F,T,T. Odd n gets the extra
    challenger-side ballot (the half-accept capacity probe)."""
    head = [False] * (n // 2)
    return head + [True] * (n - len(head))


def _ensure_choices_placeholder(prompt: str) -> str:
    """Append a choices line when the optimized output lacks ``{choices}``
    (upstream SPO-main warns and appends). spo's full-template semantics live in
    the mcq raw_template guard / kie spec rendering; an offspring losing the
    placeholder would fail fast immediately, so the generation side must provide
    this fallback."""
    if "{choices}" in prompt:
        return prompt
    return prompt.rstrip() + "\nChoices: [{choices}]"


@dataclass
class SpoConfig:
    rounds: int = 2
    #: samples per voting round (official EVALUATION_REPETITION pairing draws); a
    #: round takes the NEXT ids from the one-shuffle schedule, never resamples
    num_pairs: int = 16
    #: ballots per pair, ABBA position-swapped within the group (official 4)
    votes_per_pair: int = 4
    #: cite the {choices} placeholder rule in the optimize prompt (official C3
    #: template includes it; off for tasks whose rendering has no choices line)
    #: and gate the generation-side backfill that appends a missing {choices}
    #: line to every offspring
    prompt_includes_choices: bool = True


@dataclass
class SpoState(OptimizerState):
    """The phase machine's durable shape: everything needed to re-enter the
    exact in-flight phase (which candidates, which schedule, how far the cursor
    consumed) survives the checkpoint.json JSON path.

    ``schedule`` is the flat one-shuffle id sequence (a "pair" = one id's
    incumbent/challenger answer pair); ``cursor`` indexes the ids already
    consumed by items rounds. A crash-retry re-entry re-derives the tail from
    the SAME stored prefix, so appending is idempotent across retries.
    ``gens_served`` counts optimize calls actually issued: it is stamped into
    every optimize request, so a resumed run re-derives optimize prompts
    identical modulo the round stamp even when the deployment's generation
    stream restarts its reply script (a resume against a restarted engine
    re-serves the script's reply #1 — without the stamp that reply silently
    becomes a different challenger and the trajectory diverges). Note: a crash
    strictly INSIDE a step re-derives the in-flight request from the last
    checkpoint's counter, so its stamp may lag the interrupted attempt's
    in-memory value — only requests re-derived at checkpoint boundaries are
    byte-identical stamps included."""

    candidates: list[Prompt] = field(default_factory=list)
    round_idx: int = 0
    phase: Phase = "seed"
    incumbent: Prompt | None = None
    challenger: Prompt | None = None
    cursor: int = 0
    schedule: list[str] = field(default_factory=list)
    gens_served: int = 0
    #: cursor value AFTER the window whose items round evaluated the current
    #: challenger; ``_NO_WINDOW`` while the live challenger has no evaluated
    #: window. The evaluatedness of an items round is prompt × window — an
    #: already-archived challenger text must NOT shortcut a NEW window's round
    #: (a resumed/stale challenger with an old record would skip straight to
    #: re-voting the old window), so the window identity, not view.evaluated,
    #: decides whether this round's items pass has run.
    eval_cursor: int = _NO_WINDOW

    def to_json(self) -> dict[str, Any]:
        return {
            "candidates": [p.text for p in self.candidates],
            "round_idx": self.round_idx,
            "phase": self.phase,
            "incumbent": self.incumbent.text if self.incumbent else _NO_PROMPT,
            "challenger": self.challenger.text if self.challenger else _NO_PROMPT,
            "cursor": self.cursor,
            "schedule": list(self.schedule),
            "gens_served": self.gens_served,
            "eval_cursor": self.eval_cursor,
        }

    @classmethod
    def from_json(cls, data: Any) -> SpoState:
        def _prompt(text: Any) -> Prompt | None:
            return Prompt(text) if text else None

        return cls(
            candidates=[Prompt(t) for t in data.get("candidates", [])],
            round_idx=int(data.get("round_idx", 0)),
            phase=data.get("phase", "seed"),  # type: ignore[arg-type]
            incumbent=_prompt(data.get("incumbent", _NO_PROMPT)),
            challenger=_prompt(data.get("challenger", _NO_PROMPT)),
            cursor=int(data.get("cursor", 0)),
            schedule=[str(pid) for pid in data.get("schedule", [])],
            gens_served=int(data.get("gens_served", 0)),
            eval_cursor=int(data.get("eval_cursor", _NO_WINDOW)),
        )


@register_strategy("spo")
class SpoStrategy(Strategy):
    config_cls: ClassVar[type] = SpoConfig
    state_cls: ClassVar[type] = SpoState
    capabilities: ClassVar[CapSpec] = CapSpec(name="spo", requires_eval=frozenset({ABSOLUTE_EVAL}))

    def __init__(self, config: Any, ctx: Any) -> None:
        super().__init__(config, ctx)
        self._judge = AnswerPairwiseJudge()

    def expected_rounds(self) -> int | None:
        return self.config.rounds + 2  # seed round + voting rounds + closing

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        if not seeds:
            raise ValueError("spo requires at least one seed prompt")
        if len(seeds) > 1:
            # SPO is single-incumbent-chained (Findings of EMNLP 2025, arXiv:2502.06855 §3):
            # P0 is the incumbent and the only seed carried through the loop. Dropping a
            # provided seed silently would violate the fail-fast contract, so
            # the drop is named here.
            _log.warning(
                "spo: %d seeds provided but SPO is single-incumbent-chained "
                "(paper §3) — seeds beyond the "
                "first are ignored (%d dropped: %s)",
                len(seeds),
                len(seeds) - 1,
                ", ".join(p.text for p in seeds[1:]),
            )
        # the paper is single-candidate chained: P0 is the incumbent and the
        # only seed carried through the loop
        return SpoState(candidates=[seeds[0]], phase="seed")

    # -- schedule ---------------------------------------------------------------

    def _next_pairs(self, state: SpoState, n: int) -> list[str]:
        """One-shuffle schedule: ids shuffled once by ``derived_rng()``, consumed
        without replacement; an exhausted schedule appends a fresh reshuffled
        coverage cycle of ``ctx.sample_ids`` (the cycle source is the constant
        split ids, never a schedule slice). The cursor advances ONLY with the
        round-end checkpoint, so a crash-retry re-entry (schedule already
        extended in memory, cursor not yet persisted) re-derives the same
        window — the frozen prefix the vote re-reads never shifts. The window
        is consumed EXACTLY once per phase-machine round trip: every step that
        calls this sets ``eval_cursor`` to the new cursor, so the vote branch
        and the crash-retry re-entry never draw a second window for one round."""
        rng = self.derived_rng()
        if not state.schedule:
            pool = list(self.ctx.sample_ids)
            rng.shuffle(pool)
            state.schedule = pool
        while state.cursor + n > len(state.schedule):
            pool = list(self.ctx.sample_ids)
            rng.shuffle(pool)
            state.schedule.extend(pool)
        taken = state.schedule[state.cursor : state.cursor + n]
        state.cursor += len(taken)
        return list(taken)

    # -- requests -----------------------------------------------------------------

    def _samples_by_id(self) -> dict[str, Any]:
        """Split samples keyed by id (cached — the loop split is immutable for
        the run's lifetime). Both LLM channels read the sample's QUESTION text
        from here: the judge never receives a bare id, and never receives a
        reference either (the vote channel has no reference field at all)."""
        cached = getattr(self, "_sample_cache", None)
        if cached is None:
            cached = {s.id: s for s in self.ctx.task.load_split(self.ctx.eval_split)}
            self._sample_cache = cached
        return cached

    def _renderable(self, prompt: str) -> bool:
        """A wholesale rewrite can drop the task's required placeholders (raw_template
        boards raise at build_request), or an untagged reply that leaks the reference
        prompt duplicates the media placeholders until the rendered request blows the
        context window. Probe EVERY sample of the loop split, not just the first: the
        vote windows are drawn from these ids without replacement (coverage cycles
        revisit all of them), so a sample the planner rejects dooms some round's window
        — rejected samples never persist records, and the vote tripwire would abort the
        run mid-flight. The whole split is a superset of every window and probing it
        draws ZERO rng (the window itself is only drawn later by ``_next_pairs``), so
        the byte-level resume replay stays intact. Any unusable sample stops the loop
        on the incumbent here instead."""
        for sample in self._samples_by_id().values():
            try:
                request = self.ctx.task.build_request(prompt, sample, Variant("identity"))
            except ValueError:
                return False
            if isinstance(self.ctx.evaluator.planner.plan(request), Rejected):
                return False
        return True

    def _optimize_request(self, state: SpoState, view: EvalView) -> ChatRequest:
        assert state.incumbent is not None
        key = self.ctx.primary_objective.key
        # archive-best is CORRECT here — the error
        # list must come from the incumbent's surviving (not-superseded) evidence;
        # a subset round leaves a subset record, so zip outcomes with ITS sample
        # ids, never the full split (unlike the vote channel, this read has no
        # recent-window coverage assumption to guarantee — see _vote_requests).
        record = view.record(state.incumbent)
        errors = [o for o in record.per_sample if not o.scores.get(key, 0.0)]
        # deterministic error order: id-stable (top-3 official)
        errors.sort(key=lambda o: o.sample_id)
        by_id = self._samples_by_id()
        # the model's answer line FIRST (upstream spo order), then the reference
        error_block = "\n".join(
            "\n".join(
                (
                    f"Q: {by_id[o.sample_id].inputs.get('question', o.sample_id)}",
                    f"Answer (model): {o.prediction}",
                    "Answer: " + render_sample_reference(by_id[o.sample_id].references),
                )
            )
            for o in errors[:3]
            if o.sample_id in by_id
        )
        parts = [
            "Task requirements: follow the reference prompt and improve it.",
            _CHOICES_RULE if self.config.prompt_includes_choices else "",
            f"Reference prompt:\n{state.incumbent.text}",
        ]
        if error_block:
            parts.append(f"Errors (question / reference / model answer):\n{error_block}")
        system = (
            f"[Round {state.gens_served + 1}] {GEN_MARKER} Optimize the prompt of a "
            f"vision-language model so it answers "
            f"the questions below correctly. Analyze the errors, describe the "
            f"modification, then output the FULL new prompt. Reply EXACTLY in the "
            f"contract format <analyse>...</analyse><modification>...</modification>"
            f"<prompt>...the optimized prompt...</prompt>."
        )
        return ChatRequest(messages=(system_message(system), user_message("\n\n".join(parts))))

    def _vote_requests(self, state: SpoState, pairs: Sequence[str]) -> list[ChatRequest]:
        assert state.incumbent is not None and state.challenger is not None
        # answers read from the DURABLE record store (keyed prompt_hash ×
        # sample_id × split), not from the archive-best record. The items
        # round above just evaluated BOTH prompts on this exact window, so
        # every voted id has a stored answer no matter how the round scored:
        # Archive.add only displaces the best on a strictly higher score (or a
        # tie with larger evidence), so an incumbent that scores below its
        # historical peak on this window leaves the best record pointing at an
        # old window, and the just-paid-for answers would be invisible to it.
        # Predictions are durable per prompt_hash × sample_id — reading them
        # from the store is what makes a below-peak round still votable.
        # The raise below stays as a tripwire: it now means a genuine store
        # bug (a window sample never recorded — planner rejection or lost
        # write), never a weaker round — vote with a fabricated empty answer never
        # silently, fail fast (same discipline as _vote_pairs). The closing
        # pass reads coverage from the same durable source (_full_record).
        store = self.ctx.evaluator.store
        answers = {
            p.id: {sid: store.get(p.id, sid, self.ctx.eval_split) for sid in pairs}
            for p in (state.incumbent, state.challenger)
        }
        by_id = self._samples_by_id()
        reqs: list[ChatRequest] = []
        for sample_id in pairs:
            missing = [
                p.text[:40]
                for p in (state.incumbent, state.challenger)
                if answers[p.id][sample_id] is None
            ]
            if missing:
                raise RuntimeError(
                    f"spo vote window not covered by the durable record store: "
                    f"sample {sample_id!r} absent from {missing}"
                )
            inc_outcome = answers[state.incumbent.id][sample_id]
            ch_outcome = answers[state.challenger.id][sample_id]
            assert inc_outcome is not None and ch_outcome is not None  # tripwire above
            inc_ans = inc_outcome.prediction
            ch_ans = ch_outcome.prediction
            question = (
                by_id[sample_id].inputs.get("question", sample_id)
                if sample_id in by_id
                else sample_id
            )
            for challenger_first in _abba_order(self.config.votes_per_pair):
                if challenger_first:
                    a_text, b_text = ch_ans, inc_ans
                else:
                    a_text, b_text = inc_ans, ch_ans
                reqs.append(self._judge.build_request(question, a_text, b_text))
        return reqs

    def _tally(self, replies: Sequence[str]) -> tuple[float, float]:
        """Win/loss sums from the group-aligned reply order (ABBA mirrors are
        positional): the side layout comes from the SAME ``_abba_order`` that
        built the ballots — ``order[i]`` True means the challenger sits at
        position A in ballot i. Unparsable ballots count 0.5 for BOTH sides
        (the neutral-tie convention — the official code keeps the
        incumbent when neutral votes dominate, so a tie never helps the
        challenger; 0.5/0.5 is the smooth form of that rule and makes an
        all-neutral group exactly non-accepting)."""
        order = _abba_order(self.config.votes_per_pair)
        wins = losses = 0.0
        for ballot, text in enumerate(replies):
            verdict = self._judge.parse(text)
            champ = "A" if order[ballot % len(order)] else "B"
            if verdict == champ:
                wins += 1.0
            elif verdict in ("A", "B"):
                losses += 1.0
            else:
                wins += 0.5
                losses += 0.5
        return wins, losses

    # -- closing ----------------------------------------------------------------------

    def _vote_pairs(self, state: SpoState, n: int) -> list[str]:
        """The round's vote window: the committed slice from the persisted
        schedule/cursor, read with ZERO rng draws (the items round consumed it:
        ``eval_cursor == cursor``). There is deliberately NO re-derivation
        branch: the phase machine never persists ``phase=eval`` with an
        unevaluated window (both cursors advance in the step that requests the
        items round, and a crash before that checkpoint restores
        ``phase=propose`` instead), so re-drawing here would mean a broken
        checkpoint — fail fast rather than vote on a fabricated window."""
        if state.eval_cursor != state.cursor:
            raise RuntimeError("spo voting before the round's items evaluation")
        return state.schedule[state.cursor - n : state.cursor]

    def _full_record(self, state: SpoState) -> bool:
        assert state.incumbent is not None
        # Durable coverage, not archive-best coverage (same durable-source rule
        # as _vote_requests): a best SUBSET record above the full-split record
        # would otherwise re-request the closing round every step — all-store-
        # hit no-op rounds up to max_steps, finishing with the subset record
        # as the reported entry. The seed pass wrote the full split into the
        # store, so the closing correctly skips.
        store = self.ctx.evaluator.store
        split = self.ctx.eval_split
        return all(
            store.get(state.incumbent.id, sid, split) is not None for sid in self.ctx.sample_ids
        )

    def _finish_decision(self, state: SpoState) -> Decision:
        assert state.incumbent is not None
        if self._full_record(state):
            return Decision("finish", best_id=state.incumbent.id)
        return Decision("evaluate", items=(WorkItem(state.incumbent, ()),))

    # -- step -------------------------------------------------------------------------

    def _seed_phase(self, state: SpoState, view: EvalView) -> Decision | None:
        """Decision: evaluate the seed on the full split once, then crown it
        incumbent and fall through to propose."""
        seed = state.candidates[0]
        if not view.evaluated(seed):
            return Decision("evaluate", prompts=(seed,))
        state.incumbent = seed
        state.phase = "propose"
        return None

    def _propose_phase(self, state: SpoState, view: EvalView) -> Decision:
        # The generation happens AFTER the loop applied the previous decision;
        # the challenger's evaluation round is requested on the NEXT step
        # (one batch per round keeps the crash/step boundary deterministic at
        # the head of every evaluate batch — the resume harness's fault
        # shape). Emitting a no-op decision costs no requests.
        raw = self.generate([self._optimize_request(state, view)], "gradient")[0]
        # the request WAS issued (this counter records service, not
        # acceptance): stamp + counter together make the optimize channel
        # replay-deterministic across a restarted reply script
        state.gens_served += 1
        instruction = extract_prompt_body(raw)
        if self.config.prompt_includes_choices:
            instruction = _ensure_choices_placeholder(instruction)
        challenger = Prompt(instruction, metadata={"label": "spo"}) if instruction else None
        if challenger is None or challenger.id == state.incumbent.id:  # type: ignore[union-attr]
            # an unparsable completion or a verbatim echo: nothing left to try
            state.phase = "done"
            return self._finish_decision(state)
        if not self._renderable(instruction):
            # unusable offspring (missing placeholder): same class as an
            # unparsable completion — stop on the incumbent instead of
            # letting the evaluator's build_request abort the whole run
            state.phase = "done"
            return self._finish_decision(state)
        state.candidates.append(challenger)
        state.challenger = challenger
        state.eval_cursor = _NO_WINDOW  # this challenger has no window yet
        state.phase = "eval"
        return Decision("propose")

    def _items_round(self, state: SpoState) -> Decision:
        """Items round for THIS round's window: incumbent + challenger on the
        round's pairs. The guard is the WINDOW, not the prompt:
        view.evaluated(challenger) is true for a text an earlier round already
        archived, and shortcutting on it would skip this window's evaluation
        and re-vote the old one."""
        cfg = self.config
        pairs = self._next_pairs(state, cfg.num_pairs)
        state.eval_cursor = state.cursor
        return Decision(
            "evaluate",
            items=tuple(
                WorkItem(p, tuple(pairs))
                for p in (state.incumbent, state.challenger)
                if p is not None
            ),
        )

    def _vote_round(self, state: SpoState) -> Decision:
        """This window's items round completed (the checkpoint persisted
        eval_cursor == cursor): the vote batch rides the compare tier."""
        cfg = self.config
        replies = self.generate(
            self._vote_requests(state, self._vote_pairs(state, cfg.num_pairs)),
            "compare",
        )
        wins, losses = self._tally(replies)
        accepted = wins > losses
        if accepted:
            state.incumbent = state.challenger
        state.challenger = None
        state.eval_cursor = _NO_WINDOW  # the next challenger always gets a fresh window
        # The rejected challenger STAYS in candidates: the next round's
        # optimize request may cite its answers (view.record resolves its
        # archive record) when it is the incumbent's historical best — the
        # paper's error context over everything the loop has executed. There
        # is no offspring dedup: the loop is a chained incumbent + one live
        # challenger, and a re-proposed prompt replays from the record
        # store rather than corrupting the schedule.
        state.round_idx += 1
        state.phase = "done" if state.round_idx >= cfg.rounds else "propose"
        if state.phase == "done":
            return self._finish_decision(state)
        return Decision("propose")

    def _eval_phase(self, state: SpoState, view: EvalView) -> Decision:
        """Decision: run this round's items round when the live challenger has
        no evaluated window yet (``eval_cursor != cursor``), otherwise vote the
        committed window and accept/reject."""
        assert state.challenger is not None
        if state.eval_cursor != state.cursor:
            return self._items_round(state)
        return self._vote_round(state)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        """Phase machine: seed → propose → eval (items round, then vote) → done.
        The seed phase falls through to propose on the same step once the seed
        is evaluated (the original if-chain); every other phase returns its
        decision from its own handler."""
        assert isinstance(state, SpoState)
        if state.phase == "seed":
            decision = self._seed_phase(state, view)
            if decision is not None:
                return decision
        if state.phase == "propose":
            return self._propose_phase(state, view)
        if state.phase == "eval":
            return self._eval_phase(state, view)
        if state.phase == "done":
            return self._finish_decision(state)

        raise RuntimeError(f"spo cannot step from phase {state.phase!r}")


__all__ = ["SpoConfig", "SpoState", "SpoStrategy", "_ensure_choices_placeholder"]
