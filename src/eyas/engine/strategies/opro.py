"""opro: unbounded history-pool hill climbing — (text, score) history compressed into a
meta-prompt, from which the LLM produces K new candidates.

Fidelity contract (arXiv:2309.03409, ICLR 2024 + official google-deepmind/opro): the meta-prompt
shows the **highest-scoring** trajectory entries (official ``[-max_num_instructions:]``
on the ascending list) in **ascending** order with **integer 0–100** scores, plus task
exemplars; each raw completion yields **at most one** new candidate (first bracketed
slice, meta-prompt echo dropped — ``genio`` ``single_bracket``). Candidate set stays
monotonically growing (never evicted); ``==accuracy`` stays dropped (the objective
being producible is guaranteed by the capability declaration / assembly validation).

Empty-round rule: the official code has **no early stop** (the fixed step budget is
its only termination), so the paper's prose clause "unable to propose new solutions"
is honored only after a same-meta-prompt retry — one weak round where the optimizer
model parrots the incumbent back (deduped to nothing) must not silently end the run.
When the retry also yields nothing, the run finishes **with a warning**, not silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from eyas.backends.base import ChatRequest, system_message, user_message, user_text
from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, rank_by_score
from eyas.engine.genio import (
    GEN_MARKER,
    SINGLE_BRACKET,
    extract_candidates,
    render_sample_reference,
)
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy
from eyas.logging import get_logger
from eyas.tasks.base import Variant

_log = get_logger(__name__)

#: The trajectory's leading line — the paper states the ascending order and the
#: 0–100 integer score scale explicitly (Fig. 3: "The score ranges from 0 to 100").
_TRAJECTORY_INTRO = (
    "Below are previous instructions with their scores, sorted in ascending order. "
    "The score is an integer from 0 to 100, higher is better."
)

#: The generate-a-better-one directive + the paper §2.2 informal regularizer
#: ("concise", "generally applicable") + the square-bracket output format (Fig. 3).
_META_INSTRUCTIONS = (
    f"{GEN_MARKER} Optimize an instruction for the task exemplified above. "
    "Generate a new instruction that is different from all the instructions below and "
    "would achieve a higher score than all of them. The new instruction should be "
    "concise, effective, and generally applicable to all problems above. "
    "Write the new instruction in square brackets, for example "
    "[Answer the question carefully.]"
)


def serialize_meta_prompt(history: Sequence[tuple[str, float]], keep: int) -> str:
    """Ascending display of the top-scoring ``keep`` entries — the official
    ``[-max_num_instructions:]`` slice on the score-ascending list ("We only keep
    instructions with the highest scores"), scores as integer percents
    (the official 100-bucket display, ``round(score * 100)``)."""
    ordered = sorted(history, key=lambda pair: pair[1])[-keep:] if keep > 0 else []
    return "\n".join(f"[{round(score * 100)}] {text}" for text, score in ordered)


@dataclass
class OproConfig:
    rounds: int = 3
    num_generated: int = 4
    max_num_instructions: int = 8
    #: Task exemplars embedded in the meta-prompt (paper: "several (three for
    #: example)"); 0 disables the section.
    num_exemplars: int = 3
    #: One empty generation round (the meta model parroted/dropped everything) is
    #: retried once with the same meta-prompt before the run may finish; 0 disables
    #: the retry. The official loop has no early stop — the retry keeps the paper's
    #: prose stop clause from firing on a single lazy decode (see module docstring).
    retry_on_empty: int = 1


@dataclass
class OproState(OptimizerState):
    candidates: list[Prompt] = field(default_factory=list)
    history: list[tuple[str, float]] = field(default_factory=list)
    round_idx: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "candidates": [p.text for p in self.candidates],
            "history": self.history,
            "round_idx": self.round_idx,
        }

    @classmethod
    def from_json(cls, data: Any) -> OproState:
        return cls(
            candidates=[Prompt(t) for t in data.get("candidates", [])],
            history=[(str(t), float(s)) for t, s in data.get("history", [])],
            round_idx=int(data.get("round_idx", 0)),
        )


@register_strategy("opro")
class OproStrategy(Strategy):
    config_cls: ClassVar[type] = OproConfig
    state_cls: ClassVar[type] = OproState
    capabilities: ClassVar[CapSpec] = CapSpec(name="opro", requires_eval=frozenset({ABSOLUTE_EVAL}))

    def _exemplar_lines(self, prompt_text: str) -> list[str]:
        """Up to ``num_exemplars`` train-split exemplars, text-only: the question
        is extracted with the same ``user_text`` path the evaluation (and compare
        judging) uses for the sample's first message — never a reverse rebuild —
        so image parts never enter the meta-prompt; the answer is the sample's
        ground truth under the shared reference-key convention
        (``render_sample_reference`` — label wins, never the positional first
        value: KIE stores its entities list before the label)."""
        n = self.config.num_exemplars
        if n <= 0:
            return []
        lines: list[str] = []
        for sample in self.ctx.task.load_split(self.ctx.eval_split):
            if len(lines) >= n:
                break
            question = user_text(
                self.ctx.task.build_request(prompt_text, sample, Variant("identity"))
            )
            answer = render_sample_reference(sample.references)
            lines.append(f"Question: {question} Answer: {answer}")
        return lines

    def _meta_request(self, meta: str) -> ChatRequest:
        return ChatRequest(messages=(system_message(_META_INSTRUCTIONS), user_message(meta)))

    def _meta_user(self, history: Sequence[tuple[str, float]], prompt_text: str) -> str:
        exemplars = self._exemplar_lines(prompt_text)
        parts: list[str] = []
        if exemplars:
            parts.append("Examples:\n" + "\n".join(exemplars))
        parts.append(_TRAJECTORY_INTRO)
        block = serialize_meta_prompt(history, self.config.max_num_instructions)
        if block:
            parts.append(block)
        return "\n\n".join(parts)

    def expected_rounds(self) -> int | None:
        return self.config.rounds + 1  # seed round + per-round generate/eval

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        return OproState(candidates=list(seeds), history=[], round_idx=0)

    def _record_history(self, state: OproState, view: EvalView) -> None:
        """Fold every newly-scored candidate into the (text, score) trajectory
        pool: an entry is appended once per distinct text the history has not
        seen yet (``==accuracy`` stays dropped — the objective being producible
        is guaranteed upstream)."""
        scored = {t for t, _ in state.history}
        for p in state.candidates:
            sc = view.score(p.id)
            if sc is not None and p.text not in scored:
                state.history.append((p.text, sc))
                scored.add(p.text)

    def _generate_candidates(self, state: OproState, best: Prompt) -> list[Prompt]:
        """One meta-prompt round: K copies of the meta request on the generate
        tier, each raw completion yielding AT MOST one new candidate (official
        extraction), deduped against the known candidate set."""
        cfg = self.config
        meta = self._meta_user(state.history, best.text)
        request = self._meta_request(meta)
        outs = self.generate([request] * cfg.num_generated, "generate")
        known = {p.id for p in state.candidates}
        new: list[Prompt] = []
        for out in outs:
            # one raw completion yields at most one candidate (official extraction)
            for instr in extract_candidates(out, SINGLE_BRACKET):
                child = Prompt(instr, metadata={"label": "opro"})
                if child.id not in known:
                    known.add(child.id)
                    new.append(child)
        return new

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, OproState)
        cfg = self.config
        unevaluated = [p for p in state.candidates if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))

        self._record_history(state, view)

        best = rank_by_score(state.candidates, view.score)[0]
        if state.round_idx >= cfg.rounds:
            return Decision("finish", best_id=best.id)
        state.round_idx += 1

        new = self._generate_candidates(state, best)
        for _attempt in range(cfg.retry_on_empty):
            if new:
                break
            _log.warning(
                "opro round %d: meta model yielded %d new candidate(s) (parroted "
                "incumbent / unparsable format) — retrying generation once",
                state.round_idx,
                len(new),
            )
            new = self._generate_candidates(state, best)
        if not new:
            if cfg.retry_on_empty:
                _log.warning(
                    "opro round %d: retry still yielded nothing — finishing "
                    "(paper stop clause 'unable to propose new solutions'; the "
                    "official code has no early stop, set retry_on_empty / rounds "
                    "to keep budget burning instead)",
                    state.round_idx,
                )
            return Decision("finish", best_id=best.id)
        state.candidates.extend(new)
        return Decision("propose", prompts=tuple(new))
