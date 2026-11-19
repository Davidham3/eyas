"""PairwiseJudge extension point: extracts the "how to ask + how to
parse" of pairwise comparison out of CompareEvalModel.

The compare evaluation semantics (CompareEvalModel) owns round-robin
orchestration, win-rate scoring, and cost accounting; "how to ask A/B and how
to parse the winner" is a pluggable judging strategy. The default is LLM
judging (DefaultPairwiseJudge), swappable for rubric/custom prompts — all
members of the same ``judge`` family as the per-sample judges.

``AnswerPairwiseJudge`` serves the same contract with a different object of
comparison: the two texts are model ANSWERS to the
question, not instructions, and the request carries NO reference field — SPO's
selection signal is reference-free (Findings of EMNLP 2025, arXiv:2502.06855),
so a gold label is structurally impossible to leak through this judge's request.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import ClassVar

from eyas.backends.base import ChatRequest, system_message, user_message
from eyas.engine.registry import register_pairwise_judge

__all__ = ["PairwiseJudge", "DefaultPairwiseJudge", "AnswerPairwiseJudge"]


class PairwiseJudge(ABC):
    """Pairwise verdict of two instructions on the same question: produces the
    comparison request + parses A/B/tie.

    The request shape is shared by the whole family: the system message is the
    judge directive (the one thing a subclass varies), the user message carries
    the question/payload. That role split follows the strategies' gen-request
    convention: a system-only request raises in the real Qwen3.5 HF chat
    template ("No user query found"), so the (A)/(B)
    payload always rides the user turn.
    """

    name: ClassVar[str]

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Judge directive for this comparison flavour."""

    def build_request(self, question: str, a_text: str, b_text: str) -> ChatRequest:
        task = f"Question: {question}\n(A) {a_text}\n(B) {b_text}\nAnswer with A or B."
        return ChatRequest(messages=(system_message(self.system_prompt), user_message(task)))

    def parse(self, text: str) -> str:
        """Returns "A" / "B" / "tie".

        The verdict is the first standalone uppercase A/B token anywhere in
        the reply ("Answer: B" is B, not A — a prefix rule would bias every
        such reply toward position A). Lowercase is deliberately not matched:
        uppercasing first would turn the article "a" in prose into a phantom
        verdict. No standalone token -> "tie" (the no-verdict sentinel)."""
        match = re.search(r"\b([AB])\b", text)
        return match.group(1) if match is not None else "tie"


@register_pairwise_judge("default")
class DefaultPairwiseJudge(PairwiseJudge):
    name = "default"
    system_prompt: ClassVar[str] = (
        "You compare two instructions for answering the same question. "
        "Pick the instruction that would produce a better answer."
    )


@register_pairwise_judge("answer")
class AnswerPairwiseJudge(PairwiseJudge):
    """Reference-free output-vs-output verdict (SPO's φeval): two ANSWERS to the
    same question, which better matches the image intent / task requirements.

    ``a_text`` / ``b_text`` are model outputs (execution predictions), never
    instructions and never gold references — the prompt has no reference field,
    so the selection channel cannot see ground truth even by accident. A/B
    position randomization (the 4-vote ABBA group) is the CALLER's business
    (spo's vote schedule); this class only asks and parses, same as the default."""

    name = "answer"
    system_prompt: ClassVar[str] = (
        "You compare two answers to the same question about an image. "
        "Pick the answer that better matches the image intent and the task "
        "requirements. Judge only from the question and the two answers shown."
    )
