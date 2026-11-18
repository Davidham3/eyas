from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from eyas.backends.base import ChatResponse, InferenceBackend, SamplingConfig
from eyas.registry import register_judge
from eyas.tasks.base import Prediction, Sample, Variant

__all__ = ["ChainedJudge", "Judge", "JudgeContext", "Verdict", "register_judge"]


def _merge_usage(
    acc: Mapping[str, Any] | None, new: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """Accumulate backend-call billing across chain links (None = nothing
    spent yet; a dict — even empty — = at least one billed request)."""
    if new is None:
        return acc
    if acc is None:
        return dict(new)
    merged = dict(new)
    merged["prompt_tokens"] = int(acc.get("prompt_tokens", 0)) + int(new.get("prompt_tokens", 0))
    merged["completion_tokens"] = int(acc.get("completion_tokens", 0)) + int(
        new.get("completion_tokens", 0)
    )
    return merged


@dataclass
class Verdict:
    """Judging result. scores shares the Task.score schema; decided=False
    means this judge could not decide (e.g. parse failure) and the next
    judge in the chain takes over as fallback."""

    scores: dict[str, float]
    decided: bool = True
    note: str | None = None
    # usage of the judging backend call. Convention: None = this verdict
    # consumed no backend request (no judge / pure-logic judge); a dict —
    # empty when the server omitted usage — = one billed request. The
    # evaluator bills it into the run's judge-channel cost so the extra backend
    # call per judged sample is visible rather than folded into the eval channel
    usage: Mapping[str, Any] | None = None


@dataclass
class JudgeContext:
    """Judge runtime context. backend is the judging-model slot (may differ
    from the evaluated model).

    variant / instruction / user_text are filled by the Evaluator on every
    judging call: variant serves protocol-aware judges (e.g. mcq shifted
    choices), instruction is the candidate prompt's original text (e.g. the
    judge prompt's "Original Task"), and user_text is the plain text the
    classify request actually rendered (when a baseline judge's "Original
    Task" needs the fully assembled text, use it directly to avoid drift
    from rebuilding it per config).
    """

    backend: InferenceBackend | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    variant: Variant | None = None
    instruction: str | None = None
    user_text: str | None = None


class Judge(ABC):
    """Standalone extension point: judges (response, prediction, sample)
    into a verdict/score.

    LLM-as-judge is just one member of the judge family; deterministic
    parsing, pairwise, and rubric judges extend along the same interface. A
    deterministic judge only needs to implement sync ``judge``; judges that
    call an inference backend (LLM judging) override async ``ajudge`` (the
    Evaluator always goes through ``ajudge`` in async contexts; the default
    implementation delegates to ``judge``).
    """

    name: ClassVar[str]

    @abstractmethod
    def judge(
        self, response: ChatResponse, prediction: Prediction, sample: Sample, ctx: JudgeContext
    ) -> Verdict: ...

    async def ajudge(
        self, response: ChatResponse, prediction: Prediction, sample: Sample, ctx: JudgeContext
    ) -> Verdict:
        return self.judge(response, prediction, sample, ctx)


@register_judge("chain")
class ChainedJudge(Judge):
    """Tries the chained judges in order, taking the first decided result;
    if none decides, decided=False."""

    name = "chain"

    def __init__(self, judges: list[Judge]) -> None:
        if not judges:
            raise ValueError("ChainedJudge requires at least one judge")
        self._judges = list(judges)

    @property
    def judges(self) -> list[Judge]:
        return list(self._judges)

    def judge(
        self, response: ChatResponse, prediction: Prediction, sample: Sample, ctx: JudgeContext
    ) -> Verdict:
        spent: Mapping[str, Any] | None = None
        for j in self._judges:
            verdict = j.judge(response, prediction, sample, ctx)
            spent = _merge_usage(spent, verdict.usage)
            if verdict.decided:
                return replace(verdict, usage=spent)
        return Verdict(scores={}, decided=False, note="no judge decided", usage=spent)

    async def ajudge(
        self, response: ChatResponse, prediction: Prediction, sample: Sample, ctx: JudgeContext
    ) -> Verdict:
        spent: Mapping[str, Any] | None = None
        for j in self._judges:
            verdict = await j.ajudge(response, prediction, sample, ctx)
            spent = _merge_usage(spent, verdict.usage)
            if verdict.decided:
                return replace(verdict, usage=spent)
        return Verdict(scores={}, decided=False, note="no judge decided", usage=spent)
