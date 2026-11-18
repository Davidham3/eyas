from __future__ import annotations

from eyas.backends.base import ChatResponse
from eyas.judges.base import Judge, JudgeContext, Verdict
from eyas.registry import register_judge
from eyas.tasks.base import Prediction, Sample

__all__ = ["ExactMatchJudge"]


@register_judge("exact_match")
class ExactMatchJudge(Judge):
    """Deterministic judging: exact match of prediction.value against the
    reference label (default references["label"]).

    On parse failure (value=None), decided=False and the next judge in the
    chain (e.g. llm_judge) takes over as fallback. A sample whose references
    lack the configured key raises ValueError (fail fast, never a silent 0);
    a present-but-None value scores as an ordinary mismatch.
    """

    name = "exact_match"

    def __init__(self, reference_key: str = "label", case_sensitive: bool = False) -> None:
        self._reference_key = reference_key
        self._case_sensitive = case_sensitive

    def judge(
        self, response: ChatResponse, prediction: Prediction, sample: Sample, ctx: JudgeContext
    ) -> Verdict:
        if prediction.value is None:
            return Verdict(scores={}, decided=False, note="prediction unparseable")
        if self._reference_key not in sample.references:
            raise ValueError(
                f"exact_match: sample {sample.id!r} has no reference key {self._reference_key!r} "
                f"(available keys: {sorted(sample.references)}) — fix the judge's reference_key "
                "or the dataset annotations"
            )
        reference = sample.references[self._reference_key]
        pred, ref = prediction.value, reference
        if not self._case_sensitive:
            pred = pred.lower() if isinstance(pred, str) else pred
            ref = ref.lower() if isinstance(ref, str) else ref
        return Verdict(scores={"accuracy": float(pred == ref)}, note="exact_match")
