"""LLM judge for MCQ (classify -> text-only second pass -> normalized comparison).

Flow: classify (with image) -> raw output -> text-only second pass (same
model) -> ``match_prediction`` normalizes against the **shifted** choice
list -> compare against the original label. Runs for every variant (no
deterministic short-circuit); verified=None scores 0.

Judging-model slot: the Evaluator's ``judge_backend``; when None, the
evaluated model is reused (the "same-model judge" mode).

``variant`` selects the judging convention:
``variant="default"`` compact prompt + ``match_prediction`` longest-first
normalization. variant="cot" is the CoT/EvoPrompt baseline judging
convention: prompt blank lines are ``\\n\\n`` and fuzzy matching takes the
first hit in choice order (not longest-first). variant="opro" is the OPRO
baseline judging convention: the prompt matches cot (blank-line format)
while normalization uses ``match_prediction``'s longest-first rule (same as
default). With ``classify_original_task=True`` the "Original Task" section
uses ``ctx.user_text`` — the fully assembled text the classify request
actually rendered (auto-filled by the Evaluator); default False passes the
bare instruction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    user_message,
)
from eyas.judges.base import Judge, JudgeContext, Verdict
from eyas.registry import register_judge
from eyas.tasks.base import Prediction, Sample
from eyas.tasks.mcq import match_prediction

__all__ = ["McqLLMJudge", "build_judge_prompt", "build_cot_judge_prompt"]


def build_judge_prompt(original_prompt: str, prediction: str, choices: Sequence[str]) -> str:
    """Build the plain-text prompt for the judge stage.

    Note Choices uses Python list repr format (different from the classify
    stage's ``", ".join(f"'{c}'"`` format); judging and classify rendering
    keep their own conventions.
    """
    return (
        f"Original Task: {original_prompt}\n"
        f'Model\'s Prediction: "{prediction}"\n'
        f"Based on the prediction above, which of the following choices does it correspond to?\n"
        f"Choices: {list(choices)}\n"
        f"Output ONLY the chosen category name from the list. "
        f"If the prediction is invalid or unclear, output 'None'."
    )


def build_cot_judge_prompt(original_prompt: str, prediction: str, choices: Sequence[str]) -> str:
    """Judge prompt for the CoT baseline (in the compact version no blank line
    remains after the Prediction/Choices lines).

    Difference from the default version: a blank line (``\\n\\n``) follows
    the Prediction line and the Choices line.
    """
    return (
        f"Original Task: {original_prompt}\n"
        f'Model\'s Prediction: "{prediction}"\n\n'
        f"Based on the prediction above, which of the following choices does it correspond to?\n"
        f"Choices: {list(choices)}\n\n"
        f"Output ONLY the chosen category name from the list. "
        f"If the prediction is invalid or unclear, output 'None'."
    )


def match_judge_result_cot(result: str, choices: Sequence[str]) -> str | None:
    """Judging normalization for the CoT baseline (the same rule the reference
    evaluation applies).

    After strip, exact hit -> otherwise take the first substring hit in
    **choice list order** (unlike match_prediction's longest-first rule).
    """
    result = result.strip()
    if result == "None" or result in choices:
        return result if result != "None" else None
    for choice in choices:
        if choice in result:
            return choice
    return None


@register_judge("mcq_llm")
class McqLLMJudge(Judge):
    """Maps the classify free-text output to the current variant's choice list
    via LLM judging."""

    name = "mcq_llm"

    def __init__(
        self,
        variant: Literal["default", "cot", "opro"] = "default",
        *,
        classify_original_task: bool = False,
    ) -> None:
        self.variant = variant
        self._classify_original_task = classify_original_task

    def judge(
        self, response: ChatResponse, prediction: Prediction, sample: Sample, ctx: JudgeContext
    ) -> Verdict:
        raise NotImplementedError("McqLLMJudge requires async backend calls; use it via ajudge")

    async def ajudge(
        self, response: ChatResponse, prediction: Prediction, sample: Sample, ctx: JudgeContext
    ) -> Verdict:
        """Resolve the required context slots, run the single text-only judging
        call, turn its reply into the verdict.

        The one ``await`` stays here (no helper may move an await across an
        exception/cancellation scope: that silently changes cancellation
        semantics); every decision is a sync helper below.
        """
        backend, original_task = self._resolve_slots(ctx)
        shifted = list(sample.inputs["choices"])
        request = self._judge_request(response, original_task, prediction.raw, shifted)
        (resp,) = await backend.agenerate([request], ctx.sampling)
        return self._verdict(resp, sample, shifted)

    def _resolve_slots(self, ctx: JudgeContext) -> tuple[InferenceBackend, str]:
        """Check the context slots this judge hard-depends on, in order, and
        return ``(judging backend, "Original Task" text)``.

        Checking and returning together keeps the fail-fast order (backend ->
        variant -> instruction -> user_text) intact *and* leaves both values
        narrowed for the type checker.
        """
        if ctx.backend is None:
            raise RuntimeError(
                "McqLLMJudge requires a judge backend (the Evaluator's judge_backend slot)"
            )
        if ctx.variant is None:
            raise RuntimeError(
                "McqLLMJudge requires ctx.variant (filled by the Evaluator when rendering variants)"
            )
        if ctx.instruction is None:
            raise RuntimeError(
                "McqLLMJudge requires ctx.instruction (the candidate prompt's original text)"
            )
        if self._classify_original_task:
            if ctx.user_text is None:
                raise RuntimeError(
                    "classify_original_task=True requires ctx.user_text (auto-filled by "
                    "the Evaluator; pass the rendered classify text when constructing "
                    "JudgeContext manually)"
                )
            return ctx.backend, ctx.user_text
        return ctx.backend, ctx.instruction

    def _judge_request(
        self, response: ChatResponse, original_task: str, raw: str, shifted: list[str]
    ) -> ChatRequest:
        """Assemble the judging request: the variant's prompt convention over
        the **shifted** choices, tagged ``judge:`` so backends/tests can route
        the judging channel, and chained to the classify request id."""
        if self.variant == "default":
            prompt = build_judge_prompt(original_task, raw, shifted)
        else:
            prompt = build_cot_judge_prompt(original_task, raw, shifted)
        return ChatRequest(
            messages=(user_message(prompt),),
            request_id=f"judge:{response.request_id}",
        )

    def _verdict(self, resp: ChatResponse, sample: Sample, shifted: list[str]) -> Verdict:
        """Map the judging reply onto the sample's label."""
        if resp.finish_reason == OVER_WINDOW_FINISH:
            # planner chokepoint rejection: the request never left the process,
            # so it is UNBILLED (usage None per the Verdict convention) and the
            # verdict is not decided. Two fallback paths own the outcome:
            # inside a ChainedJudge the next link takes over (decided=False);
            # as the sole judge the sample lands on the evaluator's parse-fail
            # zero-score convention.
            return Verdict(scores={}, decided=False, note="planner-rejected", usage=None)
        if self.variant == "cot":
            verified = match_judge_result_cot(resp.text, shifted)
        else:
            verified = match_prediction(resp.text, shifted)
        return Verdict(
            scores={"accuracy": float(verified == sample.references["label"])},
            note=verified,
            # a call was issued: bill one request even when the server omitted
            # usage (empty dict = tokens unknown, per the Verdict convention)
            usage=dict(resp.usage) if resp.usage else {},
        )
