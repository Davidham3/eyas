"""Numeric open-ended math QA task (GSM8K-style; text-only).

Data format: see ``tasks/datasets.py::load_math_qa_samples`` — a
``data_dir/<split>.json`` array whose elements carry ``question`` /
``answer``, where ``answer`` is already the final numeric value (GSM8K's
``####`` marker is stripped by the preparation script).

Semantic notes:
- request assembly mirrors VqaTask without the media part: the instruction
  (candidate prompt) with {image}/<image>/{video} markers stripped, then the
  fixed skeleton ``{instruction}\\nQuestion: {q}\\nAnswer:`` as a single
  TextPart (an instruction that is empty after rstrip is dropped, so the text
  starts at ``Question:``); ``assistant_prefix`` appends a continuation
  assistant turn when non-empty (CoT baseline, mirroring McqTask);
- ``raw_template=True`` (SPO full-template board): the candidate prompt is the
  complete user text and must contain ``{question}``; a ``{choices}``
  placeholder is rejected — this task has no choice list, so an MCQ template
  would silently render a meaningless prompt;
- parsing takes the last ``Answer: <number>`` line; with no such line it falls
  back to the last number in the text — a known deviation: the fallback can pick
  a reasoning step's number instead of the final answer, which is exactly what
  the default prompt's ``Answer:`` line contract prevents; no number at all ->
  value=None, falling back to the judge;
- numbers are normalized (``$``/``,``/whitespace stripped, integral values as
  int) and compared exactly with a 1e-6 tolerance; parse failure scores 0.0
  (framework convention);
- variants are meaningless here (no choice order exists); build_request
  ignores variant; pair with the identity protocol.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from eyas.backends.base import ChatRequest, ChatResponse, TextPart, assistant_message, user_message
from eyas.core.objective import Objective
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.datasets import load_math_qa_samples

__all__ = ["MathQaTask", "parse_numeric_answer"]

# Media placeholder markers that may appear in candidate prompts (stripped at
# render time; this task has no media part)
_MEDIA_MARKERS = ("{image}", "<image>", "{video}", "<video>")

# Last "Answer: <X>" line wins; both ASCII and full-width colons are accepted
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_numeric_answer(text: str) -> int | float | None:
    """Extract the final numeric answer: the last ``Answer:`` line's last
    number, else the text's last number; ``None`` when there is no number."""
    matches = list(_ANSWER_LINE.finditer(text))
    if matches:
        inner = _NUMBER.findall(matches[-1].group(1))
        if not inner:
            return None
        token = inner[-1]
    else:
        tokens = _NUMBER.findall(text)
        if not tokens:
            return None
        token = tokens[-1]
    token = token.replace(",", "")
    return float(token) if "." in token else int(token)


@register_task("math_qa")
class MathQaTask(Task):
    """Text-only numeric open-ended QA task.

    Sample convention: inputs={"question": question text},
    references={"answer": final numeric value as a string}.
    """

    name = "math_qa"
    objectives: ClassVar[tuple[Objective, ...]] = (
        Objective(key="accuracy", maximize=True, display_name="accuracy"),
    )
    default_prompt = (
        "Solve the math problem. Show your reasoning, then write the final answer "
        "on the last line exactly as `Answer: <number>`."
    )

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        samples: list[Sample] | None = None,
        assistant_prefix: str | None = None,
        raw_template: bool = False,
    ) -> None:
        if samples is None and data_dir is None:
            raise ValueError("MathQaTask requires one of data_dir or explicit samples")
        self._data_dir = data_dir
        self._samples = samples
        # CoT baseline continuation primer: when set, append an assistant turn
        self._assistant_prefix = assistant_prefix
        # SPO full-template assembly: the candidate prompt is the complete user text
        self._raw_template = raw_template
        self._split_cache: dict[str, list[Sample]] = {}

    def load_split(self, split: str) -> list[Sample]:
        if self._samples is not None:
            return list(self._samples)
        if split not in self._split_cache:
            assert self._data_dir is not None, "data_dir must be provided when samples is not"
            self._split_cache[split] = load_math_qa_samples(self._data_dir, split)
        return list(self._split_cache[split])

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        question = str(sample.inputs["question"])
        if self._raw_template:
            text = self._render_raw_template(prompt, question)
        else:
            text = self._render_text(prompt, question)
        messages = [user_message(TextPart(text))]
        if self._assistant_prefix:
            messages.append(assistant_message(self._assistant_prefix))
        return ChatRequest(
            messages=tuple(messages),
            request_id=f"{sample.id}:{variant.id}",
        )

    def _render_text(self, prompt: str, question: str) -> str:
        """Plain assembly: media markers stripped, instruction (rstripped) then
        the fixed Question/Answer skeleton."""
        instruction = prompt
        for marker in _MEDIA_MARKERS:
            instruction = instruction.replace(marker, "")
        head = instruction.rstrip()
        return (
            f"{head}\nQuestion: {question}\nAnswer:" if head else f"Question: {question}\nAnswer:"
        )

    def _render_raw_template(self, prompt: str, question: str) -> str:
        """SPO full template: ``{question}`` is required and ``{choices}`` is
        rejected (this task has no choice list)."""
        if "{question}" not in prompt:
            raise ValueError(
                "raw_template mode requires the prompt to contain a {question} placeholder, "
                f"got {prompt[:60]!r}"
            )
        if "{choices}" in prompt:
            raise ValueError(
                "raw_template mode for math_qa rejects a {choices} placeholder "
                f"(this task has no choices), got {prompt[:60]!r}"
            )
        return prompt.replace("{question}", question)

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=parse_numeric_answer(response.text), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        gold = parse_numeric_answer(str(sample.references["answer"]))
        if prediction.value is None or gold is None:
            return {"accuracy": 0.0}
        return {"accuracy": float(abs(prediction.value - gold) <= 1e-6)}
