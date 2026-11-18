"""VQA open-ended QA task (verifies the Task abstraction generalizes
beyond multiple-choice).

Data format: see ``tasks/datasets.py::load_vqa_samples`` — a
``data_dir/<split>.json`` array whose elements carry ``image`` /
``question`` / ``answers`` (annotated answer list, VQAv2 convention).

Semantic notes:
- request assembly mirrors McqTask (instruction with {image}/<image> markers
  stripped, image first) around the fixed skeleton
  ``{instruction}\nQuestion: {q}\nAnswer:``;
- parsing takes the first non-empty line and normalizes it as the answer
  (the official VQA eval's number-word normalization is not implemented —
  known deviation); empty output -> value=None, falling back to the judge;
- scoring uses standard VQA accuracy ``min(1, #equal answers/3)`` — both
  sides of the equality check go through ``normalize_vqa_answer``; without
  an annotation judge, parse failure scores 0.0 (framework convention);
- variants are meaningless for VQA (no choice order exists);
  build_request ignores variant; pair with the identity protocol;
- the default prompt states the short-answer contract on purpose: the scorer
  (first-line match against annotated short answers) grades a correct-but-verbose
  answer as 0 — the output-format clause is the task's contract to the model,
  exactly like KIE's default demands a JSON array (field text itself stays neutral;
  under this scorer, verbose answering scores far below short-phrase answering).
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import ClassVar

from eyas.backends.base import ChatRequest, ChatResponse, ImagePart, Part, TextPart, user_message
from eyas.core.objective import Objective
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.datasets import load_vqa_samples

__all__ = ["VqaTask", "normalize_vqa_answer"]

# Image placeholder markers that may appear in candidate prompts (stripped at
# render time, same rule as McqTask)
_IMAGE_MARKERS = ("{image}", "<image>")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_vqa_answer(text: str) -> str:
    """VQA answer normalization: lowercase, strip punctuation, collapse
    whitespace (official number-word normalization not implemented)."""
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


@register_task("vqa")
class VqaTask(Task):
    """Image open-ended QA task.

    Sample convention: inputs={"image": path, "question": question text},
    references={"answers": [annotated answers, ...]} (VQAv2 convention: >=1).
    """

    name = "vqa"
    objectives: ClassVar[tuple[Objective, ...]] = (
        Objective(key="accuracy", maximize=True, display_name="accuracy"),
    )
    default_prompt = (
        "Answer the question about the target image with a short factual phrase "
        "(one to five words)."
    )

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        images_dir: str | Path | None = None,
        samples: list[Sample] | None = None,
    ) -> None:
        if samples is None and (data_dir is None or images_dir is None):
            raise ValueError("VqaTask requires one of data_dir + images_dir or explicit samples")
        self._data_dir = data_dir
        self._images_dir = images_dir
        self._samples = samples
        self._split_cache: dict[str, list[Sample]] = {}

    def load_split(self, split: str) -> list[Sample]:
        if self._samples is not None:
            return list(self._samples)
        if split not in self._split_cache:
            assert self._data_dir is not None and self._images_dir is not None, (
                "data_dir and images_dir must be provided when samples is not"
            )
            self._split_cache[split] = load_vqa_samples(self._data_dir, split, self._images_dir)
        return list(self._split_cache[split])

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        instruction = prompt
        for marker in _IMAGE_MARKERS:
            instruction = instruction.replace(marker, "")
        text = f"{instruction.rstrip()}\nQuestion: {sample.inputs['question']}\nAnswer:"
        return ChatRequest(
            messages=(user_message(ImagePart(image=sample.inputs["image"]), TextPart(text)),),
            request_id=f"{sample.id}:{variant.id}",
        )

    def demo_media_parts(self, sample: Sample) -> list[Part]:
        """A few-shot demo carries the sample image, like evaluation."""
        return [ImagePart(image=str(sample.inputs["image"]))]

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        first_line = next((line.strip() for line in response.text.splitlines() if line.strip()), "")
        value = normalize_vqa_answer(first_line) or None
        return Prediction(value=value, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        if prediction.value is None:
            return {"accuracy": 0.0}
        answers = sample.references["answers"]
        eq = sum(1 for a in answers if normalize_vqa_answer(str(a)) == prediction.value)
        return {"accuracy": min(1.0, eq / 3.0)}
