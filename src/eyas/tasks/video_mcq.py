"""Video MCQ task (NExT-QA style): the MCQ skeleton plus a per-sample Question line.

Sample convention: inputs={"video": path, "question": question text,
"choices": [choice names]}, references={"label": correct choice name}.

Semantic notes:
- the task is McqTask with the media hook swapped: ``media_key="video"`` and
  ``_media_part`` returns a VideoPart (the base task's ImagePart would
  mislabel the clip), so classify/raw-template/OPRO assembly all come from
  the base class;
- every non-raw assembly inserts one ``Question: {q}`` line above the
  ``Choices:`` line (the OPRO mode feeds that line into gen_scorer_prompt and
  ends with ``Target Video:`` instead of ``Target Image:``);
- raw_template mode additionally requires a ``{question}`` placeholder and
  accepts both clip markers (``{video}`` / ``<video>``) plus the image
  spellings the base class strips, so an optimizer seeded on image prompts
  still renders.

Demos carry no clip: a per-candidate request with N videos multiplies decode
cost and trips engine media limits.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from eyas.backends.base import Part, TextPart, VideoPart
from eyas.registry import register_task
from eyas.tasks.base import Sample
from eyas.tasks.datasets import load_video_mcq_samples
from eyas.tasks.mcq import McqTask, _require_input, gen_scorer_prompt

__all__ = ["VideoMcqTask"]

# Media placeholder markers that may appear in candidate prompts (the clip
# markers plus the image spellings inherited from McqTask's marker stripping)
_MEDIA_MARKERS = ("{image}", "<image>", "{video}", "<video>")


@register_task("video_mcq")
class VideoMcqTask(McqTask):
    """Video QA over a fixed choice list: the MCQ skeleton plus one Question line.

    Demos carry no clip: a per-candidate request with N videos multiplies decode
    cost and trips engine media limits.
    """

    name = "video_mcq"
    default_prompt = "Classify the content of the target video."

    def __init__(
        self,
        dataset: str | None = None,
        *,
        data_dir: str | Path | None = None,
        images_dir: str | Path | None = None,
        videos_dir: str | Path | None = None,
        samples: list[Sample] | None = None,
        train_per_group: int = 50,
        train_sample_seed: int = 42,
        assistant_prefix: str | None = None,
        target_image_suffix: bool = False,
        instruction_pos: str | None = None,
        include_qa: bool = False,
        raw_template: bool = False,
        media_key: str = "video",
        loader: Callable[[Any, str, Any], list[Sample]] | None = load_video_mcq_samples,
    ) -> None:
        super().__init__(
            dataset,
            data_dir=data_dir,
            images_dir=images_dir,
            videos_dir=videos_dir,
            samples=samples,
            train_per_group=train_per_group,
            train_sample_seed=train_sample_seed,
            assistant_prefix=assistant_prefix,
            target_image_suffix=target_image_suffix,
            instruction_pos=instruction_pos,
            include_qa=include_qa,
            raw_template=raw_template,
            media_key=media_key,
            loader=loader,
        )

    def _media_part(self, sample: Sample) -> Part:
        """The sample's clip as a VideoPart (the base hook would build an
        ImagePart from the media key)."""
        return VideoPart(_require_input(sample, self._media_key))

    def demo_media_parts(self, sample: Sample) -> list[Part]:
        """Demos carry no clip: a per-candidate request with N videos multiplies
        decode cost and trips engine media limits."""
        return []

    def _render_classify_text(
        self, prompt: str, sample: Sample, choices: list[str], choices_str: str
    ) -> str:
        instruction = prompt
        for marker in _MEDIA_MARKERS:
            instruction = instruction.replace(marker, "")
        question = _require_input(sample, "question")
        line = f"Question: {question}"
        if self._instruction_pos is not None:
            return gen_scorer_prompt(
                instruction,
                choices,
                self._instruction_pos,
                self._include_qa,
                question=line,
                target="Target Video:",
            )
        tail = "\nTarget Video:" if self._target_image_suffix else ""
        return f"{instruction}\n{line}\nChoices: [{choices_str}]{tail}"

    def _render_raw_template(
        self, prompt: str, sample: Sample, choices_str: str, media_part: Part
    ) -> list[Part]:
        if "{choices}" not in prompt:
            raise ValueError(
                "raw_template mode requires the prompt to contain a {choices} placeholder"
            )
        if "{question}" not in prompt:
            raise ValueError(
                "video raw_template mode requires the prompt to contain a {question} placeholder"
            )
        text = prompt.replace("{choices}", choices_str).replace(
            "{question}", _require_input(sample, "question")
        )
        marker = next((m for m in _MEDIA_MARKERS if m in text), None)
        if marker is None:
            return [media_part, TextPart(text)]
        parts: list[Part] = []
        for idx, segment in enumerate(text.split(marker)):
            if idx:
                # vLLM re-encodes the prompt around the frame block when the
                # token-level target misses (InternVL's <video> is plain text);
                # the re-encoded block is only findable when it starts a line,
                # so pin a newline before an in-place clip. A media-first
                # template already starts on one.
                if parts:
                    parts.append(TextPart("\n"))
                parts.append(media_part)
            if segment:
                parts.append(TextPart(segment))
        return parts
