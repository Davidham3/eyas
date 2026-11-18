"""Modality-mixed MCQ: one split, seven per-sample modality compositions.

combo = the non-empty subset of {text, image, video} a sample carries; "text"
is the sample's own passage (instruction/question/choices are the query and
are always present, never part of the combo). Non-raw rendering is fixed
order: passage -> images -> videos -> instruction/question/choices, and the
passage stays its own TextPart (the text-only combo is exactly two text
parts). In raw_template mode the {text}/{image}/{video} markers are replaced
in place and a marker whose modality the sample lacks disappears (an SPO
genotype may name all three; it must render every combo); a payload the
template names no marker for -- passage, images or clips -- is placed at the
head of the message in the non-raw order, so a combo never loses a payload.
Adjacent raw text segments merge into one TextPart; in raw mode a clip that
does not start the message is preceded by exactly one newline (vLLM/InternVL
re-encodes the prompt around the frame block and the block is only findable
when it starts a line -- same rule as VideoMcqTask; the non-raw order is the
plain passage-then-media order and inserts none).

Demos carry images only: the planner charges 16,384 tok per clip, so two demo
clips plus the evaluated clip can never fit a 32k window.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eyas.backends.base import (
    ChatRequest,
    ImagePart,
    Part,
    TextPart,
    VideoPart,
    assistant_message,
    user_message,
)
from eyas.registry import register_task
from eyas.tasks.base import Sample, Variant
from eyas.tasks.datasets import MIXED_COMBOS, load_mixed_mcq_samples
from eyas.tasks.mcq import McqTask, _require_input, gen_scorer_prompt

__all__ = ["MIXED_COMBOS", "MixedMcqTask"]

# Every modality placeholder that may appear in a candidate prompt: the in-place
# raw-template markers plus the image spellings the base class strips.
_MARKERS_TO_STRIP = ("{image}", "<image>", "{video}", "<video>", "{text}")

_MARKER_RE = re.compile(r"(\{text\}|\{image\}|\{video\}|<image>|<video>)")

_MARKER_KINDS = {
    "{text}": "text",
    "{image}": "image",
    "<image>": "image",
    "{video}": "video",
    "<video>": "video",
}


def _push_text(parts: list[Part], text: str) -> None:
    """Append text, merging into the previous part when it is text too (one
    text segment must never be split across TextParts, and no empty TextPart
    is ever pushed)."""
    if not text:
        return
    if parts and isinstance(parts[-1], TextPart):
        parts[-1] = TextPart(parts[-1].text + text)
    else:
        parts.append(TextPart(text))


def _push_clip(parts: list[Part], clip: Part) -> None:
    """Insert a clip, pinned to the start of a line unless it opens the message
    (vLLM re-encodes the prompt around the frame block when the token-level
    target misses -- InternVL's <video> is plain text -- and the re-encoded
    block is only findable when it starts a line; same rule as VideoMcqTask)."""
    if parts and not (isinstance(parts[-1], TextPart) and parts[-1].text.endswith("\n")):
        parts.append(TextPart("\n"))
    parts.append(clip)


@register_task("mixed_mcq")
class MixedMcqTask(McqTask):
    """MCQ over one split whose samples each declare their own modality combo.

    Sample convention: inputs={"passage": str, "images": [paths], "videos":
    [paths], "question": str, "choices": [choice names], "combo": str},
    references={"label": correct choice name}. The loader is
    ``load_mixed_mcq_samples`` (media paths relative to the data dir's parent).
    """

    name = "mixed_mcq"
    default_prompt = "Answer the question about the given sample."

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
        media_key: str = "image",
        loader: Callable[[Any, str, Any], list[Sample]] | None = load_mixed_mcq_samples,
    ) -> None:
        for key, value in (("images_dir", images_dir), ("videos_dir", videos_dir)):
            if value is not None:
                raise ValueError(
                    f"MixedMcqTask derives its media root from data_dir's parent; "
                    f"{key} must not be set (it would double-prefix every media path)"
                )
        super().__init__(
            dataset,
            data_dir=data_dir,
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

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        choices = list(_require_input(sample, "choices"))
        choices_str = ", ".join(f"'{c}'" for c in choices)
        if self._raw_template:
            parts = self._render_raw_parts(prompt, sample, choices_str)
        else:
            parts = self._render_parts(prompt, sample, choices, choices_str)
        messages = [user_message(*parts)]
        if self._assistant_prefix is not None:
            messages.append(assistant_message(self._assistant_prefix))
        return ChatRequest(messages=tuple(messages), request_id=f"{sample.id}:{variant.id}")

    def _media_parts(self, sample: Sample, kind: str) -> list[Part]:
        key = "images" if kind == "image" else "videos"
        cls = ImagePart if kind == "image" else VideoPart
        return [cls(value) for value in sample.inputs.get(key) or ()]

    def _render_parts(
        self, prompt: str, sample: Sample, choices: list[str], choices_str: str
    ) -> list[Part]:
        """Non-raw assembly: passage, then images, then clips, then the
        instruction/question/choices text."""
        parts: list[Part] = []
        passage = sample.inputs.get("passage") or ""
        if passage:
            parts.append(TextPart(passage))
        parts.extend(self._media_parts(sample, "image"))
        for clip in self._media_parts(sample, "video"):
            _push_clip(parts, clip)
        text = self._render_classify_text(prompt, sample, choices, choices_str)
        if text:
            parts.append(TextPart(text))
        return parts

    def _render_classify_text(
        self, prompt: str, sample: Sample, choices: list[str], choices_str: str
    ) -> str:
        """Query text: every modality marker stripped, then the OPRO scorer
        assembly, the ``Target:`` suffix assembly, or the plain instruction +
        question + choices block."""
        instruction = prompt
        for marker in _MARKERS_TO_STRIP:
            instruction = instruction.replace(marker, "")
        line = f"Question: {_require_input(sample, 'question')}"
        if self._instruction_pos is not None:
            return gen_scorer_prompt(
                instruction,
                choices,
                self._instruction_pos,
                self._include_qa,
                question=line,
                target="Target:",
            )
        tail = "\nTarget:" if self._target_image_suffix else ""
        return f"{instruction}\n{line}\nChoices: [{choices_str}]{tail}"

    def _render_raw_parts(self, prompt: str, sample: Sample, choices_str: str) -> list[Part]:
        """SPO full template: {question}/{choices} filled per sample, the
        {text}/{image}/{video} markers replaced in place (a marker for a
        modality the sample lacks vanishes); payloads the template names no
        marker for go to the head of the message.

        Named apart from the base class's ``_render_raw_template`` hook: that
        one places a single media part and a mixed sample has no single one.
        """
        for required in ("{choices}", "{question}"):
            if required not in prompt:
                raise ValueError(f"mixed raw_template mode requires a {required} placeholder")
        text = prompt.replace("{choices}", choices_str).replace(
            "{question}", _require_input(sample, "question")
        )
        images = self._media_parts(sample, "image")
        videos = self._media_parts(sample, "video")
        named = {"text": False, "image": False, "video": False}
        parts: list[Part] = []
        chunks = _MARKER_RE.split(text)
        for pos, chunk in enumerate(chunks):
            kind = _MARKER_KINDS.get(chunk)
            if kind is None:
                _push_text(parts, chunk)
                continue
            first_naming = not named[kind]
            named[kind] = True
            # a genotype can mutate into naming one marker twice; each payload is
            # emitted once (a second clip would exceed the 32k planner window and
            # score 0 with no visible cause)
            if not first_naming:
                continue
            if kind == "text":
                _push_text(parts, self._passage_between(sample, parts, chunks, pos))
            elif kind == "image":
                parts.extend(images)
            else:
                for clip in videos:
                    _push_clip(parts, clip)
        head: list[Part] = []
        if not named["text"]:
            _push_text(head, sample.inputs.get("passage") or "")
        if not named["image"]:
            head.extend(images)
        if not named["video"]:
            for clip in videos:
                _push_clip(head, clip)
        return head + parts

    def _passage_between(
        self, sample: Sample, parts: list[Part], chunks: list[str], pos: int
    ) -> str:
        """The passage for a ``{text}`` marker, kept off the template prose.

        A seed like ``"{text}{image}{video}Answer the question."`` names the
        marker with no surrounding whitespace, so the passage would fuse into
        the prose ("The object is blue.Answer ..."); one newline is added on
        each side that would otherwise touch, and the merge rule keeps it one
        TextPart. Markers between the passage and the following prose are
        skipped when the sample has no payload for them -- a marker that
        renders nothing cannot separate the two.
        """
        passage = sample.inputs.get("passage") or ""
        if not passage:
            return ""
        prev = parts[-1].text if parts and isinstance(parts[-1], TextPart) else ""
        if prev and not prev[-1].isspace():
            passage = "\n" + passage
        following = self._next_text_after(sample, chunks, pos)
        if following[:1] and not following[0].isspace() and not passage[-1].isspace():
            passage += "\n"
        return passage

    def _next_text_after(self, sample: Sample, chunks: list[str], pos: int) -> str:
        """First text that will actually follow ``chunks[pos]`` in the message:
        empty chunks and markers the sample renders nothing for are skipped, and
        a rendered media part ends the search (it separates the segments)."""
        for chunk in chunks[pos + 1 :]:
            kind = _MARKER_KINDS.get(chunk)
            if kind is None:
                if chunk:
                    return chunk
                continue
            if kind == "text" or self._media_parts(sample, kind):
                return ""
        return ""

    def demo_media_parts(self, sample: Sample) -> list[Part]:
        """Demos carry the sample's images but never its clips (the planner
        charges 16,384 tok per clip: two demo clips plus the evaluated clip
        can never fit a 32k window). At most one image: APE writes one
        ``[Image k]`` line per demo and its ``ResourceProfile`` caps images at
        ``demos_per_prompt``, so a multi-image demo would overflow it."""
        return self._media_parts(sample, "image")[:1]
