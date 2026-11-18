"""Response normalization (single chokepoint): ``text`` is always the clean
answer; thinking content is kept separate from the answer.

Motivation: thinking-model responses come in two shapes — a dedicated field
(``message.reasoning_content``, DeepSeek-style servers, already extracted into
ChatResponse.reasoning by each backend) and inline ``<think>...</think>``
(client-side rendering engines plus some servers). Unstripped inline content
would pollute judge prompts and parsing (the evaluator's scoring chain consumes
response.text). All three inline variants are handled: both tags present, the
open tag missing (some engines emit only the close tag),
and unclosed (the generation window is fully eaten by thinking — the whole
segment is thinking and the answer is empty; together with finish_reason=length
this identifies truncation). Idempotent: calling twice on already-stripped text
is a no-op (cache-hit replays go through the same path).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from eyas.backends.base import (
    BackendWrapper,
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    SamplingConfig,
)

__all__ = ["NormalizingBackend", "split_think"]

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _scan_think_tags(text: str) -> tuple[str, list[str]]:
    """Scan text for open/close think tags, consuming them left to right.

    Returns the leftover answer text with all tag markup removed, plus the
    inline thinking segments in encounter order. Unclosed open tags absorb
    the remainder and end the scan; a close tag seen first (open tag missing)
    marks everything before it as thinking.
    """
    inline: list[str] = []
    while True:
        open_idx = text.find(_THINK_OPEN)
        close_idx = text.find(_THINK_CLOSE)
        if open_idx == -1 and close_idx == -1:
            break
        if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
            end = text.find(_THINK_CLOSE, open_idx)
            if end == -1:
                # Unclosed: everything remaining is thinking; consume it and stop
                inline.append(text[open_idx + len(_THINK_OPEN) :])
                text = text[:open_idx]
                break
            inline.append(text[open_idx + len(_THINK_OPEN) : end])
            text = text[:open_idx] + text[end + len(_THINK_CLOSE) :]
        else:
            # Open tag missing: everything before the close tag is thinking
            inline.append(text[:close_idx])
            text = text[close_idx + len(_THINK_CLOSE) :]
    return text, inline


def _merge_think(text: str, inline: list[str], reasoning: str | None) -> tuple[str, str | None]:
    """Build the final (answer, reasoning) pair from the extracted think segments.

    Field-extracted reasoning comes first; whitespace-only segments drop out.
    """
    if not inline:
        return text, reasoning
    text = text.strip()
    merged = "\n\n".join(part.strip() for part in inline if part.strip())
    if merged:
        reasoning = f"{reasoning}\n\n{merged}" if reasoning else merged
    return text, reasoning


def split_think(text: str, reasoning: str | None = None) -> tuple[str, str | None]:
    """Strip inline think blocks from text and merge them with the extracted reasoning (field content first)."""
    text, inline = _scan_think_tags(text)
    return _merge_think(text, inline, reasoning)


class NormalizingBackend(BackendWrapper, InferenceBackend):
    """Response normalization wrapper.

    Sits at the outermost layer of the assembly chain (raw→traced→normalized):
    stripping happens uniformly on every response, so text reaching the
    optimizer is always the clean answer regardless of engine or replay source.
    """

    def __init__(self, inner: InferenceBackend) -> None:
        super().__init__(inner)
        # declared limits pass through the wrapper chain (runtime arms the
        # planner guard from the outermost backend, raw→traced→normalized)
        self.context_window = inner.context_window
        self.max_images = inner.max_images

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        responses = await self.inner.agenerate(requests, sampling)
        out: list[ChatResponse] = []
        for resp in responses:
            text, reasoning = split_think(resp.text, resp.reasoning)
            out.append(replace(resp, text=text, reasoning=reasoning))
        return out
