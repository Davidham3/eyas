"""split_think / NormalizingBackend contract tests (zero GPU).

Covers the three inline-think variants (both tags / missing open tag — the
engine-observed shape / unclosed truncation), idempotence, merging with the
reasoning_content field.
"""

from __future__ import annotations

import asyncio

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    SamplingConfig,
    user_message,
)
from eyas.backends.normalize import NormalizingBackend, split_think


def _req(request_id: str = "r1") -> ChatRequest:
    return ChatRequest(messages=(user_message("q"),), request_id=request_id)


class TestSplitThink:
    def test_no_think_is_noop(self):
        assert split_think("plain answer") == ("plain answer", None)

    def test_both_tags_stripped(self):
        text, reasoning = split_think("<think>\nlet me see\n</think>\n\n42")
        assert text == "42"
        assert reasoning == "let me see"

    def test_missing_open_tag(self):
        """Missing open tag: everything before the closing tag is thinking (the engine only emits the closing tag)."""
        text, reasoning = split_think("reasoning got beheaded</think>answer")
        assert text == "answer"
        assert reasoning == "reasoning got beheaded"

    def test_unclosed_think_is_truncation(self):
        """Unclosed: the generation window was fully consumed by thinking —
        empty answer, thinking fully retained."""
        text, reasoning = split_think("<think>thinking got cut off")
        assert text == ""
        assert reasoning == "thinking got cut off"

    def test_empty_think_block(self):
        assert split_think("<think></think>ans") == ("ans", None)

    def test_merges_with_field_reasoning(self):
        text, reasoning = split_think("<think>inline</think>ans", "field")
        assert text == "ans"
        assert reasoning == "field\n\ninline"

    def test_idempotent_on_mixed_form(self):
        once = split_think("a</think>b<think>c")
        twice = split_think(*once)
        assert once == twice


class TestNormalizingBackend:
    def test_strips_and_preserves_other_fields(self):
        class Inner(InferenceBackend):
            async def agenerate(self, requests, sampling: SamplingConfig) -> list[ChatResponse]:
                return [
                    ChatResponse(
                        text="<think>r</think>ans",
                        request_id=req.request_id,
                        usage={"total_tokens": 7},
                        raw={"echo": True},
                        reasoning="field",
                        finish_reason="stop",
                    )
                    for req in requests
                ]

        (resp,) = asyncio.run(NormalizingBackend(Inner()).agenerate([_req()], SamplingConfig()))
        assert resp.text == "ans"
        assert resp.reasoning == "field\n\nr"
        assert resp.finish_reason == "stop"
        assert resp.usage == {"total_tokens": 7}
        assert resp.raw == {"echo": True}

    def test_clean_response_unchanged(self):
        class Inner(InferenceBackend):
            async def agenerate(self, requests, sampling: SamplingConfig) -> list[ChatResponse]:
                return [ChatResponse(text="ans\n", request_id=req.request_id) for req in requests]

        (resp,) = asyncio.run(NormalizingBackend(Inner()).agenerate([_req()], SamplingConfig()))
        assert resp.text == "ans\n"  # no redundant strip when there is no think
        assert resp.reasoning is None
