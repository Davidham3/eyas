"""BackendWrapper pass-through contract: every wrapper in the
raw→traced→normalized chain must forward ``slot`` and ``close`` to the inner
backend. Pinning reach-through with a fake inner is the point — a wrapper that
swallows either makes engine release a silent no-op (the next engine start then
collides with an unreleased one; engine processes do not exit with GC)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    SamplingConfig,
)
from eyas.backends.normalize import NormalizingBackend
from eyas.engine.capability import ResourceProfile
from eyas.engine.planner import PlannedBackend, RequestPlanner
from eyas.runtime.trace import TracedBackend, TraceRecorder


class FakeInner(InferenceBackend):
    """Minimal inner engine: a slot sentinel, a close latch, and declared limits."""

    name = "fake"

    def __init__(self) -> None:
        self.slot = object()
        self.closed = False
        self.context_window = 123
        self.max_images = 2

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        return [ChatResponse(text="ok") for _ in requests]

    def close(self) -> None:
        self.closed = True


def wrap_normalized(inner: InferenceBackend, recorder: Any) -> InferenceBackend:
    return NormalizingBackend(inner)


def wrap_traced(inner: InferenceBackend, recorder: Any) -> InferenceBackend:
    return TracedBackend(inner, recorder, trace_slot="candidate")


def wrap_planned(inner: InferenceBackend, recorder: Any) -> InferenceBackend:
    return PlannedBackend(inner, RequestPlanner(ResourceProfile()))


@pytest.mark.parametrize("wrap", [wrap_normalized, wrap_traced, wrap_planned])
def test_slot_and_close_reach_inner(wrap: Any, tmp_path: Any) -> None:
    inner = FakeInner()
    outer = wrap(inner, TraceRecorder(tmp_path / "trace.jsonl"))
    assert outer.slot is inner.slot
    assert outer.context_window == 123  # declared limits pass through the chain too
    assert outer.max_images == 2
    outer.close()
    assert inner.closed
