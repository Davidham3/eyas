"""Request planner: single chokepoint for resources and context.

Every ``ChatRequest`` headed for the engine must pass through
``RequestPlanner.plan``: the multi-image cap and context-reserve checks live in
one place rather than inside individual methods. Over-limit requests become a
**plan-time typed result** ``Rejected``, judged in exactly one place, instead of
a runtime string sentinel (``OVER_WINDOW_FINISH``).

Videos are not estimated frame-by-frame here: one ``VideoPart`` is charged a
fixed coarse budget (a conservative upper bound), and exact accounting happens
engine-side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    BackendWrapper,
    ChatRequest,
    ChatResponse,
    ImagePart,
    InferenceBackend,
    SamplingConfig,
    TextPart,
    VideoPart,
)
from eyas.engine.capability import ResourceProfile
from eyas.logging import get_logger
from eyas.watchdog import bump

__all__ = ["PlannedBackend", "PlannedRequest", "Rejected", "RequestPlanner"]

_log = get_logger(__name__)

#: Per-image character allowance for the window guard: one image is charged
#: ``_IMAGE_CHAR_BUDGET // 4`` ≈ 256 estimated tokens. Images are not tokenized
#: here — the estimate only needs to be generous enough to keep a request inside
#: the context window; exact accounting happens in the backend's own tokenizer.
_IMAGE_CHAR_BUDGET = 1024

#: Per-video character allowance for the window guard: one video is charged
#: ``_VIDEO_CHAR_BUDGET // 4`` ≈ 16384 estimated tokens. A 32-frame NExT-QA
#: clip measured ~11.6k real tokens on Qwen3.5/32k, so the charge must sit
#: above that for the guard to bite: a request repeating the media placeholder
#: (2+ clips) cannot fit the window and must be rejected here, because vLLM
#: raises ``VLLMValidationError`` at ``add_request`` and aborts the run. One
#: clip plus a normal prompt stays well inside the window; it is a guard
#: threshold, not a tokenizer.
_VIDEO_CHAR_BUDGET = 65536


@dataclass(frozen=True)
class PlannedRequest:
    request: ChatRequest


@dataclass(frozen=True)
class Rejected:
    """Plan-time rejection (typed, not an exception): the rest of the batch is
    unaffected; the caller scores it under the 0-score convention. ``reason``
    is the diagnostic payload (fail-fast culture: a rejection must explain
    itself)."""

    reason: str


def _count_images(request: ChatRequest) -> int:
    n = 0
    for message in request.messages:
        n += sum(1 for part in message.content if isinstance(part, ImagePart))
    return n


def _estimate_tokens(request: ChatRequest) -> int:
    """Rough token estimate: char count // 4 (sufficient for window/reserve
    guard-threshold checks, not exact billing)."""
    chars = 0
    for message in request.messages:
        for part in message.content:
            if isinstance(part, TextPart):
                chars += len(part.text)
            elif isinstance(part, ImagePart):
                chars += _IMAGE_CHAR_BUDGET
            else:  # VideoPart — the rest of a closed union
                assert isinstance(part, VideoPart), part
                chars += _VIDEO_CHAR_BUDGET
    return max(1, chars // 4)


class RequestPlanner:
    """Plan a single request against the ``ResourceProfile``; over-limit
    returns ``Rejected`` (does not raise)."""

    def __init__(self, profile: ResourceProfile, context_window: int | None = None) -> None:
        self.profile = profile
        self.context_window = context_window
        self._rejected: dict[str, int] = {}

    @property
    def reject_counts(self) -> dict[str, int]:
        """Rejections by reason (aggregate counter feeding the DEBUG report —
        ``plan`` is the only writer; callers read a copy)."""
        return dict(self._rejected)

    def _reject(self, reason: str) -> Rejected:
        # DEBUG breadcrumb per rejection (counts/reason only — no payloads);
        # the same call bumps the watchdog so "console silent" stays
        # distinguishable from "zero progress" once the level is open
        _log.debug("[planner] rejected: %s", reason)
        bump(f"[planner] rejected reason={reason}")
        self._rejected[reason] = self._rejected.get(reason, 0) + 1
        return Rejected(reason=reason)

    def plan(self, request: ChatRequest) -> PlannedRequest | Rejected:
        n_images = _count_images(request)
        est = _estimate_tokens(request)

        if self.profile.max_images is not None and n_images > self.profile.max_images:
            return self._reject(f"images per request {n_images} > limit {self.profile.max_images}")
        if self.context_window is not None:
            usable = self.context_window - self.profile.ctx_reserve
            if est > usable:
                return self._reject(
                    f"estimated input {est} tok exceeds usable window {usable}"
                    f" (window {self.context_window} - reserve {self.profile.ctx_reserve})"
                )
        return PlannedRequest(request=request)


class PlannedBackend(BackendWrapper, InferenceBackend):
    """Adapter: force every request reaching the wrapped backend through the
    planner chokepoint.

    Exists for the paths that legitimately hold a *backend slot* rather than the
    evaluator — a judge via ``JudgeContext.backend``, pairwise games via
    ``CompareEvalModel`` — so they cannot physically bypass the planner.
    A rejected request gets an empty ``ChatResponse`` with
    ``finish_reason=OVER_WINDOW_FINISH`` (the caller's typed no-verdict /
    parse-fail signal — the same convention the evaluation scoring path
    applies), and the rest of the batch is unaffected. Requests the planner
    accepts are forwarded UNCHANGED: wrapping never alters a request byte.

    ``slot`` and ``close`` are forwarded, never swallowed: this adapter also
    sits in front of a real engine, so engine release must not become a silent
    no-op.
    """

    name = "planned"

    def __init__(self, inner: InferenceBackend, planner: RequestPlanner) -> None:
        super().__init__(inner)
        self._planner = planner
        # declared limits pass through (transparent adapter)
        self.context_window = inner.context_window
        self.max_images = inner.max_images

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        idx_map: list[int] = []
        planned_reqs: list[ChatRequest] = []
        out: list[ChatResponse] = []
        for i, request in enumerate(requests):
            planned = self._planner.plan(request)
            if isinstance(planned, PlannedRequest):
                planned_reqs.append(planned.request)
                idx_map.append(i)
                out.append(ChatResponse(text=""))  # placeholder, overwritten below
            else:
                assert isinstance(planned, Rejected)
                # rejection already logged by RequestPlanner._reject (no second crumb)
                out.append(
                    ChatResponse(text="", finish_reason=OVER_WINDOW_FINISH)
                )  # placeholder, overwritten for accepted slots
        if planned_reqs:
            responses = await self.inner.agenerate(planned_reqs, sampling)
            if len(responses) != len(planned_reqs):
                raise RuntimeError(
                    f"planned backend response count mismatch: "
                    f"sent {len(planned_reqs)} got {len(responses)}"
                )
            for slot, response in zip(idx_map, responses, strict=True):
                out[slot] = response
        return out
