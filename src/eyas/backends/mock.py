from __future__ import annotations

from collections.abc import Callable, Sequence

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    InferenceBackend,
    LoadedImage,
    SamplingConfig,
    TextPart,
    VideoPart,
)
from eyas.registry import register_backend

__all__ = [
    # only the mock's own surface: the message types live at eyas.backends.base
    # (single source); the decorator is re-exported so a mock-backend author
    # gets template + decorator from one module.
    "ScriptedMockBackend",
    "register_backend",
]


@register_backend("mock")
class ScriptedMockBackend(InferenceBackend):
    """Deterministic scripted stub for tests and demos.

    Two modes, pick one:
    - responses: a fixed script consumed in request order; exhaustion raises
      AssertionError immediately, unless cycle=True (loops back to the script
      start for reuse — used by demo configs; default False keeps the fail-fast
      behavior).
    - responder: a function (request, sampling, index) -> ChatResponse, suited
      to closed-loop tests that "answer deterministically from the input".

    Every request is recorded in full: ``recorded_requests`` holds
    ``(rendered_text, per-message part counts)`` per request in call order
    (``render_request_text`` renders every message with its role, TextParts
    concatenated verbatim, ImageParts as ``<image:path>`` text descriptors
    (prefetched payloads render as ``<image:loaded:path>``), and VideoParts as
    ``<video:path>``). The full-text recording lets resume-fidelity tests
    fingerprint the exact request stream an interrupted attempt issued and check
    that the resumed run replays it.

    Every optimizer / evaluator end-to-end contract test must run against this
    stub, with no GPU dependency.
    """

    name = "mock"

    def __init__(
        self,
        responses: Sequence[str | ChatResponse] | None = None,
        responder: Callable[[ChatRequest, SamplingConfig, int], ChatResponse] | None = None,
        cycle: bool = False,
    ) -> None:
        if (responses is None) == (responder is None):
            raise ValueError(
                "ScriptedMockBackend requires one of responses or responder (exactly one)"
            )
        self._scripted = [self._coerce(r) for r in responses] if responses is not None else None
        self._responder = responder
        self._cycle = cycle
        self._cursor = 0
        self.call_sizes: list[int] = []  # batch size of each agenerate call
        self.call_count = 0
        #: Full per-request recording in call order: (rendered_text, per-message
        #: part counts). The part-count tuple pins the image/text structure a
        #: plain-text render cannot distinguish (e.g. one TextPart "ab" vs two
        #: "a"+"b"), so fingerprint comparisons are exact.
        self.recorded_requests: list[tuple[str, tuple[int, ...]]] = []

    @staticmethod
    def render_request_text(request: ChatRequest) -> str:
        """Full plain-text render of a request: one line per message
        (``role|parts``), TextParts concatenated verbatim, ImageParts as
        ``<image:path>`` text descriptors (prefetched payloads: ``<image:loaded:path>``),
        VideoParts as ``<video:path>``."""
        lines: list[str] = []
        for message in request.messages:
            rendered: list[str] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    rendered.append(part.text)
                elif isinstance(part, ImagePart):
                    if isinstance(part.image, LoadedImage):
                        rendered.append(f"<image:loaded:{part.image.path}>")
                    else:
                        rendered.append(f"<image:{part.image}>")
                elif isinstance(part, VideoPart):
                    rendered.append(f"<video:{part.video}>")
                else:  # defensive: an unhandled part type here would break resume fingerprints
                    raise TypeError(f"unrenderable part: {type(part).__name__}")
            lines.append(f"{message.role}|" + "".join(rendered))
        return "\n".join(lines)

    @staticmethod
    def _coerce(r: str | ChatResponse) -> ChatResponse:
        return r if isinstance(r, ChatResponse) else ChatResponse(text=r)

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        self.call_sizes.append(len(requests))
        out: list[ChatResponse] = []
        for req in requests:
            self.call_count += 1
            self.recorded_requests.append(
                (
                    self.render_request_text(req),
                    tuple(len(m.content) for m in req.messages),
                )
            )
            if self._responder is not None:
                resp = self._responder(req, sampling, self.call_count)
            else:
                assert self._scripted is not None
                if self._cursor >= len(self._scripted):
                    assert self._cycle, (
                        f"script exhausted: call {self.call_count} exceeds the {len(self._scripted)} scripted responses"
                    )
                    self._cursor = 0
                resp = self._scripted[self._cursor]
                self._cursor += 1
            out.append(
                ChatResponse(
                    text=resp.text,
                    request_id=req.request_id,
                    usage=resp.usage,
                    raw=resp.raw,
                    reasoning=resp.reasoning,
                    finish_reason=resp.finish_reason,
                )
            )
        return out
