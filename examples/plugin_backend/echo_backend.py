"""A minimal third-party backend, registered through the config ``plugins`` key.

This module is the worked example behind the "Adding your own backend"
tutorial: nothing imports it at framework startup, nothing
in ``configs/`` or ``eyas.builtin`` names ``echo_demo`` — the name only becomes
resolvable once a config lists this module under ``plugins`` (or an installed
distribution advertises it via the ``eyas.plugins`` entry-points group). That
is the whole registration story a built-in backend obeys, minus the in-tree
location.

The backend itself answers deterministically: each response echoes the last
user turn's text content, truncated at 80 chars. No sleeps, no IO, no network —
the point is the contract, not the answers (the echoed question text makes
``demo_shapes`` score a flat 2/8; the metric carries no algorithmic signal).
"""

from __future__ import annotations

from collections.abc import Sequence

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    SamplingConfig,
    TextPart,
)
from eyas.registry import register_backend

__all__ = ["EchoBackend"]


@register_backend("echo_demo")
class EchoBackend(InferenceBackend):
    """Deterministic echo (example only — not for real optimization runs).

    Response construction mirrors the framework's own ``ScriptedMockBackend``:
    one ``ChatResponse`` per request carrying at least ``text`` and the
    request's ``request_id``.
    """

    name = "echo_demo"

    def __init__(self) -> None:
        # Readable proof of the close() contract (echo_backend_test pins it);
        # the framework only calls close(), it never reads this back.
        self.closed: bool = False

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        return [
            ChatResponse(text=self._echo_text(req), request_id=req.request_id) for req in requests
        ]

    def close(self) -> None:
        """Release underlying resources. Nothing to release here — the flag only
        proves the run's post-loop close() reached a plugin backend."""
        self.closed = True

    @staticmethod
    def _echo_text(request: ChatRequest) -> str:
        """Last user turn's text content (TextParts concatenated in order),
        truncated at 80 chars. Same traversal as ``eyas.backends.base.user_text``
        — messages hold typed parts, so image parts are filtered out, never
        assumed absent."""
        text = ""
        for message in request.messages:
            if message.role == "user":
                text = "".join(part.text for part in message.content if isinstance(part, TextPart))
        return text[:80]
