from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from eyas.backends.scheduling import run_sync
from eyas.logging import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inference request/response (canonical form = the messages array; template
# expansion is pushed down to concrete backends)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class LoadedImage:
    """Image payload after prefetch (produced by MediaPrefetchLoader).

    ``pil`` is the decoded PIL.Image.Image (used directly by the vllm
    in-process engine, skipping disk read and decode); ``raw`` is the raw file
    bytes (used directly by openai_api for base64, avoiding a second transcoding
    pass). The core package does not depend on Pillow, so ``pil`` is typed Any.
    """

    path: str
    raw: bytes
    pil: Any


@dataclass(frozen=True)
class LoadedVideo:
    """Video payload after prefetch: the connector's ``(frames, metadata)``.

    ``path`` keeps the original clip identity (dedup key, logging). The media
    payload is engine-native and may hold numpy arrays, so the core package
    stays independent of it and callers compare by attribute, never by ``==``.
    """

    path: str
    media: Any


@dataclass(frozen=True)
class ImagePart:
    """Image: local path, URL, or prefetched payload (LoadedImage)."""

    image: str | LoadedImage


@dataclass(frozen=True)
class VideoPart:
    """Video: local path, URL, or prefetched payload (LoadedVideo)."""

    video: str | LoadedVideo


Part = TextPart | ImagePart | VideoPart


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: tuple[Part, ...]


def system_message(text: str) -> Message:
    return Message(role="system", content=(TextPart(text),))


def user_message(*parts: Part | str) -> Message:
    content = tuple(TextPart(p) if isinstance(p, str) else p for p in parts)
    return Message(role="user", content=content)


def assistant_message(text: str) -> Message:
    """Intermediate assistant turn (continuation guidance such as a CoT prefix; the generation-turn opener is appended by profile rendering)."""
    return Message(role="assistant", content=(TextPart(text),))


def user_text(request: ChatRequest) -> str:
    """Plain text inside the user message (TextParts concatenated in order).

    Judges use it to reconstruct "the classification question the model actually
    saw" (e.g. the "Original Task" section of CoT/EvoPrompt/OPRO), avoiding the
    drift from the true rendering that a config-driven reverse rebuild causes.
    """
    for message in request.messages:
        if message.role == "user":
            return "".join(part.text for part in message.content if isinstance(part, TextPart))
    return ""


@dataclass
class ChatRequest:
    messages: tuple[Message, ...]
    request_id: str | None = None
    # Escape hatch: when the caller supplies a pre-expanded raw prompt, skip
    # template expansion to serve the radix-cache-friendly performance path
    raw_prompt: str | None = None


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    max_new_tokens: int = 1024
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    # Three-state: None = no intervention — the template default applies (the
    # openai client sends no thinking kwarg; a vllm profile forwards the
    # tri-state only when it declares ``thinking_kwarg``, where None renders
    # off). True/False = explicitly on/off.
    thinking: bool | None = None
    seed: int | None = None


def merge_sampling(base: SamplingConfig, override: SamplingConfig) -> SamplingConfig:
    """Stage sampling inherits deployment-level defaults: explicit override fields
    win; three-state/optional fields fall back to base when None.

    An optimizer stage's SamplingConfig only describes algorithm knobs
    (temperature/top_p/max_new_tokens); deployment knobs (thinking/seed etc.)
    live only in RunContext.sampling (config ``backend.sampling``). Building a
    stage from scratch drops the latter — on OpenAI-compatible endpoints
    thinking=None hits the server default (enabled), thinking burns through the
    max_new_tokens cap, and compare/attr/gradient plus ape generation
    degrade into wholesale truncation. So every
    stage sampling must be merged with ctx.sampling through this function.
    temperature / max_new_tokens always take the override (algorithm budget
    fields, always set explicitly by stages).
    """
    return SamplingConfig(
        temperature=override.temperature,
        max_new_tokens=override.max_new_tokens,
        top_k=override.top_k if override.top_k is not None else base.top_k,
        top_p=override.top_p if override.top_p is not None else base.top_p,
        repetition_penalty=(
            override.repetition_penalty
            if override.repetition_penalty is not None
            else base.repetition_penalty
        ),
        thinking=override.thinking if override.thinking is not None else base.thinking,
        seed=override.seed if override.seed is not None else base.seed,
    )


# Per-request error sentinel: when the prompt exceeds the server's context
# window and a thinking-off retry is still rejected, the openai backend returns
# an empty response with this finish_reason instead of raising — the rest of the
# batch is unaffected, and scoring lands at 0 under the parse_fail=1.0 rule.
# Media load failures (missing/undecodable local file) in both inference
# backends reuse the same sentinel shape, so they land the same durable error
# row (null prediction, parse_fail=1.0) in the eval-record store. A new prompt
# (optimized shorter) is a new cache key, so the next round retries naturally.
OVER_WINDOW_FINISH = "over_window"


@dataclass
class ChatResponse:
    text: str
    request_id: str | None = None
    usage: dict[str, Any] | None = None
    raw: Any = None  # engine-native return value, for debugging
    reasoning: str | None = (
        None  # thinking content (reasoning_content convention / inline think stripped), separate from the answer
    )
    finish_reason: str | None = (
        None  # stop/length/over_window/... — truncation (length) and over-window skips observable
    )


def resolve_max_inflight(configured: int) -> int:
    """Backend in-flight window resolution: the ``EYAS_MAX_INFLIGHT`` env var
    explicitly overrides the config value.

    Lets you raise parallelism for engine scale-up/down without editing config
    (a one-line export in the run command). Applies only to inference backends
    (every batched call on the VLM request path); other concurrent workloads
    size their batches independently. An invalid override value fails fast; a
    silent fallback would mask the tuning intent. A SET-but-empty value is
    invalid too (an empty export is a common slip) — only an UNSET variable
    keeps the configured value.
    """
    raw = os.environ.get("EYAS_MAX_INFLIGHT")
    if raw is None:
        return configured
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"EYAS_MAX_INFLIGHT must be an integer >= 1, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"EYAS_MAX_INFLIGHT must be >= 1, got {raw!r}")
    _log.info("max_inflight=%d (EYAS_MAX_INFLIGHT override, config=%d)", value, configured)
    return value


class InferenceBackend(ABC):
    """Unified inference backend interface. vllm / openai / mock all
    implement it, guaranteeing consistent evaluation behavior."""

    name: ClassVar[str]

    #: The engine's true context window (tokens) when the backend knows it.
    #: Consumed at assembly: it arms the RequestPlanner's over-window
    #: guard. None = unknown (e.g. a remote OpenAI-compatible endpoint that
    #: was not told) — the guard stays honestly disarmed.
    context_window: int | None = None

    #: Images per request the engine accepts, when the backend knows it.
    #: Consumed at assembly: the strategy's ResourceProfile.max_images is
    #: intersected with it (over-limit assemblies fail at startup).
    #: None = unknown/server-side — the intersection skips this bound.
    max_images: int | None = None

    @abstractmethod
    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]: ...

    def generate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        """Synchronous convenience entry. Note: must not be called inside a running event loop."""
        return run_sync(self.agenerate(requests, sampling))

    def close(self) -> None:  # noqa: B027  optional override hook: default no-op is intentional
        """Release underlying resources (engine/process pool/connections). Default no-op; backends holding heavy resources override."""


class BackendWrapper:
    """Common pass-through behavior for inference-backend wrappers (the
    raw→traced→normalized chain): ``slot`` and ``close`` must always reach the
    inner backend.

    Why reach-through is not optional: a wrapper that swallows ``slot`` turns
    the optimizer's engine release into a silent no-op, so the next engine
    start collides with an unreleased one and runs out of GPU memory; a
    wrapper that swallows ``close`` makes the sweep's next run crash into an
    unreleased engine (engine processes do not exit with GC).

    Subclasses keep their own ``__init__`` (they differ by trace bookkeeping)
    and must call ``super().__init__(inner)``.
    """

    def __init__(self, inner: InferenceBackend) -> None:
        self.inner = inner

    @property
    def slot(self) -> Any:
        """Pass through the inner engine slot: the optimizer releases the engine
        via ``getattr(backend, "slot")``."""
        return getattr(self.inner, "slot", None)

    def close(self) -> None:
        """Pass through close (the run end explicitly releases the engine)."""
        self.inner.close()
