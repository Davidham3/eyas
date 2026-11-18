"""OpenAI-compatible HTTP client backend.

In contrast to VllmBackend (in-process Engine, client-side
rendering), this backend leaves chat template rendering to the server (the
vllm server picks the qwen2-vl template automatically) and passes messages
through directly — fitting any OpenAI-compatible endpoint:
a local server process, a remote machine, a third-party API. The module is
importable without the ``eyas[api]`` (httpx) extra; the httpx import is deferred
to the first request.

- Images / videos: local paths → base64 data URLs (mime inferred from the
  extension, defaulting to image/jpeg / video/mp4); http(s) / data: URLs pass
  through as-is; file:// URLs name a client-local file — the scheme is stripped
  and the file read like a bare path (the server cannot fetch file://).
- SamplingConfig: temperature / max_tokens are always sent; the rest (top_p /
  top_k / repetition_penalty / seed / thinking) only when not None — top_k,
  repetition_penalty and chat_template_kwargs are extension fields on vllm
  endpoints; leave them None on strict OpenAI endpoints. The thinking
  three-state is forwarded via ``chat_template_kwargs.enable_thinking``
  (True/False explicit on/off; None is not sent and the server default applies).
- extra_body: constructor-level escape hatch, shallow-merged into the payload
  last (same-named keys override sampling fields).
- Credentials: pass api_key directly, or name an env var via api_key_env
  (validated on construction, fail fast) — the key never lands in config/git.
- raw_prompt escape hatch unsupported — the OpenAI chat API has no raw
  prompt channel; fail fast.
- Retries: exponential backoff for 429/5xx/transport errors and malformed 200
  bodies (default 2 retries, 1s base); once the ladder is exhausted a
  BackendTransientError is raised for batch-level recovery — agenerate backs
  off and re-runs the whole batch (default 2 × 60s exponential). Persistent
  failures (over-window / config error / non-retryable codes) are neither
  retried nor re-run; fail fast.
- Over-window guard: rejections where prompt+max_tokens exceed the
  server window (non-retryable; the message contains window keywords) go through
  a recovery ladder — aggregate warning → one thinking-off retry → if still
  rejected the request returns an empty sentinel response
  (finish_reason=over_window) and the rest of the batch is unaffected;
  evaluation lands a durable error row (null prediction, parse_fail=1.0), and
  once the prompt is optimized shorter the new prompt hash retries naturally.
- Media failures: a missing/undecodable local media file degrades to the SAME
  per-request sentinel (finish_reason=over_window) instead of killing the
  batch — one WARNING per batch summarizes the failed media count. Media
  failures never enter the over-window rescue ladder and are counted
  separately; config-level errors (raw_prompt, bad prefetch settings, ...)
  still raise.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ChatResponse,
    ImagePart,
    InferenceBackend,
    LoadedImage,
    Message,
    SamplingConfig,
    TextPart,
    VideoPart,
    resolve_max_inflight,
)
from eyas.backends.prefetch import MediaPrefetchLoader
from eyas.backends.scheduling import gather_bounded
from eyas.logging import get_logger
from eyas.registry import register_backend

if TYPE_CHECKING:
    import httpx

__all__ = ["OpenAIBackend"]

# 514 is a non-standard engine error code with a dual nature: the over-window
# rejection shape is diverted into the sentinel ladder by the marker check first;
# generic engine errors
# ("Internal server error") are server-side transients, on par with transport
# errors, and enter the retry ladder.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 514})

_log = get_logger(__name__)

# Message signatures of over-window prompt rejections:
# "You passed 32753 input tokens and requested 16 output tokens. However, the
# model's context length is only 32768 tokens, resulting in a maximum input
# length of 32752 tokens. Please reduce the length of the input prompt.
# (parameter=input_tokens, value=32753)". The latter two generalize to the
# vLLM serve / OpenAI family. The fourth: when a vision
# request's input alone already exceeds the window, the engine skips input
# validation and instead clamps max_tokens to a negative number before
# erroring — "max_tokens must be at least 1, got -6221.
# (parameter=max_tokens, value=-6221)". Matching must include "got -" (the
# negative-value signature): a legitimate max_tokens=0 misconfiguration is
# "got 0" and must not be misjudged as over-window.
_OVER_WINDOW_MARKERS = (
    "input tokens and requested",
    "reduce the length of the input prompt",
    "maximum context length",
    "max_tokens must be at least 1, got -",
)


class OverWindowError(RuntimeError):
    """prompt + max_tokens exceed the server's context window (non-retryable rejection; the message contains window keywords)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(
            f"prompt exceeds the server's context window (HTTP {status_code}): {detail}"
        )
        self.status_code = status_code
        self.detail = detail


class BackendTransientError(RuntimeError):
    """Request-level retry ladder exhausted (transport/5xx all failed; the signal
    caught by batch-level recovery).

    Semantics: a single request burned its entire retry budget on transient
    errors — the server is unreachable right now, but this is not an intrinsic
    error of the request (unlike the persistent failures of over-window/config
    errors). Batch callers (agenerate batch-level recovery, Evaluator,
    optimizer) can safely back off and re-run the whole batch.
    """


def _is_over_window(detail: str) -> bool:
    low = detail.lower()
    return any(marker in low for marker in _OVER_WINDOW_MARKERS)


@register_backend("openai")
class OpenAIBackend(InferenceBackend):
    """OpenAI-compatible server backend (/v1/chat/completions).

    ``transport`` is for test injection only (httpx.MockTransport); the
    default path is always None.
    """

    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        api_key_env: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 300.0,
        max_inflight: int = 16,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        batch_retries: int = 2,
        batch_retry_backoff: float = 60.0,
        prefetch_workers: int = 0,
        context_window: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # httpx joins relative paths against base_url, so it must end with / for
        # /v1/chat/completions to come out right
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.model = model
        if api_key_env is not None:
            if api_key != "EMPTY":
                raise ValueError(
                    "api_key and api_key_env cannot both be set (choose one credential source)"
                )
            api_key = os.environ.get(api_key_env) or ""
            if not api_key:
                raise ValueError(
                    f"env var {api_key_env} unset or empty — API credentials come from"
                    " env vars (the key never lands in config/git)"
                )
        self.api_key = api_key
        self.extra_body = dict(extra_body) if extra_body else None
        self.timeout = timeout
        self.max_inflight = resolve_max_inflight(max_inflight)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.batch_retries = batch_retries
        self.batch_retry_backoff = batch_retry_backoff
        # the one "telling" mechanism behind base.py's "endpoint that was not
        # told" note: config `backend.params.context_window` arms the planner guard
        self.context_window = context_window
        self._loader = MediaPrefetchLoader(prefetch_workers) if prefetch_workers > 0 else None
        self._transport = transport

    # ------------------------------------------------------------------
    # Request construction (public for zero-GPU test assertions)
    # ------------------------------------------------------------------
    def prepare(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[dict[str, Any]]:
        """ChatRequest → OpenAI chat/completions payload."""
        payloads: list[dict[str, Any]] = []
        for req in requests:
            if req.raw_prompt is not None:
                raise ValueError(
                    "the OpenAI chat API has no raw prompt channel; the raw_prompt escape"
                    " hatch is unsupported — use VllmBackend for"
                    " pre-expanded rendering"
                )
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": self._to_openai_messages(req.messages),
            }
            payload.update(self._sampling_params(sampling))
            if self.extra_body:
                payload.update(self.extra_body)
            payloads.append(payload)
        return payloads

    def _sampling_params(self, sampling: SamplingConfig) -> dict[str, Any]:
        params: dict[str, Any] = {
            "temperature": sampling.temperature,
            "max_tokens": sampling.max_new_tokens,
        }
        if sampling.top_p is not None:
            params["top_p"] = sampling.top_p
        if sampling.top_k is not None:
            params["top_k"] = sampling.top_k
        if sampling.repetition_penalty is not None:
            params["repetition_penalty"] = sampling.repetition_penalty
        if sampling.seed is not None:
            params["seed"] = sampling.seed
        if sampling.thinking is not None:
            params["chat_template_kwargs"] = {"enable_thinking": sampling.thinking}
        return params

    def _to_openai_messages(self, messages: tuple[Message, ...]) -> list[dict[str, Any]]:
        """Media parts become ``image_url`` / ``video_url`` content blocks; part
        order is preserved as-is (the VLM template places vision tokens following
        the content array order)."""
        out: list[dict[str, Any]] = []
        for message in messages:
            parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": self._media_url(part.image, "image/jpeg")},
                        }
                    )
                elif isinstance(part, VideoPart):
                    if not isinstance(part.video, str):
                        raise ValueError("openai_api does not decode videos; pass a path or URL")
                    parts.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": self._media_url(part.video, "video/mp4")},
                        }
                    )
            content: str | list[dict[str, Any]] = parts
            if len(parts) == 1 and parts[0]["type"] == "text":
                content = parts[0]["text"]
            out.append({"role": message.role, "content": content})
        return out

    @staticmethod
    def _media_url(value: str | LoadedImage, default_mime: str) -> str:
        if isinstance(value, LoadedImage):
            # Prefetched payload: raw bytes were already read in the worker process;
            # mime is inferred from the original path
            mime = mimetypes.guess_type(value.path)[0] or default_mime
            return f"data:{mime};base64,{base64.b64encode(value.raw).decode('ascii')}"
        if value.startswith(("http://", "https://", "data:")):
            return value
        if value.startswith("file://"):
            # file:// names a CLIENT-local file (the prefetch layer passes the
            # scheme through as a URL): the server cannot fetch it — strip the
            # scheme and read it into a data URL exactly like a bare path.
            value = value.removeprefix("file://")
        mime = mimetypes.guess_type(value)[0] or default_mime
        data = Path(value).read_bytes()
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        if not requests:
            return []
        requests = list(requests)
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            transport=self._transport,
        ) as client:
            # Media materialization (the prefetch harvest) is a synchronous
            # blocking op; run it in a thread pool to avoid stalling the event
            # loop. It produces RAW media only — base64 payload encoding happens
            # per request inside _generate_once, so only in-flight requests hold
            # encoded media (no whole-batch pre-encoding). Per-file media
            # failures degrade to a per-request sentinel; config errors raise.
            materialized, media_errors = await asyncio.to_thread(self._harvest_media, requests)
            # Batch-level recovery: the countermeasure against transient
            # burn-through killing the batch. The coverage ceiling of the
            # request-level retry ladder is the sum of its backoffs;
            # instead of lengthening the ladder, the structural endpoint for
            # request-intrinsic faults is here — when the whole batch dies of
            # BackendTransientError, back off and re-run (in-batch completed
            # progress is not preserved; the eval-record store dedupes
            # outside the batch, so a re-run only costs time, not semantics).
            # Persistent failures (over-window / config error / non-retryable
            # codes) are not re-run; fail fast.
            attempt = 0
            while True:
                try:
                    responses, skips, rescues = await self._generate_once(
                        requests, materialized, media_errors, sampling, client
                    )
                    break
                except BackendTransientError as exc:
                    attempt += 1
                    if attempt > self.batch_retries:
                        raise
                    delay = self.batch_retry_backoff * 2 ** (attempt - 1)
                    _log.warning(
                        "[batch_retry] whole batch died of request-level retry ladder"
                        " exhaustion (%s); backing off %.0fs then batch-level re-run"
                        " %d/%d (in-batch completed progress not preserved; the cache"
                        " layers dedupe outside the batch, semantics unchanged)",
                        exc,
                        delay,
                        attempt,
                        self.batch_retries,
                    )
                    await asyncio.sleep(delay)
            if skips or rescues:
                _log.warning(
                    "[over_window] %d/%d requests rejected as over-window: %d rescued"
                    " by the thinking-off retry, %d skipped (sentinel response"
                    " finish_reason=over_window; unscored / no gradient this round,"
                    " automatic retry next round); first rejection: %s",
                    len(skips) + len(rescues),
                    len(requests),
                    len(rescues),
                    len(skips),
                    skips[0] if skips else rescues[0],
                )
            if media_errors:
                # separate accounting from the over-window ladder: a missing
                # file is a per-request error, never a thinking-off rescue
                _log.warning(
                    "[media] %d/%d requests failed media load: per-request sentinel"
                    " (finish_reason=%s) returned, batch continues; first failure: %s",
                    len(media_errors),
                    len(requests),
                    OVER_WINDOW_FINISH,
                    media_errors[min(media_errors)],
                )
            return responses

    def _harvest_media(
        self, requests: Sequence[ChatRequest]
    ) -> tuple[list[ChatRequest], dict[int, Exception]]:
        """Materialize RAW media off the event loop (the thread form of the
        prefetch harvest); NO payload encoding here — base64 construction is
        per request inside ``_generate_once`` so only in-flight requests hold
        encoded media.

        With the prefetch pool, ``materialize_partial`` swaps local paths for
        LoadedImage raw bytes and reports each failed request by index (the
        request is returned UNmaterialized). Without the pool the requests pass
        through as-is and the local file read happens later, per request, inside
        ``prepare``. Config errors (raw_prompt escape hatch, decoded-video
        payload shape) are ValueError and still raise at prepare time — fail
        fast. Pool infrastructure death (BrokenProcessPool) is recovered inside
        ``materialize_partial`` (discard + rebuild + retry once); a second
        consecutive breakage raises with an actionable message. Environment
        errors (ImportError) propagate immediately.
        """
        if self._loader is not None:
            materialized, media_errors = self._loader.materialize_partial(requests)
        else:
            materialized, media_errors = list(requests), {}
        for i, err in sorted(media_errors.items()):
            _log.error("[media] load failed request=%s error=%s", requests[i].request_id, err)
        return materialized, media_errors

    async def _generate_once(
        self,
        requests: Sequence[ChatRequest],
        materialized: list[ChatRequest],
        media_errors: dict[int, Exception],
        sampling: SamplingConfig,
        client: httpx.AsyncClient,
    ) -> tuple[list[ChatResponse], list[OverWindowError], list[OverWindowError]]:
        """One batch attempt: fan out all requests under a concurrency cap; return
        (responses, over-window skips, over-window rescues).

        Each request builds its OWN payload (file read / base64) on a thread
        pool, then consumes-and-drops its raw prefetched media: the batch slot is
        swapped back to the lightweight path-bearing request and the local raw
        reference released, so only in-flight requests hold encoded media (the
        refcount frees shared raw bytes once their last consumer is done —
        mirroring vllm's ``MediaFutures.consume``). A batch-level re-run therefore
        re-reads consumed media from the original path (files still exist;
        prefetch is an optimization), keeping the memory bound and re-run
        correctness together.

        Transient burn-through raises BackendTransientError (the signal caught by
        batch-level recovery); over-window rejections do not raise — the
        sentinel response plus counted lists come back with the results
        (recovery ladder). A per-file media failure (harvest-reported or an
        OSError at prepare time) returns the sentinel directly — no server call,
        no rescue ladder, no skip counting; config errors (ValueError) raise.
        """
        skips: list[OverWindowError] = []
        rescues: list[OverWindowError] = []

        def _factory(i: int, req: ChatRequest) -> Callable[[], Awaitable[ChatResponse]]:
            async def _one() -> ChatResponse:
                def _sentinel() -> ChatResponse:
                    return ChatResponse(
                        text="", request_id=req.request_id, finish_reason=OVER_WINDOW_FINISH
                    )

                if i in media_errors:
                    return _sentinel()
                # consume-and-drop: take this request's raw media out of the batch
                # list (swap back to the path-bearing original) so a re-run re-reads
                # from disk; the local ``mat`` reference is released after encoding.
                mat = materialized[i]
                materialized[i] = req
                try:
                    payload = (await asyncio.to_thread(self.prepare, [mat], sampling))[0]
                except OSError as exc:
                    media_errors[i] = exc
                    _log.error("[media] load failed request=%s error=%s", req.request_id, exc)
                    return _sentinel()
                del mat
                try:
                    data = await self._post(client, payload)
                except OverWindowError as exc:
                    # Recovery ladder: thinking already explicitly off → no
                    # budget left to give up, skip directly; otherwise retry once with
                    # thinking off (squeezing back into the window margin) and skip
                    # only if still rejected.
                    if sampling.thinking is False:
                        skips.append(exc)
                        return _sentinel()
                    retry = dict(payload)
                    retry["chat_template_kwargs"] = {"enable_thinking": False}
                    try:
                        data = await self._post(client, retry)
                    except OverWindowError:
                        skips.append(exc)
                        return _sentinel()
                    rescues.append(exc)
                choice = data["choices"][0]
                message = choice.get("message") or {}
                # Thinking content convention: reasoning_content (DeepSeek-style
                # servers) or reasoning (OpenAI-style servers)
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                return ChatResponse(
                    text=message.get("content") or "",
                    request_id=req.request_id,
                    usage=data.get("usage"),
                    raw=data,
                    reasoning=reasoning or None,
                    finish_reason=choice.get("finish_reason"),
                )

            return _one

        responses = await gather_bounded(
            (_factory(i, req) for i, req in enumerate(requests)),
            self.max_inflight,
        )
        return responses, skips, rescues

    async def _post(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                await asyncio.sleep(self.retry_backoff * 2 ** (attempt - 1))
            try:
                resp = await client.post("chat/completions", json=payload)
            except httpx.TransportError as exc:
                last = f"transport error: {exc}"
                continue
            if resp.status_code == 200:
                # A malformed 200 body (non-JSON gateway junk, missing/empty
                # choices, a non-object entry, or a non-object message /
                # non-string content / non-string reasoning_content / reasoning /
                # non-string finish_reason / non-object usage deeper down) is a
                # server-side transient, not a request-intrinsic fault: it
                # burns a retry rung here and, once the ladder is exhausted,
                # surfaces BackendTransientError for batch-level recovery —
                # never a raw decode/KeyError/AttributeError that kills the
                # batch outside the retry classification. The validation is
                # total over exactly the shapes _one consumes (choices[0],
                # message, message.content, message.reasoning_content /
                # message.reasoning, finish_reason, usage), so its extraction
                # can neither raise nor smuggle a non-str text/reasoning /
                # non-dict usage into ChatResponse.
                try:
                    data = resp.json()
                    choice = data["choices"][0]
                    if not isinstance(choice, dict):
                        raise TypeError(f"choices[0] is {type(choice).__name__}, expected object")
                    message = choice.get("message")
                    if message is not None and not isinstance(message, dict):
                        raise TypeError(
                            f"choices[0].message is {type(message).__name__}, expected object"
                        )
                    content = message.get("content") if message else None
                    if content is not None and not isinstance(content, str):
                        raise TypeError(
                            f"choices[0].message.content is {type(content).__name__}, "
                            "expected string"
                        )
                    for key in ("reasoning_content", "reasoning"):
                        value = message.get(key) if message else None
                        if value is not None and not isinstance(value, str):
                            raise TypeError(
                                f"choices[0].message.{key} is {type(value).__name__}, "
                                "expected string"
                            )
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None and not isinstance(finish_reason, str):
                        raise TypeError(
                            f"choices[0].finish_reason is {type(finish_reason).__name__}, "
                            "expected string"
                        )
                    usage = data.get("usage")
                    if usage is not None and not isinstance(usage, dict):
                        raise TypeError(f"usage is {type(usage).__name__}, expected object")
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    last = f"malformed 200 body: {exc}"
                    continue
                return data
            detail = resp.text[:500]
            # Over-window detection precedes the retry set: 514 is dual-natured, and
            # over-window is a request-intrinsic error (a retry is guaranteed to be
            # rejected again) — it must go straight into the sentinel ladder, never
            # spin through retries.
            if _is_over_window(detail):
                raise OverWindowError(resp.status_code, detail)
            # A "got 0"-style clamp to a legitimate max_tokens value is a client
            # config error — persistent, retries are pointless, fail fast (distinct
            # from the "got -" negative-clamp over-window shape).
            if "max_tokens must be at least 1" in detail.lower():
                raise RuntimeError(
                    f"OpenAI backend request failed (HTTP {resp.status_code}): {detail}"
                )
            if resp.status_code not in _RETRYABLE_STATUS:
                raise RuntimeError(
                    f"OpenAI backend request failed (HTTP {resp.status_code}): {detail}"
                )
            last = f"HTTP {resp.status_code}: {detail}"
        # Transient errors burned through the request ladder — dedicated exception
        # for batch-level recovery to catch; persistent failures (over-window /
        # config error / non-retryable codes) already raised as bare RuntimeError
        # above and never reach this path.
        raise BackendTransientError(
            f"OpenAI backend still failing after {self.max_retries} retries: {last}"
        )

    def close(self) -> None:
        """Release the prefetch process pool at run end (Y18).

        Mirrors VllmBackend's loader shutdown; there is no engine/resident loop
        to release here — the httpx client is per-batch (``async with``), so
        shutting the loader is the whole teardown. Idempotent.
        """
        if self._loader is not None:
            self._loader.close()
            self._loader = None
