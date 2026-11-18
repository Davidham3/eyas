"""Multiprocess media prefetch (torch DataLoader style, framework-level,
backend-agnostic).

Local media in evaluation requests are materialized here in batches on a
resident process pool, leaving only processor numeric work on the engine
critical path:

- images: local read + PIL decode → LoadedImage;
- videos: vLLM media-connector decode (CPU only) → LoadedVideo, opt-in via
  ``decode_videos`` (openai_api keeps raw paths/URLs and never enables it).

All backends share this component:

- vllm: consume ``pil`` / ``media`` (in-process engines skip disk read + decode);
- openai_api: consume ``raw`` (base64 used directly, no second transcoding).

``begin`` submits a whole batch up front and ``resolve_*_async`` harvest one
request at a time, so decoding overlaps engine startup and generation;
``materialize`` is the blocking batch form kept for the HTTP backend.

Failure semantics: the loader itself never silently skips — missing/corrupt
media raise at harvest time (``materialize`` / ``resolve_*_async``). The
inference backends degrade PER-FILE failures to a per-request error response
(the over-window sentinel shape): vllm catches around its resolve/decode step,
openai harvests through ``materialize_partial``, which reports each failed
request by index instead of raising. Pool infrastructure death
(BrokenProcessPool) is NEVER sentinelized — it is recovered or raised: the
synchronous batch paths (``materialize`` / ``materialize_partial`` / ``begin``)
discard the dead pool, rebuild once and retry the batch; a second consecutive
breakage (no successful harvest between) raises with an actionable message.
The vllm in-flight resolve path is fail-fast on the first breakage (the batch
dies, siblings are cancelled and engine ids aborted upstream) but discards the
pool so the next ``begin`` rebuilds; a second consecutive breakage raises the
actionable message there too. Environment errors (ImportError) always
propagate immediately — a rebuild cannot heal a broken environment. URLs
(http/https/data:/file://) are not local files and pass through as-is."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeGuard, TypeVar

from eyas.backends.base import (
    ChatRequest,
    ImagePart,
    LoadedImage,
    LoadedVideo,
    Message,
    VideoPart,
)
from eyas.logging import get_logger
from eyas.watchdog import bump

__all__ = ["MediaFutures", "MediaPrefetchLoader", "decode_video"]

_log = get_logger(__name__)

_URL_PREFIXES = ("http://", "https://", "data:", "file://")

_MediaKey = tuple[str, str]  # (kind, path)

_T = TypeVar("_T")

# Actionable message for a pool that keeps dying: a rebuild already failed once,
# so the operator needs a lever (disable prefetch) or a root cause (worker OOM).
_POOL_BREAK_MESSAGE = (
    "media prefetch process pool broke twice consecutively (workers keep dying"
    " even after a rebuild); set prefetch_workers=0 to disable multiprocess"
    " prefetch, or check worker OOM / system process limits"
)


class MediaFutures(dict[_MediaKey, Future[Any]]):
    """One batch's media futures plus a consumer count per key.

    A decoded clip is tens of megabytes, so ``resolve_*`` drops a payload as
    soon as its last consumer has harvested it instead of pinning the whole
    batch (the old inline cache held at most 32 clips).
    """

    def __init__(self) -> None:
        super().__init__()
        self.remaining: Counter[_MediaKey] = Counter()

    def consume(self, key: _MediaKey) -> None:
        if key not in self.remaining:
            return
        if self.remaining[key] <= 1:
            self.remaining.pop(key)
            self.pop(key, None)
        else:
            self.remaining[key] -= 1


def _load_image_task(path: str) -> LoadedImage:
    """Process pool worker: read once from disk, decode to an RGB PIL (module-level function so spawn can pickle it)."""
    from io import BytesIO

    from PIL import Image

    raw = Path(path).read_bytes()
    img = Image.open(BytesIO(raw))
    img.load()
    rgb = img.convert("RGB")
    return LoadedImage(path=path, raw=raw, pil=rgb)


def decode_video(path: str, video_io_kwargs: dict[str, Any] | None = None) -> LoadedVideo:
    """Decode one clip through vLLM's media connector (CPU only)."""
    from vllm.multimodal.utils import fetch_video

    url = path if "://" in path else Path(path).expanduser().resolve().as_uri()
    fetched = fetch_video(url, video_io_kwargs)
    return LoadedVideo(path=path, media=getattr(fetched, "media", fetched))


def _load_video_task(path: str, video_io_kwargs: dict[str, Any] | None = None) -> LoadedVideo:
    # decode is CPU-only: hide the GPU so a spawned worker never creates a CUDA
    # context next to the resident engine
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return decode_video(path, video_io_kwargs)


def _is_local_path(item: Any) -> TypeGuard[str]:
    return isinstance(item, str) and not item.startswith(_URL_PREFIXES)


class MediaPrefetchLoader:
    """Resident multiprocess media prefetch pool.

    ``materialize`` materializes a whole batch: same-path requests within the
    batch are deduplicated (one future reused); an already-materialized
    LoadedImage/LoadedVideo passes through idempotently. Workers start in a
    spawn context — the Engine process may already have initialized CUDA,
    where fork is unsafe.
    """

    def __init__(
        self,
        workers: int = 8,
        *,
        decode_videos: bool = False,
        video_io_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if workers <= 0:
            raise ValueError(f"prefetch workers must be positive, got {workers}")
        self.workers = workers
        self.decode_videos = decode_videos
        self.video_io_kwargs = video_io_kwargs
        self._pool: ProcessPoolExecutor | None = None
        # consecutive pool breakages with no successful harvest between them;
        # the second one raises the actionable message instead of rebuilding
        self._consecutive_breaks = 0

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            ctx = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx)
        return self._pool

    def _discard_pool(self) -> None:
        """Shut down and drop the current pool (broken or not) so the next
        ``_ensure_pool`` builds a fresh one — no executor/worker leak."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _note_break(self, exc: BrokenProcessPool) -> None:
        """Record a pool breakage and discard the dead pool. The SECOND
        consecutive breakage (no successful harvest reset the count) raises the
        actionable message; the first returns so the caller can rebuild+retry."""
        self._consecutive_breaks += 1
        self._discard_pool()
        if self._consecutive_breaks >= 2:
            raise BrokenProcessPool(_POOL_BREAK_MESSAGE) from exc
        _log.warning(
            "[prefetch] process pool broke (%s); discarded — rebuilding and retrying once",
            exc,
        )

    def _with_pool_recovery(self, op: Callable[[], _T]) -> _T:
        """Run a synchronous batch op (submit + harvest); on BrokenProcessPool
        discard the dead pool, rebuild once and retry. ``_note_break`` raises the
        actionable message on the second consecutive breakage, so the loop is
        bounded to one retry. ImportError is not caught — it propagates."""
        while True:
            try:
                return op()
            except BrokenProcessPool as exc:
                self._note_break(exc)

    def _submit_batch(self, requests: Sequence[ChatRequest]) -> MediaFutures:
        """Submit every local media item of the batch (deduplicated) and return
        kind-keyed futures; no submissions means no pool is spawned. Raw form
        (no recovery) — the public ``begin`` and the materialize ops wrap it."""
        paths: dict[_MediaKey, str] = {}
        counts: Counter[_MediaKey] = Counter()
        for req in requests:
            for message in req.messages:
                for part in message.content:
                    if isinstance(part, ImagePart) and _is_local_path(part.image):
                        key: _MediaKey = ("image", part.image)
                        counts[key] += 1
                        paths.setdefault(key, part.image)
                    elif (
                        isinstance(part, VideoPart)
                        and self.decode_videos
                        and _is_local_path(part.video)
                    ):
                        key = ("video", part.video)
                        counts[key] += 1
                        paths.setdefault(key, part.video)
        futures = MediaFutures()
        futures.remaining = counts
        if not paths:
            return futures
        pool = self._ensure_pool()
        for key, path in paths.items():
            if key[0] == "image":
                futures[key] = pool.submit(_load_image_task, path)
            else:
                futures[key] = pool.submit(_load_video_task, path, self.video_io_kwargs)
        return futures

    def begin(self, requests: Sequence[ChatRequest]) -> MediaFutures:
        """Submit every local media item of the batch (deduplicated) and return
        kind-keyed futures; no submissions means no pool is spawned. A
        submit-time pool breakage is recovered (discard + rebuild + resubmit
        once); the second consecutive breakage raises the actionable message."""
        return self._with_pool_recovery(lambda: self._submit_batch(requests))

    def _materialize_messages(
        self, messages: tuple[Message, ...], futures: dict[_MediaKey, Future[Any]]
    ) -> tuple[Message, ...]:
        out: list[Message] = []
        for message in messages:
            parts: list[Any] = []
            changed = False
            for part in message.content:
                if isinstance(part, ImagePart) and _is_local_path(part.image):
                    # bump comes after result(): a hang on the filesystem read must
                    # be visible to the watchdog; materialize is the only long
                    # blocking point outside the scheduling layer
                    image = futures[("image", part.image)].result()
                    bump(f"prefetch.materialize [{part.image}]")
                    parts.append(replace(part, image=image))
                    changed = True
                elif (
                    isinstance(part, VideoPart)
                    and self.decode_videos
                    and _is_local_path(part.video)
                ):
                    video = futures[("video", part.video)].result()
                    bump(f"prefetch.materialize [{part.video}]")
                    parts.append(replace(part, video=video))
                    changed = True
                else:
                    parts.append(part)
            out.append(replace(message, content=tuple(parts)) if changed else message)
        return tuple(out)

    def materialize(self, requests: Sequence[ChatRequest]) -> list[ChatRequest]:
        """Materialize a whole batch: local paths → LoadedImage/LoadedVideo (order preserved, duplicates share a future).

        A pool breakage is recovered (discard + rebuild + retry the batch once);
        the second consecutive breakage raises the actionable message."""

        def _op() -> list[ChatRequest]:
            futures = self._submit_batch(requests)
            n_images = sum(1 for kind, _path in futures if kind == "image")
            n_videos = len(futures) - n_images
            # DEBUG breadcrumb + bump around the batch load: this is the only long
            # blocking point outside the scheduling layer — while it runs, the
            # console is legitimately silent; these lines are what distinguish
            # that from a true hang once EYAS_LOG_LEVEL=DEBUG
            _log.debug(
                "[prefetch] batch start requests=%d images=%d videos=%d",
                len(requests),
                n_images,
                n_videos,
            )
            bump(f"[prefetch] batch start requests={len(requests)} images={n_images}")
            t0 = time.monotonic()
            out = [
                replace(req, messages=self._materialize_messages(req.messages, futures))
                for req in requests
            ]
            _log.debug(
                "[prefetch] batch done requests=%d images=%d videos=%d elapsed=%.2fs",
                len(requests),
                n_images,
                n_videos,
                time.monotonic() - t0,
            )
            self._consecutive_breaks = 0  # a successful harvest heals the count
            return out

        return self._with_pool_recovery(_op)

    def materialize_partial(
        self, requests: Sequence[ChatRequest]
    ) -> tuple[list[ChatRequest], dict[int, Exception]]:
        """``materialize`` with per-file degradation: a request whose media
        fails to load is returned UNmaterialized and its exception reported by
        request index, instead of raising and killing the batch. The caller
        (openai_api) turns each reported failure into a per-request error
        response; the batch keeps its healthy requests.

        Pool infrastructure death (BrokenProcessPool) is recovered (discard +
        rebuild + retry the batch once) and NEVER degraded to a per-file error;
        a second consecutive breakage raises the actionable message. Environment
        errors (ImportError) still raise immediately — those are not per-file
        failures, and degrading them would silently null the whole batch.
        """

        def _op() -> tuple[list[ChatRequest], dict[int, Exception]]:
            futures = self._submit_batch(requests)
            n_images = sum(1 for kind, _path in futures if kind == "image")
            n_videos = len(futures) - n_images
            _log.debug(
                "[prefetch] batch start requests=%d images=%d videos=%d",
                len(requests),
                n_images,
                n_videos,
            )
            bump(f"[prefetch] batch start requests={len(requests)} images={n_images}")
            t0 = time.monotonic()
            out: list[ChatRequest] = []
            errors: dict[int, Exception] = {}
            for i, req in enumerate(requests):
                try:
                    out.append(
                        replace(req, messages=self._materialize_messages(req.messages, futures))
                    )
                except (BrokenProcessPool, ImportError):
                    raise
                except Exception as exc:
                    # worker exceptions arrive pickled (FileNotFoundError, PIL
                    # decode errors, ...); the taxonomy is heterogeneous, so the
                    # per-file catch is broad by design — the infra classes above
                    # are the explicit exclusions
                    errors[i] = exc
                    out.append(req)
            _log.debug(
                "[prefetch] batch done requests=%d images=%d videos=%d elapsed=%.2fs",
                len(requests),
                n_images,
                n_videos,
                time.monotonic() - t0,
            )
            self._consecutive_breaks = 0  # a completed harvest heals the count
            return out, errors

        return self._with_pool_recovery(_op)

    async def resolve_image_async(
        self, image: str | LoadedImage, futures: MediaFutures
    ) -> str | LoadedImage:
        """Await one image's payload; non-local/already-loaded/unsubmitted values pass through.

        A pool breakage here is mid-window (vllm): fail-fast on the first one
        (the batch dies upstream) but discard the dead pool so the next ``begin``
        rebuilds; the second consecutive breakage raises the actionable message.
        """
        if not _is_local_path(image):
            return image
        key: _MediaKey = ("image", image)
        future = futures.get(key)
        if future is None:
            return image
        try:
            loaded = await asyncio.wrap_future(future)
        except BrokenProcessPool as exc:
            self._note_break(exc)  # raises actionable on the second consecutive break
            raise
        self._consecutive_breaks = 0  # a successful harvest heals the count
        futures.consume(key)
        bump(f"prefetch.resolve.image [{image}]")
        return loaded

    async def resolve_video_async(
        self, video: str | LoadedVideo, futures: MediaFutures
    ) -> str | LoadedVideo:
        """Await one clip's payload; non-local/already-loaded/unsubmitted values pass through.

        Same mid-window pool-breakage contract as ``resolve_image_async``."""
        if not (_is_local_path(video) and self.decode_videos):
            return video
        key: _MediaKey = ("video", video)
        future = futures.get(key)
        if future is None:
            return video
        try:
            loaded = await asyncio.wrap_future(future)
        except BrokenProcessPool as exc:
            self._note_break(exc)  # raises actionable on the second consecutive break
            raise
        self._consecutive_breaks = 0  # a successful harvest heals the count
        futures.consume(key)
        bump(f"prefetch.resolve.video [{video}]")
        return loaded

    def close(self) -> None:
        self._discard_pool()
        self._consecutive_breaks = 0
