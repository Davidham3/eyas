"""vLLM local inference backend (in-process AsyncLLM engine, client-side
rendering).

Rendering likewise goes through ModelProfile (client-side ChatML + vision
placeholder expansion); images are passed via ``multi_modal_data["image"]`` and
expanded by vLLM's VLM processor at the ``<|image_pad|>`` placeholder, and
videos via ``multi_modal_data["video"]`` as decoded frame arrays: vLLM 0.29's
parser no longer takes bare paths, so clips are decoded through vLLM's own
media connector in the prefetch process pool (client-side). vllm is an
optional extra (``eyas[vllm]``); every vllm import in this module is deferred
until the engine is actually launched, so the core package can import without
the extra.

Engine choice: AsyncLLM (vLLM's V1 engine API, the class behind its own OpenAI
server) rather than the
synchronous ``LLM``. Per-request ``generate()`` async streams converge into
EngineCore through a single internal output_handler — concurrent submission is
naturally safe and batched by the engine scheduler. Concurrent sync
``LLM.generate`` calls interleave ZMQ frame writes and kill the reader thread;
regression to it is forbidden.
The engine must be created on a running
event
loop (the output handler binds to the current loop) — agenerate all run on
run_sync's process-level resident loop, which satisfies this.

Note: GPU memory unload for the vLLM in-process engine goes through
``AsyncLLM.shutdown()`` terminating the EngineCore process (more thorough than the
synchronous ``LLM`` path's best-effort), but process-level full reclamation is still most
reliable at exit. vLLM workers are spawned, so the driver script entry needs the
``if __name__ == "__main__":`` guard (library code is unaffected).
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from itertools import count
from typing import Any

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    LoadedImage,
    LoadedVideo,
    SamplingConfig,
    resolve_max_inflight,
)
from eyas.backends.prefetch import MediaFutures, MediaPrefetchLoader, decode_video
from eyas.backends.prepare import EngineBackendMixin, PreparedRequest
from eyas.backends.profiles import ModelProfile, resolve_profile
from eyas.backends.scheduling import BatchProgress, gather_cancel_on_error, run_sync
from eyas.logging import get_logger
from eyas.registry import register_backend
from eyas.watchdog import bump

__all__ = ["VllmBackend", "launch_vllm_engine"]

_log = get_logger(__name__)


def _usage_from_output(out: Any) -> dict[str, int] | None:
    """vLLM RequestOutput → usage dict (token id sequence length is the token count; missing shapes → None)."""
    try:
        prompt_ids = out.prompt_token_ids
        completion_ids = out.outputs[0].token_ids
    except (AttributeError, IndexError):
        return None
    if prompt_ids is None or completion_ids is None:
        return None
    return {"prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids)}


async def _abort_inflight(engine: Any, inflight: set[str]) -> None:
    """Best-effort abort of still-in-flight engine requests after a batch
    failure — cancelling the client tasks alone can leave the engine generating
    for a batch nobody will read (GPU burn). Abort failures are logged, never
    raised: they must not mask the original batch exception. Handles both the
    coroutine ``abort`` shape (AsyncLLM.abort in current vllm) and a synchronous
    one."""
    abort = getattr(engine, "abort", None)
    if abort is None:
        return
    for request_id in sorted(inflight):
        try:
            result = abort(request_id)
            if inspect.isawaitable(result):
                # bounded wait: the most common batch-failure cause is a wedged
                # engine — an abort that never returns must not block exception
                # propagation until the watchdog fires (GPU keeps burning anyway;
                # the kill path is the backstop, not this loop)
                await asyncio.wait_for(result, timeout=5.0)
        except Exception:
            _log.warning(
                "[vllm] abort failed or timed out request_id=%s", request_id, exc_info=True
            )


def launch_vllm_engine(
    model_path: str,
    tp_size: int = 1,
    mem_fraction: float = 0.85,
    context_length: int = 32768,
    dtype: str = "auto",
    max_running_requests: int | None = None,
    **engine_kwargs: Any,
) -> Any:
    """Launch the vLLM AsyncLLM engine (called by ``VllmSlot.ensure``).

    Must be called inside a running event loop (AsyncLLM's output handler task
    binds to the current loop); VllmBackend.agenerate all run on the run_sync
    resident loop, which satisfies this.
    """
    # The flashinfer sampler's JIT compilation always fails in CUDA<12.9
    # containers (--compress-mode=size is a CUDA 12.9+ nvcc option); native
    # torch sampling overhead is negligible
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    _log.info("🚀 Launching vLLM AsyncLLM")
    _log.info("   model     : %s", model_path)
    _log.info("   TP        : %s, gpu_memory_utilization: %s", tp_size, mem_fraction)
    _log.info("   ctx length: %s, dtype: %s", context_length, dtype)
    kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": mem_fraction,
        "max_model_len": context_length,
        "dtype": dtype,
        "disable_log_stats": True,
        **engine_kwargs,
    }
    # gpu_memory_utilization is folded in by the lifecycle manager (VllmSlot
    # narrows it by measured free memory and writes it back to the slot so
    # release's reclamation check uses the same yardstick); the same-named key in
    # engine_kwargs must not override it — the folded value is measured from free
    # memory: a config 0.85 passed in via **engine_kwargs expansion makes the engine
    # fill to the 0.85 budget and blow up at warmup.
    if "gpu_memory_utilization" in engine_kwargs:
        _log.warning(
            "   engine_kwargs.gpu_memory_utilization=%s overridden by the lifecycle-folded value %s",
            engine_kwargs["gpu_memory_utilization"],
            mem_fraction,
        )
    kwargs["gpu_memory_utilization"] = mem_fraction
    if max_running_requests is not None:
        kwargs["max_num_seqs"] = max_running_requests
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    _log.info("   ✅ Engine ready")
    return engine


# The warmup safety margin for _effective_mem_fraction. After the KV cache fills
# to the f budget, the warmup phase still allocates beyond-budget float32 state
# plus CUDA context and fragmentation headroom. engine_kwargs must never override
# the folded value (launch_vllm_engine forces the overwrite above); the margin
# keeps the post-fold warmup spike within the folded budget. Default 4.5 GiB;
# `EYAS_VLLM_WARMUP_RESERVE_GIB` overrides it for cards whose warmup spike differs.
_DEFAULT_WARMUP_RESERVE_GIB = 4.5
#: Upper bound for the same knob: a reserve beyond this is a typo, and keeping the
#: value finite also keeps ``gib * 2**30`` inside int range.
_MAX_WARMUP_RESERVE_GIB = 1024.0


def _warmup_reserve_bytes() -> int:
    """Parse ``EYAS_VLLM_WARMUP_RESERVE_GIB``; out-of-range or unparseable values
    keep the default so a typo cannot break every CLI verb at import time."""
    raw = os.environ.get("EYAS_VLLM_WARMUP_RESERVE_GIB")
    gib = _DEFAULT_WARMUP_RESERVE_GIB
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            parsed = math.nan
        if math.isfinite(parsed) and 0.0 <= parsed <= _MAX_WARMUP_RESERVE_GIB:
            gib = parsed
        else:
            _log.warning(
                "EYAS_VLLM_WARMUP_RESERVE_GIB=%r is not a finite value in [0, %g]; using %.1f GiB",
                raw,
                _MAX_WARMUP_RESERVE_GIB,
                _DEFAULT_WARMUP_RESERVE_GIB,
            )
    return int(gib * 2**30)


_WARMUP_RESERVE_BYTES = _warmup_reserve_bytes()


@dataclass
class VllmSlot:
    """Resident single-model slot (release is best-effort)."""

    model_path: str
    tp_size: int = 1
    mem_fraction: float = 0.85
    context_length: int = 32768
    dtype: str = "auto"
    max_running_requests: int | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    engine: Any | None = None

    @property
    def is_active(self) -> bool:
        return self.engine is not None

    def ensure(self) -> Any:
        if self.engine is None:
            self.mem_fraction = self._effective_mem_fraction()
            self.engine = launch_vllm_engine(
                model_path=self.model_path,
                tp_size=self.tp_size,
                mem_fraction=self.mem_fraction,
                context_length=self.context_length,
                dtype=self.dtype,
                max_running_requests=self.max_running_requests,
                **self.engine_kwargs,
            )
        return self.engine

    def _effective_mem_fraction(self) -> float:
        """Narrow gpu_memory_utilization by measured free memory at creation
        (written back to self so release's reclamation check uses the same
        yardstick).

        Motivation: f is a share of the whole machine, so memory held by another
        process on the GPU eats into the engine's headroom, and the configured
        budget plus the warmup spike can stack over the line.
        Cap = (free - warmup safety margin)/total;
        on a cold start (nothing else resident) the cap exceeds the configured
        value and behavior is unchanged (on a dedicated card the fold is
        effectively a no-op); a folded result < 0.5 means the card is
        too occupied for the engine to fit — fail fast rather than silently
        shrink the KV cache. Single-GPU semantics: mem_get_info reads the current
        device (with tp>1 workers span cards; this is only a conservative cap)."""
        try:
            import torch
        except ImportError:
            return self.mem_fraction
        if not torch.cuda.is_available():
            return self.mem_fraction
        free, total = torch.cuda.mem_get_info()
        f_cap = (free - _WARMUP_RESERVE_BYTES) / total
        f = min(self.mem_fraction, f_cap)
        if f < 0.5:
            raise RuntimeError(
                f"not enough free GPU memory to start the engine at f={self.mem_fraction:.2f}"
                f" (free {free / 2**30:.1f}GiB, folded cap f={f_cap:.2f}) —"
                " another process on the GPU is using memory; check GPU memory"
                " usage first"
            )
        return f

    def release(self) -> None:
        if self.engine is None:
            return
        _log.info("   🛑 Releasing vLLM engine (AsyncLLM.shutdown terminates EngineCore)...")
        shutdown = getattr(self.engine, "shutdown", None)
        self.engine = None
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass
        self._wait_gpu_memory_reclaimed(mem_fraction=self.mem_fraction)
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _wait_gpu_memory_reclaimed(mem_fraction: float, timeout: float = 300.0) -> None:
        """EngineCore is a subprocess: shutdown returning ≠ memory reclaimed.
        Sampling "free memory unchanged across two consecutive samples" is fooled
        by teardown lag — from receiving shutdown to actually releasing the CUDA
        context, EngineCore needs time (teardown is not instantaneous), so a flat
        curve can be misread as reclaimed while the engine still holds its share. Threshold check
        instead: while the engine lives at gpu_memory_utilization f, free ≤
        (1-f)·total, and after full reclamation free is far above that — the
        threshold is (1-f/2)·total, with ample margin on both sides; timeout
        fails fast (continuing with the engine still holding memory is a
        guaranteed OOM; letting it pass silently only defers and obscures the
        cause of death)."""
        try:
            import torch
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        required = int(torch.cuda.mem_get_info()[1] * (1.0 - mem_fraction / 2.0))
        deadline = time.monotonic() + timeout
        while True:
            free = torch.cuda.mem_get_info()[0]
            if free >= required:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"vLLM EngineCore memory not reclaimed after {timeout:.0f}s"
                    f" (free {free / 2**30:.1f}GiB < target {required / 2**30:.1f}GiB);"
                    " continuing with the engine still holding GPU memory is a guaranteed OOM"
                )
            time.sleep(2.0)


@register_backend("vllm")
class VllmBackend(EngineBackendMixin, InferenceBackend):
    """Local vLLM backend (in-process LLM; client-side rendering)."""

    name = "vllm"

    # SamplingParams pinned item by item (contract anchor); the token cap is
    # named max_tokens here
    sampling_tokens_key = "max_tokens"

    def __init__(
        self,
        model_path: str,
        *,
        tp_size: int = 1,
        mem_fraction: float = 0.85,
        context_length: int = 32768,
        dtype: str = "auto",
        max_running_requests: int | None = None,
        max_inflight: int = 512,
        prefetch_workers: int = 0,
        profile: ModelProfile | str | dict[str, Any] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self.max_inflight = resolve_max_inflight(max_inflight)
        self.profile = resolve_profile(profile, model_path)
        # declared window = the effective engine value (engine_kwargs can
        # override max_model_len at launch); arms the planner guard
        self.context_window = (engine_kwargs or {}).get("max_model_len", context_length)
        # declared image cap = the engine's limit_mm_per_prompt image entry;
        # unset means vllm's own default (1 image per prompt), not "unknown"
        limit_mm = (engine_kwargs or {}).get("limit_mm_per_prompt") or {}
        self.max_images = limit_mm.get("image", 1)
        media_io = (engine_kwargs or {}).get("media_io_kwargs") or {}
        self._video_io_kwargs = media_io.get("video") or None
        self._loader = (
            MediaPrefetchLoader(
                prefetch_workers,
                decode_videos=True,
                video_io_kwargs=self._video_io_kwargs,
            )
            if prefetch_workers > 0
            else None
        )
        # Engine-globally unique request_id sequence (AsyncLLM errors outright on
        # duplicate ids of live requests)
        self._request_seq = count()
        self.slot = VllmSlot(
            model_path=model_path,
            tp_size=tp_size,
            mem_fraction=mem_fraction,
            context_length=context_length,
            dtype=dtype,
            max_running_requests=max_running_requests,
            engine_kwargs=dict(engine_kwargs or {}),
        )

    def _multi_modal_data(self, prepared: PreparedRequest) -> dict[str, Any] | None:
        """PreparedRequest → vLLM ``multi_modal_data`` (keys ``"image"`` / ``"video"``).

        Images are handed over decoded (PIL), so the engine skips disk read and
        decode. Videos are decoded too: vLLM 0.29's ``multi_modal_data`` parser
        accepts frame arrays (optionally with HF video metadata) but no longer
        accepts a bare path — a path string is iterated character-wise and trips
        ``assert_never`` in ``MultimodalDataParser._get_video_with_metadata``.
        The prefetch pool normally holds the payload already; a bare path (no
        prefetch pool) is decoded here through vLLM's own media connector with
        the engine's ``media_io_kwargs["video"]``, so codec selection, frame
        sampling and URL support stay the engine's.
        """
        mm: dict[str, Any] = {}
        if prepared.images:
            from PIL import Image

            loaded = [
                p.pil if isinstance(p, LoadedImage) else Image.open(p) for p in prepared.images
            ]
            mm["image"] = loaded[0] if len(loaded) == 1 else loaded
        if prepared.videos:
            loaded_videos = [
                p.media
                if isinstance(p, LoadedVideo)
                else decode_video(p, self._video_io_kwargs).media
                for p in prepared.videos
            ]
            mm["video"] = loaded_videos[0] if len(loaded_videos) == 1 else loaded_videos
        return mm or None

    async def _resolve_media(
        self, item: PreparedRequest, futures: MediaFutures | None
    ) -> PreparedRequest:
        """Swap local media paths for their prefetched payloads (inside the
        in-flight window, so decode overlaps already-running generations)."""
        if futures is None:
            return item
        loader = self._loader
        assert loader is not None
        images = tuple([await loader.resolve_image_async(p, futures) for p in item.images])
        videos = tuple([await loader.resolve_video_async(p, futures) for p in item.videos])
        return replace(item, images=images, videos=videos)

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        if not requests:
            return []
        requests = list(requests)
        # submit every local decode up front: the pool works while the engine
        # launches and while earlier requests generate
        futures = self._loader.begin(requests) if self._loader is not None else None
        t0 = time.monotonic()
        engine = self.slot.ensure()
        params = self._sampling_params(sampling)
        # vLLM's V1 processor only accepts SamplingParams instances; a dict gets
        # misreported as "Pooling models"
        from vllm import SamplingParams

        sp = SamplingParams(**params)
        prepared = self.prepare(requests, sampling)
        n_images = sum(len(item.images) for item in prepared)
        n_videos = sum(len(item.videos) for item in prepared)
        if n_images or n_videos:
            _log.info(
                "[media] decode submitted requests=%d images=%d videos=%d",
                len(requests),
                n_images,
                n_videos,
            )
        decode_done: dict[str, float | None] = {"t": None}
        # max_inflight semantics = the client-side in-flight request window
        # (sliding window, no inter-batch sync barrier): submit all requests as
        # early as possible; the engine scheduler batches continuously by KV
        # admission — in-batch parallelism is entirely the engine's to absorb
        window = asyncio.Semaphore(self.max_inflight)
        progress = BatchProgress(len(requests))
        watcher = progress.watch()
        # engine-side ids currently in flight: on batch failure every one of
        # them must be aborted (cancelling the client task alone can leave the
        # engine generating for a batch nobody will read)
        inflight: set[str] = set()
        # per-file media failures degrade to a per-request sentinel response
        # (the over-window shape); collected for the one batch-level WARNING
        media_failures: list[Exception] = []

        async def _generate_one(req: ChatRequest, item: PreparedRequest) -> ChatResponse:
            async with window:
                try:
                    item = await self._resolve_media(item, futures)
                    if (item.images or item.videos) and decode_done["t"] is None:
                        decode_done["t"] = time.monotonic()
                    mm = self._multi_modal_data(item)
                except (BrokenProcessPool, ImportError):
                    # pool death / broken environment are not per-file media
                    # failures — degrading them would silently null the batch
                    raise
                except Exception as exc:
                    # missing/undecodable file: a per-REQUEST error, not a
                    # batch failure. The sentinel reuses the over-window shape
                    # so evaluation lands the same durable parse_fail row.
                    # CancelledError is a BaseException and never lands here —
                    # first-error sibling cancellation stays intact.
                    media_failures.append(exc)
                    _log.error(
                        "[media] load failed request=%s images=%d videos=%d error=%s",
                        req.request_id,
                        len(item.images),
                        len(item.videos),
                        exc,
                    )
                    progress.step()
                    return ChatResponse(
                        text="", request_id=req.request_id, finish_reason=OVER_WINDOW_FINISH
                    )
                out: Any = None
                try:
                    entry: dict[str, Any] = {"prompt": item.prompt}
                    if mm is not None:
                        entry["multi_modal_data"] = mm
                    request_id = f"eyas-{next(self._request_seq)}"
                    inflight.add(request_id)
                    try:
                        async for out_item in engine.generate(entry, sp, request_id):
                            out = out_item
                            bump(f"vllm.generate [{request_id}]")
                    except Exception:
                        # failed engine-side: nothing left to abort for this id.
                        # On cancellation the id stays tracked so the batch
                        # failure path aborts it engine-side.
                        inflight.discard(request_id)
                        raise
                    inflight.discard(request_id)
                except Exception:
                    # engine failures stay batch-fatal (cancel siblings + abort
                    # in-flight below); only the media step above degrades
                    _log.error(
                        "[media] generate failed request=%s prompt_chars=%d"
                        " prompt_head=%r prompt_tail=%r images=%d videos=%d",
                        req.request_id,
                        len(item.prompt),
                        item.prompt[:120],
                        item.prompt[-120:],
                        len(item.images),
                        len(item.videos),
                    )
                    raise
                progress.step()
                return ChatResponse(
                    text=out.outputs[0].text,
                    request_id=req.request_id,
                    raw=out,
                    finish_reason=out.outputs[0].finish_reason,
                    usage=_usage_from_output(out),
                )

        tasks = [
            asyncio.create_task(_generate_one(req, item))
            for req, item in zip(requests, prepared, strict=True)
        ]
        try:
            # same first-error semantics as gather_bounded: cancel pending
            # siblings, then abort their engine-side requests before propagating
            responses = await gather_cancel_on_error(tasks)
        except BaseException:
            await _abort_inflight(engine, inflight)
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if media_failures:
            _log.warning(
                "[media] %d/%d requests failed media load: per-request sentinel"
                " (finish_reason=%s) returned, batch continues; first failure: %s",
                len(media_failures),
                len(requests),
                OVER_WINDOW_FINISH,
                media_failures[0],
            )
        ready = (decode_done["t"] or t0) - t0
        _log.info(
            "[media] batch done requests=%d videos=%d ready=%.1fs wall=%.1fs",
            len(requests),
            n_videos,
            ready,
            time.monotonic() - t0,
        )
        return responses

    def close(self) -> None:
        """Release the prefetch process pool and engine slot.

        The engine and output handler are bound to run_sync's process-level
        resident loop; shutdown must be dispatched back onto that loop
        (cross-thread shutdown is unverified and must not be called directly).
        """
        if self._loader is not None:
            self._loader.close()
            self._loader = None
        run_sync(self._release_on_loop())

    async def _release_on_loop(self) -> None:
        self.slot.release()
