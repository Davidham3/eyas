"""Shared batch scheduling primitives (batch scheduling pushed down into
backends).

The pair shared by the inference backends and the Evaluator:

- ``gather_bounded``: the request-level concurrency primitive — semaphore rate
  limiting + order-preserving gather; the result order always equals the
  factory input order. The OpenAI backend's HTTP concurrency and the Evaluator's
  judge concurrency are the same pattern. On the first exception pending
  siblings are cancelled (``gather_cancel_on_error``) and the original exception
  type propagates — a failed batch must not keep burning tokens/GPU.
- Large-batch in-batch observability lives uniformly at this layer:
  ``BatchProgress`` bumps on every completion (the watchdog activity signal) and
  emits ``[batch.progress]`` at reporting points; ``watch()`` starts a
  zero-progress guard — ``[batch.progress]`` only appears on completions, so
  "dispatched but zero completions" (queuing behind other work — an engine, a
  rate limiter) is exactly its blind spot; ``[batch.stall]`` fills that gap every
  10min (a dispatched batch can stay at zero completions indefinitely). The
  vllm sliding window / openai gather paths all share the same signals.
- ``run_sync``: the unified driving loop for synchronous convenience entries — a
  process-level resident event loop. On first inference the vllm in-process
  Engine binds resident asyncio tasks such as the output handler to the
  then-running loop (the handler task is created only once); if every call built
  a fresh loop via asyncio.run, the second call onwards would wait forever for a
  response (GPU idle, a silent hang with no logs or errors) — the
  same long-lived loop must be shared.

Scheduling duties NOT in this layer: prompt-major request ordering lives in the
Evaluator (radix-cache prefix reuse depends on candidate-first expansion); local
image prefetching lives in ``eyas.backends.prefetch`` (``MediaPrefetchLoader``) and
is opt-in via ``prefetch_workers`` — it is not a scheduling-layer duty.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Sequence
from typing import Any

from eyas.logging import get_logger
from eyas.watchdog import bump

__all__ = ["BatchProgress", "gather_bounded", "gather_cancel_on_error", "run_sync"]

_log = get_logger(__name__)

_shared_loop: asyncio.AbstractEventLoop | None = None
# Guards loop CREATION only (double-checked): concurrent first callers from
# multiple threads must share one resident loop — an unlocked check-then-create
# lets two threads each build a loop, and the engine tasks bound to the losing
# loop would never be driven again. Loop execution stays single-caller (the
# framework issues one synchronous call at a time).
_loop_lock = threading.Lock()


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number of seconds, got {raw!r}") from exc


# Zero-progress alert threshold: by default alert after 10min with no new
# completion, then repeat every 10min; <=0 disables. Difference from the watchdog
# (``EYAS_WATCHDOG_STALL_SEC``, default 1200s): the watchdog handles "process
# self-termination", this handles "making a human see it".
BATCH_STALL_SEC = _env_seconds("EYAS_BATCH_STALL_SEC", 600.0)
BATCH_STALL_REPEAT_SEC = _env_seconds("EYAS_BATCH_STALL_REPEAT_SEC", 600.0)
BATCH_WATCH_INTERVAL_SEC = _env_seconds("EYAS_BATCH_WATCH_INTERVAL_SEC", 30.0)
# Time-based backstop throttling for the progress heartbeat: besides emitting at
# every multiples of the completion count, also force an emission when the time
# since the last report exceeds these seconds — a slow engine can take minutes
# per 500 items, and under pure count throttling the log
# goes quiet long enough to look hung; fast engines still throttle by count to
# avoid log flooding. <=0 disables the time backstop (pure count throttling).
BATCH_PROGRESS_MIN_INTERVAL_SEC = _env_seconds("EYAS_BATCH_PROGRESS_MIN_INTERVAL_SEC", 60.0)


def run_sync[T](awaitable: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine synchronously on the process-level resident event loop
    (all synchronous convenience entries go through here).

    Same fail-fast rule as asyncio.run: calling inside a running event loop
    raises RuntimeError immediately. The loop is created on first call and reused
    until process exit — the engine's handle_loop / MM preprocessing tasks bind
    only once and stay alive across evaluate/generate calls.
    """
    global _shared_loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("run_sync cannot be called inside a running event loop")
    if _shared_loop is None:
        with _loop_lock:
            if _shared_loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _shared_loop = loop
    return _shared_loop.run_until_complete(awaitable)


async def gather_cancel_on_error[T](tasks: Sequence[asyncio.Task[T]]) -> list[T]:
    """Order-preserving gather with first-error sibling cancellation.

    Plain ``asyncio.gather`` propagates the first exception immediately but
    leaves siblings running (burning tokens/GPU for a batch that already
    failed). On the first exception this cancels every pending sibling, awaits
    their unwind (swallowing their CancelledError / late failures) and re-raises
    the ORIGINAL exception — the exception type is preserved (no TaskGroup /
    ExceptionGroup: callers catch the original type). External cancellation and
    a factory raising CancelledError propagate as cancellation.
    """
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def gather_bounded[T](
    factories: Iterable[Callable[[], Awaitable[T]]], concurrency: int
) -> list[T]:
    """Semaphore rate limiting + order-preserving gather: result order = the
    ``factories`` input order, independent of completion order. On the first
    exception pending siblings are cancelled and the original exception
    propagates (see ``gather_cancel_on_error``).

    Every completion bumps via ``BatchProgress.step`` (the watchdog activity
    signal, counted for small batches too) and emits ``[batch.progress]`` at
    large-batch reporting points; the zero-progress guard via ``watch()`` is on
    duty throughout.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    items = list(factories)
    semaphore = asyncio.Semaphore(concurrency)
    progress = BatchProgress(len(items))
    watcher = progress.watch()

    async def _bounded(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            result = await factory()
            progress.step()
            return result

    tasks = [asyncio.create_task(_bounded(f)) for f in items]
    try:
        return await gather_cancel_on_error(tasks)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


async def _watch_stall(progress: BatchProgress) -> None:
    """Zero-progress guard coroutine: alert once the time since the latest
    completion exceeds ``BATCH_STALL_SEC``, then repeat every
    ``BATCH_STALL_REPEAT_SEC``. ``[batch.progress]`` only appears on completions —
    the delay before the first completion and mid-batch pauses are its blind
    spots, which this guard covers.
    """
    if BATCH_STALL_SEC <= 0:
        return
    last_report = 0.0
    try:
        while True:
            await asyncio.sleep(BATCH_WATCH_INTERVAL_SEC)
            now = time.monotonic()
            idle = now - progress.last_activity
            if idle < BATCH_STALL_SEC:
                continue
            if last_report >= progress.last_activity and now - last_report < BATCH_STALL_REPEAT_SEC:
                continue
            last_report = now
            _log.warning(
                "[batch.stall] %d/%d completed, %.0fs with no new completion — requests"
                " stuck queuing/routing/engine-side (a healthy large batch has"
                " [batch.progress] heartbeats; this alert = true zero progress)",
                progress.done,
                progress.total,
                idle,
            )
            bump("batch.stall")
    except asyncio.CancelledError:
        pass


class BatchProgress:
    """Large-batch in-batch completion progress + activity signal: bump on every
    completion, log at intervals for large batches.

    A single large agenerate batch (the Evaluator's full miss set, a whole
    compare round, 60k scale) produces zero caller-side output from
    submission to return — in-batch progress is the only liveness signal, and a
    silent batch ≠ a hang (absence of output is not evidence of a stall; the
    stall alert below exists because of that). Batches smaller than
    log_threshold (judge fan-outs etc.) do not log but still bump — small-batch
    completions are watchdog activity signals too.
    """

    def __init__(self, total: int, *, every: int = 500, log_threshold: int = 500) -> None:
        self.total = total
        self.every = every
        self.log_threshold = log_threshold
        self.done = 0
        self._t0 = time.monotonic()
        self._last_emit = self._t0
        self.last_activity = self._t0

    def step(self) -> None:
        """Record one request completion: always bump; log separately when a large
        batch reaches a reporting point.

        Reporting points = completion count hitting a multiple of every
        (including the tail) or more than ``BATCH_PROGRESS_MIN_INTERVAL_SEC``
        since the last report (time backstop for slow engines); the line carries
        rate and ETA.
        """
        self.done += 1
        now = time.monotonic()
        self.last_activity = now
        bump(f"batch.progress [{self.done}/{self.total}]")
        if self.total < self.log_threshold:
            return
        if self.done % self.every != 0 and self.done != self.total:
            if (
                BATCH_PROGRESS_MIN_INTERVAL_SEC <= 0
                or now - self._last_emit < BATCH_PROGRESS_MIN_INTERVAL_SEC
            ):
                return
        self._emit(now)

    def _emit(self, now: float) -> None:
        self._last_emit = now
        elapsed = now - self._t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = f", eta ~{(self.total - self.done) / rate:.0f}s" if rate > 0 else ""
        _log.info(
            "[batch.progress] %d/%d completed in %.1fs (%.2f req/s%s)",
            self.done,
            self.total,
            elapsed,
            rate,
            eta,
        )

    def watch(self) -> asyncio.Task[None]:
        """Start the zero-progress guard task; the caller cancels it at batch end."""
        return asyncio.create_task(_watch_stall(self))
