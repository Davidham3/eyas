"""Trace recorder: captures all LLM calls at a single point in the backend layer, one
JSONL per run.

The capture point is the InferenceBackend decorator (TracedBackend): evaluation, judging,
and operator LLM calls all go through backends, so wrapping at assembly time gives full
coverage with zero changes to user code. Two event kinds: llm_batch (batch metadata
+ actual sampling params + batch-level latency — per-request latency under batched
generate is not the true value) and llm_call (single-request messages + full response
text). Text is captured in full; images and videos record only the path, with a
digest for data: URLs (so base64 cannot blow up the file). Layering: the evaluation
record store is the metrics index, trace is the full process record. record() is the
single point: inside the lock, beyond serializing the JSONL line it synchronously
accumulates the call ledger (eval/judge/opt buckets); ``totals()`` feeds run_meta["cost"]
directly — report prefers the ledger and falls back to scanning the trace for run
dirs without one. Disk IO rides a dedicated per-recorder writer thread (same
isolation as eyas.logging): the caller only enqueues the serialized line into a
bounded queue, so a stalled trace file never freezes the event loop; ``close()``
is the flush barrier (no event lost, ordering preserved).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import queue
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eyas.backends.base import (
    BackendWrapper,
    ChatRequest,
    ChatResponse,
    ImagePart,
    InferenceBackend,
    LoadedImage,
    LoadedVideo,
    SamplingConfig,
    VideoPart,
)

__all__ = ["TraceRecorder", "TracedBackend", "bucket_batch", "trace_context"]

_QUEUE_MAXSIZE = 10_000  # bounded memory; a sustained disk stall backpressures the caller
_FLUSH_POLL_SEC = 0.005

_trace_labels: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "eyas_trace_labels", default=None
)


@contextmanager
def trace_context(**labels: str) -> Iterator[None]:
    """Attach dynamic labels to LLM call events occurring within this context (merged
    with any existing labels).

    The Evaluator uses it to tag phase=evaluate/judge; operator calls set no labels —
    attribution relies on the request_id convention.
    """
    token = _trace_labels.set({**(_trace_labels.get() or {}), **labels})
    try:
        yield
    finally:
        _trace_labels.reset(token)


def _redact_part(part: Any) -> dict[str, Any]:
    if isinstance(part, ImagePart):
        value = part.image
        if isinstance(value, LoadedImage):
            return {
                "type": "image",
                "path": value.path,
                "inline_sha16": hashlib.sha256(value.raw).hexdigest()[:16],
                "bytes": len(value.raw),
            }
        if value.startswith("data:"):
            return {
                "type": "image",
                "inline_sha16": hashlib.sha256(value.encode()).hexdigest()[:16],
                "bytes": len(value),
            }
        return {"type": "image", "path": value}
    if isinstance(part, VideoPart):
        video = part.video
        if isinstance(video, LoadedVideo):
            return {"type": "video", "path": video.path}
        if video.startswith("data:"):
            return {
                "type": "video",
                "inline_sha16": hashlib.sha256(video.encode()).hexdigest()[:16],
                "bytes": len(video),
            }
        return {"type": "video", "path": video}
    return {"type": "text", "text": part.text}


def _redact_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": [_redact_part(p) for p in m.content]} for m in messages]


def bucket_batch(slot: str | None, context_phase: str | None) -> str:
    """Bucket an llm_batch (the ledger's convention, shared with report.py so the two
    implementations cannot drift): judge slot or judge phase -> judge (when config has
    no dedicated judge backend, judge calls borrow the candidate slot and the evaluator
    tags phase="judge" — slot-only matching would mis-count judge calls as opt);
    evaluate phase -> eval; everything else (operators /
    analysis / rewrites) -> opt."""
    if slot == "judge" or context_phase == "judge":
        return "judge"
    if context_phase == "evaluate":
        return "eval"
    return "opt"


class TraceRecorder:
    """Per-run trace file (JSONL, append-only, flushed per line by a dedicated
    writer thread) + call ledger under a lock.

    Thread-safe: the candidate / judge TracedBackend slots share one recorder.
    labels are run-level static labels (given at assembly time, e.g. optimizer /
    run_id), merged into the top level of each event.

    Off-loop writes: record() serializes the event under the lock and only
    enqueues the line into a bounded queue (ordering = enqueue order = seq
    order); a background daemon thread does the disk IO, so a stalled trace
    file never freezes the caller. When the queue is full the caller
    backpressures (trace is the full process record — events are never
    silently dropped). flush() waits for the queue to drain; close() is the
    flush barrier + writer shutdown (no event lost on the healthy path).
    """

    def __init__(self, path: Path, labels: dict[str, str] | None = None) -> None:
        self.path = Path(path)
        self.labels = dict(labels or {})
        self._lock = threading.Lock()
        self._event_seq = 0
        self._batch_seq = 0
        self._totals: dict[str, int] = {
            "eval_calls": 0,
            "judge_calls": 0,
            "opt_calls": 0,
            "n_batches": 0,
            "total_requests": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "usage_requests": 0,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._queue: queue.Queue[str | None] = queue.Queue(_QUEUE_MAXSIZE)
        self._closed = False
        self._writer_error: Exception | None = None
        self._writer = threading.Thread(target=self._write_loop, name="eyas-trace", daemon=True)
        self._writer.start()

    def _write_loop(self) -> None:
        """Background writer: the only thread touching the file handle. A
        write failure parks the error (surfaced fail-fast by record/flush)
        and exits — the queue's bounded put can then never wedge forever
        because close/flush watch the thread's liveness."""
        while True:
            line = self._queue.get()
            try:
                if line is None:  # close() sentinel: every prior event is already written
                    return
                self._fh.write(line)
                self._fh.flush()
            except Exception as exc:
                self._writer_error = exc
                return
            finally:
                self._queue.task_done()

    def record(self, kind: str, **fields: Any) -> None:
        with self._lock:
            if self._closed:
                raise ValueError(f"trace recorder is closed: {self.path}")
            self._event_seq += 1
            context = _trace_labels.get() or {}
            event = {
                "seq": self._event_seq,
                "ts": round(time.time(), 6),
                "kind": kind,
                **self.labels,
                **fields,
                "context": context,
            }
            line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
            self._put_line(line)
            self._tally(kind, fields, context)

    def _put_line(self, line: str) -> None:
        """Bounded enqueue (called under the lock): a sustained disk stall
        backpressures the caller instead of growing memory or dropping
        events; a dead writer (disk failure) surfaces fail-fast rather than
        wedging the caller on a full queue."""
        while True:
            if self._writer_error is not None:
                raise RuntimeError(f"trace writer thread failed ({self.path})") from (
                    self._writer_error
                )
            try:
                self._queue.put(line, timeout=0.1)
                return
            except queue.Full:
                if not self._writer.is_alive():
                    raise RuntimeError(f"trace writer thread died ({self.path})") from (
                        self._writer_error
                    )

    def _tally(self, kind: str, fields: dict[str, Any], context: dict[str, str]) -> None:
        totals = self._totals
        if kind == "llm_batch":
            n = int(fields.get("n_requests", 0))
            totals["n_batches"] += 1
            totals["total_requests"] += n
            totals[f"{bucket_batch(fields.get('slot'), context.get('phase'))}_calls"] += n
        elif kind == "llm_call":
            usage = fields.get("usage")
            if isinstance(usage, dict):
                # usage follows the billing convention: cache hits never enter the trace
                # -> not counted
                totals["usage_requests"] += 1
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if prompt is not None:
                    totals["prompt_tokens"] += int(prompt)
                if completion is not None:
                    totals["completion_tokens"] += int(completion)
                tokens = usage.get("total_tokens")
                if tokens is None:
                    tokens = int(prompt or 0) + int(completion or 0)
                totals["total_tokens"] += int(tokens)

    def totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._totals)

    def next_batch_id(self) -> str:
        with self._lock:
            self._batch_seq += 1
            return f"b{self._batch_seq}"

    def flush(self) -> None:
        """Flush barrier: every event recorded before this call has reached
        the file (external readers may scan it while the recorder stays open).
        A failed writer surfaces its error instead of hanging."""
        while self._queue.unfinished_tasks and self._writer.is_alive():
            time.sleep(_FLUSH_POLL_SEC)
        if self._writer_error is not None:
            raise RuntimeError(f"trace writer thread failed ({self.path})") from (
                self._writer_error
            )

    def close(self) -> None:
        """Flush barrier + writer shutdown: the sentinel queues behind every
        recorded event, the join waits for the writer to finish, then the
        file handle closes — no event lost on the healthy path. Idempotent;
        teardown never raises (a failed writer already surfaced via
        record/flush)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # liveness loop: close stays finite even when the writer died
            # with a full queue
            while self._writer.is_alive():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
        self._writer.join()
        self._fh.close()


class TracedBackend(BackendWrapper, InferenceBackend):
    """Recording backend: delegates to inner, writes llm_batch + llm_call events around
    agenerate.

    Wrapped at assembly time (the Evaluator is prebuilt inside RunContext, so wrapping
    must happen before its construction):

        recorder = TraceRecorder(run_dir / "trace.jsonl", labels={"optimizer": "cot"})
        backend = TracedBackend(raw_backend, recorder, trace_slot="candidate")
    """

    def __init__(
        self, inner: InferenceBackend, recorder: TraceRecorder, trace_slot: str = "candidate"
    ) -> None:
        super().__init__(inner)
        self.recorder = recorder
        # Caution: our own trace_slot is a trace bucket name (a string) and must
        # not shadow the inner VllmSlot — when an optimizer releases
        # the engine via getattr(backend, "slot"), the BackendWrapper property
        # must reach through to the real slot. (BackendWrapper.close also passes
        # through; closing the recorder belongs to run_experiment.)
        self.trace_slot = trace_slot
        # declared limits pass through the wrapper chain (runtime arms the
        # planner guard from the outermost backend, raw→traced→normalized)
        self.context_window = inner.context_window
        self.max_images = inner.max_images

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        batch_id = self.recorder.next_batch_id()
        start = time.perf_counter()
        try:
            responses = await self.inner.agenerate(requests, sampling)
        except BaseException as exc:
            self._record_batch(batch_id, requests, sampling, start, error=repr(exc))
            raise
        self._record_batch(batch_id, requests, sampling, start)
        for index, (req, resp) in enumerate(zip(requests, responses, strict=True)):
            call: dict[str, Any] = {
                "batch_id": batch_id,
                "index": index,
                "request_id": req.request_id,
                "messages": _redact_messages(req.messages),
                "response_text": resp.text,
            }
            if req.raw_prompt is not None:
                call["raw_prompt"] = req.raw_prompt
            if resp.request_id is not None:
                call["response_request_id"] = resp.request_id
            if resp.usage is not None:
                call["usage"] = resp.usage
            if resp.finish_reason is not None:
                call["finish_reason"] = resp.finish_reason
            self.recorder.record("llm_call", **call)
        return responses

    def _record_batch(
        self,
        batch_id: str,
        requests: Sequence[ChatRequest],
        sampling: SamplingConfig,
        start: float,
        error: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "batch_id": batch_id,
            "slot": self.trace_slot,
            "backend": type(self.inner).__name__,
            "n_requests": len(requests),
            "sampling": asdict(sampling),
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        }
        if error is not None:
            fields["error"] = error
        self.recorder.record("llm_batch", **fields)
