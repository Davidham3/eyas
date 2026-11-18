"""Framework-wide logging (stdlib logging wrapper, zero new dependencies).

Conventions:
- Component code takes ``_log = get_logger(__name__)`` and only emits logs, never
  configures;
- Configuration happens only at the run entry point (run_experiment): the calling
  thread does nothing but a **bounded queue enqueue** (drops and counts when full;
  emit never blocks, never raises). Two channels: diagnostic key
  events (``key_event`` / ``extra={"eyas_priority": True}``) ride an UNBOUNDED
  priority queue the writer thread drains first — evidence logs survive
  backpressure; ordinary INFO rides the droppable queue.
  Console and FileHandler all hang off the
  background writer thread — a stalled log sink only freezes the
  writer thread, never the caller (no handler may run on the caller's thread: the
  same stall degrades to log loss, with a drop report after recovery — log loss is
  never silent; sink exceptions are counted and surfaced in the same report);
- The console handler holds a dynamic stdout proxy (each write resolves the current
  ``sys.stdout``, so a pytest capsys buffer swap is followed automatically); the
  FileHandler writes run-dir framework.log (full format: timestamp + level + logger
  name, distinct from the console's bare message lines);
- Before setup, INFO level is silent (the "eyas" logger has no handlers, root defaults
  to WARNING) — zero noise for library users, and contract tests exercising components
  directly do not flood the output;
- The watchdog (diagnostics before os._exit, must be emitted without depending on
  logging initialization) and CLI command output (command results, not logs) keep
  using print, not this module.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "drain",
    "dropped_count",
    "get_logger",
    "key_event",
    "resolve_log_level",
    "setup",
    "sink_error_count",
]

_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_QUEUE_MAXSIZE = 10_000
_PRIORITY_POLL_SEC = 0.5  # writer-thread poll while blocked on the droppable queue
_STOP = object()

_console_handler: logging.Handler | None = None
_file_handler: logging.FileHandler | None = None
_enqueue: _EnqueueHandler | None = None
_listener: threading.Thread | None = None
_sinks: list[logging.Handler] = []
_sink_lock = threading.Lock()
# Sink exceptions swallowed by _handle_all, counted under _sink_lock and
# surfaced in the writer thread's recovery report ("log loss is never silent").
_sink_errors = 0


class _CurrentStdout:
    """Dynamic proxy for the current sys.stdout — the handler is constructed once and
    follows capsys buffer swaps automatically."""

    def write(self, s: str) -> int:
        return sys.stdout.write(s)

    def flush(self) -> None:
        sys.stdout.flush()


class _EnqueueHandler(logging.Handler):
    """Caller-side handler: bounded-queue enqueue, drops and counts when full.

    emit never blocks, never raises — when the write path (a blocked FileHandler /
    a full stdout pipe) is wedged, the caller at worst loses logs, never control;
    the drop count is exposed via dropped_count(), and the writer thread's drop report
    restores visibility after recovery.

    Two channels: records flagged ``extra={"eyas_priority": True}``
    (key diagnostic events — assembly, round echo, planner rejects) go on an
    UNBOUNDED priority queue the writer thread drains first; ordinary INFO rides
    the droppable bounded queue. Key evidence therefore survives backpressure —
    "evidence logs also dropped whenever full" is structurally closed. The caller
    still never blocks: the priority queue is unbounded (key events are rare by
    construction; the queue depth is the only growth risk, never a stall).
    """

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.q: queue.Queue = queue.Queue(maxsize)
        self.priority_q: queue.Queue = queue.Queue()  # unbounded by contract
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if getattr(record, "eyas_priority", False):
                self.priority_q.put_nowait(record)
            else:
                self.q.put_nowait(record)
        except queue.Full:
            with _sink_lock:  # emit runs on many caller threads: keep the count exact
                self.dropped += 1

    def pick(self) -> tuple[queue.Queue, logging.LogRecord]:
        """Writer-thread selection: priority queue fully first, then one droppable
        record (blocking on the droppable queue only, polled so a late-arriving
        priority record is not held hostage to ordinary traffic)."""
        try:
            return self.priority_q, self.priority_q.get_nowait()
        except queue.Empty:
            pass
        while True:
            try:
                return self.q, self.q.get(timeout=_PRIORITY_POLL_SEC)
            except queue.Empty:
                try:
                    return self.priority_q, self.priority_q.get_nowait()
                except queue.Empty:
                    pass


def _handle_all(record: logging.LogRecord) -> None:
    global _sink_errors
    with _sink_lock:
        sinks = list(_sinks)
    for handler in sinks:
        try:
            handler.handle(record)
        except Exception:
            # one broken sink must not take down the others; the loss is
            # counted and surfaced in the next recovery report, never silent
            with _sink_lock:
                _sink_errors += 1


def _listen(enqueue: _EnqueueHandler) -> None:
    """Background writer thread: priority-retained records drain first, then
    droppable INFO; every record goes to all sinks. On noticing a drop or
    sink-exception increment, emits the make-up report after recovery."""
    reported_drops = 0
    reported_sink_errors = 0
    while True:
        source, record = enqueue.pick()
        try:
            if record is _STOP:
                return
            _handle_all(record)
            with _sink_lock:
                dropped_now = enqueue.dropped
                sink_errors_now = _sink_errors
            if dropped_now > reported_drops or sink_errors_now > reported_sink_errors:
                pending_drops = dropped_now - reported_drops
                pending_sink_errors = sink_errors_now - reported_sink_errors
                reported_drops = dropped_now
                reported_sink_errors = sink_errors_now
                clauses = []
                if pending_drops:
                    clauses.append(
                        f"{pending_drops} log records were dropped earlier when the queue "
                        "was full (write path blocked: sink stalled / pipe full — the run "
                        "itself is unaffected)"
                    )
                if pending_sink_errors:
                    clauses.append(
                        f"{pending_sink_errors} sink writes raised earlier (exception "
                        "swallowed per sink — the run itself is unaffected)"
                    )
                _handle_all(
                    logging.LogRecord(
                        "eyas.logging",
                        logging.WARNING,
                        __file__,
                        0,
                        f"[logging] writer thread recovered; {'; '.join(clauses)}",
                        None,
                        None,
                    )
                )
        finally:
            source.task_done()


def get_logger(name: str) -> logging.Logger:
    """Accepts a module ``__name__`` (eyas.xxx) or a short name, normalizing to the
    single "eyas." prefix."""
    return logging.getLogger(f"eyas.{name.removeprefix('eyas.')}")


_NAMED_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def resolve_log_level(env: Mapping[str, str] | None = None) -> int:
    """Resolve the framework log level from ``EYAS_LOG_LEVEL`` (case-insensitive
    DEBUG/INFO/WARNING/ERROR); unset → INFO. Invalid values fail fast — never a
    silent fallback to the default."""
    raw = (os.environ if env is None else env).get("EYAS_LOG_LEVEL")
    if raw is None:
        return logging.INFO
    level = _NAMED_LEVELS.get(raw.upper())
    if level is None:
        accepted = " | ".join(_NAMED_LEVELS)
        raise ValueError(f"invalid EYAS_LOG_LEVEL={raw!r}; accepted: {accepted}")
    return level


def key_event(logger: logging.Logger, msg: str, *args: Any) -> None:
    """Emit a diagnostic KEY EVENT at INFO on the priority-retained channel:
    the record rides the unbounded priority queue, so a full or
    stalled droppable queue can never bury the run's evidence trail (assembly,
    round echo, planner rejects). Key events are RARE by construction — an emit
    site that fires per request is ordinary traffic and must not use this."""
    logger.info(msg, *args, extra={"eyas_priority": True})


def setup(*, file: Path | None = None, level: int = logging.INFO) -> None:
    """Idempotently configure the "eyas" root logger; repeated calls with the same file
    have no side effects."""
    global _console_handler, _file_handler, _enqueue, _listener

    logger = logging.getLogger("eyas")
    logger.setLevel(level)
    logger.propagate = False

    if _enqueue is None:
        _enqueue = _EnqueueHandler(_QUEUE_MAXSIZE)
        logger.addHandler(_enqueue)
        with _sink_lock:
            if _console_handler is None:
                _console_handler = logging.StreamHandler(_CurrentStdout())
                _console_handler.setFormatter(logging.Formatter("%(message)s"))
                _sinks.append(_console_handler)
        _listener = threading.Thread(
            target=_listen, args=(_enqueue,), name="eyas-logging", daemon=True
        )
        _listener.start()
        atexit.register(drain, 30.0)

    with _sink_lock:
        if _file_handler is not None and (
            file is None or Path(_file_handler.baseFilename) != file.resolve()
        ):
            _sinks.remove(_file_handler)
            _file_handler.close()
            _file_handler = None
        if _file_handler is None and file is not None:
            file.parent.mkdir(parents=True, exist_ok=True)
            _file_handler = logging.FileHandler(file, encoding="utf-8")
            _file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
            _sinks.append(_file_handler)


def drain(timeout: float = 10.0) -> bool:
    """Wait for the background writer thread to drain BOTH queues (run teardown
    and tests); returns False on timeout. The priority queue counts too — a
    drain returning True means every emitted record (either channel) has
    reached the sinks."""
    if _enqueue is None:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not (_enqueue.q.unfinished_tasks or _enqueue.priority_q.unfinished_tasks):
            return True
        time.sleep(0.01)
    return not (_enqueue.q.unfinished_tasks or _enqueue.priority_q.unfinished_tasks)


def dropped_count() -> int:
    """Cumulative records dropped when the queue was full (0 = no degradation)."""
    return _enqueue.dropped if _enqueue is not None else 0


def sink_error_count() -> int:
    """Cumulative sink exceptions swallowed on the writer thread (0 = no
    degradation); each increment is surfaced in the next recovery report."""
    with _sink_lock:
        return _sink_errors


def _reset() -> None:
    """Test helper: stop the writer thread, remove all handlers, restore the
    unconfigured state."""
    global _console_handler, _file_handler, _enqueue, _listener, _sinks, _sink_errors
    if _enqueue is not None and _listener is not None and _listener.is_alive():
        _enqueue.q.put(_STOP)
        _listener.join(timeout=5)
    logger = logging.getLogger("eyas")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    with _sink_lock:
        sinks, _sinks = _sinks, []
    for handler in sinks:
        try:
            handler.close()
        except Exception:
            pass
    _console_handler = None
    _file_handler = None
    _enqueue = None
    _listener = None
    with _sink_lock:
        _sink_errors = 0
