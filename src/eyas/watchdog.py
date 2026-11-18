"""Process-level activity watchdog: a hang costs more than a crash (an idle GPU), so
self-terminate once inactivity exceeds the threshold.

Background: an engine call can block indefinitely — the process is alive, the GPU
idle, the job never exits. Failures must fail fast, and so must hangs:
the scheduling layer (gather_bounded) bumps the activity timestamp after
every engine interaction; when the watchdog daemon thread sees inactivity past the
threshold it prints diagnostics and calls ``os._exit(99)``, letting the job exit as a
failure and release the GPU.

Follow-up form: a stalled log write path can wedge the stdout pipe; a blocking
``print`` would then never finish and the kill would never land —
the diagnose-then-kill order therefore carries a hard constraint: stderr
diagnostics go FIRST (best-effort O_NONBLOCK write of the message AND an
unconditional all-thread stack snapshot, abandoned if the pipe is full,
affecting nothing beyond visibility), then a best-effort file write to
``$EYAS_DIAG_DIR`` (default ``/tmp``) under a hard 2s bound (daemon thread +
join timeout — a stalled shared disk is abandoned, never waited on; even a
thread-start failure is swallowed), then an unconditional ``os._exit(99)``.
The kill must never depend on any IO succeeding.

The threshold defaults to 1200s; set it above the longest legitimate silence
(engine load/warmup, a single long-generation block). ``EYAS_WATCHDOG_STALL_SEC``
tunes it, <=0 disables it explicitly.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

__all__ = ["bump", "start_watchdog", "stop_watchdog", "stalled_for"]

_DEFAULT_STALL_SEC = 1200.0
_last_activity = 0.0
_last_source = "startup"
_started = False
_stop_event: threading.Event | None = None


def _diag_dir() -> str:
    """Where hang diagnostics land (``EYAS_DIAG_DIR``, default ``/tmp``), read
    at call time so an environment set after this module's import still takes
    effect. Set ``EYAS_DIAG_DIR`` if ``/tmp`` is not writable or not persisted."""
    return os.environ.get("EYAS_DIAG_DIR", "/tmp")


def bump(source: str = "") -> None:
    """Record one activity (engine call completed, batch progress, etc.). source only
    makes hang diagnostics readable."""
    global _last_activity, _last_source
    _last_activity = time.monotonic()
    if source:
        _last_source = source


def stalled_for(now: float | None = None) -> float:
    """Seconds since the last activity (the now parameter lets tests inject a clock)."""
    return (now if now is not None else time.monotonic()) - _last_activity


#: Hard bound on the best-effort diagnostics file write: a stalled
#: ``EYAS_DIAG_DIR`` (e.g. a network filesystem) must never delay the kill beyond this.
_DIAG_FILE_BOUND_SEC = 2.0


def _stderr_write(payload: str) -> None:
    """Best-effort O_NONBLOCK stderr write: a wedged pipe means partial drops,
    affecting nothing beyond visibility. Any failure is swallowed — diagnostics
    are a byproduct of the kill and must never retaliate against it."""
    try:
        import fcntl

        flags = fcntl.fcntl(2, fcntl.F_GETFL)
        fcntl.fcntl(2, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            os.write(2, payload.encode("utf-8"))
        finally:
            fcntl.fcntl(2, fcntl.F_SETFL, flags)
    except Exception:
        pass


def _write_diag_file(message: str) -> None:
    """Write the message + a full-thread stack snapshot (faulthandler) to the
    diagnostics file. An activity label such as "last activity: engine.agenerate"
    only gives coarse localization — the stack snapshot pins the hang point
    down to a frame. Any failure is swallowed: the stderr channel already
    carries message + stacks unconditionally, the file is a bonus copy."""
    try:
        diag_path = Path(_diag_dir()) / f"eyas-watchdog-{os.getpid()}.log"
        diag_path.write_text(message, encoding="utf-8")
        import faulthandler

        with diag_path.open("a", encoding="utf-8") as fh:
            faulthandler.dump_traceback(file=fh, all_threads=True)
    except Exception:
        pass


def _stderr_write_stacks() -> None:
    """Unconditional all-thread stack snapshot to stderr, independent of the
    diagnostics file: when ``EYAS_DIAG_DIR`` stalls (the very scenario the
    watchdog exists for) the file write is abandoned, and stderr — typically
    captured by the caller — is the only channel that outlives the process.
    The snapshot is captured through a non-blocking anonymous pipe
    (faulthandler writes at the C level straight to the fd; O_NONBLOCK means
    a full pipe buffer drops the tail instead of blocking — no disk IO is
    involved in the capture), then handed to the same best-effort
    non-blocking stderr write as the message. Any failure is swallowed —
    diagnostics never retaliate against the kill."""
    try:
        import faulthandler

        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(write_fd, False)
            with os.fdopen(write_fd, "w", encoding="utf-8", closefd=False) as fh:
                faulthandler.dump_traceback(file=fh, all_threads=True)
            os.close(write_fd)
            write_fd = -1
            chunks: list[bytes] = []
            while chunk := os.read(read_fd, 65536):
                chunks.append(chunk)
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
        _stderr_write(b"".join(chunks).decode("utf-8", errors="replace"))
    except Exception:
        pass


def _emit_diagnostics(message: str) -> None:
    """Best-effort dual write of hang diagnostics, kill-first ordered:

    1. stderr FIRST (O_NONBLOCK, non-blocking): the message, then the
       all-thread stack snapshot UNCONDITIONALLY — both land even when the
       diagnostics disk is stalled (the snapshot is captured in memory, never
       gated on the file write);
    2. the file write (``$EYAS_DIAG_DIR``, message + stack snapshot) runs on a
       daemon thread under a hard ``_DIAG_FILE_BOUND_SEC`` join timeout — a
       stalled shared disk is abandoned, never waited on beyond the bound.
       Thread creation/start/join failures (e.g. thread exhaustion in a wedged
       process) are swallowed like any IO failure.

    No file echo: the stacks already reached stderr in step 1, so echoing the
    file content would double-dump them. Every failure on every path is
    swallowed; the caller's ``os._exit(99)`` fires unconditionally once this
    returns (within the bound).
    """
    _stderr_write(message)
    _stderr_write_stacks()
    try:
        writer = threading.Thread(
            target=_write_diag_file, args=(message,), name="eyas-watchdog-diag", daemon=True
        )
        writer.start()
        writer.join(timeout=_DIAG_FILE_BOUND_SEC)
    except Exception:
        pass


def start_watchdog(
    stall_sec: float | None = None,
    *,
    interval: float = 30.0,
    kill: Callable[[int], object] | None = None,
) -> None:
    """Start the daemon watchdog thread (idempotent). stall_sec <= 0 disables it
    explicitly.

    On detecting inactivity past the threshold, prints diagnostics and calls kill
    (default ``os._exit(99)``) — os._exit is mandatory: hangs often originate in the
    event loop / thread pool itself, and a graceful exit never arrives.
    """
    global _started
    if _started:
        return
    if stall_sec is None:
        stall_sec = float(os.environ.get("EYAS_WATCHDOG_STALL_SEC", _DEFAULT_STALL_SEC))
    if stall_sec <= 0:
        return
    _started = True
    bump("watchdog started")
    do_kill = kill if kill is not None else os._exit
    stop_event = threading.Event()

    global _stop_event
    _stop_event = stop_event

    def _loop() -> None:
        while not stop_event.wait(interval):
            idle = stalled_for()
            if idle > stall_sec:
                _emit_diagnostics(
                    f"EYAS-WATCHDOG: no process activity for {idle:.0f}s "
                    f"(threshold {stall_sec:.0f}s, last activity: {_last_source}) "
                    f"— judged a hang, self-terminating to free the GPU "
                    f"(diagnostics written to {_diag_dir()}/eyas-watchdog-{os.getpid()}.log)\n"
                )
                do_kill(99)
                break

    threading.Thread(target=_loop, name="eyas-watchdog", daemon=True).start()


def stop_watchdog() -> None:
    """Stop the watchdog thread and reset the started flag (for tests and process
    teardown)."""
    global _started
    if _stop_event is not None:
        _stop_event.set()
    _started = False
