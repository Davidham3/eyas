"""Activity watchdog: no activity beyond the threshold -> kill(99) (a hang
costs more than a crash; idle GPU spinning is intolerable).

Covers the failure shape: after engine IPC death, generate blocks
forever, the scheduling layer stops bumping, and the watchdog triggers
self-termination. Zero GPU, real clock + short thresholds.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from eyas import watchdog
from eyas.backends.scheduling import gather_bounded


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(watchdog, "_started", False)
    monkeypatch.setattr(watchdog, "_last_activity", 0.0)
    monkeypatch.setattr(watchdog, "_last_source", "startup")
    yield
    watchdog.stop_watchdog()


def test_watchdog_kills_on_stall():
    killed: list[int] = []
    watchdog.bump()
    watchdog.start_watchdog(stall_sec=0.1, interval=0.02, kill=killed.append)
    deadline = time.monotonic() + 2.0
    while not killed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert killed == [99]  # the break guarantees a single kill, no repeated triggering


def test_watchdog_spared_by_periodic_activity():
    killed: list[int] = []
    watchdog.bump()
    watchdog.start_watchdog(stall_sec=0.2, interval=0.05, kill=killed.append)
    deadline = time.monotonic() + 0.6
    while time.monotonic() < deadline:
        watchdog.bump("test")
        time.sleep(0.05)
    assert killed == []


def test_watchdog_start_idempotent():
    killed: list[int] = []
    watchdog.bump()
    watchdog.start_watchdog(stall_sec=0.1, interval=0.02, kill=killed.append)
    watchdog.start_watchdog(stall_sec=0.1, interval=0.02, kill=killed.append)
    deadline = time.monotonic() + 2.0
    while not killed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert killed == [99]  # the second start is a no-op, no dual-thread double kill


def test_watchdog_disabled_below_zero():
    watchdog.start_watchdog(stall_sec=0)
    watchdog.start_watchdog(stall_sec=-1)
    assert not watchdog._started


def test_watchdog_kill_survives_wedged_stdout(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """kill must still fire when the stdout pipe is wedged (reader stuck -> pipe
    full -> os.write all failing): a blocking diagnostics write would keep the
    kill from ever arriving."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))

    def _wedged(*args: object, **kwargs: object) -> None:
        raise OSError("pipe full")

    monkeypatch.setattr(watchdog.os, "write", _wedged)
    killed: list[int] = []
    watchdog.bump()
    watchdog.start_watchdog(stall_sec=0.1, interval=0.02, kill=killed.append)
    deadline = time.monotonic() + 2.0
    while not killed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert killed == [99]
    diag = tmp_path / f"eyas-watchdog-{os.getpid()}.log"
    assert "EYAS-WATCHDOG" in diag.read_text(encoding="utf-8")


def test_emit_diagnostics_swallows_all_io_errors(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every diagnostic channel failure must be swallowed: no raise when fcntl
    is unavailable (non-POSIX), /tmp still written."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "fcntl", None)
    watchdog._emit_diagnostics("EYAS-WATCHDOG: test")
    assert "EYAS-WATCHDOG" in (tmp_path / f"eyas-watchdog-{os.getpid()}.log").read_text(
        encoding="utf-8"
    )


def test_emit_diagnostics_includes_thread_stacks(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The diagnostic file carries a full thread-stack snapshot (faulthandler):
    an activity label only gives coarse localization; stack snapshots pinpoint
    the hang to a frame."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))
    watchdog._emit_diagnostics("EYAS-WATCHDOG: test")
    content = (tmp_path / f"eyas-watchdog-{os.getpid()}.log").read_text(encoding="utf-8")
    assert "EYAS-WATCHDOG" in content
    # faulthandler's stable signature (single-threaded runs print "Current thread"; "Thread" only appears with multiple threads)
    assert "most recent call first" in content
    # this test's own frame is visible in the stack (message write and stack dump append to the same file)
    assert "test_emit_diagnostics_includes_thread_stacks" in content


def test_emit_diagnostics_stacks_reach_stderr(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Stack snapshots must reach stderr on a best-effort basis: the private
    /tmp vanishes with the process, and stderr is the only durable channel."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))
    seen: list[bytes] = []
    real_write = os.write

    def _capture(fd: int, data: bytes) -> int:
        seen.append(data)
        return real_write(fd, data)

    monkeypatch.setattr(watchdog.os, "write", _capture)
    watchdog._emit_diagnostics("EYAS-WATCHDOG: test")
    payload = b"".join(seen).decode("utf-8", errors="replace")
    assert "EYAS-WATCHDOG" in payload
    assert "most recent call first" in payload
    assert "test_emit_diagnostics_stacks_reach_stderr" in payload


def test_gather_bounded_bumps_activity():
    watchdog.bump()
    before = watchdog._last_source

    async def _one() -> int:
        return 1

    async def _run() -> list[int]:
        return await gather_bounded([_one, _one], 1)

    assert asyncio.run(_run()) == [1, 1]
    assert watchdog._last_source == "batch.progress [2/2]"
    assert watchdog._last_source != before
    assert watchdog.stalled_for() < 1.0


def test_diag_dir_env_set_after_import_takes_effect(tmp_path, monkeypatch):
    # EYAS_DIAG_DIR is read at CALL time: an os.environ set after the module
    # import must still steer the diagnostics file. The write itself is
    # intercepted (recording the target path) so the test never touches the
    # real default directory.
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))
    seen: list[Path] = []

    def _intercept(self: Path, *args: object, **kwargs: object) -> int:
        seen.append(self)
        raise OSError("intercepted: no actual write in this test")

    monkeypatch.setattr(Path, "write_text", _intercept)
    watchdog._emit_diagnostics("EYAS-WATCHDOG: env-after-import")
    assert seen, "diagnostics must attempt a file write"
    assert seen[0].parent == tmp_path


def test_kill_lands_within_bound_when_diag_file_write_blocks(tmp_path, monkeypatch):
    """EYAS_DIAG_DIR on a stalling network filesystem: the diagnostics file write blocks
    indefinitely. The kill must still land within the hard 2s file bound —
    stderr diagnostics go FIRST (non-blocking), the file write is best-effort
    on a daemon thread with a bounded join, then os._exit(99) unconditionally."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))
    file_started: list[float] = []
    stderr_at: list[float] = []
    real_write = os.write

    def _blocking_write(self: Path, *args: object, **kwargs: object) -> int:
        file_started.append(time.monotonic())
        time.sleep(30)  # stalled shared disk: never returns within the bound
        raise OSError("blocked write abandoned")

    def _capture(fd: int, data: bytes) -> int:
        if fd == 2 and b"EYAS-WATCHDOG" in data:
            stderr_at.append(time.monotonic())
        return real_write(fd, data)

    monkeypatch.setattr(Path, "write_text", _blocking_write)
    monkeypatch.setattr(watchdog.os, "write", _capture)
    killed: list[int] = []
    watchdog.bump()
    t0 = time.monotonic()
    watchdog.start_watchdog(stall_sec=0.1, interval=0.02, kill=killed.append)
    deadline = time.monotonic() + 3.5  # the 2s hard bound + scheduling margin
    while not killed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert killed == [99], "the kill must never wait on the blocked diagnostics file"
    assert time.monotonic() - t0 < 3.5
    assert stderr_at, "stderr diagnostics must land even when the file write blocks"
    assert file_started, "the file write must still be attempted (best-effort)"
    assert stderr_at[0] <= file_started[0], "stderr diagnostics come FIRST, file second"


def test_emit_diagnostics_returns_within_hard_bound(tmp_path, monkeypatch):
    """_emit_diagnostics itself is bounded: a blocking file write is abandoned
    after the hard 2s join timeout, and the message already reached stderr."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))

    def _blocking_write(self: Path, *args: object, **kwargs: object) -> int:
        time.sleep(30)
        raise OSError("blocked write abandoned")

    monkeypatch.setattr(Path, "write_text", _blocking_write)
    seen: list[bytes] = []
    real_write = os.write

    def _capture(fd: int, data: bytes) -> int:
        if fd == 2:
            seen.append(data)
        return real_write(fd, data)

    monkeypatch.setattr(watchdog.os, "write", _capture)
    done = threading.Event()

    def _run() -> None:
        watchdog._emit_diagnostics("EYAS-WATCHDOG: bounded")
        done.set()

    threading.Thread(target=_run, daemon=True).start()
    t0 = time.monotonic()
    assert done.wait(timeout=2.5), "diagnostics must return within the 2s file bound"
    assert time.monotonic() - t0 < 2.5
    assert any(b"EYAS-WATCHDOG: bounded" in chunk for chunk in seen)


def test_kill_lands_when_diag_writer_thread_start_fails(tmp_path, monkeypatch):
    """Thread exhaustion (a plausible shape in exactly the wedged-process
    scenarios the watchdog exists for): the diagnostics writer thread cannot
    start. The failure must be swallowed like any other IO failure — the
    watchdog thread must survive and os._exit(99) must still land."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))
    real_start = threading.Thread.start

    def _start(self: threading.Thread) -> None:
        if self.name == "eyas-watchdog-diag":
            raise RuntimeError("can't start new thread")
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", _start)
    killed: list[int] = []
    watchdog.bump()
    watchdog.start_watchdog(stall_sec=0.1, interval=0.02, kill=killed.append)
    deadline = time.monotonic() + 2.0
    while not killed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert killed == [99], "the kill must never depend on the writer thread starting"


def test_stack_snapshot_reaches_stderr_when_diag_file_write_blocks(tmp_path, monkeypatch):
    """Filesystem stall (the very scenario the watchdog exists for): the diagnostics
    file write is abandoned at the 2s bound. The all-thread stack snapshot —
    the primary hang-localization tool — must STILL reach stderr, the only
    channel that outlives the process; it must not be gated on the file
    landing."""
    monkeypatch.setenv("EYAS_DIAG_DIR", str(tmp_path))

    def _blocking_write(self: Path, *args: object, **kwargs: object) -> int:
        time.sleep(30)  # stalled shared disk: never returns within the bound
        raise OSError("blocked write abandoned")

    monkeypatch.setattr(Path, "write_text", _blocking_write)
    seen: list[bytes] = []
    real_write = os.write

    def _capture(fd: int, data: bytes) -> int:
        if fd == 2:
            seen.append(data)
        return real_write(fd, data)

    monkeypatch.setattr(watchdog.os, "write", _capture)
    watchdog._emit_diagnostics("EYAS-WATCHDOG: stalled-disk")
    payload = b"".join(seen).decode("utf-8", errors="replace")
    assert "EYAS-WATCHDOG: stalled-disk" in payload
    assert "most recent call first" in payload, "stacks must reach stderr even when the file stalls"
    # this test's own frame is visible: the snapshot covers the calling thread
    assert "test_stack_snapshot_reaches_stderr_when_diag_file_write_blocks" in payload
