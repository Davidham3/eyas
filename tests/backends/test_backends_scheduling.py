"""Unit tests for the shared batch scheduling primitives (scheduling.py):
concurrency cap and order preservation, parameter validation, huge-batch
heartbeat (BatchProgress) and zero-progress alert ([batch.stall]).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import pytest

from eyas import watchdog
from eyas.backends import scheduling
from eyas.backends.scheduling import BatchProgress, gather_bounded, run_sync


# ---------------------------------------------------------------------------
# gather_bounded
# ---------------------------------------------------------------------------
async def test_gather_bounded_caps_concurrency_and_preserves_order():
    state = {"active": 0, "peak": 0}

    async def work(i: int) -> int:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01 * ((i % 3) + 1))  # completion order != submission order
        state["active"] -= 1
        return i

    factories: list[Callable[[], Awaitable[int]]] = [lambda i=i: work(i) for i in range(24)]
    out = await gather_bounded(factories, concurrency=4)
    assert out == list(range(24))
    assert state["peak"] <= 4


async def test_gather_bounded_accepts_generator_and_empty_input():
    async def work(i: int) -> int:
        return i * 2

    gen = (lambda i=i: work(i) for i in range(6))
    assert await gather_bounded(gen, concurrency=2) == [0, 2, 4, 6, 8, 10]
    assert await gather_bounded(iter([]), concurrency=2) == []


async def test_gather_bounded_first_error_cancels_siblings_keeps_original_type():
    """[fast-fail, slow-sleeper]: the ORIGINAL exception type propagates (no
    ExceptionGroup) and the sleeper is actually cancelled — siblings must not
    keep burning tokens/GPU after the batch has already failed."""
    sleeper_started = asyncio.Event()
    sleeper_cancelled = asyncio.Event()

    async def fast_fail() -> int:
        await sleeper_started.wait()  # the sleeper must be running before the failure fires
        raise ValueError("first failure")

    async def slow_sleeper() -> int:
        sleeper_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            sleeper_cancelled.set()
            raise
        return -1  # pragma: no cover

    with pytest.raises(ValueError, match="first failure"):
        await gather_bounded([fast_fail, slow_sleeper], concurrency=2)
    assert sleeper_cancelled.is_set()
    # no leaked worker or watcher tasks after the failure unwind
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_gather_bounded_later_sibling_failure_does_not_mask_first():
    async def first_fail() -> int:
        raise ValueError("original")

    async def later_fail() -> int:
        await asyncio.sleep(0.02)
        raise RuntimeError("later sibling")

    with pytest.raises(ValueError, match="original"):
        await gather_bounded([first_fail, later_fail], concurrency=2)
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_gather_bounded_factory_cancellation_propagates_as_cancellation():
    async def cancelled_factory() -> int:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await gather_bounded([cancelled_factory], concurrency=1)
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_gather_bounded_external_cancellation_unwinds_workers_and_watcher():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow() -> int:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return -1  # pragma: no cover

    task = asyncio.create_task(gather_bounded([slow, slow], concurrency=2))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.parametrize("bad", [0, -3])
async def test_gather_bounded_rejects_non_positive_concurrency(bad: int):
    async def work() -> None:
        return None

    with pytest.raises(ValueError, match="concurrency"):
        await gather_bounded([work], bad)


# ---------------------------------------------------------------------------
# run_sync
# ---------------------------------------------------------------------------
def test_run_sync_reuses_shared_loop_across_calls():
    async def loop_id() -> int:
        return id(asyncio.get_running_loop())

    assert run_sync(loop_id()) == run_sync(loop_id())


def test_run_sync_returns_value_and_propagates_exception():
    async def ok() -> str:
        return "v"

    async def boom() -> None:
        raise ValueError("boom")

    assert run_sync(ok()) == "v"
    with pytest.raises(ValueError, match="boom"):
        run_sync(boom())


def test_run_sync_rejects_running_loop():
    async def nested() -> None:
        inner = asyncio.sleep(0)
        try:
            run_sync(inner)
        finally:
            inner.close()  # the rejection raises before any await; do not leak the coroutine

    with pytest.raises(RuntimeError, match="run_sync"):
        run_sync(nested())


def test_run_sync_concurrent_first_callers_share_one_loop(monkeypatch: pytest.MonkeyPatch):
    """Loop CREATION is double-checked under a lock: concurrent first callers
    from multiple threads must all drive the SAME resident loop (the engine's
    handle_loop binds to it once). An unlocked check-then-create lets two
    threads each build a loop — the second caller's engine tasks then bind to a
    loop nobody drives again (the silent-hang shape).

    Harness: new_event_loop is slowed to widen the race window deterministically;
    run_until_complete is serialized because concurrent loop EXECUTION is outside
    run_sync's contract (the framework issues one synchronous call at a time) —
    this test pins the creation invariant only."""
    import threading
    import time

    original_loop = scheduling._shared_loop
    monkeypatch.setattr(scheduling, "_shared_loop", None)
    real_new_loop = asyncio.new_event_loop

    def slow_new_loop() -> asyncio.AbstractEventLoop:
        time.sleep(0.05)  # every racing thread observes None before creation lands
        return real_new_loop()

    monkeypatch.setattr(asyncio, "new_event_loop", slow_new_loop)
    real_run = asyncio.BaseEventLoop.run_until_complete
    run_lock = threading.Lock()

    def serialized_run(self: asyncio.BaseEventLoop, coro):  # type: ignore[no-untyped-def]
        with run_lock:
            return real_run(self, coro)

    monkeypatch.setattr(asyncio.BaseEventLoop, "run_until_complete", serialized_run)

    async def probe() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    barrier = threading.Barrier(4)
    loops: list[asyncio.AbstractEventLoop] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            loops.append(run_sync(probe()))
        except BaseException as exc:  # noqa: BLE001 - surfaced as an assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    try:
        assert not errors, f"worker raised: {errors}"
        assert len(loops) == 4
        assert all(loop is loops[0] for loop in loops), "first callers created separate loops"
        assert loops[0] is scheduling._shared_loop
    finally:
        if loops and loops[0] is not original_loop and not loops[0].is_closed():
            loops[0].close()


# ---------------------------------------------------------------------------
# BatchProgress: intra-batch progress for huge batches (batch silence != hang, a liveness signal)
# ---------------------------------------------------------------------------
def test_batch_progress_reports_at_interval_and_total():
    watchdog.bump("before")
    progress = BatchProgress(600, every=500)
    for _ in range(500):
        progress.step()
    assert watchdog._last_source == "batch.progress [500/600]"
    for _ in range(99):
        progress.step()
    assert (
        watchdog._last_source == "batch.progress [599/600]"
    )  # every step bumps (watchdog activity signal)
    progress.step()
    assert watchdog._last_source == "batch.progress [600/600]"


def test_batch_progress_bumps_below_threshold():
    watchdog.bump("before")
    progress = BatchProgress(32)
    for _ in range(32):
        progress.step()
    assert progress.done == 32
    assert (
        watchdog._last_source == "batch.progress [32/32]"
    )  # small batches bump as usual (the watchdog depends on it)


def test_batch_progress_logs_large_batch_only(eyas_logs: pytest.LogCaptureFixture):
    progress = BatchProgress(600, every=500)
    for _ in range(499):
        progress.step()
    assert "[batch.progress]" not in eyas_logs.text
    progress.step()
    assert "[batch.progress] 500/600" in eyas_logs.text
    for _ in range(99):
        progress.step()
    progress.step()
    assert "[batch.progress] 600/600" in eyas_logs.text


def test_batch_progress_time_floor_and_eta(eyas_logs: pytest.LogCaptureFixture, monkeypatch):
    """Time-based fallback for slow engines: even at a non-count milestone,
    exceeding the threshold since the last report forces a log line, with
    rate/ETA inline."""
    monkeypatch.setattr("eyas.backends.scheduling.BATCH_PROGRESS_MIN_INTERVAL_SEC", 60.0)
    progress = BatchProgress(600, every=500)
    progress.step()
    assert "[batch.progress]" not in eyas_logs.text
    progress._last_emit -= 120.0  # simulate 2min since the last report (slow engine)
    progress.step()
    assert "[batch.progress] 2/600" in eyas_logs.text
    assert "req/s" in eyas_logs.text and "eta ~" in eyas_logs.text


def test_batch_progress_no_log_below_threshold(eyas_logs: pytest.LogCaptureFixture):
    progress = BatchProgress(32)
    for _ in range(32):
        progress.step()
    assert "[batch.progress]" not in eyas_logs.text


# ---------------------------------------------------------------------------
# [batch.stall]: zero-progress guard ("dispatched but zero completions" is [batch.progress]'s blind spot)
# ---------------------------------------------------------------------------
@pytest.fixture
def eyas_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture eyas logs: force propagate (setup() turns it off) + raise capture level to INFO."""
    eyas_root = logging.getLogger("eyas")
    old_propagate = eyas_root.propagate
    eyas_root.propagate = True
    caplog.set_level(logging.INFO, logger="eyas.backends.scheduling")
    yield caplog
    eyas_root.propagate = old_propagate


@pytest.fixture
def fast_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduling, "BATCH_STALL_SEC", 0.05)
    monkeypatch.setattr(scheduling, "BATCH_STALL_REPEAT_SEC", 0.5)
    monkeypatch.setattr(scheduling, "BATCH_WATCH_INTERVAL_SEC", 0.01)


async def test_gather_bounded_stall_alerts_on_first_completion_delay(
    fast_stall: None, eyas_logs: pytest.LogCaptureFixture
):
    async def work(i: int) -> int:
        await asyncio.sleep(
            0.15
        )  # concurrency 1: first segment zero completions -> 0/2 alert, second -> 1/2
        return i

    factories: list[Callable[[], Awaitable[int]]] = [lambda i=i: work(i) for i in range(2)]
    assert await gather_bounded(factories, concurrency=1) == [0, 1]
    assert "batch.stall] 0/2" in eyas_logs.text
    assert "batch.stall] 1/2" in eyas_logs.text


async def test_stall_watch_disabled_by_nonpositive_threshold(
    monkeypatch: pytest.MonkeyPatch, eyas_logs: pytest.LogCaptureFixture
):
    monkeypatch.setattr(scheduling, "BATCH_STALL_SEC", 0.0)
    monkeypatch.setattr(scheduling, "BATCH_WATCH_INTERVAL_SEC", 0.01)

    async def work(i: int) -> int:
        await asyncio.sleep(0.15)
        return i

    factories: list[Callable[[], Awaitable[int]]] = [lambda i=i: work(i) for i in range(2)]
    assert await gather_bounded(factories, concurrency=1) == [0, 1]
    assert "batch.stall" not in eyas_logs.text
