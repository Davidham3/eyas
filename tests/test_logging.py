"""eyas.logging: async isolation contract + setup idempotence, console
rebinding, dual file write, silent when unconfigured.

Async isolation: callers only enqueue into
a bounded queue; a blocking write path (a stalled sink) only freezes the
background writer thread — callers stay unblocked, drops are counted, and a
recovery report follows.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from eyas import logging as eyas_logging
from eyas.logging import _reset, drain, dropped_count, get_logger, setup


@pytest.fixture(autouse=True)
def _clean_logging():
    _reset()
    yield
    _reset()


def test_get_logger_normalizes_prefix():
    assert get_logger("backends.demo").name == "eyas.backends.demo"
    assert get_logger("eyas.backends.demo").name == "eyas.backends.demo"


class _BlockingSink(logging.Handler):
    """A fake sink that blocks inside emit (simulates a wedged log sink)."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[str] = []
        self.gate = threading.Event()
        self.released = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        self.released.set()
        assert self.gate.wait(5), "must be released before the test wraps up"
        self.received.append(record.getMessage())


def test_enqueue_handler_never_blocks_and_counts_drops():
    handler = eyas_logging._EnqueueHandler(maxsize=2)
    record = logging.LogRecord("eyas.t", logging.INFO, __file__, 1, "m", None, None)
    for _ in range(5):
        handler.emit(record)
    assert handler.dropped == 3
    assert handler.q.qsize() == 2
    assert dropped_count() == 0  # the unsetup global counter is unaffected by local instances


def test_blocked_writer_does_not_block_caller(tmp_path):
    log = tmp_path / "framework.log"
    setup(file=log)
    blocker = _BlockingSink()
    with eyas_logging._sink_lock:
        eyas_logging._sinks.append(blocker)

    t0 = time.monotonic()
    get_logger("demo").info("hello")
    assert time.monotonic() - t0 < 1.0, "the caller must not wait for the writer thread"
    assert blocker.released.wait(5)
    assert dropped_count() == 0  # queue not full, the writer thread is just waiting

    blocker.gate.set()
    assert drain(5)
    assert "hello" in log.read_text(encoding="utf-8")
    assert blocker.received == ["hello"]


def test_drops_counted_and_reported_after_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(eyas_logging, "_QUEUE_MAXSIZE", 4)
    log = tmp_path / "framework.log"
    setup(file=log)
    blocker = _BlockingSink()
    with eyas_logging._sink_lock:
        eyas_logging._sinks.append(blocker)

    for i in range(20):
        get_logger("demo").info("line %d", i)
    # writer thread blocked on the first line: queue 4 + 1 in the writer's hand, 15-16 dropped
    assert 10 < dropped_count() < 20

    blocker.gate.set()
    assert drain(5)
    text = log.read_text(encoding="utf-8")
    assert "[logging] writer thread recovered" in text and "dropped" in text
    assert "line 0" in text  # first line persisted via the writer thread


class TestSetup:
    def test_console_rebinds_current_stdout(self, capsys):
        setup()
        get_logger("demo").info("hello %s", "world")
        assert drain(5)
        assert "hello world" in capsys.readouterr().out

    def test_setup_idempotent_no_duplicate_console(self, capsys):
        setup()
        setup()
        get_logger("demo").info("once")
        assert drain(5)
        assert capsys.readouterr().out.count("once") == 1

    def test_file_handler_writes_full_format(self, tmp_path, capsys):
        log = tmp_path / "framework.log"
        setup(file=log)
        get_logger("backends.demo").info("file line")
        assert drain(5)
        text = log.read_text(encoding="utf-8")
        assert "INFO eyas.backends.demo: file line" in text
        # console side is still the bare message line (timestamp/level go to the file only)
        assert capsys.readouterr().out == "file line\n"

    def test_same_file_repeated_setup_keeps_single_handler(self, tmp_path):
        log = tmp_path / "framework.log"
        setup(file=log)
        setup(file=log)
        files = [h for h in eyas_logging._sinks if isinstance(h, logging.FileHandler)]
        assert len(files) == 1

    def test_switch_file_closes_previous(self, tmp_path):
        first, second = tmp_path / "a.log", tmp_path / "b.log"
        setup(file=first)
        setup(file=second)
        get_logger("demo").info("to b")
        assert drain(5)
        assert first.read_text(encoding="utf-8") == ""
        assert "to b" in second.read_text(encoding="utf-8")

    def test_unconfigured_info_is_silent(self, capsys):
        get_logger("quiet").info("dropped")
        assert capsys.readouterr().out == ""


def test_priority_channel_survives_a_full_droppable_queue(tmp_path, monkeypatch):
    """docs/10 §10 two channels: a full droppable queue loses ordinary INFO but
    can never bury a key_event — the writer thread drains the UNBOUNDED priority
    queue first, so the run's evidence trail survives backpressure and lands in
    the framework.log ahead of the recovered backlog."""
    monkeypatch.setattr(eyas_logging, "_QUEUE_MAXSIZE", 4)
    log = tmp_path / "framework.log"
    setup(file=log)
    blocker = _BlockingSink()
    with eyas_logging._sink_lock:
        eyas_logging._sinks.append(blocker)

    for i in range(20):
        get_logger("demo").info("flood %d", i)  # overflow the droppable channel
    dropped_before = dropped_count()
    assert dropped_before > 0  # the droppable channel is provably saturated

    eyas_logging.key_event(get_logger("demo"), "evidence %s", "trail")
    assert dropped_count() == dropped_before  # the priority record was never a drop

    blocker.gate.set()
    assert drain(5)
    text = log.read_text(encoding="utf-8")
    assert "evidence trail" in text  # the evidence SURVIVED the full queue
    # priority drains first: the evidence precedes the backlog (line 0 was already
    # in the writer's hand when the queue saturated)
    assert text.index("evidence trail") < text.index("flood 1")


def test_priority_record_selection_is_priority_first():
    """Handler-level pick contract: emit routing by the eyas_priority flag, and
    pick() fully prefers the priority queue; the droppable queue's overflow
    counter never charges a priority record."""
    handler = eyas_logging._EnqueueHandler(maxsize=1)

    def _rec(msg: str, priority: bool = False) -> logging.LogRecord:
        record = logging.LogRecord("eyas.t", logging.INFO, __file__, 1, msg, None, None)
        if priority:
            record.eyas_priority = True  # what key_event() attaches via extra=
        return record

    handler.emit(_rec("ordinary-a"))  # fills the bounded droppable queue
    handler.emit(_rec("evidence", priority=True))
    handler.emit(_rec("ordinary-b"))  # droppable queue full -> counted drop
    assert handler.dropped == 1

    src, record = handler.pick()
    assert src is handler.priority_q and record.getMessage() == "evidence"
    src, record = handler.pick()
    assert src is handler.q and record.getMessage() == "ordinary-a"


class _OneShotRaisingSink(logging.Handler):
    """Raises on the first emit only — a permanently raising sink would also
    swallow the recovery report itself and re-arm the counter forever."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("sink boom")


def test_sink_exceptions_counted_and_reported_after_recovery(tmp_path):
    # "log loss is never silent": a sink exception is counted and surfaced in
    # the writer thread's recovery report (same mechanism as queue drops)
    log = tmp_path / "framework.log"
    setup(file=log)
    bad = _OneShotRaisingSink()
    with eyas_logging._sink_lock:
        eyas_logging._sinks.append(bad)
    get_logger("demo").info("first")
    assert drain(5)
    assert eyas_logging.sink_error_count() == 1
    text = log.read_text(encoding="utf-8")
    assert "first" in text  # the healthy sinks still received the record
    assert "[logging] writer thread recovered" in text
    assert "sink" in text
