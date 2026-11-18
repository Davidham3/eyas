"""Trace recorder contract tests (zero GPU): event stream structure, image
redaction, error path, dynamic labels, two slots sharing a recorder, append
continuation."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    SamplingConfig,
    TextPart,
    VideoPart,
    system_message,
    user_message,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.backends.normalize import NormalizingBackend
from eyas.runtime.trace import (
    TracedBackend,
    TraceRecorder,
    _redact_part,
    bucket_batch,
    trace_context,
)


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_traced(tmp_path, responses, trace_slot="candidate", labels=None):
    recorder = TraceRecorder(tmp_path / "trace.jsonl", labels=labels)
    backend = TracedBackend(
        ScriptedMockBackend(responses=responses), recorder, trace_slot=trace_slot
    )
    return recorder, backend


class _SlotBearer(ScriptedMockBackend):
    def __init__(self, responses, slot_obj):
        super().__init__(responses=responses)
        self.slot = slot_obj


def test_wrapper_chain_forwards_slot(tmp_path):
    """Engine-slot pass-through anchor: the optimizer's
    getattr(.slot) through the outer wrapper must reach the real slot — a
    wrapper that does not forward makes engine release a silent no-op, so the
    next engine start collides with an unreleased one."""
    sentinel = object()
    raw = _SlotBearer(["x"], sentinel)
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    traced = TracedBackend(raw, recorder, trace_slot="candidate")
    normalized = NormalizingBackend(traced)

    assert traced.trace_slot == "candidate"
    assert traced.slot is sentinel
    assert normalized.slot is sentinel  # the outermost layer is what the optimizer sees


def test_wrapper_chain_slot_absent_returns_none(tmp_path):
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    normalized = NormalizingBackend(TracedBackend(ScriptedMockBackend(responses=["x"]), recorder))
    assert normalized.slot is None  # slot-less backends like api/mock: getattr default None


def test_records_batch_and_call_events(tmp_path):
    recorder, backend = make_traced(tmp_path, ["A", "B"], labels={"optimizer": "toy"})
    requests = [
        ChatRequest(messages=(system_message("p"), user_message("q1")), request_id="r1"),
        ChatRequest(messages=(system_message("p"), user_message("q2")), request_id="r2"),
    ]
    responses = backend.generate(requests, SamplingConfig(temperature=0.3))
    assert [r.text for r in responses] == ["A", "B"]

    recorder.flush()  # writer-thread barrier before reading the file
    events = read_events(tmp_path / "trace.jsonl")
    assert [e["kind"] for e in events] == ["llm_batch", "llm_call", "llm_call"]
    batch = events[0]
    assert batch["n_requests"] == 2
    assert batch["slot"] == "candidate"
    assert batch["optimizer"] == "toy"  # static labels at the event top level
    assert batch["sampling"]["temperature"] == 0.3
    assert isinstance(batch["latency_ms"], float)
    assert batch["context"] == {}  # no dynamic labels
    assert all(e["batch_id"] == batch["batch_id"] for e in events[1:])

    call = events[1]
    assert call["request_id"] == "r1" and call["index"] == 0
    assert call["messages"][0] == {"role": "system", "content": [{"type": "text", "text": "p"}]}
    assert call["response_text"] == "A"
    # seq strictly increasing and unique
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_llm_call_records_finish_reason(tmp_path):
    """finish_reason goes into llm_call events — the only observability entry for truncation (length)."""
    recorder, backend = make_traced(tmp_path, [ChatResponse(text="A", finish_reason="length")])
    backend.generate(
        [ChatRequest(messages=(system_message("p"), user_message("q")), request_id="r1")],
        SamplingConfig(),
    )
    recorder.flush()
    call = [e for e in read_events(tmp_path / "trace.jsonl") if e["kind"] == "llm_call"][0]
    assert call["finish_reason"] == "length"


def test_image_redaction_no_base64(tmp_path):
    data_url = "data:image/png;base64,QUFBQUFB"
    recorder, backend = make_traced(tmp_path, ["ok"])
    request = ChatRequest(
        messages=(
            user_message(
                ImagePart(image="/data/images/s1.jpg"),
                ImagePart(image=data_url),
                TextPart("classify"),
            ),
        ),
        request_id="r",
    )
    backend.generate([request], SamplingConfig())

    recorder.flush()
    raw = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    assert "QUFBQUFB" not in raw  # base64 never persisted
    call = json.loads(raw.splitlines()[1])
    parts = call["messages"][0]["content"]
    assert parts[0] == {"type": "image", "path": "/data/images/s1.jpg"}
    assert parts[1]["type"] == "image"
    assert len(parts[1]["inline_sha16"]) == 16
    assert parts[1]["bytes"] == len(data_url)
    assert parts[2] == {"type": "text", "text": "classify"}


def test_redact_part_video():
    assert _redact_part(VideoPart(video="a.mp4")) == {"type": "video", "path": "a.mp4"}


def test_error_records_batch_and_reraises(tmp_path):
    def bad_responder(request, sampling, index):
        raise RuntimeError("boom")

    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    backend = TracedBackend(ScriptedMockBackend(responder=bad_responder), recorder)
    with pytest.raises(RuntimeError, match="boom"):
        backend.generate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())

    recorder.flush()
    events = read_events(tmp_path / "trace.jsonl")
    assert len(events) == 1
    assert events[0]["kind"] == "llm_batch"
    assert "boom" in events[0]["error"]
    assert events[0]["n_requests"] == 1


def test_trace_context_merges_nested_labels():
    with trace_context(phase="a"):
        with trace_context(round="3"):
            assert_trace_labels({"phase": "a", "round": "3"})
        # restored after exiting the inner context
        assert_trace_labels({"phase": "a"})


def assert_trace_labels(expected):
    from eyas.runtime.trace import _trace_labels

    assert _trace_labels.get() == expected


def test_two_slots_share_recorder(tmp_path):
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    candidate = TracedBackend(
        ScriptedMockBackend(responses=["x"]), recorder, trace_slot="candidate"
    )
    judge = TracedBackend(ScriptedMockBackend(responses=["y"]), recorder, trace_slot="judge")
    candidate.generate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    judge.generate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())

    recorder.flush()
    events = read_events(tmp_path / "trace.jsonl")
    batches = [e for e in events if e["kind"] == "llm_batch"]
    assert [b["slot"] for b in batches] == ["candidate", "judge"]
    assert len({b["batch_id"] for b in batches}) == 2  # batch ids globally unique


def test_ledger_judge_phase_on_candidate_slot(tmp_path):
    """When judge calls borrow the candidate slot (no separate judge backend
    in config) they bucket under phase="judge" and must not count into opt."""
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    recorder.record("llm_batch", batch_id="b1", slot="candidate", n_requests=3)
    with trace_context(phase="judge"):
        recorder.record("llm_batch", batch_id="b2", slot="candidate", n_requests=5)
    with trace_context(phase="evaluate", split="train"):
        recorder.record("llm_batch", batch_id="b3", slot="candidate", n_requests=2)
    with trace_context(phase="judge"):
        recorder.record("llm_batch", batch_id="b4", slot="judge", n_requests=7)
    recorder.close()

    totals = recorder.totals()
    assert totals["opt_calls"] == 3
    assert totals["judge_calls"] == 12  # union of phase and slot: candidate 5 + judge 7
    assert totals["eval_calls"] == 2
    assert bucket_batch("candidate", None) == "opt"
    assert bucket_batch(None, "evaluate") == "eval"


def test_ledger_usage_tokens(tmp_path):
    """Usage ledger's three columns: prompt/completion counted separately,
    missing total reconstructed by summation, events without usage excluded
    from usage_requests (cache-hit accounting matches billing)."""
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    recorder.record("llm_batch", batch_id="b1", slot="candidate", n_requests=2)
    recorder.record(
        "llm_call",
        batch_id="b1",
        slot="candidate",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    recorder.record("llm_call", batch_id="b1", slot="candidate", usage={"total_tokens": 30})
    recorder.record("llm_call", batch_id="b1", slot="candidate")
    recorder.close()

    totals = recorder.totals()
    assert totals["prompt_tokens"] == 10
    assert totals["completion_tokens"] == 5
    assert totals["usage_requests"] == 2
    assert totals["total_tokens"] == 45  # 15 + 30 + no-usage events excluded


def test_trace_writes_happen_off_the_caller_thread(tmp_path):
    """Disk IO rides a dedicated writer thread (same isolation logging has):
    a stalled trace file must never freeze the caller / event loop. close()
    is the flush barrier — no event lost, ordering preserved."""
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    real_fh = recorder._fh
    write_threads: list[int] = []

    class _SpyFile:
        def write(self, s: str) -> int:
            write_threads.append(threading.get_ident())
            return real_fh.write(s)

        def flush(self) -> None:
            real_fh.flush()

        def close(self) -> None:
            real_fh.close()

    recorder._fh = _SpyFile()
    caller_thread = threading.get_ident()
    for i in range(3):
        recorder.record("llm_batch", batch_id=f"b{i}", slot="candidate", n_requests=1)
    recorder.close()  # flush barrier: every event must have landed
    assert len(write_threads) == 3
    assert all(t != caller_thread for t in write_threads)
    events = read_events(tmp_path / "trace.jsonl")
    assert [e["seq"] for e in events] == [1, 2, 3]  # ordering preserved


def test_recorder_flush_barrier_and_record_after_close(tmp_path):
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    recorder.record("llm_batch", batch_id="b1", slot="candidate", n_requests=1)
    recorder.flush()  # barrier without closing: events visible to an external reader
    assert [e["batch_id"] for e in read_events(tmp_path / "trace.jsonl")] == ["b1"]
    recorder.close()
    with pytest.raises(ValueError, match="closed"):
        recorder.record("llm_batch", batch_id="b2")


def test_recorder_reopens_in_append_mode(tmp_path):
    """Append mode: reopening the recorder continues the file (seq is
    per-writer, monotonic within each instance)."""
    recorder = TraceRecorder(tmp_path / "run" / "trace.jsonl", labels={"optimizer": "cot"})
    recorder.record("llm_batch", batch_id="b1", slot="candidate", n_requests=2)
    recorder.close()

    recorder2 = TraceRecorder(tmp_path / "run" / "trace.jsonl")
    recorder2.record("llm_batch", batch_id="b-next", slot="candidate", n_requests=0)
    recorder2.close()

    events = read_events(tmp_path / "run" / "trace.jsonl")
    assert [e["batch_id"] for e in events] == ["b1", "b-next"]
    assert events[0]["optimizer"] == "cot" and "optimizer" not in events[1]
