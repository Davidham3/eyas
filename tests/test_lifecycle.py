"""Run lifecycle invariants (fast tier, zero GPU).

Two propositions pinned here:

1. **Started-run close** — every run that wrote its run-start status ends with
   either ``result.json`` or ``status.json`` at phase ``failed``. A crashed run
   whose status still reads ``starting``/``optimizing`` is indistinguishable
   from a live one; the failed write closes that window (assembly failures,
   trace-recorder construction, mid-loop failures, and finalization writes).
2. **Thread / close hygiene** — a completed run leaves no new non-daemon
   thread behind and closes its backend exactly once.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    system_message,
    user_message,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.core.objective import Objective
from eyas.engine.runtime import run_experiment
from eyas.engine.status import read_status
from eyas.logging import _reset
from eyas.registry import register_backend, register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant


@pytest.fixture(autouse=True)
def _isolated_logging():
    """run_experiment wires framework logging into module-global state; reset around each test."""
    _reset()
    yield
    _reset()


@register_task("toy_lc")
class ToyLC(Task):
    name = "toy_lc"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"})]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _cfg(**run: object) -> dict[str, object]:
    return {
        "run": {"name": "toy_lc_run", "seed": 0, **run},
        "task": {"name": "toy_lc"},
        "protocol": {"name": "identity"},
        "backend": {"name": "mock", "params": {"responses": ["cat"], "cycle": True}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {"name": "hill_climb", "params": {"rounds": 2}},
    }


def test_assembly_failure_marks_status_failed(tmp_path):
    # An unresolvable strategy name blows up inside the assembly window (after
    # the run-start status write, before any round). The run dir must say so:
    # phase "failed" as the last word, and no result.json pretending otherwise.
    cfg = _cfg()
    cfg["strategy"] = {"name": "does_not_exist"}
    with pytest.raises(Exception, match="does_not_exist"):
        run_experiment(cfg, output_dir=tmp_path)
    status = read_status(tmp_path)
    assert status is not None
    assert status["phase"] == "failed"
    assert status["history"][-1]["phase"] == "failed"
    # the invariant's other half: failed and result.json never coexist
    assert not (tmp_path / "result.json").exists()


def test_mid_loop_failure_marks_status_failed(tmp_path):
    # Failure after the loop started: round 0 checkpoints (status "optimizing",
    # round 1), then the second backend batch explodes mid-round-1. The status
    # must close at "failed" without moving the round cursor backwards.
    cfg = _cfg()
    cfg["backend"] = {
        "name": "exploding_mock",
        "params": {"responses": ["cat"], "cycle": True, "ok_batches": 1},
    }
    with pytest.raises(RuntimeError, match="injected mid-loop failure"):
        run_experiment(cfg, output_dir=tmp_path)
    status = read_status(tmp_path)
    assert status is not None
    assert status["phase"] == "failed"
    assert status["round"] == 1  # the persisted round-0 checkpoint stands
    assert status["history"][-1]["phase"] == "failed"
    assert not (tmp_path / "result.json").exists()


def test_trace_recorder_failure_marks_status_failed(tmp_path):
    # TraceRecorder construction does eager filesystem IO.
    # Occupying the trace path with a directory pins which call bites: the
    # parent mkdir passes (parent exists, exist_ok=True), while open("a") on
    # the directory raises IsADirectoryError. Construction sits after the
    # run-start status write, so its failure must land in the failed-phase
    # window instead of leaving the run stuck at "starting".
    (tmp_path / "trace.jsonl").mkdir()
    cfg = _cfg(trace=True)
    with pytest.raises(IsADirectoryError):
        run_experiment(cfg, output_dir=tmp_path)
    status = read_status(tmp_path)
    assert status is not None
    assert status["phase"] == "failed"
    assert status["history"][-1]["phase"] == "failed"
    assert not (tmp_path / "result.json").exists()


def test_finalization_failure_marks_status_failed(tmp_path):
    # The loop runs to completion (every round checkpointed), but the write
    # that finalizes the run fails: a directory occupies the result.json path,
    # so export's write_text raises IsADirectoryError. Finalization sits
    # inside the failed-phase window — the started-run invariant also covers
    # the post-loop writes, so the run must close at "failed", never stay at
    # "optimizing" with neither result.json nor a terminal failure.
    (tmp_path / "result.json").mkdir()
    with pytest.raises(IsADirectoryError):
        run_experiment(_cfg(), output_dir=tmp_path)
    status = read_status(tmp_path)
    assert status is not None
    assert status["phase"] == "failed"
    assert status["history"][-1]["phase"] == "failed"
    # the loop itself finished before the finalization bit: rounds were echoed
    assert status["round"] >= 1
    # export is the first finalization statement, so nothing after it was written
    assert not (tmp_path / "run_config.json").exists()


def test_happy_path_never_writes_failed_phase(tmp_path):
    # The three happy-path phases stay untouched by the failure write.
    run_experiment(_cfg(), output_dir=tmp_path)
    status = read_status(tmp_path)
    assert status is not None
    assert status["phase"] == "done"
    phases = [entry["phase"] for entry in status["history"]]
    assert phases[0] == "starting" and phases[-1] == "done"
    assert "failed" not in phases
    assert (tmp_path / "result.json").exists()


@register_backend("exploding_mock")
class ExplodingMockBackend(ScriptedMockBackend):
    """Mid-loop failure injector: answers ``ok_batches`` agenerate batches, then raises."""

    def __init__(self, ok_batches: int = 1, **params: object):
        super().__init__(**params)  # type: ignore[arg-type]
        self._ok_batches = ok_batches

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        if len(self.call_sizes) >= self._ok_batches:
            raise RuntimeError("injected mid-loop failure")
        return await super().agenerate(requests, sampling)


CLOSE_CALLS: list[str] = []


@register_backend("lc_close_spy")
class LifecycleCloseSpyBackend(ScriptedMockBackend):
    """close-count spy local to this file (test_runtime keeps its own copy)."""

    def close(self) -> None:
        CLOSE_CALLS.append("close")
        super().close()


def _close_spy_cfg() -> dict[str, object]:
    cfg = _cfg()
    cfg["backend"] = {"name": "lc_close_spy", "params": {"responses": ["cat"], "cycle": True}}
    return cfg


def test_completed_run_leaks_no_non_daemon_thread(tmp_path):
    # (a) A run's helper threads (logging writer, watchdog, any engine-side
    # helper) are daemons by contract: a completed run must not leave a new
    # non-daemon thread that would keep the process (or a pytest worker) alive
    # after main. Snapshot the live non-daemon set first so pre-existing
    # threads — MainThread, pytest internals — never count as leaks.
    del CLOSE_CALLS[:]
    before = {t.name for t in threading.enumerate() if not t.daemon}
    run_experiment(_close_spy_cfg(), output_dir=tmp_path)
    leaked = sorted(
        t.name
        for t in threading.enumerate()
        if not t.daemon and t.name != "MainThread" and t.name not in before
    )
    assert leaked == []


def test_completed_run_closes_backend_exactly_once(tmp_path):
    # (b) The close chain (runtime → Traced/Normalizing wrappers → engine)
    # runs exactly once per run: zero leaves the next sweep entry on an
    # unreleased engine, twice means a double-close on a dead engine.
    del CLOSE_CALLS[:]
    run_experiment(_close_spy_cfg(), output_dir=tmp_path)
    assert CLOSE_CALLS == ["close"]


def test_completed_run_done_phase_with_daemon_writer(tmp_path):
    # (c) The happy path closes at phase "done", and the background logging
    # writer ("eyas-logging") is alive as a daemon — the run never blocks on,
    # or is blocked by, its own write path.
    run_experiment(_cfg(), output_dir=tmp_path)
    status = read_status(tmp_path)
    assert status is not None
    assert status["phase"] == "done"
    writer = next(t for t in threading.enumerate() if t.name == "eyas-logging")
    assert writer.daemon
