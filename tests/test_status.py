"""status.json — schema v1 contract + atomic single-writer updates.

The run dir's machine-readable status file is the observation contract every
future consumer (``eyas status``, a dashboard) reads. Propositions under test:

1. writer roundtrip + atomicity (temp file **in the same dir** written
   completely, then one ``os.replace`` onto the final name — a reader never
   sees a half-written file) + ``history`` bounded to the last 50 entries
2. reader: absent → ``None``; valid → dict; corrupt → ``json.JSONDecodeError``
   propagates (a corrupted status is loud, never silently re-initialised)
3. the budget water level is a live section: a refresh seeded with a frozen
   allocator state (the checkpoint's ``budget_state``) shows it, and a later
   refresh without one carries the last water level forward
4. runtime integration: run-start ``starting`` → per-round echoes → final
   ``done`` (schema / round / cost / cache asserted against the run result), and
   a resumed run inherits the prior status's cumulative counters instead of
   resetting them
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eyas.backends.base import ChatRequest, ChatResponse, system_message, user_message
from eyas.core.objective import Objective
from eyas.engine import runtime as runtime_mod
from eyas.engine import status as status_mod
from eyas.engine.budget import FlatBudget
from eyas.engine.domain import Cost
from eyas.engine.runtime import run_experiment
from eyas.logging import _reset
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant

write_status = status_mod.write_status
read_status = status_mod.read_status

# ---------------------------------------------------------------------------
# writer / reader unit contract
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run": {"name": "toy_run", "strategy": "cot"},
        "phase": "optimizing",
        "round": 3,
        "budget": {"name": "flat"},
        "cost": {"requests": 4},
        "cache": {"hits": 1, "misses": 3},
        "rejections": {},
        "history": [{"round": 0, "phase": "optimizing", "elapsed_sec": 0.1}],
    }
    write_status(tmp_path, payload)
    assert (tmp_path / "status.json").is_file()
    assert not (tmp_path / "status.json.tmp").exists()  # the temp never lingers
    stored = read_status(tmp_path)
    assert stored is not None
    # roundtrip is exact apart from the writer's own observation stamp
    assert stored == {**payload, "updated_at": stored["updated_at"]}
    assert stored["updated_at"]


def test_write_status_is_atomic_tmp_then_replace(tmp_path: Path, monkeypatch) -> None:
    # Atomicity proof: the whole payload lands in a temp file
    # in the SAME directory first, and the only move is a single os.replace onto
    # the final name — so a reader sees either the old or the new file, never a
    # half-written one.
    observed: list[dict[str, Any]] = []
    real_replace = os.replace

    def spy(src, dst):  # noqa: ANN001 - mirrors os.replace
        src_p, dst_p = Path(src), Path(dst)
        observed.append(
            {
                "src": src_p,
                "dst": dst_p,
                "src_exists": src_p.exists(),
                "src_json": json.loads(src_p.read_text(encoding="utf-8")),
                "dst_exists_before": dst_p.exists(),
            }
        )
        real_replace(src, dst)

    monkeypatch.setattr(status_mod.os, "replace", spy)
    payload = {"schema_version": 1, "phase": "done", "round": 1}
    write_status(tmp_path, payload)

    assert len(observed) == 1  # exactly one move, no partial renames
    rec = observed[0]
    assert rec["src"].name == "status.json.tmp"
    assert rec["dst"].name == "status.json"
    assert rec["src"].parent == tmp_path and rec["dst"].parent == tmp_path  # same dir
    assert rec["src_exists"]  # tmp fully written before the move
    assert rec["src_json"] == {**payload, "updated_at": rec["src_json"]["updated_at"]}
    assert not rec["dst_exists_before"]  # the final name is not written in place


def test_write_status_bounds_history_to_50(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "history": [
            {"round": i, "phase": "optimizing", "elapsed_sec": float(i)} for i in range(60)
        ],
    }
    write_status(tmp_path, payload)
    stored = read_status(tmp_path)
    assert stored is not None
    assert len(stored["history"]) == 50  # bounded to the last 50 entries
    assert [h["round"] for h in stored["history"]] == list(range(10, 60))
    assert len(payload["history"]) == 60  # the caller's payload is never mutated


def test_merge_status_carries_forward_and_keeps_round_monotone() -> None:
    prior = status_mod.merge_status(
        None,
        phase="optimizing",
        round=4,
        run={"name": "r"},
        cost={"requests": 40, "wall_seconds": 8.0},
        cache={"hits": 1, "misses": 2},
        rejections={"images per request 2 > limit 1": 3},
    )
    # a refresh without a field keeps the prior value (nothing measured yet)
    held = status_mod.merge_status(prior, phase="optimizing", round=5)
    assert held["cost"]["requests"] == 40
    assert held["rejections"] == {"images per request 2 > limit 1": 3}
    assert held["run"] == {"name": "r"}
    assert [h["round"] for h in held["history"]] == [4, 5]
    assert held["history"][0]["elapsed_sec"] == 0.0  # the builder keeps no clock of its own
    timed = status_mod.merge_status(prior, phase="optimizing", round=5, elapsed_sec=1.25)
    assert timed["history"][-1]["elapsed_sec"] == 1.25  # the writer supplies the measurement
    # a round never moves backwards within a status lineage (resume from an
    # older checkpoint must not re-report a lower round)
    backwards = status_mod.merge_status(prior, phase="done", round=2, cost={"requests": 1})
    assert backwards["round"] == 4
    assert backwards["cost"]["requests"] == 1  # cost is a fresh reading, not a max
    # an unknown key from a future schema version is carried forward untouched
    future = {**prior, "next_phase_preview": {"k": 1}}
    assert status_mod.merge_status(future, phase="done", round=5)["next_phase_preview"] == {"k": 1}


def test_budget_section_refresh_keeps_water_level() -> None:
    # The budget water level is a live section: a refresh seeded with a frozen
    # allocator state (a checkpoint's budget_state) shows it, and a refresh
    # without one keeps the last water level instead of reverting to fresh.
    start = status_mod.merge_status(
        None,
        phase="starting",
        round=0,
        budget=status_mod.budget_section(FlatBudget, None),
    )
    assert start["budget"] == {"name": "flat", "state": {}}
    frozen = {"round": 2, "counts": [3, 4]}  # a checkpoint's budget_state shape
    echo = status_mod.merge_status(
        start,
        phase="optimizing",
        round=1,
        budget=status_mod.budget_section(FlatBudget, frozen),
    )
    assert echo["budget"]["state"] == frozen
    held = status_mod.merge_status(echo, phase="done", round=1)
    assert held["budget"]["state"] == frozen  # carried forward, never re-zeroed


def test_read_status_missing_returns_none(tmp_path: Path) -> None:
    assert read_status(tmp_path) is None
    assert read_status(tmp_path / "does-not-exist") is None


def test_read_status_corrupt_propagates(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_status(tmp_path)


def test_status_module_is_pure() -> None:
    # Single-writer discipline: the run loop is the only writer; the status
    # layer itself owns no threads and no module-level mutable state.
    assert "threading" not in dir(status_mod)
    assert not [
        n
        for n, v in vars(status_mod).items()
        if not n.startswith("_") and isinstance(v, (dict, list, set))
    ]


def test_cost_keys_mirror_cost_to_json() -> None:
    # The comment on _COST_KEYS promises it mirrors Cost.to_json's names
    # exactly (the status cost section is the same bill, not a second
    # accounting) — pin the mirror so a Cost field cannot drift unnoticed.
    assert status_mod._COST_KEYS == tuple(Cost().to_json())


def test_advance_stamps_elapsed_on_the_appended_history_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The elapsed interval is measured on the run's monotonic clock and must
    # land on the entry advance() APPENDS — pin the exact value with a frozen
    # clock and the prior entry's value unchanged, read back from the status
    # document a reader would see (a ">= 0" assertion cannot catch a
    # wrong-target stamp).
    ticks = iter([100.0, 100.5])  # construction tick, then the advance tick
    monkeypatch.setattr(runtime_mod, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = runtime_mod._StatusTracker(
        _cfg("elapsed_stamp"), run_dir, runtime_mod._RunParts(), None, None
    )
    seeded = read_status(run_dir)
    assert seeded is not None
    assert [h["round"] for h in seeded["history"]] == [0]  # the run-start entry to keep

    tracker.advance(status_mod.merge_status(seeded, phase="optimizing", round=1))

    status = read_status(run_dir)
    assert status is not None
    assert [h["phase"] for h in status["history"]] == ["starting", "optimizing"]
    assert status["history"][-1]["elapsed_sec"] == 0.5  # 100.5 - 100.0, on the NEW entry
    assert status["history"][0]["elapsed_sec"] == seeded["history"][0]["elapsed_sec"]


# ---------------------------------------------------------------------------
# runtime integration (mock end-to-end, zero GPU)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_logging():
    """run_experiment wires framework logging into module-global state; reset around each test."""
    _reset()
    yield
    _reset()


@register_task("toy_status")
class ToyStatus(Task):
    name = "toy_status"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"}),
            Sample(id="q2", inputs={"question": "q2"}, references={"label": "dog"}),
        ]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _cfg(name: str = "toy_status_run") -> dict[str, Any]:
    return {
        "run": {"name": name, "seed": 0, "max_steps": 8},
        "task": {"name": "toy_status"},
        "protocol": {"name": "identity"},
        "backend": {"name": "mock", "params": {"responses": ["cat", "dog"], "cycle": True}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {
            "name": "hill_climb",
            "params": {"rounds": 2, "keep_size": 2, "per_parent": 2},
        },
    }


def test_run_start_writes_initial_status(tmp_path: Path, monkeypatch) -> None:
    # Run-start payload contract: phase "starting", round 0, zeroed counters —
    # the file exists before assembly, so a slow assembly is never mistaken for
    # a run dir that never started.
    captured: list[dict[str, Any]] = []
    real = status_mod.write_status

    def spy(run_dir: Path, payload: dict[str, Any]) -> None:
        captured.append(dict(payload))
        real(run_dir, payload)

    monkeypatch.setattr(status_mod, "write_status", spy)
    run_experiment(_cfg(), output_dir=tmp_path)

    assert captured[0]["schema_version"] == 1
    assert captured[0]["phase"] == "starting"
    assert captured[0]["round"] == 0
    assert captured[0]["cost"]["requests"] == 0
    assert captured[0]["cache"] == {"hits": 0, "misses": 0}
    assert captured[0]["rejections"] == {}
    assert captured[0]["run"]["strategy"] == "hill_climb"
    assert captured[0]["run"]["task"] == "toy_status"


def test_run_finalizes_status(tmp_path: Path) -> None:
    result = run_experiment(_cfg(), output_dir=tmp_path)
    status = read_status(tmp_path)
    assert status is not None
    assert status["schema_version"] == 1
    assert status["phase"] == "done"
    assert status["round"] == len(result.history)  # final round index
    assert status["cost"]["requests"] == result.total_cost.requests
    assert status["cost"]["requests"] > 0
    assert status["cache"]["hits"] >= 0
    assert isinstance(status["rejections"], dict)
    assert status["updated_at"]
    # one history entry per refresh: starting + one per completed round + done
    assert all(e["elapsed_sec"] >= 0 for e in status["history"])
    assert [h["phase"] for h in status["history"]] == (
        ["starting"] + ["optimizing"] * len(result.history) + ["done"]
    )


def test_round_echo_refreshes_budget_water_level(tmp_path: Path, monkeypatch) -> None:
    # The budget water level must move with the checkpoint: every round echo
    # re-seeds the budget section from the checkpoint's frozen budget_state,
    # so status.json never shows the run-start level while rounds burn budget.
    cfg = _cfg("budget_water_level")
    cfg["budget"] = {"name": "bandit", "params": {"eval_rounds": 8, "samples_per_eval": 2}}
    captured: list[dict[str, Any]] = []
    real = status_mod.write_status

    def spy(run_dir: Path, payload: dict[str, Any]) -> None:
        captured.append(json.loads(json.dumps(payload)))  # the frozen-on-disk shape
        real(run_dir, payload)

    monkeypatch.setattr(status_mod, "write_status", spy)
    run_experiment(cfg, output_dir=tmp_path)

    start, *echoes = captured
    assert start["budget"] == {"name": "bandit", "state": {}}  # as-fresh-as-run-start
    round_echoes = [p for p in echoes if p["phase"] == "optimizing"]
    assert len(round_echoes) >= 2
    # the ucb allocator keeps a non-empty state once rounds have burned arms
    assert all(p["budget"]["state"] for p in round_echoes)
    # the final status mirrors the last checkpoint's frozen budget_state exactly
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    status = read_status(tmp_path)
    assert status is not None
    assert status["budget"]["state"] == checkpoint["budget_state"]


def test_resumed_run_does_not_reset_status(tmp_path: Path, monkeypatch) -> None:
    # End to end: interrupt a run after its first round (the
    # checkpoint + status.json of the interrupted attempt stay in the run dir),
    # then resume it. The resumed run's observation must continue from where the
    # interrupted one stopped instead of restarting from zero.
    cfg = _cfg()
    calls: list[int] = []
    real = status_mod.write_status

    def interrupt_after_first_round(run_dir: Path, payload: dict[str, Any]) -> None:
        real(run_dir, payload)
        if payload["phase"] == "optimizing":
            calls.append(int(payload["round"]))
            if len(calls) == 1:
                raise KeyboardInterrupt  # the round echo already persisted checkpoint.json

    monkeypatch.setattr(status_mod, "write_status", interrupt_after_first_round)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(cfg, output_dir=tmp_path)
    assert calls == [1]
    interrupted = read_status(tmp_path)
    assert interrupted is not None and interrupted["round"] == 1
    monkeypatch.undo()  # the resumed run writes status normally again

    full = run_experiment(_cfg(), output_dir=tmp_path.parent / "full")  # uninterrupted reference
    full_round = read_status(tmp_path.parent / "full")["round"]
    run_experiment(cfg, resume=tmp_path)

    status = read_status(tmp_path)
    assert status is not None
    assert status["phase"] == "done"
    assert status["round"] >= interrupted["round"]  # never reset below the checkpoint's round
    assert status["round"] == full_round  # same observation as the uninterrupted run
    assert status["cost"]["requests"] >= interrupted["cost"]["requests"]
    assert status["cost"]["requests"] == full.total_cost.requests  # same bill as a clean run
    assert status["history"][0]["round"] == 0  # the pre-resume history entry survived


def test_sweep_writes_status_per_run(tmp_path: Path) -> None:
    from eyas.engine.runtime import run_sweep

    dirs = run_sweep([_cfg("sweep_sa"), _cfg("sweep_sb")], tmp_path / "root")
    phases = [read_status(d) for d in dirs]
    assert [s["phase"] if s else None for s in phases] == ["done", "done"]
