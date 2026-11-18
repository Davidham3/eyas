"""Run-dir durability contract: every durable run-dir JSON (status / checkpoint / result)
is written atomically, no temp residue survives a run, and a corrupt checkpoint fails with a
clear message instead of a raw JSONDecodeError."""

from __future__ import annotations

import json

import pytest
from test_final_eval import _cfg

from eyas.engine.runtime import run_experiment
from eyas.engine.status import write_json_atomic
from eyas.logging import _reset


@pytest.fixture(autouse=True)
def _isolated_logging():
    """run_experiment wires framework logging into module-global state; reset around each test."""
    _reset()
    yield
    _reset()


def test_write_json_atomic_replaces_and_leaves_no_tmp(tmp_path):
    doc = tmp_path / "doc.json"
    write_json_atomic(doc, json.dumps({"a": 1}))
    write_json_atomic(doc, json.dumps({"a": 2}))
    assert json.loads(doc.read_text(encoding="utf-8")) == {"a": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_run_dir_durable_json_written_atomically(tmp_path):
    run_experiment(_cfg(), output_dir=tmp_path)
    run_dir = next(p.parent for p in tmp_path.rglob("result.json"))
    # the three durable documents all parse, and the atomic pattern never
    # leaves its temp file behind
    for name in ("status.json", "checkpoint.json", "result.json"):
        assert (run_dir / name).is_file(), name
        json.loads((run_dir / name).read_text(encoding="utf-8"))
    assert list(run_dir.glob("*.tmp")) == []


def test_corrupt_checkpoint_fails_clearly(tmp_path):
    run_experiment(_cfg(), output_dir=tmp_path)
    run_dir = next(p.parent for p in tmp_path.rglob("checkpoint.json"))
    # simulate a torn write (external damage)
    (run_dir / "checkpoint.json").write_text('{"round": 1, "history": [trunc', encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt"):
        run_experiment(_cfg(), resume=run_dir)


def test_result_json_history_carries_per_round_invoices(tmp_path):
    """Per-round cost + wall seconds ride into result.json: the
    per-round spend curve — which round burned what, judge split included, and
    how long it took on the loop's clock — must be readable from the final
    artifact alone, not stitched from checkpoint/status side files."""
    run_experiment(_cfg(), output_dir=tmp_path)
    run_dir = next(p.parent for p in tmp_path.rglob("result.json"))
    exported = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert exported["history"], "toy cot run must complete at least one round"
    for entry in exported["history"]:
        # full Cost shape (same seven fields as total_cost, not a stub)
        assert set(entry["cost"]) == set(exported["total_cost"])
        assert isinstance(entry["elapsed_sec"], float) and entry["elapsed_sec"] >= 0.0
        # the round's request count is the same bill from both vantage points
        assert entry["n_requests"] == entry["cost"]["requests"]
    # ledger consistency: per-round bills sum to the loop total (final_eval, when
    # configured, is accounted separately and lands in neither side)
    assert (
        sum(e["cost"]["requests"] for e in exported["history"])
        == exported["total_cost"]["requests"]
    )


def test_result_json_history_best_so_far_is_a_learning_curve(tmp_path):
    """The per-round ``best_so_far`` points must form the best-so-far curve:
    monotone non-decreasing on the primary metric (append-only archive, stable
    ranking rule), and the tail point must coincide with the run's reported
    best (same ledger leader, end of loop)."""
    run_experiment(_cfg(), output_dir=tmp_path)
    run_dir = next(p.parent for p in tmp_path.rglob("result.json"))
    exported = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    curve = [h["best_so_far"] for h in exported["history"]]
    assert all(c is not None for c in curve)
    acc = [c["metrics"]["accuracy"] for c in curve]
    assert acc == sorted(acc), f"best-so-far curve dipped: {acc}"
    assert curve[-1]["prompt_hash"] == exported["best"]["prompt_hash"]
    assert curve[-1]["metrics"] == exported["best"]["metrics"]
