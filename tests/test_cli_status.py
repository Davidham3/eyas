"""``eyas status`` — the read-only reference consumer of the run-dir contract.

The observation contract lives in files (``status.json`` / ``checkpoint.json`` /
``result.json``); this CLI renders them for a human. Propositions under test:

1. complete run dir (status + checkpoint + result) → every fixed section is
   rendered and each printed value is the value the file carries (nothing is
   invented by the renderer — the design's hard rule)
2. mid-run dir (status only) → exit 0, the sections it can fill, no final block,
   no traceback text
3. pre-status dir (no status.json at all) → exit 0, the degraded line, and the
   checkpoint/result facts it can still show
4. a corrupt ``status.json`` stays loud (``read_status`` propagates the
   ``JSONDecodeError``) — the renderer adds no fallback

Fixtures are written through ``write_status`` so the documents have exactly the
shape the run loop's single writer produces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eyas.cli import main
from eyas.engine.status import read_status, write_status

# ---------------------------------------------------------------------------
# fixtures: run-dir documents shaped like the real writer's output
# ---------------------------------------------------------------------------

_COST = {
    "requests": 120,
    "input_tokens": 9000,
    "output_tokens": 1500,
    "wall_seconds": 61.5,
    "judge_requests": 12,
    "judge_input_tokens": 700,
    "judge_output_tokens": 90,
}


def _status_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": "demo_run",
            "strategy": "hill_climb",
            "task": "toy_status",
            "seed": 0,
            "run_dir": "/nonexistent/outputs/demo_run/20261012-010203",
        },
        "phase": "optimizing",
        "round": 2,
        "budget": {"name": "bandit", "state": {"arms": {"generate": 4, "evolve": 2}}},
        "cost": dict(_COST),
        "cache": {"hits": 80, "misses": 40},
        "rejections": {"images per request 2 > limit 1": 3},
        "history": [{"round": 1, "phase": "optimizing", "elapsed_sec": 20.5}],
    }
    doc.update(overrides)
    return doc


def _checkpoint_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "strategy": "hill_climb",
        "round_idx": 3,
        "state": {"kind": "HillClimbState"},
        "archive": [],
        "history": [{"round": 0, "n_candidates": 2, "n_requests": 4}],
        "total_cost": dict(_COST),
        "texts": {"abc123def456": "Classify the animal."},
        "budget_state": {"arms": {"generate": 6, "evolve": 3}},
        "rng_state": [3, [0], None],
    }
    doc.update(overrides)
    return doc


def _result_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "best": {
            "prompt_hash": "abc123def456",
            "text": "Classify the animal.",
            "metrics": {"accuracy": 0.875},
        },
        "ranked": [],
        "history": [{"round": 0, "n_candidates": 2, "n_requests": 4}],
        "total_cost": dict(_COST),
        "cost_by_tier": {"generate": 8, "eval": 12},
        "artifacts": {},
        "final_eval": {
            "split": "test",
            "prompt_hash": "abc123def456",
            "metrics": {"accuracy": 0.9},
            "cost": {"requests": 2},
        },
    }
    doc.update(overrides)
    return doc


def _status(code: int, out: str) -> None:
    assert code == 0, f"eyas status must exit 0, got {code}"
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# (a) complete run dir: status + checkpoint + result
# ---------------------------------------------------------------------------


def test_status_full_run_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_status(tmp_path, _status_doc(phase="done", round=3))
    (tmp_path / "checkpoint.json").write_text(json.dumps(_checkpoint_doc()), encoding="utf-8")
    (tmp_path / "result.json").write_text(json.dumps(_result_doc()), encoding="utf-8")

    code = main(["status", str(tmp_path)])
    out = capsys.readouterr().out
    _status(code, out)
    assert "no status.json" not in out  # the degraded line is for dirs without one

    # run identity + phase + round come from status.json's own fields
    assert "run: demo_run" in out
    assert "strategy=hill_climb task=toy_status seed=0" in out
    assert "phase: done" in out
    assert "round: 3" in out
    # budget: allocator name + its own (opaque) state
    assert "budget: bandit" in out
    assert "arms" in out
    # cost: the main channel and the judge channel, keys mirroring result.json
    assert "cost: requests=120 input_tokens=9000 output_tokens=1500 wall_seconds=61.5" in out
    assert "judge: requests=12 input_tokens=700 output_tokens=90" in out
    # cache hit-rate is the ratio of the file's two counters
    assert "cache: hits=80 misses=40 hit_rate=66.7%" in out
    # rejections: the planner's tally by reason
    assert "images per request 2 > limit 1=3" in out
    # the writer's observation stamp, verbatim from the file
    assert f"updated_at: {read_status(tmp_path)['updated_at']}" in out
    # phase history tail (round / phase / seconds)
    assert "r1 optimizing 20.5s" in out
    # final block: result.json's best + metrics + report-split numbers
    assert "metrics: accuracy=0.875" in out
    assert "best: 'Classify the animal.'" in out
    assert "hash=abc123def456" in out
    assert "final_eval: split=test" in out
    assert "accuracy=0.9" in out
    assert "cost by tier: generate=8 eval=12" in out  # the file's own key order
    # status.json is the live observation: the checkpoint fallback stays unprinted
    assert "budget state (last checkpoint)" not in out


def test_status_prints_only_what_the_files_carry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A budget section without state, no rejections, no history: the renderer
    # prints what is there and nothing that only exists in the terminal.
    write_status(
        tmp_path,
        _status_doc(budget={"name": "flat"}, rejections={}, history=[]),
    )

    code = main(["status", str(tmp_path)])
    out = capsys.readouterr().out
    _status(code, out)
    budget_line = next(line for line in out.splitlines() if line.startswith("budget:"))
    assert budget_line == "budget: flat"
    assert "rejections: none" in out
    assert not any(line.startswith("history:") for line in out.splitlines())
    assert "final_eval" not in out


def test_status_run_name_falls_back_to_the_run_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # status.run may legitimately be empty (a hand-written or partial document);
    # the name line then shows the directory itself, never a made-up name.
    write_status(tmp_path, _status_doc(run={}))

    code = main(["status", str(tmp_path)])
    out = capsys.readouterr().out
    _status(code, out)
    assert f"run: {tmp_path.name}" in out
    assert "demo_run" not in out


# ---------------------------------------------------------------------------
# (b) mid-run dir: status.json only
# ---------------------------------------------------------------------------


def test_status_mid_run_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_status(tmp_path, _status_doc(phase="optimizing", round=2))

    code = main(["status", str(tmp_path)])
    out = capsys.readouterr().out
    _status(code, out)  # exit 0, no traceback text
    assert "phase: optimizing" in out
    assert "round: 2" in out
    assert "cost: requests=120" in out
    # nothing final to report yet
    assert "best:" not in out
    assert "metrics:" not in out
    assert "no status.json" not in out


# ---------------------------------------------------------------------------
# (c) pre-status dir: no status.json at all
# ---------------------------------------------------------------------------


def test_status_without_status_file_degrades(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["status", str(tmp_path)])
    out = capsys.readouterr().out
    _status(code, out)
    assert "no status.json (pre-status run or older version)" in out
    # nothing to render beyond the degraded line but the directory itself
    assert "phase:" not in out
    assert "round:" not in out
    assert "cost:" not in out


def test_status_of_a_missing_dir_still_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A typo'd path is a run-dir shape like any other: degraded, exit 0, no traceback.
    code = main(["status", str(tmp_path / "no-such-run")])
    out = capsys.readouterr().out
    _status(code, out)
    assert "no status.json (pre-status run or older version)" in out


def test_status_without_status_file_shows_checkpoint_and_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "checkpoint.json").write_text(json.dumps(_checkpoint_doc()), encoding="utf-8")
    (tmp_path / "result.json").write_text(json.dumps(_result_doc()), encoding="utf-8")

    code = main(["status", str(tmp_path)])
    out = capsys.readouterr().out
    _status(code, out)
    assert "no status.json (pre-status run or older version)" in out
    # checkpoint facts
    assert "strategy: hill_climb" in out
    assert "round: 3" in out
    assert "cost: requests=120" in out
    assert "judge: requests=12" in out
    assert "budget state (last checkpoint): arms=" in out
    # result facts
    assert "metrics: accuracy=0.875" in out
    assert "best: 'Classify the animal.'" in out
    assert "final_eval: split=test" in out


# ---------------------------------------------------------------------------
# a broken observation stays loud (the reader's contract, not the renderer's)
# ---------------------------------------------------------------------------


def test_status_requires_a_run_dir(capsys: pytest.CaptureFixture[str]) -> None:
    # No dir at all is a usage error, not a degraded report.
    with pytest.raises(SystemExit) as excinfo:
        main(["status"])
    assert excinfo.value.code == 2
    assert "run dir" in capsys.readouterr().err


def test_status_corrupt_status_file_is_loud(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        main(["status", str(tmp_path)])


def test_status_accepts_the_report_style_run_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # ``--run`` mirrors ``eyas report``'s flag; the positional form is primary.
    write_status(tmp_path, _status_doc())
    assert main(["status", "--run", str(tmp_path)]) == 0
    assert "phase: optimizing" in capsys.readouterr().out
