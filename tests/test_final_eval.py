"""final_eval contract: train-loop selection + ``run.final_eval.split`` reporting.

Covers: archived best re-evaluated on the final split with cost accounted independently (excluded from total_cost);
the two absolute-side skip paths (no config / same split — the compare-semantics skip is pinned in test_contract); ``run.eval_split`` can move the loop to another split.
"""

from __future__ import annotations

import json

from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.runtime import run_experiment
from tests.support.engine import ToyFinalEval  # noqa: F401  registered by import
from tests.support.engine import toy_final_eval_cfg as _cfg


def test_final_eval_reports_best_on_test_split(tmp_path):
    result = run_experiment(_cfg(final_eval={"split": "test"}), output_dir=tmp_path)
    fe = result.final_eval
    assert fe is not None
    assert fe["split"] == "test"
    assert (
        fe["metrics"]["accuracy"] == 0.0
    )  # the mock cannot answer test samples -> proves the final eval ran on test
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0  # the loop selects on train
    assert (
        fe["prompt_hash"] == result.best.prompt_id
    )  # archived best re-evaluated verbatim -> same hash
    # cost independence: total_cost only carries loop accounting; the final re-eval is accounted separately
    assert result.total_cost.requests == sum(r.n_requests for r in result.history)
    assert fe["cost"]["requests"] >= 1
    exported = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert exported["final_eval"]["split"] == "test"


def test_final_eval_none_without_config(tmp_path):
    result = run_experiment(_cfg(), output_dir=tmp_path)
    assert result.final_eval is None
    exported = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert exported["final_eval"] is None


def test_final_eval_skipped_when_same_split(tmp_path):
    result = run_experiment(_cfg(final_eval={"split": "train"}), output_dir=tmp_path)
    assert (
        result.final_eval is None
    )  # the archive already holds numbers for the same split, no re-evaluation


def test_run_eval_split_moves_loop_off_train(tmp_path):
    result = run_experiment(_cfg(eval_split="test"), output_dir=tmp_path)
    assert result.final_eval is None
    assert result.best is not None
    assert result.best.scores["accuracy"] == 0.0  # the loop ran on test throughout


def test_report_surfaces_final_eval_metrics_with_test_prefix(tmp_path):
    from eyas.engine.report import report_runs

    run_experiment(_cfg(final_eval={"split": "test"}), output_dir=tmp_path)
    md = report_runs([tmp_path], fmt="md")
    assert (
        "test_accuracy" in md
    )  # final split metrics enter the dynamic columns with the test_ prefix (framework convention)
    csv_text = report_runs([tmp_path], fmt="csv")
    assert "test_accuracy" in csv_text.splitlines()[0]


def test_report_prefix_follows_the_configured_split(tmp_path):
    """The prefix is the reporting split's own name, not a hardcoded "test_" — a
    final_eval.split=val run must not publish its columns as test metrics."""
    from eyas.engine.report import report_runs

    run_experiment(_cfg(final_eval={"split": "val"}), output_dir=tmp_path)
    md = report_runs([tmp_path], fmt="md")
    assert "val_accuracy" in md
    assert "test_accuracy" not in md


# The compare-semantics skip path (compare runs never re-evaluate on the final
# split) is pinned in tests/test_contract.py::
# test_compare_only_fixture_runs_on_compare_task — it needs a compare-capable
# strategy, which only the assembly fixture provides.
