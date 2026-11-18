"""final_eval contract: train-loop selection + ``run.final_eval.split`` reporting.

Covers: archived best re-evaluated on the final split with cost accounted independently (excluded from total_cost);
the two absolute-side skip paths (no config / same split — the compare-semantics skip is pinned in test_contract); ``run.eval_split`` can move the loop to another split.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    system_message,
    user_message,
)
from eyas.core.objective import Objective
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.runtime import run_experiment
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant


@register_task("toy_final_eval")
class ToyFinalEval(Task):
    """The mock can answer train samples (q1 -> cat) but not test samples (the mock always answers cat) --
    final metric 0.0 vs train 1.0 forms a readable signal that "the final evaluation really ran on test"."""

    name = "toy_final_eval"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        if split == "test":
            return [Sample(id="q_test", inputs={"question": "q_test"}, references={"label": "dog"})]
        return [Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"})]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _cfg(**run_extra: object) -> dict[str, object]:
    return {
        "run": {"name": "toy_final_eval_run", "seed": 0, **run_extra},
        "task": {"name": "toy_final_eval"},
        "protocol": {"name": "identity"},
        "backend": {"name": "mock", "params": {"responses": ["cat"], "cycle": True}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {
            "name": "cot",
            "params": {"base_prompt": "Classify the animal.", "cot_suffix": "Think step by step."},
        },
    }


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
