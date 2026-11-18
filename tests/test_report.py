"""eyas.engine.report: the run-dir contract and the report-level failure modes."""

import json

import pytest

from eyas.engine.report import report_runs, summarize_run


def _write_run(run_dir, *, result, config):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if config is not None:
        (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_missing_result_json_rejects_the_dir_naming_it(tmp_path):
    """A dir that never held a run is rejected, and the message says where."""
    stray = tmp_path / "not_a_run"
    stray.mkdir()
    (stray / "status.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as excinfo:
        summarize_run(stray)
    assert "not a valid run dir (missing result.json)" in str(excinfo.value)
    assert str(stray) in str(excinfo.value)


def test_run_without_seed_renders_dash(tmp_path):
    """A run config that never recorded a seed reports it as the empty marker."""
    _write_run(
        tmp_path / "run",
        result={
            "best": {"text": "p", "metrics": {"accuracy": 0.5}},
            "total_cost": {"requests": 2},
        },
        config={"run": {"name": "seedless"}, "strategy": {"name": "raw"}, "task": {"name": "t"}},
    )
    assert summarize_run(tmp_path / "run").seed is None
    md = report_runs([tmp_path / "run"], fmt="md")
    assert "| seedless | raw | t | - | 0.5 |" in md


def test_report_requires_at_least_one_run_dir():
    with pytest.raises(ValueError, match="report requires at least one run dir"):
        report_runs([])
