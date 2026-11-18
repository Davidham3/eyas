"""``eyas run`` / ``eyas sweep`` — argument- and plan-level contracts.

Run verb proposition: ``--resume`` and ``--output-dir`` are mutually
exclusive by documentation (resume reuses an existing run dir; it does not
create a new one at ``--output-dir``). Supplying both is a usage error — the
parser rejects it with exit code 2 — never a silent precedence rule where one
flag quietly wins over the other.

Sweep verb: the plan TSV (``_parse_sweep_plan``) is a strict two-column
contract — comments/blank lines skipped, anything that is not exactly
``<config yaml>\\t<run dir>`` fails loudly with ``plan:lineno``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eyas.cli import _parse_sweep_plan, main


def _write_cfg(dest: Path, name: str, seed: int) -> Path:
    """Minimal sectioned YAML that passes load_config's required-section check
    (task/backend/strategy mappings) — same shape the sweep verb reads."""
    path = dest / f"{name}.yaml"
    path.write_text(
        "task:\n  name: mcq\n"
        "backend:\n  name: mock\n"
        "strategy:\n  name: cot\n"
        f"run:\n  seed: {seed}\n",
        encoding="utf-8",
    )
    return path


def test_parse_sweep_plan_skips_comments_and_blanks(tmp_path: Path) -> None:
    cfg_a = _write_cfg(tmp_path, "a", 1)
    cfg_b = _write_cfg(tmp_path, "b", 2)
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "# first config\n"
        "\n"
        f"{cfg_a}\t{tmp_path / 'runs/a'}\n"
        f"  {cfg_b} \t runs/b \n"  # padded fields are stripped; run dir may stay CWD-relative
        "   # indented comment\n",
        encoding="utf-8",
    )
    entries = _parse_sweep_plan(plan)
    # one (loaded config, run dir) entry per real line, in file order;
    # configs come back through load_config (full dict, sections intact)
    assert [(e[0]["run"]["seed"], e[0]["task"]["name"], e[1]) for e in entries] == [
        (1, "mcq", tmp_path / "runs/a"),
        (2, "mcq", Path("runs/b")),
    ]


def test_parse_sweep_plan_rejects_malformed_line(tmp_path: Path) -> None:
    cfg = _write_cfg(tmp_path, "a", 1)
    plan = tmp_path / "plan.tsv"
    # a space-joined pair is a usage error, not a silent single-column line;
    # the error names plan:lineno counting skipped comments/blank lines
    plan.write_text(
        f"# comment\n\n   \n{cfg} {tmp_path / 'runs/a'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"plan\.tsv:4: expected '<config yaml>"):
        _parse_sweep_plan(plan)
    # a third column is equally rejected (exactly two tab-separated fields)
    plan.write_text(f"{cfg}\truns/a\truns/b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plan.tsv:1"):
        _parse_sweep_plan(plan)


def test_run_rejects_resume_combined_with_output_dir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Parser-level rejection: nothing is loaded or executed before the check,
    # so the config path never has to exist.
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run",
                "--config",
                "unused.yaml",
                "--output-dir",
                "some-new-run-dir",
                "--resume",
                "some-existing-run-dir",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--resume" in err
    assert "--output-dir" in err
    assert "mutually exclusive" in err


# ---------------------------------------------------------------------------
# dispatch contract: main -> _COMMANDS -> _cmd_* handlers. The handler bodies
# moved out of main verbatim; these tests pin the dispatch wiring and each
# verb's stdout/exit-code contract with the runtime seam monkeypatched (no
# experiment actually runs) — the branches main shed must not lose their pins.
# ---------------------------------------------------------------------------


class _RuntimeCalls:
    """Records the runtime seam's arguments so dispatch is pinned by evidence."""

    def __init__(self) -> None:
        self.plan_entries: list[tuple[dict, Path]] = []
        self.sweep_configs: list[dict] = []
        self.sweep_root = None
        self.sweep_share = None
        self.report_dirs: list[Path] = []

    def patch(self, monkeypatch: pytest.MonkeyPatch, *, outcomes, run_dirs) -> None:
        from eyas.engine import report as report_mod
        from eyas.engine import runtime

        def _plan(entries, *, continue_on_error=True):
            self.plan_entries = list(entries)
            return dict(outcomes)

        def _sweep(configs, output_root, *, share_backends=False):
            self.sweep_configs = list(configs)
            self.sweep_root = output_root
            self.sweep_share = share_backends
            return list(run_dirs)

        def _report(run_dirs, fmt="md"):
            self.report_dirs = list(run_dirs)
            return f"TABLE[{fmt}]"

        monkeypatch.setattr(runtime, "run_plan", _plan)
        monkeypatch.setattr(runtime, "run_sweep", _sweep)
        monkeypatch.setattr(report_mod, "report_runs", _report)


def test_sweep_plan_dispatch_runs_plan_then_reports_completed_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _RuntimeCalls()
    ok_dir = tmp_path / "runs/ok"
    ok_dir.mkdir(parents=True)
    (ok_dir / "result.json").write_text("{}", encoding="utf-8")
    calls.patch(
        monkeypatch,
        outcomes={str(ok_dir): None, str(tmp_path / "runs/dead"): "RuntimeError('boom')"},
        run_dirs=[],
    )
    cfg = _write_cfg(tmp_path, "a", 1)
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        f"{cfg}\t{tmp_path / 'runs/ok'}\n{cfg}\t{tmp_path / 'runs/dead'}\n", encoding="utf-8"
    )

    code = main(["sweep", "--plan", str(plan), "--set", "run.seed=42"])
    assert code == 1  # any contained failure surfaces as a non-zero exit (verbatim rule)
    out = capsys.readouterr().out
    assert "plan done: 1/2 runs ok" in out
    assert "FAILED " in out and "RuntimeError('boom')" in out
    # the failed exit returns before the summary is printed (old shape, pinned)
    assert "TABLE[md]" not in out
    # the plan entry went through apply_overrides: --set reached the config
    assert calls.plan_entries[0][0]["run"]["seed"] == 42
    # the failed exit returns before any report is built
    assert calls.report_dirs == []


def test_sweep_plan_dispatch_all_ok_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _RuntimeCalls()
    ok_dir = tmp_path / "runs/ok"
    ok_dir.mkdir(parents=True)
    (ok_dir / "result.json").write_text("{}", encoding="utf-8")
    calls.patch(
        monkeypatch, outcomes={str(ok_dir): None, str(tmp_path / "runs/bare"): None}, run_dirs=[]
    )
    cfg = _write_cfg(tmp_path, "a", 1)
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        f"{cfg}\t{tmp_path / 'runs/ok'}\n{cfg}\t{tmp_path / 'runs/bare'}\n", encoding="utf-8"
    )

    assert main(["sweep", "--plan", str(plan)]) == 0
    out = capsys.readouterr().out
    assert "plan done: 2/2 runs ok" in out
    assert "FAILED" not in out
    assert "TABLE[md]" in out
    # only run dirs that actually carry result.json reach the report
    assert calls.report_dirs == [ok_dir]


def test_sweep_config_dispatch_defaults_output_root_under_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _RuntimeCalls()
    run_dir = tmp_path / "outputs/00-a"
    calls.patch(monkeypatch, outcomes={}, run_dirs=[run_dir])
    cfg = _write_cfg(tmp_path, "a", 1)

    code = main(["sweep", "--config", str(cfg), "--share-backends"])
    assert code == 0
    out = capsys.readouterr().out
    assert "sweep done: 1 runs" in out
    assert "TABLE[md]" in out
    assert calls.sweep_configs[0]["task"]["name"] == "mcq"
    assert calls.sweep_share is True
    assert str(calls.sweep_root).startswith("outputs" + "/" + "sweep-")


def test_sweep_rejects_mixed_or_missing_selectors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write_cfg(tmp_path, "a", 1)
    for argv in (
        ["sweep"],
        ["sweep", "--config", str(cfg), "--plan", "p.tsv"],
        ["sweep", "--matrix", "m.yaml", "--config", str(cfg)],
        ["sweep", "--config", str(cfg), "--filter", "tag=q2b"],  # --filter needs --matrix
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert (
            "exactly one of --config (repeatable), --plan, or --matrix" in err
            or "--filter only" in err
        )


def _write_matrix_fixture(tmp_path: Path) -> Path:
    """Toy matrix spec over an absolute base pattern (CWD-independent): 2 tags x
    2 methods, one combo cap."""
    base = tmp_path / "base.yaml"
    base.write_text(
        "task:\n  name: mcq\nbackend:\n  name: mock\n  params:\n    model_path: placeholder\nstrategy:\n  name: cot\nrun:\n  seed: 42\n",
        encoding="utf-8",
    )
    spec = tmp_path / "matrix.yaml"
    spec.write_text(
        f"base: {base}\n"
        "run_name: mx_{tag}_{method}\n"
        "set:\n  run.trace: true\n"
        "axes:\n"
        "  tag:\n"
        "    t1:\n      backend.params.model_path: /m/t1\n"
        "    t2:\n      backend.params.model_path: /m/t2\n"
        "  method: [raw, cot]\n",
        encoding="utf-8",
    )
    return spec


def test_sweep_matrix_dispatch_expands_cells_and_shares_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _RuntimeCalls()
    run_dirs = [tmp_path / f"outputs/{i}-mx" for i in range(2)]
    calls.patch(monkeypatch, outcomes={}, run_dirs=run_dirs)
    spec = _write_matrix_fixture(tmp_path)

    code = main(
        [
            "sweep",
            "--matrix",
            str(spec),
            "--filter",
            "tag=t1",
            "--set",
            "run.seed=43",
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "matrix matrix.yaml: 2 cells (tag=t1, method=raw|cot)" in out
    assert "sweep done: 2 runs" in out
    # expansion: t1 only, both methods, run.name stamped, per-cell --set applied
    assert [c["run"]["name"] for c in calls.sweep_configs] == ["mx_t1_raw", "mx_t1_cot"]
    assert all(c["run"]["seed"] == 43 for c in calls.sweep_configs)
    assert all(c["backend"]["params"]["model_path"] == "/m/t1" for c in calls.sweep_configs)
    assert all(c["run"]["trace"] is True for c in calls.sweep_configs)
    # matrix mode always shares the engine across identical backend blocks
    assert calls.sweep_share is True


def test_sweep_matrix_rejects_over_filtered_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _write_matrix_fixture(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["sweep", "--matrix", str(spec), "--filter", "tag=nope"])
    assert excinfo.value.code == 2
    assert "not a value of axis" in capsys.readouterr().err


def test_report_dispatch_prints_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _RuntimeCalls()
    calls.patch(monkeypatch, outcomes={}, run_dirs=[])
    code = main(["report", "--run", str(tmp_path / "r1"), "--format", "csv"])
    assert code == 0
    assert capsys.readouterr().out == "TABLE[csv]\n"
    assert calls.report_dirs == [tmp_path / "r1"]
