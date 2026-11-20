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

import re
import subprocess
import sys
import types
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
        self.run_config: dict | None = None
        self.run_kwargs: dict = {}

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

        def _experiment(config, **kwargs):
            self.run_config = config
            self.run_kwargs = kwargs

            class _Result:
                total_cost = types.SimpleNamespace(requests=0)

                def best_text(self):
                    return "best"

            return _Result()

        monkeypatch.setattr(runtime, "run_plan", _plan)
        monkeypatch.setattr(runtime, "run_sweep", _sweep)
        monkeypatch.setattr(runtime, "run_experiment", _experiment)
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
        ["sweep", "--spec", "specs/benchmark.yaml", "--config", str(cfg)],
        ["sweep", "--config", str(cfg), "--filter", "tag=q2b"],  # --filter needs an expansion mode
        ["sweep", "--config", str(cfg), "--dry-run"],  # --dry-run needs an expansion mode
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert (
            "exactly one of --config (repeatable), --plan, --matrix, or --spec" in err
            or "--filter only" in err
            or "--dry-run only" in err
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


# ---------------------------------------------------------------------------
# E2 command surface: run --cell / sweep --spec / gen-configs. The cell bodies
# come from the packaged spec in this checkout (the same assembly the golden-parity
# tests pin); here the CLI boundary is what's under test: mutual exclusion,
# the spec_sha stamp, expansion plumbing, and the generated-file contract.


_SPEC_DIR = Path(__file__).resolve().parents[2] / "src" / "eyas" / "spec"
_GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "goldens" / "configspec"
_GOLDEN_CELLS = _GOLDEN / "cells"
_QUICKSTART_MANIFEST = _GOLDEN / "quickstart.sha256"


def test_run_requires_exactly_one_of_config_or_cell(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (
        ["run"],  # neither selector
        ["run", "--config", "a.yaml", "--cell", "pets/raw"],  # both
        ["run", "--config", "a.yaml", "--variant", "openai"],  # --variant needs --cell
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "exactly one of --config or --cell" in err or "--variant only" in err


def test_run_cell_assembles_config_and_stamps_spec_sha(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from eyas.configspec import load_spec, spec_sha

    calls = _RuntimeCalls()
    calls.patch(monkeypatch, outcomes={}, run_dirs=[])

    code = main(
        [
            "run",
            "--cell",
            "pets/raw",
            "--spec-dir",
            str(_SPEC_DIR),
            "--set",
            "backend.name=mock",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == "Done: best='best' requests=0\n"
    cfg = calls.run_config
    assert cfg["run"]["name"] == "quickstart_pets_raw"
    assert cfg["task"]["name"] == "mcq"
    assert cfg["strategy"]["name"] == "raw"
    # the CLI boundary stamps provenance; --set still applies on top
    assert cfg["run"]["spec_sha"] == spec_sha(load_spec(_SPEC_DIR))
    assert cfg["backend"]["name"] == "mock"
    assert calls.run_kwargs == {"output_dir": None, "resume": None}


def test_run_config_path_carries_no_spec_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a plain --config run ships exactly what the file says — no spec_sha invented
    calls = _RuntimeCalls()
    calls.patch(monkeypatch, outcomes={}, run_dirs=[])
    cfg = _write_cfg(tmp_path, "a", 1)
    assert main(["run", "--config", str(cfg)]) == 0
    assert "spec_sha" not in calls.run_config["run"]


def test_run_cell_rejects_unknown_cell(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls = _RuntimeCalls()
    calls.patch(monkeypatch, outcomes={}, run_dirs=[])
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--cell", "nope/raw", "--spec-dir", str(_SPEC_DIR)])
    assert excinfo.value.code == 2
    assert "unknown dataset" in capsys.readouterr().err


def _sweep_spec_args(*extra: str) -> list[str]:
    return [
        "sweep",
        "--spec",
        str(_SPEC_DIR / "benchmark.yaml"),
        "--spec-dir",
        str(_SPEC_DIR),
        *extra,
    ]


def test_sweep_spec_expands_cells_stamps_sha_and_shares_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from eyas.configspec import load_spec, spec_sha

    calls = _RuntimeCalls()
    run_dirs = [tmp_path / f"r{i}" for i in range(8)]
    calls.patch(monkeypatch, outcomes={}, run_dirs=run_dirs)

    code = main(
        _sweep_spec_args(
            "--filter",
            "tag=q2b",
            "--filter",
            "dataset=pets",
            "--set",
            "run.seed=42",
            "--output-root",
            str(tmp_path / "out"),
        )
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "benchmark.yaml: 8 cells" in out
    assert "sweep done: 8 runs" in out
    assert len(calls.sweep_configs) == 8
    sha = spec_sha(load_spec(_SPEC_DIR))
    names = [c["run"]["name"] for c in calls.sweep_configs]
    assert names == [
        f"matrix_q2b_pets_{m}"
        for m in ("raw", "cot", "hill_climb", "evoprompt", "opro", "ape", "protegi", "spo")
    ]
    assert all(c["run"]["spec_sha"] == sha for c in calls.sweep_configs)
    assert all(c["run"]["seed"] == 42 for c in calls.sweep_configs)
    assert all(c["run"]["trace"] is True for c in calls.sweep_configs)
    # bench axes: the dataset axis carries bench dirs, the tag axis carries the model path
    assert all(
        c["task"]["params"]["data_dir"] == "datasets/bench/pets/data" for c in calls.sweep_configs
    )
    assert all(
        c["backend"]["params"]["model_path"] == "/path/to/Qwen3.5-2B" for c in calls.sweep_configs
    )
    assert calls.sweep_share is True


def test_sweep_spec_dry_run_prints_and_executes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _RuntimeCalls()
    calls.patch(monkeypatch, outcomes={}, run_dirs=[])

    code = main(_sweep_spec_args("--filter", "tag=q2b", "--filter", "dataset=pets", "--dry-run"))
    assert code == 0
    out = capsys.readouterr().out
    assert "matrix_q2b_pets_raw" in out
    assert "dry-run: 8 cells" in out
    assert "spec_sha=" in out
    assert calls.sweep_configs == []


def test_sweep_spec_rejects_foreign_benchmark_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _RuntimeCalls()
    calls.patch(monkeypatch, outcomes={}, run_dirs=[])
    other = tmp_path / "benchmark.yaml"
    other.write_text("run_name: x\naxes:\n  dataset: [pets]\n  method: [raw]\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["sweep", "--spec", str(other), "--spec-dir", str(_SPEC_DIR), "--dry-run"])
    assert excinfo.value.code == 2
    assert "not the benchmark section" in capsys.readouterr().err


def _parse_generated(text: str) -> dict:
    import yaml

    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    return yaml.safe_load(body)


def test_gen_configs_single_cell_round_trips_golden(tmp_path: Path, capsys) -> None:
    import json

    from eyas.configspec import load_spec, spec_sha

    code = main(
        ["gen-configs", "--cell", "gsm8k/cot", "--spec-dir", str(_SPEC_DIR), "--out", str(tmp_path)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "gen-configs: 1 YAMLs" in out
    path = tmp_path / "gsm8k_cot.yaml"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# GENERATED from configs/spec — do not edit")
    # provenance stamp (rc2): the header carries the spec content digest —
    # the same 12-hex value spec_sha records in run_config.json — as the
    # last token before the closing paren
    header = text.splitlines()[0]
    stamp = re.search(r" spec=([0-9a-f]{12})\)$", header)
    assert stamp is not None, header
    assert stamp.group(1) == spec_sha(load_spec(_SPEC_DIR))
    assert "# [method cot]" in text and "# [dataset gsm8k]" in text
    parsed = _parse_generated(text)
    golden = json.loads((_GOLDEN_CELLS / "gsm8k_cot.json").read_text(encoding="utf-8"))
    assert parsed == golden  # GENERATED header + notes strip to the shipped dict


def test_gen_configs_variant_mirror_and_availability_guard(tmp_path: Path, capsys) -> None:
    import json

    code = main(
        [
            "gen-configs",
            "--cell",
            "pets/raw",
            "--variant",
            "openai",
            "--spec-dir",
            str(_SPEC_DIR),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    text = (tmp_path / "pets_raw_openai.yaml").read_text(encoding="utf-8")
    assert "--variant openai" in text.splitlines()[0]
    parsed = _parse_generated(text)
    golden = json.loads((_GOLDEN_CELLS / "pets_raw_openai.json").read_text(encoding="utf-8"))
    assert parsed == golden

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "gen-configs",
                "--cell",
                "gsm8k/spo",
                "--variant",
                "openai",
                "--spec-dir",
                str(_SPEC_DIR),
                "--out",
                str(tmp_path / "nope"),
            ]
        )
    assert excinfo.value.code == 2
    assert "no 'openai' variant" in capsys.readouterr().err
    assert not (tmp_path / "nope").exists()  # fail-fast: nothing written


def test_gen_configs_default_dump_equals_shipped_surface(tmp_path: Path) -> None:
    import json

    code = main(["gen-configs", "--spec-dir", str(_SPEC_DIR), "--out", str(tmp_path)])
    assert code == 0
    files = sorted(p.stem for p in tmp_path.glob("*.yaml"))
    # the quickstart digest manifest is the shipped-cell surface; the default
    # dump must produce exactly those cells, no extras and nothing missing
    manifest = _QUICKSTART_MANIFEST.read_text(encoding="utf-8").splitlines()
    ids = sorted(line.split("  ", 1)[1] for line in manifest if line and not line.startswith("#"))
    assert files == ids
    # the representative cells also round-trip field for field
    for golden in _GOLDEN_CELLS.glob("*.json"):
        parsed = _parse_generated((tmp_path / f"{golden.stem}.yaml").read_text(encoding="utf-8"))
        expected = json.loads(golden.read_text(encoding="utf-8"))
        assert parsed == expected, golden.stem


def test_gen_configs_rejects_variant_without_cell(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["gen-configs", "--variant", "openai", "--out", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "--variant requires --cell" in capsys.readouterr().err


def test_gen_configs_default_spec_dir_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the packaged default must not assume a checkout layout under CWD: an
    # installed user runs gen-configs from any directory (rc2 ruling: the spec
    # travels inside the package, not next to the CWD)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "cells"
    assert main(["gen-configs", "--cell", "pets/raw", "--out", str(out)]) == 0
    assert (out / "pets_raw.yaml").exists()


def test_run_cell_dry_run_offline_and_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --dry-run assembles from the packaged spec and stops: no checkout under
    # CWD, no engine import, no run dir — the release smoke leans on all three
    monkeypatch.chdir(tmp_path)
    code = main(["run", "--cell", "pets/raw", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "nothing executed" in out
    assert re.search(r"spec_sha=[0-9a-f]{12}", out)
    assert not (tmp_path / "outputs").exists()
    # the smoke-bearing property: in a fresh interpreter the dry path imports
    # no eyas.engine.* module at all (in-process sys.modules is contaminated
    # by sibling tests, so probe a subprocess)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from eyas.cli import main\n"
            "assert main(['run', '--cell', 'pets/raw', '--dry-run']) == 0\n"
            "leaked = sorted(m for m in sys.modules if m.startswith('eyas.engine'))\n"
            "assert not leaked, leaked\n",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_run_dry_run_without_cell_is_a_usage_error(tmp_path: Path) -> None:
    cfg = _write_cfg(tmp_path, "cfg", 42)
    with pytest.raises(SystemExit):
        main(["run", "--config", str(cfg), "--dry-run"])


def test_run_dry_run_rejects_resume_and_output_dir(tmp_path: Path, capsys) -> None:
    # house style of _cmd_run: parser-enforced mutual exclusion, never silent
    # precedence — --dry-run executes nothing, so --resume/--output-dir have
    # nothing left to say (final-review finding: they were accepted and ignored)
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--cell", "pets/raw", "--dry-run", "--resume", str(tmp_path / "nope")])
    assert excinfo.value.code == 2
    assert "--dry-run" in capsys.readouterr().err
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--cell", "pets/raw", "--dry-run", "--output-dir", str(tmp_path / "out")])
    assert excinfo.value.code == 2
    assert "--dry-run" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
