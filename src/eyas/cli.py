from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from eyas import __version__, registry


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # 'prepare' forwards its whole argument tail to eyas.prepare.main's own parser:
    # the dataset knobs are many and evolve independently of the core CLI, so they
    # must not collide with this wrapper's parser (and the capability ships with
    # the package, not only with a repo checkout).
    if argv and argv[0] == "prepare":
        return _cmd_prepare(argv[1:])
    parser = argparse.ArgumentParser(
        prog="eyas",
        description="Train sharp-eyed prompts for multimodal models.",
    )
    parser.add_argument("--version", action="version", version=f"eyas {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    list_p = sub.add_parser("list", help="List all registered components")
    list_p.add_argument(
        "--config",
        help="Import this experiment config's plugins before listing (plugin registrations are invisible without it)",
    )

    sub.add_parser(
        "prepare",
        help="Download and build quickstart dataset splits (own flag set: see 'eyas prepare --help')",
    )

    run_p = sub.add_parser("run", help="Run one contract-driven optimization from a sectioned YAML")
    run_p.add_argument(
        "--config", help="Experiment config YAML (a custom or gen-configs-generated file)"
    )
    run_p.add_argument(
        "--cell",
        metavar="DATASET/METHOD",
        help=(
            "Assemble one quickstart cell from the packaged config spec and run "
            "it directly (no YAML file), e.g. --cell pets/raw"
        ),
    )
    run_p.add_argument(
        "--variant",
        help="(--cell mode) assembled variant, e.g. openai (see the packaged spec's cells.yaml)",
    )
    run_p.add_argument(
        "--spec-dir",
        default=None,
        metavar="DIR",
        help=(
            "(--cell mode) config-spec directory "
            "(default: the spec shipped inside the eyas package)"
        ),
    )
    run_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-path config overrides (repeatable), e.g. --set run.seed=43",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="(--cell mode) assemble the cell from the packaged spec and exit without running",
    )
    run_p.add_argument("--output-dir", help="Run dir (defaults to outputs/<name>/<timestamp>)")
    run_p.add_argument(
        "--resume", help="Resume from checkpoint.json in an existing run dir (not --output-dir)"
    )

    report_p = sub.add_parser("report", help="Summarize run dir outputs into a comparison table")
    report_p.add_argument("--run", action="append", required=True, help="Run dir (repeatable)")
    report_p.add_argument("--format", choices=["md", "csv"], default="md", help="Output format")

    status_p = sub.add_parser(
        "status", help="Render one run dir's observation contract (read-only)"
    )
    status_p.add_argument("run_dir", nargs="?", help="Run dir to inspect")
    status_p.add_argument(
        "--run", dest="run_flag", help="Run dir (same as the positional, report-style)"
    )

    sweep_p = sub.add_parser("sweep", help="Run multiple configs in order; abort all if any fails")
    sweep_p.add_argument(
        "--config", action="append", help="Config YAML, repeatable (one output root layout)"
    )
    sweep_p.add_argument(
        "--plan",
        help=(
            "Plan TSV, one '<config yaml>\\t<run dir>' per line (# comments ok). "
            "One process, engine shared across all lines (a model tag's whole "
            "batch pays one weight load); per-run failure containment."
        ),
    )
    sweep_p.add_argument(
        "--matrix",
        metavar="SPEC.yaml",
        help=(
            "Matrix spec YAML: expand facet axes into cells and run them in "
            "order (base config + override-dict axes). Backend blocks are "
            "shared across cells with an identical backend section (one weight "
            "load per model tag). Combine with --filter and --set (e.g. one "
            "sweep cell set = --filter tag=X --filter dataset=Y --set "
            "run.seed=NN)."
        ),
    )
    sweep_p.add_argument(
        "--spec",
        metavar="BENCHMARK.yaml",
        help=(
            "Benchmark spec inside a config spec (benchmark.yaml of the spec "
            "dir): expand the published matrix from the packaged spec with no "
            "per-cell "
            "files — cell bases are assembled (axes × combos; --filter/--set as "
            "in --matrix mode)."
        ),
    )
    sweep_p.add_argument(
        "--spec-dir",
        default=None,
        metavar="DIR",
        help=(
            "(--spec mode) config-spec directory holding benchmark.yaml "
            "(default: the spec shipped inside the eyas package)"
        ),
    )
    sweep_p.add_argument(
        "--dry-run",
        action="store_true",
        help="(expansion modes) print the expansion and exit without running",
    )
    sweep_p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="AXIS=VALUE[,VALUE]",
        help="(--matrix/--spec mode) restrict an axis to listed values, repeatable",
    )
    sweep_p.add_argument(
        "--share-backends",
        action="store_true",
        help="(--config mode) reuse core backends across configs with an identical backend block",
    )
    sweep_p.add_argument("--output-root", help="Output root (default outputs/sweep-<timestamp>)")
    sweep_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-path overrides (apply to every config, repeatable)",
    )

    gen_p = sub.add_parser(
        "gen-configs",
        help="Export editable cell YAMLs from the packaged spec (escape hatch; products are gitignored)",
    )
    gen_p.add_argument(
        "--cell",
        action="append",
        metavar="DATASET/METHOD",
        help="Cell to export (repeatable; default: every dataset x method in the spec)",
    )
    gen_p.add_argument(
        "--variant",
        help="Export the variant mirror of each --cell (requires --cell), e.g. openai",
    )
    gen_p.add_argument(
        "--out",
        default="configs/generated",
        help="Output directory (default: configs/generated, gitignored)",
    )
    gen_p.add_argument(
        "--spec-dir",
        default=None,
        metavar="DIR",
        help="Config-spec directory (default: the spec shipped inside the eyas package)",
    )

    args = parser.parse_args(argv)
    return _COMMANDS[args.command](parser, args)


def _cmd_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Summarize run dir outputs into a comparison table."""
    from eyas.engine.report import report_runs

    print(report_runs([Path(r) for r in args.run], fmt=args.format))
    return 0


def _cmd_sweep(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run multiple configs in order; abort all if any fails."""
    from datetime import datetime

    from eyas.engine.report import report_runs
    from eyas.engine.runtime import load_config, run_plan, run_sweep
    from eyas.runtime.config import apply_overrides

    if sum(bool(x) for x in (args.config, args.plan, args.matrix, args.spec)) != 1:
        parser.error(
            "sweep needs exactly one of --config (repeatable), --plan, --matrix, or --spec"
        )
    if args.filter and not (args.matrix or args.spec):
        parser.error("--filter only applies to --matrix/--spec")
    if args.dry_run and not (args.matrix or args.spec):
        parser.error("--dry-run only applies to --matrix/--spec")
    if args.spec:
        from eyas.configspec import (
            benchmark_axes_spec,
            benchmark_cells,
            load_spec,
            spec_sha,
        )
        from eyas.runtime.matrix import describe_expansion, parse_filters

        try:
            spec = load_spec(args.spec_dir)
            axes_spec = benchmark_axes_spec(spec)
            if Path(args.spec).resolve() != axes_spec.path.resolve():
                raise ValueError(
                    f"--spec {args.spec} is not the benchmark section of {spec.path} "
                    f"(expected {axes_spec.path})"
                )
            filters = parse_filters(args.filter, axes_spec)
            cells = benchmark_cells(spec, filters)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"--spec {args.spec}: {exc}")
        if not cells:
            parser.error(f"--spec expansion produced no cells (filters: {args.filter})")
        sha = spec_sha(spec)
        configs = []
        for _, cfg in cells:
            cfg["run"]["spec_sha"] = sha
            configs.append(apply_overrides(cfg, args.set))
        print(describe_expansion(axes_spec, filters, len(configs)))
        if args.dry_run:
            for name, _ in cells:
                print(name)
            print(f"dry-run: {len(configs)} cells, spec_sha={sha} — nothing executed")
            return 0
        output_root = args.output_root or (
            Path("outputs") / f"sweep-{datetime.now():%Y%m%d-%H%M%S}"
        )
        run_dirs = run_sweep(configs, output_root, share_backends=True)
        print(f"sweep done: {len(run_dirs)} runs, output root: {output_root}")
        print(report_runs(run_dirs, fmt="md"))
        return 0
    if args.matrix:
        from eyas.runtime.matrix import (
            default_base_resolver,
            describe_expansion,
            expand_matrix,
            load_matrix_spec,
            parse_filters,
        )

        try:
            matrix = load_matrix_spec(args.matrix)
            filters = parse_filters(args.filter, matrix)
            configs = [
                cfg
                for _, cfg in expand_matrix(
                    matrix, filters, resolve_base=default_base_resolver(matrix, load_config)
                )
            ]
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"--matrix {args.matrix}: {exc}")
        if not configs:
            parser.error(f"--matrix expansion produced no cells (filters: {args.filter})")
        # per-cell overrides ride on top of the expansion (--set run.seed=NN)
        configs = [apply_overrides(cfg, args.set) for cfg in configs]
        print(describe_expansion(matrix, filters, len(configs)))
        if args.dry_run:
            print(f"dry-run: {len(configs)} cells — nothing executed")
            return 0
        output_root = args.output_root or (
            Path("outputs") / f"sweep-{datetime.now():%Y%m%d-%H%M%S}"
        )
        # matrix cells are facet products: within one leading-axis group the
        # backend block is identical by construction, so sharing the engine is
        # the whole point of sweeping a matrix in one process.
        run_dirs = run_sweep(configs, output_root, share_backends=True)
        print(f"sweep done: {len(run_dirs)} runs, output root: {output_root}")
        print(report_runs(run_dirs, fmt="md"))
        return 0
    if args.plan:
        entries = _parse_sweep_plan(Path(args.plan))
        entries = [(apply_overrides(cfg, args.set), run_dir) for cfg, run_dir in entries]
        outcomes = run_plan(entries)
        failed = {k: v for k, v in outcomes.items() if v is not None}
        print(f"plan done: {len(outcomes) - len(failed)}/{len(outcomes)} runs ok")
        for run_dir, err in failed.items():
            print(f"  FAILED {run_dir}: {err}")
        if failed:
            return 1
        dirs = [Path(k) for k in outcomes]
        print(report_runs([d for d in dirs if (d / "result.json").exists()], fmt="md"))
        return 0
    configs = [apply_overrides(load_config(c), args.set) for c in args.config]
    output_root = args.output_root or (Path("outputs") / f"sweep-{datetime.now():%Y%m%d-%H%M%S}")
    run_dirs = run_sweep(configs, output_root, share_backends=args.share_backends)
    print(f"sweep done: {len(run_dirs)} runs, output root: {output_root}")
    print(report_runs(run_dirs, fmt="md"))
    return 0


def _cmd_gen_configs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Export editable cell YAMLs from the packaged spec (the escape hatch): default
    every dataset x method plus the variant mirrors the spec declares available.
    Products live under configs/generated (gitignored) and carry a GENERATED
    header — the committed surface stays the spec. Validation is fail-fast:
    the whole plan is checked before a single file is written."""
    from eyas.configspec import cell_notes, load_spec, render_cell_yaml

    if args.variant and not args.cell:
        parser.error(
            "--variant requires --cell (the default export already covers "
            "every available variant mirror)"
        )
    try:
        spec = load_spec(args.spec_dir)
        if args.variant and args.variant not in spec.variants:
            raise ValueError(f"unknown variant {args.variant!r} (known: {sorted(spec.variants)})")
        cells: list[tuple[str, str, str | None, str]] = []
        if args.cell:
            for item in args.cell:
                if "/" not in item or item.count("/") != 1:
                    raise ValueError(f"--cell must be DATASET/METHOD, got {item!r}")
                dataset, method = item.split("/")
                stem = f"{dataset}_{method}" + (f"_{args.variant}" if args.variant else "")
                cells.append((dataset, method, args.variant, stem))
            if args.variant:
                available_list = spec.variants[args.variant].available
                for dataset, method, _, _ in cells:
                    if f"{dataset}/{method}" not in available_list:
                        available = ", ".join(available_list) or "(none)"
                        raise ValueError(
                            f"cell {dataset}/{method} has no {args.variant!r} variant "
                            f"(available: {available})"
                        )
        else:
            cells = [
                (dataset, method, None, f"{dataset}_{method}")
                for dataset in spec.datasets
                for method in spec.methods
            ]
            for name, variant in sorted(spec.variants.items()):
                for cell in variant.available:
                    ds, meth = cell.split("/")
                    cells.append((ds, meth, name, f"{ds}_{meth}_{name}"))
        # render everything in memory first: a mid-batch assembly failure must
        # not leave a half-written output dir behind
        rendered = [
            (stem, render_cell_yaml(spec, dataset, method, variant))
            for dataset, method, variant, stem in cells
        ]
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, text in rendered:
            (out_dir / f"{stem}.yaml").write_text(text, encoding="utf-8")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    notes = sum(len(cell_notes(spec, d, m, v)) for d, m, v, _ in cells)
    print(
        f"gen-configs: {len(rendered)} YAMLs -> {out_dir}/ (GENERATED headers, {notes} why-notes)"
    )
    return 0


def _cmd_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run one contract-driven optimization: a sectioned YAML, or one cell
    assembled live from the packaged spec (``--cell``)."""
    if args.resume and args.output_dir:
        # Documented contract, parser-enforced: resume reuses an existing
        # run dir, so --output-dir has nothing left to say (no silent
        # precedence between the two).
        parser.error("--resume and --output-dir are mutually exclusive")
    if sum(bool(x) for x in (args.config, args.cell)) != 1:
        parser.error("run needs exactly one of --config or --cell")
    if args.variant and not args.cell:
        parser.error("--variant only applies to --cell")
    if args.dry_run and not args.cell:
        parser.error("--dry-run only applies to --cell")
    if args.dry_run and (args.resume or args.output_dir):
        # the dry path executes nothing, so resume/output-dir have nothing to
        # say — parser-enforced, matching the resume/output-dir rule above
        parser.error("--dry-run cannot be combined with --resume or --output-dir")

    if args.cell:
        from eyas.configspec import cell_config, load_spec, spec_sha

        if "/" not in args.cell or args.cell.count("/") != 1:
            parser.error(f"--cell must be DATASET/METHOD, got {args.cell!r}")
        dataset, method = args.cell.split("/")
        try:
            spec = load_spec(args.spec_dir)
            # spec + run dir is the reproducibility record: the content hash
            # rides the run section next to the effective config the runtime
            # persists (run_config.json). Stamped here, at the CLI boundary —
            # cell_config stays the golden-pure assembly.
            config = cell_config(spec, dataset, method, args.variant)
            config["run"]["spec_sha"] = spec_sha(spec)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"--cell {args.cell}: {exc}")
        if args.dry_run:
            # the offline preview: packaged spec in, assembled config out,
            # nothing executed. Deliberately free of eyas.engine imports —
            # the bare-wheel release smoke depends on that.
            from eyas.runtime.config import apply_overrides

            config = apply_overrides(config, args.set)
            print(
                f"run --cell {args.cell}: config assembled, "
                f"spec_sha={config['run']['spec_sha']} — nothing executed"
            )
            return 0
    else:
        from eyas.engine.runtime import load_config

        assert args.config is not None  # exactly-one-of check above
        config = load_config(args.config)
    from eyas.engine.runtime import run_experiment
    from eyas.runtime.config import apply_overrides

    config = apply_overrides(config, args.set)
    result = run_experiment(config, output_dir=args.output_dir, resume=args.resume)
    best = result.best_text()
    print(f"Done: best={best!r} requests={result.total_cost.requests}")
    return 0


def _parse_sweep_plan(path: Path) -> list[tuple[dict, Path]]:
    """Plan TSV: one '<config yaml>\\t<run dir>' per line; # comments and blank
    lines are skipped. Run dirs may be absolute or CWD-relative."""
    from eyas.engine.runtime import load_config

    entries: list[tuple[dict, Path]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected '<config yaml>\\t<run dir>', got {line!r}")
        entries.append((load_config(parts[0].strip()), Path(parts[1].strip())))
    return entries


# ---------------------------------------------------------------------------
# ``eyas status`` — the read-only reference consumer of the run-dir contract
#
# Hard rule: every printed field comes from the observation files (status.json /
# checkpoint.json / result.json). The terminal shows the same facts the files
# carry — nothing is invented here (no ETA, no extrapolation), so a future
# dashboard reading the same files renders the same sections.
# ---------------------------------------------------------------------------

#: ``cost`` mirrors result.json's ``total_cost`` keys: the main channel first,
#: the judge channel under its own label (both rendered in the file's key order).
_JUDGE_PREFIX = "judge_"


def _val(value: object) -> str:
    """Render one JSON value: floats lose a redundant ``.0``, nesting keeps the
    document's own encoding (JSON), so the line never paraphrases the file."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str | int | bool) or value is None:
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _kv(source: object, keys: tuple[str, ...] = ()) -> str:
    """``key=value`` pairs for ``keys`` — or for every entry the file carries when
    ``keys`` is empty. Keys a document does not carry are never printed."""
    if not isinstance(source, Mapping):
        return ""
    items = ((key, source[key]) for key in keys if key in source) if keys else source.items()
    return " ".join(f"{key}={_val(value)}" for key, value in items)


def _section(label: str, value: str) -> str:
    return f"{label}: {value}" if value else f"{label}: -"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Read one of the run dir's JSON documents; ``None`` when the file is absent.

    Absent is a legitimate run-dir state (mid-run, pre-status, cancelled before
    the first round); a present-but-unusable document is not, so it raises just
    like :func:`eyas.engine.status.read_status` — a broken observation is loud.
    """
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(doc).__name__}")
    return doc


def _cost_lines(cost: object) -> list[str]:
    """Cost section as two lines: the main channel and the judge channel."""
    if not isinstance(cost, Mapping):
        return [_section("cost", "")]
    main = " ".join(
        f"{key}={_val(value)}" for key, value in cost.items() if not key.startswith(_JUDGE_PREFIX)
    )
    judge = " ".join(
        f"{key.removeprefix(_JUDGE_PREFIX)}={_val(value)}"
        for key, value in cost.items()
        if key.startswith(_JUDGE_PREFIX)
    )
    return [_section("cost", main), _section("judge", judge)]


def _history_entry(entry: object) -> str:
    item = entry if isinstance(entry, Mapping) else {}
    parts = [f"r{item.get('round', '?')}", str(item.get("phase", "?"))]
    if "elapsed_sec" in item:
        parts.append(f"{_val(item['elapsed_sec'])}s")
    return " ".join(parts)


def _run_line(run: Mapping[str, Any], run_dir: Path) -> str:
    """Identity line: the file's name (else the dir it lives in) + strategy/task/seed."""
    identity = _kv(run, ("strategy", "task", "seed"))
    name = run.get("name") or run_dir.name
    return _section("run", f"{name} ({identity})" if identity else str(name))


def _budget_line(budget: object) -> str:
    if isinstance(budget, Mapping):
        budget_line = str(budget.get("name", ""))
        if budget.get("state"):
            budget_line += f" state={_val(budget['state'])}"
    else:
        budget_line = str(budget or "")
    return _section("budget", budget_line)


def _cache_line(cache: object) -> str:
    if isinstance(cache, Mapping) and "hits" in cache and "misses" in cache:
        hits, misses = cache["hits"], cache["misses"]
        total = hits + misses
        # the hit-rate is the ratio of the file's two counters, nothing else
        rate = f" hit_rate={100.0 * hits / total:.1f}%" if total else ""
        return f"cache: hits={hits} misses={misses}{rate}"
    return _section("cache", _kv(cache))


def _rejections_line(rejections: object) -> str:
    return _section(
        "rejections",
        " ".join(f"{key}={_val(value)}" for key, value in sorted(rejections.items()))
        if isinstance(rejections, Mapping) and rejections
        else "none",
    )


def _status_section(run_dir: Path, status: dict[str, Any]) -> list[str]:
    """The sections status.json carries: identity, progress, budget, cost,
    cache, rejections, stamp and the phase-history tail. Each line comes from
    its own renderer; this only fixes the order."""
    run_section = status.get("run")
    run = run_section if isinstance(run_section, Mapping) else {}
    lines = [
        _run_line(run, run_dir),
        _section("phase", str(status.get("phase", ""))),
        _section("round", _val(status["round"]) if "round" in status else ""),
        _budget_line(status.get("budget")),
    ]
    lines += _cost_lines(status.get("cost"))
    lines.append(_cache_line(status.get("cache")))
    lines.append(_rejections_line(status.get("rejections")))
    lines.append(_section("updated_at", str(status.get("updated_at", ""))))
    history = status.get("history")
    if isinstance(history, list) and history:
        lines.append("history: " + " | ".join(_history_entry(entry) for entry in history))
    return lines


def _checkpoint_section(checkpoint: dict[str, Any]) -> list[str]:
    """What a pre-status run dir still says about itself through its checkpoint."""
    lines = []
    if "strategy" in checkpoint:
        lines.append(_section("strategy", str(checkpoint["strategy"])))
    if "round_idx" in checkpoint:
        lines.append(_section("round", _val(checkpoint["round_idx"])))
    lines += _cost_lines(checkpoint.get("total_cost"))
    lines.append(_section("budget state (last checkpoint)", _kv(checkpoint.get("budget_state"))))
    return lines


def _result_section(result: dict[str, Any]) -> list[str]:
    """The final block: result.json's best + metrics + report-split numbers."""
    lines = []
    best = result.get("best")
    if isinstance(best, Mapping):
        lines.append(_section("metrics", _kv(best.get("metrics"))))
        if best.get("text") is not None:
            hash_part = f" hash={_val(best['prompt_hash'])}" if "prompt_hash" in best else ""
            lines.append(_section("best", f"{best['text']!r}{hash_part}"))
    elif "best" in result:
        lines.append("best: none (no archived prompt)")
    final_eval = result.get("final_eval")
    if isinstance(final_eval, Mapping):
        parts = [_kv(final_eval, ("split",)), _kv(final_eval.get("metrics"))]
        if "prompt_hash" in final_eval:
            parts.append(f"hash={_val(final_eval['prompt_hash'])}")
        lines.append(_section("final_eval", " ".join(part for part in parts if part)))
    lines.append(_section("cost by tier", _kv(result.get("cost_by_tier"))))
    return lines


def _status_lines(
    run_dir: Path,
    status: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> list[str]:
    """Render the fixed sections from whichever files carry them, in fixed order."""
    lines = [f"run dir: {run_dir}"]
    if status is None:
        # A run dir without status.json still tells what it can.
        lines.append("no status.json (pre-status run or older version)")
        if checkpoint is not None:
            lines += _checkpoint_section(checkpoint)
    else:
        lines += _status_section(run_dir, status)
    if result is not None:
        lines += _result_section(result)
    return lines


def _cmd_list(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Print the registered components; ``--config`` first imports that experiment's plugins."""
    import eyas.builtin  # noqa: F401 — trigger built-in component registration
    import eyas.engine.builtin  # noqa: F401 — trigger engine component registration

    if args.config:
        # Deliberately NOT load_config: listing has no business rejecting
        # a config for missing run sections — we only want its plugins list.
        import yaml

        from eyas.runtime.config import load_plugins

        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            parser.error(f"--config must be a YAML mapping: {args.config}")
        load_plugins(raw)

    for group in registry.groups():
        names = registry.registered(group)
        print(f"{group:<10} {' '.join(names) if names else '(none)'}")
    from eyas.engine import registry as engine_registry

    for group in engine_registry.groups():
        names = engine_registry.registered(group)
        print(f"engine:{group:<7} {' '.join(names) if names else '(none)'}")
    return 0


def _cmd_status(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Render a run dir's observation contract; read-only, exit 0 on every shape.

    A corrupt ``status.json`` deliberately propagates ``json.JSONDecodeError`` —
    reading a broken observation is loud by contract (``read_status``), and this
    renderer adds no fallback of its own.
    """
    if not (args.run_dir or args.run_flag):
        parser.error("status needs a run dir: eyas status <run dir>")
    from eyas.engine.status import read_status

    run_dir = Path(args.run_dir or args.run_flag)
    status = read_status(run_dir)
    # status.json is the live observation; checkpoint.json is only consulted as
    # the fallback source for a dir that has no status document at all.
    checkpoint = None if status is not None else _read_json_object(run_dir / "checkpoint.json")
    for line in _status_lines(
        run_dir,
        status,
        checkpoint,
        _read_json_object(run_dir / "result.json"),
    ):
        print(line)
    return 0


def _cmd_prepare(argv: list[str]) -> int:
    """Forward to ``eyas.prepare.main`` (its own parser owns the flags)."""
    from eyas import prepare

    try:
        return prepare.main(list(argv))
    except ModuleNotFoundError as exc:
        print(
            f"eyas prepare needs the '{exc.name}' module: install the prepare extra "
            "(uv sync --extra prepare, or pip install 'eyas[prepare]')",
            file=sys.stderr,
        )
        return 2


#: subcommand name -> handler(parser, args) -> exit code. Every handler is a
#: top-level ``_cmd_*`` function; ``main`` only builds the parser and dispatches.
_COMMANDS: dict[str, Callable[[argparse.ArgumentParser, argparse.Namespace], int]] = {
    "list": _cmd_list,
    "gen-configs": _cmd_gen_configs,
    "run": _cmd_run,
    "report": _cmd_report,
    "status": _cmd_status,
    "sweep": _cmd_sweep,
}


if __name__ == "__main__":
    raise SystemExit(main())
