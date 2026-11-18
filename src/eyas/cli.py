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
    run_p.add_argument("--config", required=True, help="Experiment config YAML")
    run_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-path config overrides (repeatable), e.g. --set run.seed=43",
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
            "order (base config + override-dict axes; see configs/matrix.yaml). "
            "Backend blocks are shared across cells with an identical backend "
            "section (one weight load per model tag). Combine with --filter and "
            "--set (e.g. one sweep cell set = --filter tag=X --filter dataset=Y "
            "--set run.seed=NN)."
        ),
    )
    sweep_p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="AXIS=VALUE[,VALUE]",
        help="(--matrix mode) restrict an axis to listed values, repeatable",
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

    if sum(bool(x) for x in (args.config, args.plan, args.matrix)) != 1:
        parser.error("sweep needs exactly one of --config (repeatable), --plan, or --matrix")
    if args.filter and not args.matrix:
        parser.error("--filter only applies to --matrix")
    if args.matrix:
        from eyas.runtime.matrix import (
            describe_expansion,
            expand_matrix,
            load_matrix_spec,
            parse_filters,
        )

        try:
            spec = load_matrix_spec(args.matrix)
            filters = parse_filters(args.filter, spec)
            configs = [cfg for _, cfg in expand_matrix(spec, filters, load_base=load_config)]
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"--matrix {args.matrix}: {exc}")
        if not configs:
            parser.error(f"--matrix expansion produced no cells (filters: {args.filter})")
        # per-cell overrides ride on top of the expansion (--set run.seed=NN)
        configs = [apply_overrides(cfg, args.set) for cfg in configs]
        print(describe_expansion(spec, filters, len(configs)))
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


def _cmd_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run one contract-driven optimization from a sectioned YAML."""
    if args.resume and args.output_dir:
        # Documented contract, parser-enforced: resume reuses an existing
        # run dir, so --output-dir has nothing left to say (no silent
        # precedence between the two).
        parser.error("--resume and --output-dir are mutually exclusive")
    from eyas.engine.runtime import load_config, run_experiment
    from eyas.runtime.config import apply_overrides

    config = apply_overrides(load_config(args.config), args.set)
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
    "run": _cmd_run,
    "report": _cmd_report,
    "status": _cmd_status,
    "sweep": _cmd_sweep,
}


if __name__ == "__main__":
    raise SystemExit(main())
