#!/usr/bin/env python3
"""Aggregate benchmark run directories into the markdown fragments published in the docs.

Single source of truth for every published benchmark number: the quickstart
effect tables, the E1 backend-parity table and the E3 cost profile, the bench
A/B effect + cost tables, the InternVL family tables, the README benchmark
scoreboard, and the method cards. Stdlib only; reads
``result.json`` files from the sweep run-directory layout
``<outputs-root>/<sweep>_<dataset>_s<seed>/NN-<run.name>/`` and prints markdown
fragments to stdout (warnings to stderr). The stems each subcommand reads
(``eyas sweep --output-root ...`` produces the first two):

* ``qs_<ds>_s<seed>`` — quickstart E2 effect and cost cells;
* ``qs_e1_openai_<ds>_s42`` — E1 arm (the vllm arm is the plain
  ``qs_<ds>_s42`` sweep);
* ``<sweep>_<tag>_<ds>_s<seed>`` — bench / InternVL sweeps, ``--sweep`` selecting
  the family
  (default ``matrix``).

Number-extraction rules (binding):

* effect numbers come from ``final_eval.metrics`` (report split, always ``test``
  here); if a cell has no ``final_eval`` the script falls back to ``best.metrics``
  and says so on stderr — never silently;
* primary metric per dataset: pets / cub / textvqa / nextqa / gsm8k ->
  ``accuracy``, sroie -> ``field_acc`` (``strict_acc`` shown as the
  parenthetical auxiliary in §3);
* aggregation: mean ± population σ over the seeds present in the outputs root
  (n noted per column); Δpp is against the raw row of the same model block;
* a cell whose mean ``parse_fail`` reaches 5% also carries the parsed-only
  accuracy — ``pf X% → parsed Y%``, where parsed = accuracy / (1 − parse_fail)
  per seed, then averaged. Protocol accuracy charges unparseable predictions as
  wrong, so cells that differ in format compliance are not comparable on
  accuracy alone; cells below the threshold render unchanged;
* cost rows use ``total_cost`` (optimization loop only; the shared final_eval test
  pass is accounted separately). A cell whose loop billed zero requests was
  fully served by the run-dir record store on its final pass — the bulk cost was
  billed on an earlier attempt and is not visible in result.json; such cells are
  marked "(resume: billed on first attempt)" and excluded from cost means.

Usage examples (run from the repo root):

    python scripts/aggregate_bench_results.py quickstart-effect --dataset pets
    python scripts/aggregate_bench_results.py quickstart-cost
    python scripts/aggregate_bench_results.py e1-parity
    python scripts/aggregate_bench_results.py bench-effect --dataset sroie
    python scripts/aggregate_bench_results.py internvl-effect --dataset pets
    python scripts/aggregate_bench_results.py headline
    python scripts/aggregate_bench_results.py method-cards --dataset textvqa
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Quickstart + E1 parity datasets (Qwen3.5-9B only).
DATASETS = ("pets", "cub", "sroie", "textvqa")
# Every dataset the published matrix sweeps:
# the business four plus the video (nextqa) and math (gsm8k) sets.
BENCH_DATASETS = (*DATASETS, "nextqa", "gsm8k")
PRIMARY_METRIC = {
    "pets": "accuracy",
    "cub": "accuracy",
    "textvqa": "accuracy",
    "sroie": "field_acc",
    "nextqa": "accuracy",
    "gsm8k": "accuracy",
}
# Published sweep order (the NN- directory prefix order of every sweep).
METHODS = ("raw", "cot", "evoprompt", "hill_climb", "opro", "protegi", "spo", "ape")
# sroie has no generative-prompt demos (opro/ape need demo questions): by design.
DATASET_METHODS = {
    ds: METHODS if ds != "sroie" else tuple(m for m in METHODS if m not in ("opro", "ape"))
    for ds in BENCH_DATASETS
}
QS_MODELS = ("42", "43", "44")  # quickstart seeds
E1_SEED = "42"  # E1 backend-parity seed
E1_METHODS = ("raw", "cot")  # E1 arms (sweep order: raw=00, cot=01)
# E1 columns: (column header, sweep stem). The vllm arm is the effect-table
# seed-42 quickstart cells; the other arm is the dedicated qs_e1_* rerun
# sweep produced by the openai variants of the same cells
# (eyas run --cell <ds>/raw --variant openai, ditto cot).
E1_ARMS = (
    ("vllm in-process", "qs"),
    ("OpenAI-compatible API", "qs_e1_openai"),
)
BENCH_TAGS = ("q2b", "q4b")  # bench sweep columns
# Extension column (q38b): Qwen3.8-27B runs the protocol-identical matrix
# cells on the 4 open datasets (12 sweeps / 96 runs) and joins §4 + the README
# scoreboard.
EXT_BENCH_TAGS = ("q38b",)
IVL_TAGS = ("ivl2b", "ivl4b", "ivl8b")  # InternVL family columns
BENCH_SEEDS = ("42", "43", "44")
IVL2B_SEEDS = BENCH_SEEDS
IVL48_SEEDS = BENCH_SEEDS
# Every tag × dataset × seed runs 42/43/44 with all 8 methods on every dataset —
# including the generative-demo pair (opro/ape) on sroie (the matrix spec's
# synthesized KIE columns), so the quickstart DATASET_METHODS whitelist does not
# apply to bench aggregation.


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def load_cell(
    outputs_root: Path, sweep: str, dataset: str, method: str, seed: str | None
) -> dict[str, Any] | None:
    """Load one cell's result.json. Sweep dirs are ``<stem>[_s<seed>]``."""
    seed_part = f"_s{seed}" if seed else ""
    sweep_dir = outputs_root / f"{sweep}_{dataset}{seed_part}"
    if not sweep_dir.is_dir():
        return None
    candidates = sorted(
        d
        for d in sweep_dir.iterdir()
        if d.name.split("-", 1)[-1].endswith(f"_{method}") and (d / "result.json").is_file()
    )
    if not candidates:
        warn(f"{sweep_dir.name}: no result.json for method {method}")
        return None
    if len(candidates) > 1:
        warn(
            f"{sweep_dir.name}: multiple cells for method {method}: {[c.name for c in candidates]}"
        )
    return json.loads((candidates[0] / "result.json").read_text(encoding="utf-8"))


def cell_metric(result: dict[str, Any], cell_id: str, metric: str) -> float:
    """Effect number for one cell: final_eval.metrics first, best.metrics fallback."""
    final = result.get("final_eval")
    if final is not None:
        split = final.get("split")
        if split != "test":
            warn(f"{cell_id}: final_eval.split={split!r} (expected test)")
        return float(final["metrics"][metric])
    warn(f"{cell_id}: no final_eval, falling back to best.metrics (selection split)")
    return float(result["best"]["metrics"][metric])


def aggregate(results: list[float]) -> tuple[float, float]:
    return statistics.fmean(results), statistics.pstdev(results)


# Below this mean parse-fail rate a cell is comparable on accuracy alone; at or
# above it the published cell also states the parsed-only accuracy.
PARSE_FAIL_AUX_MIN = 0.05


def cell_parse_fail(result: dict[str, Any]) -> float | None:
    """The cell's ``parse_fail`` metric (final_eval first, same source rule as accuracy)."""
    final = result.get("final_eval")
    metrics = (final if final is not None else result["best"])["metrics"]
    pf = metrics.get("parse_fail")
    return None if pf is None else float(pf)


def parsed_only_aux(runs: list[dict[str, Any]], values: list[float]) -> str | None:
    """``pf X% → parsed Y%`` for a cell with material format failures, else None.

    Parsed accuracy is computed per seed (accuracy / (1 - parse_fail)) and then
    averaged. A seed that parses nothing is left out rather than dividing by
    zero; a seed without the metric suppresses the auxiliary entirely (mixed
    provenance is not a figure to publish).
    """
    pairs: list[tuple[float, float]] = []
    for run, acc in zip(runs, values, strict=True):
        pf = cell_parse_fail(run["result"])
        if pf is None:
            return None
        if pf >= 1.0:
            continue
        pairs.append((pf, acc / (1.0 - pf)))
    if not pairs:
        return None
    pf_mean = statistics.fmean(pf for pf, _ in pairs)
    if pf_mean < PARSE_FAIL_AUX_MIN:
        return None
    parsed_mean = statistics.fmean(parsed for _, parsed in pairs)
    return f"pf {100.0 * pf_mean:.1f}% → parsed {100.0 * parsed_mean:.1f}%"


# ---------------------------------------------------------------- formatting


def fmt_frac(mean: float, sd: float, places: int = 3) -> str:
    return f"{mean:.{places}f} ± {sd:.{places}f}"


def fmt_pct(mean: float, sd: float, n: int) -> str:
    if n == 1:
        return f"{100.0 * mean:.1f}% (n=1)"
    return f"{100.0 * mean:.1f}% ±{100.0 * sd:.1f}% (n={n})"


def cost_line(result: dict[str, Any]) -> str:
    cost = result["total_cost"]
    if cost["requests"] == 0:
        return "req (resume: billed on first attempt)"
    return f"req {cost['requests']} (judge {cost['judge_requests']}) tok {cost['input_tokens']}/{cost['output_tokens']}"


def fmt_k(tokens: int) -> str:
    return f"{tokens / 1000.0:.1f}k"


# ---------------------------------------------------------------- quickstart


class QuickstartAgg:
    """Quickstart effect tables and cost profile (Qwen3.5-9B, vllm arm)."""

    def __init__(self, outputs_root: Path) -> None:
        self.root = outputs_root

    def cells(self, dataset: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for method in DATASET_METHODS[dataset]:
            runs: list[dict[str, Any]] = []
            for seed in QS_MODELS:
                result = load_cell(self.root, "qs", dataset, method, seed)
                if result is None:
                    warn(f"qs_{dataset}_s{seed}: missing cell {method}")
                    continue
                runs.append({"seed": seed, "result": result})
            out[method] = runs
        return out

    def effect(self, dataset: str) -> str:
        metric = PRIMARY_METRIC[dataset]
        cells = self.cells(dataset)
        means: dict[str, tuple[float, float, int]] = {}
        for method, runs in cells.items():
            values = [
                cell_metric(r["result"], f"qs_{dataset}_s{r['seed']}/{method}", metric)
                for r in runs
            ]
            mean, sd = aggregate(values)
            means[method] = (mean, sd, len(values))
        raw_mean = means["raw"][0]
        best_method = max((m for m in means if m != "raw"), key=lambda m: means[m][0])
        # bold = strictly best row in the table, raw included — when no
        # optimizer beats the seed prompt, the honest highlight is raw itself
        if raw_mean > means[best_method][0]:
            best_method = "raw"
        lines = [f"| Method | test {metric} | Δ vs raw (pp) |", "| --- | --- | --- |"]
        for method in DATASET_METHODS[dataset]:
            mean, sd, n = means[method]
            delta = "" if method == "raw" else f"{100.0 * (mean - raw_mean):+.1f}"
            strict = ""
            if dataset == "sroie":
                strict_vals = [
                    float(r["result"]["final_eval"]["metrics"]["strict_acc"]) for r in cells[method]
                ]
                strict = f" (strict {statistics.fmean(strict_vals):.3f})"
            body = fmt_frac(mean, sd) + strict
            if method == best_method:
                body = f"**{body}**"
            lines.append(f"| {method} | {body} | {delta} |")
        # Cross-seed range facts for the section preamble.
        ranges = []
        for method, runs in cells.items():
            values = [cell_metric(r["result"], "range-scan", metric) for r in runs]
            if len(values) > 1:
                ranges.append((max(values) - min(values), method))
        widest = max(ranges) if ranges else (0.0, "-")
        lines.append("")
        lines.append(f"<!-- widest cross-seed range: {100 * widest[0]:.1f}pp ({widest[1]}) -->")
        return "\n".join(lines)

    def cost(self) -> str:
        """Cost profile: seed-42 vllm cells, format `wall / reqs+judge / input-tokens`."""
        lines = [
            f"| Method | {' | '.join(DATASETS)} |",
            "| --- | " + " | ".join(["---"] * len(DATASETS)) + " |",
        ]
        for method in METHODS:
            row = [f"| {method} "]
            for dataset in DATASETS:
                if method not in DATASET_METHODS[dataset]:
                    row.append("| — ")
                    continue
                result = load_cell(self.root, "qs", dataset, method, "42")
                if result is None:
                    row.append("| MISSING ")
                    continue
                cost = result["total_cost"]
                if cost["requests"] == 0:
                    row.append("| *(resume)* ")
                    continue
                row.append(
                    f"| {cost['wall_seconds']:.0f}s / {cost['requests']}+{cost['judge_requests']} / {fmt_k(cost['input_tokens'])} "
                )
            lines.append("".join(row) + "|")
        # final_eval pass request counts (the separately accounted test pass).
        # Report the first seed-42 cell of the dataset whose test pass was actually
        # billed (resume-replayed cells bill zero).
        lines.append("")
        facts = {}
        for ds in DATASETS:
            for probe in ("opro", "spo", "cot", "raw"):
                if probe not in DATASET_METHODS[ds]:
                    continue
                r = load_cell(self.root, "qs", ds, probe, "42")
                if r and r["final_eval"]["cost"]["requests"] > 0:
                    facts[ds] = (
                        probe,
                        r["final_eval"]["cost"]["requests"],
                        r["final_eval"]["cost"]["judge_requests"],
                    )
                    break
        lines.append(f"<!-- final_eval test pass (first billed seed-42 cell): {facts} -->")
        return "\n".join(lines)


# ------------------------------------------------------------------- e1 parity


def e1_cell(outputs_root: Path, sweep: str, dataset: str, method: str) -> dict[str, Any] | None:
    """Load one E1 parity cell's result.json.

    Same NN-prefixed discovery convention as :func:`load_cell`, but the e1
    sweeps' run names carry the arm suffix (``quickstart_pets_raw_openai``),
    so the method is matched as a name token rather than a plain ``_raw``
    suffix. Absent sweep dir / cell stays silent — the caller emits the one
    per-missing-cell stderr line.
    """
    sweep_dir = outputs_root / f"{sweep}_{dataset}_s{E1_SEED}"
    if not sweep_dir.is_dir():
        return None
    candidates = sorted(
        d
        for d in sweep_dir.iterdir()
        if (f"_{method}_" in d.name or d.name.endswith(f"_{method}"))
        and (d / "result.json").is_file()
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        warn(
            f"{sweep_dir.name}: multiple cells for method {method}: {[c.name for c in candidates]}"
        )
    return json.loads((candidates[0] / "result.json").read_text(encoding="utf-8"))


def e1_parity(outputs_root: Path) -> str:
    """E1 parity: raw / cot x two backend arms x pets/cub/sroie, seed 42."""
    lines = [
        "| Dataset | Metric | Arm | " + " | ".join(name for name, _ in E1_ARMS) + " |",
        "| --- | --- | --- | " + " | ".join(["---"] * len(E1_ARMS)) + " |",
    ]
    for dataset in ("pets", "cub", "sroie"):
        metric = PRIMARY_METRIC[dataset]
        for method in E1_METHODS:
            row = [f"| {dataset} | {metric} | {method} "]
            for _arm, stem in E1_ARMS:
                cell_id = f"{stem}_{dataset}_s{E1_SEED}/{method}"
                result = e1_cell(outputs_root, stem, dataset, method)
                if result is None:
                    warn(f"{cell_id}: missing cell")
                    row.append("| — ")
                    continue
                # same extraction rule as the rest of the script (final_eval first)
                text = f"{cell_metric(result, cell_id, metric):.4f}"
                if dataset == "sroie":
                    source = (
                        result["final_eval"]
                        if result.get("final_eval") is not None
                        else result["best"]
                    )
                    # published §4 convention: the strict label is spelled out on
                    # the first sroie row only, parenthetical (bare) below it
                    label = "strict " if method == E1_METHODS[0] else ""
                    text += f" ({label}{float(source['metrics']['strict_acc']):.2f})"
                row.append(f"| {text} ")
            lines.append("".join(row) + "|")
    return "\n".join(lines)


# ---------------------------------------------------------------------- bench


class BenchAgg:
    """Bench effect sweeps (q2b/q4b/q9b + extension full sweeps) and the InternVL
    family."""

    def __init__(self, outputs_root: Path, family: str = "matrix") -> None:
        self.root = outputs_root
        self.family = family

    def methods(self, dataset: str) -> tuple[str, ...]:
        return METHODS

    def runs(
        self, stem: str, dataset: str, method: str, seeds: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        out = []
        for seed in seeds:
            result = load_cell(self.root, stem, dataset, method, seed)
            if result is not None:
                out.append({"seed": seed, "result": result})
        return out

    def _column(
        self, runs: list[dict[str, Any]], dataset: str, method: str, label: str
    ) -> tuple[str, float | None, str]:
        metric = PRIMARY_METRIC[dataset]
        if not runs:
            return "—", None, label
        values = [
            cell_metric(r["result"], f"{label}_{dataset}/{method}/s{r['seed']}", metric)
            for r in runs
        ]
        mean, sd = aggregate(values)
        text = fmt_pct(mean, sd, len(runs))
        aux = parsed_only_aux(runs, values)
        if aux is not None:
            text += f"<br>{aux}"
        billed = [r["result"] for r in runs if r["result"]["total_cost"]["requests"] > 0]
        if billed:
            cost = statistics.fmean(r["total_cost"]["requests"] for r in billed)
            judge = statistics.fmean(r["total_cost"]["judge_requests"] for r in billed)
            tin = statistics.fmean(r["total_cost"]["input_tokens"] for r in billed)
            tout = statistics.fmean(r["total_cost"]["output_tokens"] for r in billed)
            line = f"req {cost:.0f} (judge {judge:.0f}) tok {tin:.0f}/{tout:.0f}"
            if len(billed) < len(runs):
                line += f" [cost n={len(billed)}]"
        else:
            line = "req (resume: billed on first attempt)"
        return f"{text}<br>{line}", mean, label

    def _table(self, dataset: str, columns: list[tuple[str, str, tuple[str, ...] | None]]) -> str:
        """columns: (column header, stem, seeds); every column bills identically."""
        header = (
            "| method | "
            + " | ".join(f"{name} {PRIMARY_METRIC[dataset]}" for name, _, _ in columns)
            + " "
        )
        header += "| " + " | ".join(f"Δ {name} (pp)" for name, _, _ in columns) + " |"
        lines = [header, "|---" * (1 + 2 * len(columns)) + "|"]
        texts: dict[str, list[str]] = {}
        means: dict[tuple[int, str], float | None] = {}
        for i, (name, stem, seeds) in enumerate(columns):
            for method in self.methods(dataset):
                runs = self.runs(stem, dataset, method, seeds or ())
                text, mean, _ = self._column(runs, dataset, method, name)
                means[(i, method)] = mean
                texts.setdefault(method, []).append(text)
        for method in self.methods(dataset):
            row = [f"| {method} "]
            for i in range(len(columns)):
                row.append(f"| {texts[method][i]} ")
            for i in range(len(columns)):
                mean, base = means[(i, method)], means[(i, "raw")]
                cell = (
                    ""
                    if method == "raw" or mean is None or base is None
                    else f"{100 * (mean - base):+.1f}"
                )
                row.append(f"| {cell} ")
            lines.append("".join(row) + "|")
        return "\n".join(lines)


# ------------------------------------------------------------------ method cards


def method_cards(outputs_root: Path, dataset: str) -> str:
    """Method-card evidence lines: one per quickstart method.

    A pure formatting layer over :class:`QuickstartAgg` — the same cell loading,
    extraction rule (``final_eval.metrics``, test split) and mean ± population-σ
    aggregation the ``quickstart-effect`` tables use, so the cards can never
    drift from the quickstart tables. One bullet per method: primary metric with
    n over the
    seeds present, Δ vs the raw row in pp (raw itself carries none), and the
    seed-42 cost summary (optimization loop billed requests + judge; the shared
    final_eval test pass accounted separately).
    """
    metric = PRIMARY_METRIC[dataset]
    cells = QuickstartAgg(outputs_root).cells(dataset)
    cells = {m: runs for m, runs in cells.items() if runs}
    if "raw" not in cells:
        warn(f"qs_{dataset}: no raw baseline — no Δ anchor, emitting no cards")
        return ""
    means: dict[str, tuple[float, float, int]] = {}
    for method, runs in cells.items():
        values = [
            cell_metric(r["result"], f"qs_{dataset}_s{r['seed']}/{method}", metric) for r in runs
        ]
        mean, sd = aggregate(values)
        means[method] = (mean, sd, len(values))
    raw_mean = means["raw"][0]
    lines = [
        f"<!-- method-cards {dataset}: {metric}, Qwen3.5-9B quickstart tier, "
        f"seeds {'/'.join(QS_MODELS)}; loop cost from the seed-42 cell -->"
    ]
    for method in DATASET_METHODS[dataset]:
        if method not in means:
            continue
        mean, sd, n = means[method]
        line = f"- `{method}` — test {metric} {fmt_pct(mean, sd, n)}"
        if method != "raw":
            line += f" · Δ {100.0 * (mean - raw_mean):+.1f} pp vs raw"
        seed42 = next((r["result"] for r in cells[method] if r["seed"] == "42"), None)
        if seed42 is not None:
            loop = seed42["total_cost"]
            test = seed42["final_eval"]["cost"]
            loop_txt = (
                "req (resume: billed on first attempt)"
                if loop["requests"] == 0
                else f"req {loop['requests']} (judge {loop['judge_requests']})"
            )
            line += (
                f" · cost {loop_txt}"
                f" · test pass req {test['requests']} (judge {test['judge_requests']})"
            )
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------- headline


def bench_columns(family: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """Bench / README columns: the three published Qwen3.5 tags plus the
    extension tag (q38b)."""
    tags = (*BENCH_TAGS, "q9b", *EXT_BENCH_TAGS)
    return [(t, f"{family}_{t}", BENCH_SEEDS) for t in tags]


def headline(outputs_root: Path, family: str = "matrix") -> str:
    """README scoreboard: per dataset × model column, raw -> best optimized cell.

    The published run set (matrix) covers all seven model columns (six
    tags of the Qwen/InternVL families + the Qwen3.8-27B extension) over the
    business four plus nextqa and gsm8k; every column is a full optimization
    sweep over 3 seeds, so README and the bench tables read one source.
    A column/dataset with missing cells degrades to "—" (no raw baseline) or
    "raw → —" (no optimized cell) with a stderr warning, instead of a silent
    number.
    """
    columns = [(name, stem) for name, stem, _ in bench_columns(family)]
    columns += [(t, f"{family}_{t}") for t in IVL_TAGS]
    agg = BenchAgg(outputs_root, family=family)
    lines = ["| dataset | " + " | ".join(name for name, _ in columns) + " |"]
    lines.append("| --- |" + " --- |" * len(columns))
    for dataset in BENCH_DATASETS:
        row = [f"| {dataset} "]
        for label, stem in columns:
            stats: dict[str, float] = {}
            for method in agg.methods(dataset):
                runs = agg.runs(stem, dataset, method, BENCH_SEEDS)
                if not runs:
                    continue
                metric = PRIMARY_METRIC[dataset]
                values = [
                    cell_metric(r["result"], f"{label}_{dataset}/{method}", metric) for r in runs
                ]
                stats[method] = statistics.fmean(values)
            raw = stats.get("raw")
            optimized = {m: v for m, v in stats.items() if m != "raw"}
            if raw is None:
                # missing column: explicit em dash, never a silent number
                warn(f"{stem}_{dataset}: no raw baseline yet")
                row.append("| — ")
                continue
            if not optimized:
                row.append(f"| {100 * raw:.1f} → — ")
                continue
            best_method = max(optimized, key=lambda m: optimized[m])
            best = optimized[best_method]
            tag = best_method if best > raw else "saturated"
            row.append(f"| {100 * raw:.1f} → **{100 * best:.1f}** ({tag}) ")
        lines.append("".join(row) + "|")
    return "\n".join(lines)


# ------------------------------------------------------------------------- cli


def resolve_datasets(value: str, datasets: tuple[str, ...]) -> tuple[str, ...]:
    return datasets if value == "all" else (value,)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="root of the sweep run directories (default: outputs)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_dataset(p: argparse.ArgumentParser, datasets: tuple[str, ...]) -> None:
        p.add_argument("--dataset", default="all", choices=("all", *datasets))

    def add_family(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--sweep",
            default="matrix",
            help="sweep stem family (default: matrix, the published run set)",
        )

    p = sub.add_parser("quickstart-effect", help="per-dataset E2 tables (mean ± population σ)")
    add_dataset(p, DATASETS)
    p = sub.add_parser("quickstart-cost", help="E3 cost profile (seed 42)")
    sub.add_parser("e1-parity", help="E1 backend-parity table (raw/cot × 2 arms, seed 42)")
    p = sub.add_parser(
        "bench-effect",
        help="per-dataset bench tables (q2b/q4b/q9b + extension columns on "
        "full-coverage sweeps, cost in cell)",
    )
    add_dataset(p, BENCH_DATASETS)
    add_family(p)
    p = sub.add_parser(
        "internvl-effect",
        help="InternVL family tables (ivl2b/ivl4b/ivl8b, cost in cell)",
    )
    add_dataset(p, BENCH_DATASETS)
    add_family(p)
    p = sub.add_parser("headline", help="README bench scoreboard (raw → best per model column)")
    p.add_argument("--sweep", default="matrix", help="sweep stem family (default: matrix)")
    p = sub.add_parser(
        "method-cards",
        help="per-method evidence lines (quickstart E2 tier: "
        "mean ± population σ, Δ vs raw, seed-42 cost)",
    )
    add_dataset(p, DATASETS)

    args = parser.parse_args(argv)
    root = Path(args.outputs_root)
    if not root.is_dir():
        print(f"error: outputs root {root} not found", file=sys.stderr)
        return 2

    if args.command == "quickstart-effect":
        for ds in resolve_datasets(args.dataset, DATASETS):
            print(f"### quickstart {ds} ({PRIMARY_METRIC[ds]})\n")
            print(QuickstartAgg(root).effect(ds))
            print()
    elif args.command == "quickstart-cost":
        print(QuickstartAgg(root).cost())
    elif args.command == "e1-parity":
        print(e1_parity(root))
    elif args.command == "bench-effect":
        agg = BenchAgg(root, family=args.sweep)
        for ds in resolve_datasets(args.dataset, BENCH_DATASETS):
            print(f"### bench {ds} ({PRIMARY_METRIC[ds]})\n")
            print(agg._table(ds, bench_columns(args.sweep)))
            print()
    elif args.command == "internvl-effect":
        agg = BenchAgg(root, family=args.sweep)
        for ds in resolve_datasets(args.dataset, BENCH_DATASETS):
            print(f"### internvl {ds} ({PRIMARY_METRIC[ds]})\n")
            print(
                agg._table(
                    ds,
                    [
                        ("ivl2b", f"{args.sweep}_ivl2b", IVL2B_SEEDS),
                        ("ivl4b", f"{args.sweep}_ivl4b", IVL48_SEEDS),
                        ("ivl8b", f"{args.sweep}_ivl8b", IVL48_SEEDS),
                    ],
                )
            )
            print()
    elif args.command == "headline":
        print(headline(root, family=args.sweep))
    elif args.command == "method-cards":
        for ds in resolve_datasets(args.dataset, DATASETS):
            print(method_cards(root, ds))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
