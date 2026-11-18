"""Cross-run summary (``eyas report``): reads result.json + run_config.json
from run dirs.

Markdown / CSV dual format, dynamic metric columns (multi-metric), seed, best
prompt, cost (total requests + token totals + judge-channel split + breakdown
by sampling tier). Evaluation semantics live inside the EvalModel, so cost is
reported as total_requests / judge_requests / tokens + cost_by_tier.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunSummary", "summarize_run", "report_runs", "render_markdown", "render_csv"]


@dataclass
class RunSummary:
    run_dir: str
    name: str
    strategy: str
    task: str
    seed: int | None
    best_prompt: str
    metrics: dict[str, float] = field(default_factory=dict)
    requests: int = 0
    judge_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_by_tier: dict[str, int] = field(default_factory=dict)
    wall_seconds: float = 0.0

    @property
    def cost_tier_str(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.cost_by_tier.items())) or "-"


def _identity_fields(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """What the run calls itself: display name, strategy, task and seed."""
    run_info = config.get("run", {})
    return {
        "name": run_info.get("name", run_dir.name),
        "strategy": str(config.get("strategy", {}).get("name", "?")),
        "task": str(config.get("task", {}).get("name", "?")),
        "seed": run_info.get("seed"),
    }


def _best_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Best prompt + the reported metric columns: best's own metrics merged
    with the final-eval numbers."""
    best = result.get("best") or {}
    metrics = dict(best.get("metrics") or {})
    final_eval = result.get("final_eval") or {}
    # column prefix follows the actual reporting split (run.final_eval.split),
    # not a hardcoded "test_" — a val-split final eval must not read as test
    split = str(final_eval.get("split") or "test")
    for key, value in (final_eval.get("metrics") or {}).items():
        metrics[f"{split}_{key}"] = value
    return {"best_prompt": best.get("text", ""), "metrics": metrics}


def _cost_fields(result: dict[str, Any]) -> dict[str, Any]:
    """The cost section: total-cost counters, per-tier split and wall time."""
    total = result.get("total_cost", {})
    return {
        "requests": int(total.get("requests", 0)),
        "judge_requests": int(total.get("judge_requests", 0)),
        "input_tokens": int(total.get("input_tokens", 0)),
        "output_tokens": int(total.get("output_tokens", 0)),
        "cost_by_tier": dict(result.get("cost_by_tier") or {}),
        "wall_seconds": float(total.get("wall_seconds", 0.0)),
    }


def summarize_run(run_dir: str | Path) -> RunSummary:
    """Read one run dir's result.json + run_config.json; each section of the
    summary is filled by its own extractor, this only fixes the sections."""
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    config_path = run_dir / "run_config.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"not a valid run dir (missing result.json): {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    return RunSummary(
        run_dir=str(run_dir),
        **_identity_fields(config, run_dir),
        **_best_fields(result),
        **_cost_fields(result),
    )


def _metric_columns(summaries: list[RunSummary]) -> list[str]:
    seen: dict[str, None] = {}
    for s in summaries:
        for key in s.metrics:
            seen.setdefault(key, None)
    return sorted(seen)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(summaries: list[RunSummary]) -> str:
    metric_keys = _metric_columns(summaries)
    header = [
        "run",
        "strategy",
        "task",
        "seed",
        *metric_keys,
        "requests",
        "judge_reqs",
        "in_tok",
        "out_tok",
        "by_tier",
        "wall_s",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for s in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    s.name,
                    s.strategy,
                    s.task,
                    _fmt(s.seed),
                    *(_fmt(s.metrics.get(k)) for k in metric_keys),
                    str(s.requests),
                    str(s.judge_requests),
                    str(s.input_tokens),
                    str(s.output_tokens),
                    s.cost_tier_str,
                    _fmt(s.wall_seconds),
                ]
            )
            + " |"
        )
    lines += ["", "## Best prompts", ""]
    for s in summaries:
        lines += [f"### {s.name}", "", "```\n" + s.best_prompt + "\n```", ""]
    return "\n".join(lines)


def render_csv(summaries: list[RunSummary]) -> str:
    metric_keys = _metric_columns(summaries)
    header = [
        "run",
        "strategy",
        "task",
        "seed",
        *metric_keys,
        "requests",
        "judge_requests",
        "input_tokens",
        "output_tokens",
        "cost_by_tier",
        "wall_s",
        "best_prompt",
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for s in summaries:
        writer.writerow(
            [
                s.run_dir,
                s.strategy,
                s.task,
                s.seed if s.seed is not None else "",
                *(s.metrics.get(k, "") for k in metric_keys),
                s.requests,
                s.judge_requests,
                s.input_tokens,
                s.output_tokens,
                s.cost_tier_str,
                s.wall_seconds,
                s.best_prompt,
            ]
        )
    return buf.getvalue()


def report_runs(run_dirs: list[Any], *, fmt: str = "md") -> str:
    summaries = [summarize_run(d) for d in run_dirs]
    if not summaries:
        raise ValueError("report requires at least one run dir")
    if fmt == "csv":
        return render_csv(summaries)
    if fmt == "md":
        return render_markdown(summaries)
    raise ValueError(f"unknown format: {fmt!r} (supported: md / csv)")
