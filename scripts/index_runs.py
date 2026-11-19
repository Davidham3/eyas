"""Index every Eyas run dir into flat queryable tables (CSV out).

The runs written by this version share one
observation contract — result.json / run_config.json / rounds / per-sample
record store — so downstream analysis should never hand-roll directory
traversal again. Point this at one or more roots and it emits four CSVs:

- ``runs.csv``       one row per run (identity + config facts + best/final_eval
                     metrics flattened to columns + the total_cost bill);
- ``rounds.csv``     one row per optimizer round (n_candidates / n_requests /
                     full per-round cost / elapsed_sec / best_so_far curve
                     point); runs without those fields keep the columns empty;
- ``candidates.csv`` one row per evaluated candidate (archive leaderboard,
                     rank / prompt_hash / prompt text — the prompt-evolution
                     corpus);
- ``samples.csv``    one row per (prompt x sample x split) outcome straight
                     from the run's sqlite record store (scores_json /
                     correct / prediction). Schema-3 stores decode with the
                     store's own type-tag semantics: str answers export as
                     plain text, non-str (KIE-style) as JSON text, None as
                     empty — the tag never leaks into the CSV. Older (or
                     unversioned) stores carry plain untagged predictions
                     and export verbatim (NULL stays empty). Pass
                     --no-predictions to drop the prediction column (keeps
                     the file light for full-matrix indexing; the text always
                     stays in the run dir).

Metrics dicts are flattened to ``<prefix><key>`` columns over a first-pass
key union; anything unexpected also survives in the adjacent ``*_json``
column. Usage:

    python scripts/index_runs.py outputs/matrix_q2b_pets_s42 ... -o analysis/
    python scripts/index_runs.py outputs -o analysis/ --no-predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from eyas.engine.evaluation import _decode_prediction


def _find_run_dirs(roots: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        if (root / "result.json").is_file():
            dirs.append(root)
            continue
        for p in sorted(root.rglob("result.json")):
            dirs.append(p.parent)
    # dedupe preserving order
    seen: set[Path] = set()
    out = []
    for d in dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def _flatten(row: dict[str, Any], prefix: str, flat: dict[str, str], raw: str | None = None):
    for k, v in row.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[f"{prefix}{k}"] = "" if v is None else v
        else:
            flat[f"{prefix}{k}"] = json.dumps(v, ensure_ascii=False)
    if raw is not None:
        flat[f"{prefix.rstrip('_')}_json"] = raw


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _run_row(run_dir: Path) -> dict[str, str]:
    result = _read_json(run_dir / "result.json") or {}
    cfg = _read_json(run_dir / "run_config.json") or {}
    status = _read_json(run_dir / "status.json") or {}
    run = cfg.get("run", {})
    task = cfg.get("task", {})
    backend = cfg.get("backend", {})
    strategy = cfg.get("strategy", {})
    params = task.get("params", {})
    data_dir = str(params.get("data_dir", ""))
    flat: dict[str, str] = {
        "run_dir": str(run_dir),
        "sweep": run_dir.parent.name,
        "name": result.get("name") or run.get("name") or run_dir.name,
        "strategy": strategy.get("name", ""),
        "task": task.get("name", ""),
        "dataset": Path(data_dir).parent.name if data_dir else "",
        "model": Path(str(backend.get("params", {}).get("model_path", ""))).name,
        "protocol": cfg.get("protocol", {}).get("name", "identity"),
        "eval_model": cfg.get("eval_model", {}).get("name", "absolute"),
        "budget": cfg.get("budget", {}).get("name", ""),
        "budget_params": json.dumps(cfg.get("budget", {}).get("params", {}), ensure_ascii=False),
        "seed": run.get("seed", ""),
        "trace": bool(run.get("trace", False)),
        "phase": status.get("phase", ""),
        "best_prompt_hash": (result.get("best") or {}).get("prompt_hash", ""),
        "best_prompt_chars": len(((result.get("best") or {}).get("text")) or ""),
        "n_rounds": len(result.get("history") or []),
        "n_ranked": len(result.get("ranked") or []),
    }
    _flatten(result.get("metrics") or {}, "final__", flat)
    _flatten((result.get("final_eval") or {}).get("metrics") or {}, "final_eval__", flat)
    _flatten(result.get("total_cost") or {}, "total__", flat)
    return flat


def _round_rows(run_dir: Path) -> list[dict[str, str]]:
    result = _read_json(run_dir / "result.json") or {}
    name = result.get("name") or run_dir.name
    rows = []
    for h in result.get("history") or []:
        row: dict[str, str] = {
            "run": name,
            "run_dir": str(run_dir),
            "round": h.get("round", ""),
            "n_candidates": h.get("n_candidates", ""),
            "n_requests": h.get("n_requests", ""),
            "elapsed_sec": h.get("elapsed_sec", ""),
        }
        _flatten(h.get("cost") or {}, "cost__", row)
        leader = h.get("best_so_far")
        if isinstance(leader, dict):
            row["best_so_far__prompt_hash"] = leader.get("prompt_hash", "")
            _flatten(leader.get("metrics") or {}, "best_so_far__", row)
        rows.append(row)
    return rows


def _candidate_rows(run_dir: Path) -> list[dict[str, str]]:
    result = _read_json(run_dir / "result.json") or {}
    name = result.get("name") or run_dir.name
    rows = []
    for rank, cand in enumerate(result.get("ranked") or []):
        row: dict[str, str] = {
            "run": name,
            "run_dir": str(run_dir),
            "rank": rank,
            "prompt_hash": cand.get("prompt_hash", ""),
            "n_chars": len(cand.get("text") or ""),
            "text": (cand.get("text") or "").replace("\r", ""),
        }
        _flatten(cand.get("metrics") or {}, "", row)
        rows.append(row)
    return rows


def _prediction_text(encoded: str | None) -> str:
    """Decode the store's type-tagged prediction column into CSV text.

    Reuses the store's decode semantics (tag-driven, fail fast on a corrupt
    row — no duplicated decode logic): str -> the text itself, non-str
    (KIE-style list/dict) -> JSON text without the tag (the historical CSV
    shape), None -> empty.
    """
    decoded = _decode_prediction(encoded)
    if decoded is None:
        return ""
    if isinstance(decoded, str):
        return decoded
    return json.dumps(decoded, ensure_ascii=False)


def _store_schema(conn: sqlite3.Connection) -> int:
    """Read the store's schema stamp (``meta.schema``) read-only.

    A missing meta table / stamp (the oldest unversioned stores) or an
    unreadable one counts as 0 — the historical shape. Never writes.
    """
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
    except sqlite3.DatabaseError:
        return 0
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def _sample_rows(run_dir: Path, with_predictions: bool) -> list[dict[str, str]]:
    db = run_dir / "cache" / "eyas.sqlite"
    if not db.is_file():
        return []
    name = (
        (_read_json(run_dir / "result.json") or {}).get("name")
        or (_read_json(run_dir / "status.json") or {}).get("name")
        or run_dir.name
    )
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = []
    try:
        # schema >= 3 stores type-tag predictions; older (or unversioned)
        # stores carry plain historical text — export it verbatim instead
        # of misreading it as a corrupt tag.
        tagged = _store_schema(conn) >= 3
        cur = conn.execute(
            "SELECT namespace, prompt_id, sample_id, split, prediction, correct, scores_json, error"
            " FROM outcomes"
        )
        for ns, pid, sid, split, prediction, correct, scores_json, error in cur:
            row = {
                "run": name,
                "run_dir": str(run_dir),
                "namespace": ns,
                "prompt_hash": pid,
                "sample_id": sid,
                "split": split,
                "correct": correct if correct is not None else "",
                "scores_json": scores_json,
                "error": error or "",
            }
            if with_predictions:
                if tagged:
                    text = _prediction_text(prediction)
                else:
                    text = prediction if prediction is not None else ""
                row["prediction"] = text.replace("\r", "")
            rows.append(row)
    except (sqlite3.DatabaseError, RuntimeError) as exc:
        # a mid-run/corrupt store (bad sqlite pages, or a corrupt schema-3
        # prediction tag) must not abort the whole index — but the truncation
        # of this store's rows must not be silent either (a silently incomplete
        # analysis export is worse than a loud partial one)
        print(f"[index] WARNING: {db}: rows truncated at corruption ({exc})", file=sys.stderr)
        return rows
    finally:
        conn.close()
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path, help="run dirs or roots containing them")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output directory for CSVs")
    parser.add_argument(
        "--no-predictions", action="store_true", help="drop prediction text from samples.csv"
    )
    args = parser.parse_args(argv)

    run_dirs = _find_run_dirs(args.roots)
    if not run_dirs:
        print("no run dirs (result.json) found", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [_run_row(d) for d in run_dirs]
    rounds: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    samples: list[dict[str, str]] = []
    for d in run_dirs:
        rounds.extend(_round_rows(d))
        candidates.extend(_candidate_rows(d))
        samples.extend(_sample_rows(d, with_predictions=not args.no_predictions))

    _write_csv(args.out / "runs.csv", runs)
    _write_csv(args.out / "rounds.csv", rounds)
    _write_csv(args.out / "candidates.csv", candidates)
    _write_csv(args.out / "samples.csv", samples)
    print(
        f"indexed {len(run_dirs)} runs -> runs={len(runs)} rounds={len(rounds)} "
        f"candidates={len(candidates)} samples={len(samples)} in {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
