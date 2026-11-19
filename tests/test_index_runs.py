"""scripts/index_runs.py contract: run dirs -> flat queryable tables.

Synthetic run dirs pin the four tables (runs / rounds / candidates / samples)
and the honesty rules of the indexer: pre-field runs keep the new columns
ABSENT (never fabricated), the sqlite store is opened read-only — indexing
must never mutate a run dir (the store is the checkpoint) — and samples.csv
predictions are decoded with the store's type-tag semantics (the tag never
leaks into the CSV).
"""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _ROOT / "scripts" / "index_runs.py"


def _make_run_dir(root: Path, name: str, *, with_curve: bool) -> Path:
    run_dir = root / "wave_x" / name
    (run_dir / "cache").mkdir(parents=True)
    history = [
        {
            "round": 0,
            "n_candidates": 1,
            "n_requests": 10,
            "cost": {
                "requests": 10,
                "input_tokens": 100,
                "output_tokens": 5,
                "wall_seconds": 1.0,
                "judge_requests": 0,
                "judge_input_tokens": 0,
                "judge_output_tokens": 0,
            },
            "elapsed_sec": 2.0,
        }
    ]
    if with_curve:
        history[0]["best_so_far"] = {"prompt_hash": "abc123", "metrics": {"accuracy": 0.5}}
    result = {
        "name": name,
        "metrics": {"accuracy": 0.9, "parse_fail": 0.0},
        "final_eval": {"metrics": {"accuracy": 0.8}},
        "total_cost": history[0]["cost"],
        "best": {"prompt_hash": "abc123", "text": "do the thing", "metrics": {"accuracy": 0.9}},
        "ranked": [
            {"prompt_hash": "abc123", "text": "do the thing", "metrics": {"accuracy": 0.9}},
            {"prompt_hash": "def456", "text": "older prompt", "metrics": {"accuracy": 0.5}},
        ],
        "history": history,
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": {"name": name, "seed": 42, "trace": True},
                "task": {"name": "mcq", "params": {"data_dir": "datasets/demo/foo/data"}},
                "backend": {"name": "vllm", "params": {"model_path": "/models/Qwen3.5-2B"}},
                "strategy": {"name": "hill_climb", "params": {"rounds": 3}},
                "budget": {"name": "flat", "params": {}},
            }
        ),
        encoding="utf-8",
    )
    conn = sqlite3.connect(run_dir / "cache" / "eyas.sqlite")
    conn.execute(
        "CREATE TABLE outcomes (namespace TEXT, prompt_id TEXT, sample_id TEXT, split TEXT,"
        " prediction TEXT, correct INTEGER, scores_json TEXT, error TEXT,"
        " PRIMARY KEY (namespace, prompt_id, sample_id, split))"
    )
    # schema-3 stores stamp meta.schema at init (SqliteRecordStore shape)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema', '3')")
    # prediction column carries the store's type tag (schema 3: "s:" str)
    conn.execute(
        "INSERT INTO outcomes VALUES (?, 'abc123', 's0', 'train', 's:B', 1, '{\"accuracy\": 1.0}', NULL)",
        (name,),
    )
    conn.commit()
    conn.close()
    return run_dir


def _read(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _index(root: Path, out: Path, *extra: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(root), "-o", str(out), *extra],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_index_tables(tmp_path):
    _make_run_dir(tmp_path, "run_new", with_curve=True)
    _make_run_dir(tmp_path, "run_old", with_curve=False)
    out = tmp_path / "analysis"
    _index(tmp_path, out)

    runs = {r["name"]: r for r in _read(out / "runs.csv")}
    assert set(runs) == {"run_new", "run_old"}
    row = runs["run_new"]
    assert row["strategy"] == "hill_climb"
    assert row["dataset"] == "foo"
    assert row["model"] == "Qwen3.5-2B"
    assert row["final__accuracy"] == "0.9"
    assert row["final_eval__accuracy"] == "0.8"
    assert row["total__requests"] == "10"
    assert row["trace"] == "True"

    rounds = _read(out / "rounds.csv")
    new = next(r for r in rounds if r["run"] == "run_new")
    old = next(r for r in rounds if r["run"] == "run_old")
    assert new["best_so_far__prompt_hash"] == "abc123"
    assert new["best_so_far__accuracy"] == "0.5"
    # pre-field runs leave the curve columns EMPTY, never fabricate a point
    assert old["best_so_far__prompt_hash"] == ""

    cands = [c for c in _read(out / "candidates.csv") if c["run"] == "run_new"]
    assert [c["rank"] for c in cands] == ["0", "1"]
    assert cands[0]["prompt_hash"] == "abc123"

    samples = _read(out / "samples.csv")
    assert len(samples) == 2  # one per run
    s = next(x for x in samples if x["run"] == "run_new")
    assert s["sample_id"] == "s0" and s["correct"] == "1" and s["split"] == "train"
    assert s["prediction"] == "B"


def test_samples_predictions_decode_type_tags(tmp_path: Path) -> None:
    """samples.csv exports DECODED predictions: the store's type tag never
    leaks into the CSV (naive gold-vs-prediction comparison stays valid).
    str -> plain text, non-str (KIE-style) -> JSON text without the tag,
    None -> empty."""
    run_dir = _make_run_dir(tmp_path, "run_tags", with_curve=False)
    kie = [{"field": "name", "value": "x"}]
    conn = sqlite3.connect(run_dir / "cache" / "eyas.sqlite")
    conn.executemany(
        "INSERT INTO outcomes VALUES (?, 'abc123', ?, 'train', ?, 1, '{}', NULL)",
        [
            ("run_tags", "s_kie", "j:" + json.dumps(kie, ensure_ascii=False)),
            ("run_tags", "s_none", None),
            ("run_tags", "s_json_str", 's:{"a":1}'),
            ("run_tags", "s_empty", "s:"),
        ],
    )
    conn.commit()
    conn.close()
    out = tmp_path / "analysis"
    _index(tmp_path, out)
    samples = {s["sample_id"]: s["prediction"] for s in _read(out / "samples.csv")}
    assert samples["s0"] == "B"
    assert samples["s_kie"] == json.dumps(kie, ensure_ascii=False)
    assert samples["s_none"] == ""
    assert samples["s_json_str"] == '{"a":1}'
    assert samples["s_empty"] == ""


def test_index_no_predictions_and_readonly_store(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run_a", with_curve=True)
    before = (run_dir / "cache" / "eyas.sqlite").read_bytes()
    out = tmp_path / "analysis"
    _index(tmp_path, out, "--no-predictions")
    samples = _read(out / "samples.csv")
    assert "prediction" not in samples[0]
    # read-only open: the store bytes are untouched
    assert (run_dir / "cache" / "eyas.sqlite").read_bytes() == before


def _make_legacy_run_dir(root: Path, name: str, *, stamp: str | None) -> Path:
    """Rewrite a run's store into the historical pre-schema-3 shape: plain
    untagged prediction text, meta stamped with an older version (or the
    meta table absent entirely — the oldest unversioned stores)."""
    run_dir = _make_run_dir(root, name, with_curve=False)
    conn = sqlite3.connect(run_dir / "cache" / "eyas.sqlite")
    conn.execute("DELETE FROM outcomes")
    conn.execute("DROP TABLE meta")
    if stamp is not None:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema', ?)", (stamp,))
    conn.executemany(
        "INSERT INTO outcomes VALUES (?, 'abc123', ?, 'train', ?, 1, '{}', NULL)",
        [(name, "s0", "B"), (name, "s_json", '{"a":1}'), (name, "s_none", None)],
    )
    conn.commit()
    conn.close()
    return run_dir


def test_index_tolerates_pre_schema3_stores(tmp_path: Path) -> None:
    """A heterogeneous runs tree (historical schema-2 / unversioned stores
    beside a schema-3 store) must index fully: old predictions export
    VERBATIM (plain text, NULL -> empty, no tag decoding), new ones decode,
    exit 0 — one old store must not abort the whole index."""
    _make_run_dir(tmp_path, "run_v3", with_curve=False)
    _make_legacy_run_dir(tmp_path, "run_v2", stamp="2")
    _make_legacy_run_dir(tmp_path, "run_unstamped", stamp=None)
    out = tmp_path / "analysis"
    _index(tmp_path, out)

    runs = {r["name"] for r in _read(out / "runs.csv")}
    assert runs == {"run_v3", "run_v2", "run_unstamped"}
    by_run: dict[str, dict[str, str]] = {}
    for s in _read(out / "samples.csv"):
        by_run.setdefault(s["run"], {})[s["sample_id"]] = s["prediction"]
    assert set(by_run) == {"run_v3", "run_v2", "run_unstamped"}
    # schema 3: tag-decoded
    assert by_run["run_v3"]["s0"] == "B"
    # pre-schema-3: verbatim historical text, never tag-decoded
    for old in ("run_v2", "run_unstamped"):
        assert by_run[old]["s0"] == "B"
        assert by_run[old]["s_json"] == '{"a":1}'
        assert by_run[old]["s_none"] == ""


def test_corrupt_schema3_tag_does_not_abort_index(tmp_path: Path) -> None:
    """A genuinely corrupt schema-3 prediction tag follows the per-store
    posture (like a corrupt sqlite page): that store's export stops, the
    rest of the index is still written, exit 0."""
    _make_run_dir(tmp_path, "run_ok", with_curve=False)
    bad = _make_run_dir(tmp_path, "run_bad", with_curve=False)
    conn = sqlite3.connect(bad / "cache" / "eyas.sqlite")
    conn.execute(
        "INSERT INTO outcomes VALUES ('run_bad', 'abc123', 's_bad', 'train',"
        " 'j:{not json', 1, '{}', NULL)"
    )
    conn.commit()
    conn.close()
    out = tmp_path / "analysis"
    _index(tmp_path, out)
    samples = _read(out / "samples.csv")
    assert any(s["run"] == "run_ok" and s["prediction"] == "B" for s in samples)
    assert {r["name"] for r in _read(out / "runs.csv")} == {"run_ok", "run_bad"}
