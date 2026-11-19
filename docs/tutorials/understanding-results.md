# Understanding results

**What you'll learn:** what each file in a run directory means, how the evaluation record store works, how to resume a run, and how to compare runs with `uv run eyas report`.

## The run directory

Every run writes a self-contained directory under `outputs/`. This tree is the real output of the zero-data hill-climb run captured in [Your first optimization](first-optimization.md) (its config set `run.trace: true`):

```text
outputs/tutorial_shapes_hc/20261117-173941/
├── status.json        # live phase / round / cost / cache state
├── result.json        # final artifact: best, ranked, history, cost
├── checkpoint.json    # snapshot written at each evaluate decision — resume enters here
├── run_config.json    # the effective config (overrides applied)
├── framework.log      # full-format log (the console shows the bare lines)
├── cache/
│   └── eyas.sqlite    # per-sample evaluation records (SQLite, WAL)
└── trace.jsonl        # every request/response (only with run.trace: true)
```

A `artifacts/` subdirectory joins the listing only when a strategy emits cross-strategy artifacts. The stamp is `%Y%m%d-%H%M%S`; a same-second collision appends `-2`, `-3`, ...

The directory is a **checkpoint**: copy it elsewhere and the run's knowledge moves with it. Deleting it deletes everything the run knows. (`--output-dir DIR` replaces the whole `outputs/<name>/<timestamp>` path with `DIR` itself. The run dir is exactly what you passed, no extra nesting.)

## `result.json`

The shape mirrors what the loop returns:

| Field | Meaning |
| --- | --- |
| `best` | Winning evaluation — `best.text` is your final prompt, `best.metrics` its scores |
| `ranked` | All evaluated prompts, best-first |
| `history` | Per-round records: round, candidates requested, requests actually issued, that round's `cost`, `elapsed_sec`, and `best_so_far` (a round is an `evaluate` decision — propose/finish steps produce no history row) |
| `total_cost` | `requests` / `input_tokens` / `output_tokens` / `wall_seconds`, plus the `judge_requests` / `judge_input_tokens` / `judge_output_tokens` split already included in those totals |
| `cost_by_tier` | Optimizer-side generation requests, keyed by sampling tier (`generate`, `gradient`, ...) |
| `final_eval` | Best prompt re-scored on the held-out reporting split (`null` when unconfigured or same split) |
| `artifacts` | Manifest of artifacts the strategy emitted, content-addressed |

For texture, the story run's artifact (the TextVQA × OPRO run of [Getting started](getting-started.md), abridged, with `best.text` elided and `ranked` and `history` collapsed):

```json
{
  "best": {
    "prompt_hash": "4a52ea3e833d",
    "metrics": { "accuracy": 0.8633333333333333, "parse_fail": 0.0 }
  },
  "total_cost": {
    "requests": 1600,
    "input_tokens": 1304208,
    "output_tokens": 6441,
    "wall_seconds": 193.89097208715975,
    "judge_requests": 0,
    "judge_input_tokens": 0,
    "judge_output_tokens": 0
  },
  "cost_by_tier": { "generate": 16 },
  "final_eval": {
    "split": "test",
    "prompt_hash": "4a52ea3e833d",
    "metrics": { "accuracy": 0.8133333333333332, "parse_fail": 0.0 },
    "cost": {
      "requests": 100,
      "input_tokens": 78419,
      "output_tokens": 402,
      "wall_seconds": 8.65102164261043,
      "judge_requests": 0,
      "judge_input_tokens": 0,
      "judge_output_tokens": 0
    }
  }
}
```

The same prompt wins on both sides of the two-stage protocol (`best.prompt_hash` re-scored on `test`), the loop's bill and the reporting re-score are accounted apart, and the judge columns are zero because TextVQA is rule-scored. The `history` rows, one per round with their own `cost` / `best_so_far`, are walked line by line in [Your first optimization](first-optimization.md).

Prompts are identified by `prompt_hash`, the SHA-256 of their normalized text, so identical rewrites collapse everywhere: archive, records, reports.

## The evaluation record store

`cache/eyas.sqlite` stores one durable record per (prompt, sample, split), answering *what did this prompt score on this sample?* Each record carries the prediction and its scores, keyed by the prompt's content hash. A store hit short-circuits everything: no rendering, no inference, no judge. **Each run gets a fresh run directory, so a plain re-run starts from an empty store.** Records are reused only when the new run is pointed at an existing one: `--resume <run_dir>` re-enters that directory, and two runs share a store only if you set the same `run.cache_dir` (or run both against one fixed `--output-dir`).

```{warning} Semantic changes need a new run directory
The record keys assume the scoring semantics, the seed and the algorithm did not change. If you change any of those, start a **new run directory**: `--resume` on the old one would replay records produced under the old semantics as-is.
```

## Resuming and overriding

```bash
# Resume into an existing run directory (mutually exclusive with --output-dir)
uv run eyas run --config exp.yaml --resume outputs/exp/20260926-101530
```

Resume restores the strategy's typed state, the archive, per-round history, the budget allocator's state and the master RNG from `checkpoint.json`, then lets the strategy keep deciding. Records replay from the store, so already-scored prompts cost zero requests.

```bash
# Override any config key from the CLI (CLI > config > default)
uv run eyas run --config exp.yaml --set run.seed=43
```

`--set` takes dotted paths and fails fast if the path does not exist in the config. The open doors: a `params` section is the component's kwargs namespace (assembled as `cls(**params)`), and any optional section the config leaves `null` can be materialized; so `--set backend.params.some_key=...` may create keys the config never listed, and a typo there fails at startup on the constructor's `TypeError` instead. Everywhere else a missing path still fails fast. Changing the seed deliberately gives you an independent replicate: the standard protocol is three seeds (e.g. 42/43/44) and mean ± std.

## Reading the objectives

A task's `objectives` is an ordered tuple: the first entry is the headline metric every strategy optimizes and that `best` is chosen by. The remaining entries are reported alongside in `best.metrics` (e.g. a KIE task carries `field_acc` / `strict_acc` / `char_sim`) so you can eyeball secondary quality, but they are never traded off during the search; there is no non-dominated front to read.

## Cross-run comparison

```bash
# One run, human-readable
uv run eyas report --run outputs/exp/20260926-101530 --format md

# Several runs into a comparison table (markdown or csv)
uv run eyas report --run outputs/exp_a/<stamp> --run outputs/exp_b/<stamp> \
    --format csv > comparison.csv
```

Reports read `result.json` + `run_config.json` and pull the same fields the result files carry: best metrics plus split-prefixed `final_eval` columns (`val_*`, `test_*`), cost and `cost_by_tier`. Comparing strategies is therefore a file-listing exercise, not a log-diving one. (`report` does not read the per-round `history`; open `result.json` for that.)

## Cost accounting

Every run tallies evaluation and judge-channel requests together in `total_cost.requests` (the `judge_*` fields split the judge share back out), optimizer-side generation per tier in `cost_by_tier`, and wall seconds overall. When you compare two strategies, read their cost next to their scores: a method that wins by 1 point while spending 5x the requests is a trade, not a win.

## What you just learned

- The run directory is a portable checkpoint; the directory itself is the key.
- One cache layer does the work: per-sample evaluation records keyed by prompt hash.
- `--set` overrides any config key; `--resume` re-enters an existing run directory from `checkpoint.json`.
- Secondary objectives are reported in `best.metrics` but never traded off; `total_cost` and `cost_by_tier` expose the price.

## Next steps

- [Methods](../methods.md): which method should I use next? Compare the eight shipped strategies and pick one.
- [Custom tasks](custom-task.md): make the loop score *your* metric.
- [Custom optimization methods](custom-method.md): add your own strategy to `uv run eyas list`.
