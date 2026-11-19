# Guided tour — one run, start to finish

The topic tutorials each teach one surface (configs, results, tasks, methods).
This page instead walks **one journey end to end**: start a run, watch its
state while it lives, read every artifact it leaves, extend it with a reporting
split, replicate it across seeds, and recover it from a crash. All of it runs
with zero GPU, so you can follow along in a terminal right now. Every output
block below was captured from a real execution of the commands above it
(timestamps and wall-clock values from that capture), not written by hand.

What this page runs: the built-in `demo_shapes` task, declared in
[`configs/tutorial_shapes.yaml`](../../configs/tutorial_shapes.yaml). It renders
solid-color square PNGs in the repo (no dataset, no weights) and asks "what
color?", against any OpenAI-compatible chat endpoint. This page
teaches **loop mechanics**: the shape of the artifacts, the live state
document, and the resume path, all identical in a real run. The numbers are not:
the endpoint behind the docs URL answers `red` to every request, so metrics
here carry no algorithmic meaning. The real-score story lives in
[Getting started](getting-started.md). Nothing on this page can rot
silently: every command block below is executed in CI, verbatim, against an
in-repo stub of the OpenAI API.

## 1. The materials

One YAML file is the entire experiment definition
([`configs/tutorial_shapes.yaml`](../../configs/tutorial_shapes.yaml)):

```yaml
run:
  name: tutorial_shapes
  seed: 42

task:
  name: demo_shapes   # built-in zero-data task: auto-generates solid-color square PNGs
  params: {}

protocol:
  name: identity

backend:
  name: openai
  params:
    base_url: http://localhost:8000/v1
    model: qwen3.5-9b
    api_key_env: EYAS_DOC_STUB_KEY
    max_inflight: 16
  sampling:
    thinking: false

eval_model:
  name: absolute      # absolute scoring (accuracy); task objective can produce it -> assembly validation passes

budget:
  name: flat
  params: {}  # FlatBudget is parameterless: every candidate, full split (explicit pin)

# Sampling tiers (optional): algorithm-facing knobs only -- thinking/seed always
# come from backend.sampling (the deployment layer)
sampling_profiles:
  generate: { temperature: 0.9, max_new_tokens: 512 }

strategy:
  name: cot           # degenerate method: the CoT identity lands in the rendered prompt text
  params:
    base_prompt: "What color is the square? Answer with one color name."
    cot_suffix: "Let's think step by step."
```

Split into a decision half and an execution half: `run` is run metadata, the
other sections are components, each resolved through its own decorator
registry (`eyas list` prints them all), and each section's `params` become the
component's constructor kwargs. That structure *is* the framework's
contract-first design: if a combination is semantically illegal (a
compare-only method fitted to an accuracy leaderboard), assembly refuses to
start rather than burning GPU time to discover it.

## 2. Start the run

```bash
uv run eyas run --config configs/tutorial_shapes.yaml --output-dir outputs/eyas_tour
```

Console output (captured, verbatim):

```text
Done: best="What color is the square? Answer with one color name. Let's think step by step." requests=8
round 1/400: 8 reqs in 0.83s eta≈0.00s
```

Two lines, both worth reading. Their order carries a fact of its own: the CLI's
final `Done:` print can land before the round echo, which travels the logging
module's background writer thread, a deliberate design where log writes never
block the run, with console ordering as its accepted cosmetic cost:

- **The round echo**: `round 1/400: 8 reqs in 0.83s eta≈0.00s`. Every round
  the loop charges its requests to the budget, evaluates them cache-first, and
  echoes progress with an ETA. On a real run this is your heartbeat: rounds
  land every few minutes, and the watchdog treats *silence* (20 min without
  activity) as a hang, not a crash. The kill is deliberately IO-independent.
- **Why `cot` ran exactly one round**: `cot` is the degenerate method: one
  fixed prompt (seed + suffix), evaluated once on the 8-sample split. `finish`
  comes back immediately; `400` is the default `run.max_steps` (set `run.max_steps`
  to change the ceiling), never a bound.

The run dir is now the whole experiment:

```text
outputs/eyas_tour
├── status.json        # live observation contract (phase, cost, cache, history)
├── checkpoint.json    # round-granular resume state
├── result.json        # final answer: best, ranked, history, cost
├── run_config.json    # the fully-resolved config this run was assembled from
├── framework.log      # full-format log (the console shows the bare lines)
└── cache/eyas.sqlite  # the durable evaluation record store (schema-versioned)
```

## 3. Read the live state

`uv run eyas status outputs/eyas_tour` renders the run dir's observation
documents read-only: it reads `status.json` (falling back to
`checkpoint.json`) and appends `result.json` when present (captured after the
run; mid-run it is equally valid, phase `optimizing`):

```text
run dir: outputs/eyas_tour
run: tutorial_shapes (strategy=cot task=demo_shapes seed=42)
phase: done
round: 1
budget: flat
cost: requests=8 input_tokens=976 output_tokens=8 wall_seconds=0.537433
judge: requests=0 input_tokens=0 output_tokens=0
cache: hits=0 misses=8 hit_rate=0.0%
rejections: none
updated_at: 2026-11-17T09:24:04+00:00
history: r0 starting 0s | r1 optimizing 1.197s | r1 done 0.035s
metrics: accuracy=0.25 parse_fail=0
best: "What color is the square? Answer with one color name. Let's think step by step." hash=b6174e9477dc
cost by tier: -
```

The fields cover the whole accounting in one screen:

- **`cost` vs `judge`**: the optimization channel and the judging channel are
  metered separately (judging rides its own sampling tier, pinned greedy so a
  verdict never rides the run's seed).
- **`cache: hits=0 misses=8`**: every evaluation consults the record store
  first. A cold run misses everything; a resumed run (or a repeat pointed at the
  same run directory) replays records at zero request cost. This is why run dirs are *the* unit of work.
- **`rejections`**: every request goes through one planner chokepoint;
  anything over the declared resource limits is refused *before* reaching the
  backend and counted here, never silently dropped.
- **`history`**: the phase timeline (`starting → optimizing → done`), appended
  per status write, bounded (`HISTORY_LIMIT`) so the document stays small.

All three durable documents (`status.json`, `checkpoint.json`, `result.json`)
are written temp-then-`os.replace`: a kill landing mid-write can never tear
them, and a corrupt checkpoint (external damage, or a pre-atomic run dir)
fails resume with a clear message instead of a raw `JSONDecodeError`.

## 4. Read the answer

`result.json` is the reportable artifact (captured, abridged):

```text
{
  "best": {
    "prompt_hash": "b6174e9477dc",
    "text": "What color is the square? Answer with one color name. Let's think step by step.",
    "metrics": { "accuracy": 0.25, "parse_fail": 0.0 }
  },
  "ranked": [ ... ],
  "history": [ { "round": 0, "n_candidates": 1, "n_requests": 8,
                 "cost": { ... }, "elapsed_sec": 0.8347,
                 "best_so_far": { ... } } ],
  "total_cost": { "requests": 8, "input_tokens": 976, "output_tokens": 8, ... },
  "cost_by_tier": {},
  "final_eval": null
}
```

- `accuracy=0.25` is exactly right *and meaningless*: the docs endpoint answers
  `red` to every request, and the demo's train labels cycle over a four-color
  palette, so exactly two of the eight train samples are labeled red. Fixed by
  construction: if it ever moves, that is a pipeline bug worth reporting.
- `best.prompt_hash` is the prompt's **sole identity** (sha256 of the
  normalized text, first 12 hex): labels and metadata are display-only and can
  never become cache or dedup keys.
- `total_cost` covers the optimization loop only. Any final reporting split is
  accounted apart, which the next section covers.

## 5. Add a reporting split (and learn `--set`'s honesty)

The two-stage protocol: the loop selects on `run.eval_split` (default `train`);
after the loop, the archived best is re-evaluated on `run.final_eval.split` to
produce the reporting numbers. Natural attempt:

```bash
uv run eyas run --config configs/tutorial_shapes.yaml --set run.final_eval.split=val
```

This **fails on purpose** (the last line of the traceback):

```text
ValueError: --set path does not exist: 'run.final_eval.split' ('final_eval' is not a mapping)
```

`--set` overrides keys that exist; it will not invent structure (typo
protection). The open doors are a `params` kwargs namespace and any optional section
the config leaves `null`; a typo in `params` instead fails at startup on the
constructor's `TypeError`. So structure gets added where structure lives:
copy the config and grow the `run:` section by the two `final_eval` lines.
The complete edited file follows, run and reported (watch the run dir fill
a `final_eval` block):

```bash
cat > /tmp/tutorial_shapes_val.yaml <<'YAML'
run:
  name: tutorial_shapes
  seed: 42
  final_eval:
    split: val

task:
  name: demo_shapes   # built-in zero-data task: auto-generates solid-color square PNGs
  params: {}

protocol:
  name: identity

backend:
  name: openai
  params:
    base_url: http://localhost:8000/v1
    model: qwen3.5-9b
    api_key_env: EYAS_DOC_STUB_KEY
    max_inflight: 16
  sampling:
    thinking: false

eval_model:
  name: absolute      # absolute scoring (accuracy); task objective can produce it -> assembly validation passes

budget:
  name: flat
  params: {}  # FlatBudget is parameterless: every candidate, full split (explicit pin)

# Sampling tiers (optional): algorithm-facing knobs only -- thinking/seed always
# come from backend.sampling (the deployment layer)
sampling_profiles:
  generate: { temperature: 0.9, max_new_tokens: 512 }

strategy:
  name: cot           # degenerate method: the CoT identity lands in the rendered prompt text
  params:
    base_prompt: "What color is the square? Answer with one color name."
    cot_suffix: "Let's think step by step."
YAML
uv run eyas run --config /tmp/tutorial_shapes_val.yaml --output-dir outputs/eyas_tour_val
uv run eyas report --run outputs/eyas_tour_val --format md
```

Report output (captured; the md report also reprints the best prompt under a
`## Best prompts` heading, trimmed here). Note the two new columns:

```text
| run | strategy | task | seed | accuracy | parse_fail | val_accuracy | val_parse_fail | requests | judge_reqs | in_tok | out_tok | by_tier | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tutorial_shapes | cot | demo_shapes | 42 | 0.25 | 0 | 0.25 | 0 | 8 | 0 | 976 | 8 | - | 0.4538 |
```

Final-split metric columns carry the **split's own name** (`val_*` here,
`test_*` when the reporting split is `test`), so a val-split run can never
masquerade its numbers as test metrics. The accounting independence is
visible in `result.json`: the loop billed 8 requests, and the val re-evaluation
billed its own 4 separately under `final_eval.cost`; reporting cost never
pollutes the optimization bill.

## 6. Replicate, don't trust

A single run is weak evidence, so replicate it. `run.seed` is a real, existing
key, so this time `--set` works:

```bash
uv run eyas run --config configs/tutorial_shapes.yaml --output-dir outputs/eyas_tour43 --set run.seed=43
uv run eyas run --config configs/tutorial_shapes.yaml --output-dir outputs/eyas_tour44 --set run.seed=44
```

Three seeds, then aggregate:

```bash
uv run eyas report --run outputs/eyas_tour --run outputs/eyas_tour43 --run outputs/eyas_tour44 --format md
```

```text
| run | strategy | task | seed | accuracy | parse_fail | requests | judge_reqs | in_tok | out_tok | by_tier | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tutorial_shapes | cot | demo_shapes | 42 | 0.25 | 0 | 8 | 0 | 976 | 8 | - | 0.5374 |
| tutorial_shapes | cot | demo_shapes | 43 | 0.25 | 0 | 8 | 0 | 976 | 8 | - | 0.426 |
| tutorial_shapes | cot | demo_shapes | 44 | 0.25 | 0 | 8 | 0 | 976 | 8 | - | 0.4288 |
```

The published benchmark protocol is exactly this shape at scale: every cell
is `mean ± population σ` over three seeds (see
[Reproducing the benchmark](benchmark.md)). The published seed-discipline rule
is worth knowing before you read your own table: **never conclude
from a single-seed gap under 2pp; seed std underestimates long-generation
drift.**

## 7. Crash, then resume

The run dir *is* the checkpoint. If a long run dies (preemption, OOM, watchdog
kill), its records survive twice over: the SQLite store commits per write
(before the checkpoint), and `checkpoint.json` froze the loop position at the
last round end. Recovery is one command: `--resume` re-enters the run dir
(`--resume` and `--output-dir` are mutually exclusive; the resume *is* the
dir), evaluation records replay so already-burned requests cost nothing, the
strategy's typed state restores, and the loop picks up round-by-round.

A tour run finishes in a second, too fast to kill mid-round, so the block
below earns the same lesson a different way: a fresh dir, then `--resume`
pointed at the *finished* run, which is the identical re-entry path a crashed
run takes:

```bash
uv run eyas run --config configs/tutorial_shapes.yaml --output-dir outputs/eyas_tour_resume
uv run eyas run --config configs/tutorial_shapes.yaml --resume outputs/eyas_tour_resume
```

The resume's `Done:` line re-bills the run's own total (`requests=8`) while the
endpoint served nothing new. The run dir's status after the resume (captured,
two lines) shows the replay:

```text
cache: hits=8 misses=0 hit_rate=100.0%
history: r0 starting 0s | r1 optimizing 1.083s | r1 done 0.031s | r1 starting 0s | r1 done 0.277s
```

`hits=8 misses=0`: the evaluations came from the record store, and the phase
history grew the re-entry pair (`r1 starting | r1 done`). Resume fidelity is
contract-tested: a run interrupted at round *k* and resumed lands on the same
`best`, the same per-round request counts, the same bill as an uninterrupted run.

## 8. From this file to the published cells

Everything above drove a hand-written file — `configs/tutorial_shapes.yaml` is
the tour's own teaching artifact. The cells behind the published numbers are
not files at all: each one is an address in the config spec (`configs/spec/`),
assembled into the full config at run time. Two commands own that surface, and
you have met both shapes already.

*Run the address.* Same `run` command, a `dataset/method` pair in place of
`--config` (real cells read a real dataset and need a real model, so this
block is the tour's only command that the built-in demo cannot serve):

```bash
uv run eyas run --cell pets/raw
```

*Export to edit.* Want a cell as a file — to read the whole config, or to
change something the spec does not cover? The escape hatch writes the exact
effective YAML, GENERATED header included, and a `why` note per spec layer:

```bash
uv run eyas gen-configs --cell pets/raw --cell pets/cot --out tmp/tour_cells
head -1 tmp/tour_cells/pets_raw.yaml
```

```text
# GENERATED from configs/spec — do not edit (eyas gen-configs --cell pets/raw)
```

The exported cell re-enters §2 unchanged:
`eyas run --config tmp/tour_cells/pets_raw.yaml`. For an endpoint served over
the OpenAI protocol the address takes a variant —
`--cell pets/raw --variant openai` — and a whole factorial study expands the
same spec in one wave (`--spec configs/spec/benchmark.yaml` + `--filter`s) in
[Reproducing the benchmark](benchmark.md).

One habit to keep: exported cells belong in `tmp/` (already gitignored). The
spec is the source; a file you edit is a fork you now maintain.

## 9. Where to go next

| Now you want to ... | Read |
| --- | --- |
| Define the run config for *your* task | [Your first optimization](first-optimization.md) |
| Go deeper on records, resume, reports | [Understanding results](understanding-results.md) — including [resuming a crashed run](understanding-results.md#resuming-and-overriding) |
| Plug in your dataset / metric | [Custom tasks](custom-task.md) |
| Write a ninth optimization method | [Custom optimization methods](custom-method.md) |
| Drive a real model (vLLM / API) | [Inference backends](backends.md) |
| Read and reproduce the published tables | [Reproducing the benchmark](benchmark.md), then the [bench benchmark design note](../08-bench-benchmark.md) |

One reading habit to keep from this tour: trust the artifacts in the run dir
before you trust any summary. Check `status.json` while the run lives,
`result.json` after, and `framework.log` when either surprises you.
