# Your first optimization

**What you'll learn:** every section of a real run config, what the strategy and the managed loop each do per round, and how two methods share one task and one backend.

[Getting started](getting-started.md) ran the docs' flagship story with one config: the TextVQA × OPRO profile, printed there in inline form as `quickstart_textvqa_opro.yaml`. Small enough to read whole, real enough to be the file behind published numbers, and the best place to see the anatomy of an Eyas run. (Repository checkouts can regenerate the same config from
[`src/eyas/spec`](../../src/eyas/spec) with `uv run eyas gen-configs --cell textvqa/opro --variant openai` — a developer path; the spec stays the source.)

## The config, piece by piece

```yaml
# The textvqa/opro openai cell in inline form. Repository checkouts export the same file:
#   uv run eyas gen-configs --cell textvqa/opro --variant openai
run:
  name: quickstart_textvqa_opro_openai
  seed: 42
  final_eval:
    split: test

task:
  name: vqa
  params:
    data_dir: datasets/quickstart/textvqa/data
    images_dir: datasets/quickstart/textvqa/images

backend:
  name: openai
  params:
    base_url: http://localhost:8000/v1
    model: qwen3.5-9b
    api_key_env: EYAS_DOC_STUB_KEY
    max_inflight: 16
  sampling:
    thinking: false

protocol:
  name: identity

eval_model:
  name: absolute

budget:
  name: flat
  params: {}  # FlatBudget is parameterless: every candidate, full split (explicit pin)

strategy:
  name: opro
  params:
    rounds: 4
```

Every section is a component resolved through a decorator registry (`uv run eyas list` prints them all), and each section's `params` mapping is the component's constructor kwargs. Three sections are required (`task` / `backend` / `strategy`); the rest are companions with sensible defaults. Walk the pieces in the order the loop feels them (task, backend, strategy, budget), then the protocol pair.

### `task` — what "good" means

`vqa` is a built-in task: it loads the splits the [Getting started](getting-started.md) data script built (seeded train 100 / val 30 / test 100 questions under `data_dir`, images under `images_dir`), renders each prompt into a text+image request, parses the model's answer, and scores it with rule-based VQAv2 soft accuracy against the ten human answers per question, with no LLM judge anywhere. Tasks own four responsibilities: loading samples, building requests, parsing responses, and scoring. (You will write your own in [Custom tasks](custom-task.md).)

### `backend` — where requests go

The `openai` backend is a plain HTTP client for any server exposing `/v1/chat/completions`: `base_url` + `api_key_env` are the whole connection (the key itself lives in the environment, never in the config). `max_inflight` is the client-side concurrency window; the optional `backend.sampling` block sets **deployment-level** sampling defaults (temperature, `max_new_tokens`, `thinking`, `seed`). The very same config shape runs the in-process vLLM engine instead: swap the section for `name: vllm` with a `model_path` and nothing else changes ([Inference backends](backends.md)).

### `strategy` — the method

`opro` keeps the (prompt, score) trajectory in the optimizer model's context and asks it for the next candidate; `rounds: 4` is four propose-and-score rounds on top of the seed-prompt evaluation. That is the entire footprint of the algorithm in the config, and swapping this one section is all it takes to run a different method on the same task, data and endpoint.

### `budget` — how much data each candidate sees

`flat` gives every candidate the full evaluation split, which is why the story's 4-round OPRO search, 16 evaluated candidates including the seed, bills exactly 100 requests each (the 1,600 in its `total_cost` is the arithmetic). `even_split` divides the sample budget across candidates; `bandit` (UCB1, upper confidence bound) adaptively concentrates evaluation on candidates that look promising. The allocator is stateful and the loop feeds it back every round's scores.

### `run`, `protocol`, `eval_model` — identity and the scoring pair

- `run.name` names the run and is part of the default output path (`outputs/<name>/<stamp>/`). Each launch creates a fresh run directory, so a plain re-run recomputes; reuse records by pointing the new run at an existing directory: `--resume <run_dir>`, or a fixed `run.output_dir` / shared `run.cache_dir`.
- `run.seed` seeds the master RNG (random number generator) that strategy decisions, data shuffling and derived child RNGs draw from, so two runs with the same seed are reproducible. Deployment sampling is configured separately under `backend.sampling` (including its own `seed`, if set).
- `run.final_eval.split: test` is the two-stage split protocol: the loop optimizes on `run.eval_split` (default `train`), and afterwards the archived best is re-scored once on the held-out reporting split. Optional here: omit it and `final_eval` is `null`. `run.trace: true` turns on request-level tracing.
- `protocol: identity` means each sample is asked once, as-is; other protocols can derive multiple variants per sample (the task's `build_request` sees the variant id).
- `eval_model: absolute` scores each prediction against the task's rules (accuracy, `field_acc`, ...). `compare` is the pairwise sibling: it asks which of two candidates is better, optionally through a pluggable `pairwise_judge`. Both are first-class: a strategy declares which semantics it requires, and `validate_assembly` rejects mismatched combinations at startup.

The story runs baseline first, then the search. Same endpoint, same splits, one strategy
swap (the baseline clears OPRO's `params`; `raw` takes none):

```bash
uv run eyas run --config quickstart_textvqa_opro.yaml --set strategy.name=raw --set strategy.params={}
uv run eyas run --config quickstart_textvqa_opro.yaml
```

## A real search loop: hill climbing

Same task, same data, same endpoint: copy the story config and swap only the strategy
section (with the quickstart data built and an endpoint wired, per
[Getting started](getting-started.md)):

```bash
cp quickstart_textvqa_opro.yaml your_hill_climb.yaml
```

Repository checkouts can instead start from the generated cell:
`uv run eyas gen-configs --cell textvqa/opro --variant openai` (developer path).

Set `run.name: my_hill_climb` (the default output directory follows that name) and replace the `strategy:` block with one that searches without an optimizer model:

```yaml
strategy:
  name: hill_climb
  params:
    rounds: 3
    keep_size: 3
    per_parent: 2
```

```bash
uv run eyas run --config your_hill_climb.yaml
```

`hill_climb` is deliberately small (`src/eyas/engine/strategies/hill_climb.py` in a repository checkout) and is the template the framework's contract tests ride on. Its proposer is a fixed edit menu, so it spends zero optimizer-side LLM requests. Conceptually, each `step`:

1. **Evaluate anything unseen.** If candidates in state have no scores yet, return `Decision("evaluate", prompts=...)`: the loop does the scoring.
2. **Rank by score.** The read-only `EvalView` answers "evaluated?" and "score?" from the archive; the strategy never mutates evaluation results.
3. **Mutate or finish.** After `rounds` rounds, `Decision("finish", best_id=...)`. Otherwise the top `keep_size` parents each spawn `per_parent` suffix mutations, which enter state and get evaluated next step.

The managed loop does everything else: charges the budget, batches and caches evaluation, records per-round history, bumps the watchdog, writes the checkpoint.

## Watch the loop move

The examples below are captured from a real execution of exactly this shape (`hill_climb` with `rounds: 3` / `keep_size: 3` / `per_parent: 2`) on the zero-data built-in `demo_shapes` task against a local endpoint (the same in-repo endpoint the [guided tour](guided-tour.md) runs on, so the metrics carry no algorithmic meaning; the plumbing is the real thing, and on your run everything below reads the same with real numbers). Open the run directory (pass `--output-dir DIR` to pin it; otherwise it is `outputs/<name>/<stamp>/`) and look at `result.json`'s `history`. One round row, verbatim:

```json
{
  "round": 1,
  "n_candidates": 2,
  "n_requests": 16,
  "cost": {
    "requests": 16,
    "input_tokens": 1840,
    "output_tokens": 16,
    "wall_seconds": 0.16444439813494682,
    "judge_requests": 0,
    "judge_input_tokens": 0,
    "judge_output_tokens": 0
  },
  "elapsed_sec": 0.7741222754120827,
  "best_so_far": {
    "prompt_hash": "2084e6fd166e",
    "metrics": {
      "parse_fail": 0.0,
      "accuracy": 0.25
    }
  }
}
```

- `round`: loop round counter.
- `n_candidates`: prompts the strategy asked to evaluate this round.
- `n_requests`: inference requests actually issued (record-store hits issue none, so replays shrink to zero).
- `cost` / `elapsed_sec`: that round's full bill (judge split included) and the loop's own clock. `best_so_far` is the archive leader after the round, and consecutive leaders form the best-so-far learning curve.

Then look at `total_cost`: requests and wall seconds aggregated across the whole run (with the `judge_*` fields splitting out the judge channel), plus `cost_by_tier` if the strategy makes optimizer-side LLM calls. The story's OPRO run, for instance, records `"cost_by_tier": { "generate": 16 }` (16 optimizer-side prompt generations) next to a `total_cost` of 1,700 evaluation requests. `final_eval` (when `run.final_eval.split` is configured) holds the best prompt's score on the held-out reporting split: the story run's seed-42 best prompt scored 0.750 test-side while the loop had selected it at 0.873 on train (the held-out transfer of a train-selected winner is noisy at these split sizes, see the honesty note on the [getting-started page](getting-started.md)); it is `null` otherwise.

```{note} Why the first objective matters
A task declares an ordered tuple of `objectives` (e.g. `accuracy`, or `field_acc` + `strict_acc` + `char_sim`). Every strategy optimizes the **first** objective (it becomes `ctx.primary_objective`), and the loop ranks by it; the remaining entries are reported alongside as metrics but never traded off during the search.
```

## What you see while it runs

The terminal and the run dir split the work: the console carries progress, the files carry detail. Every round the loop finishes prints one line at the default level, from the captured hill-climb run above, whose four console lines match the four `history` rows one for one:

```text
round 1/400: 8 reqs in 1.11s eta≈3.33s
round 2/400: 16 reqs in 0.77s eta≈1.88s
round 3/400: 32 reqs in 2.33s eta≈1.40s
round 4/400: 32 reqs in 1.23s eta≈0.00s
```

- `round 2/400`: rounds whose checkpoint has persisted, against the loop's enforced ceiling (`run.max_steps`). A method that reaches its own target first simply stops printing.
- `16 reqs`: inference requests that round actually issued (a record-store hit issues none, so it matches `history[].n_requests`).
- `0.77s`: what that round really took (`history[].elapsed_sec`, the same round's loop clock).
- `eta≈1.88s`: mean completed-round duration × rounds left under the horizon (the method's own declared round count when it is smaller than the ceiling, else `run.max_steps`). When no round has a measured duration (a fresh attempt, or a resumed run whose status carries no round timing), it reads `eta≈n/a`: no projection is fabricated from zero measurements.

Everything finer lives in the run dir: `status.json` is the live machine-readable state (phase, round cursor, cost, cache hits/misses, rejections), `framework.log` holds the same lines with timestamp, level and logger, and `trace.jsonl` the per-request record when `run.trace: true`. Two knobs widen the view without leaving the terminal:

```bash
# every phase visible: planner rejections, record-store hit/miss, prefetch batches
EYAS_LOG_LEVEL=DEBUG uv run eyas run --config your_hill_climb.yaml

# render the live state of a run in progress (read-only, any moment)
uv run eyas status outputs/my_hill_climb/<timestamp>
```

## What you just learned

- A run config has three required sections (`task` / `backend` / `strategy`) and the usual companions (`run`, `protocol`, `eval_model`, `budget`, `sampling_profiles`, `judge`, `plugins`). You just walked a shipped one section by section.
- Methods differ only in the strategy block; tasks, backends, budgets and eval models are swappable without touching the method, and `validate_assembly` catches illegal pairings before launch.
- The search loop is observable: one `round <i>/<max>` progress line per round on the console (with an eta from the run's own round times), per-round history, content-addressed prompt identity, cost accounting.

## Next steps

- [Understanding results](understanding-results.md): records, resume semantics, and cross-run reports.
- [Guided tour](guided-tour.md). The full lifecycle on one run: live `status.json`, every artifact, seed replication, crash recovery.
- [Custom tasks](custom-task.md): point the same loop at your own data.
