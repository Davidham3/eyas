# Eyas

**Train sharp-eyed prompts for multimodal models.**

Eyas is a performance-first framework for **automatic prompt optimization (APO) on multimodal
models**: bring a task, a seed prompt and a round budget (`run.max_steps`), and Eyas runs a real
optimization loop (hill climbing, evolutionary search, gradient-style rewriting) and hands you
back a better prompt, with the cost accounting to prove what it spent.

## The story, in one table

TextVQA (scene-text QA, rule-scored, no LLM judge), Qwen3.5-9B, held-out test split (n = 100),
mean ± σ over the fixed seeds 42/43/44:

| raw (seed prompt) | opro (optimized prompt) | Δ (pp) |
| --- | --- | --- |
| 0.736 ± 0.004 | **0.788 ± 0.029** | +5.2 |

No weights touched: the search replaced one sentence of the prompt, and the held-out gain above
is what it bought, measured with the two config files you are about to run, against a self-hosted
OpenAI-compatible endpoint. At n = 100 a single reading carries a ±4 to 5 pp noise band, so the
table quotes three-seed means; every seed lifted, and the σ shows the loop's real run-to-run
spread. The published in-process-vLLM wave of the same configs tells the same story with an
unusually tight three-seed cluster (+8.7 pp,
[docs/07 §E2, the three-seed effect table](docs/07-quickstart-benchmark.md)); the commands and every artifact are walked end
to end in [Getting started](docs/tutorials/getting-started.md).

The full published scoreboard, 7 models × 6 datasets × 8 methods × 3 seeds (1008 completed runs,
judge-channel and token accounting on every cell), is the
[bench benchmark](docs/08-bench-benchmark.md).

## Run the story

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Davidham3/eyas.git && cd eyas
uv sync --extra api --extra prepare
```

Build the exact splits the story config points at, then run the baseline and the optimization:

```bash
uv run eyas prepare --dataset textvqa
uv run eyas run --cell textvqa/raw --variant openai
uv run eyas run --cell textvqa/opro --variant openai
```

Each run writes a self-contained run directory under `outputs/<run.name>/<timestamp>` holding
`result.json` (best prompt, history, cost), `checkpoint.json`, and `status.json` (live phase /
round / cost). One config, two backend shapes: an in-process vLLM engine (`uv sync --extra vllm`,
the recommended default) or any OpenAI-compatible endpoint (`uv sync --extra api`); the config's
two lines `base_url` and `api_key_env` are the whole connection to your model. No endpoint yet?
The [guided tour](docs/tutorials/guided-tour.md) executes the complete loop on a built-in
synthetic task: zero downloads, zero keys, zero GPU.

## What every method shares

- Eight methods on one strategy contract. `raw`, `cot`, `hill_climb`, `evoprompt`, `opro`,
  `ape`, `protegi`, `spo`: a method's algorithm identity lives only in its strategy's `step`
  decisions; the managed loop owns every invariant (budgets, evaluation, accounting, resume).
  That is what makes eight methods comparable on equal budgets, and writing your ninth a
  ~100-line exercise.
- Every run bills the judge channel separately (`judge_requests`) and totals tokens; the
  published scoreboard reads accuracy against price.
- A run directory is a self-contained checkpoint: per-sample evaluation records (SQLite in
  WAL (write-ahead logging) mode), content-addressed prompt identity, resumable and movable.

Modality is a sample-level choice, not a framework fork: the same loop covers text, image, video
or any mix of the three, with shipped benchmark recipes for `gsm8k`, `cub`, `pets`, `textvqa`,
`sroie`, `nextqa` and a mixed `mixed_mcq`.

## Documentation

The docs site lives in [`docs/`](docs/index.md) (build:
`uv sync --group docs && uv run sphinx-build -b html docs docs/_build/html`). Tutorials:
[getting started](docs/tutorials/getting-started.md), [guided tour](docs/tutorials/guided-tour.md),
[your first optimization](docs/tutorials/first-optimization.md),
[understanding results](docs/tutorials/understanding-results.md),
[custom tasks](docs/tutorials/custom-task.md),
[custom optimization methods](docs/tutorials/custom-method.md),
[inference backends](docs/tutorials/backends.md),
[reproducing the benchmark](docs/tutorials/benchmark.md). Runnable starter packages live in
[`examples/`](examples/). The [contract-driven design](docs/10-architecture.md) is the
authoritative architecture reference; the benchmark protocols are
[docs/07](docs/07-quickstart-benchmark.md) and [docs/08](docs/08-bench-benchmark.md), and the
per-method conformance contract is [baseline fidelity](docs/baselines-fidelity.md).

## Data, models and provenance

Eyas ships code, not data or weights: benchmark datasets are downloaded at prepare time from
their upstream hosts, `datasets/` is gitignored and nothing is redistributed. The Apache-2.0
license covers the code only, while each dataset and model checkpoint keeps its own license and
terms, and several benchmark sets are research-use (sources and the model-checkpoint rule:
[Data and models](docs/datasets.md)).

> **Provenance note**: every published number is script-generated from the runs' `result.json`
> files by `scripts/aggregate_bench_results.py`, so the published tables are measurements of runs
> you can reproduce. The GPU run directories themselves are not distributed; regenerate them with
> the [benchmark tutorial](docs/tutorials/benchmark.md) and docs/08 §2-3.

## Contributing

Two decorator-based registries cover the extension surface: **Task / Backend / Judge / Protocol**
(what gets asked, where requests go, how responses become scores) and **Strategy / EvalModel /
Budget / PairwiseJudge** (the method itself, its scoring semantics, its budget policy, pluggable
pairwise judging); duplicate names fail fast. Start with
[Custom optimization methods](docs/tutorials/custom-method.md); `hill_climb` in
`src/eyas/engine/strategies/` is the reference implementation, and every shipped strategy runs
end-to-end on the framework's scripted in-repo test backend in CI (the fast suite is `uv run pytest`, zero GPU).

## License

Apache-2.0, see [LICENSE](LICENSE); it covers the code only (see
[Data, models and provenance](#data-models-and-provenance)).
