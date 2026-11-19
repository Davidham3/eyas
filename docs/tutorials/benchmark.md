# Reproducing the benchmark

**What you'll learn:** how to rebuild the published quickstart benchmark (4 public datasets, 8 strategies, unified budgets, 3 seeds) and how to read the result tables.

The full study design, the result tables and the conclusions live in the [quickstart benchmark design note](../07-quickstart-benchmark.md). This tutorial is the hands-on path through it.

## What is being reproduced

| Dataset | Task | train / test | Metric |
| --- | --- | --- | --- |
| pets (Oxford-IIIT Pet) | fine-grained breed MCQ | 74 / 111 | accuracy |
| cub (CUB-200-2011 snapshot) | bird-species MCQ, 8 groups | 110 / 220 | accuracy |
| sroie (SROIE receipts) | key-information extraction | 100 / 100 | field accuracy |
| textvqa (TextVQA labeled-val slice) | open-ended scene-text QA | 100 / 100 | accuracy (VQAv2 soft) |

All runs share one open-weight VLM (Qwen3.5-9B), one budget tier per method, and one scoring protocol; every method runs on seeds 42/43/44. The benchmark answers three questions: do backends agree (E1), which strategies beat which baselines (E2), and what does each strategy cost (E3).

A note on data: every dataset below is downloaded at prepare time from its upstream host and none is redistributed by this repository; sources and their separate terms are listed in [Data and models](../datasets.md).

## Step 1 — install and prepare data

```bash
uv sync --extra vllm    # or --extra api
# --dataset is required (no default build); the quickstart tables need these four:
uv run --extra prepare eyas prepare --dataset both --source auto   # pets + sroie
uv run --extra prepare eyas prepare --dataset cub --source auto
uv run --extra prepare eyas prepare --dataset textvqa --source auto
# --dataset all additionally builds gsm8k / nextqa (bench-scale cells, docs/08);
# nextqa needs the user-supplied clip pack via --nextqa-videos <zip|dir>.
```

The script builds `datasets/quickstart/<ds>/{data,images}/`, downloading through a fallback chain (HF direct -> hf-mirror, plus ModelScope for `pets`). Point the `model_path` placeholder in the configs at your local Qwen3.5-9B checkpoint directory.

## Step 2 — single run

Cells are assembled from [`configs/spec`](../../configs/spec) — there are no per-cell files to read before running; `--cell <dataset>/<method>` is the whole address:

```bash
uv run eyas run --cell pets/raw
```

Each cell pins the whole protocol: data splits, backend, eval model, budget tier, and (where a method needs one) an evaluation judge — the spec layers (defaults / datasets / methods / cells) carry it, and the run directory's `run_config.json` records the effective config plus the spec content hash (`run.spec_sha`). Independent replicates are a one-key override:

```bash
uv run eyas run --cell pets/raw --set run.seed=43
```

## Step 3 — sweep everything

```bash
uv run eyas gen-configs --cell pets/raw --cell pets/cot
uv run eyas sweep --config configs/generated/pets_raw.yaml \
                  --config configs/generated/pets_cot.yaml
```

`sweep` runs configs sequentially (one `--config` per file) and aborts the sweep if any run fails. Each cell gets its own run directory; re-running a cell reuses records only when it re-enters that directory (`--resume`, or a fixed output/cache dir).

For a large factorial study there is no need to enumerate one file per cell: `--spec` expands the published benchmark spec (facet axes over assembled cells) directly; the published bench study runs exactly this way (`configs/spec/benchmark.yaml`, docs/08 §2-3). It reads the larger `datasets/bench*` directories, built with the recipe in docs/08 §1 (Step 1 above builds the quickstart `datasets/quickstart/` tree):

```bash
uv run eyas sweep --spec configs/spec/benchmark.yaml \
  --filter tag=q2b --filter dataset=pets \
  --set run.seed=42 \
  --set backend.params.model_path=/path/to/Qwen3.5-2B
```

## Step 4 — collect results

```bash
uv run eyas report --run outputs/quickstart_pets_raw/<stamp> --format md
```

Each run directory's `result.json` carries the final test-split score (configs set `run.final_eval.split: test`, reported under `final_eval`), the per-round history, and the cost aggregates. The cost table (E3) is just the `total_cost` and `cost_by_tier` fields across runs: the runtime counts requests and wall time, bills the judge channel separately (`judge_requests`), and aggregates response tokens; `python scripts/aggregate_bench_results.py quickstart-effect / quickstart-cost` re-emits the published tables from the run dirs.

## Reading the tables

Three headline findings from the published study; all numbers and caveats are in the design note:

- **Backends agree.** The in-process vLLM engine and an OpenAI-compatible endpoint produce identical scores for identical weights, templates and scoring: the raw pets/sroie cells are bit-identical across both arms and the widest cell (cub cot, 3.6pp) sits inside the binomial sampling range (E1, design note §4).
- **Strategies diverge by task shape.** No method dominates everywhere, which is why the framework makes them swappable. Structured extraction rewards hill climbing (+8.0pp on sroie, with cot close behind at a tenth of the requests); on nearly-saturated pets there is nothing to gain: the seed prompt wins on every seed and every optimizer is net-negative; on cub only evoprompt clears noise (+2.6pp); and on judgeless textvqa prompts buy answer-format gains (opro +8.7pp) while the CoT invitation collapses the first-line answer contract (−42.8pp).
- **Costs vary orders of magnitude**, from single-pass baselines to heavy multi-round optimizers. The scoreboard pairs every score with its price; a method that wins 1 point for 5x requests is a trade.

## Adding a dataset

Your dataset slots into the same protocol if it fits one of the shipped task shapes: grouped multiple choice (`mcq`), open-ended VQA (`vqa`), or entity extraction (`kie`). Build split manifests in the documented formats, export a cell as an editable base (`uv run eyas gen-configs --cell <ds>/<method>`), swap the `task` block, and you inherit the entire comparison machinery: strategies, seeds, record caching, cost accounting. See [Custom tasks](custom-task.md) for genuinely new scoring semantics.

## Next steps

- [Understanding results](understanding-results.md): what `result.json`, `checkpoint.json` and `report` give you.
- The [quickstart benchmark design note](../07-quickstart-benchmark.md): full protocol, tables, and known boundaries.
- The [bench benchmark design note](../08-bench-benchmark.md): the full-scale study behind the README scoreboard, covering full splits, the model-size sweep, and the cost frontier.
