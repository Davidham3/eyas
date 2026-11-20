# Eyas — Automatic Prompt Optimization for Multimodal Models

**Train sharp-eyed prompts for multimodal models.**

## The prompt is the last hand-crafted part of the pipeline

Look at what office automation asks of a model these days: sort a warehouse photo album into catalog categories, lift the totals and dates out of a stack of scanned invoices, watch a surveillance clip and answer what happened next. The previous decade solved each such problem expensively: a team paid annotators to label the domain, trained a bespoke model on those labels, and threw both away the week the paperwork changed its look. The vision-language model collapsed that assembly line; for anything not proprietary, a single general model now does the reading, no training step required. One station of the old line survived the collapse, though: the human who decides what to ask the model and how to phrase the ask. That station matters more than people expect, since swapping two phrasings of the same request moves the score on your data by points. And the shortcut tricks that once seemed to help the asking, the canned incantations everyone copy-pasted, now mostly move nothing: models have been trained on those phrases until the phrases stopped being special. The human judgment that remains therefore has to be exercised differently, by running a loop that measures candidate sentences against real data and lets the numbers pick the winner instead of hunting for one clever sentence.

## Automatic prompt optimization, in one paragraph

That loop has a name: **automatic prompt optimization (APO)**. Treat the prompt as the variable that gets trained, without touching the model's weights: an optimizer generates candidate prompts, every candidate is scored by the task metric on real data, whatever wins survives to seed the next round, and the cycle runs under an explicit budget of requests and tokens. The field organizes around four method families. *Rewrite-and-propose*: sample or induce candidate instructions and score the pool. *Textual gradient*: have a model write out, in words, why the current prompt failed on sampled errors, then revise the prompt in the opposite direction of that critique. *Score-trajectory*: keep the (prompt, score) history inside the optimizer's own context and have a model write the next candidate from the trajectory. *Evolution and self-play*: maintain a population that an LLM mutates and recombines, or let candidates compete in reference-free pairwise votes. These are the field's map, not one tool's taxonomy, and Eyas ships eight methods as points on it, from the `raw` measurement baseline through hill climbing to the paper methods `ape`, `protegi`, `opro`, `evoprompt` and `spo`, all on one strategy contract ([the method gallery](methods.md)). The flagship story at the bottom of this page is an `opro` run (score-trajectory), so the page's example is one of the families the map shows.

## What hurts in production — and what this framework does about it

Four problems come with the territory of running an optimization loop on real multimodal work. They belong to the domain, not to any one tool; Eyas was built hitting all four, and every answer below links to measured evidence rather than an adjective.

- **The optimization bill is a black box.** Judge calls, retries and comparison rounds hide inside one "total requests" number, so nobody can say what a round of the loop actually cost or where the money went. *Here:* every run bills the judge channel separately (`judge_requests`, judge tokens) and records per-round cost into the run's own ledger, and every published accuracy travels with its price tag: [quickstart benchmark §E3, the cost profile](07-quickstart-benchmark.md) profiles wall time, requests and tokens per cell, judge channel included.

- **The loop never meets your deployment.** Optimizers get exercised against one hosted API, production runs a self-hosted engine, and the results have to be re-earned across the bridge. *Here:* one config runs both shapes, an in-process vLLM engine or any OpenAI-compatible endpoint (`base_url` + `api_key_env` are the whole connection), and the two arms were measured against the same weights: parity inside noise, the raw rows identical to four decimals ([quickstart benchmark §E1, the backend-parity table](07-quickstart-benchmark.md)).

- **Images and video are second-class citizens.** Loops designed for text bolt media on afterwards, so a picture task inherits half the scheduling and none of the accounting. *Here:* image, video and mixed-modality tasks are first-class in the task registry: the published tables carry fine-grained image classification, receipt field extraction, scene-text QA and video QA, and a locally synthesized mix of seven per-sample passage/image/clip combinations is smoke-verified end to end per combination on a scripted in-repo backend (scope note in the [quickstart benchmark](07-quickstart-benchmark.md): it is not a benchmark column), all under one config, one render order and one cost ledger ([Data and models](datasets.md)).

- **There is no comparable public record.** Published improvements are usually single-seed on private protocols, a headline "+20%" no outsider can check, with no two methods measured the same way. *Here:* a public scoreboard built to be argued with: 6 datasets × 7 models × 8 methods × 3 seeds, 1008 completed runs, every cell mean ± σ over the fixed seeds 42/43/44, tables machine-regenerated from the run directories, plus a per-method paper-conformance record ([bench benchmark](08-bench-benchmark.md), [baseline fidelity](baselines-fidelity.md)).

## One contract, eight methods

```{image} _static/strategy-loop.svg
:alt: Strategy and OptimizerLoop exchanging Decision and EvalView
:align: center
:width: 680px
```

What makes those comparisons fair, and the eight methods interchangeable, is a single contract: a method is only a **strategy** whose every `step` decides *evaluate these candidates, propose these new ones, or finish with a best*; everything else (budgeting, cache-first evaluation, checkpointing, judging, cost accounting) belongs to one managed loop that all eight share, byte for byte. That is why eight methods are comparable on equal budgets, why writing a ninth is a ~100-line exercise ([methods](methods.md), [contract-driven design](10-architecture.md)), and why a run directory (per-sample records, prompts, history, bill) is self-contained enough to resume from and to audit afterwards.

Five of the eight are reproduced from published SOTA prompt optimization papers — APE, OPRO, EvoPrompt, ProTeGi, SPO — each with a conformance record against the official repo pinned at commit level ([baseline fidelity](baselines-fidelity.md)). Δ is the published TextVQA test gain over the seed prompt (mean ± σ over seeds 42/43/44, quickstart tier, machine-regenerated from the run dirs); each row links its method card.

| Method | Published as | Algorithmic identity | Δ vs `raw` |
| --- | --- | --- | --- |
| [`opro`](methods.md#opro) | **OPRO** — Yang et al., *Optimization by PROmpting*, ICLR 2024 ([arXiv:2309.03409](https://arxiv.org/abs/2309.03409)) | keeps an (instruction, score) trajectory in context; the model proposes its successor from the ranking | **+8.7 pp** |
| [`spo`](methods.md#spo) | **SPO** — Xiang et al., *Self-Supervised Prompt Optimization*, Findings of EMNLP 2025 ([arXiv:2502.06855](https://arxiv.org/abs/2502.06855)) | reference-free answer voting: ABBA pairwise ballots, the selection channel never sees a gold label | +6.4 pp |
| [`evoprompt`](methods.md#evoprompt) | **EvoPrompt** — Guo et al., ICLR 2024 ([arXiv:2309.08532](https://arxiv.org/abs/2309.08532)) | evolves a prompt population with LLM-run genetic / differential-evolution operators | +5.8 pp |
| [`ape`](methods.md#ape) | **APE** — Zhou et al., *Large Language Models Are Human-Level Prompt Engineers*, ICLR 2023 ([arXiv:2211.01910](https://arxiv.org/abs/2211.01910)) | one-shot forward induction: induce instructions from demo pools, score the whole pool, keep the best | +5.6 pp |
| [`protegi`](methods.md#protegi) | **ProTeGi** — Pryzant et al., EMNLP 2023 ([arXiv:2305.03495](https://arxiv.org/abs/2305.03495)) | textual gradients on sampled errors + beam search + UCB bandit candidate selection | +1.3 pp |
| [`hill_climb`](methods.md#hill_climb) | framework baseline (classic iterative refinement) | greedy local search over a fixed edit menu | +0.2 pp |
| [`cot`](methods.md#cot) | framework baseline (chain-of-thought rendering) | identity lives in the rendered prompt, not the loop | −42.8 pp on this judgeless QA task — the published anti-pattern row |
| [`raw`](methods.md#raw) | framework baseline (evaluation floor) | evaluates the seed prompt once; every Δ above is measured against it | 0 |

One level up: the object this loop optimizes is a *textualized component*, and a prompt is simply the component the field optimizes today. The same propose-measure-keep machinery is what the field is now turning toward agent skills and instruction libraries; this framework's contract, a textual candidate plus a managed measurement loop, is the shape such a component would need. That is an observation about the domain's map, not a promise: Eyas ships prompt optimization, and that is the whole claim.

## The story, in one table

TextVQA: scene-text visual question answering, rule-scored against ten human answers per question, no LLM judge anywhere. Qwen3.5-9B, held-out test split (n = 100), mean ± σ over the fixed seeds 42/43/44:

| raw (seed prompt) | opro (optimized prompt) | Δ (pp) |
| --- | --- | --- |
| 0.736 ± 0.004 | **0.788 ± 0.029** | +5.2 |

No weights were touched: the search rewrote one sentence of the prompt, and the held-out gain above is what it bought, measured by re-running the exact configs of the getting-started page against a self-hosted OpenAI-compatible endpoint. At n = 100 a single reading carries a ±4 to 5 pp noise band, which is why the table quotes three-seed means, and the σ shown is the loop's honest run-to-run spread (every seed lifted). The published in-process-vLLM wave of the same configs, the textvqa column of [quickstart benchmark §E2, the three-seed effect table](07-quickstart-benchmark.md), measures the same lift with an unusually tight three-seed cluster (+8.7 pp).

**Start here:** [Getting started](tutorials/getting-started.md) walks this exact run end to end: the three-command setup (data prep, baseline, optimization), the real round-by-round log, the before/after prompts, and the full bill. No endpoint yet? The [guided tour](tutorials/guided-tour.md) executes a complete loop with zero downloads, zero keys, zero GPU.

## Where to go next

| If you want to ... | Read |
| --- | --- |
| Walk the flagship story end to end (real data, real lift, real bill) | [Getting started](tutorials/getting-started.md) |
| Run a complete loop right now — zero downloads, zero keys | [Guided tour](tutorials/guided-tour.md) |
| See what each method does and pick one | [Methods](methods.md) |
| Understand a run config line by line | [Your first optimization](tutorials/first-optimization.md) |
| Read run outputs, caches and reports | [Understanding results](tutorials/understanding-results.md) |
| Plug in your own dataset / task | [Custom tasks](tutorials/custom-task.md) |
| Write your own optimization method | [Custom optimization methods](tutorials/custom-method.md) |
| Run against a real model (vLLM / API) | [Inference backends](tutorials/backends.md) |
| Reproduce the published benchmark numbers | [Reproducing the benchmark](tutorials/benchmark.md) |
| Check dataset sources and their separate terms | [Data and models](datasets.md) |

The repo also ships runnable starter packages under `examples/`: `strategy_skeleton/` is the copy-and-rename kit for writing a new method (used by the custom-method tutorial), `plugin_backend/` a working out-of-tree backend registration.

The design notes document the architecture behind the tutorials: the [contract-driven design](10-architecture.md) is the authoritative reference, and the [bench benchmark](08-bench-benchmark.md) is the **single authoritative effect scoreboard** (6 open datasets × 7 models × 8 strategies × 3 seeds, 1008 runs), machine-regenerated from the runs, with per-cell cost profiles. The [quickstart benchmark](07-quickstart-benchmark.md) documents the out-of-the-box protocol tier (backend parity, quickstart cost), and [baseline fidelity](baselines-fidelity.md) records how faithfully each published baseline matches its paper.

## License

Apache-2.0. See the `LICENSE` file in the repository root.

```{toctree}
:hidden:
:caption: Tutorials

tutorials/getting-started
tutorials/guided-tour
tutorials/first-optimization
tutorials/understanding-results
tutorials/custom-task
tutorials/custom-method
tutorials/backends
tutorials/benchmark
```

```{toctree}
:hidden:
:caption: Design notes

methods
datasets
baselines-fidelity
07-quickstart-benchmark
08-bench-benchmark
10-architecture
```
