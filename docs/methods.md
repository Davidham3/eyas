# Methods: the eight shipped strategies

What each of the eight registered optimization methods does, when it is the right choice, and the published evidence behind that claim. Every evidence line is machine-regenerated from the same runs as the [quickstart benchmark](07-quickstart-benchmark.md), so it can never drift. End-to-end usage of one of them (OPRO on TextVQA) is the [getting-started story](tutorials/getting-started.md); writing a ninth method is the [custom-method tutorial](tutorials/custom-method.md).

## One contract, eight identities

Every method is a **strategy**: a decision half sitting on top of one shared execution half. Each `step` decides: *evaluate* these candidates, *propose* these new ones, or *finish* with a best; the managed loop does everything else (budgeting, cache-first evaluation, checkpointing, judging, cost accounting) identically for all eight.

```{image} _static/strategy-loop.svg
:alt: Strategy and OptimizerLoop exchanging Decision and EvalView
:align: center
:width: 680px
```

That split is what "method identity" means here: **a method is nothing but the sequence of decisions its `step` makes.** The degenerate cases make the point. `raw` proposes nothing; its whole identity is evaluating one seed prompt and stopping. `cot` also proposes nothing; its identity lives in the *rendering* (the chain-of-thought instruction carried in the prompt), which the contract admits as a first-class place to put an algorithm. Conversely `opro`, `evoprompt` and `spo` differ from each other only in *what they propose, from what signal*: the scoring, budgeting and accounting underneath is byte-for-byte the same machinery. That is why eight methods are comparable on equal budgets, and why the strategy registry (`eyas list` prints it) is the whole method catalogue.

A method never hard-codes a temperature. Each declares sampling **tiers** by name (`generate` / `evolve` / `gradient` / `synonym` / `compare`) and the deployment layer owns the numbers, with cost accounted per tier ([design note §6](10-architecture.md#6-sampling-tiers-samplingprofiles)).

## How to read the evidence lines

Each card carries one line from the **quickstart tier**: TextVQA test accuracy (mean ± population σ over seeds 42/43/44, n=3), the gain or loss against the `raw` row in percentage points, and the seed-42 request cost of the optimization loop (the shared 100-sample test pass is accounted separately). The lines are pasted **verbatim** from

> `uv run python scripts/aggregate_bench_results.py method-cards --dataset textvqa`

The command is the same aggregation that generates the full per-dataset tables in [quickstart benchmark §3](07-quickstart-benchmark.md#3-e2-main-tables-test-split-mean--population-σ-over-seeds-424344), so re-running it reproduces this page byte-for-byte. TextVQA is the story dataset because it is where all eight methods separate; the multi-model record (6 datasets × 7 models × 8 methods × 3 seeds) is the [bench benchmark §4](08-bench-benchmark.md), and each card links its conformance record in [baseline fidelity §4](baselines-fidelity.md#4-per-method-conformance). Read every line as mean ± σ: on the n=100 quickstart test split a single reading carries a ±4 to 5 pp noise band.

### raw

**What it does:** evaluates the task's seed prompt once and stops: the degenerate strategy, and the leaderboard floor every other row is measured against. A framework baseline with no paper claim ([fidelity §4.1](baselines-fidelity.md#41-raw--cot-framework-baselines-no-paper)).

**When to use:** always, as the first run. On near-saturated tasks it is also the
right call: on pets the seed prompt beat every optimizer on every seed (the whole Δ
column is negative, [quickstart §3](07-quickstart-benchmark.md#3-e2-main-tables-test-split-mean--population-σ-over-seeds-424344)).

**Evidence (TextVQA):**

- `raw` — test accuracy 72.9% ±0.2% (n=3) · cost req 100 (judge 0) · test pass req 100 (judge 0)

### cot

**What it does:** the classic chain-of-thought prompting trick, carried as a rendered variant of the seed prompt, under the same degenerate evaluate-and-stop loop; its existence shows the contract admits "identity lives in the rendering" methods. A framework baseline; the conformance record covers its rendering and judged-path merging, not a paper ([fidelity §4.1](baselines-fidelity.md#41-raw--cot-framework-baselines-no-paper)).

**When to use:** when the scoring channel can consume reasoning: judged multiple choice (the judge reads the reasoning back) or structured extraction, where the quickstart's KIE column rewards it. Keep it away from judgeless short-answer span matching, where justification text leaks into the graded first line (see the −42.8 pp row below).

**Evidence (TextVQA):**

- `cot` — test accuracy 30.1% ±0.6% (n=3) · Δ -42.8 pp vs raw · cost req 100 (judge 0) · test pass req 100 (judge 0)

### hill_climb

**What it does:** greedy local search over prompts: evaluate the kept candidates, mutate the incumbent with a small edit menu, keep any improvement. The classic iterative-refinement baseline that the APE and OPRO papers compare against, with no single source paper of its own; as a paper-less framework baseline it has no per-method subsection and is covered by the shared conventions of [fidelity §4](baselines-fidelity.md#4-per-method-conformance).

**When to use:** the simplest *real* optimizer (its proposer is a fixed edit menu, no demo questions and no optimizer-model choreography) and a strong choice on structured tasks: it posts the best quickstart cell on sroie ([quickstart §3](07-quickstart-benchmark.md#3-e2-main-tables-test-split-mean--population-σ-over-seeds-424344)).

**Evidence (TextVQA):**

- `hill_climb` — test accuracy 73.1% ±0.4% (n=3) · Δ +0.2 pp vs raw · cost req 900 (judge 0) · test pass req 100 (judge 0)

### ape

**What it does:** one-shot **forward induction**: sample input→output demonstrations, ask the model to induce the instruction they follow, clean and dedup the candidates, score the whole pool, keep the best ([APE, ICLR 2023](https://arxiv.org/abs/2211.01910); registered deviations incl. pool size: [fidelity §4.2](baselines-fidelity.md#42-ape--iclr-2023-zhou-et-al)).

**When to use:** a single cheap proposal round when you have demonstration samples to induce from: one round, one pool. Not shipped for the generative-prompt KIE task (it needs demo questions); its quickstart cells therefore skip sroie.

**Evidence (TextVQA):**

- `ape` — test accuracy 78.4% ±4.0% (n=3) · Δ +5.6 pp vs raw · cost req 400 (judge 0) · test pass req 100 (judge 0)

### opro

**What it does:** trajectory-guided search: keep an (instruction, score) history, show its best entries in ascending order inside a meta-prompt, and have the model propose the next candidates from that trajectory ([OPRO, ICLR 2024](https://arxiv.org/abs/2309.03409); registered deviations incl. cap and K: [fidelity §4.3](baselines-fidelity.md#43-opro--iclr-2024-yang-et-al)).

**When to use:** when a competent optimizer model is available to read the trajectory, and the win channel is instruction *wording*; it is the quickstart's TextVQA champion. Not shipped for the generative-prompt KIE task (same demo-question requirement as ape).

**Evidence (TextVQA):**

- `opro` — test accuracy 81.6% ±0.3% (n=3) · Δ +8.7 pp vs raw · cost req 1600 (judge 0) · test pass req 100 (judge 0)

### evoprompt

**What it does:** evolves a *population* of prompts with LLM-run evolutionary operators: GA (roulette parents, one crossover→merge-then-mutate template call per child) or DE (four-prompt position-aligned operator), elitist update; the official N_P=10 seed pool ships in the config ([EvoPrompt, ICLR 2024](https://arxiv.org/abs/2309.08532); registered deviations incl. seed-pool provenance: [fidelity §4.4](baselines-fidelity.md#44-evoprompt--iclr-2024-guo-et-al)).

**When to use:** broad exploration when the budget can feed a population and pay for its breadth. It is the only positive optimizer on the quickstart's hardest fine-grained column (cub, [quickstart §3](07-quickstart-benchmark.md#3-e2-main-tables-test-split-mean--population-σ-over-seeds-424344)) and the biggest positive delta on the bench TextVQA 9B column ([bench §4](08-bench-benchmark.md)).

**Evidence (TextVQA):**

- `evoprompt` — test accuracy 78.7% ±0.0% (n=3) · Δ +5.8 pp vs raw · cost req 1100 (judge 0) · test pass req 100 (judge 0)

### protegi

**What it does:** beam search over candidates expanded by **textual gradients**: a critique call lists reasons a sampled error group was mispredicted, edit calls revise the prompt per reason, each child fans out paraphrases, and a bandit allocator picks which candidates earn further evaluation ([ProTeGi, EMNLP 2023](https://arxiv.org/abs/2305.03495); registered deviations incl. budget scaling: [fidelity §4.5](baselines-fidelity.md#45-protegi--emnlp-2023-pryzant-et-al)).

**When to use:** when per-error feedback is the informative signal (clear systematic failure modes) and evaluation cost per candidate must stay low; the bandit prunes weak arms early. Budget-honest: at quickstart scale it is the most variance- and budget-sensitive method of the eight (worst quickstart delta on cub, [quickstart §3](07-quickstart-benchmark.md#3-e2-main-tables-test-split-mean--population-σ-over-seeds-424344)).

**Evidence (TextVQA):**

- `protegi` — test accuracy 74.2% ±1.6% (n=3) · Δ +1.3 pp vs raw · cost req 324 (judge 0) · test pass req 100 (judge 0)

### spo

**What it does:** self-supervised **answer voting**: the incumbent and a challenger (proposed from the incumbent plus its wrong samples) face off on paired samples, and a reference-free pairwise judge casts ABBA position-swapped ballots; the challenger advances only on a strict win surplus ([SPO, Findings of EMNLP 2025](https://arxiv.org/abs/2502.06855); registered deviations incl. window size: [fidelity §4.6](baselines-fidelity.md#46-spo--findings-of-emnlp-2025-xiang-et-al)).

**When to use:** when absolute scores feel unreliable and "which answer is better" is the more trustworthy question: the selection channel structurally never sees a gold label (gold may appear only on the proposal side), so selection quality does not ride on the metric's calibration.

**Evidence (TextVQA):**

- `spo` — test accuracy 79.3% ±4.2% (n=3) · Δ +6.4 pp vs raw · cost req 600 (judge 0) · test pass req 100 (judge 0)

## Which one should I pick?

| Your situation | Start with | Why |
| --- | --- | --- |
| Any task, first run | `raw` | The floor. On a near-saturated task it also *wins*: every quickstart pets optimizer is below the seed prompt |
| Judgeless short-answer / span-matched task (VQA-style) | `opro`, then `evoprompt` / `spo` / `ape` | The quickstart TextVQA gains (+5.6 to +8.7 pp, the evidence lines above) ride the answer-format channel — and `cot` is the documented anti-pattern there |
| Structured extraction with a field metric (KIE) | `hill_climb`, `cot` | The quickstart sroie column's best cells ([quickstart §3](07-quickstart-benchmark.md#3-e2-main-tables-test-split-mean--population-σ-over-seeds-424344)) |
| Fine-grained classification far from its ceiling | `evoprompt` | The only positive optimizer on the quickstart cub column; watch its population cost |
| Tiny budget, one cheap exploration round | `ape` | A single propose round over an induced pool — no multi-round loop to feed, and its loop lands in the middle of the TextVQA cost lines above |
| Absolute scoring unreliable; pairwise preference acceptable | `spo` | Selection is gold-free answer voting; see the bench tables for where pairwise-style gains land ([bench §4](08-bench-benchmark.md)) |
| Clear systematic errors + tight per-candidate eval budget | `protegi` | Bandit selection stops spending on weak arms early; give it real budget — the quickstart cells show how sharp its downside gets when the signal is thin |

Two caveats apply across the board. First, every published row above is a three-seed mean ± σ; the quickstart §3 preamble's cross-seed ranges reach 9.7 pp on this very dataset, so never pick a method from a single run. Second, gains are channel-specific: on judgeless open-ended QA the prompt is cheap to steer toward *format*, and the numbers above mostly price that channel. Where perception, not wording, is the limit (fine-grained classification, or a model whose raw score already reads strong), no method on this page reliably buys much; the bench scoreboard's saturated columns say the same thing at full scale ([bench §4](08-bench-benchmark.md)).

## Illegal pairings are rejected before they cost you anything

A strategy declares its evaluation semantics, and the assembly check ([design note §3](10-architecture.md#3-capability-declaration-and-assembly-time-validation-capability-negotiation)) refuses impossible pairings **at startup, before any request is billed**. The case that shapes how to read the cards above: a method that only produces pairwise-comparison scores can never enter an accuracy leaderboard. Mounting it on an accuracy target raises a `ValueError` naming the objective it cannot produce ("this method must not enter the leaderboard for these objectives"), instead of running to completion and reporting `—` after you have paid for the GPU time. The guarantee is a framework property, not a per-method check: the contract tests pin it with a synthetic compare-only strategy mounted on an accuracy board (`tests/test_contract.py`).

## Read next

- [Getting started](tutorials/getting-started.md): the `opro` card above, walked end to end on TextVQA with your own endpoint.
- [Custom optimization methods](tutorials/custom-method.md): the ~100-line path to a ninth strategy.
- [Quickstart benchmark](07-quickstart-benchmark.md): the full per-dataset E2 tables and cost profile behind every evidence line on this page.
- [Bench benchmark](08-bench-benchmark.md): the authoritative multi-model scoreboard.
- [Baseline fidelity](baselines-fidelity.md): per-method paper-conformance tables and registered deviations.
