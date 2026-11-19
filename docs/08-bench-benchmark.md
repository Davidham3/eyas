# Bench benchmark: full-scale splits, model-size sweep, cost frontier

> Protocol + reproduction doc, and the **single authoritative effect scoreboard**:
> every published accuracy figure in this repo (README scoreboard included) is emitted
> from the run set described here by `scripts/aggregate_bench_results.py` (default
> `--sweep matrix`). The quickstart benchmark (docs/07) answers "does every method
> run out of the box"; this doc covers the **headline benchmark** built on top of it,
> addressing its documented limits (docs/07 authority note and §9): mini splits
> (σ≈3 percentage points (pp)), a single model, and a single budget tier.
>
> - **B, model sweep**: the same cells at Qwen3.5-2B / 4B / 9B and
>   InternVL3.5-2B / 4B / 8B. Headline hypothesis: optimized small ≈ unoptimized
>   big (the cost-performance frontier that makes APO pay for itself).
> - **C, cost frontier**: judge-channel billing + token totals (landed in
>   `Cost.judge_*` / report `judge_reqs`/`in_tok`/`out_tok`); every bench run emits the
>   full cost profile, so accuracy tables can be read against requests/tokens.
>
> **The published run set**: every table below is emitted from the
> open-dataset matrix: 7 model tags × 6 datasets ×
> **all 8 methods** × seeds 42/43/44 = 126 sweeps / **1008 runs**, every run completed,
> zero failures. Six nextqa 2B cells replayed from the run-dir record store after an
> interrupted first attempt (their bulk was billed there; §4 marks them), every other
> loop bill is the cold billing of its run. Every table is regenerated from the run dirs by
> `scripts/aggregate_bench_results.py`:
>
> ```bash
> uv run python scripts/aggregate_bench_results.py bench-effect --sweep matrix
> uv run python scripts/aggregate_bench_results.py internvl-effect --sweep matrix
> uv run python scripts/aggregate_bench_results.py headline --sweep matrix
> ```
>
> The seventh column is the extension tag **q38b**, the legacy tag label for
> **Qwen3.8-27B** (the tag reads like "3.8B", but the model is 27B), a public 27B
> open-weight checkpoint: the same
> in-process vllm shape on the same generated cells,
> scoped to all six open datasets (18 sweeps / 144 runs); Qwen3.8 shares the
> `qwen3_5` architecture family, so the profile/thinking contract is inherited
> verbatim.
> The cells run full-scale splits, the quickstart
> method contracts (hyperparameters, budgets, judging protocol) at bench scale,
> every cell rendering its own Hugging Face chat template (`hf_template` profile),
> and all 8 methods on every
> dataset, including the generative-demo pair (ape / opro) on sroie: the
> single-bucket degenerate label path on KIE (the docs/07 textvqa precedent),
> publishing like any other cell.

## 1. Full-scale datasets (`datasets/bench/`; gitignored)

Built with the same recipe as quickstart (`eyas prepare`, shipped in the package as
`eyas.prepare`), only the
split knobs change:

```bash
uv run --extra prepare eyas prepare \
    --dataset pets  --output-root datasets/bench \
    --pets-train-per-label 8 --pets-val-per-label 4 --pets-test-per-label 30
uv run --extra prepare eyas prepare \
    --dataset cub   --output-root datasets/bench --source hf-mirror \
    --cub-families 25 --cub-species-per-family 8 --cub-imgs-per-species 10 \
    --cub-train-per-species 3 --cub-val-per-species 1 --cub-test-per-species 6
uv run --extra prepare eyas prepare \
    --dataset sroie --output-root datasets/bench \
    --sroie-train-n 248 --sroie-val-n 128 --sroie-test-n 250
uv run --extra prepare eyas prepare \
    --dataset textvqa --output-root datasets/bench \
    --textvqa-train-n 248 --textvqa-val-n 128 --textvqa-test-n 250
uv run --extra prepare eyas prepare \
    --dataset nextqa --output-root datasets/bench --nextqa-videos <NExT-QA.zip> \
    --nextqa-train-n 248 --nextqa-val-n 128 --nextqa-test-n 250
uv run --extra prepare eyas prepare \
    --dataset gsm8k --output-root datasets/bench \
    --gsm8k-train-n 248 --gsm8k-val-n 128 --gsm8k-test-n 250
```

The split-size flags are explicit on every command: the script's quickstart
defaults (100/30/100) are NOT the bench splits, and the published bandit cells
pin `8×31 = 248` on the 248-sample train splits; a tree built at the defaults
fails fast at run time.

The upstream source of each dataset, and its separate terms, are listed in
[Data and models](datasets.md); nothing is redistributed by this repository.

| dataset | split (train/val/test) | vs quickstart | notes |
| --- | --- | --- | --- |
| pets | 296 / 148 / 1110 | 74/111 | 37 breeds, ≤30 images/label on test; σ(test)≈1.5pp |
| cub | 261 / 87 / 522 | 110/220 | 15 family groups / 87 species (families with ≥4 species with ≥10 imgs); grouped-MCQ protocol as in quickstart |
| sroie | 248 / 128 / 250 | 100/100 | full available pool (626 receipts); test = 250, the complete upstream test split |
| textvqa | 248 / 128 / 250 | 100/100 | same labeled-val slice source, enlarged to the bench-scale budget tier |
| nextqa | 248 / 128 / 250 | — | video MCQ: one clip per sample (32 frames), one QA picked per video from the pooled official CSVs; clips come from the 6.5 GB NExT-QA pack (2.6 GB extracted, gitignored, never committed); σ(test)≈2.6pp |
| gsm8k | 248 / 128 / 250 | — | text-only numeric math QA (official `main` config; val carved out of the train split); one final answer, rule-scored; σ(test)≈2.6pp |

nextqa / gsm8k have no quickstart twin: both joined the matrix at bench scale only
(the `--dataset all` path builds all six).

sroie note: the loader now pools the upstream test split as well (full-scale requirement);
quickstart slices are unchanged on disk (not rebuilt).

## 2. Config matrix (one spec in the repo)

The published cells are ONE spec, `configs/spec/benchmark.yaml`, expanded by
`eyas sweep --spec` (facet axes whose values are override dicts; cell bases are
assembled from the config spec, no base files read). No per-cell YAMLs exist:
each run directory persists its effective config and the spec content hash
(`run_config.json`), so the spec + the run dirs are the reproducibility record.
A cell's base is exactly the assembled quickstart cell of the same
dataset+method, so every cell inherits its
method's exact assembly contract (hyperparameters, protocols, budgets, judges)
verbatim from E2: **B differs from E2 only by data scale and model size**, the
same invariant the predecessor generator encoded. The expansion is bound by
contract tests (`tests/test_matrix_configs.py`: 336 cells across the six boards, registry binding,
run identity, placeholder paths, InternVL load traits, bandit caps, curated
pools) and golden-parity to the published wave (`tests/test_configspec.py`). What the axes carry:

- `tag` (`q2b,q4b,q9b,ivl2b,ivl4b,ivl8b,q38b`) → placeholder model path; InternVL
  tags additionally replace the rendering profile (custom code, `trust_remote_code`)
  and carry the engine trait. The q38b extension tag is scoped to all six open
  datasets.
- `dataset` (`pets,cub,sroie,textvqa,nextqa,gsm8k`) → the data directories the
  published runs used (`datasets/bench/...`, carried by each dataset's `bench:`
  bundle in `configs/spec/datasets.yaml`).
- `method` (the 8 published strategies) → assembles the method identity onto the
  dataset contract — including the KIE `ape`/`opro` columns, which need no
  synthesis (the kie dataset contract + those method identities assemble to the
  exact cells the retired base-swap combos produced; golden parity proved it);
  two combos pin the bandit eval caps to the published bench plan (32/32/31/31).

`run.seed` 43/44 via `--set run.seed=NN` (quickstart seed convention); `run.trace:
true` is the spec's global `set`.

Run dirs: `outputs/matrix_<tag>_<ds>_s<seed>/NN-<run.name>/` (sweep layout, cache =
run dir, resubmit resumes).

## 3. Execution

Each (model tag, dataset, seed) runs as a 1×A100-80G job (vllm extra):
one matrix expansion of the 8 method cells through a single engine.
The published run set covers the open-dataset matrix (q2b / q4b / q9b / ivl2b /
ivl4b / ivl8b / q38b × seeds
42/43/44, six datasets, all 8 methods each): **126 sweeps / 1008 runs, zero
failures**. The executable command:

```
uv run eyas sweep --spec configs/spec/benchmark.yaml \
  --filter tag=q2b --filter dataset=pets \
  --set run.seed=42 \
  --set backend.params.model_path=<MODELS_ROOT>/Qwen/Qwen3.5-2B \
  --output-root outputs/matrix_q2b_pets_s42
```

(`--filter` selects the cells; `--set` layers per-cell overrides on top
of the expansion: model paths are placeholders in the spec, and `<MODELS_ROOT>`
stands for the local directory holding your model checkpoints, seeds 43/44 ride
the same flag. The data-path overrides are spec'd:
`task.params.data_dir`/`images_dir` carry the
published directories, so only the weights path needs a real value at run time.)

The format-controlled arms quoted in §4 and §D are the same raw cell re-run with
the task's default prompt plus one fixed answer-format sentence, under their own
output-root prefix so they never touch a published run dir (cub replaces video
with image; the q9b arm is the sentence-cost control):

```
uv run eyas sweep --spec configs/spec/benchmark.yaml \
  --filter tag=q4b --filter dataset=nextqa --filter method=raw \
  --set run.seed=42 \
  --set 'strategy.seeds=["Classify the content of the target video. Reply with exactly one of the listed options and nothing else."]' \
  ...
```

They are diagnostics, not part of the 1008-run set.

## 4. Results (Qwen family)

Every cell succeeded (Qwen half: 4 tags × 6 datasets × 3 seeds = 72
sweeps / 576 runs, all 8 methods each). Tables are emitted by
`scripts/aggregate_bench_results.py`: accuracy = mean ± population σ over the three
seeds; the cost line = per-seed mean over seeds whose loop bill is non-zero (a cell
replayed from the run-dir record store bills zero requests and is excluded)
(`total_cost` = optimization loop only; the shared final_eval test pass is accounted
separately). Six nextqa 2B cells carry the *(resume)* treatment: q2b and ivl2b
raw / cot / hill_climb at seed 42 replayed from the run-dir record store (loop and
final-eval bill both read zero; the spend was billed on an earlier attempt of the
same wave), so their cost means run over the two billed seeds and are marked
`[cost n=2]`. Accuracy is unaffected: a replayed cell reports the stored outcome of
the same computation. Every other loop bill in these tables is the cold billing of
its run.

A cell whose mean `parse_fail` reaches 5% also states the parsed-only accuracy
(`pf X% → parsed Y%`: accuracy / (1 − parse_fail), computed per seed, then
averaged). Protocol accuracy charges an unparseable prediction as wrong, so a
cell that loses more answers to format than to reasoning is not comparable to its
neighbours on accuracy alone. Ten cells of the run set meet the bar: nextqa
(raw 4B, spo 2B/ivl2b), sroie cot (2B/4B/ivl2b), cub (raw ivl2b/ivl4b, spo and
ape ivl2b); every other cell reads exactly as before.

**Reading the model columns**: all four columns run the *full optimization* on
bench-scale train under one protocol, every cell rendering its own `hf_template`
profile, one generation end to end (the q38b extension column inherits the contract:
Qwen3.8 shares the `qwen3_5` architecture family and runs the identical cells on the
same in-process vllm shape). The run set confirms two signature anomalies in the
9B column: the pets raw sits within 1.2 pp of its
column's optimized ceiling, and the sroie 9B raw (81.1) lands at its own 2B level
(81.4) instead of the 4B one (88.5); the latter is exactly what opens that
column's ≈7pp prompt headroom.

### pets (test = 1110, σ(test) ≈ 1.5pp)

| method | q2b accuracy | q4b accuracy | q9b accuracy | q38b accuracy | Δ q2b (pp) | Δ q4b (pp) | Δ q9b (pp) | Δ q38b (pp) |
|---|---|---|---|---|---|---|---|---|
| raw | 80.9% ±0.3% (n=3)<br>req 296 (judge 0) tok 90412/43942 | 84.2% ±0.3% (n=3)<br>req 296 (judge 0) tok 90412/62181 | 86.2% ±0.2% (n=3)<br>req 296 (judge 0) tok 90412/74559 | 82.4% ±0.7% (n=3)<br>req 296 (judge 0) tok 90412/131094 |  |  |  |  |
| cot | 79.4% ±0.5% (n=3)<br>req 592 (judge 296) tok 263101/82188 | 80.0% ±0.3% (n=3)<br>req 592 (judge 296) tok 301354/120399 | 84.5% ±0.3% (n=3)<br>req 592 (judge 296) tok 285887/104982 | 81.7% ±0.4% (n=3)<br>req 592 (judge 296) tok 297112/116178 | -1.5 | -4.2 | -1.7 | -0.7 |
| evoprompt | 83.9% ±1.4% (n=3)<br>req 19141 (judge 9571) tok 8141319/2247110 | 83.5% ±0.1% (n=3)<br>req 8091 (judge 4045) tok 3347607/872335 | 87.4% ±1.1% (n=3)<br>req 6512 (judge 3256) tok 2746268/770019 | 84.1% ±0.0% (n=3)<br>req 8880 (judge 4440) tok 3966952/1253262 | +3.0 | -0.7 | +1.3 | +1.7 |
| hill_climb | 78.6% ±2.5% (n=3)<br>req 5328 (judge 2664) tok 1870225/528110 | 84.2% ±0.2% (n=3)<br>req 5328 (judge 2664) tok 2103910/762156 | 87.3% ±0.5% (n=3)<br>req 5723 (judge 2861) tok 2250991/804193 | 81.8% ±0.0% (n=3)<br>req 5328 (judge 2664) tok 1864273/522377 | -2.3 | +0.0 | +1.1 | -0.6 |
| opro | 81.6% ±1.7% (n=3)<br>req 8091 (judge 4045) tok 3493706/783831 | 82.8% ±1.0% (n=3)<br>req 10064 (judge 5032) tok 4556214/1271299 | 86.2% ±0.6% (n=3)<br>req 9669 (judge 4835) tok 4572427/1468974 | 83.8% ±1.2% (n=3)<br>req 10064 (judge 5032) tok 5129565/1854387 | +0.7 | -1.4 | +0.1 | +1.4 |
| protegi | 80.5% ±2.2% (n=3)<br>req 2368 (judge 1184) tok 984296/321706 | 82.0% ±1.1% (n=3)<br>req 2155 (judge 1077) tok 918122/333756 | 85.8% ±1.0% (n=3)<br>req 2368 (judge 1184) tok 1297064/603142 | 82.5% ±1.1% (n=3)<br>req 2368 (judge 1184) tok 1162047/534942 | -0.5 | -2.2 | -0.4 | +0.1 |
| spo | 76.3% ±1.2% (n=3)<br>req 1013 (judge 0) tok 767896/235331 | 81.8% ±1.7% (n=3)<br>req 1027 (judge 0) tok 705176/227844 | 86.8% ±2.0% (n=3)<br>req 976 (judge 0) tok 587984/167950 | 81.1% ±1.0% (n=3)<br>req 1103 (judge 0) tok 651361/415190 | -4.6 | -2.4 | +0.6 | -1.3 |
| ape | 81.8% ±0.1% (n=3)<br>req 2763 (judge 1381) tok 1094597/263014 | 83.2% ±1.5% (n=3)<br>req 2368 (judge 1184) tok 982190/257261 | 85.6% ±0.8% (n=3)<br>req 2565 (judge 1283) tok 1180412/335121 | 84.3% ±1.6% (n=3)<br>req 2368 (judge 1184) tok 1089239/345106 | +0.9 | -1.0 | -0.6 | +1.9 |

- **The raw ladder is alive**: 80.9 → 84.2 → 86.2 across
  q2b / q4b / q9b.
- **The B headline lives on cub; pets sits at the tie line**: evoprompt at 2B reaches
  83.9, 0.3 pp under the 4B raw (84.2), so "optimized small ≈
  unoptimized one-size-bigger" is carried by cub (39.9 vs 37.9, § cub),
  with pets as the near-tie witness. On 4B, pets is the matrix's
  do-not-optimize column: nothing clears zero (best hill_climb +0.0, an exact tie;
  evoprompt −0.7, the rest ≤ −1.0).
- **spo-at-2B reads −4.6 ±1.2, and the cell is rerun-unstable**: independent reruns
  of the same cell land at −4.7 ±3.2 / −16.9 ±8.5 / −4.6 ±1.2. Single readings of it
  are meaningless; the published finding is that pets-spo swings across reruns.
- **9B: near-saturation**: evoprompt 87.4 (+1.3) tops the column with hill_climb
  (+1.1) close behind; everything else sits ≤ +0.6.
- **q38b inverts the raw ladder at 27B**: the 27B raw (82.4) lands *below* both the
  4B (84.2) and the 9B (86.2) raws: pets is fine-grained breed recognition and this
  model's out-of-box misses are breed-label confusions, not instruction failures.
  The column optimizes modestly (ape +1.9 / evoprompt +1.7 / opro +1.4), real but
  well under what the raw's deficit "should" have left on the table: prompt search
  recovers style, not breed perception.

### cub (test = 522, grouped fine-grained MCQ)

| method | q2b accuracy | q4b accuracy | q9b accuracy | q38b accuracy | Δ q2b (pp) | Δ q4b (pp) | Δ q9b (pp) | Δ q38b (pp) |
|---|---|---|---|---|---|---|---|---|
| raw | 36.1% ±0.3% (n=3)<br>req 261 (judge 0) tok 64061/46689 | 37.9% ±0.5% (n=3)<br>req 261 (judge 0) tok 64061/59197 | 50.3% ±0.4% (n=3)<br>req 261 (judge 0) tok 64061/74892 | 49.7% ±0.4% (n=3)<br>req 261 (judge 0) tok 64061/137412 |  |  |  |  |
| cot | 35.8% ±0.1% (n=3)<br>req 522 (judge 261) tok 204613/95911 | 37.2% ±0.5% (n=3)<br>req 522 (judge 261) tok 215288/106445 | 46.1% ±1.3% (n=3)<br>req 522 (judge 261) tok 219937/111107 | 53.6% ±0.0% (n=3)<br>req 522 (judge 261) tok 239577/130769 | -0.3 | -0.7 | -4.2 | +3.9 |
| evoprompt | 39.9% ±0.2% (n=3)<br>req 16008 (judge 8004) tok 5870549/2503702 | 39.0% ±0.5% (n=3)<br>req 7308 (judge 3654) tok 2655184/1141839 | 50.8% ±1.0% (n=3)<br>req 6090 (judge 3045) tok 2266187/1016029 | 52.7% ±1.8% (n=3)<br>req 8178 (judge 4089) tok 3836361/2127125 | +3.8 | +1.1 | +0.4 | +2.9 |
| hill_climb | 34.2% ±0.2% (n=3)<br>req 5046 (judge 2523) tok 1463318/510481 | 38.2% ±0.3% (n=3)<br>req 5394 (judge 2697) tok 1637639/616642 | 46.8% ±0.6% (n=3)<br>req 5742 (judge 2871) tok 2086215/998865 | 49.4% ±0.0% (n=3)<br>req 4698 (judge 2349) tok 1528078/645271 | -1.9 | +0.3 | -3.5 | -0.3 |
| opro | 36.6% ±1.8% (n=3)<br>req 8178 (judge 4089) tok 3075497/1066505 | 38.8% ±1.9% (n=3)<br>req 8874 (judge 4437) tok 3406902/1401999 | 48.8% ±0.1% (n=3)<br>req 8874 (judge 4437) tok 3835024/1839359 | 49.7% ±2.3% (n=3)<br>req 8874 (judge 4437) tok 4457633/2406558 | +0.5 | +0.8 | -1.5 | -0.1 |
| protegi | 38.6% ±0.5% (n=3)<br>req 2368 (judge 1184) tok 817376/308579 | 37.1% ±1.2% (n=3)<br>req 2368 (judge 1184) tok 1095274/541758 | 47.5% ±3.0% (n=3)<br>req 2368 (judge 1184) tok 1155931/605217 | 52.0% ±1.3% (n=3)<br>req 2368 (judge 1184) tok 1198510/665006 | +2.6 | -0.8 | -2.8 | +2.2 |
| spo | 32.5% ±3.2% (n=3)<br>req 1022 (judge 0) tok 822884/425437 | 34.9% ±0.7% (n=3)<br>req 942 (judge 0) tok 565626/257807 | 40.9% ±1.7% (n=3)<br>req 968 (judge 0) tok 612792/349984 | 46.2% ±1.2% (n=3)<br>req 1080 (judge 0) tok 610504/533059 | -3.6 | -3.1 | -9.4 | -3.6 |
| ape | 39.7% ±0.4% (n=3)<br>req 1914 (judge 957) tok 621724/238470 | 37.7% ±1.0% (n=3)<br>req 2610 (judge 1305) tok 902662/359400 | 51.3% ±1.2% (n=3)<br>req 2088 (judge 1044) tok 783158/316675 | 52.9% ±1.0% (n=3)<br>req 2610 (judge 1305) tok 1223968/645805 | +3.6 | -0.2 | +1.0 | +3.1 |

- **The B headline cell**: evoprompt at 2B (39.9, +3.8pp) clears the 4B raw
  (37.9) by +2.0pp, so optimized small beats unoptimized bigger, at 61× the raw's
  request bill (16008 vs 261).
- At 2B the field turns positive (evoprompt +3.8, ape +3.6, protegi +2.6, opro +0.5;
  hill_climb −1.9, cot −0.3, spo −3.6); at 4B **nothing clears noise**: evoprompt
  +1.1 leads a band that straddles zero (−3.1..+1.1). Fine-grained perception caps
  the absolute level.
- **9B: the raw is the ceiling**: ape +1.0 is the column's only meaningful positive
  Δ (evoprompt +0.4 is noise); spo −9.4, hill_climb −3.5, cot −4.2 and protegi −2.8
  sit well below. Fine-grained perception is the ceiling at every size up to 9B.
- **cot at q38b is the table's biggest Δ of any column**: 53.6 (+3.9), edging the
  2B evoprompt's +3.8; ape +3.1 is the q38b column's second, over
  a raw (49.7) that sits *just below* the 9B raw. Whatever caps
  optimization at 4B/9B does not cap the next generation's 27B: this column's
  headroom is model-generation-specific, not a monotone function of size.
- cot swings with size, until it doesn't (−0.3 → −0.7 → −4.2 → **+3.9** at 27B):
  the 27B extension column shows the sign is a property of the model-task pairing,
  not of size itself.

### sroie (test = 250, KIE field_acc; zero-judge channel)

| method | q2b field_acc | q4b field_acc | q9b field_acc | q38b field_acc | Δ q2b (pp) | Δ q4b (pp) | Δ q9b (pp) | Δ q38b (pp) |
|---|---|---|---|---|---|---|---|---|
| raw | 81.4% ±0.1% (n=3)<br>req 248 (judge 0) tok 354210/24869 | 88.5% ±0.1% (n=3)<br>req 248 (judge 0) tok 354210/24938 | 81.1% ±0.4% (n=3)<br>req 248 (judge 0) tok 354210/25619 | 87.4% ±0.1% (n=3)<br>req 248 (judge 0) tok 354210/25991 |  |  |  |  |
| cot | 72.6% ±0.6% (n=3)<br>pf 7.1% → parsed 78.2%<br>req 248 (judge 0) tok 363634/86330 | 70.5% ±1.2% (n=3)<br>pf 17.9% → parsed 85.9%<br>req 248 (judge 0) tok 363634/106112 | 87.7% ±0.1% (n=3)<br>req 248 (judge 0) tok 363634/64299 | 78.3% ±0.3% (n=3)<br>req 248 (judge 0) tok 363634/97797 | -8.8 | -18.0 | +6.6 | -9.1 |
| evoprompt | 81.6% ±0.1% (n=3)<br>req 8184 (judge 0) tok 11783087/821837 | 88.3% ±0.0% (n=3)<br>req 4381 (judge 0) tok 6285238/451994 | 87.4% ±0.4% (n=3)<br>req 3141 (judge 0) tok 4493025/327802 | 88.5% ±0.1% (n=3)<br>req 4712 (judge 0) tok 6770993/458484 | +0.2 | -0.2 | +6.3 | +1.1 |
| hill_climb | 82.2% ±0.1% (n=3)<br>req 2728 (judge 0) tok 3929046/275884 | 88.7% ±0.1% (n=3)<br>req 2232 (judge 0) tok 3211698/208068 | 88.4% ±0.0% (n=3)<br>req 2728 (judge 0) tok 3930038/246465 | 87.5% ±0.3% (n=3)<br>req 2232 (judge 0) tok 3211698/186374 | +0.8 | +0.2 | +7.3 | +0.1 |
| opro | 83.2% ±0.5% (n=3)<br>req 3720 (judge 0) tok 5395238/399772 | 88.4% ±0.4% (n=3)<br>req 4216 (judge 0) tok 6114405/424099 | 88.4% ±0.4% (n=3)<br>req 4216 (judge 0) tok 6103410/437179 | 88.3% ±0.3% (n=3)<br>req 4216 (judge 0) tok 6150613/404264 | +1.8 | -0.1 | +7.3 | +0.9 |
| protegi | 80.9% ±0.7% (n=3)<br>req 1147 (judge 0) tok 1643612/113314 | 88.0% ±0.1% (n=3)<br>req 1147 (judge 0) tok 1648799/119403 | 86.6% ±1.4% (n=3)<br>req 1147 (judge 0) tok 1646185/116264 | 87.6% ±0.8% (n=3)<br>req 1147 (judge 0) tok 1654916/112383 | -0.5 | -0.5 | +5.5 | +0.2 |
| spo | 80.3% ±0.8% (n=3)<br>req 820 (judge 0) tok 1442749/133290 | 85.4% ±2.1% (n=3)<br>req 863 (judge 0) tok 1504318/100301 | 85.7% ±1.2% (n=3)<br>req 981 (judge 0) tok 1833862/130373 | 87.6% ±0.2% (n=3)<br>req 968 (judge 0) tok 1696143/93497 | -1.1 | -3.1 | +4.6 | +0.2 |
| ape | 81.2% ±0.1% (n=3)<br>req 1157 (judge 0) tok 1647028/113833 | 88.5% ±0.3% (n=3)<br>req 992 (judge 0) tok 1418411/106257 | 87.0% ±0.7% (n=3)<br>req 1157 (judge 0) tok 1659263/115522 | 88.2% ±0.0% (n=3)<br>req 744 (judge 0) tok 1068417/77852 | -0.2 | -0.0 | +5.9 | +0.8 |

- **Full 8-method coverage held**: the generative-demo pair ran sroie through the
  single-bucket label path; they land mid-band at 2B (ape −0.2 / opro +1.8) but
  take the 9B column with it (opro +7.3 tied for the win, ape +5.9).
- **4B is saturated**: raw 88.5 and every non-CoT optimizer lands −3.1..+0.2
  (hill_climb takes +0.2, inside noise; cot's format collapse is the −18.0 outlier,
  parsed-only −2.6, with 17.9% of its 4B outputs not parsing);
  extraction saturates at 4B.
- **2B gains are real but modest**: opro +1.8 / hill_climb +0.8, the rest ≤ +0.2
  except spo (−1.1) and the cot collapse (−8.8 at 2B, −18.0 at 4B; parsed-only
  −3.2 / −2.6, with 7.1% / 17.9% of the outputs unparseable: the JSON-output
  KIE first-line contract punishes the CoT tax everywhere except the one 9B
  exception).
- **The 9B reversal is consistent across independent runs, same direction, same
  magnitude**: every optimizer beats the 9B raw (opro +7.3 / hill_climb +7.3 / cot
  +6.6 / evoprompt
  +6.3 / ape +5.9 / protegi +5.5 / spo +4.6) because the raw lands at its own 2B
  level instead of the 4B one, leaving ≈7pp of prompt headroom. Three independent
  runs read ≈+7pp: the reversal is a property of the model-task pair, not of a
  lucky run.
- **q38b behaves like 4B, not like 9B**: the 27B raw (87.4) sits near the 4B level
  (88.5), so the 9B reversal does not extend up the model generations: it is that
  model's own quirk, not a size effect. The column then reproduces the
  4B saturation shape: evoprompt +1.1 / opro +0.9 clear the band only marginally,
  and cot collapses again (−9.1): the KIE format contract, not model size, owns
  that tax.

### textvqa (test = 250, σ(test) ≈ 2.6pp)

| method | q2b accuracy | q4b accuracy | q9b accuracy | q38b accuracy | Δ q2b (pp) | Δ q4b (pp) | Δ q9b (pp) | Δ q38b (pp) |
|---|---|---|---|---|---|---|---|---|
| raw | 78.8% ±0.4% (n=3)<br>req 248 (judge 0) tok 196942/879 | 79.2% ±0.2% (n=3)<br>req 248 (judge 0) tok 196942/955 | 76.9% ±0.0% (n=3)<br>req 248 (judge 0) tok 196942/967 | 83.5% ±0.1% (n=3)<br>req 248 (judge 0) tok 196942/884 |  |  |  |  |
| cot | 71.4% ±0.3% (n=3)<br>req 248 (judge 0) tok 201902/1512 | 68.4% ±0.0% (n=3)<br>req 248 (judge 0) tok 201902/2772 | 36.7% ±0.4% (n=3)<br>req 248 (judge 0) tok 201902/5779 | 78.8% ±0.0% (n=3)<br>req 248 (judge 0) tok 201902/6798 | -7.4 | -10.8 | -40.2 | -4.7 |
| evoprompt | 77.3% ±1.4% (n=3)<br>req 9093 (judge 0) tok 7260887/88414 | 84.2% ±0.3% (n=3)<br>req 3555 (judge 0) tok 2820686/31365 | 85.9% ±0.0% (n=3)<br>req 2728 (judge 0) tok 2159418/39434 | 83.4% ±0.4% (n=3)<br>req 4051 (judge 0) tok 3216802/34805 | -1.4 | +5.0 | +8.9 | -0.1 |
| hill_climb | 78.4% ±0.3% (n=3)<br>req 2232 (judge 0) tok 1795294/7772 | 79.2% ±0.2% (n=3)<br>req 2232 (judge 0) tok 1796286/8387 | 76.9% ±0.0% (n=3)<br>req 2232 (judge 0) tok 1795294/9332 | 83.1% ±0.0% (n=3)<br>req 2232 (judge 0) tok 1795294/6648 | -0.3 | +0.0 | +0.0 | -0.4 |
| opro | 78.4% ±0.3% (n=3)<br>req 2893 (judge 0) tok 2317249/96334 | 81.9% ±1.3% (n=3)<br>req 4051 (judge 0) tok 3240693/20277 | 82.4% ±1.2% (n=3)<br>req 4216 (judge 0) tok 3358347/23612 | 82.8% ±0.6% (n=3)<br>req 4216 (judge 0) tok 3385710/15433 | -0.3 | +2.7 | +5.5 | -0.8 |
| protegi | 78.4% ±0.3% (n=3)<br>req 837 (judge 0) tok 682215/13902 | 78.9% ±2.2% (n=3)<br>req 1147 (judge 0) tok 937855/47148 | 76.0% ±2.6% (n=3)<br>req 1147 (judge 0) tok 939446/13460 | 79.8% ±1.9% (n=3)<br>req 1147 (judge 0) tok 946297/4056 | -0.3 | -0.3 | -0.9 | -3.7 |
| spo | 76.4% ±0.0% (n=3)<br>req 992 (judge 0) tok 1076604/45434 | 78.4% ±8.3% (n=3)<br>req 800 (judge 0) tok 804584/20243 | 82.5% ±1.1% (n=3)<br>req 912 (judge 0) tok 987787/23722 | 79.8% ±4.5% (n=3)<br>req 1035 (judge 0) tok 1008546/13260 | -2.4 | -0.8 | +5.6 | -3.7 |
| ape | 78.4% ±0.3% (n=3)<br>req 1240 (judge 0) tok 977931/90163 | 79.4% ±0.5% (n=3)<br>req 1157 (judge 0) tok 915756/60179 | 76.9% ±0.0% (n=3)<br>req 909 (judge 0) tok 723195/29997 | 82.8% ±0.3% (n=3)<br>req 827 (judge 0) tok 665319/21507 | -0.3 | +0.2 | +0.0 | -0.7 |

- The 4B column is where prompts pay: evoprompt +5.0 / opro +2.7 / ape +0.2 over a
  raw that sits at the 2B level (79.2 vs 78.8). At 2B nothing beats raw: four
  methods pin at 78.4 and the raw's 78.8 is the column ceiling.
- cot is negative at every size (−7.4 / −10.8 / −40.2 vs raw), the docs/07 §3
  answer-span mechanism, reproduced at scale; the 9B cell collapses hardest (36.7):
  the CoT prompt's long reasoning output breaks the judgeless exact-match span, and
  the tax grows with model verbosity.
- **9B is the matrix's biggest prompt win**: evoprompt 85.9 (+8.9, one of the
  run set's two largest positive deltas, tied with cub ivl2b cot) / spo 82.5 (+5.6) /
  opro 82.4 (+5.5). The raw row itself reads
  *below* its 2B sibling (76.9 vs 78.8): textvqa's prompt headroom *grows* with model
  size within Qwen3.5, the one column in the matrix where that happens.
- **q38b closes the headroom the 9B row opened**: the 27B raw (83.5) is the highest
  textvqa raw in the matrix and the column is saturated: nothing clears it (best
  cell evoprompt −0.1). The headroom curve is not a size curve: it peaked where the
  9B raw underperformed its own siblings and vanished where the raw finally reads
  strong: prompt headroom tracks *raw deficit*, not scale.

### nextqa (test = 250, video MCQ; one 32-frame clip per sample)

| method | q2b accuracy | q4b accuracy | q9b accuracy | q38b accuracy | Δ q2b (pp) | Δ q4b (pp) | Δ q9b (pp) | Δ q38b (pp) |
|---|---|---|---|---|---|---|---|---|
| raw | 66.4% ±0.0% (n=3)<br>req 248 (judge 0) tok 1070634/1054 [cost n=2] | 63.5% ±0.8% (n=3)<br>pf 18.9% → parsed 78.3%<br>req 248 (judge 0) tok 1070634/9962 | 81.5% ±0.2% (n=3)<br>req 248 (judge 0) tok 1070634/1025 | 80.0% ±1.0% (n=3)<br>req 248 (judge 0) tok 1070634/5151 |  |  |  |  |
| cot | 63.9% ±1.6% (n=3)<br>req 496 (judge 248) tok 1120835/12396 [cost n=2] | 76.3% ±1.0% (n=3)<br>req 496 (judge 248) tok 1153434/45031 | 79.9% ±1.0% (n=3)<br>req 496 (judge 248) tok 1130463/21874 | 80.1% ±0.7% (n=3)<br>req 496 (judge 248) tok 1135344/26716 | -2.5 | +12.8 | -1.6 | +0.1 |
| evoprompt | 65.5% ±1.3% (n=3)<br>req 14715 (judge 7357) tok 33127620/102653 | 80.5% ±0.4% (n=3)<br>req 6283 (judge 3141) tok 14113994/86453 | 80.7% ±0.5% (n=3)<br>req 5621 (judge 2811) tok 12568222/23595 | 82.0% ±0.9% (n=3)<br>req 7275 (judge 3637) tok 16382988/127230 | -0.9 | +17.1 | -0.8 | +2.0 |
| hill_climb | 66.8% ±0.0% (n=3)<br>req 4464 (judge 2232) tok 9925938/34383 [cost n=2] | 76.8% ±0.3% (n=3)<br>req 4464 (judge 2232) tok 9990476/101317 | 81.2% ±0.3% (n=3)<br>req 4464 (judge 2232) tok 9956986/65601 | 80.0% ±1.8% (n=3)<br>req 4464 (judge 2232) tok 10028946/137447 | +0.4 | +13.3 | -0.3 | -0.0 |
| opro | 68.0% ±3.4% (n=3)<br>req 6283 (judge 3141) tok 14134179/29489 | 77.5% ±0.7% (n=3)<br>req 8432 (judge 4216) tok 19044162/145019 | 80.7% ±0.5% (n=3)<br>req 7605 (judge 3803) tok 17083732/31583 | 81.6% ±1.4% (n=3)<br>req 8267 (judge 4133) tok 19029993/432344 | +1.6 | +14.0 | -0.8 | +1.6 |
| protegi | 68.0% ±0.9% (n=3)<br>req 1984 (judge 992) tok 4393033/19378 | 76.9% ±1.6% (n=3)<br>req 2294 (judge 1147) tok 5136395/51336 | 80.0% ±1.0% (n=3)<br>req 2294 (judge 1147) tok 5069538/22704 | 80.1% ±2.5% (n=3)<br>req 2087 (judge 1044) tok 4626112/34665 | +1.6 | +13.5 | -1.5 | +0.1 |
| spo | 60.1% ±4.7% (n=3)<br>pf 11.3% → parsed 67.9%<br>req 472 (judge 0) tok 2164667/63651 | 75.9% ±4.1% (n=3)<br>req 264 (judge 0) tok 1158190/4042 | 81.6% ±0.0% (n=3)<br>req 341 (judge 0) tok 1512344/29593 | 79.3% ±0.4% (n=3)<br>req 347 (judge 0) tok 1531098/26893 | -6.3 | +12.4 | +0.1 | -0.7 |
| ape | 66.3% ±0.5% (n=3)<br>req 1984 (judge 992) tok 4449509/9905 | 79.5% ±0.4% (n=3)<br>req 1984 (judge 992) tok 4482852/29341 | 80.8% ±0.9% (n=3)<br>req 2480 (judge 1240) tok 5566854/10426 | 80.7% ±2.1% (n=3)<br>req 2315 (judge 1157) tok 5261987/40230 | -0.1 | +16.0 | -0.7 | +0.7 |

- **The raw ladder is not monotonic on video**: 66.4 (2B) → 63.5 (4B) → 81.5 (9B) →
  80.0 (27B). The 4B raw sits *below* its 2B sibling, and that deficit is exactly where
  the column's headroom opens: every one of the seven optimizers clears +12 pp of
  protocol accuracy on 4B
  (evoprompt +17.1, ape +16.0, opro +14.0, protegi +13.5, hill_climb +13.3, cot +12.8,
  spo +12.4), the only column in the matrix where that happens.
- **The 4B "win" is answer-format compliance, not video reasoning**: the q4b raw
  cell writes 9 962 output tokens across the test set, 9.5× its 2B sibling on the
  identical prompt (1 054), and 19% of that prose does not parse. A
  format-controlled arm, the same default prompt plus one fixed sentence ("Reply
  with exactly one of the listed options and nothing else."), reads 80.0% with
  zero parse failures on all three seeds and 1 010 output tokens: +16.5 pp over the
  published raw, above the parsed-only estimate (78.3), level with the column's best
  optimizer (evoprompt 80.5, +0.5) and ahead of the other six (0.5-4.1 pp below the
  control). The same sentence costs 1.4 pp on q9b (80.1 vs 81.5), a column that
  already parses, so the format-corrected 4B level is ≈81.4; no optimizer clears
  it. The column's +12.4..+17.1 pp is what the default prompt's output contract
  costs on video inputs, not learned video reasoning. The arm is an out-of-band
  control (§3), not part of the published 1008-run set.
- **The top of the ladder is saturated**: q9b raw 81.5 is the column ceiling (best
  cell +0.1), q38b raw 80.0 buys at most +2.0 (evoprompt), ivl8b raw 83.2 is untouched
  (0.0) and ivl2b raw 70.0 is untouched too: on the second family the 2B and 8B
  columns are saturated while only 4B pays (+2.4 ape / +2.3 evoprompt).
- **cot is negative in six of seven columns** (−1.6..−9.6 protocol, ivl8b worst;
  parsed-only −1.5..−9.5, ivl8b still worst: this is the video CoT tax, not
  format); long reasoning text breaks the option-letter span the judge extracts.
  spo's small-model protocol deficits, by contrast, are format: its q2b and ivl2b
  prompts carried 11-13% parse failures, and parsed-only it reads 67.9 (raw 67.8)
  and 70.1 (raw 70.0).

### gsm8k (test = 250, numeric math QA; text only)

| method | q2b accuracy | q4b accuracy | q9b accuracy | q38b accuracy | Δ q2b (pp) | Δ q4b (pp) | Δ q9b (pp) | Δ q38b (pp) |
|---|---|---|---|---|---|---|---|---|
| raw | 79.9% ±0.2% (n=3)<br>req 248 (judge 0) tok 25697/98462 | 92.4% ±0.3% (n=3)<br>req 248 (judge 0) tok 25697/82163 | 96.4% ±0.0% (n=3)<br>req 248 (judge 0) tok 25697/92058 | 96.8% ±0.0% (n=3)<br>req 248 (judge 0) tok 25697/96576 |  |  |  |  |
| cot | 81.7% ±0.2% (n=3)<br>req 248 (judge 0) tok 29665/77778 | 91.6% ±0.0% (n=3)<br>req 248 (judge 0) tok 29665/79875 | 95.6% ±0.0% (n=3)<br>req 248 (judge 0) tok 29665/86766 | 95.7% ±0.2% (n=3)<br>req 248 (judge 0) tok 29665/90888 | +1.9 | -0.8 | -0.8 | -1.1 |
| evoprompt | 77.3% ±1.5% (n=3)<br>req 7523 (judge 0) tok 809897/2352774 | 90.9% ±0.5% (n=3)<br>req 3803 (judge 0) tok 387573/1080936 | 93.2% ±0.0% (n=3)<br>req 2893 (judge 0) tok 286076/776083 | 96.0% ±0.0% (n=3)<br>req 4877 (judge 0) tok 524057/1703100 | -2.5 | -1.5 | -3.2 | -0.8 |
| hill_climb | 79.1% ±0.4% (n=3)<br>req 2232 (judge 0) tok 254089/906959 | 92.5% ±1.3% (n=3)<br>req 3224 (judge 0) tok 375725/1046435 | 95.5% ±0.2% (n=3)<br>req 2893 (judge 0) tok 336172/1032693 | 96.1% ±0.4% (n=3)<br>req 2893 (judge 0) tok 335180/1121068 | -0.8 | +0.1 | -0.9 | -0.7 |
| opro | 78.9% ±1.1% (n=3)<br>req 3885 (judge 0) tok 428874/1504048 | 93.2% ±0.3% (n=3)<br>req 4133 (judge 0) tok 455398/1356353 | 94.8% ±0.3% (n=3)<br>req 4216 (judge 0) tok 443545/1482163 | 96.9% ±0.7% (n=3)<br>req 4133 (judge 0) tok 454323/1437194 | -0.9 | +0.8 | -1.6 | +0.1 |
| protegi | 76.4% ±0.9% (n=3)<br>req 672 (judge 0) tok 71357/207938 | 91.7% ±2.0% (n=3)<br>req 754 (judge 0) tok 91444/311674 | 93.2% ±2.5% (n=3)<br>req 837 (judge 0) tok 118393/449373 | 93.7% ±3.8% (n=3)<br>req 1147 (judge 0) tok 140445/470377 | -3.5 | -0.7 | -3.2 | -3.1 |
| spo | 75.1% ±0.5% (n=3)<br>req 248 (judge 0) tok 18076/66160 | 91.1% ±0.2% (n=3)<br>req 248 (judge 0) tok 18076/66057 | 94.8% ±0.0% (n=3)<br>req 248 (judge 0) tok 18076/62610 | 91.2% ±0.0% (n=3)<br>req 253 (judge 0) tok 19154/76283 | -4.8 | -1.3 | -1.6 | -5.6 |
| ape | 80.1% ±0.5% (n=3)<br>req 744 (judge 0) tok 72627/219269 | 91.6% ±0.9% (n=3)<br>req 909 (judge 0) tok 93726/312757 | 94.8% ±0.6% (n=3)<br>req 827 (judge 0) tok 89873/292930 | 96.8% ±0.3% (n=3)<br>req 909 (judge 0) tok 102902/238300 | +0.3 | -0.8 | -1.6 | +0.0 |

- **The most saturated board in the matrix**: raw 79.9 / 92.4 / 96.4 / 96.8 on Qwen,
  83.1 / 92.4 / 94.8 on InternVL. Only the 2B column has real headroom: cot +1.9 at
  q2b and evoprompt +1.6 at ivl2b are the only ≥1 pp positives across all seven columns.
- **No optimizer wins twice**: q2b cot, q4b opro (+0.8), q9b saturated (−0.8 at best),
  q38b opro (+0.1), ivl2b evoprompt, ivl4b protegi (+0.8), ivl8b evoprompt (+0.8). On a
  board whose raw prompt already elicits the chain and the final line, the method layer
  has nothing structural to add.
- **spo is negative in every column and the largest negative in the matrix**
  (−4.8 / −1.3 / −1.6 / −5.6 on Qwen, −8.9 / −8.1 / −10.4 on InternVL). Its seeds carry
  the answer-format suffix the `raw_template` path renders; on math the default prompt
  already emits the line the scorer extracts, so the template buys nothing and moves
  the answer. Format-carrying templates are not free where the format is already right.
- Parse failures are ~0 on this board (≤0.4% per cell): the final-answer extractor
  tolerates a whole reasoning passage, so gsm8k's numbers are not format-diluted the
  way nextqa's q4b raw is.

### C — cost frontier (judge channel + tokens)

Visible in every row above (`req` includes `judge` requests; tokens per run):

- Judge-using methods bill the judge as ~half their requests (exactly the eval-split
  count per full evaluation); sroie / textvqa bill zero judge: KIE and VQA are
  rule-scored by design.
- spo's OvO selection votes ride the optimizer *generate* channel (compare tier), so
  they appear in `req` but never in `judge_requests`; its loop is still
  judge-channel-free and cheap (pets 2B: 1013 req), votes already inside that number.
- **The price of pool search**: pets evoprompt-at-2B bills 19141 req (9571 judge) +
  8.1 M input tokens for its +3.0pp; the same method on the 27B extension bills
  8880 req (4440 judge) + 4.0 M input tokens for +1.7pp (≈6.4 vs ≈5.2 kreq/pp):
  pool search spends at a rate set by the search, not by the payoff, and spends
  hardest where the raw's error mode is perceptual rather than instructive.
- **The cheapest real wins are judge-free**: sroie-at-9B hill_climb (+7.3pp, 2728 req,
  zero judge) and textvqa-at-9B evoprompt (+8.9pp, 2728 req, zero judge); on the
  judge side, cub-at-2B ape buys +3.6pp for 1914 req (957 judge).
- **Math is judge-free and request-cheap**: gsm8k bills zero judge on all seven
  columns (rule-scored final answer), the raw / cot / spo cells are single 248-request
  passes, and even evoprompt's q2b cell bills 7523 req, the smallest search bill of
  any board. What these columns spend is *decode time*, not requests: the q38b cell
  takes 24-35 min of generation on 2.9-4.9 kreq.
- **Video bills one judge per evaluated sample like the business boards, but pays in
  tokens**: nextqa's judge-using cells bill half their requests through the judge
  (cot 496 = 248 + 248), spo stays judge-free at 264-531 req, and the q38b column
  spends 51-103 min on 2-8 kreq, 3-5× the identical request count on a 9B, because
  every request carries a 32-frame clip (~12 k video prompt tokens).

### D — Second family: InternVL3.5-2B / 4B / 8B

The family-independence claim tested: the same bench protocol re-run on a non-Qwen
family at three sizes (InternVL3.5-2B / 4B / 8B), all over all six datasets at
seeds 42/43/44, all 8 methods (sroie included). Every cell renders with the
model's own HF chat template (`profile: hf_template`), so the cross-family reading
is template-for-template. Every §D cell is n=3.

**Reading discipline**: cross-family raw accuracies are *not* comparable:
different vision encoders and image-token budgets (the identical pets raw cell
bills 0.77 M input tokens on InternVL vs 0.09 M on Qwen, same 296 requests). The
valid cross-family reading is the **delta pattern**: each cell minus its own
family's own raw, compared cell by cell.

**pets** (test = 1110)

| method | ivl2b accuracy | ivl4b accuracy | ivl8b accuracy | Δ ivl2b (pp) | Δ ivl4b (pp) | Δ ivl8b (pp) |
|---|---|---|---|---|---|---|
| raw | 47.0% ±0.1% (n=3)<br>req 296 (judge 0) tok 772448/15468 | 51.4% ±0.0% (n=3)<br>req 296 (judge 0) tok 772448/5263 | 53.6% ±0.1% (n=3)<br>req 296 (judge 0) tok 772448/23637 |  |  |  |
| cot | 43.4% ±0.6% (n=3)<br>req 592 (judge 296) tok 963764/104341 | 48.9% ±0.3% (n=3)<br>req 592 (judge 296) tok 907393/47647 | 48.5% ±0.1% (n=3)<br>req 592 (judge 296) tok 923850/63933 | -3.7 | -2.5 | -5.1 |
| evoprompt | 49.0% ±1.4% (n=3)<br>req 23285 (judge 11643) tok 34891129/929077 | 56.7% ±0.1% (n=3)<br>req 13221 (judge 6611) tok 19441820/196145 | 58.0% ±0.0% (n=3)<br>req 8091 (judge 4045) tok 11987929/223280 | +2.0 | +5.2 | +4.4 |
| hill_climb | 47.9% ±0.1% (n=3)<br>req 6512 (judge 3256) tok 9718083/610210 | 53.6% ±0.3% (n=3)<br>req 5328 (judge 2664) tok 7649740/188863 | 53.9% ±0.2% (n=3)<br>req 6117 (judge 3059) tok 8839380/282261 | +0.9 | +2.2 | +0.3 |
| opro | 48.6% ±0.1% (n=3)<br>req 9867 (judge 4933) tok 14929060/487125 | 53.2% ±0.4% (n=3)<br>req 8485 (judge 4243) tok 12451828/94171 | 54.7% ±1.0% (n=3)<br>req 6315 (judge 3157) tok 9287729/113616 | +1.6 | +1.8 | +1.1 |
| protegi | 47.9% ±1.2% (n=3)<br>req 2368 (judge 1184) tok 3421583/189757 | 52.7% ±0.1% (n=3)<br>req 2368 (judge 1184) tok 3329219/88053 | 53.8% ±2.0% (n=3)<br>req 2368 (judge 1184) tok 3336032/101353 | +0.9 | +1.3 | +0.2 |
| spo | 45.6% ±2.1% (n=3)<br>req 987 (judge 0) tok 2812312/78593 | 51.9% ±0.8% (n=3)<br>req 741 (judge 0) tok 1966923/13514 | 47.9% ±4.7% (n=3)<br>req 768 (judge 0) tok 2028304/24931 | -1.5 | +0.4 | -5.7 |
| ape | 48.6% ±0.6% (n=3)<br>req 2763 (judge 1381) tok 4090855/82825 | 52.7% ±0.5% (n=3)<br>req 2368 (judge 1184) tok 3460649/26468 | 55.2% ±1.3% (n=3)<br>req 1579 (judge 789) tok 2328292/43261 | +1.6 | +1.2 | +1.6 |

**cub** (test = 522)

| method | ivl2b accuracy | ivl4b accuracy | ivl8b accuracy | Δ ivl2b (pp) | Δ ivl4b (pp) | Δ ivl8b (pp) |
|---|---|---|---|---|---|---|
| raw | 21.8% ±0.5% (n=3)<br>pf 14.6% → parsed 25.6%<br>req 261 (judge 0) tok 602994/28379 | 27.6% ±0.0% (n=3)<br>pf 8.4% → parsed 30.1%<br>req 261 (judge 0) tok 602994/7196 | 34.1% ±0.5% (n=3)<br>req 261 (judge 0) tok 602994/24780 |  |  |  |
| cot | 30.8% ±0.7% (n=3)<br>req 522 (judge 261) tok 717994/73542 | 27.9% ±0.2% (n=3)<br>req 522 (judge 261) tok 707112/62693 | 36.7% ±0.2% (n=3)<br>req 522 (judge 261) tok 715106/70688 | +8.9 | +0.3 | +2.6 |
| evoprompt | 29.1% ±0.9% (n=3)<br>req 20706 (judge 10353) tok 27599594/1839404 | 30.5% ±0.1% (n=3)<br>req 10614 (judge 5307) tok 13439951/321178 | 36.1% ±0.6% (n=3)<br>req 7308 (judge 3654) tok 9412297/408219 | +7.2 | +2.9 | +2.0 |
| hill_climb | 30.2% ±0.1% (n=3)<br>req 5742 (judge 2871) tok 7689147/720627 | 31.2% ±0.8% (n=3)<br>req 4698 (judge 2349) tok 5924249/210038 | 36.5% ±1.0% (n=3)<br>req 5742 (judge 2871) tok 7398621/422838 | +8.4 | +3.6 | +2.4 |
| opro | 27.9% ±1.6% (n=3)<br>req 8700 (judge 4350) tok 11584763/707218 | 33.3% ±0.4% (n=3)<br>req 7482 (judge 3741) tok 9370781/88708 | 33.4% ±0.3% (n=3)<br>req 5046 (judge 2523) tok 6364721/298613 | +6.1 | +5.7 | -0.7 |
| protegi | 29.2% ±0.2% (n=3)<br>req 2368 (judge 1184) tok 3126670/216703 | 31.4% ±2.0% (n=3)<br>req 2368 (judge 1184) tok 3007896/82711 | 36.3% ±0.9% (n=3)<br>req 2368 (judge 1184) tok 3057858/156658 | +7.4 | +3.8 | +2.2 |
| spo | 26.6% ±2.9% (n=3)<br>pf 7.5% → parsed 28.7%<br>req 894 (judge 0) tok 2202249/120220 | 31.7% ±2.8% (n=3)<br>req 618 (judge 0) tok 1468510/10201 | 35.4% ±0.9% (n=3)<br>req 762 (judge 0) tok 1794071/60599 | +4.8 | +4.1 | +1.3 |
| ape | 27.3% ±0.6% (n=3)<br>pf 5.4% → parsed 28.9%<br>req 1914 (judge 957) tok 2469374/116035 | 33.4% ±0.6% (n=3)<br>req 2262 (judge 1131) tok 2819668/31312 | 36.5% ±1.5% (n=3)<br>req 1740 (judge 870) tok 2194979/66277 | +5.5 | +5.8 | +2.4 |

**sroie** (test = 250, field_acc)

| method | ivl2b field_acc | ivl4b field_acc | ivl8b field_acc | Δ ivl2b (pp) | Δ ivl4b (pp) | Δ ivl8b (pp) |
|---|---|---|---|---|---|---|
| raw | 79.0% ±0.1% (n=3)<br>req 248 (judge 0) tok 540736/24622 | 83.7% ±0.2% (n=3)<br>req 248 (judge 0) tok 540736/24134 | 79.9% ±0.1% (n=3)<br>req 248 (judge 0) tok 540736/24440 |  |  |  |
| cot | 63.9% ±0.3% (n=3)<br>pf 8.7% → parsed 70.0%<br>req 248 (judge 0) tok 550408/68536 | 84.8% ±0.0% (n=3)<br>req 248 (judge 0) tok 550408/100564 | 82.2% ±0.2% (n=3)<br>req 248 (judge 0) tok 550408/124493 | -15.1 | +1.0 | +2.3 |
| evoprompt | 79.9% ±0.5% (n=3)<br>req 9341 (judge 0) tok 20524707/989054 | 85.7% ±0.7% (n=3)<br>req 3472 (judge 0) tok 7579315/340667 | 80.8% ±0.6% (n=3)<br>req 2728 (judge 0) tok 5950824/275935 | +0.9 | +2.0 | +0.8 |
| hill_climb | 79.6% ±0.2% (n=3)<br>req 2728 (judge 0) tok 5981824/463452 | 84.9% ±0.2% (n=3)<br>req 2893 (judge 0) tok 6343637/234629 | 81.2% ±0.1% (n=3)<br>req 2728 (judge 0) tok 5980171/402217 | +0.6 | +1.2 | +1.2 |
| opro | 80.1% ±0.8% (n=3)<br>req 3968 (judge 0) tok 8685835/405665 | 85.7% ±1.6% (n=3)<br>req 3637 (judge 0) tok 7900621/353686 | 82.0% ±1.5% (n=3)<br>req 2480 (judge 0) tok 5382395/240020 | +1.1 | +1.9 | +2.1 |
| protegi | 79.6% ±1.6% (n=3)<br>req 1147 (judge 0) tok 2492210/115500 | 84.9% ±0.2% (n=3)<br>req 1147 (judge 0) tok 2498917/113698 | 80.4% ±2.6% (n=3)<br>req 1147 (judge 0) tok 2516308/115233 | +0.5 | +1.2 | +0.4 |
| spo | 78.4% ±0.8% (n=3)<br>req 640 (judge 0) tok 1400328/64573 | 85.6% ±0.2% (n=3)<br>req 592 (judge 0) tok 1296627/62514 | 81.0% ±3.6% (n=3)<br>req 767 (judge 0) tok 1700953/82967 | -0.6 | +1.9 | +1.0 |
| ape | 80.5% ±1.3% (n=3)<br>req 1075 (judge 0) tok 2358152/112795 | 86.8% ±0.2% (n=3)<br>req 1157 (judge 0) tok 2535091/108948 | 80.3% ±0.3% (n=3)<br>req 909 (judge 0) tok 1986584/89657 | +1.4 | +3.1 | +0.4 |

**textvqa** (test = 250)

| method | ivl2b accuracy | ivl4b accuracy | ivl8b accuracy | Δ ivl2b (pp) | Δ ivl4b (pp) | Δ ivl8b (pp) |
|---|---|---|---|---|---|---|
| raw | 75.3% ±0.0% (n=3)<br>req 248 (judge 0) tok 653302/954 | 69.3% ±0.0% (n=3)<br>req 248 (judge 0) tok 653302/1056 | 75.9% ±0.0% (n=3)<br>req 248 (judge 0) tok 653302/937 |  |  |  |
| cot | 65.6% ±0.0% (n=3)<br>req 248 (judge 0) tok 658262/3274 | 56.5% ±0.0% (n=3)<br>req 248 (judge 0) tok 658262/8686 | 69.6% ±0.0% (n=3)<br>req 248 (judge 0) tok 658262/6741 | -9.7 | -12.8 | -6.3 |
| evoprompt | 76.5% ±0.4% (n=3)<br>req 9837 (judge 0) tok 25937707/65503 | 75.3% ±0.0% (n=3)<br>req 5208 (judge 0) tok 13705123/44258 | 80.7% ±0.0% (n=3)<br>req 2893 (judge 0) tok 7615905/18777 | +1.2 | +6.0 | +4.8 |
| hill_climb | 73.3% ±0.0% (n=3)<br>req 2728 (judge 0) tok 7220050/13015 | 70.1% ±0.0% (n=3)<br>req 1736 (judge 0) tok 4587994/12295 | 76.5% ±0.0% (n=3)<br>req 2232 (judge 0) tok 5904518/13487 | -2.0 | +0.8 | +0.7 |
| opro | 75.9% ±0.8% (n=3)<br>req 4051 (judge 0) tok 10645221/200099 | 75.5% ±0.9% (n=3)<br>req 3885 (judge 0) tok 10202163/47100 | 78.4% ±0.6% (n=3)<br>req 2563 (judge 0) tok 6725078/58580 | +0.5 | +6.2 | +2.5 |
| protegi | 75.6% ±0.1% (n=3)<br>req 1147 (judge 0) tok 3010813/28150 | 73.3% ±0.8% (n=3)<br>req 1147 (judge 0) tok 3001771/15593 | 76.0% ±2.3% (n=3)<br>req 1147 (judge 0) tok 3000293/6612 | +0.3 | +4.0 | +0.2 |
| spo | 75.1% ±0.0% (n=3)<br>req 847 (judge 0) tok 2366382/14123 | 71.3% ±0.3% (n=3)<br>req 619 (judge 0) tok 1650209/9724 | 73.9% ±0.4% (n=3)<br>req 709 (judge 0) tok 1918711/11964 | -0.3 | +2.0 | -2.0 |
| ape | 75.3% ±0.0% (n=3)<br>req 1157 (judge 0) tok 3043204/57293 | 69.9% ±0.8% (n=3)<br>req 1240 (judge 0) tok 3277587/27042 | 75.9% ±0.0% (n=3)<br>req 1240 (judge 0) tok 3274198/46302 | +0.0 | +0.5 | +0.0 |

**nextqa** (test = 250, video MCQ)

| method | ivl2b accuracy | ivl4b accuracy | ivl8b accuracy | Δ ivl2b (pp) | Δ ivl4b (pp) | Δ ivl8b (pp) |
|---|---|---|---|---|---|---|
| raw | 70.0% ±0.0% (n=3)<br>req 248 (judge 0) tok 2099306/3207 [cost n=2] | 74.4% ±0.0% (n=3)<br>req 248 (judge 0) tok 2099306/2305 | 83.2% ±0.0% (n=3)<br>req 248 (judge 0) tok 2099306/1987 |  |  |  |
| cot | 64.5% ±0.9% (n=3)<br>req 496 (judge 248) tok 2157062/26608 [cost n=2] | 71.6% ±0.9% (n=3)<br>req 496 (judge 248) tok 2170714/36773 | 73.6% ±0.9% (n=3)<br>req 496 (judge 248) tok 2178963/45009 | -5.5 | -2.8 | -9.6 |
| evoprompt | 69.3% ±0.8% (n=3)<br>req 17691 (judge 8845) tok 76564726/302282 | 76.7% ±0.7% (n=3)<br>req 11243 (judge 5621) tok 48478808/79312 | 81.2% ±0.0% (n=3)<br>req 6613 (judge 3307) tok 28498331/58454 | -0.7 | +2.3 | -2.0 |
| hill_climb | 70.0% ±0.0% (n=3)<br>req 4960 (judge 2480) tok 21376842/161708 [cost n=2] | 73.9% ±2.5% (n=3)<br>req 4464 (judge 2232) tok 19300562/168552 | 83.2% ±0.0% (n=3)<br>req 5125 (judge 2563) tok 22215979/273599 | +0.0 | -0.5 | +0.0 |
| opro | 63.9% ±2.0% (n=3)<br>req 8101 (judge 4051) tok 35019182/167987 | 74.0% ±0.0% (n=3)<br>req 7109 (judge 3555) tok 30604579/45345 | 81.6% ±1.2% (n=3)<br>req 4795 (judge 2397) tok 20623117/25571 | -6.1 | -0.4 | -1.6 |
| protegi | 64.4% ±1.2% (n=3)<br>req 1881 (judge 940) tok 8121335/64183 | 74.8% ±1.0% (n=3)<br>req 2294 (judge 1147) tok 9893371/29150 | 82.4% ±0.9% (n=3)<br>req 2294 (judge 1147) tok 9875207/23152 | -5.6 | +0.4 | -0.8 |
| spo | 60.9% ±0.2% (n=3)<br>pf 13.1% → parsed 70.1%<br>req 531 (judge 0) tok 4614254/59577 | 73.5% ±2.1% (n=3)<br>req 619 (judge 0) tok 5291551/5062 | 81.1% ±1.5% (n=3)<br>req 704 (judge 0) tok 5977340/5670 | -9.1 | -0.9 | -2.1 |
| ape | 66.8% ±2.3% (n=3)<br>req 1157 (judge 579) tok 5005287/14971 | 76.8% ±0.3% (n=3)<br>req 2315 (judge 1157) tok 9996478/17292 | 82.4% ±0.7% (n=3)<br>req 2149 (judge 1075) tok 9293849/18128 | -3.2 | +2.4 | -0.8 |

**gsm8k** (test = 250, text only)

| method | ivl2b accuracy | ivl4b accuracy | ivl8b accuracy | Δ ivl2b (pp) | Δ ivl4b (pp) | Δ ivl8b (pp) |
|---|---|---|---|---|---|---|
| raw | 83.1% ±0.4% (n=3)<br>req 248 (judge 0) tok 24228/54641 | 92.4% ±0.0% (n=3)<br>req 248 (judge 0) tok 24228/69061 | 94.8% ±0.0% (n=3)<br>req 248 (judge 0) tok 24228/73624 |  |  |  |
| cot | 81.1% ±1.1% (n=3)<br>req 248 (judge 0) tok 27204/49162 | 91.6% ±0.0% (n=3)<br>req 248 (judge 0) tok 27204/60282 | 92.7% ±0.2% (n=3)<br>req 248 (judge 0) tok 27204/53338 | -2.0 | -0.8 | -2.1 |
| evoprompt | 84.7% ±2.5% (n=3)<br>req 9837 (judge 0) tok 1058756/2389085 | 92.3% ±0.2% (n=3)<br>req 4051 (judge 0) tok 384151/1112471 | 95.6% ±0.0% (n=3)<br>req 2728 (judge 0) tok 253364/766441 | +1.6 | -0.1 | +0.8 |
| hill_climb | 83.9% ±0.2% (n=3)<br>req 2232 (judge 0) tok 240868/524656 | 92.4% ±0.0% (n=3)<br>req 2232 (judge 0) tok 240868/601219 | 94.0% ±0.0% (n=3)<br>req 2232 (judge 0) tok 240868/618742 | +0.8 | +0.0 | -0.8 |
| opro | 82.9% ±0.2% (n=3)<br>req 3803 (judge 0) tok 346283/885200 | 92.4% ±0.3% (n=3)<br>req 3389 (judge 0) tok 292015/890895 | 94.9% ±0.2% (n=3)<br>req 2480 (judge 0) tok 186728/619203 | -0.1 | +0.0 | +0.1 |
| protegi | 83.6% ±2.0% (n=3)<br>req 1044 (judge 0) tok 122766/267206 | 93.2% ±0.9% (n=3)<br>req 754 (judge 0) tok 79521/216920 | 93.3% ±3.4% (n=3)<br>req 837 (judge 0) tok 109122/263782 | +0.5 | +0.8 | -1.5 |
| spo | 74.1% ±0.2% (n=3)<br>req 248 (judge 0) tok 16853/55501 | 84.3% ±0.8% (n=3)<br>req 259 (judge 0) tok 22938/73342 | 84.4% ±0.0% (n=3)<br>req 253 (judge 0) tok 19298/67347 | -8.9 | -8.1 | -10.4 |
| ape | 84.0% ±0.6% (n=3)<br>req 992 (judge 0) tok 95341/213752 | 92.4% ±0.0% (n=3)<br>req 827 (judge 0) tok 81421/239060 | 95.2% ±0.3% (n=3)<br>req 992 (judge 0) tok 102203/282790 | +0.9 | +0.0 | +0.4 |

Readings:

- **The protocol and the contract hold on a second family at three sizes**: all 432
  InternVL cells (3 sizes × 6 datasets × 8 methods × 3 seeds) ran on the `hf_template`
  profile with no framework change, and the generation channel produced winners at
  every size.
- **cub is the family-conditional column, and most of the 2B band is answer
  format**: at 2B *every* method moves positive, +4.8..+8.9pp (cot +8.9,
  hill_climb +8.4, protegi +7.4, evoprompt +7.2, opro +6.1, ape +5.5, spo +4.8)
  over a raw at 21.8, while the same column on Qwen 2B peaks at +3.8 with three
  methods negative. But the raw fails to parse on 14.6% of samples, and a
  format-controlled arm (§3) rebuilds it at 28.5% with zero parse failures: +6.7
  over the published raw and above the parsed-only estimate (25.6: conditioning on
  the parseable outputs selects the wrong subpopulation once the format is
  repaired). Against that control the published ordering survives intact at a
  quarter of its magnitude: cot +2.3 and hill_climb +1.7 clear noise,
  evoprompt/protegi +0.6/+0.7 sit at it, opro/ape/spo fall to −0.6/−1.2/−1.9.
  Fine-grained perception caps absolute levels; what "every optimizer wins" was
  mostly measuring is a 2B model that cannot keep the default answer contract on
  small images.
- **pets inverts the Qwen saturation**: evoprompt leads at every size (+2.0 / +5.2 /
  +4.4): on InternVL, pets *optimizes well all the way up the ladder*, unlike the
  Qwen do-not-optimize column; cot is negative everywhere (−2.5..−5.1).
- **sroie gains are broad but shallow** (+0.4..+3.1, ape the 2B winner at +1.4); cot
  collapses at 2B (−15.1 protocol, −9.0 parsed-only: 8.7% of its outputs do not
  parse) then flips positive at 4B/8B (+1.0 / +2.3), the Qwen
  column's mirror pattern, reproduced on the second family.
- **Method-level anti-claims**: spo is best-on-cub everywhere (+4.8 / +4.1 / +1.3)
  but negative elsewhere at these sizes, including pets-8B at −5.7 ±4.7 (its one
  apparent small-model miss on video, nextqa ivl2b −9.1, is 13.1% parse failures:
  parsed-only it sits at raw, 70.1 vs 70.0); and
  protegi's textvqa 2B reading is **rerun-dependent, not a constant**: independent
  reruns land at −12.8 ±4.4, −1.5 ±1.7 and +0.3 ±0.1. The critique-edit loop's
  instability on judgeless VQA is real; that its sign and magnitude are a seed
  lottery is itself the warning.
- **Video repeats the Qwen pattern on the second family**: ivl2b and ivl8b are
  saturated (raw untouched, or ties only), the 4B column is the one that pays (+2.4
  ape / +2.3 evoprompt), and cot is negative at every size (−2.8 / −9.6, ivl8b worst):
  the option-letter span breaks under long reasoning on video for both families.
- **Math is saturated on InternVL too, with one sign flip**: evoprompt +1.6 at 2B and
  +0.8 at 8B are the only gains, and spo is the family's largest systematic negative
  (−8.9 / −8.1 / −10.4): the answer-format template a `raw_template` seed imposes is
  what the default prompt already does right.

### Boundary notes

- **Provenance and rerun noise**: these tables come from a from-scratch run of the
  full matrix (126 sweeps / 1008 runs, all completed; six nextqa 2B cells replayed
  their loop bill from an earlier attempt, §4). Cell spread
  between independent reruns is re-roll noise inside the seed bands except pets-spo,
  whose instability across three independent readings (−4.7 / −16.9 / −4.6) is itself
  a published finding (§4). The extra rerun-stability runs are not part of the
  distributed run set; the distributed tables are the seed-42/43/44 matrix.
- **One clip per sample is a hard ceiling on the video board**: the planner charges
  16 384 tokens per clip, so two clips plus a question exceed the 32 768-token window
  and the request is rejected at plan time. nextqa satisfies this by construction (one
  clip per sample, 32 frames); the mixed-modality loader enforces the same cap.
- **nextqa's q4b column is the one place answer format, not reasoning, moves the
  number**: its raw cell fails to parse on 18-20% of samples (§4 nextqa), and the
  format-controlled arm lands at 80.0, so the column's +12..+17 pp is the output
  contract, while parsed-only raw (0.783) and the control (0.800) bracket the
  reasoning level.
