# Baseline Fidelity Report — Paper Conformance of the Published Baselines

> Trust artifact for the benchmark baselines: for every shipped method, which paper
> and which official repo it claims to implement (at a pinned commit), which contract
> items are verified, which deviations from the paper/official code exist **and are
> admitted here**, and which test pins each contract item. The pinned commits below are
> what the verification was performed against.
>
> Reading rule: anything **not** listed as a deviation and not flagged below is
> implemented per the cited paper section / official file. Where paper and official
> code disagree, the authority order is **paper text > official repo > this
> implementation**, and the disagreement itself is recorded.

## 1. Scope

The framework ships **8 published optimization methods** (registry: raw, cot,
hill_climb, evoprompt, opro, ape, protegi, spo).

The compare **machinery** (`COMPARE_EVAL` capability, `validate_assembly`, the
`PairwiseJudge` extension point) stays implemented and tested: SPO's OvO
(one-vs-one) selection exercises it, and the assembly-rejection guarantee is pinned by a
synthetic in-test compare-only fixture (`tests/test_contract.py::
test_compare_only_fixture_rejected_from_accuracy_board`).

## 2. Authorities (paper + official repo at pinned commit)

Verified by cloning each repo at the pinned commit (sparse checkouts where noted)
and reading the cited files at those hashes. A clone at a different hash is not
authority.

| Method | Paper | Official repo (pinned commit) | Cite scope | Verification |
| --- | --- | --- | --- | --- |
| raw | — (framework baseline: task seed prompt evaluated as-is) | — | — | — |
| cot | — (framework rendering variant: seed prompt + chain-of-thought suffix) | — | — | — |
| hill_climb | — (classical baseline shape; no paper claim made) | — | — | — |
| ape | [APE, Zhou et al., ICLR 2023](https://arxiv.org/abs/2211.01910) | [keirp/automatic_prompt_engineer @ `eac521c7` (main)](https://github.com/keirp/automatic_prompt_engineer/commit/eac521c79a78965245ce7745dcc9f6b0792c7ec7) | forward induction + `demos_template` | re-verified at the pinned commit |
| opro | [OPRO, Yang et al., ICLR 2024](https://arxiv.org/abs/2309.03409) | [google-deepmind/opro @ `a76bdce2`](https://github.com/google-deepmind/opro/commit/a76bdce2cbf6d4a0d1e570a6fcfe17be9c2abdd7) | `opro/` package decision process | re-verified at the pinned commit |
| evoprompt | [EvoPrompt, Guo et al., ICLR 2024](https://arxiv.org/abs/2309.08532) | [beeevita/EvoPrompt @ `94caff33`](https://github.com/beeevita/EvoPrompt/commit/94caff336555df99acc5c338c7930c40cc550ad9) | genetic-algorithm / differential-evolution (GA/DE) operators + selection | re-verified at the pinned commit |
| protegi | [ProTeGi, Pryzant et al., EMNLP 2023](https://arxiv.org/abs/2305.03495) | [microsoft/LMOps @ `4f2a9deb` (sparse `prompt_optimization/`)](https://github.com/microsoft/LMOps/commit/4f2a9deb5f08e459fd44c2e4792344d78ca89fc3) | textual gradient + beam + UCB | re-verified at the pinned commit |
| spo | [SPO, Xiang et al., Findings of EMNLP 2025](https://arxiv.org/abs/2502.06855) | [FoundationAgents/SPO @ `e8381f07`](https://github.com/FoundationAgents/SPO/commit/e8381f073a543a8b324c9f3f993c7563d02d5087) | OvO loop + `EVALUATION_REPETITION` | re-verified at the pinned commit |

## 3. Cross-method decisions and framework-side fixes

Decisions that apply across methods:

- **Judge / optimizer model identity.** Compare judging and OvO ballots use the
  **same deployment model** as generation; there is no separate judge model. This
  is a deliberate decision (it keeps the cost profile honest and the deployment
  surface single-knob).
- **`AnswerPairwiseJudge` vs `DefaultPairwiseJudge`.** `AnswerPairwiseJudge`
  (registry name `answer`) is answer-vs-answer, the paper's OvO: two model
  *answers* to one question, verdict A/B/tie, **no reference field at all**.
  `DefaultPairwiseJudge` compares *instruction texts* (the framework
  `CompareEvalModel` path). SPO selection structurally cannot see gold labels.
- **Extraction profiles (`eyas/engine/genio.py`).** `single_bracket` (opro),
  `single_tag` (evoprompt, spo, protegi edits/synonyms) and `lines` (multi-line
  instruction pools, via `parse_instructions`, available to custom methods) replace
  the one-size line-split. APE does not take a profile: one `clean_instruction`
  pass yields its single candidate. The 500-char cap lives in
  `clean_instruction` (and therefore in `lines`) and is registered as a
  deviation there (D6).
- **Split-qualified, type-tagged record store (schema stamp 3).** Outcome rows
  are keyed by `(namespace, prompt_id, sample_id, split)` (the store's
  namespace plus the prompt's content hash), and the prediction column is
  type-tagged (`s:` str / `j:` JSON / NULL for None), so a free-text answer
  that merely looks like JSON round-trips as a str (resume byte-fidelity); a
  store written under a different schema is refused at
  open with an explicit version-mismatch error (silent record-miss replay would
  hide the mismatch). Such stores are deliberately **not** resume-compatible and
  there is no migration tool; starting a fresh run dir is the supported path.
- **Deterministic ranking.** All strategy sorts key on `(-score, prompt_hash)`
  (`tests/test_ranking.py`). The larger-evidence tie-break lives in
  **`Archive.add`** (not `Archive.ranked`): on an equal score the record with the
  **larger per-sample evidence set** replaces the incumbent, so a tied subset
  record cannot permanently mask the closing full-split record; equal score *and*
  equal size keep the first writer. `Archive.ranked` itself only sorts by
  descending score with remaining ties broken ascending by prompt id.
- **`Decision.items` (typed `WorkItem(prompt, sample_ids)`)**: an evaluate round
  carrying explicit work items; `budget.takeover(round_idx)` tells allocators the
  round's load is strategy-owned (BanditBudget discards its arm state and
  re-baselines). Primitives for SPO's per-round sampling; pinned in
  `tests/test_contract.py`. The loop calls `observe` **unconditionally** after
  every evaluate round, including a strategy-owned one, so a post-`takeover`
  `BanditBudget.observe` is a documented no-op (nothing to credit: the arms were
  discarded; the next `propose_load` re-baselines UCB).
- **`BudgetAllocator.rank()`**: beam ordering by pooled bandit Q̂ for ProTeGi
  (implemented by `BanditBudget`, `None` elsewhere → archive-score "Unif"
  fallback).
- **Sampling tiers** (`eyas/engine/sampling.py`, `DEFAULT_TIERS`):
  `generate/evolve/gradient/synonym/compare` (no `eval` tier: evaluation
  rendering rides the evaluator's deployment sampling); the deployment layer owns
  thinking/seed for generation/eval rendering (the judging channel is pinned
  greedy by design, docs/10 §6). No perturbation tier exists.
- **Resume fidelity protocol**: one mid-round resume-equivalence test per strategy
  (`tests/test_resume.py`), checkpoint→resume→byte-identical continuation.
  Checkpoints are written only at round boundaries, so a mid-phase
  resume **re-derives** the in-flight round's generations from the checkpointed
  state (round-atomic redo: the generation work is redone, never replayed from a
  store; "the generation stream replays byte-identically" is a test-side property
  of the deterministic responder, not a framework guarantee).

## 4. Per-method conformance

### 4.1 raw / cot (framework baselines, no paper)

| Contract item | Implementation | Test |
| --- | --- | --- |
| raw = task seed prompt, one train round + final_eval | `strategies/raw.py`, seed from `strategy.seeds` / `task.default_prompt` | `test_raw_degenerate_path` |
| cot = seed prompt + CoT guidance rendered into the task prompt (`task.params.assistant_prefix` on the mcq datasets, the `base_prompt` text on sroie/textvqa); shipped cot cells leave `cot_suffix` empty; reasoning merged back for judging on the judged (mcq) cells only | task-side `assistant_prefix` / `base_prompt` + `backends/normalize.py::split_think` | `test_cot_identity_in_rendering_variant`, `tests/test_normalize.py` |

No deviations: these make no paper claim.

### 4.2 ape — ICLR 2023 (Zhou et al.)

Contract verified against the author repo (`keirp/automatic_prompt_engineer`,
forward induction path):

| # | Contract item | Authority | Verified by |
| --- | --- | --- | --- |
| A1 | Candidates are few-shot **induction instructions** built from forward demos, then scored on the eval set | paper §2 (instruction-induction framing); `automatic_prompt_engineer/generate.py` (`get_query` / `generate_prompts`) | `test_ape_generates_clean_candidate`, `test_ape_induction_framing_is_the_paper_template` |
| A2 | Demo format `Input: …\nOutput: …` — demonstrations condition on **input→output pairs** (the forward-induction demo fix: completions restored to the demo groups) | `automatic_prompt_engineer/ape.py:20/62` — `demos_template='Input: [INPUT]\nOutput: [OUTPUT]'` | `test_ape_induction_conditions_on_input_output_pairs` |
| A3 | MCQ phrasing: `[Image {k}]` position markers on demo Input lines are plain-text markers (`user_text()` renders text parts only — ImageParts are dropped, no paths or image tokens leak) | framework adaptation (see A2) | `test_ape_demo_renders_list_reference_as_text`, `test_ape_induction_conditions_on_input_output_pairs` |
| A4 | One candidate survives per proposal; the completion is cleaned by marker stripping + first-segment selection (`clean_instruction`, 500-char cap) | verified against the pinned commit; `eyas/engine/genio.py` (`OUT_START`/`OUT_END`, `_MAX_LEN`) | `test_ape_generates_clean_candidate` |

Deviations (registered):

| # | Deviation | Why |
| --- | --- | --- |
| D1 | Pool size 4 per call (paper/official default **50 candidates** via multiple rounds) | budget scaling; registered as a deviation in this section |
| D2 | Initial task instruction stays **in** the candidate pool as a monotonicity floor; the official pipeline scores only generated candidates | deliberate guard against crowning a degenerate best (the paper text does not claim the initial prompt is a candidate); design decision recorded in `ape.py` |
| D3 | Demos are drawn from the selection split (the loop's eval split) | the loop split is the optimization data per the two-stage protocol; paper uses the training set |
| D4 | No multi-stage top-k% candidate filtering (paper Alg. 1 lines 2–9) — flat full-split evaluation of every candidate | ranking-equivalent endpoint, cost-saving stages dropped; budget simplification |
| D5 | Demo draws are **stratified by label** (`stratified_demo_groups`): when the group size is at least the class count, every group covers every class (backfilled to the group size, class-coverage fail-fast); when `demos_per_prompt` is smaller than the class count (the shipped 2-demo budget), each group covers a per-group-random subset of classes, rotating across groups — where the official sampler draws uniformly at random; recorded as a strengthening, not a deviation | class coverage can only improve; pinned by `tests/test_contract.py::test_ape_stratified_demos_labelless_single_bucket` |
| D6 | List-valued references are rendered through a top-level `", ".join` (`genio.render_reference`: scalars pass through `str`, tuple behaves as list), used by ape demos, opro exemplars, the spo optimize `Answer:` line and the protegi critique `Label:` slot | the official templates never pin the text form of a multi-value answer | `tests/test_genio.py::test_render_reference_joins_top_level_list` + `test_ape_demo_renders_list_reference_as_text` / `test_opro_exemplar_renders_list_reference_as_text` / `test_spo_optimize_prompt_renders_list_reference_as_text` / `test_protegi_critique_renders_list_reference_as_text` |
| D7 | **Two demos per induction prompt** (`demos_per_prompt=2`) where the paper samples five input-output pairs | budget scaling; the demo count is also the per-request image bound (`ResourceProfile(max_images=demos_per_prompt)`), pinned by `tests/test_planner_channel.py` |
| D8 | Empty references produce a blank `Output:` line | unreachable in practice with shipped tasks |

Scoring matches the official pipeline: candidates are ranked by their exact evaluation score (`rank_by_score`), no softening, no floor.

### 4.3 opro — ICLR 2024 (Yang et al.)

| # | Contract item | Authority | Verified by |
| --- | --- | --- | --- |
| O1 | Trajectory = (instruction, score) pairs; the **top-k scoring** entries are kept | `opro/optimization/opt_utils.py:64 gen_ins_and_score_pairs_substr` (ascending sort then `[-max_num_instructions:]` keeps the top) | `test_opro_trajectory_keeps_top_entries_ascending` |
| O2 | Displayed **ascending** (low score first) | official sorts ascending at `opt_utils.py:62-64` before formatting | same test |
| O3 | One candidate per completion via `single_bracket` (first `[...]` slice; a candidate containing `INS` — a meta-prompt-echo guard — is dropped). **Ours, not official**: the official extractor takes the first bracket slice verbatim (`opt_utils.py:33-38`, `extract_string_in_square_brackets`), applies `polish_sentence`, dedups by md5 against all previously evaluated instructions, and drops candidates over 500 chars (`opt_utils.py:785-806`) — no echo/marker filter | echo guard: our design (spec-inherited); official pipeline cites at the pinned commit | `test_opro_one_candidate_per_completion`, `test_single_bracket_drops_echoed_marker` (`tests/test_genio.py`) |
| O4 | Score displayed as integer percent (×100) | official meta-instruction format | `test_opro_score_prefix_is_integer_percent` |
| O5 | Meta-prompt carries the regularizer sentence ("concise, effective, generally applicable") | official meta-instruction text (`opt_utils.py:278, 287-288`); paper §2.2 meta-instruction figure | `test_opro_meta_request_states_regularizer` |
| O6 | 3 train exemplars, **text-only** (question + gold answer; images excluded from optimizer prompts) | design decision for cost | `test_opro_meta_request_carries_text_only_exemplars` |
| O7 | Selection signal = the candidate's **full-selection-split mean score**, re-measured over the whole split every round (`FlatBudget`: `budget: {name: flat}` pinned in every shipped opro config by the pin-scan test — there is no per-round subset window) | official evaluates every candidate over the **entire train index** (`eval_index_all=train_index`, `opt_utils.py:526` initial pass / `:909` per-round candidates); the subset in the official setup is the *fixed* `train_ratio` slice of the raw dataset (gsm8k 0.035 / bbh 0.2 / mmlu 0.8 — `optimize_instructions.py:640-649`, drawn once at `:660-667`), not a per-round sample window; the shipped code evaluates the full selection split every round | `test_opro_live_run_keeps_top_trajectory_entry`; allocator-pin scan `tests/test_configs.py::test_shipped_configs_pin_the_allocator_and_its_caps` |
| O8 | Termination = the round budget; the paper's prose "unable to propose new solutions" stop clause fires only **after one same-meta-prompt retry** of an empty generation round (`OproConfig.retry_on_empty`, default 1, 0 = fire immediately), and the finish logs a warning — never silent | official code has **no early stop** (`num_search_steps` fixed budget is the only termination, `optimize_instructions.py`); paper §2 prose carries the stop clause the code never implemented — we split the difference: keep the prose clause (a budget fully spent on parrot rounds is real waste) but require it to survive one retry (one lazy decode is not "unable") | `test_opro_empty_round_retries_and_recovers`, `test_opro_empty_retry_exhausted_finishes_with_seed`, `test_opro_retry_disabled_finishes_on_first_empty_round` |

Deviations (registered):

| # | Deviation | Why |
| --- | --- | --- |
| D1 | Cap 8 (paper 20), K 4 (paper 8), rounds/temp ladder at shipped values | budget scaling; the shipped values are the registered deviations |
| D2 | The official display filter `old_instruction_score_threshold` (default **0.1** — trajectory entries with a score below it never appear in the meta-prompt; `opt_utils.py:54, 68-69`) is **absent** here: all entries are displayed, `OproConfig` has no threshold knob | display-bandwidth knob unshipped; the asymmetry is registered here |
| D3 | Proposal temperature 0.9 vs official fixed 1.0 — and temperatures are owned by the **deployment layer** (tiers), never hard-coded per method | framework deployment contract |
| D4 | Over-cap overflow **truncates the tail** instead of dropping oldest entries | display-bandwidth equivalence; identity of kept entries matches official for ≤ cap |
| D5 | Exemplars are the first-N of the selection split, not random per step (the paper's own criterion options admit more than random) | resume determinism |
| D6 | Displayed percent is `round(score × 100)` — this **matches** the official display exactly (`_bucketize_float`, `opt_utils.py:47-49`, run with `num_score_buckets = 100`); recorded as an equivalence note, not a deviation | verified at the pinned commit |
| D7 | `single_bracket` drops any candidate whose text contains `INS` — an **ours-only** echo guard with no official counterpart (the official extractor is a verbatim first-bracket slice; `opt_utils.py:33-38`); known over-drop of legitimate uppercase-INS candidates | ours, spec-inherited; paired with the O3 marker-echo mechanism |

Paper-vs-code disagreement: the paper's prose describes score-bias orderings;
the official code simply keeps the top-`max_num_instructions` by score. We
follow the code (authority order §2 rationale). Official candidate filtering
beyond the trajectory display is md5 dedup + a 500-char length skip
(`opt_utils.py:785-806`); our over-cap behavior is a tail
truncation (D4) and no length skip is shipped.

### 4.4 evoprompt — ICLR 2024 (Guo et al.)

| # | Contract item | Authority | Verified by |
| --- | --- | --- | --- |
| E1 | GA: two-step **crossover → mutation** templates, one-shot example, ported semantics | GA prompt templates `BBH/data/template_ga.py` (`templates_2`, imported at `BBH/evoluter.py:15`); selection loop `BBH/evoluter.py`; paper Alg. 1 | `test_evoprompt_ga_request_carries_two_step_merge_template` |
| E2 | DE four-step: identify differences → mutate parts → combine with the historical champion (`cur_best` in the "Prompt 3" slot) → crossover with the basic prompt | paper Alg. 2; DE prompt templates (`BBH`) | `test_evoprompt_de_request_is_the_four_step_operator` |
| E3 | Donors r1 ≠ r2, both ≠ base index | `BBH/evoluter.py` GA/DE parent sampling (`random.sample`, tournament groups) | `test_evoprompt_de_donors_exclude_the_base` |
| E4 | Parent selection defaults to **roulette wheel**; tournament is a knob (`sel_mode`) | official `--sel_mode` default `wheel` (`BBH/args.py:34`, choices wheel/random/tour); paper Alg. 2 shows tournament — disagreement recorded, code wins | `test_evoprompt_ga_parents_are_fitness_proportional` |
| E5 | Population: **evaluate full seed pool, then truncate** to popsize by rank | documented semantics: the seed pool is evaluated in full before truncation | `test_evoprompt_init_evaluates_pool_then_truncates_top_popsize` |
| E6 | `trials` / `cur_best` serialized; resume at trial-eval boundary | checkpoint/resume contract (§3) | `test_evoprompt_mid_round_resume_survives_trial_state`, `test_evoprompt_state_serializes_trials_and_cur_best` |

Deviations (registered):

| # | Deviation | Why |
| --- | --- | --- |
| D1 | **DE donor slots** are **padded** when population \ {base} has fewer than two members, so the per-slot trial shape still runs (a single individual cannot supply distinct donors; the official code hard-requires popsize ≥ 4). GA has **no pad** — an initial pool under 2 distinct members raises at `init` (D7) | fixed-point degeneracy guard, DE-only by design |
| D2 | DE tie (`fitness equal`) **takes the trial** vector; official keeps base | tie rule stated explicitly here |
| D3 | Roulette second parent is drawn from the **renormalized remainder** (no self-pairing), vs official replace-true twins; equal-fitness case is exactly uniform, first-parent marginal exact | immaterial micro-deviation, design decision |
| D4 | Evolve temperature 0.7 vs official 0.5; the two operators share one tier (`evolve`) | deployment-layer temperature |
| D5 | Templates are text-only (no image token — the renderer adds the image) | framework rendering contract |
| D6 | **Initial-pool provenance**: the official pipeline **LLM-generates** the N_P=10 initial instructions per task at run time; ours ships **10 curated static English seed prompts per dataset** via `strategy.seeds` — config-visible, deterministic, and identical across seeds/runs (entry 1 is `task.default_prompt`, entries 2–10 vary phrasing/emphasis/format like the official generated pools). The **count N_P=10 matches the paper** (`--popsize 10`); only the candidate provenance deviates | reproducibility discipline (the official pool changes with the generator model/temperature); pools live in the 4 quickstart cells, the bench-scale cells inherit them verbatim through the dataset prompt asset in `configs/spec/datasets.yaml`, pinned by `test_evoprompt_ship_cells_carry_the_official_np10_pool` + `tests/test_matrix_configs.py` |
| D7 | **No silent single-prompt fallback for evoprompt GA**: `init` fail-fasts on an assembled pool of < 2 distinct prompts (message names the arithmetic + the `strategy.seeds` fix), `_ga_parents` keeps a mid-run tripwire, and the shipped evoprompt cells (the four quickstart bases, inherited by every bench cell) carry the D6 pools | fail-fast (no silent fallback): structural infeasibility must never reach the GPU; the configurator fixes configs (same doctrine: the configurator owns sample quantities) |
| E-note | Official defaults: `--popsize 10` (`BBH/args.py:30`), `--sel_mode wheel` (`BBH/args.py:34`). Our `popsize` config name follows the official argparse flag | naming alignment, recorded |

### 4.5 protegi — EMNLP 2023 (Pryzant et al.)

| # | Contract item | Authority | Verified by |
| --- | --- | --- | --- |
| P1 | Two-call **textual gradient → edit**: separate critique request ("write N reasons why the prompt got these examples wrong") and edit request ("revise the prompt using this feedback: {gradient}") | `prompt_optimization/optimizers.py`: critique prompt `_get_gradients` (`:58`, "give {num_feedbacks} reasons…" `:69`), edit prompt `apply_gradient` (`:80`, "Based on these examples the problem with this prompt is that {feedback_str}" `:91`); paper §3 | two-call sequence fingerprint `test_protegi_gradient_is_a_critique_then_edit_dialogue` |
| P2 | Error groups **sampled randomly** (size 4), never a head slice | paper §3 (random minibatches of errors) | `test_protegi_error_groups_are_seeded_samples` |
| P3 | Paraphrase expansion pass, separate sampling tier `synonym`, capped by `max_expansion_factor` | paper §3 paraphrase step; official `generate_synonyms` (`optimizers.py:105`, "Generate a variation … keeping the semantic meaning") | `test_protegi_expansion_pools_paraphrases_under_the_cap` |
| P4 | Beam truncation by pooled bandit **Q̂ ranking** (`BudgetAllocator.rank()`), carryover beam behind ranked arms, `arm_order` identity mapping (arm-identity fix) | paper §3.4 UCB formulation; official `UCBBandits` (`evaluators.py:140`) | `test_protegi_beam_ranks_arms_by_identity_not_position` + `test_protegi_beam_ranks_by_bandit_q_hat_not_archive_scores` |
| P5 | Flat-budget full-evaluation remains as the paper's **"Unif" comparator** (opt-in) | paper's uniform-sampling baseline variant | bandit config pins + Unif fallback tests |
| P6 | Beam order = `rank()` else archive-score Unif; final pick = truncation head (bandit) or Unif pick | paper Alg. 1 `SelectTop_b` | `test_bandit_rank_orders_by_q_hat_with_unvisited_first`, end-to-end bandit test |
| P7 | `offspring` / `seen` serialized; mid-phase resume replays | checkpoint/resume contract (§3) | `test_protegi_state_serializes_offspring_and_seen`, `test_mid_phase_resume_equivalence[protegi]` |
| P8 | One beam parameter b (expansion breadth + `SelectTop_b` truncation), carried by `beam_size`; a separate `keep_size` field stays unimplemented — none exists officially | recorded erratum; traceability comment in `ProtegiConfig` |

Deviations (registered):

| # | Deviation | Why |
| --- | --- | --- |
| D1 | Error pool is built from **durable `SampleOutcome` records** (evaluation record store), not re-run per step as the official pipeline re-scores per step | framework cache-first contract; semantically the same pool with replayed scores |
| D2 | UCB exploration constant `c = 2.0` (paper §3.4) applied as **standard UCB1**; the paper's Q̂ formulation has its own variance treatment | simplification recorded |
| D3 | Paper-derived config defaults (minibatch size, expansion factor, mc samples) **scaled down** for benchmark budgets | registered as a deviation in this section |
| D4 | `EDIT_TEMPLATE` uses "OPPOSITE to the flaw" wording (official phrasing differs slightly) | wording drift, semantics pinned by fingerprint |
| D5 | Numbered-without-tags gradient output merges into one gradient (parser strictness) | parser trade-off, registered |
| D6 | `clean_instruction` keeps its 500-char cap, so a protegi gradient line longer than that is truncated | registered as a deviation in this section |
| D7 | The bandit's **last pull may be a PARTIAL window**: the official algorithm draws a fixed-size batch of N per round (paper Ar. explicitly reuses samples across rounds), so our fixed **sequential** windows deviate in two registered ways: **no inter-round reuse**, and a **shorter final round when the split is not a multiple of `samples_per_eval`** (a data-end fact, not parameter rewriting). The partial final window is legal. The 14 bench cells on the 248-sample sroie/textvqa splits pin `samples_per_eval: 31` (`8×31 = 248` exact) in the benchmark spec's bandit combos (`configs/spec/benchmark.yaml`); the cub 261 / pets 296 cells keep the official `8×32` (256 ≤ n — full windows, the split tail beyond the plan is never consumed); the partial-tail shape remains legal for any split × plan that lands on it | the official caps are fixed while the selection split is whatever the benchmark provides — the **configurator**, not the runtime, reconciles the two; registered so a short final round is not mistaken for a bug |
| D8 | **Official 8×32 does not fit the quickstart toy splits** — the four quickstart protegi cells run on 74 (pets) / 100 (sroie, textvqa) / 110 (cub) sample selection splits, i.e. only `ceil(n/32) = 3` (pets) / `4` (the other three) distinct windows exist. The shipped contract is **config-authoritative fail-fast with remediation**: the plan executes exactly as configured (partial final window legal), a pull starting at or beyond the split end raises `ValueError` carrying the arithmetic (pull, start, n, planned) **and both computed fix values** — `samples_per_eval ≤ n // eval_rounds` and `eval_rounds ≤ n // samples_per_eval` (with an extra clause when both compute < 1: the split cannot host the plan at any cap) — the framework refuses to guess, the human fixes the config. **The configurator sets the caps (runtime never adapts): the four quickstart protegi cells pin `floor(n/8)` — pets `9` (74-sample split), cub `13` (110), sroie `12` (100), textvqa `12` (100) — with `eval_rounds=8` kept (paper structure), so every configured pull serves a full window and the split tail (2/6/4/4) is never consumed; the official `8×32` caps remain on bench scale where the splits fit. The bench-side table (`32`/`32`/`31`/`31`) is carried by the benchmark spec's bandit combos (`configs/spec/benchmark.yaml`, bound in `tests/test_matrix_configs.py`), so toy caps can never leak into bench cells, and the non-optimizing baseline cells carry a flat budget explicitly — regeneration is idempotent. Shipped bench truth: the 14 sroie/textvqa cells run `8×31` exact in-config; cub 261 / pets 296 run full windows | fail-fast contract ("no silent fallback"): the error must carry actionable remediation for the configurator; silent degradation and silent adaptation are both excluded |

### 4.6 spo — Findings of EMNLP 2025 (Xiang et al.)

| # | Contract item | Authority | Verified by |
| --- | --- | --- | --- |
| S1 | Loop = OvO **answer voting**: incumbent vs challenger on paired samples; accept iff `wins > losses` | paper §3 + abstract ("simple yet effective" pairwise preference); official paired-swap evaluation | `test_spo_accept_rule_challenger_wins_advance_incumbent`, `test_spo_accept_rule_challenger_loses_keep_incumbent` |
| S2 | ABBA position swap with **4 votes per pair**; unparsable/tie tally 0.5 per side | official `EVALUATION_REPETITION = 4` (`utils/evaluation_utils.py:9`) | `test_spo_vote_group_of_four_swaps_abba_under_single_pair` + `test_spo_tie_counts_half_needs_strict_win_sum` |
| S3 | Selection vote is **structurally gold-free**: `AnswerPairwiseJudge` has no reference field at all; gold may appear only on the OPTIMIZE side (paper allows execution feedback incl. reference answers there) | paper §3 framing; official prompt templates | `test_spo_ovo_votes_carry_no_gold_and_swap_ab_within_four`, `test_answer_judge_request_has_no_gold_field` (`tests/test_pairwise.py`) |
| S4 | Optimize prompt carries the incumbent + its wrong samples' `Q:` / `Answer (model):` / `Answer:` lines and demands the `<analyse>/<modification>/<prompt>` contract; stamped `[Round n]` from the persisted counter | official optimize-prompt templates (`prompts/optimize_prompt.py`; the voting side is `prompts/evaluate_prompt.py`) | `test_spo_optimize_request_carries_incumbent_outputs_and_contract` |
| S5 | Pair schedule: split ids shuffled **once**, consumed **without replacement** across rounds; exhausted schedule appends a reshuffled cycle | deterministic-resume contract (§3) | `test_spo_pair_schedule_shuffled_once_without_replacement` |
| S6 | Seed pass + one **closing full-split** evaluate when the incumbent lacks a full record; `Flat + takeover` budget | framework contract (§3 `Decision.items`) | `test_spo_items_decision_evaluates_exactly_the_subsets`, `test_takeover_protocol_all_allocators_accept_and_bandit_invalidates` |
| S7 | Phases/incumbent/challenger/cursor/schedule/gens_served persisted; resumed runs never resample voted pairs; `items` rounds billed like any round | state-persistence/resume contract (§3) | `test_spo_state_serializes_phase_incumbent_cursor_and_schedule`, `test_spo_midphase_resume_from_voting_checkpoint_continues_exact_trajectory` |
| S8 | Generations reuse the record store; judge calls billed to the `compare` tier | framework cost convention | `test_spo_votes_bill_the_compare_channel_only` |

Deviations (registered):

| # | Deviation | Why |
| --- | --- | --- |
| D1 | Shipped matrix uses `rounds=20`; paper A.4.3 scans up to 10 for the reported tables | 20 sits in the appendix's scan region but is above the published rows — knob values are ours, registered |
| D2 | Zero offspring dedup (official keeps a deduplicated pending set) | accepted drift; identity via prompt_hash only |
| D3 | Neutral tie smoothing: unparsable/tie → **0.5/0.5 split** per side (official: missing verdict loses the ballot) | deliberate smoothing, stated in docs/07 §6 |
| D4 | `{choices}` guard: mcq choice-list rendering kept inside the task protocol so optimized prompts never inline a frozen class list | task-protocol adaptation (framework contract) |
| D5 | Seed derivation is deterministic per run (`derived_rng()`); the official pipeline offers a `general_seed` argument and seeds ad-hoc | reproducibility discipline |
| D6 | The shipped spo cells (the four quickstart bases, inherited by every bench cell) all carry **two seeds** while the loop carries one: `init` keeps `seeds[0]` (the paper's loop is single-incumbent — one challenger against one incumbent per round, S1 — so the extra seed is dropped) and **names the drop on a WARNING** — a supplied input is never discarded without a trace | fail-fast discipline; `test_spo_init_warns_when_extra_seeds_are_dropped`, `test_spo_init_stays_silent_on_single_seed` |
| D7 | `num_pairs = 16` samples per voting round where the paper's sample-size curve settles on **three**; the per-round window is ours | budget scaling — the larger window buys a stabler ABBA vote at the shipped round count; stated in the series bridge (article 06) |

## 5. Benchmark generation status (read before quoting any number)

All **published** numbers (docs/07 §3/§4/§5, docs/08 and the README
scoreboard) are emitted mechanically by
`scripts/aggregate_bench_results.py` from the runs' `result.json`, so every
published table row can be regenerated by re-running that script over the run
directories. Per-sample prose observations come from the same runs' durable
per-sample records. docs/08 and the README scoreboard come from the bench run set
(7 tags × 6 datasets × 8 methods × 3 seeds = 126 sweeps / 1008 runs, the
Qwen3.8-27B extension column included): every run completed, every bill cold
except the six nextqa 2B cells whose loop replayed from the record store (the
spend was billed on an earlier attempt of the same wave, and their cost means
say so); docs/07
comes from the quickstart run set (15 sweep invocations covering 90 E2 cells and
6 E1 arm cells) and is unchanged by the bench regeneration. The seed-42 directional readings all reproduced in the full-method
q9b sweep, which supplies the published 9B columns; the whole chain
(configs → runs → tables) is one generation.

## 6. Reproduction pointers

- Re-clone each authority repo at the pinned commit of §2 and read the cited
  files; a clone at any other hash is not authority.
- Contract evidence lives in `tests/` (fast tier, zero GPU): per-method
  fingerprints in `test_contract.py`, extraction in `test_genio.py`, resume
  fidelity in `test_resume.py`, ranking pins in `test_ranking.py`, store/schema
  in `test_record_store.py`, pairwise judging in `test_pairwise.py`, config
  allocator pinning in `test_configs.py`.
- Method end-to-end matrices run against `ScriptedMockBackend`; every claim in
  §4 that cites a test is reproducible with `uv run pytest`.
