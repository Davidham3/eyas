# Eyas contract (frozen — the shipped architecture)

> Status: **frozen contract** (project convention: any signature change must be reviewed
> here in sync). The core delivers "contract-driven, capability-validatable,
> resource-single-chokepoint, typed-state". §12 records the shipped status and the
> known deviations. Interface signatures are the ones under `src/eyas/engine/`.

## 1. Design axioms

- **A1 Contracts over shapes**: components declare pre/post invariants, capabilities,
  and resource requirements in addition to signatures; these are validated statically
  at assembly time, so illegal assemblies `fail fast` at startup.
- **A2 Algorithm identity can land in any of five places** (rendering variant /
  propose / budget policy / evaluation semantics / selection). The abstraction unit
  is the "capability contributor": an optimizer is a composition of capabilities.
- **A3 Budgets are first-class and sovereign**: consumption explicit, auditable,
  adaptive. `BudgetAllocator` is stateful (choose/update), orthogonal to the pure
  ranking step `BudgetAllocator.rank()`.
- **A4 Evaluation semantics are pluggable**: absolute scoring
  (per-sample→aggregate→mean) and relative comparison (pairwise/listwise) are peer
  `EvalModel`s; the former is not hardcoded.
- **A5 Resources and context have a single chokepoint**: every engine-bound request
  passes through one `RequestPlanner`; token / multi-image / context-window limits
  and budget accounting are enforced in exactly one place.
- **A6 Cross-round state is typed**: optimizer state is a schema'd `OptimizerState`
  (serializable = natural checkpoint/resume), not an `extra` dict.
- **A7 per-sample observation is first-class**: every (candidate, sample)'s
  prediction/error is durable, never lost to cache hits.

## 2. Core domain model

```python
@dataclass(frozen=True)
class Prompt:
    text: str                           # compare=False: display-only, not identity
    metadata: Mapping[str, str] = ...   # label etc. display-only, not identity
    id: str                             # init=False, set in __post_init__ to prompt_hash(text):
                                        # sha256[:12] after normalize — the sole identity

@dataclass(frozen=True)
class Cost:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    judge_requests: int = 0        # judge channel, already included in the totals above
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

@dataclass(frozen=True)
class SampleOutcome:          # A7: durable, persisted to cache with the Evaluation
    sample_id: str
    prediction: Any           # the task's parse output as-is (str, list[dict[str, str]], ...)
    scores: Mapping[str, float]
    correct: bool | None = None  # None = judge undecided
    error: str | None = None  # directly usable by textual gradients, no need to infer from metrics

@dataclass(frozen=True)
class Evaluation:
    prompt_id: str
    split: str
    eval_model: str           # "absolute:mcq" / "compare:spo"
    scores: Mapping[str, float]           # after aggregation
    per_sample: tuple[SampleOutcome, ...] # durable across cache hits
    cost: Cost
    round_idx: int = 0

class Archive(Protocol):      # append-only: known prompt -> best Evaluation (opro unbounded pool / reuse)
    def add(self, ev: Evaluation) -> None: ...
    def best(self, prompt_id: str) -> Evaluation | None: ...
    def history(self) -> Iterator[Evaluation]: ...

class OptimizerState(ABC):    # A6: typed, serializable = checkpoint. One subclass per method
    def to_json(self) -> dict[str, Any]: ...
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "OptimizerState": ...
```

## 3. Capability declaration and assembly-time validation (capability negotiation)

Every strategy carries a declarative `CapSpec`; `capability.validate_assembly`
runs a compatibility check, and illegal assemblies **fail at startup**.

> The sketch below is the pre-landing spelling of this contract. The shipped form is
> `@register_strategy("name")` with the declaration attached as a class attribute:
> `capabilities: ClassVar[CapSpec] = CapSpec(name=…, requires_eval=…, optimize_keys=…,
> resource_profile=…, produces_artifacts=…, consumes_artifacts=…)`; tasks, backends and
> judges register by name only (`@register_task("mcq")`) and expose their declarations
> through their own classes (`TaskCap`, `EvalModelCap`, `BackendLimits`). There is no
> `register_optimizer`. See `src/eyas/engine/registry.py`, `src/eyas/engine/capability.py` and
> `src/eyas/engine/strategies/raw.py`.

```python
@register_task("mcq", metrics=["accuracy", "macro_f1"], supports={"absolute_eval"},
               splits={"train", "val", "test"})
@register_eval_model("absolute_mcq", needs={"absolute_eval"}, produces_metrics={"accuracy"})
@register_eval_model("compare_spo",  needs={"compare_eval"},  produces_metrics={"compare_score"})

@register_optimizer(
    name="compare_only",
    requires_eval={"compare_eval"},
    produces_task_metrics=set(),          # <- empty: cannot produce accuracy
    needs_backend=True, needs_judge=False,
    sampling_profiles={"rewrite", "compare"},
    resource_profile=ResourceProfile(max_images=8, ctx_reserve=4096),
    consumes_artifacts=set(),             # no declared dependency: ape.candidate_prompts is a soft read (§9)
)
```

`build_run_context(config)` validations (all raise `ValueError`):

1. `opt.requires_eval ⊆ union of the assembled eval_model semantics`.
2. Every optimization objective key of the run (`opt.optimize_keys`, falling back to
   the task's primary objective) is produced by some installed eval model;
   unsatisfied → reject, with the message "this method must not enter the
   leaderboard for these objectives".
3. `opt.resource_profile ⊆ backend.resource_limits` (multi-image count, context
   window).
4. Pipeline dependencies (`consumes_artifacts`) satisfiable (sweep DAG / artifacts
   present).

The backend side of rule 3 is declared on `InferenceBackend` itself:
`context_window` and `max_images` (both `int | None`; `None` = unknown/server-side,
that bound is skipped). vllm reports its effective engine window and its
`limit_mm_per_prompt` image cap (default 1, vllm's own default); a remote
OpenAI-compatible endpoint declares `None` unless told. `build_run_context`
intersects the strategy's `ResourceProfile.max_images` with the declared cap, so
a multi-image strategy over the backend limit is rejected at startup.

> **Effect**: a method that would have run to completion and reported `—` is a red light at
> second 0 of `eyas run`, and a method that declares objectives no installed eval model
> can produce never reaches the leaderboard.

## 4. Evaluation layer: two peer semantics

```python
class EvalModel(ABC):                       # extension point: semantics + produced metrics
    def produces(self) -> frozenset[str]: ...
    def semantics(self) -> frozenset[str]: ...   # {"absolute_eval"} / {"compare_eval"}
    async def compare(self, prompts, samples, evaluator, split, round_idx) -> list[Evaluation]: ...

class AbsoluteEvalModel(EvalModel):         # absolute semantics:
    ...                                     # per-sample -> protocol aggregate -> cross-sample mean

class CompareEvalModel(EvalModel):          # compare semantics:
    ...                                     # round-robin pairwise -> relative win-rate

@dataclass
class Evaluator:                            # orchestration: cache-first + planner + cost accounting + watchdog bump
    model: EvalModel
    async def aevaluate(self, prompts, load: LoadPlan, split="train", round_idx=0) -> list[Evaluation]: ...
```

Absolute / relative **share the orchestration** (same planner, same cost accounting,
same trace/bump); only the scoring kernel differs. The absolute path is cache-first;
compare-semantics pairwise goes through the planner, so it gains cost accounting and
liveness detection (its games are always freshly sampled).

Compare cost and per-sample semantics: the whole-batch game bill is carried **once**.
Only the first `Evaluation` of a compare run carries the batch `Cost` (requests =
`n*(n-1)*len(samples)`, tokens summed over the games, wall span of the batch); every
other `Evaluation` carries `Cost()`. Round-level cost summing therefore counts each
game exactly once (no n-fold inflation). Each prompt's `per_sample` view is its **real
mean game outcome on that sample** (win 1.0 / tie 0.5 / loss 0.0 across its games,
always in [0,1]); the aggregate `compare_score` is the win rate over all its games.
Fewer than two candidates stays uncontested: neutral 0.5, no games, zero cost.

## 5. Budget allocator (first-class, stateful, composable)

```python
class BudgetAllocator(Protocol):
    def propose_load(self, round_idx, state, prompts, all_sample_ids) -> LoadPlan: ...  # what to evaluate this round, how many samples each
    def observe(self, evals: list[Evaluation], rewards: list[float]) -> list[str]: ...  # early-stop/eviction -> surviving prompt_id
    def rank(self) -> list[int] | None: ...   # final selection order over the arm list (SelectTop_b(Q̂)); None = no bandit ranking
    def takeover(self, round_idx: int) -> None: ...  # a `Decision(items=...)` round owns its load out of band
    def get_state(self) -> dict[str, Any] | None: ...  # serializable allocator state for the checkpoint; stateless allocators return None
    def set_state(self, data: Any) -> None: ...        # restore allocator state from a checkpoint (--resume)
```

`takeover` is called by the loop for **every** round that carries explicit work
items (`Decision(items=...)`): the round bypasses `propose_load`, so planners
must not assume one fixed allocation shape. `FlatBudget`/`EvenSplitBudget`
(no-ops) and `BanditBudget` (discards its **entire** arm state: arms unbound,
counts/values/samples/round zeroed; the next `propose_load` re-baselines on the
freshly proposed arm set; there is no keep-arms semantics across a takeover
round) implement it. Because the loop calls `observe` **unconditionally** after
every evaluate round, a strategy-owned one included, `BanditBudget.observe` is
a documented **no-op** while the arms are unbound (post-takeover): the round's
rewards are dropped rather than credited to discarded arms, and no state is
(re)created; asserting on the unbound state instead would make the first
strategy-owned round of any future bandit user crash.

Built-ins: `FlatBudget` (hill_climb/evoprompt/ape full flat, `rank()→None`),
`EvenSplitBudget` (protegi bf even split, `rank()→None`), `BanditBudget` (protegi
UCB/UCB-E, choose/update; `rank()` = pooled-Q̂-descending positions over the
arm list of the most recent `propose_load`, zero-pull arms first, and the caller
maps positions through the arm identities it committed (protegi:
`ProtegiState.arm_order`, carryover beam members rank behind every ranked arm)
instead of ranking the beam by archive best-subset scores; None while
unrouted).
`BanditBudget`'s window contract (**config is law**): the configured window plan
(`eval_rounds × samples_per_eval`) executes exactly as stated. The runtime
never rewrites a configured value (sample quantities belong to the configurator;
auto-shrinking IS a silent downgrade). Consecutive windows may legitimately end
on a **partial final window** when the split is not a multiple of
`samples_per_eval` (a data-end fact: the official algorithm draws fixed-size N
batches with explicit inter-round reuse; our registered deviations are no
inter-round reuse and a shorter final round,
`docs/baselines-fidelity.md` §4.5 D7/D8). A pull starting **at or beyond** the
split end raises `ValueError` carrying the arithmetic (pull, start, n, planned)
AND both computed remediation values for the configurator
(`samples_per_eval ≤ n // eval_rounds` and `eval_rounds ≤ n // samples_per_eval`):
the framework refuses to guess; the human fixes the config.
Budgets are explicit objects, so cost becomes predictable and
profilable (the core selling point).

## 6. Sampling tiers (SamplingProfiles)

Methods declare the tier names they use
(`generate/evolve/gradient/synonym/compare`); at assembly time these
merge with the deployment baseline `ctx.sampling`: for candidate generation
and evaluation rendering, thinking/seed come from the deployment layer. Optimizers
carry no `SamplingConfig(temperature=0.7)` constants of their own; cost is accounted
per tier. The **judging channel is the deliberate exception**: `JudgeContext`
sampling is pinned greedy (temperature 0, no seed/thinking). A verdict that
rode the run's seed or thinking setting would make relative scoring itself
irreproducible, so pinning is a methodological requirement, not a waiver
(pinned by tests/test_planner_channel.py: two runs with different deployment
seed/thinking produce byte-identical judging requests).

## 7. Single chokepoint for resources and context (RequestPlanner)

```python
@dataclass(frozen=True)
class ResourceProfile:
    max_images: int | None
    ctx_reserve: int

@dataclass(frozen=True)
class Rejected:                            # plan-time typed result, NOT an exception;
    reason: str                            # the rest of the batch is unaffected

class RequestPlanner:                        # every build_request product must pass through here
    def plan(self, request: ChatRequest) -> PlannedRequest | Rejected: ...
```

Over-window / over-limit escalates from a runtime string sentinel to a
**first-class planning-time result**.
`ResourceProfile` is validated at assembly time by intersecting with the backend's
declared limits. APE multi-image single requests, protegi's erroneous multi-image
requests and SROIE large images blowing the window all hit this one guard rail.

**Shipped as structure, not prose.** Three closings:

- `PlannedBackend` (`planner.py`): the paths that legitimately hold a *backend
  slot* rather than the evaluator (a judge via `JudgeContext.backend`, pairwise
  games via `CompareEvalModel`) ride this adapter, so bypassing the planner is
  physically impossible. Accepted requests forward unchanged (wrapping never
  alters a request byte; verified by a byte-identical request-stream fingerprint
  across all 8 strategies); a rejected request gets an
  empty response with `finish_reason=OVER_WINDOW_FINISH` and the rest of the
  batch is unaffected.
- The window guard is ARMED from the window the backend declares
  (`InferenceBackend.context_window`; the vllm engine reports its engine value, a
  remote OpenAI-compatible endpoint that was not told declares `None`, and there
  the guard stays disarmed: its over-window signal rides the server
  error). `build_run_context` plumbs it into both `validate_assembly` and the
  `RequestPlanner`, so the guard is armed before the first request is planned.
  The same declaration path carries `InferenceBackend.max_images` (vllm: its
  `limit_mm_per_prompt` image entry, default 1; openai: `None` = server-side):
  the assembly-time intersection with `ResourceProfile.max_images` is live, so
  an over-limit multi-image strategy fails at startup, not request by request
  at the planner.
- A planner-rejected JUDGE verdict is unbilled: `McqLLMJudge` returns
  `Verdict(decided=False, usage=None)` on the rejection response: a request
  that never left the process enters no cost profile (tests/test_planner_channel.py).

## 8. New optimizer contract: managed loop + typed step

A free-form `run()` guarantees no invariant.
Change it to **the skeleton owns the invariants; methods write only one-step
decisions**:

```python
class Strategy(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[CapSpec]
    config_cls: ClassVar[type]
    state_cls: ClassVar[type[OptimizerState]]
    def __init__(self, config, ctx: RunContext) -> None: ...  # ctx constructor-carried
    def init(self, seeds: Sequence[Prompt]) -> OptimizerState: ...
    def step(self, state: OptimizerState, view: EvalView) -> Decision: ...
    def expected_rounds(self) -> int | None: ...  # eta horizon only, never a bound
    def restore(self, data: Mapping[str, Any]) -> OptimizerState: ...  # --resume

Action = Literal["evaluate", "propose", "finish"]

@dataclass(frozen=True)
class WorkItem:                               # an explicit (prompt, sample-subset) work unit
    prompt: Prompt
    sample_ids: tuple[str, ...]               # () = the full evaluation split

@dataclass(frozen=True)
class Decision:                               # one step does one thing, composable
    action: Action
    prompts: tuple[Prompt, ...] = ()
    best_id: str | None = None                # finish: the method's declared winner (observability;
                                              # the loop's ledger best ranks archive scores —
                                              # a divergence logs a WARNING, the rule stands)
    items: tuple[WorkItem, ...] | None = None  # evaluate: explicit (prompt, subset) work items
```

When `items` is set, the loop builds the round's plan from the items directly
(the round is a *subset round*, its archive score averages the subset only),
calls `budget.takeover(round_idx)` (the round's work is decided by the strategy,
not by the allocator) and bills the round against the model's declared
capabilities as usual. This is the loop primitive that lets a method run rounds
whose purpose is not "score a prompt on the split": spo's per-round one-vs-one (OvO) vote
batches ride the `compare` tier through the same accounting/trace/cache rails.

`OptimizerLoop` (the framework) handles: calling the planner, debiting the
`BudgetAllocator`, running the `Evaluator`, recording cost, writing history,
bumping the watchdog, serializing `OptimizerState`, draining logs. The round-end
checkpoint also freezes the per-tier request ledger (`cost_by_tier`'s source),
the artifact manifest, and the run-config identity (`config_fingerprint`, a
sha256 of the canonical config JSON with the path-only keys `run.output_dir` /
`run.output_root` / `run.cache_dir` excluded, plus `config_sections`,
per-top-level-section hashes
for the mismatch diagnostic); `--resume` restores the ledger, reloads the
persisted artifacts (§9), and rejects a changed config (naming the changed
sections), a dir without `checkpoint.json`, and a pre-fingerprint checkpoint
(with re-run guidance); it never starts fresh silently. `RunContext` carries the
identity (`config_fingerprint` / `config_sections`, set by the runtime; `None`
for library use, frozen as null). **Methods never
get the bare backend**: only the orchestrated `ctx.evaluator`, structurally
forbidding "bypassing the evaluator to run inference yourself".

Degenerate methods carry over the same way: raw/cot = `init` yields the seed
pool (raw: the first seed; cot: one rendered prompt per seed, the CoT suffix
appended), their first `step` returns `Decision("evaluate", …)` for that seed (so the
leaderboard entry exists), and the next `step` returns
`Decision("finish", best_id=…)`; `final_eval` supplies the reporting split. There is no separate `is_terminal` hook and no free-run escape
hatch: termination is expressed **only** through `finish`, so every method passes
through step and the invariants hold in exactly one place.

## 9. Cross-method artifacts = a first-class Artifact pipeline

`Artifact` (typed, hash-addressed). APE hard-declares
`produces_artifacts={"ape.candidate_prompts"}`; evoprompt soft-consumes that
address: when a store provides the artifact it merges the upstream candidates
into its initial pool, and it runs standalone otherwise. Assembly-time validation
therefore enforces declared dependencies only (rule 4: a declared consumer with no
producer is rejected), which is why an evoprompt-only sweep needs no APE step. A
consumer always reads its producer's output through the store, never by parsing a
shared file. A path-bound store persists every emit to `run_dir/artifacts/`;
`--resume` reloads those files before assembly validation and cross-checks them
against the checkpoint's frozen manifest (a missing or altered file fails fast),
so a restarted run's consumers, declared or soft, still find their upstream
outputs.

## 10. Observability by contract

cost/trace/history/round-echo are produced uniformly by `OptimizerLoop`; methods
cannot omit them. Logs split into **two channels**: records emitted via
`eyas.logging.key_event` (`extra={"eyas_priority": True}`) ride an **unbounded**
priority queue the writer thread drains first; ordinary INFO rides the droppable
bounded queue, so a full or stalled write path can never bury the run's evidence
trail (the per-round echo is the resident key event; per-request traffic such as
planner rejects and engine progress is NOT key by construction and must never be
marked priority). Watchdog kill-diagnostics deliberately bypass logging entirely
(non-blocking stderr prints): they must emit before `os._exit` without depending
on the queue or any handler.

`EYAS_LOG_LEVEL=DEBUG` opens the phase-breadcrumb minimum inventory: `[loop]
round=N start / step done / end` (elapsed sec, bracketing `strategy.step`),
`[planner] rejected` plus the aggregate `RequestPlanner.reject_counts` (by
reason, read-only), `[eval] start / cache hit/miss / generated`, `[compare]
start` (pair count before the fan-out batch), `[store] flush / close` plus
`SqliteRecordStore.hits` / `.misses`, `[prefetch] batch start / done` with
counts, and the engine backend's `[media] decode submitted / batch done`
(request, image and clip counts, decode-ready and wall seconds). Messages carry
counts/elapsed only, never media payloads, with bounded per-request
exceptions: `[media] generate failed` (ERROR) logs the failed request id, its
media counts and 120 characters of the prompt head and tail, and `[media] load
failed` (ERROR) logs the request id plus the load error, because a failed
request must be identifiable from the log alone. Per-file media load failures
(a missing/undecodable image or clip) degrade to the same per-request
`OVER_WINDOW_FINISH` sentinel in both inference backends: the batch survives,
one `[media]` WARNING per batch summarizes the failed count, and evaluation
lands the durable error row (null prediction, `parse_fail=1.0`); engine
failures and pool/config errors still raise. The breadcrumbs also carry the
watchdog bump tag, so trace silence alone cannot mask a hang once the level is
open.
`reject_counts` / `hits` / `misses` are the in-process counter surface the
status/report layer consumes.

The run dir also carries `status.json`, the machine-readable observation every
external reader (`eyas status`, a dashboard, a cluster wrapper) uses. Schema v1:
`{schema_version, run, phase, round, budget, cost, cache, rejections, history,
updated_at}`, where `phase` walks `starting → optimizing → done` (with
`failed` as the terminal phase when the run dies before `result.json`),
`round` is the loop's round cursor, `cost` mirrors `result.json`'s
`total_cost` keys, and `cache` / `rejections` come from the counter surface above. `eyas.engine.status` owns
it as pure functions: `write_status` (temp file **in the same dir** +
`os.replace`, so a reader never sees a torn document; stamps `updated_at`;
`history` bounded to the last 50 entries) and `read_status` (absent → `None`,
corrupt → the `JSONDecodeError` propagates, so a broken observation is loud).
`run_experiment` is the **single writer** at four points: run start
(`starting`, round 0, zeroed counters, before assembly), the `on_round` checkpoint
echo, finalization (`done`), and the error path (`failed`). A refresh never
moves `round` below what the lineage already reported, so a resumed run does not
restart its observation at zero. The error write closes the **started-run
invariant**: every run that wrote its run-start status ends with either
`result.json` or phase `failed` (assembly failures and mid-loop failures alike);
the last phase alone cannot tell "still running" from "dead". The original
exception always propagates unchanged, and a failed status write is a lost
observation, never a second error. Unknown keys ride along: the contract is
write-only forward compatible.

`eyas status <run dir>` is that contract's reference consumer: read-only, no
color, exit 0 for every run-dir shape. It prints the fixed sections (run name /
phase / round / budget / cost with its judge channel / cache hit-rate /
rejections / `updated_at`, then the history tail) and appends result.json's
final best + metrics + report-split numbers when the run got that far. Every
printed field comes from the files: the renderer invents nothing (no ETA, no
extrapolation), which is what lets a later dashboard render the same sections.
A dir without `status.json` degrades to checkpoint + result behind the line
`no status.json (pre-status run or older version)`; a corrupt `status.json`
stays loud, since the reader propagates the `JSONDecodeError` by contract.

The **terminal** carries progress, the files carry detail: the loop echoes one line
per completed round at the default INFO level: `round <i>/<max>: <n_requests> reqs
in <x>s eta≈<y>s`. `i` counts the rounds whose checkpoint persisted (the number
`status.json`'s `round` cursor reports) against the loop's enforced step ceiling
`run.max_steps`: it counts **strategy steps** (every `evaluate` or `propose`
decision consumes one; only evaluate rounds advance the round cursor), the only
bound the loop owns, so a method with a smaller
self-declared target just stops printing early; `n_requests` is the round record's
own count and the elapsed seconds the loop's clock for the round that just ended.
The ack fires **after** the checkpoint write: the console never announces a round
the run dir has not frozen. `eta≈y` is
`mean completed-round duration × rounds left under min(run.max_steps, strategy.expected_rounds())`,
the method's own declared round horizon (seed round included; `None` = no
declaration) projected when it is smaller than the ceiling, because a method that
stops after 4 rounds must not be shown as if 396 were left. The
declaration only ever lowers the horizon: never a floor, never a bound; above
the ceiling it is ignored and the ceiling projection stands. Averaged over
`RunContext.round_seconds`, the loop's own
per-round clock, because the round record's `wall_seconds` is the per-evaluation
bill, not the round's elapsed time. `--resume` refills that clock from
`<run_dir>/status.json` (the runtime passes `RunContext.run_dir`; a loop driven
directly as a library has no run dir, so its average starts from this attempt's
rounds). With an empty clock (no measured round, e.g. a resumed run whose
status carries no round timing), the field reads `eta≈n/a` verbatim: a
projection over zero measurements is a fabricated number.

`result.json`'s `history` entries carry the **per-round invoices and the learning
curve**, not just the echo-visible counts: `{round, n_candidates, n_requests,
cost, elapsed_sec, best_so_far}` where `cost` is the round's full `Cost` (all
seven fields, judge split included), `elapsed_sec` is the loop's own clock for
the round (the same quantity `round_seconds` carries, distinct from
`cost.wall_seconds`, the per-evaluation bill), and `best_so_far` is the archive
leader AFTER this round folded in (`{prompt_hash, metrics}` under the ledger's
ranking rule): consecutive points are the best-so-far curve, their deltas the
per-round marginal gains. The per-round spend curve and the improvement curve
must be readable from the final artifact alone: checkpoint.json can be truncated
by a later round and status.json's history is bounded, so neither is a durable
substitute, and the final `ranked` list carries no round attribution; without
the curve point it is unrecoverable post hoc. Measured quantities are
per-attempt: a resumed run replays the pre-crash rounds VERBATIM from the
checkpoint payload (never re-measures them), which is what the resume-fidelity
test pins; the curve test pins monotonicity and the tail == reported best.

Backend lifecycles gain the `loop_affinity: process_singleton | reentrant | thread`
capability declaration; `RunContext` holds the backend-owned loop. `run_sync`'s
process-level singleton loop is a convenience API
provided only when all backends are reentrant, which unlocks async/notebook/thread
concurrency.

## 11. Carrying matrix (the new abstraction fits the 8 methods)

| Method | Strategy form | BudgetAllocator | EvalModel (semantics) | SamplingProfiles | Special capabilities (new abstraction slots) | Assembly-time |
| --- | --- | --- | --- | --- | --- | --- |
| raw | degenerate | Flat | absolute | — | metrics come from final_eval | pass |
| cot | degenerate | Flat | absolute | — | identity = rendering variant (carried by task) | pass |
| hill_climb | standard | Flat | absolute | — | standard | pass |
| evoprompt | standard(pos-aligned) | Flat | absolute | evolve | Soft artifact consume `ape.candidate_prompts` (no hard declared deps); typed State cur_best; position alignment explicit index | pass |
| opro | archive-only(unbounded) | Flat | absolute | generate | typed OproState.history; removes accuracy hardcoding | pass |
| ape | degenerate-once | Flat | absolute | generate | Artifact produce; resource profile bounds demo images (`max_images` = demos per prompt) — over-limit requests rejected at the planner; inline completion sanitizer (extract, drop unparsable, dedupe) | pass |
| protegi | beam | Bandit (shipped) / EvenSplitBudget | absolute | gradient,synonym | gradients = critique→edit two-call dialogue over durable SampleOutcome errors (randomly-sampled groups); paraphrase expansion cap `max_expansion_factor`; beam order = `BudgetAllocator.rank()` (SelectTop_b(Q̂)) else archive-score Unif; Lineage explicit parents | pass |
| spo | items-round | Flat + takeover | absolute (+ direct OvO ballots) | gradient,compare | OvO answer voting: `Decision(items)` pair eval + `AnswerPairwiseJudge` ballots, no gold, ABBA; two generations only — the optimize call on `gradient`, the vote ballots on `compare` (evaluate-round traffic is the loop's, not a generation tier) | pass |

Fixing a new bug goes from "cramming a special case into a call site" to
"implementing a new `BudgetAllocator` / `EvalModel` / `Strategy` / task
rendering variant": every patch gets a clear slot and no longer erodes the
core.
## 12. Shipped status and known deviations

Everything in §1-§11 is implemented as written: the eight strategies
(`raw`, `cot`, `hill_climb`, `evoprompt`, `opro`, `ape`, `protegi`, `spo`), the two
registries, the managed loop, the typed checkpoint/resume path, and the config spec
`configs/spec/` (the assembled `<ds>/<method>` cells plus the benchmark spec
`configs/spec/benchmark.yaml`: 7 tags × 6 datasets × 8 methods; published results in
`docs/08-bench-benchmark.md`). The contract tests under `tests/` are the executable
form of this document.

The deviations below are the only places where the shipped code is narrower than the
contract:

- **No sweep-aware artifact DAG (§9)**: artifacts are typed, addressed and persisted,
  and assembly-time validation checks `consumes_artifacts` against what is available;
  but producers are not auto-wired to consumers by the sweep: they are connected
  manually/pipeline-style through the same `ArtifactStore`.
- **No backend `loop_affinity` capability (§10)**: `run_sync` remains a process-level
  singleton event loop instead of a capability advertised only when every backend is
  reentrant.
- **No sampling-profile × backend-capability check at assembly time**: a profile's
  `thinking` / tier requirements are not validated against the backend when the run is
  assembled; the four rules in §3 are the complete set.
