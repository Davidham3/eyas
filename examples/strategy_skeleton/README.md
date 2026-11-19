# Strategy skeleton — your method, running in three commands

A complete optimization method in one file, plus the scripted contract test and the
configs
that prove it works. Copy this directory, rename the symbols, and you have your
method; nothing here is framework-internal, and nothing in it needs a GPU.

Pair it with [`docs/tutorials/custom-method.md`](../../docs/tutorials/custom-method.md)
for the contract in prose. Everything you need to write and validate a method is
in these two files — you do not have to read framework source.

## Contents

| File | What it is |
| --- | --- |
| `my_method.py` | The method: typed config + capability declaration + `init`/`step` |
| `my_method_test.py` | Scripted contract test — toy task, scripted backend, expected outcome, all in one file |
| `demo_config.yaml` | The same method on the zero-data `demo_shapes` task and a scripted in-repo test backend |
| `run_config.yaml` | The same method against a real OpenAI-compatible endpoint |

## Run the tests

From the repo root, with the repo's own environment:

```bash
uv run pytest examples/strategy_skeleton -q
```

From a fresh clone, with no eyas install step — enter this directory and point
Python at the checkout's own source tree (so no stale `eyas` distribution can
answer for it). `uv run` discovers the repo's root project from any subdirectory,
so it supplies pytest from that environment while `PYTHONPATH` keeps the `eyas`
import resolved from this checkout's `src/`:

```bash
cd examples/strategy_skeleton && PYTHONPATH=../../src uv run python -m pytest -q
```

Both shapes import the package as `examples.strategy_skeleton`, which is why the
test file resolves the repo root itself instead of relying on the current
directory. The repo gate collects this file automatically (`testpaths` in
`pyproject.toml` includes `examples`), so the skeleton cannot rot silently. Keep
your copy's file name matching pytest's `test_*` / `*_test` patterns — a test file
named anything else is silently never collected, which is the one failure mode
that looks like success.

Expected output of the first command (`uv run pytest examples/strategy_skeleton -q`)
— seven tests, no GPU, no data download:

```
.......                                                                  [100%]
7 passed in 0.32s
```

(The no-argument form in the second block collects the repo-wide `testpaths` from
`pyproject.toml` — `tests` and `examples` — so it runs the whole fast suite, not just
these seven tests. Exact timings — and whether pytest reports a stray environment
warning next to the count — vary by environment; the count is the contract.)

## Run the scripted demo

```bash
PYTHONPATH=src uv run eyas run --config examples/strategy_skeleton/demo_config.yaml
```

(The `PYTHONPATH=src` prefix matters only if the `eyas` in your environment is
older than this checkout; a fresh `uv sync --group dev` environment runs the
command as-is.)

The run prints one line per completed round and finishes with the winning
prompt. It writes a run directory under `outputs/skeleton_random_search/<stamp>/`:

| File | What you get |
| --- | --- |
| `result.json` | `best` (hash + text + metrics), `ranked`, `history`, `total_cost` |
| `status.json` | live observation: `phase` (`starting` → `optimizing` → `done`), round cursor, cost |
| `checkpoint.json` | one snapshot per **evaluate** decision — `propose`/`finish` never write one (they spend a `max_steps` slot but produce no history and no checkpoint; interrupted right after a `propose`, resume replays that propose from the last evaluate checkpoint). `uv run eyas run --config <same yaml> --resume <run dir>` replays it (and `--resume` excludes `--output-dir`) |
| `cache/eyas.sqlite` | this run's evaluation record store (see “Facts” below) |
| `framework.log` | the run's log, including assembly-time rejections |

`uv run eyas status <run dir>` renders the same thing without opening JSON.
Pass `--output-dir <dir>` to choose the directory yourself.

Expect `accuracy` around 0.25 here: the scripted backend answers `"red"` to everything and
the demo task's labels cycle over four colors, so the number measures the
pipeline, not the search. It is there to confirm the parts connect.

The round echo's request counts shrink as the run goes (`1 → 3 → 2 → 1`
candidates): mutations that land on text the archive has already scored are
filtered out before they are queued, so a neighborhood of 3 is an upper bound,
not a guarantee. Watch that column when you tune your own mutation vocabulary —
if it collapses to 1 every round, the cause is on your side of the line: either your
edits are not diverse enough, or several picks collapse to the *same* normalized text
per round (dedup keeps one). Print the accepted candidates, not just the count.

## Go real

Edit `run_config.yaml` — `base_url`, `model`, and the task/dataset you want —
export the key it reads (`export EYAS_API_KEY=...`), and run it:

```bash
uv run eyas run --config examples/strategy_skeleton/run_config.yaml \
  --set backend.params.model=your-model \
  --set strategy.params.max_rounds=1
```

`--set` rewrites a key that already exists in the YAML and fails fast on a path
that is not there, which is why `run_config.yaml` spells every tunable key out.
To drop the closing evaluation on the reporting split, either delete its two
`run.final_eval` lines or set `--set run.final_eval=null` (the key exists in
`run_config.yaml`, so the override is legal).

## What to replace

| Symbol | Replace with |
| --- | --- |
| `RandomSearchStrategy` + `@register_strategy("random_search")` | your class name and your registry key — that key is what `strategy.name` in a YAML config selects |
| `RandomSearchConfig` | your knobs; the config's `params:` mapping is bound to it **by keyword**, so field names must match your YAML |
| `step()` | your policy: what to score, what to try next, when to `finish` |
| `SkeletonToyTask`, `_LABELS`, `_answering_from_the_prompt` in the test | a toy task your method can actually improve, and the assertions that prove it |
| the registration name in the two YAML files | the same key as `@register_strategy(...)` |
| `__init__.py` here and in `examples/` | keep both — they are what makes the package path below importable |
| the module path `examples.strategy_skeleton.my_method` (both configs' `plugins:` key and the import in the test) | where your copy lives — the plugins key imports a dotted path resolved against the repo root, so a copy in `examples/my_method/` becomes `examples.my_method.my_method` |

Keep the four class attributes (`name`, `config_cls`, `state_cls`,
`capabilities`) — they are what the framework reads before it runs a step.

## Your own state (when `PromptListState` is not enough)

A per-mutation-kind win counter, a beam, a population — that bookkeeping lives in
a state class of your own. The loop serializes with `to_json()` at each evaluate
checkpoint and restores by calling `state_cls.from_json(data)` — a **classmethod**
returning an instance — so the payload must be plain JSON (dict / list / str /
int / float / bool / None) and the round trip must be lossless:

```python
from __future__ import annotations  # or quote the return annotation: -> "MyState"


@dataclass
class MyState(OptimizerState):  # and: state_cls: ClassVar[type] = MyState
    candidates: list[Prompt] = field(default_factory=list)
    round_idx: int = 0
    wins: dict[str, int] = field(default_factory=dict)  # your bookkeeping

    def to_json(self) -> dict[str, Any]:
        return {
            "candidates": [p.text for p in self.candidates],
            "round_idx": self.round_idx,
            "wins": dict(self.wins),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MyState:
        return cls(
            candidates=[Prompt(t) for t in data.get("candidates", [])],
            round_idx=int(data.get("round_idx", 0)),
            wins=dict(data.get("wins", {})),
        )
```

Store prompt **text** and rebuild `Prompt` on the way back (identity is derived
from text, so it survives the trip). A field you forget in one direction does not
crash — it silently resumes as if it were empty, which is how a "passes fresh,
drifts on resume" bug is born. The inverse hazard bites the other way too: after a
resume, `step` re-runs over candidates the pre-crash run already scanned, so
bookkeeping that credits what it sees in the view each step must be **idempotent**
(derive counters from the stored outcomes, or gate crediting on a monotonic
watermark kept in your state) — a losslessly serialized counter can still
double-credit. Set `state_cls` to your class and read the
narrowed type inside `step` with an `isinstance` assert, as `my_method.py` does.

## Facts worth knowing before you write your step

Each of these is a behavior the framework has, verified against its
implementation. Your `step` is only correct if you know them.

- **The toy task is what makes the test a test.** The scripted backend here decides the answer
  from the prompt text — `_answering_from_the_prompt` returns the correct label
  only when the system prompt contains `step by step` — so the score has a peak a
  search can climb, and `accuracy == 1.0` proves the method proposed, scored and
  kept a better candidate. A backend that ignores the prompt gives every candidate
  the same score, and then the strongest thing your test can say is "it ran".
  Beware the third case: a reward that needs **two** independent cues at once
  reads as 0.0 for every single-step mutation — indistinguishable from a broken
  method. Make sure at least one of your mutations moves the score alone (or
  prove your method composes mutations within its round budget).
- **Three decisions, one at a time.** `Decision("evaluate", prompts=...)` scores
  exactly those prompts; `Decision("propose", prompts=...)` is *informational* —
  it records candidates you already put into `state`, it does not score them;
  `Decision("finish", best_id=...)` ends the run. A `propose` is normally
  followed by an `evaluate` for the same candidates.
- **Seeds cross the boundary as text.** `OptimizerLoop.run(seeds=...)` takes
  **strings** and wraps them into `Prompt`s before calling your `init(seeds)` —
  the same parameter name two frames away carries different types.
- **`max_rounds`-style knobs are your own; the loop has a separate ceiling.** The
  run's `run.max_steps` (default 400) counts *decisions*, and this method spends
  one to propose and one to evaluate per round. Hit the ceiling first and the
  loop stops you silently — no error, just a run that ended early.
- **Scores come back through the view only.** After an `evaluate` round, read
  `view.evaluated(prompt)`, `view.score(prompt_id)`, `view.per_sample(prompt_id)`.
  A prompt nobody scored reads `None` — treat it as worst, not as zero.
- **Per-sample observations are readable from the archive.**
  `view.per_sample(prompt_id)` returns every sample's prediction, per-sample
  scores and `correct` flag for that prompt's archived best — complete even when
  the round was served from the record store (zero requests) and even after
  `--resume`, because the checkpoint serializes them. Error-driven methods read
  their gradient signal here; you do not have to cache sample-level detail in
  your own state. It is the archive view, not the raw record store: a subset
  round yields subset rows.
- **Dedup: identity is the first 12 hex chars of SHA-256 over NFKC-normalized,
  stripped text.** `label`/`metadata` never take part in identity, so two
  candidates whose text normalizes to the same thing are the same candidate —
  the loop deduplicates within one `evaluate` decision by identity (the record
  store fills only after generation, so in-batch duplicates would otherwise be
  sent and billed twice; the drop is marked with a DEBUG line). Still filter with
  `view.evaluated(...)` before queueing, as `step()` here does — the store only
  knows texts scored in *earlier* rounds, and queuing a known-scored prompt
  wastes nothing but a slot.
- **The record store lives in the run directory**, keyed by (prompt, sample, split): nothing
  already scored is paid for again — not later in the run, and not on a `--resume`
  of the same run dir (the checkpoint also restores the archive, so your method is
  never even asked to re-queue a scored candidate). Cross-run sharing happens only
  if you set `run.cache_dir` to a path two runs both use — and it is per record,
  so any (prompt, sample, split) record already stored there replays.
- **Randomness: `self.derived_rng()` only.** It is derived from the run's stream,
  so a resumed run replays your mutations exactly. Never call `random.*` global
  functions, and never reseed from the clock.
- **State must serialize.** `state_cls` is an `OptimizerState` with `to_json` /
  `from_json` round-tripping through `checkpoint.json`; `PromptListState`
  (candidates + round counter) covers search-style methods. See the section above
  for the shape of your own.
- **The protocol multiplies requests per candidate; `identity` does not.** A task
  renders a prompt into one or more *variants* per sample (that is what
  `protocol:` in the config / `protocol=` in the test selects). `IdentityProtocol`
  emits exactly one, so `requests == candidates × samples` — the arithmetic the
  cost test above pins. Only reach for another protocol if your method's identity
  really lives in the rendering.
- **No judge in this kit**: the task's own `score()` decides correctness. A judge
  (`judge:` in the config, `judge=` in the test) is for when parsing the response
  is not enough to score it — budget for it, a judge verdict costs an extra
  request per sample.
- **Ranking is one objective**: the task's first objective, oriented for
  maximization. Ties are broken by ascending `prompt_id` (the stable content
  hash), never by insertion order, and the behavior among equal scores is otherwise
  unspecified — if your parent selection on a plateau depends on which co-optimum
  wins, carry your own tie-break signal (a discovery counter in your state is the
  usual choice); a hill-climber that keeps re-picking the first-seen incumbent
  never compounds its mutations. `capabilities.requires_eval` defaults to absolute
  scores; a
  compare-based method declares `COMPARE_EVAL` instead and is then rejected at
  startup against an accuracy leaderboard — that error is the framework saving
  you a GPU run.
- **If you need an LLM to propose candidates**, call `self.generate(requests,
  tier="gradient")` from `step` — it takes `ChatRequest`s and returns one response
  **text** per request, in order. The tier is a name, not a sampling config:
  `generate` / `evolve` / `gradient` / `synonym` / `compare` are the built-in
  conventions (each carries its temperature/top-p/max-tokens; there is deliberately
  no `eval` tier — evaluation rides the deployment sampling), an unknown name
  raises with the options listed, and a config can add or retune tiers through
  `sampling_profiles:`. Thinking and seed always come from the deployment layer
  (`backend.sampling`), never from your method, and requests are counted per tier.
  The contract test for such a method returns scripted generations the same way it
  returns scripted answers.
- **Few-shot grounding lives on the context**: `self.ctx.task.load_split(split)`
  gives you samples (with their `references`), `self.ctx.sample_ids` the ids of the
  split the loop optimizes. Use `eval_split`, not a hardcoded `"train"`, or your
  method quietly scores the wrong rows when a config changes the split.
- **Registries are process-global; duplicate names are an error.** Registering a
  name twice (your toy task colliding with a built-in, or two modules claiming one
  strategy name) raises `ValueError: duplicate registration` at import — which
  arrives as a pytest collection error rather than a test failure. Give a test's
  toy task a name nothing else uses, and note that `eyas list` lists what the
  framework ships: a strategy registered through a config's `plugins:` key appears
  in that run's own registry, not in the bare `eyas list` output.
- **`parse_fail` is always in the metric set.** The framework adds it next to your
  task's objectives (a sample whose parsed prediction was `None` scores 1.0 on it),
  which is why the evaluator's `produces` set carries it even when your task's
  `score()` never mentions it. A metric key missing from an evaluation reads 0.0.
- **Errors inside a run fail fast** — a missing config key, an unregistered name,
  or an incompatible assembly raises at startup. Nothing falls back quietly, so
  read the first traceback, not the last.
