# Custom optimization methods

**What you'll learn:** the strategy contract (typed config, capability declaration, `init`/`step` decisions, typed state) and a complete ~100-line strategy you can register and run.

## One role, one contract

A method is a **Strategy**: it makes one-step decisions, and the framework executes them. There are no operators or selectors to compose. Mutation rules, few-shot construction and population bookkeeping all live inside the strategy (they are what distinguishes one method from another), while the managed loop owns everything invariant-shaped: budgets, evaluation, cost accounting, history, checkpointing, the watchdog.

```python
class Strategy(ABC):
    name: ClassVar[str]                       # set by @register_strategy
    capabilities: ClassVar[CapSpec]           # declared requirements, validated at assembly
    config_cls: ClassVar[type]                # dataclass; built from the strategy's YAML params
    state_cls: ClassVar[type[OptimizerState]] # typed state; serializes for resume

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState: ...
    def step(self, state: OptimizerState, view: EvalView) -> Decision: ...
```

`step` returns exactly one of three decisions:

- `Decision("evaluate", prompts=...)`: score these prompts; the loop charges the budget allocator, batches the requests and files results into the archive.
- `Decision("propose", prompts=...)` is informational: new candidates already entered `state`; they get evaluated on a later `evaluate` step.
- `Decision("finish", best_id=...)`: stop; `best_id` names the winner (the loop double-checks against its own archive ranking).

Two rules keep strategies honest:

- **Strategies never see the raw backend.** LLM-backed propose goes through `self.generate(requests, tier="generate")`, which merges the tier's sampling profile with deployment defaults, records per-tier request counts, and routes through the resource planner.
- **Strategies never write scores.** The `EvalView` given to `step` is read-only: `view.evaluated(prompt)`, `view.score(prompt_id)`, `view.per_sample(prompt_id)`, a window into the archive, nothing else.

## Typed state

`state_cls` is any `OptimizerState` subclass with `to_json`/`from_json`; the loop snapshots it into `checkpoint.json` every round and restores it on `--resume`. `PromptListState` (a plain list of `Prompt`s plus a round counter) covers search-style methods; richer methods define their own state dataclass. Derive child RNGs from `self.derived_rng()` so resume reproduces exactly.

## The reference implementation

The template is [`examples/strategy_skeleton/`](../../examples/strategy_skeleton/): a complete method, its scripted contract test, and the two configs that run it. Copy that directory, replace the symbols its README lists, and you have your own method. It is deliberately small, and its test is the shape the framework's own contract tests ride on. The kit ships its own `random_search` (four knobs: `neighborhood_size` / `edit_probability` / `seed` / `max_rounds`); the inline walkthrough below teaches the shape with its own `my_random` variant, so the two never collide as duplicate registrations in one process. The reference method below is the same `step`, in full spirit:

```python
def step(self, state, view) -> Decision:
    unevaluated = [p for p in state.candidates if not view.evaluated(p)]
    if unevaluated:
        return Decision("evaluate", prompts=tuple(unevaluated))

    def key(p):
        score = view.score(p.id)
        return -inf if score is None else score

    ranked = sorted(state.candidates, key=key, reverse=True)
    if state.round_idx >= self.config.rounds:
        return Decision("finish", best_id=ranked[0].id)
    state.round_idx += 1

    offspring = []
    for parent in ranked[: self.config.keep_size]:
        for i in range(self.config.per_parent):
            child = Prompt(f"{parent.text} {self.config.mutations[i % len(self.config.mutations)]}")
            if not view.evaluated(child):
                offspring.append(child)
    state.candidates = ranked[: self.config.keep_size] + offspring
    return Decision("propose", prompts=tuple(offspring))
```

Evaluate-what's-unseen first, then rank, then expand. The loop interleaves the two decision types for you: after a `propose`, your next `step` will typically return `evaluate` for the fresh offspring. One caveat this pattern hides: the reference `step` evaluates the *entire* unevaluated set, so its rank-then-expand sees every child scored. If you cap the evaluate batch (any `pool_size`-style knob), unscored in-pool candidates read `None`: rank among {in-pool + scored}, or do not propose again until the pool drains; otherwise `None`-as-worst silently reverts your parent to the last scored incumbent while your queued children wait.

## Write your own: random search

A deliberately simple method that still exercises every part of the contract: each round, mutate the current best `n_candidates` ways, let the loop score everything, keep the single best.

```python
# my_strategy.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf
from typing import ClassVar

from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import OptimizerState, Prompt, PromptListState
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, Strategy


@dataclass
class MyRandomConfig:
    rounds: int = 3
    n_candidates: int = 8


@register_strategy("my_random")
class MyRandomStrategy(Strategy):
    """Each round: n_candidates mutations of the best; the best of all survives."""

    config_cls: ClassVar[type] = MyRandomConfig
    state_cls: ClassVar[type] = PromptListState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="my_random", requires_eval=frozenset({ABSOLUTE_EVAL})
    )

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        return PromptListState(candidates=list(seeds), round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, PromptListState)
        cfg = self.config

        unevaluated = [p for p in state.candidates if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))

        def key(p):
            score = view.score(p.id)
            return -inf if score is None else score

        best = max(state.candidates, key=key)
        if state.round_idx >= cfg.rounds:
            return Decision("finish", best_id=best.id)
        state.round_idx += 1

        rng = self.derived_rng()
        suffixes = ("Think carefully.", "Answer concisely.", "Double-check the image.")
        offspring = []
        for _ in range(cfg.n_candidates):
            child = Prompt(f"{best.text} {rng.choice(suffixes)}")
            if not view.evaluated(child):
                offspring.append(child)
        if not offspring:
            return Decision("finish", best_id=best.id)
        state.candidates = [best, *offspring]
        return Decision("propose", prompts=tuple(offspring))
```

Points that generalize to any strategy:

- `capabilities` is your contract with the assembly validator. `requires_eval={ABSOLUTE_EVAL}` means "I rank by absolute scores"; a compare-based method would declare the compare requirement instead, and `validate_assembly` rejects it against an accuracy leaderboard at startup, before any GPU time.
- `config_cls` is a plain dataclass: the run entry reads the strategy's `params` mapping from YAML into it.
- Dedup is free, and it is bounded: `Prompt` identity is the first 12 hex chars of the SHA-256 of its *normalized* text (NFKC + stripped), so the same candidate spelled differently is still the same candidate. The record store lives in the run dir keyed by (prompt, sample, split), so nothing is ever scored twice **within a run**, a `--resume` of the same run dir included. Sharing across runs happens only when you point `run.cache_dir` at a path two runs both use; nothing is deduped "under the same name" otherwise.
- Missing scores read as `None` in the view (treat as worst); the archive's primary ranking uses the task's first objective, oriented for maximization. Among equal scores the archive orders by ascending `prompt_id` (the stable content hash), never by insertion order, and nothing more is specified; if plateau parent-selection matters to your method, carry your own tie-break signal (a discovery counter in your state is the usual choice).

## Run it

`rs_demo.yaml` is [`configs/tutorial_shapes.yaml`](../../configs/tutorial_shapes.yaml) with two edits: a `plugins` line on top, and the `strategy` section swapped for your method. The zero-data shapes task keeps the smoke test free of datasets, and the `openai` backend section (base_url / model / api key env: point them at any OpenAI-compatible endpoint, as in [Inference backends](backends.md)) keeps it free of in-process engines:

```yaml
# rs_demo.yaml
plugins: [my_strategy]

run:
  name: rs_demo
  seed: 42

task:
  name: demo_shapes
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
  name: absolute

budget:
  name: flat

strategy:
  name: my_random
  params:
    rounds: 2
    n_candidates: 4
```

```bash
uv run eyas run --config rs_demo.yaml
```

`eyas list --config rs_demo.yaml` now shows `my_random` among the strategies (plain `eyas list` only imports the built-ins; it reads no config, so plugin registrations are invisible to it). Registration routes are the same as for tasks, and shared with backends, protocols, and judges:

- the config `plugins` key, which imports your module before any registry lookup (what the config above does):

  ```yaml
  plugins: [my_strategy] # dotted module paths resolve against the CWD — run from the repo root
  ```

- the `eyas.plugins` entry-points group for packaged plugins (no `plugins` line anywhere):

  ```toml
  # pyproject.toml of your own package
  [project.entry-points."eyas.plugins"]
  my_plugins = "my_pkg.eyas_plugins" # a module whose import performs the registration
  ```

- a `builtin.py` import for in-framework contributions.

One trap spans all routes: import your plugin module under a **single canonical dotted path**. Python re-executes a module imported via two different paths (e.g. `plugins: [my_strategy]` from the CWD plus a test importing it as `examples.foo.my_strategy`), which re-runs the decorators against fresh class objects and trips the duplicate-name guard; pick the path that both your config and your tests resolve to.

The worked example for both routes, with registration, a runnable config, and a CI test that pins the whole story, is [`examples/plugin_backend/`](../../examples/plugin_backend/README.md) (a backend; the mechanism is identical for strategies).

## Test it

The contract-test pillar applies to strategies too: your method must complete with zero GPU, and the framework's own suite proves every shipped strategy does exactly that on its scripted in-repo test backend. The in-hand equivalent is [`examples/strategy_skeleton/my_method_test.py`](../../examples/strategy_skeleton/my_method_test.py), a self-contained test that assembles a `RunContext` against that same scripted test backend, drives `OptimizerLoop(...).run(strategy, seeds)` directly, and asserts on `result.best` and the history. Running the config above is the smoke version of the same contract.

## Where to go from here

- Further reading, source access only: none of it is a prerequisite, and the skeleton plus this tutorial is the whole path. With the framework source checked out, `src/eyas/engine/strategies/opro.py` (meta-prompting) and `protegi.py` (feedback gradients) show `self.generate(..., tier=...)` and `ctx.task.load_split(...)` few-shot grounding in real use, and the `bandit` budget allocator shows how an allocator reshapes per-candidate sample loads from observed scores.
- [Understanding results](understanding-results.md): how your strategy's history and best prompt surface in `result.json`.
