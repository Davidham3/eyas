# Contributing to Eyas

Thanks for taking the time. This document is the short version of how the project is
developed; the authoritative architecture reference is
[`docs/10-architecture.md`](docs/10-architecture.md).

## Development loop

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/). The core carries zero
  heavy dependencies; the inference engine goes through the `eyas[vllm]` extra and the
  light `eyas[api]` extra composes with it.

```bash
uv sync                          # core dependencies (seconds, no torch)
uv run pytest                    # fast tier (GPU tests excluded automatically)
uv run ruff check src tests      # lint (ruff handles lint + import order)
uv run ruff format src tests     # formatting
uv run mypy src                  # type check
```

- **The gate before every PR**: `ruff check` + `ruff format --check` + `mypy src` +
  `pytest` all green — exactly what CI runs (`.github/workflows/ci.yml`). The PR
  template carries the checklist; user-facing changes also update `CHANGELOG.md`.
- Security vulnerabilities: report privately, see [`SECURITY.md`](SECURITY.md).
- Contract tests run on the `ScriptedMockBackend` (zero GPU): every strategy, backend
  switch, judge, and runtime behavior is testable without an engine. If you add a
  component, write the contract test in `tests/` before the implementation.
- **Changing the config spec (`configs/spec/`)?** Regenerate the golden manifests with
  `python scripts/gen_configspec_goldens.py --regen` and read the diff before committing
  — the manifests under `tests/goldens/configspec/` are a review surface: a drifted
  digest line is a real behavior change. CI validates the same invariant through the
  default check-level tests (`tests/test_configspec.py`).
- GPU smoke (`uv run pytest -m slow`) is opt-in and needs a real engine env; CI does
  not require it.

## Where things live

- `src/eyas/engine/` — the runtime: Strategy contract + OptimizerLoop, budgets,
  evaluators, planner, reports.
- `src/eyas/backends|tasks|judges/` — the shared inference/task/judging infrastructure
  the runtime rides on.
- `docs/` — numbered design notes: `baselines-fidelity.md` (the method-fidelity
  contract), `07`/`08` (the benchmark records), `10` (the frozen architecture
  contract). The numbers are stable identifiers — gaps in the sequence are
  intentional, and cross-references elsewhere use the same numbers.
- `configs/` — runnable YAMLs (demo / quickstart / matrix).

## Adding a component (strategy example)

1. Create a module in `src/eyas/engine/strategies/` implementing `Strategy.init(seeds)` and
   `Strategy.step(state, view)`; declare capabilities via `CapSpec`.
2. Register with `@register_strategy("my_strategy")` (`eyas.engine.registry`) and add one
   import line in `src/eyas/engine/builtin.py`.
3. Add the contract test: run `OptimizerLoop` end-to-end on a toy task +
   `ScriptedMockBackend`, assert `result.best` and the round history.
4. Update the component inventory in `docs/10-architecture.md` (§11 carrying matrix
   and the §12 status table).

## Conventions worth knowing

- Registry names are unique; duplicate registration fails fast at import.
- A `Prompt`'s only identity is its `prompt_hash` (label/metadata are display-only).
- Every evaluation path consults the run-dir record store first
  (`<run_dir>/cache/eyas.sqlite`); the run directory is the checkpoint.
- Errors fail fast — no silent fallbacks (the one exception: a missing metric value
  counts as 0.0).
- Keep IO and heavy dependencies out of `core/`-style pure modules; inference and
  persistence happen only in backends / evaluation / runtime.

## Commit style

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:` …),
imperative mood, one logical change per commit.
