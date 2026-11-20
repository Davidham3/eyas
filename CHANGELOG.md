# Changelog

All notable changes to Eyas are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the version follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0rc2] - 2026-11-20

### Added

- **Packaged config spec**: the five spec YAMLs now ship inside the `eyas`
  package (`eyas/spec`), so `pip install "eyas[api]"` alone makes
  `eyas run --cell <dataset>/<method>` work from any directory — no repo
  checkout, no `--spec-dir`.
- **Spec provenance**: generated cell YAMLs carry `spec=<sha12>` (a digest of
  the shipped spec files) in the GENERATED header — a generated file can now
  prove which shipped spec produced it; `run --cell` records the same value in
  `run_config.json`.
- **`eyas run --cell --dry-run`**: assemble a cell from the packaged spec and
  exit — offline preview, and the release smoke check that proves the wheel
  reads its own shipped spec.

### Changed

- `--spec-dir` (run/sweep/gen-configs) now defaults to the spec shipped inside
  the installed package instead of a checkout-relative `configs/spec`; explicit
  directories behave exactly as before.

## [0.1.0rc1] - 2026-11-19

Initial public release.

### Added

- **Contract-driven runtime** (`eyas.engine`): every optimization method is one
  `Strategy` (one-step decision: evaluate / propose / finish) executed by a managed
  `OptimizerLoop` (budget, batched evaluation, cost accounting, checkpoint, resume);
  capability validation rejects illegal assemblies at startup. Authoritative
  contract: `docs/10-architecture.md`.
- **Eight strategies out of the box**: `raw`, `cot`, `hill_climb`, `evoprompt`
  (genetic algorithm + differential evolution), `opro`, `ape`, `protegi`, `spo` — each contract-tested end-to-end on a
  scripted mock backend with zero GPU.
- **Backend-agnostic inference**: in-process vLLM engine or any
  OpenAI-compatible endpoint, identical behavior across both shapes; response
  normalization, multiprocess media prefetch (images and clips), radix-friendly request ordering,
  sliding concurrency windows, and a request planner chokepoint.
- **Shared evaluation infrastructure**: run-directory SQLite evaluation record store
  (cache-first; the run directory *is* the checkpoint), pluggable judges (exact /
  chained / LLM / pairwise), absolute and compare eval models, flat / even-split /
  bandit budget allocators, tiered sampling profiles.
- **Tasks**: multiple-choice (MCQ) / visual question answering (VQA) / key-information
  extraction (KIE) / numeric math QA / video MCQ task families, a mixed-modality MCQ
  task (one split, seven per-sample combinations of passage / image / clip, all rendered
  through the same typed request form) plus zero-external-data demo tasks for the
  30-second pipeline run.
- **Observability & safety**: per-run JSONL trace, live `status.json`, framework
  logging with background writer (a stalled sink never freezes the run loop), and an
  activity watchdog that self-terminates on hangs.
- **Operational CLI**: `eyas list / run / sweep / report / status`, checkpoint
  resume (`--resume`), matrix batch experiments (`sweep --matrix`: one spec —
  shared base template + override-dict axes + combos — expands to the full
  factorial cell set, `--filter` selects cells), plugin registration for
  out-of-tree components.
- **Published benchmark**: 6 open datasets × 7 open models × 8 strategies × 3 seeds
  (1008 runs) with per-cell cost profiles (`docs/08-bench-benchmark.md`,
  machine-regenerated from the run directories); quickstart protocol tier in
  `docs/07-quickstart-benchmark.md`.
- Tutorials (`docs/tutorials/`), runnable starter packages (`examples/`), and CI
  running the full gate (ruff / mypy / pytest) on master pushes and every PR.

### Changed

- **Python support widened to 3.12–3.14** (`requires-python = ">=3.12,<3.15"`),
  with 3.13 and 3.14 classifiers added. Core is covered on 3.12–3.14; the vLLM
  extra remains verified on 3.12.
- **pip-first install story**: `uv pip install "eyas[api]"` (or `[vllm]` /
  `[prepare]`) is the documented path, and the reader tutorials use inline YAML
  so an installed wheel needs no repository files; the benchmark tutorial
  remains a repository-checkout path. `git clone` + `uv sync` remains the
  developer path, documented in `CONTRIBUTING.md`.
- **Tag-driven release pipeline**: pushing a `v*` tag builds the distribution
  once, rehearses it on TestPyPI, runs a 3-OS × Python 3.12/3.13/3.14
  install-and-CLI smoke matrix, then waits for a required reviewer before
  publishing to PyPI and drafting the GitHub Release. No PyPI API tokens:
  publishing uses OIDC trusted publishing.
