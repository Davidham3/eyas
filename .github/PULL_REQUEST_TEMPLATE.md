# Pull request

## What it does

<!-- One paragraph: the behavior change, and why. Link the issue it closes if any. -->

## Gate

- [ ] `uv run ruff check src tests` green
- [ ] `uv run ruff format --check src tests` green
- [ ] `uv run mypy src` green
- [ ] `uv run pytest` green (fast tier; GPU tier is opt-in)

## Notes for the reviewer

- New/changed component goes through its registry decorator, and the contract test
  runs on `ScriptedMockBackend` (zero GPU)?
- Interface signature changed? The authoritative contract is
  `docs/10-architecture.md` — the PR should carry that edit in the same diff.
- Behavior or public surface changed? User-facing docs (`docs/index.md`,
  `docs/tutorials/`) and `CHANGELOG.md` are updated here, not later.
