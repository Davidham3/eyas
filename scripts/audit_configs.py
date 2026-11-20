#!/usr/bin/env python3
"""Audit experiment configs against the strict validators.

why: fail-fast hardening turns previously-silent paths into raises (unknown
``instruction_pos``, malformed MCQ/KIE/VQA records, unknown registry names,
bad ``--set`` shapes). A config that passes only through a silent path does
not mean what it says — such a config must be fixed, then run again.

what: audits every shipped cell — the config spec's assembled product
(``src/eyas/spec``, shipped inside the package; the primary surface since
spec-as-source) — plus every
per-run YAML under ``configs/`` (tutorial shapes, gen-configs exports) and
``outputs/**/run_config.json``, deduplicating by content hash (and by
task-identity for the expensive dataset checks), then runs the *real* package
validators on each unique config:

  1. structural sections present (run/task/backend/strategy/...)
  2. registry name resolution for every section (task/backend/judge/
     protocol groups via the base registry, the rest via the engine registry)
  3. task construction with the config's params — this is what triggers
     the instruction_pos / required-arg validation
  4. dataset loading for the train split and the configured final-eval
     split — this is what surfaces malformed records

It deliberately does NOT construct backends/judges (GPU + network); only
their names are resolved.

Exit 0 means no config hits a strict path; exit 1 prints the per-config
error classes grouped by kind.

Usage (repo root):
    uv run python scripts/audit_configs.py [--spec DIR] [--configs DIR] [--outputs DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent

import eyas.builtin  # noqa: E402,F401  (registers built-in tasks/backends/judges/protocols)
import eyas.engine.builtin  # noqa: E402,F401  (registers engine strategies/budgets/eval-models/judges)
from eyas import registry as base_registry  # noqa: E402
from eyas.engine import registry as engine_registry  # noqa: E402

# engine-registry group per config section (backend/judge/protocol/tasks live in
# the base registry instead).
_ENGINE_GROUPS = {
    "eval_model": "eval_model",
    "budget": "budget",
    "strategy": "strategy",
}
_BASE_GROUPS = {
    "backend": "backend",
    "judge": "judge",
    "protocol": "protocol",
}


def _shrink(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Drop backend/judge params: they embed model paths and GPU knobs that
    this check never constructs, and would otherwise defeat task-level
    dedup across model variants of the same task matrix."""
    if name in ("backend", "judge"):
        return {**params, "params": "<shrunk>"}
    return params


def _config_key(cfg: dict[str, Any]) -> str:
    norm = {
        sec: _shrink(sec, cfg.get(sec, {}) or {})
        for sec in (*_ENGINE_GROUPS, *_BASE_GROUPS, "task")
    }
    return hashlib.sha256(json.dumps(norm, sort_keys=True, default=str).encode()).hexdigest()


def _task_key(cfg: dict[str, Any]) -> str:
    task = cfg.get("task", {}) or {}
    return json.dumps({"task": task}, sort_keys=True, default=str)


def _display_path(path: Path | str) -> str:
    """Repo-relative when possible, otherwise the origin as given (assembled
    spec cells carry a plain "spec:<cell>" label, not a path)."""
    if isinstance(path, str):
        return path
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _resolve(path: Any) -> Any:
    """Resolve relative data paths against the repo root (configs/*.yaml use
    repo-relative paths; run_config.json files already carry absolute ones)."""
    if isinstance(path, str) and path and not path.startswith(("/", "~")):
        return str((REPO / path).resolve())
    return path


def check_config(cfg: dict[str, Any], origin: str) -> list[str]:
    """Run every cheap-then-expensive validator; return a list of error
    strings (empty = config would pass the new strict pipeline)."""
    errors: list[str] = []

    # 1. structural: the pipeline requires run + task + backend + strategy
    for sec in ("run", "task", "backend", "strategy"):
        if sec not in cfg:
            errors.append(f"structure: missing section '{sec}'")
    if errors:
        return errors

    # 2. registry name resolution (no construction)
    task_cfg = cfg["task"] or {}
    for sec in ("task", *_BASE_GROUPS, *_ENGINE_GROUPS):
        name = (cfg.get(sec) or {}).get("name")
        if name is None:
            if sec == "task":
                errors.append("structure: task.name missing")
            continue
        reg = base_registry if sec in _BASE_GROUPS or sec == "task" else engine_registry
        try:
            reg.get(sec, name)
        except Exception as exc:
            errors.append(f"registry[{sec}]: {type(exc).__name__}: {exc}")

    # 3. task construction (triggers instruction_pos / arg validation)
    task = None
    if not task_cfg.get("name"):
        return errors
    try:
        cls = base_registry.get("task", str(task_cfg["name"]))
        params = dict(task_cfg.get("params", {}) or {})
        for key in ("data_dir", "images_dir"):
            if key in params:
                params[key] = _resolve(params[key])
        task = cls(**params)
    except Exception as exc:
        errors.append(f"task-build[{type(exc).__name__}]: {exc}")
        return errors

    # 4. dataset validation for the splits a run actually consumes
    splits = {"train"}
    final_split = ((cfg.get("run") or {}).get("final_eval") or {}).get("split")
    if final_split:
        splits.add(str(final_split))
    for split in sorted(splits):
        try:
            samples = task.load_split(split)
            if not samples:
                errors.append(f"dataset[{split}]: loaded 0 samples")
        except Exception as exc:
            errors.append(f"dataset[{split}][{type(exc).__name__}]: {exc}")
    return errors


def _is_run_config(cfg: dict[str, Any]) -> bool:
    """Per-run config shape (mirrors ``tests/configs/test_configs.py:_is_run_config``):
    a run config carries ``strategy`` (no legacy ``optimizer``) plus ``run``.
    Anything else — e.g. a sweep spec with ``base``/``set``/``axes`` — is not
    a run config and must not go through the run-section checks."""
    return "strategy" in cfg and "run" in cfg and "optimizer" not in cfg


def iter_configs(configs_dir: Path, outputs_dir: Path, spec_dir: Path):
    """Yield (origin, parsed config) over every auditable face: the assembled
    spec cells first (the shipped surface), then per-run YAMLs under configs/
    (tutorial shapes, gen-configs exports), then the persisted run_config.json
    files. Benchmark specs have no run shape and are skipped by the caller's
    _is_run_config check."""
    from eyas.configspec import cell_config, load_spec

    if spec_dir.is_dir():
        spec = load_spec(spec_dir)
        for dataset in spec.datasets:
            for method in spec.methods:
                yield f"spec:{dataset}/{method}", cell_config(spec, dataset, method)
        for vname, variant in sorted(spec.variants.items()):
            for cellkey in variant.available:
                dataset, method = cellkey.split("/")
                yield (
                    f"spec:{cellkey}@{vname}",
                    cell_config(spec, dataset, method, vname),
                )
    if configs_dir.is_dir():
        for path in sorted(configs_dir.rglob("*.yaml")):
            yield path, yaml.safe_load(path.read_text(encoding="utf-8"))
    if outputs_dir.is_dir():
        for path in sorted(outputs_dir.rglob("run_config.json")):
            yield path, json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="spec directory (default: the spec shipped inside the eyas package)",
    )
    ap.add_argument("--configs", type=Path, default=REPO / "configs")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument(
        "--verbose", action="store_true", help="print each failing config as it is checked"
    )
    args = ap.parse_args()

    # the environment's shipped spec (ruling 5 of the rc2 plan: this script
    # audits what THIS install carries, unlike the pytest gates, which grade
    # the checkout)
    from eyas.configspec import default_spec_dir

    spec_dir = args.spec if args.spec is not None else default_spec_dir()

    total_files = 0
    parse_errors: list[tuple[str, str]] = []
    skipped_specs = 0  # specs and non-run shapes: counted, not checked
    # key -> (first origin seen, cfg)
    unique: dict[str, tuple[str, dict[str, Any]]] = {}
    task_done: dict[str, list[str]] = {}  # task-key -> errors
    task_by_config: dict[str, str] = {}  # config-key -> task-key

    for origin, cfg in iter_configs(args.configs, args.outputs, spec_dir):
        total_files += 1
        if cfg is None:
            parse_errors.append((_display_path(origin), "unreadable/empty config"))
            continue
        if not isinstance(cfg, dict):
            parse_errors.append(
                (_display_path(origin), f"top-level {type(cfg).__name__}, expected mapping")
            )
            continue
        if not _is_run_config(cfg):
            skipped_specs += 1
            continue
        key = _config_key(cfg)
        unique.setdefault(key, (_display_path(origin), cfg))

    # run the expensive checks once per config-key, and once again per
    # task-key dataset identity so identical task/data pairs from different
    # faces cost only one dataset read.
    results: dict[str, list[str]] = {}
    for key, (origin, cfg) in unique.items():
        tkey = _task_key(
            {
                **cfg,
                "task": {
                    **(cfg.get("task") or {}),
                    "params": {
                        k: _resolve(v) if k in ("data_dir", "images_dir") else v
                        for k, v in ((cfg.get("task") or {}).get("params") or {}).items()
                    },
                },
            }
        )
        task_by_config[key] = tkey
        if tkey in task_done:
            results[key] = task_done[tkey]
        else:
            results[key] = check_config(cfg, origin)
            task_done[tkey] = results[key]
        if args.verbose and results[key]:
            print(f"FAIL {origin}: {results[key]}")

    # report
    failed = {k: v for k, v in results.items() if v}
    by_kind: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, errs in failed.items():
        origin = unique[key][0]
        for e in errs:
            by_kind[e.split(":", 1)[0].split("[")[0]].append((origin, e))

    print("\n=== audit summary ===")
    print(f"config faces scanned : {total_files}")
    print(f"non-run specs skipped: {skipped_specs}")
    print(f"parse failures       : {len(parse_errors)}")
    print(f"unique configs       : {len(unique)} (task/data identities: {len(task_done)})")
    print(f"configs passing strict validators: {len(unique) - len(failed)}")
    print(f"configs FAILING (fix, then re-run): {len(failed)}")
    for origin, err in parse_errors:
        print(f"  PARSE  {origin}: {err}")
    for kind, items in sorted(by_kind.items()):
        print(f"\n-- {kind} ({len(items)} site(s)) --")
        for origin, err in items[:20]:
            print(f"  {origin}\n      {err}")
        if len(items) > 20:
            print(f"  ... {len(items) - 20} more of the same kind")
    return 1 if (failed or parse_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
