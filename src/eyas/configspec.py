"""Config spec: the committed human-authored configuration surface.

``configs/spec/`` is the single source for the quickstart cells and the
benchmark matrix; per-cell YAMLs are not committed (generated on demand via
``eyas gen-configs``, gitignored). Five files, split by review concern:

    defaults.yaml    shared assembly skeleton (what every cell agrees on)
    datasets.yaml    per dataset: kind, task section, bench dirs, prompt assets
    methods.yaml     per method: strategy identity, kind-gated patches, judge
    cells.yaml       per-cell exceptions + the openai variant (definition,
                     availability list, per-cell mirror exceptions)
    benchmark.yaml   the published matrix: axes / combos / run_name template

Layered merge (later wins; dotted-path bundles with the shared semantics of
:mod:`eyas.runtime.matrix` — mapping values replace, ``None`` deletes):

    defaults  <  datasets[ds].task  <  methods[m] (strategy, kind-gated
    task_patch, backend_patch, budget, kind-gated judge)  <  dataset prompt
    asset for m  <  cells["<ds>/<m>"]  <  variant (backend replacement + per-
    cell override)  <  run.name rule

A dataset's ``kind`` is the gate for the method sections that only some tasks
implement (assistant_prefix / instruction_pos / raw_template / the LLM judge):
a patch a task does not declare is never applied. ``why:`` fields are
fidelity notes kept beside the facts they explain — stripped from the assembly
output, rendered back into ``gen-configs`` file headers.

The assembly output is byte-for-byte the effective config the deleted
per-cell YAMLs carried (golden parity, ``tests/test_configspec.py``); the
runtime strict validator (``eyas.engine.runtime.load_config`` semantics)
remains the only source of validity — this module assembles, never invents.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eyas.runtime.matrix import (
    AxisValue,
    MatrixSpec,
    apply_bundle,
    expand_matrix,
    load_matrix_spec,
)

__all__ = [
    "Spec",
    "benchmark_axes_spec",
    "benchmark_cells",
    "cell_config",
    "cell_digest",
    "load_spec",
    "render_cell_yaml",
    "spec_sha",
]

#: one cell's identity inside cells.yaml / variant availability lists
_CELL_KEY = "dataset/method"


@dataclass(frozen=True)
class DatasetSpec:
    kind: str
    why: str
    task: dict[str, Any]
    bench: dict[str, Any]
    prompts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MethodSpec:
    why: str
    strategy: dict[str, Any]
    task_patch: dict[str, dict[str, Any]]  # dataset kind -> bundle
    backend_patch: dict[str, Any]
    budget: dict[str, Any] | None
    judge: dict[str, dict[str, Any]]  # dataset kind -> judge section


@dataclass(frozen=True)
class CellOverride:
    why: str
    set: dict[str, Any]


@dataclass(frozen=True)
class VariantSpec:
    why: str
    backend: dict[str, Any]
    available: tuple[str, ...]
    overrides: dict[str, CellOverride]


@dataclass(frozen=True)
class Spec:
    path: Path
    defaults: dict[str, Any]
    datasets: dict[str, DatasetSpec]
    methods: dict[str, MethodSpec]
    cells: dict[str, CellOverride]
    variants: dict[str, VariantSpec]
    benchmark: MatrixSpec | None


def _require_mapping(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(raw, f"spec file {path}")


def _entry_why(raw: dict[str, Any], what: str) -> str:
    why = raw.get("why", "")
    if not isinstance(why, str):
        raise ValueError(f"{what}: why must be a string")
    return why


def _reject_unknown(raw: dict[str, Any], known: set[str], what: str) -> None:
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{what}: unknown keys {sorted(unknown)} (known: {sorted(known)})")


def _cell_key(raw: object, what: str) -> str:
    if not isinstance(raw, str) or "/" not in raw or raw.count("/") != 1:
        raise ValueError(f"{what}: key must be in {_CELL_KEY} form, got {raw!r}")
    return raw


def _cell_bundle(raw: object, what: str) -> CellOverride:
    """One override entry: either a mapping with an optional ``why`` and a
    ``set`` bundle, or a bare bundle (no note needed)."""
    if (
        isinstance(raw, dict)
        and "set" in raw
        and all("." not in k and k in {"why", "set"} for k in raw)
    ):
        cm = _require_mapping(raw, what)
        return CellOverride(_entry_why(cm, what), _require_mapping(cm["set"], f"{what}.set"))
    return CellOverride("", _require_mapping(raw, what))


def _bundle_map(raw: object, what: str) -> dict[str, Any]:
    return {k: v for k, v in _require_mapping(raw if raw is not None else {}, what).items()}


def _parse_datasets(raw: dict[str, Any], spec_dir: Path) -> dict[str, DatasetSpec]:
    out: dict[str, DatasetSpec] = {}
    for name, body in raw.items():
        what = f"datasets.{name}"
        bm = _require_mapping(body, what)
        _reject_unknown(bm, {"kind", "why", "task", "bench", "prompts"}, what)
        kind = bm.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{what}: kind must be a non-empty string")
        task = _require_mapping(bm.get("task"), f"{what}.task")
        if not isinstance(task.get("name"), str) or not task["name"]:
            raise ValueError(f"{what}.task: missing component name")
        prompts_raw = _require_mapping(bm.get("prompts") or {}, f"{what}.prompts")
        prompts: dict[str, dict[str, Any]] = {}
        for method, asset in prompts_raw.items():
            prompts[method] = _require_mapping(asset, f"{what}.prompts.{method}")
        out[name] = DatasetSpec(
            kind=kind,
            why=_entry_why(bm, what),
            task=task,
            bench=_bundle_map(bm.get("bench"), f"{what}.bench"),
            prompts=prompts,
        )
    if not out:
        raise ValueError(f"spec {spec_dir}/datasets.yaml: no datasets")
    return out


def _parse_methods(raw: dict[str, Any], spec_dir: Path) -> dict[str, MethodSpec]:
    out: dict[str, MethodSpec] = {}
    for name, body in raw.items():
        what = f"methods.{name}"
        bm = _require_mapping(body, what)
        _reject_unknown(
            bm, {"why", "strategy", "task_patch", "backend_patch", "budget", "judge"}, what
        )
        strategy = _require_mapping(bm.get("strategy"), f"{what}.strategy")
        if not isinstance(strategy.get("name"), str) or not strategy["name"]:
            raise ValueError(f"{what}.strategy: missing component name")
        for section in ("task_patch", "judge"):
            gate_raw = bm.get(section) or {}
            gm = _require_mapping(gate_raw, f"{what}.{section}")
            for kind, gated in gm.items():
                _require_mapping(gated, f"{what}.{section}.{kind}")
        budget = bm.get("budget")
        if budget is not None:
            _require_mapping(budget, f"{what}.budget")
        out[name] = MethodSpec(
            why=_entry_why(bm, what),
            strategy=strategy,
            task_patch={k: dict(v) for k, v in (bm.get("task_patch") or {}).items()},
            backend_patch=_bundle_map(bm.get("backend_patch"), f"{what}.backend_patch"),
            budget=dict(budget) if budget is not None else None,
            judge={k: dict(v) for k, v in (bm.get("judge") or {}).items()},
        )
    if not out:
        raise ValueError(f"spec {spec_dir}/methods.yaml: no methods")
    return out


def _parse_cells(raw: dict[str, Any], spec_dir: Path) -> dict[str, CellOverride]:
    out: dict[str, CellOverride] = {}
    for key, body in raw.items():
        cell = _cell_key(key, f"cells entry of {spec_dir / 'cells.yaml'}")
        out[cell] = _cell_bundle(body, f"cells.{cell}")
    return out


def _parse_variants(raw: dict[str, Any], spec_dir: Path) -> dict[str, VariantSpec]:
    out: dict[str, VariantSpec] = {}
    for name, body in raw.items():
        what = f"variants.{name}"
        bm = _require_mapping(body, what)
        _reject_unknown(bm, {"why", "backend", "available", "overrides"}, what)
        backend = _require_mapping(bm.get("backend"), f"{what}.backend")
        available_raw = bm.get("available")
        if not isinstance(available_raw, list) or not all(
            isinstance(v, str) for v in available_raw
        ):
            raise ValueError(f"{what}.available must be a list of {_CELL_KEY} keys")
        available = tuple(_cell_key(v, f"{what}.available") for v in available_raw)
        overrides: dict[str, CellOverride] = {}
        for key, body2 in _require_mapping(bm.get("overrides") or {}, f"{what}.overrides").items():
            cell = _cell_key(key, f"{what}.overrides")
            if cell not in available:
                raise ValueError(f"{what}.overrides.{cell}: cell is not in {what}.available")
            overrides[cell] = _cell_bundle(body2, f"{what}.overrides.{cell}")
        out[name] = VariantSpec(_entry_why(bm, what), backend, available, overrides)
    return out


def load_spec(spec_dir: str | Path = "configs/spec") -> Spec:
    """Load and validate the five spec files. Fail fast on any structural
    problem (unknown keys, dangling dataset/method/cell references) — before a
    single cell runs."""
    spec_dir = Path(spec_dir)
    defaults = _load_yaml(spec_dir / "defaults.yaml")
    for section in ("run", "backend"):
        _require_mapping(defaults.get(section), f"defaults.{section}")
    datasets_raw = _load_yaml(spec_dir / "datasets.yaml")
    _reject_unknown(datasets_raw, {"datasets"}, f"spec file {spec_dir / 'datasets.yaml'}")
    datasets = _parse_datasets(datasets_raw["datasets"], spec_dir)
    methods_raw = _load_yaml(spec_dir / "methods.yaml")
    _reject_unknown(methods_raw, {"methods"}, f"spec file {spec_dir / 'methods.yaml'}")
    methods = _parse_methods(methods_raw["methods"], spec_dir)
    cells_raw = _load_yaml(spec_dir / "cells.yaml")
    _reject_unknown(cells_raw, {"cells", "variants"}, f"spec file {spec_dir / 'cells.yaml'}")
    cells = _parse_cells(cells_raw.get("cells") or {}, spec_dir)
    variants = _parse_variants(cells_raw.get("variants") or {}, spec_dir)

    for source, what in (
        (cells, "cells"),
        *[(v.overrides, f"variants.{n}.overrides") for n, v in variants.items()],
    ):
        for cell in source:
            ds, method = cell.split("/")
            if ds not in datasets:
                raise ValueError(f"{what}.{cell}: unknown dataset {ds!r}")
            if method not in methods:
                raise ValueError(f"{what}.{cell}: unknown method {method!r}")
    for ds, dspec in datasets.items():
        for method in dspec.prompts:
            if method not in methods:
                raise ValueError(f"datasets.{ds}.prompts.{method}: unknown method")

    benchmark: MatrixSpec | None = None
    bench_path = spec_dir / "benchmark.yaml"
    if bench_path.exists():
        benchmark = load_matrix_spec(bench_path)
        _validate_benchmark(benchmark, datasets, methods, spec_dir)
    return Spec(spec_dir, defaults, datasets, methods, cells, variants, benchmark)


def _validate_benchmark(
    benchmark: MatrixSpec,
    datasets: dict[str, DatasetSpec],
    methods: dict[str, MethodSpec],
    spec_dir: Path,
) -> None:
    if "dataset" not in benchmark.axes or "method" not in benchmark.axes:
        raise ValueError(f"benchmark spec {benchmark.path}: axes must declare dataset and method")
    if benchmark.base is not None:
        raise ValueError(
            f"benchmark spec {benchmark.path}: must not carry a base "
            "(cells are assembled from the config spec)"
        )
    for axis, values in benchmark.axes.items():
        for av in values:
            if axis == "dataset" and av.name not in datasets:
                raise ValueError(f"benchmark axis dataset={av.name!r}: unknown dataset")
            if axis == "method" and av.name not in methods:
                raise ValueError(f"benchmark axis method={av.name!r}: unknown method")
            if axis == "dataset" and av.overrides:
                raise ValueError(
                    f"benchmark axis dataset={av.name!r}: dataset bundles live in datasets.yaml (bench:)"
                )
    for i, combo in enumerate(benchmark.combos):
        if combo.base is not None:
            raise ValueError(
                f"benchmark combos[{i}]: base re-basing is retired — assembled cells need no file swap"
            )
        for axis, picks in combo.when.items():
            for pick in picks:
                if axis == "dataset" and pick not in datasets:
                    raise ValueError(f"benchmark combos[{i}]: unknown dataset {pick!r}")
                if axis == "method" and pick not in methods:
                    raise ValueError(f"benchmark combos[{i}]: unknown method {pick!r}")


def benchmark_axes_spec(spec: Spec) -> MatrixSpec:
    """The benchmark section as a MatrixSpec with the dataset axis's bench
    bundles injected from datasets.yaml — one MatrixSpec, one expansion engine
    (also what ``--filter`` parses against at the CLI boundary)."""
    benchmark = spec.benchmark
    if benchmark is None:
        raise ValueError(f"spec {spec.path}: no benchmark.yaml (sweep --spec has no cells)")
    axes: dict[str, tuple[AxisValue, ...]] = {}
    for axis, values in benchmark.axes.items():
        if axis == "dataset":
            axes[axis] = tuple(
                AxisValue(name=av.name, overrides=dict(spec.datasets[av.name].bench))
                for av in values
            )
        else:
            axes[axis] = values
    return MatrixSpec(
        path=benchmark.path,
        base=None,
        run_name=benchmark.run_name,
        set=dict(benchmark.set),
        axes=axes,
        combos=benchmark.combos,
    )


def cell_config(
    spec: Spec, dataset: str, method: str, variant: str | None = None
) -> dict[str, Any]:
    """The effective config of one quickstart cell, assembled layer by layer.
    Byte-for-byte the dict the committed per-cell YAML carried (golden parity
    is the contract; see ``tests/test_configspec.py``)."""
    if variant is not None and variant not in spec.variants:
        raise ValueError(f"unknown variant {variant!r} (known: {sorted(spec.variants)})")
    if method not in spec.methods:
        raise ValueError(f"unknown method {method!r} (spec {spec.path}: {sorted(spec.methods)})")
    if dataset not in spec.datasets:
        raise ValueError(f"unknown dataset {dataset!r} (spec {spec.path}: {sorted(spec.datasets)})")
    ds = spec.datasets[dataset]
    m = spec.methods[method]
    cell = f"{dataset}/{method}"

    config = copy.deepcopy(spec.defaults)
    config["task"] = copy.deepcopy(ds.task)

    apply_bundle(config, {"strategy": copy.deepcopy(m.strategy)})
    if m.budget is not None:
        apply_bundle(config, {"budget": copy.deepcopy(m.budget)})
    apply_bundle(config, m.task_patch.get(ds.kind, {}))
    apply_bundle(config, m.backend_patch)
    if ds.kind in m.judge:
        apply_bundle(config, {"judge": copy.deepcopy(m.judge[ds.kind])})

    apply_bundle(config, copy.deepcopy(ds.prompts.get(method, {})))

    override = spec.cells.get(cell)
    if override is not None:
        apply_bundle(config, copy.deepcopy(override.set))

    if variant is not None:
        v = spec.variants[variant]
        if cell not in v.available:
            available = ", ".join(v.available) or "(none)"
            raise ValueError(f"cell {cell} has no {variant!r} variant (available: {available})")
        config["backend"] = copy.deepcopy(v.backend)
        cell_variant = v.overrides.get(cell)
        if cell_variant is not None:
            apply_bundle(config, copy.deepcopy(cell_variant.set))

    run = config.setdefault("run", {})
    if not isinstance(run, dict):
        raise ValueError(f"defaults run section must be a mapping, got {type(run).__name__}")
    run["name"] = f"quickstart_{dataset}_{method}" + (f"_{variant}" if variant else "")
    return config


def benchmark_cells(
    spec: Spec, filters: dict[str, tuple[str, ...]] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """The published benchmark matrix, expanded from the spec — the same
    (run_name, config) cells the committed matrix + per-cell YAML wave produced
    (golden parity; ``--set`` overrides ride on top at the CLI boundary)."""
    return expand_matrix(
        benchmark_axes_spec(spec),
        filters,
        resolve_base=lambda values, _pattern: cell_config(
            spec, values["dataset"], values["method"]
        ),
    )


def spec_sha(spec: Spec) -> str:
    """Content hash of the committed spec files (name + bytes, sorted) — the
    provenance stamp the runtime records beside each assembled run config:
    spec + run dir is the reproducibility record."""
    h = hashlib.sha256()
    for path in sorted(spec.path.glob("*.yaml")):
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def cell_digest(config: Mapping[str, Any]) -> str:
    """Canonical sha256 digest of one assembled cell — the identity the
    golden manifests key on. ``sort_keys`` keeps dict insertion order out of
    the digest, ``ensure_ascii=False`` pins the escaping policy, and floats
    go through CPython ``repr`` (known-value test: ``tests/test_configspec.py``)."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: canonical section order of a rendered cell (reads like the historical
#: per-cell YAMLs did); unknown extras keep insertion order at the tail
_SECTION_ORDER = (
    "plugins",
    "run",
    "task",
    "backend",
    "protocol",
    "eval_model",
    "budget",
    "judge",
    "strategy",
)


def cell_notes(
    spec: Spec, dataset: str, method: str, variant: str | None = None
) -> list[tuple[str, str]]:
    """The fidelity notes touching one cell, in merge order: dataset, method,
    cell exception, variant (+ its per-cell exception). ``gen-configs`` renders
    these into the generated file header — the shipped wave's comment corpus,
    kept reviewable without the per-cell files."""
    if method not in spec.methods:
        raise ValueError(f"unknown method {method!r} (spec {spec.path}: {sorted(spec.methods)})")
    if dataset not in spec.datasets:
        raise ValueError(f"unknown dataset {dataset!r} (spec {spec.path}: {sorted(spec.datasets)})")
    cell = f"{dataset}/{method}"
    notes: list[tuple[str, str]] = [
        (f"dataset {dataset}", spec.datasets[dataset].why),
        (f"method {method}", spec.methods[method].why),
    ]
    override = spec.cells.get(cell)
    if override is not None and override.why:
        notes.append((f"cell {cell}", override.why))
    if variant is not None:
        v = spec.variants[variant]
        if v.why:
            notes.append((f"variant {variant}", v.why))
        cell_variant = v.overrides.get(cell)
        if cell_variant is not None and cell_variant.why:
            notes.append((f"variant {variant} cell {cell}", cell_variant.why))
    return [(label, why.strip()) for label, why in notes if why.strip()]


def render_cell_yaml(spec: Spec, dataset: str, method: str, variant: str | None = None) -> str:
    """One editable cell YAML (the ``eyas gen-configs`` product): GENERATED
    header + fidelity notes + the effective config, sections in canonical
    order. ``yaml.safe_load(render_cell_yaml(...))`` equals ``cell_config``."""
    config = cell_config(spec, dataset, method, variant)
    lines = [
        "# GENERATED from configs/spec — do not edit (eyas gen-configs"
        f" --cell {dataset}/{method}" + (f" --variant {variant}" if variant else "") + ")"
    ]
    for label, why in cell_notes(spec, dataset, method, variant):
        lines.append(f"# [{label}]")
        lines.extend(f"# {ln}" if ln else "#" for ln in why.splitlines())
    ordered = [k for k in _SECTION_ORDER if k in config] + [
        k for k in config if k not in _SECTION_ORDER
    ]
    body = {k: config[k] for k in ordered}
    lines.append(_yaml_cell_dump(body).rstrip("\n"))
    return "\n".join(lines) + "\n"


class _CellYamlDumper(yaml.SafeDumper):
    """Block scalars for multi-line strings (prompts read as text, not escapes)."""


def _represent_cell_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_CellYamlDumper.add_representer(str, _represent_cell_str)


def _yaml_cell_dump(data: dict[str, Any]) -> str:
    return yaml.dump(data, Dumper=_CellYamlDumper, sort_keys=False, allow_unicode=True, width=96)
