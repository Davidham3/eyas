"""Matrix spec: facet expansion for batch experiments (``eyas sweep --matrix``).

A matrix spec replaces a family of per-cell config files with one declarative
file: a shared ``base`` template (a path pattern over the axis values), a
``run_name`` pattern, and ``axes`` whose values are **override dicts** — the
same dotted-path mental model as ``--set``, so a cell's full recipe stays
visible in the spec. Expansion is a pure function of (spec, filters):

    base file -> global ``set`` -> axis bundles (declaration order) -> combo sets

The per-cell effective config is persisted by the runtime itself (the run
directory's ``run_config.json``), so committed per-cell YAMLs are not the
reproducibility anchor — the spec plus the run directory is.

Override semantics here deliberately differ from :func:`eyas.runtime.config.apply_overrides`:
``--set`` edits one concrete config and protects typos by requiring paths to
pre-exist; a matrix bundle is a *recipe fragment authored against a template*,
so intermediate mappings are created on descent, a mapping value REPLACES the
target leaf (dict-valued leaves are the explicit-replacement tool — e.g.
swapping a whole ``profile``), and a ``null`` value deletes the key. The
spec's binding test (``tests/configs/test_matrix_configs.py``: expansion shape +
equivalence of the published cells) is this freedom's counterpart guard.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AxisValue",
    "MatrixSpec",
    "apply_bundle",
    "default_base_resolver",
    "load_matrix_spec",
    "parse_filters",
    "expand_matrix",
]


@dataclass(frozen=True)
class AxisValue:
    """One value on one axis: its name (feeds the patterns) and its override bundle."""

    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class Combo:
    """Targeted intervention on matching cells: re-base them (``base``), add
    override sets (``set``), or drop them (``skip``)."""

    when: dict[str, tuple[str, ...]]
    base: str | None
    set: dict[str, Any]
    skip: bool


@dataclass(frozen=True)
class MatrixSpec:
    """A parsed, validated matrix spec. ``axes`` keeps declaration order; the
    expansion product follows it (so cells sharing the leading axes stay
    adjacent — the shape backend sharing exploits). ``base`` is the per-cell
    file pattern; specs whose cells come from an injected resolver (the config
    spec's benchmark section) carry no base at all."""

    path: Path
    base: str | None
    run_name: str
    set: dict[str, Any]
    axes: dict[str, tuple[AxisValue, ...]]
    combos: tuple[Combo, ...]


def _require_mapping(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _parse_bundle(raw: Mapping[str, Any], what: str) -> dict[str, Any]:
    """A bundle is a mapping of dotted-path keys to YAML-typed values; empty
    values are normalised to ``{}`` only at leaf assignment (``None`` is the
    delete marker, so it must survive parsing untouched)."""
    bundle: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{what}: override keys must be non-empty strings, got {key!r}")
        bundle[key] = value
    return bundle


def _parse_axis(axis: str, raw: object) -> tuple[AxisValue, ...]:
    """Two axis forms: a list of plain names (empty bundles) or a mapping
    name -> override bundle (declaration order preserved)."""
    if isinstance(raw, list):
        values = []
        for item in raw:
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"axes.{axis}: list values must be non-empty strings, got {item!r}"
                )
            values.append(AxisValue(name=item, overrides={}))
        if not values:
            raise ValueError(f"axes.{axis}: list form needs at least one value")
        return tuple(values)
    if isinstance(raw, dict):
        values = []
        for name, bundle_raw in raw.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"axes.{axis}: value names must be non-empty strings, got {name!r}"
                )
            if bundle_raw is None:
                bundle: dict[str, Any] = {}
            else:
                bundle = _parse_bundle(
                    _require_mapping(bundle_raw, f"axes.{axis}.{name}"), f"axes.{axis}.{name}"
                )
            values.append(AxisValue(name=name, overrides=bundle))
        if not values:
            raise ValueError(f"axes.{axis}: mapping form needs at least one value")
        return tuple(values)
    raise ValueError(f"axes.{axis}: must be a list of names or a name->bundle mapping")


def load_matrix_spec(path: str | Path) -> MatrixSpec:
    """Load and validate a matrix spec YAML. Fail fast on any structural
    problem: bad sections, unknown combo keys, combos referencing unknown axes
    or values — all caught before a single cell runs."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require_mapping(raw, f"matrix spec {path}")
    known = {"base", "run_name", "set", "axes", "combos"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"matrix spec {path}: unknown keys {sorted(unknown)} (known: {sorted(known)})"
        )
    for key in ("run_name", "axes"):
        if key not in raw:
            raise ValueError(f"matrix spec {path}: missing required key {key!r}")
    base = raw.get("base")
    if base is not None and (not isinstance(base, str) or not base):
        raise ValueError(f"matrix spec {path}: base must be a non-empty path pattern")
    run_name = raw["run_name"]
    if not isinstance(run_name, str) or not run_name:
        raise ValueError(f"matrix spec {path}: run_name must be a non-empty pattern")
    set_raw = raw.get("set") or {}
    global_set = _parse_bundle(_require_mapping(set_raw, f"matrix spec {path}: set"), "set")
    axes_raw = _require_mapping(raw["axes"], f"matrix spec {path}: axes")
    axes: dict[str, tuple[AxisValue, ...]] = {}
    for axis, axis_raw in axes_raw.items():
        if not isinstance(axis, str) or not axis or axis in {"base", "set"}:
            raise ValueError(f"matrix spec {path}: bad axis name {axis!r}")
        axes[axis] = _parse_axis(axis, axis_raw)

    combos: list[Combo] = []
    for i, combo_raw in enumerate(raw.get("combos") or []):
        cm = _require_mapping(combo_raw, f"matrix spec {path}: combos[{i}]")
        unknown = set(cm) - {"when", "base", "set", "skip"}
        if unknown:
            raise ValueError(f"matrix spec {path}: combos[{i}] unknown keys {sorted(unknown)}")
        when_raw = _require_mapping(cm.get("when"), f"matrix spec {path}: combos[{i}].when")
        if not when_raw:
            raise ValueError(
                f"matrix spec {path}: combos[{i}].when must list at least one axis condition"
            )
        when: dict[str, tuple[str, ...]] = {}
        for axis, values in when_raw.items():
            if axis not in axes:
                raise ValueError(
                    f"matrix spec {path}: combos[{i}].when references unknown axis {axis!r}"
                )
            names = (values,) if isinstance(values, str) else values
            if not isinstance(names, (list, tuple)) or not names:
                raise ValueError(
                    f"matrix spec {path}: combos[{i}].when.{axis} must be a value or list of values"
                )
            valid = {v.name for v in axes[axis]}
            for name in names:
                if not isinstance(name, str) or name not in valid:
                    raise ValueError(
                        f"matrix spec {path}: combos[{i}].when.{axis} references unknown value {name!r} (valid: {sorted(valid)})"
                    )
            when[axis] = tuple(names)
        combo_base = cm.get("base")
        if combo_base is not None and (not isinstance(combo_base, str) or not combo_base):
            raise ValueError(
                f"matrix spec {path}: combos[{i}].base must be a non-empty path pattern"
            )
        skip = cm.get("skip", False)
        if not isinstance(skip, bool):
            raise ValueError(f"matrix spec {path}: combos[{i}].skip must be a bool")
        combo_set = cm.get("set") or {}
        combos.append(
            Combo(
                when=when,
                base=combo_base,
                set=_parse_bundle(
                    _require_mapping(combo_set, f"matrix spec {path}: combos[{i}].set"),
                    f"combos[{i}].set",
                ),
                skip=skip,
            )
        )
    return MatrixSpec(
        path=path, base=base, run_name=run_name, set=global_set, axes=axes, combos=tuple(combos)
    )


def parse_filters(items: list[str], spec: MatrixSpec) -> dict[str, tuple[str, ...]]:
    """Parse ``axis=value[,value...]`` filter strings against the spec: every
    axis must exist, every value must be one of the axis's values. Repeated
    filters on one axis intersect."""
    filters: dict[str, tuple[str, ...]] = {}
    for item in items:
        key, sep, raw_value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--filter must be in axis=value form: {item!r}")
        if key not in spec.axes:
            raise ValueError(f"--filter unknown axis {key!r} (axes: {list(spec.axes)})")
        valid = {v.name for v in spec.axes[key]}
        names = tuple(v for v in raw_value.split(",") if v)
        if not names:
            raise ValueError(f"--filter {key}= needs at least one value: {item!r}")
        for name in names:
            if name not in valid:
                raise ValueError(
                    f"--filter {key}={name!r} is not a value of axis {key!r} (valid: {sorted(valid)})"
                )
        if key in filters:
            merged = tuple(v for v in filters[key] if v in names)
            if not merged:
                raise ValueError(
                    f"--filter axis {key!r} filtered to empty set (repeated filters must share a value)"
                )
            filters[key] = merged
        else:
            filters[key] = names
    return filters


def _set_path(config: dict[str, Any], key: str, value: Any) -> None:
    """Matrix override semantics: descend creating mappings, then assign the
    leaf (a mapping value replaces), with ``None`` deleting the leaf."""
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            node[part] = {}
        elif not isinstance(child, dict):
            raise ValueError(f"matrix override {key!r}: '{part}' is not a mapping, cannot descend")
        node = node[part]
    if value is None:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = value


def apply_bundle(config: dict[str, Any], bundle: Mapping[str, Any]) -> None:
    """Apply one override bundle (dotted-path keys, mapping values replace,
    ``None`` deletes) in place — shared with the config-spec assembler so the
    repo has exactly one bundle-merge implementation."""
    for key, value in bundle.items():
        _set_path(config, key, value)


def _format(pattern: str, values: Mapping[str, str], what: str) -> str:
    try:
        return pattern.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"matrix {what} pattern {pattern!r} references unknown axis {exc}"
        ) from exc


#: A cell resolver turns one cell's axis values (plus the effective base
#: pattern, combo re-basing included) into the cell's config dict. The default
#: reads base YAML files; the config-spec assembler injects layered assembly.
BaseResolver = Callable[[Mapping[str, str], str], dict[str, Any]]


def default_base_resolver(
    spec: MatrixSpec, load: Callable[[Path], dict[str, Any]] | None = None
) -> BaseResolver:
    """The file-based resolver: resolve the base pattern against the cell
    values, require the file, and load it (``load`` is the CLI's section-
    validating loader; the default is a bare ``yaml.safe_load``)."""
    if not spec.base:
        raise ValueError(f"matrix spec {spec.path}: no base pattern (cells need a resolver)")

    def resolve(values: Mapping[str, str], pattern: str) -> dict[str, Any]:
        base_path = Path(_format(pattern, values, "base"))
        if not base_path.exists():
            raise ValueError(f"matrix cell {dict(values)}: base config not found: {base_path}")
        if load is not None:
            return load(base_path)
        return _require_mapping(
            yaml.safe_load(base_path.read_text(encoding="utf-8")), f"base {base_path}"
        )

    return resolve


def expand_matrix(
    spec: MatrixSpec,
    filters: Mapping[str, tuple[str, ...]] | None = None,
    *,
    resolve_base: BaseResolver | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Expand the spec into ``(run_name, config)`` cells in axis-declaration
    product order. ``resolve_base(cell_values, base_pattern)`` produces a cell
    base config — by default the spec's base file pattern, resolved relative to
    the CWD (the repo-root launch convention shared with ``plugins``); a config
    spec injects its layered assembler instead."""
    if resolve_base is None:
        resolve_base = default_base_resolver(spec)
    filters = dict(filters or {})
    for axis in filters:
        if axis not in spec.axes:
            raise ValueError(f"filter references unknown axis {axis!r} (axes: {list(spec.axes)})")

    value_lists: list[list[AxisValue]] = []
    axis_names: list[str] = []
    for axis, axis_values in spec.axes.items():
        keep = filters.get(axis)
        value_lists.append([v for v in axis_values if keep is None or v.name in keep])
        axis_names.append(axis)

    cells: list[tuple[str, dict[str, Any]]] = []
    names_seen: dict[str, str] = {}
    for combo_values in itertools.product(*value_lists):
        cell_values = {axis: av.name for axis, av in zip(axis_names, combo_values, strict=True)}
        matching = [
            c
            for c in spec.combos
            if all(cell_values[axis] in picks for axis, picks in c.when.items())
        ]
        if any(c.skip for c in matching):
            continue
        base_pattern = spec.base or ""
        for c in matching:
            if c.base is not None:
                base_pattern = c.base
        config = resolve_base(cell_values, base_pattern)
        apply_bundle(config, spec.set)
        for av in combo_values:
            apply_bundle(config, av.overrides)
        for c in matching:
            apply_bundle(config, c.set)
        run_name = _format(spec.run_name, cell_values, "run_name")
        _set_path(config, "run.name", run_name)
        if run_name in names_seen:
            raise ValueError(
                f"matrix run_name collision: {run_name!r} from both {names_seen[run_name]} and {cell_values}"
                " — include the distinguishing axes in run_name"
            )
        names_seen[run_name] = str(cell_values)
        cells.append((run_name, config))
    return cells


def describe_expansion(
    spec: MatrixSpec, filters: Mapping[str, tuple[str, ...]], n_cells: int
) -> str:
    """One-line summary of what an expansion covers (CLI echo, not a promise:
    the cell list itself is what runs)."""
    facets = ", ".join(
        f"{axis}={'|'.join(filters.get(axis, tuple(v.name for v in values)))}"
        for axis, values in spec.axes.items()
    )
    return f"matrix {spec.path.name}: {n_cells} cells ({facets})"
