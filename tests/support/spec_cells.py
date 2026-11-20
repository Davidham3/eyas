"""Shared accessor for the shipped cell surface under spec-as-source.

The committed per-cell YAMLs are retired (E2): the cell set is the config
spec's cross product, and a cell's effective config is ``cell_config`` — the
same assembly the golden-parity suite (tests/configs/test_configspec.py) pins. Tests
that used to read ``configs/quickstart_<cell>.yaml`` load through here, so
"what ships" is stated once.
"""

from __future__ import annotations

from pathlib import Path

from eyas.configspec import Spec, cell_config, load_spec, render_cell_yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "src" / "eyas" / "spec"

_SPEC: Spec | None = None


def spec() -> Spec:
    global _SPEC
    if _SPEC is None:
        _SPEC = load_spec(SPEC_DIR)
    return _SPEC


def shipped_cells() -> list[tuple[str, str, str | None]]:
    """Every shipped cell as (dataset, method, variant): the full board x
    method grid (the assembly contract, including the sroie ape/opro combos
    the historical file wave synthesized through the matrix) plus the variant
    mirrors the spec declares available."""
    s = spec()
    cells = [(dataset, method, None) for dataset in s.datasets for method in s.methods]
    for name, variant in sorted(s.variants.items()):
        cells += [(*cell.split("/"), name) for cell in variant.available]
    return cells


def cell_id(dataset: str, method: str, variant: str | None) -> str:
    return f"{dataset}_{method}" + (f"_{variant}" if variant else "")


def cell(dataset: str, method: str, variant: str | None = None) -> dict:
    """The cell's effective config (dict, fresh copy per call)."""
    return cell_config(spec(), dataset, method, variant)


def cell_text(dataset: str, method: str, variant: str | None = None) -> str:
    """The cell as generated YAML text — the surface text-level assertions
    (prompt markers, prose scans) used to run against the file contents."""
    return render_cell_yaml(spec(), dataset, method, variant)
