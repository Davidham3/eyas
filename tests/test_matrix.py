"""Matrix spec contract tests (zero GPU): parsing, filters, expansion semantics.

The matrix expander (``eyas sweep --matrix``) is the batch-experiment mechanism
that replaces families of per-cell config files. Pinned contracts:

* spec validation is fail fast — unknown keys, missing sections, combos
  referencing unknown axes/values all raise at load;
* override semantics differ from ``--set`` on purpose (see matrix.py docstring):
  intermediate mappings are created, mapping values REPLACE the leaf, ``null``
  deletes it;
* application order is base -> global set -> axis bundles (declaration order)
  -> combo sets, and combos can re-base (``base``), add sets, or drop (``skip``)
  matching cells;
* run names are unique over an expansion (collisions fail, they do not
  overwrite silently); filters validate axis/value membership and intersect on
  repeated axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from eyas.runtime.matrix import describe_expansion, expand_matrix, load_matrix_spec, parse_filters

BASE_A = "run:\n  name: placeholder\n  seed: 42\nstrategy:\n  name: raw\n  params: {}\nprofile_slot:\n  type: hf_template\n  thinking_kwarg: enable_thinking\n"
BASE_B = "run:\n  name: placeholder\nstrategy:\n  name: raw\n  params: {}\n"


@pytest.fixture()
def spec_dir(tmp_path: Path) -> Path:
    (tmp_path / "base_a.yaml").write_text(BASE_A, encoding="utf-8")
    (tmp_path / "base_b.yaml").write_text(BASE_B, encoding="utf-8")
    return tmp_path


def _write_spec(dir_: Path, body: dict[str, Any]) -> Path:
    path = dir_ / "spec.yaml"
    path.write_text(yaml.dump(body, sort_keys=False), encoding="utf-8")
    return path


def _spec(dir_: Path, **over: Any):
    body: dict[str, Any] = {
        "base": str(dir_ / "base_a.yaml"),
        "run_name": "cell_{m}",
        "axes": {"m": ["x", "y"]},
    }
    body.update(over)
    return load_matrix_spec(_write_spec(dir_, body))


def _by_name(cells: list[tuple[str, dict]]) -> dict[str, dict]:
    return dict(cells)


# ---------------------------------------------------------------- version 1: loading


def test_load_rejects_unknown_top_level_key(spec_dir: Path) -> None:
    path = _write_spec(spec_dir, {"base": "b", "run_name": "n", "axes": {"m": ["x"]}, "axyes": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        load_matrix_spec(path)


def test_load_requires_base_run_name_axes(spec_dir: Path) -> None:
    for missing in ("base", "run_name", "axes"):
        body = {"base": "b", "run_name": "n", "axes": {"m": ["x"]}}
        del body[missing]
        path = _write_spec(spec_dir, body)
        with pytest.raises(ValueError, match=f"missing required key {missing!r}"):
            load_matrix_spec(path)


def test_load_rejects_empty_axis_and_bad_bundle_keys(spec_dir: Path) -> None:
    path = _write_spec(spec_dir, {"base": "b", "run_name": "n", "axes": {"m": []}})
    with pytest.raises(ValueError, match="at least one value"):
        load_matrix_spec(path)
    path = _write_spec(
        spec_dir, {"base": "b", "run_name": "n", "axes": {"m": {"x": {1: "int key"}}}}
    )
    with pytest.raises(ValueError, match="non-empty strings"):
        load_matrix_spec(path)


def test_combo_validation(spec_dir: Path) -> None:
    base = {"base": "b", "run_name": "n", "axes": {"m": ["x"]}}
    cases = [
        ({"combos": [{"when": {"m": "x"}, "nope": 1}]}, "unknown keys"),
        ({"combos": [{"when": {}}]}, "at least one axis"),
        ({"combos": [{"when": {"noaxis": "x"}}]}, "unknown axis"),
        ({"combos": [{"when": {"m": "novalue"}}]}, "unknown value"),
        ({"combos": [{"when": {"m": "x"}, "skip": "yes"}]}, "must be a bool"),
    ]
    for over, match in cases:
        path = _write_spec(spec_dir, {**base, **over})
        with pytest.raises(ValueError, match=match):
            load_matrix_spec(path)


# ---------------------------------------------------------------- version 2: expansion


def test_product_count_order_and_run_name(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {
                "base": str(spec_dir / "base_a.yaml"),
                "run_name": "cell_{t}_{m}",
                "axes": {"m": ["x", "y"], "t": ["p", "q"]},
            },
        )
    )
    cells = expand_matrix(spec)
    # axis-declaration product order: the LAST axis varies fastest
    assert [n for n, _ in cells] == ["cell_p_x", "cell_q_x", "cell_p_y", "cell_q_y"]


def test_run_name_collision_fails(spec_dir: Path) -> None:
    spec = _spec(spec_dir, axes={"m": ["x", "y"], "t": ["p", "q"]})
    with pytest.raises(ValueError, match="run_name collision"):
        expand_matrix(spec)


def test_apply_order_base_global_axes_combos(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {
                "base": str(spec_dir / "base_a.yaml"),
                "run_name": "cell_{m}",
                "set": {"run.seed": 7, "strategy.name": "global"},
                "axes": {
                    "m": {
                        "x": {"strategy.name": "axis"},
                        "y": {},
                    }
                },
                "combos": [{"when": {"m": "x"}, "set": {"strategy.name": "combo"}}],
            },
        )
    )
    cells = _by_name(expand_matrix(spec))
    assert cells["cell_x"]["strategy"]["name"] == "combo"  # combo set last
    assert (
        cells["cell_y"]["strategy"]["name"] == "global"
    )  # global set under axes (x's axis lost to combo)
    assert cells["cell_x"]["run"]["seed"] == 7  # global set applied to every cell
    assert cells["cell_x"]["run"]["name"] == "cell_x"  # run.name stamped last


def test_mapping_value_replaces_and_null_deletes(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {
                "base": str(spec_dir / "base_a.yaml"),
                "run_name": "cell",
                "axes": {
                    "m": {
                        "x": {
                            "profile_slot": {"type": "other"},
                            "run.final_eval": None,
                            "brand.new.key": 1,
                        }
                    }
                },
            },
        )
    )
    (cfg,) = [c for _, c in expand_matrix(spec)]
    # dict-valued leaf REPLACES: the Qwen-style thinking_kwarg does not survive
    assert cfg["profile_slot"] == {"type": "other"}
    assert "final_eval" not in cfg["run"]  # null deletes (idempotent when absent)
    assert cfg["brand"] == {"new": {"key": 1}}  # intermediate mappings created


def test_descend_through_scalar_fails(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {
                "base": str(spec_dir / "base_a.yaml"),
                "run_name": "cell",
                "axes": {"m": {"x": {"run.seed.boom": 1}}},
            },
        )
    )
    with pytest.raises(ValueError, match="not a mapping, cannot descend"):
        expand_matrix(spec)


def test_combo_rebase_and_skip(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {
                "base": str(spec_dir / "base_a.yaml"),
                "run_name": "cell_{m}",
                "axes": {"m": ["x", "y", "z"]},
                "combos": [
                    {"when": {"m": "y"}, "base": str(spec_dir / "base_b.yaml")},
                    {"when": {"m": "z"}, "skip": True},
                ],
            },
        )
    )
    cells = _by_name(expand_matrix(spec))
    assert set(cells) == {"cell_x", "cell_y"}
    assert "profile_slot" in cells["cell_x"]  # original base
    assert "profile_slot" not in cells["cell_y"]  # re-based onto base_b


def test_missing_base_file_fails(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {"base": str(spec_dir / "{m}.yaml"), "run_name": "cell_{m}", "axes": {"m": ["x"]}},
        )
    )
    with pytest.raises(ValueError, match="base config not found"):
        expand_matrix(spec)


def test_unknown_placeholder_fails(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {
                "base": str(spec_dir / "base_a.yaml"),
                "run_name": "cell_{nope}",
                "axes": {"m": ["x"]},
            },
        )
    )
    with pytest.raises(ValueError, match="references unknown axis"):
        expand_matrix(spec)


# ---------------------------------------------------------------- version 3: filters


def test_filters_membership_comma_and_intersect(spec_dir: Path) -> None:
    spec = load_matrix_spec(
        _write_spec(
            spec_dir,
            {
                "base": str(spec_dir / "base_a.yaml"),
                "run_name": "t{t}m_{m}",
                "axes": {"m": ["x", "y"], "t": ["p", "q"]},
            },
        )
    )
    filters = parse_filters(["m=x,y", "t=p"], spec)
    assert filters == {"m": ("x", "y"), "t": ("p",)}
    assert len(expand_matrix(spec, filters)) == 2
    # repeated axis intersects
    filters = parse_filters(["m=x,y", "m=y"], spec)
    assert filters["m"] == ("y",)
    cells = [n for n, _ in expand_matrix(spec, filters)]
    assert cells == ["tpm_y", "tqm_y"]


def test_filters_reject_unknown_axis_and_value(spec_dir: Path) -> None:
    spec = _spec(spec_dir)
    with pytest.raises(ValueError, match="unknown axis"):
        parse_filters(["nope=x"], spec)
    with pytest.raises(ValueError, match="not a value of axis"):
        parse_filters(["m=zzz"], spec)
    with pytest.raises(ValueError, match="axis=value form"):
        parse_filters(["justtext"], spec)
    with pytest.raises(ValueError, match="filtered to empty"):
        parse_filters(["m=x", "m=y"], spec)


def test_describe_expansion_lists_facets(spec_dir: Path) -> None:
    spec = _spec(spec_dir)  # single axis m: x|y
    text = describe_expansion(spec, parse_filters(["m=x"], spec), 1)
    assert "1 cells" in text and "(m=x)" in text
