"""Shipped-config validation under spec-as-source (zero GPU).

The per-cell YAMLs are retired: the shipped surface is configs/spec, and each
cell's effective config is the assembler's product (golden parity pinned by
tests/configs/test_configspec.py; the shared accessor tests/support/spec_cells.py states the
cell set). For every shipped run config — every spec cell (board x method grid
+ available mirrors) plus the standalone configs/tutorial_shapes.yaml — where
"shipped" means strategy section present, no optimizer section, validate:
required sections (load_config); task/judge/protocol names resolvable with
bindable params;
strategy name resolvable in the engine registry with params bound to a dataclass (unknown keys fail fast);
eval_model/budget names resolvable; strategy.seeds shape; and assembly-time capability matching
(strategy.requires_eval ⊆ eval_model semantics; the run's optimization objective keys must be
produced by the installed eval model -- compare semantics only produces compare_score, never enters the absolute
metric board; compare semantics needs >=2 seeds -- a single candidate degenerates to the 0.5 shortcut inside
CompareEvalModel, nothing left to optimize). The objective-key check is the assembly-time mirror of the
requires_eval check: a compare-scored config can pass the requires_eval check and still be rejected here.

No inference backend is constructed (needs a local engine / external server); real data coverage lives in the slow layer and real runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import eyas.builtin  # noqa: F401
import eyas.engine.builtin  # noqa: F401
from eyas import registry as base_registry
from eyas.engine import registry as engine_registry
from eyas.engine.capability import ABSOLUTE_EVAL, COMPARE_EVAL
from eyas.engine.runtime import load_config
from tests.support.spec_cells import cell, cell_text, shipped_cells
from tests.support.spec_cells import cell_id as make_cell_id

_ROOT = Path(__file__).resolve().parents[2]


def _bind(section: dict, group: str) -> tuple[type, object]:
    """Resolve a {name, params} section in the registry and construct it."""
    cls = base_registry.get(group, section["name"])
    return cls, cls(**(section.get("params") or {}))


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg if isinstance(cfg, dict) else {}


#: every shipped cell as (cell_id, effective-config dict): the assembler is
#: deterministic, so loading the grid once at collection is the same fact the
#: old file glob stated — just read from the spec instead of the files.
_CELL_PARAMS: list[tuple[str, dict[str, Any]]] = [
    (make_cell_id(ds, m, v), cell(ds, m, v)) for ds, m, v in shipped_cells()
]


def _cells_where(predicate) -> list[tuple[str, dict[str, Any]]]:
    """The parametrization shape the retired per-cell file globs had: only the
    matching cells are collected (no per-test skip noise)."""
    return [(cid, cfg) for cid, cfg in _CELL_PARAMS if predicate(cid, cfg)]


@pytest.mark.parametrize(
    "cell_id,cfg",
    _CELL_PARAMS + [("tutorial_shapes", _load(_ROOT / "configs" / "tutorial_shapes.yaml"))],
)
def test_run_config_resolves_and_binds(cell_id: str, cfg: dict, tmp_path: Path) -> None:
    # the strict loader runs on the serialized form (what gen-configs writes
    # and what a hand-carried file would carry), not on the in-memory dict
    path = tmp_path / f"{cell_id}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg = load_config(path)  # required-section validation for task/backend/strategy

    # task param binding (construction is lazy, no data files touched); same
    # {name, params} section convention the runtime builders use
    task_cls, _task = _bind(cfg["task"], "task")
    if cfg.get("judge"):
        _bind(cfg["judge"], "judge")
    _bind({"name": cfg.get("protocol", {}).get("name", "identity")}, "protocol")

    # strategy name resolvable + params bound to dataclass (unknown keys -> TypeError fail fast)
    strategy_cls = engine_registry.get("strategy", cfg["strategy"]["name"])
    strategy_config = strategy_cls.config_cls(**cfg["strategy"].get("params", {}))

    # eval_model / budget names resolvable
    eval_model_name = cfg.get("eval_model", {}).get("name", "absolute")
    engine_registry.get("eval_model", eval_model_name)
    engine_registry.get("budget", cfg.get("budget", {}).get("name", "flat"))

    # seeds shape
    seeds = cfg["strategy"].get("seeds") or []
    assert all(isinstance(s, str) and s for s in seeds), (
        f"{cell_id}: seeds must be a list of non-empty strings"
    )

    # assembly-time capability matching (mirrors build_run_context's validate_assembly criteria)
    semantics = (
        frozenset({ABSOLUTE_EVAL}) if eval_model_name == "absolute" else frozenset({COMPARE_EVAL})
    )
    requires = strategy_cls.capabilities.requires_eval
    assert requires <= semantics, (
        f"{cell_id}: strategy {cfg['strategy']['name']} requires {sorted(requires)}, "
        f"eval_model {eval_model_name} only provides {sorted(semantics)}"
    )
    if COMPARE_EVAL in requires:
        assert len(seeds) >= 2, (
            f"{cell_id}: compare semantics needs >=2 strategy.seeds (a single candidate degenerates to the 0.5 shortcut, nothing to optimize)"
        )

    # objective-key mirror: the run's optimization objective keys must be produced by the installed eval model
    # (absolute model builds produces from task metrics + parse_fail; compare only produces compare_score)
    objectives = getattr(task_cls, "objectives", ())
    produced = (
        frozenset({o.key for o in objectives} | {"parse_fail"})
        if eval_model_name == "absolute"
        else frozenset({"compare_score"})
    )
    optimize_keys = strategy_cls.capabilities.optimize_keys or (
        {objectives[0].key} if objectives else {"score"}
    )
    assert optimize_keys <= produced, (
        f"{cell_id}: strategy {cfg['strategy']['name']} cannot produce optimization objectives "
        f"{sorted(optimize_keys - produced)} (eval_model {eval_model_name} only produces {sorted(produced)})"
    )

    # resource intersection mirror: a vllm backend declares its
    # limit_mm_per_prompt image cap (unset = vllm's own default 1); the
    # strategy's declared image consumption must fit it — the same check
    # build_run_context enforces at startup (no backend constructed here).
    if cfg["backend"]["name"] == "vllm":
        engine_kwargs = cfg["backend"].get("params", {}).get("engine_kwargs") or {}
        backend_max_images = (engine_kwargs.get("limit_mm_per_prompt") or {}).get("image", 1)
        profile = strategy_cls.capabilities_for(strategy_config).resource_profile
        if profile.max_images is not None:
            assert profile.max_images <= backend_max_images, (
                f"{cell_id}: strategy {cfg['strategy']['name']} declares "
                f"{profile.max_images} images per request but the vllm engine accepts "
                f"{backend_max_images} (set backend.params.engine_kwargs.limit_mm_per_prompt)"
            )


# allocator -> the params keys a shipped config must name explicitly. The
# mapping follows the constructor signatures in eyas/engine/budget.py (the true cap
# fields), not intuition: FlatBudget is parameterless, so "explicit" for flat is
# an explicit empty params ({} pinned in the yaml, not the key omitted);
# EvenSplitBudget's only cap is total_sample_budget (a required argument — a
# bare name cannot even construct); BanditBudget demands all four constructor
# parameters (the pull caps every shipped cell must state; the table
# under-demands nothing). A bare ``name:``
# line with no params, and a config omitting the budget: section outright (the
# runtime's silent {"name": "flat"} default is exactly what this contract
# eliminates), are the shapes this scan rejects.
_ALLOCATOR_CAPS: dict[str, tuple[str, ...]] = {
    "flat": (),
    "even_split": ("total_sample_budget",),
    "bandit": ("eval_rounds", "samples_per_eval", "c", "keep_top"),
}


def _budget_pin_error(budget: object) -> str | None:
    """Return why this budget section violates explicit pinning, or None."""
    if not isinstance(budget, dict):
        return "missing budget: section"
    name = budget.get("name")
    if name not in _ALLOCATOR_CAPS:
        return f"budget.name {name!r} is not a shipped allocator {sorted(_ALLOCATOR_CAPS)}"
    if "params" not in budget:
        return "bare budget name with no params: allocator defaults stay implicit"
    params = budget["params"]
    if not isinstance(params, dict):
        return f"budget.params must be a mapping, got {params!r}"
    missing = [cap for cap in _ALLOCATOR_CAPS[name] if cap not in params]
    if missing:
        return f"budget.params omits caps {missing} for allocator {name!r}"
    return None


@pytest.mark.parametrize(
    "cell_id,cfg",
    _CELL_PARAMS + [("tutorial_shapes", _load(_ROOT / "configs" / "tutorial_shapes.yaml"))],
)
def test_shipped_configs_pin_the_allocator_and_its_caps(cell_id: str, cfg: dict) -> None:
    """Every shipped run config states its budget allocator AND the allocator's
    cap parameters explicitly — no run's effective budget may hinge on an
    implicit constructor default."""
    error = _budget_pin_error(cfg.get("budget"))
    assert error is None, f"{cell_id}: {error}"


def test_pinning_helper_rejects_bare_name_and_unknown_allocators() -> None:
    # the shape the scan exists to catch: a bare name line, params omitted
    assert _budget_pin_error({"name": "flat"}) is not None
    assert _budget_pin_error({"name": "bandit", "params": {"eval_rounds": 8}}) is not None
    assert _budget_pin_error({"name": "even_split", "params": {}}) is not None
    assert _budget_pin_error({"name": "made_up", "params": {}}) is not None
    # a config omitting the budget: section outright (cfg.get("budget") -> None)
    # must not slip through to the runtime's silent {"name": "flat"} default
    assert _budget_pin_error(None) is not None
    # raised bandit table: the former two-cap minimum is now under-demanding —
    # all four constructor parameters must be stated
    assert (
        _budget_pin_error({"name": "bandit", "params": {"samples_per_eval": 4, "keep_top": 0.5}})
        is not None
    )
    # explicit shapes pass
    assert _budget_pin_error({"name": "flat", "params": {}}) is None
    assert (
        _budget_pin_error(
            {
                "name": "bandit",
                "params": {"eval_rounds": 8, "samples_per_eval": 4, "c": 2.0, "keep_top": 0.5},
            }
        )
        is None
    )
    assert _budget_pin_error({"name": "even_split", "params": {"total_sample_budget": 16}}) is None


def _textvqa_cells() -> list[tuple[str, dict]]:
    return [(cid, cfg) for cid, cfg in _CELL_PARAMS if cid.startswith("textvqa_")]


@pytest.mark.parametrize("cell_id,cfg", _textvqa_cells())
def test_textvqa_column_contract(cell_id: str, cfg: dict) -> None:
    """The textvqa column joins the matrix as open-ended VQA: vqa task, judgeless
    (VQAv2 soft accuracy is rule-scored), identity protocol (variants are meaningless
    for vqa), and no {choices} marker in any prompt text -- VqaTask never substitutes
    it (the marker is MCQ/KIE-only; the pets-derived method sections must shed it)."""
    dataset, method, variant = _split_cell_id(cell_id)
    assert cfg["task"]["name"] == "vqa"
    assert "judge" not in cfg, f"{cell_id}: open-ended vqa must not carry a judge section"
    assert cfg.get("protocol", {}).get("name", "identity") == "identity"
    assert "{choices}" not in cell_text(dataset, method, variant)
    params = cfg["task"]["params"]
    assert params["data_dir"].endswith("textvqa/data")
    assert params["images_dir"].endswith("textvqa/images")


def _split_cell_id(cid: str) -> tuple[str, str, str | None]:
    """Inverse of _spec_cells.cell_id over the shipped grid (ids are unique:
    dataset and method names never contain '_openai')."""
    for ds, m, v in shipped_cells():
        if make_cell_id(ds, m, v) == cid:
            return ds, m, v
    raise KeyError(cid)


# The story configs (docs' flagship getting-started story: TextVQA raw vs OPRO on
# one OpenAI-compatible endpoint) plus the guided tour's public tutorial config.
# The run-config test above deliberately never constructs an inference backend,
# so nothing else pins a docs YAML backend key to the real constructor signature
# — a docs typo would only surface at the owner's fresh run. This is that pin.
@pytest.mark.parametrize(
    "target",
    [
        ("textvqa_raw_openai", None),
        ("textvqa_opro_openai", None),
        ("tutorial_shapes", _ROOT / "configs" / "tutorial_shapes.yaml"),
    ],
    ids=lambda t: t[0],
)
def test_story_openai_configs_bind_the_real_backend_signature(
    target: tuple[str, Path | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every docs story openai config's backend.params constructs OpenAIBackend
    as-is — the YAML keys must exist in the constructor signature (typos fail CI,
    not the owner's fresh run)."""
    from eyas.backends.openai_api import OpenAIBackend

    cid, path = target
    cfg = _load(path) if path is not None else cell(*_split_cell_id(cid))
    assert cfg["backend"]["name"] == "openai", f"{cid}: story config must use the openai backend"
    # the docs-run machinery (tests/docs/test_tutorials.py RUN_STUB) executes these
    # configs verbatim against the stub server — the URL literal is its contract
    assert cfg["backend"]["params"]["base_url"] == "http://localhost:8000/v1", (
        f"{cid}: docs-run configs must carry the canonical stub URL literal"
    )
    monkeypatch.setenv("EYAS_DOC_STUB_KEY", "dummy")  # api_key_env is validated on construction
    OpenAIBackend(**(cfg["backend"].get("params") or {}))


@pytest.mark.parametrize(
    "cell_id,cfg", _cells_where(lambda cid, cfg: cfg["strategy"]["name"] == "protegi")
)
def test_protegi_configs_run_the_bandit_selector(cell_id: str, cfg: dict) -> None:
    """ProTeGi's selection step IS bandit best-arm identification (§2.2.2); a
    flat-budget run is the paper's own "Unif" ablation, so every shipped
    protegi-strategy config declares ``budget: name: bandit`` with the pull
    caps pinned explicitly (changing one knob must never silently change
    results between runs)."""
    budget = cfg.get("budget") or {}
    assert budget.get("name") == "bandit", (
        f"{cell_id}: protegi must ship on the bandit selector, got budget={budget!r} "
        "(flat = the paper's Unif comparator, opt-in only)"
    )
    params = budget.get("params") or {}
    assert "samples_per_eval" in params, f"{cell_id}: samples_per_eval must be pinned"
    assert "keep_top" in params, f"{cell_id}: keep_top must be pinned"
    for key in ("samples_per_eval", "keep_top", "eval_rounds"):
        if key in params:
            assert isinstance(params[key], (int, float)), f"{cell_id}: {key} must be numeric"


# Quickstart bandit cells (config is law): the
# toy train splits hold 74 (pets) / 110 (cub) / 100 (sroie, textvqa, gsm8k,
# nextqa) / 28 (mixed) samples — facts derived from the shipped
# eyas.prepare recipe (the gsm8k/nextqa --*-train-n default is 100)
# and the mixed-modality smoke generator (4 samples per modality combo × 7 combos),
# restated here independently of the spec (same style as the bench table above).
# The official 8×32 fits none of them, so the configurator keeps eval_rounds=8
# (paper structure) and sizes samples_per_eval to floor(n/8): 9 / 13 / 12 / 12 / 12 / 12 / 3
# — every configured pull lands on a full window and the split tail (2/6/4/4/4/4/4)
# is simply never consumed (the configurator fixes the config; the runtime
# never shrinks a configured value). The quickstart quota is the cells/
# exception for <ds>/protegi; bench-scale quotas are the benchmark combos
# (bound in tests/configs/test_matrix_configs.py).
_QUICKSTART_BANDIT_PLAN: dict[str, tuple[int, int, int]] = {
    # dataset -> (split n, eval_rounds, samples_per_eval)
    "pets": (74, 8, 9),
    "cub": (110, 8, 13),
    "sroie": (100, 8, 12),
    "textvqa": (100, 8, 12),
    "gsm8k": (100, 8, 12),
    "nextqa": (100, 8, 12),
    "mixed": (28, 8, 3),
}


@pytest.mark.parametrize(
    "cell_id,cfg",
    _cells_where(
        lambda cid, cfg: (
            _split_cell_id(cid)[0] in _QUICKSTART_BANDIT_PLAN
            and cfg["strategy"]["name"] == "protegi"
        )
    ),
)
def test_quickstart_bandit_cells_pin_the_toy_split_exact_plan(cell_id: str, cfg: dict) -> None:
    """Each quickstart bandit cell must carry the configurator's exact floor(n/8)
    plan for its toy split:
    rounds stay 8 and rounds × spe ≤ n, so all 8 configured pulls serve real
    windows under the shipped contract."""
    dataset = _split_cell_id(cell_id)[0]
    split_n, rounds, spe = _QUICKSTART_BANDIT_PLAN[dataset]
    params = cfg["budget"]["params"]
    assert params["eval_rounds"] == rounds, (
        f"{cell_id}: eval_rounds must stay {rounds} (paper structure), got {params['eval_rounds']}"
    )
    assert params["samples_per_eval"] == spe, (
        f"{cell_id}: samples_per_eval must be {spe} = {split_n} // {rounds} "
        f"(quickstart toy-split fact), "
        f"got {params['samples_per_eval']}"
    )
    # feasibility under the shipped contract: no configured pull starts at/after the end
    assert (rounds - 1) * spe < split_n, (
        f"{cell_id}: {rounds}×{spe} arms window exhaustion on the {split_n}-sample split"
    )


# Evoprompt initial pools: the shipped GA cells run the official N_P=10 initial
# population — 10 curated, distinct seed prompts per dataset
# A cell without strategy.seeds would fall
# back to the single [task.default_prompt] and fail at the first GA round, so
# init guards the assembled pool. The pools are dataset-owned prompt assets in
# datasets.yaml; bench-scale cells carry the same asset through the benchmark
# expansion (bound in tests/configs/test_matrix_configs.py).


@pytest.mark.parametrize(
    "cell_id,cfg", _cells_where(lambda cid, cfg: cfg["strategy"]["name"] == "evoprompt")
)
def test_evoprompt_ship_cells_carry_the_official_np10_pool(cell_id: str, cfg: dict) -> None:
    """Every shipped evoprompt cell carries exactly 10
    pairwise-distinct (prompt_hash identity, NFKC+strip normalization) initial
    pool entries — the N_P=10 count the paper ships (``--popsize 10`` + 10
    initial instructions), config-visible and deterministic."""
    from eyas.engine.domain import prompt_hash

    seeds = cfg["strategy"].get("seeds") or []
    assert len(seeds) == 10, (
        f"{cell_id}: evoprompt ships the official N_P=10 initial pool, "
        f"got {len(seeds)} strategy.seeds"
    )
    assert len({prompt_hash(s) for s in seeds}) == 10, (
        f"{cell_id}: initial pool has prompts colliding under prompt_hash"
    )
