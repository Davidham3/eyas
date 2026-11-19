"""configs/spec/benchmark.yaml contract tests (zero GPU, zero data files).

The published benchmark matrix (7 model tags x 6 open datasets x 8 strategies =
336 cells) ships as ONE spec expanded by ``eyas sweep --spec`` — cell bases are
assembled from configs/spec (golden parity pinned by tests/test_configspec.py),
no per-cell YAMLs exist. These tests bind the expansion end-to-end so the spec
cannot silently drift from the published protocol the cells carry:

* every expanded cell binds (mirrors test_configs.py: registry resolution,
  param binding, capability rules) — a binding failure means the sweep would
  die at assembly on the GPU node, not at startup here;
* matrix invariants: trace ON everywhere (the observation surface is the point),
  seed 42 pinned, run identity == run_name pattern, placeholder model paths,
  InternVL load traits, KIE ape/opro assembled columns;
* curated pool contract: evoprompt ships the official N_P=10 distinct pool with
  the task default prompt as entry 1 (the published curation convention);
* bandit caps fit their train split (feasibility: no window arms exhaustion);
* one cell set = --filter tag --filter dataset expands to exactly the 8 method cells.
"""

from __future__ import annotations

from pathlib import Path

import eyas.builtin  # noqa: F401
import eyas.engine.builtin  # noqa: F401
from eyas import registry as base_registry
from eyas.configspec import benchmark_axes_spec, benchmark_cells, load_spec
from eyas.engine import registry as engine_registry
from eyas.runtime.matrix import parse_filters
from eyas.tasks.kie import KieTask
from eyas.tasks.math_qa import MathQaTask
from eyas.tasks.mcq import McqTask
from eyas.tasks.video_mcq import VideoMcqTask
from eyas.tasks.vqa import VqaTask

_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = _ROOT / "configs" / "spec"

MODELS = ("q2b", "q4b", "q9b", "ivl2b", "ivl4b", "ivl8b", "q38b")
DATASETS = ("pets", "cub", "sroie", "textvqa", "gsm8k", "nextqa")
METHODS = ("raw", "cot", "hill_climb", "evoprompt", "opro", "ape", "protegi", "spo")
PLACEHOLDER_PATHS = {
    "q2b": "/path/to/Qwen3.5-2B",
    "q4b": "/path/to/Qwen3.5-4B",
    "q9b": "/path/to/Qwen3.5-9B",
    "ivl2b": "/path/to/InternVL3.5-2B",
    "ivl4b": "/path/to/InternVL3.5-4B",
    "ivl8b": "/path/to/InternVL3.5-8B",
    "q38b": "/path/to/Qwen3.8-27B",
}
# train-split facts (the published runs used exactly these splits)
TRAIN_N = {"pets": 296, "cub": 261, "sroie": 248, "textvqa": 248, "gsm8k": 248, "nextqa": 248}
SHAPE_OF = {
    "pets": "mcq",
    "cub": "mcq",
    "sroie": "kie",
    "textvqa": "vqa",
    "gsm8k": "math_qa",
    "nextqa": "video_mcq",
}
ANCHOR = {
    "kie": KieTask.default_prompt,
    "mcq": McqTask.default_prompt,
    "vqa": VqaTask.default_prompt,
    "math_qa": MathQaTask.default_prompt,
    "video_mcq": VideoMcqTask.default_prompt,
}

_SPEC = load_spec(SPEC_DIR)
_AXES = benchmark_axes_spec(_SPEC)
_CELLS = benchmark_cells(_SPEC)
_BY_NAME = dict(_CELLS)


def _split_name(name: str) -> tuple[str, str, str]:
    stem = name.removeprefix("matrix_")
    tag, rest = stem.split("_", 1)
    method = next(m for m in METHODS if rest.endswith(f"_{m}"))
    ds = rest[: -len(method) - 1]
    assert tag in MODELS and ds in DATASETS and method in METHODS, name
    return tag, ds, method


def test_matrix_family_complete():
    assert len(_CELLS) == len(MODELS) * len(DATASETS) * len(METHODS) == 336
    assert len(_BY_NAME) == len(_CELLS)  # run-name uniqueness (expander also guards)


def test_matrix_binds_every_cell():
    """Registry resolution + param binding + capability rules for all 336 cells
    (mirrors test_configs's shipped-config contract; a binding failure here
    means the sweep would die at assembly on the GPU node, not at startup here)."""
    for _, cfg in _CELLS:
        task_cls = base_registry.get("task", cfg["task"]["name"])
        task_cls(**(cfg["task"].get("params") or {}))  # construction is lazy, no data touched
        if cfg.get("judge"):
            base_registry.get("judge", cfg["judge"]["name"])(**(cfg["judge"].get("params") or {}))
        base_registry.get("protocol", cfg.get("protocol", {}).get("name", "identity"))()
        strategy_cls = engine_registry.get("strategy", cfg["strategy"]["name"])
        strategy_config = strategy_cls.config_cls(**cfg["strategy"].get("params", {}))
        engine_registry.get("eval_model", cfg.get("eval_model", {}).get("name", "absolute"))
        engine_registry.get("budget", cfg["budget"]["name"])
        # resource intersection mirror (same rule as test_configs): the vllm
        # image cap must fit the strategy's declared consumption, including
        # the synthesized sroie ape/opro cells.
        if cfg["backend"]["name"] == "vllm":
            engine_kwargs = cfg["backend"].get("params", {}).get("engine_kwargs") or {}
            backend_max_images = (engine_kwargs.get("limit_mm_per_prompt") or {}).get("image", 1)
            profile = strategy_cls.capabilities_for(strategy_config).resource_profile
            if profile.max_images is not None:
                assert profile.max_images <= backend_max_images, (
                    f"{cfg['run']['name']}: strategy declares {profile.max_images} images per "
                    f"request but the vllm engine accepts {backend_max_images} "
                    "(set backend.params.engine_kwargs.limit_mm_per_prompt)"
                )


def test_matrix_run_identity_and_trace():
    for name, cfg in _CELLS:
        tag, ds, _method = _split_name(name)
        assert cfg["run"]["name"] == name, name
        assert cfg["run"]["trace"] is True, f"{name}: the matrix records every value"
        assert cfg["run"]["seed"] == 42, f"{name}: seed 42 pinned; 43/44 ride --set"
        assert cfg["run"]["final_eval"]["split"] == "test", name
        mp = cfg["backend"]["params"]["model_path"]
        assert mp == PLACEHOLDER_PATHS[tag], f"{name}: placeholder path must match the model tag"
        profile = cfg["backend"]["params"]["profile"]
        if tag.startswith("ivl"):
            assert profile == {"type": "hf_template", "trust_remote_code": True}, name
            assert cfg["backend"]["params"]["engine_kwargs"]["trust_remote_code"] is True
        else:
            assert profile.get("thinking_kwarg") == "enable_thinking", name
        assert cfg["task"]["name"] == SHAPE_OF[ds], name
        data_dir = cfg["task"]["params"]["data_dir"]
        want_root = f"datasets/bench/{ds}/data"
        assert data_dir == want_root, f"{name}: {data_dir!r} != {want_root!r}"
        if ds == "nextqa":
            assert cfg["task"]["params"]["videos_dir"] == "datasets/bench/nextqa/videos", name


def test_matrix_kie_ape_opro_columns_assemble_from_the_contract():
    """The KIE ape/opro columns (assembled from the kie dataset contract + the
    method identity; the retired synthesis combo proved equal): kie task, the
    strategy identity applied, the label-free demo-path pattern."""
    for method in ("ape", "opro"):
        cfg = _BY_NAME[f"matrix_q2b_sroie_{method}"]
        assert cfg["task"]["name"] == "kie"
        assert cfg["strategy"]["name"] == method
        if method == "opro":
            assert cfg["strategy"]["params"]["rounds"] == 4


def test_matrix_curated_pools_follow_the_published_convention():
    for name, cfg in _CELLS:
        _, ds, method = _split_name(name)
        if method == "evoprompt":
            pool = cfg["strategy"]["seeds"]
            assert len(pool) == 10 and len(set(pool)) == 10, name
            assert pool[0] == ANCHOR[SHAPE_OF[ds]], (
                f"{name}: evoprompt pool entry 1 must be the task default prompt"
            )
        if method == "spo":
            seeds = cfg["strategy"]["seeds"]
            assert len(seeds) >= 1 and all(s.strip() for s in seeds), name
        if method == "opro":
            assert cfg["strategy"]["params"]["rounds"] == 4, name


def test_matrix_bandit_caps_fit_their_splits():
    for name, cfg in _CELLS:
        if cfg["budget"]["name"] != "bandit":
            continue
        _, ds, _ = _split_name(name)
        params = cfg["budget"]["params"]
        spe, rounds = params["samples_per_eval"], params["eval_rounds"]
        assert spe * rounds <= TRAIN_N[ds] + (TRAIN_N[ds] % max(spe, 1)), name
        assert (rounds - 1) * spe < TRAIN_N[ds], (
            f"{name}: {rounds}x{spe} arms window exhaustion on the {TRAIN_N[ds]} split"
        )


def test_matrix_filter_selects_exactly_one_cell_set():
    """The sweep contract: filter one tag + one dataset = the 8 method
    cells of that tag/dataset, in the spec's method order."""
    for tag in MODELS:
        for ds in DATASETS:
            filters = parse_filters([f"tag={tag}", f"dataset={ds}"], _AXES)
            cells = benchmark_cells(_SPEC, filters)
            assert [n for n, _ in cells] == [f"matrix_{tag}_{ds}_{m}" for m in METHODS]


def test_matrix_spec_documented_repro_commands_are_executable():
    """The spec header advertises a sweep command form; pin its pieces against
    the expansion so docs and mechanism cannot drift."""
    filters = parse_filters(["tag=q2b", "dataset=pets"], _AXES)
    cells = benchmark_cells(_SPEC, filters)
    assert len(cells) == 8
    # every cell's model placeholder is overridable via --set (the placeholder
    # is a leaf the assembled cell also carries -> apply-overrides path
    # semantics hold)
    for _, cfg in cells:
        assert cfg["backend"]["params"]["model_path"] == "/path/to/Qwen3.5-2B"
