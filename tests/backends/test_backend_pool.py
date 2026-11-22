"""BackendPool / plan-mode contract tests (zero GPU, mock backend).

The engine-reuse feature: a matrix batch
pays the full vllm weight load (minutes on large models) ONCE per sweep/plan
instead of once per config. These tests pin the contract that makes sharing safe:

* identical backend blocks share one CORE instance; different blocks do not;
* per-run layers are never shared (each run gets its own NormalizingBackend /
  trace recorder — asserted via distinct evaluator.backend objects over the
  same core);
* close ownership: pooled cores are NOT closed between runs, exactly once at
  pool end; with share_backends=False each run builds and closes its own
  backend;
* plan semantics: runs write into their explicit run dirs, a failed entry is
  contained (later entries still run) and reported in outcomes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import eyas.builtin  # noqa: F401
import eyas.engine.builtin  # noqa: F401
from eyas import registry as base_registry
from eyas.backends.mock import ScriptedMockBackend
from eyas.backends.normalize import NormalizingBackend
from eyas.engine.runtime import run_plan, run_sweep

BUILDS: list[str] = []
CLOSES: list[str] = []
# shared build/close order — pins the eviction contract (close-before-build)
EVENTS: list[tuple[str, str]] = []


@base_registry.register_backend("counting_mock")
class CountingMock(ScriptedMockBackend):
    """ScriptedMockBackend + build/close tallies keyed by the responses param."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tag = str(kwargs.get("responses"))
        BUILDS.append(self.tag)
        EVENTS.append(("build", self.tag))

    def close(self) -> None:
        CLOSES.append(self.tag)
        EVENTS.append(("close", self.tag))
        super().close()


def make_cfg(name: str, responses: str) -> dict[str, Any]:
    return {
        "run": {"name": name, "seed": 42, "max_steps": 1},
        "task": {"name": "demo_shapes", "params": {"shapes": ["square"]}},
        "protocol": {"name": "identity"},
        "backend": {"name": "counting_mock", "params": {"responses": [responses], "cycle": True}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat", "params": {}},
        "strategy": {"name": "raw", "params": {}},
    }


@pytest.fixture(autouse=True)
def _tally():
    BUILDS.clear()
    CLOSES.clear()
    EVENTS.clear()
    yield
    BUILDS.clear()
    CLOSES.clear()
    EVENTS.clear()


def test_shared_sweep_builds_core_once_and_closes_once(tmp_path: Path):
    cfgs = [make_cfg("run_a", "red"), make_cfg("run_b", "red")]
    run_dirs = run_sweep(cfgs, tmp_path / "root", share_backends=True)
    assert len(run_dirs) == 2
    assert all((d / "result.json").exists() for d in run_dirs)
    assert BUILDS == ["['red']"]  # ONE core for both runs
    assert CLOSES == ["['red']"]  # released exactly once, at sweep end


def test_isolated_sweep_keeps_per_run_build_and_close(tmp_path: Path):
    cfgs = [make_cfg("run_a", "red"), make_cfg("run_b", "red")]
    run_sweep(cfgs, tmp_path / "root")  # share_backends defaults OFF
    assert BUILDS == ["['red']", "['red']"]  # isolated per-run assembly untouched
    assert CLOSES == ["['red']", "['red']"]


def test_plan_shares_across_entries_and_containes_failures(tmp_path: Path):
    e1 = (make_cfg("ok_one", "red"), tmp_path / "waveA" / "00-ok_one")
    e2 = ({"run": {"name": "broken"}}, tmp_path / "waveB" / "00-broken")  # fails assembly
    e3 = (make_cfg("ok_two", "red"), tmp_path / "waveC" / "00-ok_two")
    outcomes = run_plan([e1, e2, e3])

    assert outcomes[str(tmp_path / "waveA" / "00-ok_one")] is None
    assert outcomes[str(tmp_path / "waveB" / "00-broken")] is not None
    assert outcomes[str(tmp_path / "waveC" / "00-ok_two")] is None  # plan continued past e2
    assert (tmp_path / "waveC" / "00-ok_two" / "result.json").exists()
    assert len(BUILDS) == 1  # one shared core across the whole plan
    assert CLOSES == ["['red']"]  # and exactly one release at plan end


def test_plan_separates_cores_by_backend_block(tmp_path: Path):
    e1 = (make_cfg("r1", "red"), tmp_path / "1")
    e2 = (make_cfg("r2", "blue"), tmp_path / "2")
    outcomes = run_plan([e1, e2])
    assert all(v is None for v in outcomes.values())
    assert sorted(BUILDS) == ["['blue']", "['red']"]  # different blocks -> distinct engines
    assert sorted(CLOSES) == ["['blue']", "['red']"]


def test_pool_shares_core_but_keeps_per_run_layers(tmp_path: Path):
    """The shared core must stay UNDER per-run NormalizingBackend wrappers —
    trace/normalize/planner semantics are per-run even when the engine is not."""
    from eyas.engine.runtime import build_context_from_config

    cfg = make_cfg("layered", "red")
    ctx1 = build_context_from_config(cfg, run_dir=tmp_path / "r1")
    ctx2 = build_context_from_config(cfg, run_dir=tmp_path / "r2")
    try:
        assert ctx1.evaluator.backend is not ctx2.evaluator.backend  # per-run wrappers
        assert isinstance(ctx1.evaluator.backend, NormalizingBackend)
    finally:
        ctx1.evaluator.backend.close()
        ctx2.evaluator.backend.close()


def test_sweep_evicts_resident_core_before_building_a_different_one(tmp_path: Path):
    """Single-residency contract: a core engine occupies the GPU, so the pool
    must release the previous core BEFORE building a different block —
    otherwise the second engine loads while the first is still resident and
    the memory precheck (correctly) refuses it. A,B,A therefore builds three
    cores with each close strictly preceding the next build; the identical-key
    sharing win is untouched."""
    cfgs = [make_cfg("r1", "red"), make_cfg("r2", "blue"), make_cfg("r3", "red")]
    run_dirs = run_sweep(cfgs, tmp_path / "root", share_backends=True)
    assert len(run_dirs) == 3
    assert all((d / "result.json").exists() for d in run_dirs)
    assert EVENTS == [
        ("build", "['red']"),
        ("close", "['red']"),
        ("build", "['blue']"),
        ("close", "['blue']"),
        ("build", "['red']"),
        ("close", "['red']"),
    ]


def test_sweep_identical_block_never_evicts_mid_sweep(tmp_path: Path):
    """Guard against over-evicting: identical backend blocks must still share
    exactly one core for the whole sweep (the whole point of pooling)."""
    cfgs = [make_cfg("r1", "red"), make_cfg("r2", "red")]
    run_sweep(cfgs, tmp_path / "root", share_backends=True)
    assert EVENTS == [("build", "['red']"), ("close", "['red']")]
