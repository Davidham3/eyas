"""§6 sampling tiers: deployment-layer thinking/seed are always preserved, tiers only carry algorithm tuning; no magic sampling in the optimizer."""

from __future__ import annotations

from test_contract import ToyMCQ, _gen_mock

from eyas.backends.base import SamplingConfig
from eyas.engine.evaluation import AbsoluteEvalModel
from eyas.engine.sampling import DEFAULT_TIERS, merge_tier, resolve_tier
from eyas.engine.strategies.ape import ApeConfig, ApeStrategy
from eyas.engine.strategy import OptimizerLoop, build_run_context
from eyas.tasks.protocol import IdentityProtocol


def test_merge_keeps_deployment_thinking_and_seed():
    """Tiers carry algorithm tuning; deployment-layer thinking/seed must be preserved (dropping them burns straight through the budget)."""
    deployment = SamplingConfig(temperature=0.0, max_new_tokens=256, thinking=True, seed=7)
    merged = merge_tier(deployment, DEFAULT_TIERS, "gradient")
    assert merged.temperature == 1.0 and merged.max_new_tokens == 512  # tier algorithm tuning
    assert merged.thinking is True and merged.seed == 7  # deployment tuning preserved


def test_resolve_tier_fail_fast_on_unknown():
    import pytest

    with pytest.raises(KeyError):
        resolve_tier(DEFAULT_TIERS, "nonexistent")


def test_cost_by_tier_and_profile_override():
    ctx = build_run_context(
        ApeStrategy,
        ApeConfig(num_candidates=3, demos_per_prompt=2),
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        sampling=SamplingConfig(seed=7, thinking=False),
        sampling_profiles={"generate": SamplingConfig(temperature=0.3, max_new_tokens=64)},
    )
    assert ctx.sampling_profiles["generate"].temperature == 0.3  # config overrides the tier
    assert (
        ctx.sampling_profiles["evolve"] == DEFAULT_TIERS["evolve"]
    )  # the rest fall back to defaults
    result = OptimizerLoop(ctx).run(ApeStrategy(ApeConfig(3, 2), ctx), ["seed"])
    assert (
        result.cost_by_tier.get("generate") == 3
    )  # 3 generated candidates, accounted under their tier
    assert (
        "eval" not in result.cost_by_tier
    )  # evaluation does not go through generate(), not counted under the generate tier
