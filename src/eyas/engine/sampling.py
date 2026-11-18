"""Sampling tiers (§6): methods declare which tier name to use; at assembly
time it is merged with the deployment baseline ctx.sampling.

Methods declare tier names, never ``SamplingConfig(...)`` literals: constructing
one from scratch drops the deployment layer's thinking/seed, and on
openai-compatible endpoints thinking=None falls back to the server default
(enabled), burning through max_new_tokens (see the note on
backends.base.merge_sampling). The optimizer only writes tier names; the numbers
are the algorithm's convention, thinking/seed always come from the deployment
layer, and cost is accounted per tier.
"""

from __future__ import annotations

from eyas.backends.base import SamplingConfig, merge_sampling

__all__ = ["DEFAULT_TIERS", "resolve_tier"]

# Default tiers by algorithm convention (temperature/top_p/max_new_tokens);
# for candidate generation and evaluation rendering the deployment layer owns
# thinking/seed. NOTE: there is deliberately no "eval" tier —
# evaluation rendering rides the evaluator's deployment sampling (the loop's
# traffic, not a generation tier); a sampling_profiles.eval entry would be a
# silent no-op.
DEFAULT_TIERS: dict[str, SamplingConfig] = {
    "generate": SamplingConfig(temperature=0.9, max_new_tokens=512),
    "evolve": SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=512),
    "gradient": SamplingConfig(temperature=1.0, top_p=0.95, max_new_tokens=512),
    # protegi mc/paraphrase pass: the paper's "temperature … 1.0 in all other contexts"
    "synonym": SamplingConfig(temperature=1.0, max_new_tokens=512),
    "compare": SamplingConfig(temperature=0.0, max_new_tokens=16),
}


def resolve_tier(profiles: dict[str, SamplingConfig], tier: str) -> SamplingConfig:
    if tier not in profiles:
        raise KeyError(f"undeclared sampling tier: {tier!r} (options: {sorted(profiles)})")
    return profiles[tier]


def merge_tier(
    base: SamplingConfig, profiles: dict[str, SamplingConfig], tier: str
) -> SamplingConfig:
    """Deployment baseline (thinking/seed) + tier algorithm convention →
    effective sampling (via merge_sampling, thinking tri-state preserved)."""
    return merge_sampling(base, resolve_tier(profiles, tier))
