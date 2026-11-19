"""merge_sampling: merge semantics for stage sampling inheriting
deployment-level defaults."""

from __future__ import annotations

from eyas.backends.base import SamplingConfig, merge_sampling


def test_stage_overrides_win_and_none_falls_back():
    base = SamplingConfig(temperature=0.9, max_new_tokens=4096, thinking=False, seed=7, top_k=5)
    stage = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=2048)
    m = merge_sampling(base, stage)
    assert m.temperature == 0.7
    assert m.max_new_tokens == 2048
    assert m.top_p == 0.95
    assert m.thinking is False  # stage None -> falls back to the deployment default
    assert m.seed == 7
    assert m.top_k == 5


def test_greedy_temperature_is_explicit_not_default():
    # temperature has no tri-state: 0.0 is an explicit greedy value, not "unset" — it must override base
    base = SamplingConfig(temperature=0.9)
    stage = SamplingConfig(temperature=0.0, max_new_tokens=6)
    assert merge_sampling(base, stage).temperature == 0.0


def test_budget_fields_never_inherit():
    # temperature / max_new_tokens are algorithm budget fields: stage always sets them explicitly, no fallback
    base = SamplingConfig(thinking=False, repetition_penalty=1.05, seed=3, max_new_tokens=8192)
    m = merge_sampling(base, SamplingConfig())
    assert m.thinking is False
    assert m.repetition_penalty == 1.05
    assert m.seed == 3
    assert m.max_new_tokens == 1024  # stage default, not base's 8192
