"""§9 Artifact pipeline: typed/hash-addressed artifacts, cross-method soft consumption, assembly-time dependency validation."""

from __future__ import annotations

import json

import pytest

from eyas.engine.artifacts import ArtifactStore
from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec
from eyas.engine.domain import Artifact, Prompt
from eyas.engine.evaluation import AbsoluteEvalModel
from eyas.engine.strategies.ape import ApeConfig, ApeStrategy
from eyas.engine.strategies.evoprompt import EvopromptConfig, EvopromptStrategy
from eyas.engine.strategy import Strategy, build_run_context
from eyas.tasks.protocol import IdentityProtocol
from tests.contract.helpers import ToyMCQ, _gen_mock


def test_artifact_is_content_addressed():
    a = Artifact(kind="candidate_prompts", payload=["a", "b"], producer="ape")
    assert a.address() == "ape.candidate_prompts"
    same = Artifact(kind="candidate_prompts", payload=["a", "b"], producer="ape")
    diff = Artifact(kind="candidate_prompts", payload=["a", "c"], producer="ape")
    assert a.digest == same.digest and a.digest != diff.digest


def test_store_file_persistence(tmp_path):
    store = ArtifactStore(path=tmp_path / "artifacts")
    art = store.emit(Artifact(kind="candidate_prompts", payload=["x"], producer="ape"))
    assert store.available() == {"ape.candidate_prompts"}
    assert store.manifest() == {"ape.candidate_prompts": art.digest}
    on_disk = tmp_path / "artifacts" / "ape.candidate_prompts.json"
    assert json.loads(on_disk.read_text())["payload"] == ["x"]


def test_ape_emits_candidate_prompts():
    from eyas.engine.strategy import OptimizerLoop

    ctx = build_run_context(
        ApeStrategy,
        ApeConfig(num_candidates=3, demos_per_prompt=2),
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    result = OptimizerLoop(ctx).run(ApeStrategy(ApeConfig(3, 2), ctx), ["seed"])
    assert "ape.candidate_prompts" in result.artifacts  # artifact manifest lands in result


def test_cross_method_handoff_ape_to_evoprompt():
    store = ArtifactStore()
    ape_ctx = build_run_context(
        ApeStrategy,
        ApeConfig(num_candidates=3, demos_per_prompt=2),
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        artifacts=store,
    )
    from eyas.engine.strategy import OptimizerLoop

    OptimizerLoop(ape_ctx).run(ApeStrategy(ApeConfig(3, 2), ape_ctx), ["seed"])
    assert "ape.candidate_prompts" in store  # ape output lands in the pipeline

    # Hand the same store to evoprompt: init soft-consumes them into the initial population (no json path is read)
    evo_ctx = build_run_context(
        EvopromptStrategy,
        EvopromptConfig(popsize=4, rounds=1),
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        artifacts=store,
    )
    state = EvopromptStrategy(EvopromptConfig(popsize=4, rounds=1), evo_ctx).init([Prompt("seed0")])
    pop_texts = {p.text for p in state.population}
    ape_texts = set(store.get("ape.candidate_prompts").payload)
    assert pop_texts & ape_texts  # ape candidates get consumed


def test_assembly_fail_fast_on_missing_artifact():
    class Consumer(Strategy):
        name = "consumer"
        config_cls = type("C", (), {})
        state_cls = type("S", (), {})
        capabilities = CapSpec(
            name="consumer",
            requires_eval=frozenset({ABSOLUTE_EVAL}),
            consumes_artifacts=frozenset({"ape.candidate_prompts"}),
        )

        def init(self, seeds):
            raise NotImplementedError

        def step(self, state, view):
            raise NotImplementedError

    with pytest.raises(ValueError, match="cross-method artifact"):
        build_run_context(  # empty store: the dependency lacks an upstream producer -> rejected at assembly time
            Consumer,
            Consumer.config_cls(),
            ToyMCQ(),
            IdentityProtocol(),
            _gen_mock(),
            AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
            artifacts=ArtifactStore(),
        )


def test_assembly_self_produced_artifact_does_not_self_block():
    """Rule 4 consumes the declared produces_artifacts: an address the strategy
    produces itself never blocks its own assembly (empty store included)."""

    class SelfProducer(Strategy):
        name = "self_producer"
        config_cls = type("C", (), {})
        state_cls = type("S", (), {})
        capabilities = CapSpec(
            name="self_producer",
            requires_eval=frozenset({ABSOLUTE_EVAL}),
            produces_artifacts=frozenset({"ape.candidate_prompts"}),
            consumes_artifacts=frozenset({"ape.candidate_prompts"}),
        )

        def init(self, seeds):
            raise NotImplementedError

        def step(self, state, view):
            raise NotImplementedError

    ctx = build_run_context(  # empty store: self-declared producer satisfies the dependency
        SelfProducer,
        SelfProducer.config_cls(),
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        artifacts=ArtifactStore(),
    )
    assert ctx is not None
