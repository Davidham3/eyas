"""Contract tests (ProTeGi): critique-then-edit gradient fingerprints (ProTeGi,
EMNLP 2023, arXiv:2305.03495), expansion cap, beam ranking.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from eyas.backends.base import (
    ChatResponse,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.budget import BanditBudget
from eyas.engine.domain import (
    Archive,
    Cost,
    Evaluation,
    Prompt,
    SampleOutcome,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
)
from eyas.engine.genio import (
    GEN_MARKER,
    render_sample_reference,
)
from eyas.engine.strategies.protegi import ProtegiConfig, ProtegiState, ProtegiStrategy
from eyas.engine.strategy import EvalView, OptimizerLoop, build_run_context
from eyas.tasks.base import Sample, Task
from eyas.tasks.math_qa import MathQaTask
from eyas.tasks.protocol import IdentityProtocol
from tests.contract.helpers import ToyMCQ, _gen_mock


def test_protegi_gradient_and_bandit_budget():
    config = ProtegiConfig(beam_size=2, rounds=1, n_gradients=2)
    ctx = build_run_context(
        ProtegiStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        budget=BanditBudget(eval_rounds=2, samples_per_eval=1),
    )
    result = OptimizerLoop(ctx).run(ProtegiStrategy(config, ctx), ["Classify the animal."])
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0  # the gradient rewrite adopts step by step


# --- ProTeGi mechanism fingerprints ---
#
# Paper (ProTeGi, EMNLP 2023, arXiv:2305.03495) Algorithm 2: a TWO-CALL dialogue per
# parent (∇ critique → δ edit fed the parsed gradient), a paraphrase (mc) expansion pass capped at
# max_expansion_factor, and error groups RANDOMLY sampled. Selection ranks by the
# bandit's pooled Q̂ (SelectTop_b(Q_T)), never by archive best-subset scores.
# The mock never reads request text, so these must be pinned here.


def _protegi_ctx(
    config: ProtegiConfig,
    backend: ScriptedMockBackend,
    task: Task | None = None,
    seed: int = 42,
    **kw,
):
    ctx = build_run_context(
        ProtegiStrategy,
        config,
        task or ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=seed,
        **kw,
    )
    return ctx, ProtegiStrategy(config, ctx)


def test_protegi_gradient_is_a_critique_then_edit_dialogue():
    """Algorithm 2 ∇→δ: the critique request asks for tagged reasons why the prompt got
    the sampled errors wrong; the edit request embeds the PARSED gradient text after
    "using this feedback:" — never the raw tagged completion (which would echo the
    <START>/<END> markers back)."""
    from eyas.engine.strategies import protegi as pg

    assert "reasons why" in pg.CRITIQUE_TEMPLATE
    assert "using this feedback" in pg.EDIT_TEMPLATE
    assert "Paraphrase" in pg.SYNONYM_TEMPLATE
    assert "keeping the semantic meaning" in pg.SYNONYM_TEMPLATE

    backend = _gen_mock()
    ctx, strat = _protegi_ctx(ProtegiConfig(rounds=1, beam_size=2, n_gradients=1), backend)
    result = OptimizerLoop(ctx).run(strat, ["Classify the animal."])
    assert result.best is not None
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    critiques = [t for t in gen if "reasons why" in t]
    edits = [t for t in gen if "using this feedback" in t]
    assert critiques, "no critique request was issued"
    assert edits, "no edit request was issued"
    assert gen.index(critiques[0]) < gen.index(edits[0])  # critique precedes the edit
    critique = critiques[0]
    assert f'My current prompt is: "{ToyMCQ.default_prompt}"' in critique  # official A.1 slot
    assert "Text: q1 / Label: cat / Prediction: wrong" in critique  # _sample_error_str blocks
    assert "Give 1 reasons why" in critique  # official literal, num_feedbacks rendered
    assert "Wrap each reason with <START> and <END>" in critique
    gradient = "[Classify the animal carefully and think step by step.]"
    for text in edits:
        assert f"using this feedback: {gradient}," in text  # PARSED gradient injected
        # the raw tagged completion never reaches the edit prompt: critique markers
        # are parsed away (the edit template itself only ever speaks the OUT tags)
        assert "<START>" not in text and "<END>" not in text


def test_protegi_critique_renders_list_reference_as_text():
    """Fidelity doc ape D6 invariant: a wrong sample whose
    reference is a list (KIE-style multi-value answers) must render its
    ``Label:`` slot in the critique error string as natural text, never the
    Python repr (which also made the gold permanently unmatchable in the
    error-pool filter)."""

    class _ProtegiListRefToy(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id=f"q{i}",
                    inputs={"question": f"q{i}", "image": f"q{i}.png"},
                    references={"label": [f"a{i}", f"b{i}"]},
                )
                for i in (1, 2)
            ]

    backend = _gen_mock()
    ctx, strat = _protegi_ctx(
        ProtegiConfig(rounds=1, beam_size=2, n_gradients=1),
        backend,
        task=_ProtegiListRefToy(),
    )
    labels = strat._labels_by_sample()
    assert labels["q1"] == ("q1", "a1, b1")  # rendered gold, not the Python repr
    OptimizerLoop(ctx).run(strat, ["Classify the animal."])
    critiques = [t for t, _ in backend.recorded_requests if GEN_MARKER in t and "reasons why" in t]
    assert critiques, "no critique request was issued"
    assert "Label: a1, b1" in critiques[0] or "Label: a2, b2" in critiques[0]
    assert "['a1', 'b1']" not in critiques[0] and "['a2', 'b2']" not in critiques[0]


def test_protegi_labels_read_the_task_reference_key():
    """GSM8K cells: math_qa samples carry references={"answer": ...}; the critique
    label must render that value, not an empty string from a fixed "label" key."""
    samples = [
        Sample(id=f"m{k}", inputs={"question": f"{k}+1?"}, references={"answer": str(k + 1)})
        for k in range(2)
    ]
    ctx, strat = _protegi_ctx(
        ProtegiConfig(rounds=1),
        ScriptedMockBackend(responder=lambda request, sampling, idx: ChatResponse(text="x")),
        task=MathQaTask(samples=samples),
    )
    assert strat._labels_by_sample() == {"m0": ("0+1?", "1"), "m1": ("1+1?", "2")}


def test_render_sample_reference_prefers_label_over_position():
    """KIE stores its entities list BEFORE the label; a positional read would
    render dict reprs as the gold, so the label key must win."""
    label = '[{"name": "ACME", "address": "1 Main St"}]'
    assert (
        render_sample_reference(
            {"entities": [{"name": "ACME", "address": "1 Main St"}], "label": label}
        )
        == label
    )
    assert render_sample_reference({"answer": "7"}) == "7"
    assert render_sample_reference({}) == ""


def test_protegi_labels_prefer_the_label_key_when_present():
    """The sroie boards' samples carry entities first; the critique Label slot
    must render the serialized label, never the entities list."""
    label = '[{"name": "ACME", "address": "1 Main St"}]'

    class _ProtegiKieRefToy(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id="k1",
                    inputs={"question": "extract"},
                    references={
                        "entities": [{"name": "ACME", "address": "1 Main St"}],
                        "label": label,
                    },
                )
            ]

    ctx, strat = _protegi_ctx(
        ProtegiConfig(rounds=1),
        ScriptedMockBackend(responder=lambda request, sampling, idx: ChatResponse(text="x")),
        task=_ProtegiKieRefToy(),
    )
    assert strat._labels_by_sample() == {"k1": ("extract", label)}


def test_protegi_expansion_pools_paraphrases_under_the_cap():
    """MC expansion: synonym requests carry the official paraphrase wording; the
    offspring pool is UNIQUE (paraphrase-of-paraphrase duplicates dropped), parents
    included (official `new_task_sections + [task_section]`), capped at
    max_expansion_factor per parent over the shared pool before selection."""
    backend = _gen_mock()
    cfg = ProtegiConfig(rounds=1, beam_size=2, n_gradients=1, mc_samples=2)
    ctx, strat = _protegi_ctx(cfg, backend)
    pools: list[list[str]] = []

    def crash(snapshot: dict) -> None:
        state = snapshot["state"]
        if state.get("offspring"):
            pools.append(list(state["offspring"]))
            raise KeyboardInterrupt("capture the post-expansion offspring pool")

    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(strat, ["Classify the animal."], on_round=crash)
    paraphrases = [t for t, _ in backend.recorded_requests if "paraphras" in t.lower()]
    assert paraphrases, "no paraphrase (mc) request was issued"
    assert "keeping the semantic meaning" in paraphrases[0]
    # the paraphrase phase runs on the dedicated synonym tier
    assert ctx.tier_requests.get("synonym") == len(paraphrases)
    assert pools
    offspring = pools[-1]
    assert len(offspring) <= cfg.max_expansion_factor
    assert len(offspring) == len(set(offspring))  # unique pool (dedup before truncation)
    assert any("step by step" in t for t in offspring)  # edit + mc fanout produced successors
    # offspring holds SUCCESSORS only — parents re-enter selection from state.beam
    # (the pool = beam + offspring mirrors official ``new_prompts += prompts``)
    assert ToyMCQ.default_prompt not in offspring


def test_protegi_error_groups_are_seeded_samples():
    """`random.sample` semantics: 8 distinct errors, group size 4 → every seed's first
    group is a 4-sample mixture, and two seeds draw DIFFERENT groups (a head slice is
    seed-invariant and fails the cross-seed inequality)."""
    n_errors = 8

    class EightErrorToy(ToyMCQ):
        def load_split(self, split: str) -> Sequence[Sample]:
            return [
                Sample(
                    id=f"q{i}",
                    inputs={"question": f"q{i}", "image": f"q{i}.png"},
                    references={"label": "cat"},
                )
                for i in range(n_errors)
            ]

    def first_group(seed: int) -> str:
        cfg = ProtegiConfig(rounds=1, beam_size=1, n_gradients=1, n_errors_per_grad=4)
        ctx, strat = _protegi_ctx(cfg, _gen_mock(), EightErrorToy(), seed=seed)
        prompt = Prompt(ToyMCQ.default_prompt)
        outcomes = tuple(
            SampleOutcome(f"q{i}", "wrong", {"accuracy": 0.0}) for i in range(n_errors)
        )
        archive = Archive(ctx.score_of)
        archive.add(Evaluation(prompt.id, "train", "absolute", {"accuracy": 0.0}, outcomes, Cost()))
        view = EvalView(archive, ctx.score_of)
        errors = strat._error_group(view, prompt, strat._labels_by_sample())
        assert len(errors) == n_errors  # the pool is the FULL error set (no head slice)
        group = strat._sample_group(errors, strat.derived_rng())
        return ",".join(q for q, _, _ in group)

    groups = {seed: first_group(seed) for seed in (1, 2)}
    for text in groups.values():
        assert len(text.split(",")) == 4
    assert groups[1] != groups[2]


def test_protegi_beam_ranks_by_bandit_q_hat_not_archive_scores():
    """SelectTop_b(Q_T) on the DEGENERATE fixture shape (beam == arms, offspring
    empty — the arm basis coincides with the pool, documented as degenerate);
    the in-flight shape (arms == offspring, distinct from a carryover beam) is
    pinned by ``test_protegi_beam_ranks_arms_by_identity_not_position``.

    Inverted fixture: arm0 pulled high (0.9), arm1 pulled low (0.1), but arm1's
    ARCHIVE score higher (1.0) — the bandit beam keeps arm0 FIRST (pooled Q̂
    descending) while the archive-score ranking (the paper's Unif comparator)
    would put arm1 first."""
    cfg = ProtegiConfig(rounds=1, beam_size=2)
    arms = [Prompt("arm zero"), Prompt("arm one")]
    budget = BanditBudget(eval_rounds=1, samples_per_eval=1)
    ctx, _ = _protegi_ctx(cfg, _gen_mock(), budget=budget)
    budget.propose_load(0, None, arms, ["q1", "q2"])  # one pull round -> exhausted
    evals = [
        Evaluation(a.id, "train", "absolute", {"accuracy": s}, (), Cost())
        for a, s in zip(arms, [0.9, 0.1], strict=True)
    ]
    budget.observe(evals, [0.9, 0.1])
    archive = Archive(ctx.score_of)
    for a, s in zip(arms, [0.5, 1.0], strict=True):  # archive scores deliberately inverted
        archive.add(Evaluation(a.id, "train", "absolute", {"accuracy": s}, (), Cost()))
    view = EvalView(archive, ctx.score_of)
    # beam == arm basis (the degenerate coincidence): positions map 1:1 here
    state = ProtegiState(
        beam=list(arms), offspring=[], arm_order=[a.id for a in arms], round_idx=0, phase="score"
    )
    decision = ProtegiStrategy(cfg, ctx).step(state, view)
    assert decision.action == "finish"
    assert state.beam == [arms[0], arms[1]]  # bandit Q̂ order, NOT the archive-score order
    assert decision.best_id == arms[0].id
    # and the Unif comparator: FlatBudget offers no ranking → archive-score order
    flat_ctx, _ = _protegi_ctx(cfg, _gen_mock())
    state2 = ProtegiState(beam=list(arms), offspring=[], round_idx=0, phase="score")
    ProtegiStrategy(cfg, flat_ctx).step(state2, view)
    assert state2.beam == [arms[1], arms[0]]


def test_protegi_beam_ranks_arms_by_identity_not_position():
    """The arm basis is ``state.offspring`` (what the score phase
    proposes), NOT the mixed ``beam + offspring`` selection pool. A stale
    ``beam_order`` position must never address the pool: with carryover parents
    occupying pool slots 0..1, every arm index applied to the pool would be
    off-by-len(beam) — a positional pick selects a different top-b set than the
    identity-keyed mapping."""
    cfg = ProtegiConfig(rounds=1, beam_size=3)
    budget = BanditBudget(eval_rounds=1, samples_per_eval=1)
    ctx, _ = _protegi_ctx(cfg, _gen_mock(), budget=budget)
    beam = [Prompt("carryover parent A"), Prompt("carryover parent B")]
    arms = [Prompt(f"arm {i}") for i in range(4)]
    budget.propose_load(0, None, arms, ["q1", "q2"])  # arms == the proposed offspring
    budget.observe(
        [
            Evaluation(a.id, "train", "absolute", {"accuracy": r}, (), Cost())
            for a, r in zip(arms[:2], [0.9, 0.1], strict=True)
        ],
        [0.9, 0.1],
    )  # only arm0/arm1 pulled: arm2/arm3 stay zero-pull (∞-first in Q̂ order)
    archive = Archive(ctx.score_of)
    for a in arms:  # arms scored in the archive; carryover parents unscored
        archive.add(Evaluation(a.id, "train", "absolute", {"accuracy": 0.5}, (), Cost()))
    view = EvalView(archive, ctx.score_of)
    state = ProtegiState(
        beam=list(beam),
        offspring=list(arms),
        arm_order=[a.id for a in arms],  # the committed arm basis (identity mapping)
        round_idx=0,
        phase="score",
    )
    decision = ProtegiStrategy(cfg, ctx).step(state, view)
    assert decision.action == "finish"
    # Q̂ order with zero-pull arms first: [arm2, arm3, arm0, arm1]
    assert state.beam == [arms[2], arms[3], arms[0]]
    # Positional application to the mixed pool [A, B, arm0..arm3] would select
    # {pool[2]=arm0, pool[3]=arm1, pool[0]=A} — a DIFFERENT set (A in, arm2/arm3
    # out): a positional off-by-len(beam) pick fails this assertion.
    assert decision.best_id == arms[2].id  # beam head, never the pool-offset pick


def test_protegi_score_phase_skips_the_empty_successor_pool():
    """A fully unparsable/deduped expansion must not burn empty bandit rounds: with no
    successors the score phase goes straight to selection (official: no new prompts →
    nothing for the bandit to pull)."""
    cfg = ProtegiConfig(rounds=1, beam_size=2)
    budget = BanditBudget(eval_rounds=4, samples_per_eval=1)
    ctx, _ = _protegi_ctx(cfg, _gen_mock(), budget=budget)
    arm = Prompt("only survivor")
    archive = Archive(ctx.score_of)
    archive.add(Evaluation(arm.id, "train", "absolute", {"accuracy": 0.5}, (), Cost()))
    view = EvalView(archive, ctx.score_of)
    state = ProtegiState(beam=[arm], offspring=[], round_idx=0, phase="score")
    decision = ProtegiStrategy(cfg, ctx).step(state, view)
    assert decision.action == "finish"  # select + final over the surviving beam
    # the empty pull was never proposed to the bandit
    assert budget._round == 0  # noqa: SLF001


def test_bandit_rank_orders_by_q_hat_with_unvisited_first():
    from eyas.engine.budget import EvenSplitBudget, FlatBudget

    arms = [Prompt(f"arm {i}") for i in range(4)]
    ids = ["s0", "s1", "s2", "s3"]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1, keep_top=0.5)
    assert budget.rank() is None  # before the first round there is nothing to rank
    budget.propose_load(0, None, arms, ids)
    evals = [
        Evaluation(a.id, "train", "absolute", {"accuracy": r}, (), Cost())
        for a, r in zip(arms, [0.1, 0.4, 0.9, 0.7], strict=True)
    ]
    budget.observe(evals, [0.1, 0.4, 0.9, 0.7])
    assert budget.rank() == [2, 3, 1, 0]  # pooled Q̂ descending (SelectTop_b(Q_T))
    # a second pull updates the means; the ranking follows the pooled mean, not UCB
    plan = budget.propose_load(1, None, arms, ids)
    pulled = [i for i, a in enumerate(arms) if a.id in {w.prompt.id for w in plan.items}]
    assert len(pulled) == 2  # narrowed set (keep_top 0.5)
    budget.observe(
        [
            Evaluation(arms[i].id, "train", "absolute", {"accuracy": 0.0}, (), Cost())
            for i in pulled
        ],
        [0.0] * len(pulled),
    )
    ranked = budget.rank()
    qhat = [
        v / c if c else float("inf")
        for v, c in zip(budget._values, budget._counts, strict=True)  # noqa: SLF001
    ]
    assert ranked == sorted(range(4), key=lambda i: (-qhat[i], i))
    assert ranked is not None and ranked[0] == 2  # arm2 (0.45) still leads arm3 (0.35)
    # zero-pull arms rank FIRST (UCB's unvisited optimism), behind nobody but each other
    fresh = BanditBudget(eval_rounds=3, samples_per_eval=1, keep_top=0.5)
    fresh.propose_load(0, None, arms, ids)
    fresh.observe(
        [Evaluation(arms[0].id, "train", "absolute", {"accuracy": 0.9}, (), Cost())], [0.9]
    )
    assert fresh.rank() == [1, 2, 3, 0]
    # allocators without a bandit ranking return None (the Unif-comparator contract)
    assert FlatBudget().rank() is None
    assert EvenSplitBudget(4).rank() is None


def test_protegi_state_serializes_offspring_and_seen():
    """A resumed run restores offspring texts + seen hashes from the checkpoint, so
    the in-flight offspring is never silently discarded."""
    beam = [Prompt("seed prompt")]
    offspring = [Prompt("child one"), Prompt("child two")]
    state = ProtegiState(
        beam=list(beam),
        offspring=list(offspring),
        arm_order=["h1", "h2"],
        seen={"s", "a", "m"},
        round_idx=2,
        phase="score",
    )
    data = json.loads(json.dumps(state.to_json()))
    assert set(data) == {"beam", "offspring", "arm_order", "seen", "round_idx", "phase"}
    restored = ProtegiState.from_json(data)
    assert [p.text for p in restored.beam] == [p.text for p in beam]
    assert [p.text for p in restored.offspring] == ["child one", "child two"]
    assert restored.arm_order == ["h1", "h2"]
    assert restored.seen == {"s", "a", "m"}
    assert restored.round_idx == 2
    assert restored.phase == "score"
    # round-trip through a Prompt rebuild keeps identity (hash, not text equality)
    assert restored.offspring[0].id == offspring[0].id
