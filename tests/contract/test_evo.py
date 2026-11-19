"""Contract tests (EvoPrompt): GA/DE fingerprints (ICLR 2024, arXiv:2309.08532),
fitness wheel, pool init, state/resume.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    system_message,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.core.objective import Objective
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.domain import (
    Archive,
    Cost,
    Evaluation,
    Prompt,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
)
from eyas.engine.genio import (
    GEN_MARKER,
)
from eyas.engine.strategies.evoprompt import EvopromptConfig, EvopromptStrategy
from eyas.engine.strategy import EvalView, OptimizerLoop, build_run_context
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.protocol import IdentityProtocol
from tests.contract.helpers import ToyMCQ, _capture_eval_trials, _gen_mock, _run


def test_evoprompt_de_positional_replacement():
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        EvopromptStrategy,
        EvopromptConfig(popsize=2, rounds=1, mode="de"),
        ToyMCQ(),
        eval_model,
        ["Classify A.", "Classify B."],
        backend=_gen_mock(),
    )
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0  # DE greedily adopts the better trial positionally


# --- EvoPrompt operator fingerprints ---
#
# The paper's core contribution is the DESIGNED operators (EvoPrompt, ICLR 2024, arXiv:2309.08532
# §3.2/§3.3): roulette-wheel GA parents (Table 4: wheel beats tournament/random), the two-step
# merge→adjust GA template, and the four-step DE text adaptation with the champion in
# the "Prompt 3" slot. The mock never reads request text, so these must be pinned here.


def _evo_ctx(config: EvopromptConfig, backend: ScriptedMockBackend, seed: int = 42):
    ctx = build_run_context(
        EvopromptStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=seed,
    )
    return ctx, EvopromptStrategy(config, ctx)


def _evo_view(ctx, scores: Mapping[str, float]) -> EvalView:
    """A real EvalView over a fabricated archive: ``scores`` maps prompt text → accuracy.

    An Evaluation without per-sample outcomes evaluates as its aggregate scores
    (evaluation.py backfill), so injecting one per prompt gives the strategy genuine
    ``view.score`` / ``view.evaluated`` answers for arbitrary prompts."""
    archive = Archive(ctx.score_of)
    for text, accuracy in scores.items():
        prompt = Prompt(text)
        archive.add(
            Evaluation(prompt.id, ctx.eval_split, "absolute", {"accuracy": accuracy}, (), Cost())
        )
    return EvalView(archive, ctx.score_of)


def _ga_parent_draws(mode: str, pop: list[Prompt], fitness: list[float], draws: int):
    """Seeded stream of GA parent pairs for one selection mode (fresh strategy/RNG each call)."""
    _, strat = _evo_ctx(
        EvopromptConfig(popsize=len(pop), rounds=1, mode="ga", sel_mode=mode),
        _gen_mock(),
        seed=7,
    )
    rng = strat.derived_rng()
    return [strat._ga_parents(pop, fitness, rng) for _ in range(draws)]


def test_evoprompt_ga_parents_are_fitness_proportional():
    """Paper §3.2 p_i = s_i / Σ_j s_j: under the roulette DEFAULT the first draw is
    the fitness share (1/10 / 1/10 / 8/10, statistical bands over 10 000 seeded
    draws) and the second parent is the next distinct member the wheel lands on —
    a tournament (exact shares 1/6 / 1/6 / 2/3) or a renormalized second draw
    lands far outside the bands. The `sel_mode="tournament"` knob switches the
    regime (self-paired parents prove the two independent tours); the seeded
    stream is reproducible (the derived_rng resume-determinism mechanism)."""
    pop = [Prompt(f"parent {i}") for i in range(3)]
    fitness = [1.0, 1.0, 8.0]
    draws_n = 10_000
    roulette = _ga_parent_draws("roulette", pop, fitness, draws_n)
    assert _ga_parent_draws("roulette", pop, fitness, 1000) == roulette[:1000]  # seeded
    tournament = _ga_parent_draws("tournament", pop, fitness, draws_n)

    # roulette: two distinct wheel draws; marginals within ±0.02 of the paper's
    # proportions (several σ at n=10 000) — a tournament (first-draw shares
    # 1/6 / 1/6 / 2/3) lands far outside these bands
    assert all(a.id != b.id for a, b in roulette)
    counts = Counter(next(i for i, p in enumerate(pop) if p.id == a.id) for a, _ in roulette)
    assert all(abs(counts[i] / draws_n - f / 10.0) < 0.02 for i, f in enumerate(fitness))
    # fitness-8 member appears as SOME parent far more often than a fitness-1 one
    assert sum(1 for a, b in roulette if {a.id, b.id} & {pop[2].id}) > sum(
        1 for a, b in roulette if {a.id, b.id} & {pop[0].id}
    )
    # tournament knob: two INDEPENDENT 2-way tours — a member can pair with itself
    # (~41% expected under the exact shares 1/6 / 1/6 / 2/3), impossible for the wheel
    assert sum(1 for a, b in tournament if a.id == b.id) / draws_n > 0.25
    tour_counts = Counter(next(i for i, p in enumerate(pop) if p.id == a.id) for a, _ in tournament)
    assert tour_counts[2] < 7500  # ≈ 6670, clearly below the wheel's 8000
    # zero fitness degenerates to a uniform wheel (Σ s_j = 0 has no proportions)
    flat = _ga_parent_draws("roulette", pop, [0.0] * 3, 1000)
    assert all(a.id != b.id for a, b in flat)
    flat_counts = Counter(a.id for a, _ in flat)
    assert all(abs(count / 1000 - 1 / 3) < 0.06 for count in flat_counts.values())


def test_evoprompt_ga_init_raises_on_single_distinct_seed():
    """Structural infeasibility is an INIT-time fact: an evoprompt GA pool with
    fewer than 2 distinct prompts (one seed, or duplicates collapsing under
    prompt_hash) raises before any request is issued, naming the arithmetic and
    the fix — never at the first GA round after engine startup. A popsize < 2
    config is rejected even earlier, at construction, in both modes."""
    config = EvopromptConfig(popsize=10, rounds=3, mode="ga")
    # the exact loop shape: OptimizerLoop calls init() before its first evaluate
    with pytest.raises(ValueError, match="population >= 2"):
        _run(
            EvopromptStrategy,
            config,
            ToyMCQ(),
            AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
            ["Classify the animal."],
            backend=_gen_mock(),
        )
    # duplicates dedup by prompt id to one distinct member → the same raise
    _, strat = _evo_ctx(config, _gen_mock())
    with pytest.raises(ValueError, match="strategy.seeds"):
        strat.init([Prompt("same"), Prompt("same")])
    # the guard's message carries the arithmetic and the fix
    with pytest.raises(ValueError) as excinfo:
        strat.init([Prompt("alone")])
    message = str(excinfo.value)
    assert "1 distinct prompt(s)" in message and "popsize=10" in message
    # a two-distinct pool crosses the guard (init returns, seed_eval pending)
    state = strat.init([Prompt("first"), Prompt("second")])
    assert state.phase == "seed_eval"
    # popsize=1 is structurally infeasible in BOTH modes and is rejected at
    # strategy construction (the earliest point the config is knowable) —
    # the degenerate-pool pad in _de_donors stays reachable only as an
    # internal shape guard, never through a shipped popsize=1 config
    for mode in ("ga", "de"):
        bad = EvopromptConfig(popsize=1, rounds=1, mode=mode)
        with pytest.raises(ValueError, match="popsize"):
            _evo_ctx(bad, _gen_mock())


def test_evoprompt_rejects_popsize_below_two_at_construction():
    """popsize < 2 cannot cross (GA) or difference (DE) — the guard fires at
    construction for both modes, naming the knob, never at the first round
    after engine startup."""
    for mode in ("ga", "de"):
        for bad_popsize in (0, 1, -3):
            config = EvopromptConfig(popsize=bad_popsize, rounds=1, mode=mode)
            with pytest.raises(ValueError, match="popsize"):
                _evo_ctx(config, _gen_mock())
    # popsize=2 constructs cleanly in both modes (the minimum viable population)
    for mode in ("ga", "de"):
        _, strat = _evo_ctx(EvopromptConfig(popsize=2, rounds=1, mode=mode), _gen_mock())
        assert strat.config.popsize == 2


def test_evoprompt_negative_fitness_is_clamped_before_the_wheel():
    """Oriented scores can be negative (minimize objectives): a negative share
    must be clamped to 0 BEFORE rng.choices — the raw negative weight otherwise
    distorts the cumulative wheel (a non-monotonic cum_weights list silently
    starves the positive members). fitness [-0.5, 0.2, 0.4] clamps to
    [0, 0.2, 0.4]: member 0 never drawn, member 1 takes a 1/3 share."""
    pop = [Prompt(f"member {i}") for i in range(3)]
    fitness = [-0.5, 0.2, 0.4]
    draws = _ga_parent_draws("roulette", pop, fitness, 3000)
    first_counts = Counter(next(i for i, p in enumerate(pop) if p.id == a.id) for a, _ in draws)
    assert first_counts[0] == 0  # clamped to zero: no share
    assert abs(first_counts[1] / 3000 - 1 / 3) < 0.05  # the unclamped wheel starves it to 0
    assert all(a.id != b.id for a, b in draws)


def test_evoprompt_fitness_none_and_zero_both_carry_zero_share():
    """_fitness distinguishes None (never evaluated) from a real 0.0 score with
    an explicit check — both carry zero wheel share, but the None path must not
    ride truthiness (a future score type must not be silently coerced)."""
    ctx, strat = _evo_ctx(EvopromptConfig(popsize=2, rounds=1, mode="ga"), _gen_mock())
    scored_zero = Prompt("scored zero")
    never_evaluated = Prompt("never evaluated")
    view = _evo_view(ctx, {scored_zero.text: 0.0})
    assert strat._fitness(view, [scored_zero, never_evaluated]) == [0.0, 0.0]


def test_evoprompt_ga_ten_seed_pool_flows_to_population_of_ten():
    """The shipped cell's official N_P=10 pool end to end: the seed round evaluates
    ALL 10 pool members (the pool is evaluated before truncating, and popsize=10 truncates
    nothing), so the GA round that follows crosses among 10 distinct parents — with
    popsize=10 and a 10-seed pool the crossing population is never degenerate."""
    seeds = [f"Classify the animal with cue {i}." for i in range(10)]
    backend = _gen_mock()
    ctx = build_run_context(
        EvopromptStrategy,
        EvopromptConfig(popsize=10, rounds=1, mode="ga"),
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    rounds: list[dict] = []
    result = OptimizerLoop(ctx).run(
        EvopromptStrategy(EvopromptConfig(popsize=10, rounds=1, mode="ga"), ctx),
        list(seeds),
        on_round=rounds.append,
    )
    # the seed round (round 0) carried all 10 pool members as candidates
    seed_round = rounds[0]["history"][0]
    assert seed_round["round"] == 0 and seed_round["n_candidates"] == 10
    # the seed round evaluated ALL 10 pool members (the pool is evaluated in full
    # before truncating; popsize=10 then truncates nothing) — 10 members × 2
    # toy samples. "cue" isolates the seed-pool evals from the later trial eval;
    # the GEN_MARKER exclusion drops the GA calls (they cite pool members in
    # their Prompt 1/2 parent slots).
    seed_eval_requests = [
        t for t, _ in backend.recorded_requests if "cue" in t and GEN_MARKER not in t
    ]
    assert len(seed_eval_requests) == 10 * 2
    assert len({next(s for s in seeds if s in t) for t in seed_eval_requests}) == 10
    # the GA round crossed distinct parents: every child request pairs two DIFFERENT
    # pool members (population = 10 → distinctness never needs padding)
    gen_requests = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(gen_requests) == 10  # N children per generation (Algorithm 2)
    for text in gen_requests:
        slot1 = [s for s in seeds if f"Prompt 1: {s}" in text]
        slot2 = [s for s in seeds if f"Prompt 2: {s}" in text]
        assert len(slot1) == len(slot2) == 1 and slot1[0] != slot2[0]
    # GA update = elitist top-10 over pool ∪ trials: the run completes with a best
    assert result.best is not None
    assert result.history


def test_evoprompt_ga_request_carries_two_step_merge_template():
    """Official GA semantics: a one-shot example, then Step 1 merge/combine-the-core-ideas
    + Step 2 adjust-randomly, one call per population member (Algorithm 2: N children)."""
    from eyas.engine.strategies import evoprompt as evo

    assert "Merge" in evo.GA_STEP1_INSTRUCTION
    assert "combine their core idea" in evo.GA_STEP1_INSTRUCTION
    assert "Adjust" in evo.GA_STEP2_INSTRUCTION and "randomly" in evo.GA_STEP2_INSTRUCTION
    assert evo.OUT_START in evo.GA_STEP2_INSTRUCTION and evo.OUT_END in evo.GA_STEP2_INSTRUCTION

    backend = _gen_mock()
    ctx, strat = _evo_ctx(EvopromptConfig(popsize=2, rounds=1, mode="ga"), backend)
    seeds = ["Classify the animal one-shot.", "Classify the animal two-shot."]
    state = strat.init([Prompt(s) for s in seeds])
    state.phase = "gen"
    view = _evo_view(ctx, dict.fromkeys(seeds, 0.5))
    state.cur_best, state.cur_best_score = state.population[0], 0.5
    trials = strat._variation(state, view)
    assert len(trials) == len(state.population)
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(gen) == 2  # one LLM call per child
    for text in gen:
        assert evo.GA_EXAMPLE in text  # the one-shot example precedes the parents
        assert evo.GA_STEP1_INSTRUCTION in text
        assert evo.GA_STEP2_INSTRUCTION in text
    assert f"Prompt 1: {state.population[0].text}" in gen[0]
    assert "Prompt 2: " in gen[0]


def test_evoprompt_de_request_is_the_four_step_operator():
    """Official DE template slots: Step 1 identify different parts (Prompt 1/2 = the two
    donors), Step 2 mutate them, Step 3 combine with Prompt 3 = the CHAMPION (cur_best,
    not the per-round population max), Step 4 crossover with the Basic Prompt = the base."""
    from eyas.engine.strategies import evoprompt as evo

    for marker in (
        evo.DE_STEP1_INSTRUCTION,
        evo.DE_STEP2_INSTRUCTION,
        evo.DE_STEP3_INSTRUCTION,
        evo.DE_STEP4_INSTRUCTION,
    ):
        assert marker in evo.DE_TEMPLATE
    assert "different parts" in evo.DE_STEP1_INSTRUCTION
    assert "Randomly mutate" in evo.DE_STEP2_INSTRUCTION
    assert "Prompt 3" in evo.DE_STEP3_INSTRUCTION
    assert "Crossover" in evo.DE_STEP4_INSTRUCTION

    backend = _gen_mock()
    ctx, strat = _evo_ctx(EvopromptConfig(popsize=3, rounds=1, mode="de"), backend)
    seeds = [
        "Classify the animal one-shot.",
        "Classify the animal two-shot.",
        "Classify the animal three-shot.",
    ]
    state = strat.init([Prompt(s) for s in seeds])
    state.phase = "gen"
    champion = state.population[0]
    view = _evo_view(ctx, {seeds[0]: 1.0, seeds[1]: 0.5, seeds[2]: 0.25})
    state.cur_best, state.cur_best_score = champion, 1.0
    trials = strat._variation(state, view)
    assert len(trials) == 3
    de_requests = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(de_requests) == 3  # one DE call per population member (Algorithm 3)
    for base, text in zip(state.population, de_requests, strict=True):
        assert evo.DE_EXAMPLE in text  # the one-shot example precedes the slots
        for marker in (
            evo.DE_STEP1_INSTRUCTION,
            evo.DE_STEP2_INSTRUCTION,
            evo.DE_STEP3_INSTRUCTION,
            evo.DE_STEP4_INSTRUCTION,
        ):
            assert marker in text
        assert f"Prompt 3: {champion.text}" in text  # champion wired into the combine slot
        assert f"Basic Prompt: {base.text}" in text
        base_pos = text.index(f"Basic Prompt: {base.text}")
        for other in seeds:
            if other == base.text:
                continue
            donor_positions = [
                text.index(f"Prompt {slot}: {other}")
                for slot in (1, 2)
                if f"Prompt {slot}: {other}" in text
            ]
            # the non-base members are the differencing donors, in the Prompt 1/2 slots,
            # and the basic-prompt slot is the LAST prompt filled in the template
            assert donor_positions and min(donor_positions) < base_pos


def test_evoprompt_de_donors_exclude_the_base():
    """Algorithm 3: r1 ≠ r2 ≠ i — both donors drawn from population \\ {base}."""
    pop = [Prompt(f"member {i}") for i in range(5)]
    _, strat = _evo_ctx(EvopromptConfig(popsize=5, rounds=1, mode="de"), _gen_mock(), seed=3)
    rng = strat.derived_rng()
    for j in range(len(pop)):
        for _ in range(50):
            donor_a, donor_b = strat._de_donors(pop, j, rng)
            assert donor_a.id != donor_b.id
            assert donor_a.id != pop[j].id and donor_b.id != pop[j].id
    # population \ {base} smaller than 2 pads the twin slot with the base itself — the
    # minimum deviation that keeps the per-slot trial shape at degenerate popsize 1
    lone = [Prompt("only one")]
    donor_a, donor_b = strat._de_donors(lone, 0, rng)
    assert donor_a.id == donor_b.id == lone[0].id


class MarkerSeedsToy(Task):
    """Sample s{i} is correct exactly for prompts listing marker s{i} (the backend echoes
    the marker back when the prompt names it). Seed i lists s0..s{i}, so the eight seeds
    get EIGHT DIFFERENT scores and the top-popsize truncation has a unique right answer."""

    name = "toy_marker_seeds"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(
                id=f"s{i}",
                inputs={"question": f"s{i}", "image": f"s{i}.png"},
                references={"marker": f"s{i}"},
            )
            for i in range(8)
        ]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        hit = 1.0 if sample.references["marker"] in response.text else 0.0
        return Prediction(value=hit, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": float(prediction.value)}


def _marker_seed_mock() -> ScriptedMockBackend:
    """Echo backend: answers a sample with its question when the prompt names that
    marker, otherwise with text that matches nothing."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_text = "".join(
            p.text for m in request.messages if m.role == "system" for p in m.content
        )
        question = user_text(request)
        return ChatResponse(text=question if question in sys_text else "no-marker")

    return ScriptedMockBackend(responder=responder)


def test_evoprompt_init_evaluates_pool_then_truncates_top_popsize():
    """The initial population is the TOP-popsize of the EVALUATED pool — all 8 seeds
    are evaluated first; truncation happens only once the scores exist (cutting the pool
    to popsize BEFORE evaluation would make the rank/truncate path dead)."""
    backend = _marker_seed_mock()
    seeds = [f"Classify. markers: {', '.join(f's{k}' for k in range(i + 1))}" for i in range(8)]
    cfg = EvopromptConfig(popsize=4, rounds=1, mode="de")
    ctx = build_run_context(
        EvopromptStrategy,
        cfg,
        MarkerSeedsToy(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    snapshots: list[dict] = []
    eval_requests_at_round: list[int] = []

    def on_round(snapshot: dict) -> None:
        snapshots.append(snapshot)
        eval_requests_at_round.append(
            sum(1 for t, _ in backend.recorded_requests if GEN_MARKER not in t)
        )

    result = OptimizerLoop(ctx).run(EvopromptStrategy(cfg, ctx), list(seeds), on_round=on_round)
    # the FIRST round evaluated ALL EIGHT seeds (8 prompts × 8 samples); truncating
    # to popsize=4 before evaluating anything would be wrong
    assert result.history[0].n_candidates == 8
    assert eval_requests_at_round[0] == 8 * 8
    # truncation happened with the scores in hand: the survivors are the top-4 (seeds 7..4,
    # scores 1.0 / 0.875 / 0.75 / 0.625) — keeping the first four would always be
    # wrong. The last checkpoint is the round-end snapshot of the trial round (phase
    # eval_trials: the flip to done happens in the NEXT step, which emits no checkpoint).
    assert snapshots[-1]["state"]["phase"] == "eval_trials"
    assert snapshots[-1]["state"]["population"] == [seeds[i] for i in (7, 6, 5, 4)]


def test_evoprompt_state_serializes_trials_and_cur_best():
    """A mid-round resume restores trials/cur_best from the checkpoint instead of
    crashing on empty trials and losing the champion. Full round-trip (the
    checkpoint.json JSON path)."""
    from eyas.engine.strategies.evoprompt import EvoState

    pop = [Prompt("p one"), Prompt("p two")]
    state = EvoState(
        population=list(pop),
        trials=[Prompt("t one"), Prompt("t two")],
        cur_best=pop[1],
        cur_best_score=0.75,
        round_idx=2,
        phase="eval_trials",
    )
    restored = EvoState.from_json(json.loads(json.dumps(state.to_json())))
    assert [p.text for p in restored.population] == [p.text for p in pop]
    assert [p.text for p in restored.trials] == ["t one", "t two"]
    assert restored.cur_best is not None and restored.cur_best.id == pop[1].id
    assert restored.cur_best_score == 0.75
    assert restored.round_idx == 2
    assert restored.phase == "eval_trials"


def test_evoprompt_mid_round_resume_survives_trial_state():
    """Resuming from the post-generation checkpoint (phase=eval_trials, trials only in
    memory) restores the serialized trials/cur_best, so the resumed round neither
    crashes on mismatched lengths nor loses the champion identity."""
    from eyas.engine.strategies.evoprompt import EvoState

    cfg = EvopromptConfig(popsize=2, rounds=2, mode="de")
    seeds = ["Classify the animal one-shot.", "Classify the animal two-shot."]

    snapshots: list[dict] = []
    ctx, interrupted = _evo_ctx(cfg, _gen_mock())
    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(interrupted, list(seeds), on_round=_capture_eval_trials(snapshots))
    assert snapshots, "no eval_trials checkpoint was produced"
    checkpoint = json.loads(json.dumps(snapshots[-1]))
    # trials must BE in the checkpoint, not just survive one
    assert checkpoint["state"].get("trials"), "trials missing from the eval_trials checkpoint"
    assert checkpoint["state"].get("cur_best") is not None

    ctx, resumed_strategy = _evo_ctx(cfg, _gen_mock())
    resumed = OptimizerLoop(ctx).run(resumed_strategy, list(seeds), resume=checkpoint)
    assert resumed.best is not None
    restored = EvoState.from_json(checkpoint["state"])
    assert restored.cur_best is not None and restored.cur_best_score is not None
    assert len(restored.trials) == 2
