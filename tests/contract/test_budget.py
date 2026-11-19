"""Contract tests (budgets): bandit narrowing/takeover/window rules and
even-split division — the allocator machinery strategies share.
"""

from __future__ import annotations

import pytest

from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.budget import BanditBudget
from eyas.engine.domain import (
    Archive,
    Cost,
    Evaluation,
    Prompt,
    SampleOutcome,
)
from tests.contract.helpers import _noop_state


def test_bandit_budget_narrows_adaptively():
    arms = [Prompt(f"arm {i}") for i in range(4)]
    ids = [f"s{i}" for i in range(4)]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1, keep_top=0.5)
    plan0 = budget.propose_load(0, _noop_state(), arms, ids)
    assert len(plan0.items) == 4  # round 0 evaluates every arm
    rewards = [0.0, 0.0, 0.0, 1.0]
    evals = [
        Evaluation(a.id, "t", "absolute", {"accuracy": r}, (), Cost())
        for a, r in zip(arms, rewards, strict=True)
    ]
    budget.observe(evals, rewards)
    plan1 = budget.propose_load(1, _noop_state(), arms, ids)
    assert len(plan1.items) == 2  # adaptively narrowed to the top 50%
    chosen_ids = {w.prompt.id for w in plan1.items}
    assert arms[3].id in chosen_ids  # UCB prefers the high-mean arm


def test_archive_score_tie_prefers_larger_evidence_set():
    """Archive.add contract (SPO closing-pass rule): strictly higher score wins; on a score tie
    the LARGER per-sample evidence set wins — a subset round must not lock a
    prompt's reported entry below its full-split record (spo's closing pass
    overwrites its vote-subset record exactly this way; without the rule the
    final Evaluation would report subset evidence at the full-split score).
    Equal score AND equal size keep the first writer (deterministic, no
    insertion-order drift under resume)."""

    def _eval(prompt: Prompt, ids: tuple[str, ...], acc: float, round_idx: int) -> Evaluation:
        return Evaluation(
            prompt_id=prompt.id,
            split="train",
            eval_model="absolute",
            scores={"accuracy": acc},
            per_sample=tuple(
                SampleOutcome(sample_id=s, prediction="p", scores={"accuracy": acc}) for s in ids
            ),
            cost=Cost(requests=len(ids), wall_seconds=0.0),
            round_idx=round_idx,
        )

    def _score_of(e: Evaluation) -> float:
        return float(e.scores["accuracy"])

    archive = Archive(_score_of)
    p = Prompt("tied prompt")
    archive.add(_eval(p, ("q1",), 1.0, 0))
    archive.add(_eval(p, ("q1", "q2"), 1.0, 1))  # tie, larger evidence -> wins
    assert len(archive.best(p.id).per_sample) == 2  # type: ignore[union-attr]
    archive.add(_eval(p, ("q1",), 1.0, 2))  # tie, smaller evidence -> keeps best
    assert len(archive.best(p.id).per_sample) == 2  # type: ignore[union-attr]
    archive.add(_eval(p, ("q1",), 0.5, 3))  # lower score -> never displaces
    assert len(archive.best(p.id).per_sample) == 2  # type: ignore[union-attr]
    best = archive.best(p.id)
    q = Prompt("first writer")
    archive.add(_eval(q, ("q1", "q2"), 1.0, 4))
    archive.add(_eval(q, ("q1", "q2"), 1.0, 5))  # equal score + size -> first writer stays
    assert archive.best(q.id) is not None
    assert archive.best(q.id).round_idx == 4
    assert best is not None and best.round_idx == 1


def test_takeover_protocol_all_allocators_accept_and_bandit_invalidates():
    """Budget protocol: Flat/EvenSplit accept ``takeover`` as a no-op; BanditBudget
    invalidates its arm state (a strategy-owned round bypasses propose_load, so the
    allocator must not carry arms/counts over a takeover)."""
    from eyas.engine.budget import EvenSplitBudget, FlatBudget

    FlatBudget().takeover(0)  # no-op contract
    EvenSplitBudget(4).takeover(0)

    arms = [Prompt(f"arm {i}") for i in range(3)]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1)
    budget.propose_load(0, None, arms, ["s0", "s1", "s2"])
    assert budget.get_state() is not None
    budget.takeover(1)
    assert budget.get_state() is None  # arms/counts/round cleared
    # the next propose_load re-baselines on the freshly proposed arm set
    plan = budget.propose_load(2, None, arms, ["s0", "s1", "s2"])
    assert len(plan.items) == 3


def test_bandit_observe_after_takeover_is_a_documented_noop():
    """A documented no-op: the OptimizerLoop calls ``observe`` unconditionally
    after EVERY evaluate round — including a strategy-owned ``Decision(items=...)``
    round whose ``takeover`` already unbound the arms (the spo×bandit API shape).
    After such a takeover ``observe`` is a documented no-op: no raise, zero state
    corruption (arms stay unbound, counts untouched), and the next ``propose_load``
    re-baselines UCB on the fresh arm set."""
    arms = [Prompt(f"arm {i}") for i in range(2)]
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1)
    budget.propose_load(0, None, arms, ["s0", "s1"])
    evals = [Evaluation(a.id, "train", "absolute", {"accuracy": 0.5}, (), Cost()) for a in arms]
    budget.observe(evals, [0.5, 0.5])
    assert budget._counts == [1, 1]  # noqa: SLF001
    budget.takeover(1)
    # the loop's unconditional observe after the strategy-owned round: no raise
    assert budget.observe(evals, [0.9, 0.9]) == [e.prompt_id for e in evals]
    assert budget._arms is None  # noqa: SLF001  # still unbound — no silent re-binding
    assert budget.get_state() is None  # and nothing was credited
    assert budget._round == 0  # noqa: SLF001  # the takeover cursor never advanced
    # the next propose_load re-baselines: fresh arms, zero counts, window at 0
    plan = budget.propose_load(2, None, arms, ["s0", "s1"])
    assert budget._counts == [0, 0]  # noqa: SLF001
    assert all(item.sample_ids == ("s0",) for item in plan.items)


def test_bandit_window_exhaustion_raises_with_remediation():
    """Config is law: "sample
    quantities belong to the configurator — runtime auto-shrinking IS a silent
    downgrade." With the official ``8×32`` caps over the 248-sample bench splits
    the 8th pull is a legitimate PARTIAL window (``248 − 7×32 = 24`` — data-end);
    a NINTH pull starts at or
    beyond the split end and raises ``ValueError``. The message carries the
    arithmetic (pull, start, n, planned) AND both remediation values computed
    for the configurator — ``samples_per_eval ≤ n // eval_rounds`` (248 // 8 =
    31, the value the shipped 248-split cells now pin) and ``eval_rounds ≤
    n // samples_per_eval`` (248 // 32 = 7). The framework refuses to guess."""
    n, spe, rounds = 248, 32, 8
    arms = [Prompt(f"arm {i}") for i in range(4)]
    ids = [f"s{i:03d}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
    budget.propose_load(0, None, arms, ids)  # a first arm set takes pull 1 ...
    budget.takeover(0)  # ... a strategy-owned round re-baselines before the scoring pulls
    served: list[tuple[str, ...]] = []
    for pull in range(1, rounds + 1):  # the 8 scoring pulls; the 8th is PARTIAL at 224
        plan = budget.propose_load(pull, None, arms, ids)
        window = tuple(ids[(pull - 1) * spe : pull * spe])
        served.append(window)
        for item in plan.items:
            assert item.sample_ids == window
    assert len(served[-1]) == n - 7 * spe == 24  # last-partial-window serves (legal tail)
    assert len(set(served)) == rounds  # one window per pull — the silent repeat stays dead
    with pytest.raises(ValueError, match=r"eval_rounds × samples_per_eval") as exc:
        budget.propose_load(rounds + 1, None, arms, ids)
    msg = str(exc.value)
    assert "samples_per_eval ≤ 31" in msg  # fix A: n // eval_rounds = 248 // 8
    assert "eval_rounds ≤ 7" in msg  # fix B: n // samples_per_eval = 248 // 32
    # the raise is state-clean: the round cursor never advanced past the 8 pulls
    assert budget._round == rounds  # noqa: SLF001


def test_bandit_exact_split_boundary_raises_only_beyond_the_last_window():
    """n = eval_rounds × samples_per_eval exactly: every pull carries a full
    window; the pull BEYOND the horizon raises with both remediation values
    (at the exact boundary both fixes name the current values — the plan is
    already maximal)."""
    arms = [Prompt("a"), Prompt("b")]
    ids = ["s0", "s1", "s2", "s3"]
    budget = BanditBudget(eval_rounds=2, samples_per_eval=2)
    for pull, start in ((1, 0), (2, 2)):  # two full windows cover the split exactly
        plan = budget.propose_load(pull, None, arms, ids)
        assert all(item.sample_ids == tuple(ids[start : start + 2]) for item in plan.items)
    with pytest.raises(ValueError, match=r"2 × 2") as exc:
        budget.propose_load(3, None, arms, ids)
    assert "samples_per_eval ≤ 2" in str(exc.value)  # 4 // 2
    assert "eval_rounds ≤ 2" in str(exc.value)  # 4 // 2


def test_bandit_degenerate_split_raise_names_impossible_fixes():
    """n < samples_per_eval (7 samples, 8×32): pull 1 serves the 7-sample head
    (a partial window — the split end is reached inside the first pull), pull 2
    starts at 32 ≥ 7 and raises. Both remediation values compute to 0 — neither
    cap can be repaired into a fit — and the message says so explicitly: the
    only fix is more selection data (or a different split)."""
    n, spe, rounds = 7, 32, 8
    arms = [Prompt("a"), Prompt("b")]
    ids = [f"s{i}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe)
    plan = budget.propose_load(1, None, arms, ids)
    for item in plan.items:
        assert item.sample_ids == tuple(ids)  # partial head serve
    with pytest.raises(ValueError, match=r"window exhaustion") as exc:
        budget.propose_load(2, None, arms, ids)
    msg = str(exc.value)
    assert "samples_per_eval ≤ 0" in msg  # 7 // 8 — impossible
    assert "eval_rounds ≤ 0" in msg  # 7 // 32 — impossible
    assert "cannot host the plan at any cap" in msg  # the impossible-fix hint


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"eval_rounds": 0}, r"eval_rounds must be >= 1, got 0"),
        ({"samples_per_eval": 0}, r"samples_per_eval must be >= 1, got 0"),
        ({"samples_per_eval": -1}, r"samples_per_eval must be >= 1, got -1"),
        ({"c": -0.1}, r"c must be >= 0, got -0.1"),
        ({"keep_top": 0.0}, r"keep_top must be in \(0, 1\], got 0.0"),
        ({"keep_top": -0.5}, r"keep_top must be in \(0, 1\], got -0.5"),
        ({"keep_top": 1.5}, r"keep_top must be in \(0, 1\], got 1.5"),
    ],
)
def test_bandit_ctor_rejects_invalid_params(kwargs: dict, match: str) -> None:
    """Every numeric ctor param is range-validated at construction (fail fast,
    naming the offending value): a zero/negative window or round count, a
    negative UCB exploration coefficient, or a keep_top outside the (0, 1]
    arm-fraction range must die at assembly, not silently distort the plan."""
    with pytest.raises(ValueError, match=match):
        BanditBudget(**kwargs)


def test_bandit_ctor_accepts_boundary_values() -> None:
    budget = BanditBudget(eval_rounds=1, samples_per_eval=1, c=0.0, keep_top=1.0)
    assert (budget.eval_rounds, budget.samples_per_eval, budget.c, budget.keep_top) == (
        1,
        1,
        0.0,
        1.0,
    )


@pytest.mark.parametrize("n", [1, 2, 7, 24, 64, 74, 110, 248, 256, 296])
@pytest.mark.parametrize("rounds", [1, 3, 8])
@pytest.mark.parametrize("spe", [1, 8, 32])
def test_bandit_window_plan_grid_serves_or_raises(n, rounds, spe):
    """Config is law: the configured plan (``rounds`` pulls of
    ``spe``) executes exactly — a fitting plan serves the consecutive windows
    verbatim (the last one partial at the data end is legal — consecutive
    windows may end on a partial final window); a plan
    whose pull starts at or beyond the split end raises AT that pull, carrying
    both computed remediation values, and the raise is final (later pulls raise
    too — no adaptation ever kicks in)."""
    arms = [Prompt("a"), Prompt("b")]
    ids = [f"s{i}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
    raised = False
    for k in range(rounds):
        if k * spe >= n:
            with pytest.raises(ValueError, match="window exhaustion") as exc:
                budget.propose_load(k, None, arms, ids)
            msg = str(exc.value)
            assert f"samples_per_eval ≤ {n // rounds}" in msg
            assert f"eval_rounds ≤ {n // spe}" in msg
            assert budget._round == k  # noqa: SLF001  # state-clean at the raise
            # the raise is final: exhaustion is a config fact, not a transient
            with pytest.raises(ValueError, match="window exhaustion"):
                budget.propose_load(k + 1, None, arms, ids)
            assert budget._round == k  # noqa: SLF001
            raised = True
            break
        plan = budget.propose_load(k, None, arms, ids)
        window = tuple(ids[k * spe : min((k + 1) * spe, n)])
        assert window  # a served window is never empty
        for item in plan.items:
            assert item.sample_ids == window
    if not raised:
        assert budget._round == rounds  # noqa: SLF001  # the plan completed exactly


def test_bandit_normal_path_serves_the_expected_windows():
    """Normal-path invariance (guard kept by the shipped contract): the served windows are
    byte-identical to the documented shape — consecutive windows of ``spe`` from
    the plain ``samples[start : start + spe]`` slice. The cub 261 / pets 296
    bench cells (8×32 = 256 ≤ n) serve window-for-window unchanged; the
    hypothetical 8×32 over the 248 splits completes too, ending on the legal
    24-sample partial tail (pull 8 starts at 224 < 248, a legal data-end tail)."""
    for n in (248, 256, 261, 296):
        rounds, spe = 8, 32
        assert (rounds - 1) * spe < n  # no pull of the plan starts at/beyond the end
        arms = [Prompt(f"arm {i}") for i in range(3)]
        ids = [f"s{i:04d}" for i in range(n)]
        budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
        for pull in range(1, rounds + 1):
            plan = budget.propose_load(pull, None, arms, ids)
            window = tuple(ids[(pull - 1) * spe : pull * spe])  # window arithmetic, verbatim
            assert window  # every served window is nonempty
            for item in plan.items:
                assert item.sample_ids == window
        assert budget._round == rounds  # noqa: SLF001


def test_bandit_exact_fit_serves_full_windows():
    """8×31 = 248 exactly (the shape the 12 sroie/textvqa bench cells now ship):
    every pull carries a FULL 31-sample window, the eighth ends exactly at the
    split end, and no pull arms exhaustion."""
    n, spe, rounds = 248, 31, 8
    arms = [Prompt("a"), Prompt("b")]
    ids = [f"s{i:03d}" for i in range(n)]
    budget = BanditBudget(eval_rounds=rounds, samples_per_eval=spe, keep_top=0.5)
    for pull in range(1, rounds + 1):
        plan = budget.propose_load(pull, None, arms, ids)
        window = tuple(ids[(pull - 1) * spe : pull * spe])
        assert len(window) == spe
        for item in plan.items:
            assert item.sample_ids == window


def test_even_split_budget_divides_the_total_budget_evenly() -> None:
    """EvenSplitBudget contract (protegi bf semantics): the
    configured total sample budget is split across candidates as
    ``per = max(1, min(|split|, ceil(total / n)))``, every candidate gets the
    same subset, the split caps the per-candidate window, and the allocator is
    stateless (rank None / takeover no-op / get_state None). Pins the registered
    built-in's core arithmetic — the shape no shipped config selects today."""
    from eyas.engine.budget import EvenSplitBudget

    prompts = tuple(Prompt(f"p{i}") for i in range(3))
    sample_ids = tuple(f"s{i}" for i in range(20))

    plan = EvenSplitBudget(total_sample_budget=9).propose_load(0, None, prompts, sample_ids)  # type: ignore[arg-type]
    assert len(plan.items) == 3
    assert all(
        item.sample_ids == ("s0", "s1", "s2") for item in plan.items
    )  # ceil(9/3)=3, shared head slice

    capped = EvenSplitBudget(total_sample_budget=100).propose_load(0, None, prompts, sample_ids)  # type: ignore[arg-type]
    assert len(capped.items[0].sample_ids) == 20  # capped by |split|

    floor = EvenSplitBudget(total_sample_budget=1).propose_load(0, None, prompts, sample_ids)  # type: ignore[arg-type]
    assert all(len(item.sample_ids) == 1 for item in floor.items)

    budget = EvenSplitBudget(total_sample_budget=9)
    assert budget.rank() is None
    assert budget.takeover(0) is None
    assert budget.get_state() is None
    budget.set_state(None)

    with pytest.raises(ValueError, match="must be >= 1"):
        EvenSplitBudget(total_sample_budget=0)  # fail fast: the budget must be >= 1
