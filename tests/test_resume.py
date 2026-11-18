"""Checkpoint/resume: interrupted resume decides identically to a full run (typed state + archive + budget + rng fidelity).

hill_climb is deterministic -> interrupt after the first round and resume from the checkpoint; the final best,
per-round request counts, and total requests must exactly match a single uninterrupted run.

The bottom half adds the mid-phase resume harness: every registry-discovered
strategy must finish a mid-phase-interrupted run (crash strictly after a checkpoint snapshot, inside the
in-flight round's propose/evaluate phase) with the exact final result of an uninterrupted run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import pytest
from test_contract import _LABELS, ToyMCQ, _gen_mock, _mock

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    SamplingConfig,
    TextPart,
    system_message,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.backends.scheduling import run_sync
from eyas.engine import builtin as _engine_builtin  # noqa: F401 — self-containedness
from eyas.engine import runtime as runtime_mod
from eyas.engine.artifacts import ArtifactStore
from eyas.engine.budget import BanditBudget
from eyas.engine.domain import Prompt
from eyas.engine.evaluation import AbsoluteEvalModel, SqliteRecordStore
from eyas.engine.genio import GEN_MARKER, OUT_END, OUT_START
from eyas.engine.registry import get as _engine_get
from eyas.engine.registry import registered as _engine_registered
from eyas.engine.runtime import run_experiment
from eyas.engine.sampling import DEFAULT_TIERS
from eyas.engine.strategies.ape import ApeConfig, ApeStrategy
from eyas.engine.strategies.evoprompt import EvopromptConfig, EvopromptStrategy
from eyas.engine.strategies.hill_climb import HillClimbConfig, HillClimbStrategy
from eyas.engine.strategy import OptimizerLoop, _state_prompts, build_run_context
from eyas.logging import _reset
from eyas.registry import register_task
from eyas.tasks.protocol import IdentityProtocol

_CFG = HillClimbConfig(rounds=3, keep_size=2, per_parent=3)


def _ctx_strategy():
    ctx = build_run_context(
        HillClimbStrategy,
        _CFG,
        ToyMCQ(),
        IdentityProtocol(),
        _mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=42,
    )
    return ctx, HillClimbStrategy(_CFG, ctx)


def test_resume_reproduces_full_run():
    ctx, strategy = _ctx_strategy()
    full = OptimizerLoop(ctx).run(strategy, ["Classify the animal."])
    assert full.best is not None and full.best.scores["accuracy"] == 1.0
    assert (
        len(full.history) >= 2
    )  # at least seed + one evolution round, so the interruption is meaningful

    snapshots: list[dict] = []

    def interrupt_after_first(snapshot: dict) -> None:
        snapshots.append(snapshot)
        if len(snapshots) == 1:
            raise KeyboardInterrupt

    ctx, strategy = _ctx_strategy()
    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(strategy, ["Classify the animal."], on_round=interrupt_after_first)

    ctx, strategy = _ctx_strategy()
    resumed = OptimizerLoop(ctx).run(strategy, ["Classify the animal."], resume=snapshots[0])

    assert resumed.best is not None
    assert resumed.best.prompt_id == full.best.prompt_id
    assert dict(resumed.best.scores) == dict(full.best.scores)
    assert [r.n_requests for r in resumed.history] == [r.n_requests for r in full.history]
    assert resumed.total_cost.requests == full.total_cost.requests
    # per-round invoices (full Cost incl. judge split + loop-clock seconds) survive
    # the checkpoint round-trip VERBATIM: the pre-crash round is replayed from the
    # checkpoint payload, never re-measured. Measured quantities (wall_seconds /
    # elapsed_sec) are per-attempt, so the verbatim comparison is against the
    # checkpoint payload itself — only the deterministic request counts are
    # comparable across attempts (asserted above via n_requests).
    ck = json.loads(json.dumps(snapshots[0]["history"][0]))
    assert resumed.history[0].cost.to_json() == ck["cost"]
    assert resumed.history[0].elapsed_sec == ck["elapsed_sec"]
    # the learning-curve point survives too (a resume must not amputate the
    # pre-crash part of the curve)
    assert resumed.history[0].best_so_far == ck["best_so_far"]


def test_resume_priming_never_overwrites_a_populated_store(tmp_path):
    """Resume shape for ``_prime_restart``'s no-op guard: the run
    dir's sqlite is ALREADY populated when the checkpoint is replayed, so
    priming must skip every existing row (a bare ``store.put`` would
    INSERT OR REPLACE it). The pin tampers the checkpoint's archived outcome
    to a prediction the store has never seen; if priming dropped the guard,
    the tampered value would land in the durable row. With the guard, the
    row keeps the originally stored prediction."""

    store_path = tmp_path / "cache" / "eyas.sqlite"

    def make_ctx():
        store = SqliteRecordStore(path=store_path)
        ctx = build_run_context(
            HillClimbStrategy,
            _CFG,
            ToyMCQ(),
            IdentityProtocol(),
            _mock(),
            AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
            store=store,
            seed=42,
        )
        return ctx, HillClimbStrategy(_CFG, ctx)

    # interrupted first pass: completes one evaluate round into the durable store
    ctx, strategy = make_ctx()
    snapshots: list[dict] = []

    def interrupt(snapshot: dict) -> None:
        snapshots.append(snapshot)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(strategy, ["Classify the animal."], on_round=interrupt)
    stored = ctx.evaluator.store
    record = snapshots[0]["archive"][0]
    prompt_id, outcome = record["prompt_id"], record["per_sample"][0]
    sample_id, split = outcome["sample_id"], record["split"]
    expected = stored.get(prompt_id, sample_id, split)
    assert expected is not None  # the round really persisted to the run-dir store
    stored.close()

    # the checkpoint claims a DIFFERENT prediction for that row (as a corrupted /
    # out-of-date checkpoint would): priming must not push it into the store
    outcome["prediction"] = "tampered-prediction"

    ctx2, strategy2 = make_ctx()  # fresh store instance over the same file
    try:
        OptimizerLoop(ctx2).run(strategy2, ["Classify the animal."], resume=snapshots[0])
        row = ctx2.evaluator.store.get(prompt_id, sample_id, split)
        assert row is not None
        assert row.prediction == expected.prediction
        assert row.prediction != "tampered-prediction"
    finally:
        ctx2.evaluator.store.close()


def test_checkpoint_roundtrip_preserves_budget_state():
    budget = BanditBudget(eval_rounds=3, samples_per_eval=1)
    arms = [_prompt(f"arm {i}") for i in range(3)]
    budget.propose_load(0, None, arms, ["s0", "s1", "s2"])
    state = budget.get_state()
    assert state is not None and state["round"] == 1

    restored = BanditBudget(eval_rounds=3, samples_per_eval=1)
    restored.set_state(state)
    assert restored.get_state() == state


def _prompt(text: str):
    from eyas.engine.domain import Prompt

    return Prompt(text)


def _hc_ctx_strategy(rounds: int):
    # two mutation suffixes both carry "step by step" (the mock's correctness switch),
    # so the two offspring tie at the top score 1.0 while the seed scores 0.0
    cfg = HillClimbConfig(
        rounds=rounds,
        keep_size=1,
        per_parent=2,
        mutations=("Think step by step.", "Please think step by step."),
    )
    ctx = build_run_context(
        HillClimbStrategy,
        cfg,
        ToyMCQ(),
        IdentityProtocol(),
        _mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=42,
    )
    return ctx, HillClimbStrategy(cfg, ctx)


def test_resume_score_tie_picks_same_best():
    """Score-tied candidates decide identically across checkpoint/resume.

    The seed scores 0.0 (no "step by step" switch in the mock) while both mutation
    offspring score 1.0 — a top-score tie between distinct prompts. The tie winner
    must be fixed by the prompt-id tie-break (rank_by_score), not by insertion
    order — the checkpoint persists candidate texts in stored order
    (``_state_prompts`` order == the serialized list, so restore order is pinned by
    stored texts), and the resumed run must finish with the same best as the
    uninterrupted one.
    """
    seeds = ["Classify the animal."]

    ctx, strategy = _hc_ctx_strategy(rounds=1)
    full = OptimizerLoop(ctx).run(strategy, seeds)
    assert full.best is not None
    assert full.best.scores["accuracy"] == 1.0
    tied_ids = [e.prompt_id for e in full.ranked if e.scores["accuracy"] == 1.0]
    assert len(tied_ids) == 2  # the fixture really ties
    # the tie-break pins the winner to the smallest prompt_hash among the tied set
    assert full.best.prompt_id == min(tied_ids)

    snapshots: list[dict] = []

    def interrupt(snapshot: dict) -> None:
        snapshots.append(snapshot)
        if len(snapshots) == 2:  # after the round that evaluated both tied offspring
            raise KeyboardInterrupt

    ctx, strategy = _hc_ctx_strategy(rounds=1)
    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(strategy, seeds, on_round=interrupt)

    # restoration fidelity check: stored texts order pins the candidate order
    restored = _state_prompts(HillClimbStrategy.state_cls.from_json(snapshots[-1]["state"]))
    assert [p.text for p in restored] == snapshots[-1]["state"]["candidates"]

    ctx, strategy = _hc_ctx_strategy(rounds=1)
    resumed = OptimizerLoop(ctx).run(strategy, seeds, resume=snapshots[-1])
    assert resumed.best is not None
    assert resumed.best.prompt_id == full.best.prompt_id


# ---------------------------------------------------------------------------
# Mid-phase resume harness (shared by the method-level suites)
# ---------------------------------------------------------------------------

#: The four generation tiers (generate/evolve/gradient/synonym) share this
#: max_new_tokens; the eval (1024) and compare (16) tiers never collide with it,
#: so the mock can classify a served request as generation-backed from its
#: effective sampling alone.
_GEN_TIER_MAX_TOKENS = DEFAULT_TIERS["generate"].max_new_tokens


class _FaultedBackend(ScriptedMockBackend):
    """ScriptedMockBackend raising ``KeyboardInterrupt`` **before** serving the
    ``fault_at_call``-th request — the stand-in for a mid-round crash.

    A real SIGKILL loses the in-flight round and everything after the last
    checkpoint: nothing in memory survives. ``KeyboardInterrupt`` propagates
    through ``run_sync`` the same way without killing the test process, so the
    same fixture can resume right after (the same trick the checkpoint-snapshot
    interruption above uses). Only served requests reach the recordings, so a
    faulted batch counts as never issued.
    """

    def __init__(self, responder: Any, fault_at_call: int | None) -> None:
        super().__init__(responder=responder)
        self.fault_at_call = fault_at_call
        self.recorded_tiers: list[str] = []  # per served request: "generate" | "other"

    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        if self.fault_at_call is not None and self.call_count + len(requests) >= self.fault_at_call:
            raise KeyboardInterrupt(f"synthetic crash before request {self.fault_at_call}")
        out = await super().agenerate(requests, sampling)
        tier = "generate" if sampling.max_new_tokens == _GEN_TIER_MAX_TOKENS else "other"
        self.recorded_tiers.extend([tier] * len(requests))
        return out


def _spo_varying_mock() -> ScriptedMockBackend:
    """spo toy mock with per-round fresh instructions and always-A votes.

    A uniform toy mock makes every spo round generate the same offspring, so
    the loop retires its challenger early (challenger == incumbent → done) and
    the multi-round items/vote/cursor trajectory never runs — the divergence a
    dropped state field causes would then have nothing to diverge IN.
    ``Variant {n}`` bodies key on the request's ``[Round n]`` stamp (the reply
    is a deterministic function of the prompt, like a deterministic engine, and
    it survives a backend restart: a resumed run re-derives the stamped prompt
    and gets the reference run's reply), so every round's challenger stays
    distinct and the loop stays alive. The vote channel answers A (position
    bias): under the ABBA group that is exactly 2 wins / 2 losses per pair → a
    TIE → the incumbent never advances, and the reply carries no generation
    marker (a marker-bearing reply would classify the compare-tier request as
    generation-backed in the harness's stream split).
    """

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_text = "".join(
            p.text for m in request.messages if m.role == "system" for p in m.content
        )
        if sys_text.startswith("You compare two answers"):
            return ChatResponse(text="A")
        if GEN_MARKER in sys_text:
            m = re.search(r"\[Round (\d+)\]", sys_text)
            k = int(m.group(1)) if m else 0
            return ChatResponse(text=f"{OUT_START}Variant {k}: think step by step. {OUT_END}")
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    return ScriptedMockBackend(responder=responder)


def _fixture_for(
    name: str,
) -> tuple[
    type[Any],
    Any,
    Callable[[], ScriptedMockBackend],
    Any,
    list[str],
    int,
    bool,
    str | None,
    set[str],
]:
    """Registry-discovered strategy class + default config + toy mock factory +
    eval model + canonical seeds + checkpoint count to interrupt at + whether
    its propose phase is generation-backed (generation-replay check applies) +
    budget kind + the checkpoint ``state`` key-set the serialization must keep
    emitting (durable guard against silently weakened serialization).

    Configs stay at the small end yet keep >=2 evaluate rounds where the method
    has a multi-round loop, so a mid-phase interruption is meaningful.
    Generation-backed methods share the toy classification fixture whose
    determinism switches make the LLM channels reproducible ("step by step"
    answers correct, GEN_MARKER requests generate it).
    """
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    config_cls = _engine_get("strategy", name).config_cls
    seeds = ["Classify the animal."]
    mock: Callable[[], ScriptedMockBackend] = _mock
    # the checkpoint whose end-of-round snapshot the harness interrupts after;
    # single-evaluation methods (raw / cot / ape — ape generates its whole pool
    # in init, before the loop's first durable write) have exactly one
    n_checkpoints = 1
    gen_replay = False
    budget_kind: str | None = None
    # the key-set the strategy's state_cls.to_json emits today — the harness
    # asserts every captured checkpoint still carries exactly these keys, so a
    # field silently dropped from (or a strategy serializing only {}) never
    # passes unnoticed
    state_keys: set[str] = {"candidates", "round_idx"}  # PromptListState (raw/cot/hill_climb)
    if name == "hill_climb":
        config: Any = config_cls(rounds=3, keep_size=2, per_parent=3)
        n_checkpoints = 2
    elif name == "ape":
        config = config_cls(num_candidates=3, demos_per_prompt=2)
        mock = _gen_mock
    elif name == "opro":
        config = config_cls(rounds=2, num_generated=3)
        mock = _gen_mock
        n_checkpoints, gen_replay = 2, True
        state_keys = {"candidates", "history", "round_idx"}
    elif name == "evoprompt":
        # rounds=2: the seed round truncates the pool, round 1 generates+evaluates
        # the DE trials (checkpoint #2 is the post-generation phase=eval_trials
        # snapshot carrying trials/cur_best), and the crash lands at the head of
        # round 2's generation phase
        config = config_cls(popsize=2, rounds=2, mode="de")
        mock = _gen_mock
        seeds = ["Classify the animal one-shot.", "Classify the animal two-shot."]
        n_checkpoints, gen_replay = 2, True
        state_keys = {
            "population",
            "trials",
            "cur_best",
            "cur_best_score",
            "round_idx",
            "phase",
        }
    elif name == "protegi":
        # PROTEGI is beam search over bandit-scored offspring — the toy config
        # mirrors the published shape (N=2 error groups per expansion) and keeps
        # two rounds so the serialized offspring/seen fields re-enter a later expand
        # and become observable
        config = config_cls(beam_size=2, rounds=2, n_gradients=2)
        mock = _gen_mock
        n_checkpoints, gen_replay, budget_kind = 2, True, "bandit"
        state_keys = {"beam", "offspring", "arm_order", "seen", "round_idx", "phase"}
    elif name == "spo":
        # the OvO challenger loop: seed-eval + two voting items-rounds
        # checkpoints; the closing full-split pass is skipped on the 2-sample
        # toy (every items window already covers the split). Interrupting after
        # #2 lands at the head of the re-vote batch (mid-phase, voting).
        config = config_cls(rounds=2, num_pairs=2, votes_per_pair=4)
        mock = _spo_varying_mock
        seeds = ["Answer now.", "Answer later."]
        n_checkpoints, gen_replay = 2, True
        state_keys = {
            "candidates",
            "round_idx",
            "phase",
            "incumbent",
            "challenger",
            "cursor",
            "schedule",
            "gens_served",
            "eval_cursor",
        }
    else:  # raw / cot: degenerate single-round methods, config carries defaults
        config = config_cls()
    return (
        _engine_get("strategy", name),
        config,
        mock,
        eval_model,
        seeds,
        n_checkpoints,
        gen_replay,
        budget_kind,
        state_keys,
    )


def _config_kwargs(cfg: Any) -> dict[str, Any]:
    """Dataclass instance -> kwargs dict (fixture config -> mergeable defaults)."""
    if not is_dataclass(cfg):  # pragma: no cover - all toy configs are dataclasses
        raise TypeError(f"config {cfg!r} is not a dataclass")
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


def resume_equivalence(
    strategy_name: str,
    config: dict[str, Any],
    seeds: list[str],
    checkpoint_at_step: int,
    *,
    fault_at_call: int | None = None,
    check_generation_replay: bool = False,
    budget_kind: str | None = None,
    assert_after_resume: Callable[[dict[str, Any], Any], None] | None = None,
) -> dict[str, Any]:
    """Mid-phase resume fidelity harness: a crash strictly after checkpoint
    ``checkpoint_at_step`` must not change the final result.

    Parameter precedence: the fixture (``_fixture_for``) supplies the toy
    environment — mock, eval model, generation-replay switch, budget kind,
    canonical checkpoint count — and the **default config knobs**; the
    caller's ``config`` dict is authoritative for the keys it provides
    (merged over the fixture defaults), and the caller's ``seeds`` and
    ``checkpoint_at_step`` are honored verbatim.

    Procedure:

    1. uninterrupted reference run on a fresh fixture -> expected outcome;
    2. fault pinning: one probe run stopped (at the snapshot callback) exactly
       at snapshot ``checkpoint_at_step`` yields the number of requests issued
       up to and including the evaluate batch whose completion wrote that
       snapshot; the interrupted run then crashes on the **next** request
       (``fault_at_call`` overrides). The fault aborts the whole next batch
       before serving any request, so the crash lands strictly after the
       checkpoint at the **head of the in-flight phase** — the round's
       propose-generation (inside ``step``) when the phase starts with
       generation, otherwise the first request of the evaluate batch — never
       inside a served batch. If the reference run issued no request after
       that snapshot (single-evaluation methods — the checkpoint IS the end
       of work), the interruption falls back to the snapshot callback itself,
       the durable moment the existing round-boundary tests interrupt at;
    3. interrupted run on a fresh fixture; its last snapshot goes through a
       JSON round trip first — the real ``checkpoint.json`` persistence
       path, so a field that cannot survive JSON fails here rather than being
       papered over by passing the in-memory dict;
    4. resumed run on a fresh fixture with ``resume=`` the round-tripped
       snapshot.

    Asserted: the checkpoint's ``state`` carries exactly the fixture-declared
    key-set (durable serialization-completeness guard — a field silently
    dropped from ``to_json``, or a strategy serializing ``{}``, fails here);
    best prompt_hash, best scores, round count, per-round request counts,
    total requests, the full ranking (ids + scores), and the per-tier request
    ledger (``cost_by_tier``) equal the uninterrupted run. For a backend fault,
    additionally: every request issued
    AFTER the target snapshot (the round in flight at crash time — its work
    never reached the archive) replays in the resumed stream in order, by full
    request text + message part counts; requests before the snapshot need no
    replay (the archive covers them — re-issuing would be the bug). With
    ``check_generation_replay`` the generation-tier requests issued after the
    snapshot must replay **byte-identically at the head** of the resumed
    generation stream: a meta-prompt/gradient context rebuilt from a state
    field that did not survive the checkpoint (missing history, dropped
    offspring/errors) changes those request texts and is caught here, not only
    through the final score.

    What "mid-phase" means concretely: ``OptimizerLoop`` writes a checkpoint
    **only** right after a completed evaluate round (the ``on_round`` snapshot
    the runtime persists as ``checkpoint.json``) — there is no durable write
    inside a round, and that is precisely the crash window the harness targets. A
    propose decision's state mutation (new offspring appended to state, phase
    advance) is not checkpointed before the following evaluate round finishes,
    so a crash in between loses the in-memory mutation exactly as a real
    process death does. This harness therefore interrupts strictly after a
    round-end checkpoint, at the head of the next round's in-flight phase
    (propose-generation or the evaluate batch's first request), and demands
    the resumed run redo that phase faithfully — the mutation check proves the
    assertions bind (deleting one serialized state field turns this harness red).
    """
    (
        strategy_cls,
        fixture_config,
        mock_factory,
        eval_model,
        _,
        _,
        gen_replay_default,
        _,
        state_keys,
    ) = _fixture_for(strategy_name)
    # precedence: fixture supplies the toy env (mock / eval model / generation
    # replay / checkpoint count) and the default config knobs; the caller's
    # config dict is authoritative for the keys it provides
    cfg = strategy_cls.config_cls(**{**_config_kwargs(fixture_config), **config})

    def make_budget() -> Any:
        if budget_kind == "bandit":
            return BanditBudget(eval_rounds=2, samples_per_eval=1)
        return None

    def fresh(fault: int | None) -> tuple[Any, Any, _FaultedBackend]:
        backend = _FaultedBackend(responder=_toy_responder(mock_factory), fault_at_call=fault)
        ctx = build_run_context(
            strategy_cls,
            cfg,
            ToyMCQ(),
            IdentityProtocol(),
            backend,
            eval_model,
            budget=make_budget(),
            seed=42,
        )
        return ctx, strategy_cls(cfg, ctx), backend

    def outcome(result: Any) -> dict[str, Any]:
        assert result.best is not None, "run produced no best evaluation"
        return {
            "best_hash": result.best.prompt_id,
            "best_scores": dict(result.best.scores),
            "rounds": len(result.history),
            "per_round_requests": [r.n_requests for r in result.history],
            "total_requests": result.total_cost.requests,
            "ranking": [(e.prompt_id, dict(e.scores)) for e in result.ranked],
            "cost_by_tier": dict(result.cost_by_tier),
        }

    ctx, strategy, full_backend = fresh(None)
    full = outcome(OptimizerLoop(ctx).run(strategy, list(seeds)))

    # pin the mid-phase fault: stop a run at the target snapshot, count the
    # requests issued up to (and including) the batch whose completion wrote it
    probes: list[dict[str, Any]] = []

    class _ProbeStop(Exception):
        pass

    def probe(snapshot: dict[str, Any]) -> None:
        probes.append(snapshot)
        if len(probes) == checkpoint_at_step:
            raise _ProbeStop

    try:
        ctx, strategy, probe_backend = fresh(None)
        OptimizerLoop(ctx).run(strategy, list(seeds), on_round=probe)
    except _ProbeStop:
        pass
    assert len(probes) == checkpoint_at_step, (
        f"{strategy_name}: only {len(probes)} checkpoints exist, "
        f"asked to interrupt after #{checkpoint_at_step}"
    )
    issued_at_snapshot = probe_backend.call_count
    snapshot_interrupt = issued_at_snapshot >= full_backend.call_count
    fault = (
        None
        if snapshot_interrupt
        else (fault_at_call if fault_at_call is not None else issued_at_snapshot + 1)
    )
    if snapshot_interrupt:
        assert fault_at_call is None, (
            f"{strategy_name}: no request is issued after checkpoint "
            f"#{checkpoint_at_step}; a backend fault at call {fault_at_call} "
            "cannot land mid-phase — interrupt at the snapshot instead"
        )

    snapshots: list[dict[str, Any]] = []

    def capture(snapshot: dict[str, Any]) -> None:
        snapshots.append(snapshot)
        if snapshot_interrupt and len(snapshots) == checkpoint_at_step:
            raise KeyboardInterrupt("synthetic crash at the last durable write")

    ctx, strategy, backend = fresh(fault)
    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(strategy, list(seeds), on_round=capture)
    assert len(snapshots) == checkpoint_at_step, (
        f"{strategy_name}: fault at request {fault} must leave exactly "
        f"{checkpoint_at_step} checkpoints, saw {len(snapshots)}"
    )

    # real persistence path: the checkpoint must survive the JSON round trip
    checkpoint = json.loads(json.dumps(snapshots[-1]))

    # durable serialization guard: the captured checkpoint's state must carry
    # exactly the fixture-declared key-set (the transient mutation-check proof,
    # now pinned here)
    assert set(checkpoint["state"]) == state_keys, (
        f"{strategy_name}: checkpoint state keys {sorted(checkpoint['state'])} != "
        f"fixture-declared serialization contract {sorted(state_keys)}"
    )

    ctx, strategy, resumed_backend = fresh(None)
    resumed = OptimizerLoop(ctx).run(strategy, list(seeds), resume=checkpoint)
    got = outcome(resumed)

    assert got == full

    if not snapshot_interrupt:
        # only the in-flight round's requests (issued after the last checkpoint,
        # never archived) must replay; pre-checkpoint work is archive-covered and
        # re-issuing it would itself be a fidelity bug
        in_flight = backend.recorded_requests[issued_at_snapshot:]
        it = iter(resumed_backend.recorded_requests)
        missing = [fp for fp in in_flight if not any(fp == seen for seen in it)]
        assert not missing, (
            f"{strategy_name}: {len(missing)} in-flight interrupted requests never "
            "replayed in the resumed run (not a supersequence)"
        )
        if check_generation_replay or gen_replay_default:
            gen_in_flight = [
                fp
                for fp, tier in zip(
                    in_flight, backend.recorded_tiers[issued_at_snapshot:], strict=True
                )
                if tier == "generate"
            ]
            gen_resumed = [
                fp
                for fp, tier in zip(
                    resumed_backend.recorded_requests,
                    resumed_backend.recorded_tiers,
                    strict=True,
                )
                if tier == "generate"
            ]
            assert gen_resumed[: len(gen_in_flight)] == gen_in_flight, (
                f"{strategy_name}: generation requests replayed differently after "
                "resume — propose context was rebuilt from incomplete state"
            )

    if assert_after_resume is not None:
        assert_after_resume(checkpoint, resumed_backend)

    return {"full": full, "resumed": got, "checkpoint": snapshots[-1]}


def _toy_responder(mock_factory: Callable[[], ScriptedMockBackend]) -> Any:
    """Pull the responder out of a fresh toy mock (both toy mocks are
    responder-mode, so this reuses their exact determinism switches)."""
    responder = mock_factory()._responder  # noqa: SLF001 - harness deliberately reuses the toy switch
    assert responder is not None
    return responder


def _harness_args(strategy_name: str) -> tuple[dict[str, Any], list[str], int, str | None]:
    """Config kwargs + canonical seeds + checkpoint count + budget kind for the fixture."""
    _, cfg, _, _, seeds, n_checkpoints, _, budget_kind, _ = _fixture_for(strategy_name)
    return _config_kwargs(cfg), list(seeds), n_checkpoints, budget_kind


_STRATEGY_NAMES = _engine_registered("strategy")

# Every registry strategy must round-trip its full state key-set through
# to_json/from_json; the harness asserts that completeness, so every entry runs
# unmarked.
_PARAM = list(_STRATEGY_NAMES)


@pytest.mark.parametrize("strategy_name", _PARAM)
def test_mid_phase_resume_equivalence(strategy_name: str) -> None:
    """Every faithful strategy: crash mid-round strictly after the fixture's
    checkpoint -> resumed final result identical to the uninterrupted run.

    Round-trip invariant: every registry strategy's ``to_json``/``from_json``
    preserves its full state key-set (the harness asserts the checkpoint's
    ``state`` key-set exactly), so no strategy needs an xfail marker here.

    For the multi-round methods ``checkpoint_at_step=2`` is the mid-phase point:
    checkpoint #2 covers the first generation round's evaluation, so the crash
    lands strictly after the checkpoint at the head of the next round's
    in-flight phase — the propose-generation (the round's offspring are being
    built inside ``step``, i.e. exist only in the in-memory state and in no
    checkpoint) or the first request of the evaluate batch (the fault aborts
    the whole batch before any request is served, never inside a served batch).
    For the single-evaluation methods (raw / cot / ape — ape's
    whole candidate pool is generated in ``init``, before the loop's first
    durable write) the round-end checkpoint is the only durable moment that
    exists; the harness interrupts at the snapshot there and asserts the
    checkpoint + resume machinery holds at the only reachable phase.
    """
    cfg_kwargs, seeds, n_checkpoints, budget_kind = _harness_args(strategy_name)
    resume_equivalence(
        strategy_name,
        cfg_kwargs,
        seeds,
        checkpoint_at_step=n_checkpoints,
        budget_kind=budget_kind,
    )


def test_scripted_mock_backend_recording_shape() -> None:
    backend = ScriptedMockBackend(responder=lambda request, sampling, idx: ChatResponse(text="x"))
    req = ChatRequest(
        messages=(
            system_message("sys text"),
            user_message(ImagePart("q1.png"), TextPart("question"), ImagePart("q2.png")),
        )
    )
    run_sync(backend.agenerate([req], SamplingConfig()))
    text, part_counts = backend.recorded_requests[0]
    assert "sys text" in text
    assert "<image:q1.png>" in text and "<image:q2.png>" in text
    assert "question" in text
    assert part_counts == (1, 3)


def test_scripted_mock_backend_recording_distinguishes_part_splits() -> None:
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        return ChatResponse(text="x")

    a = ScriptedMockBackend(responder=responder)
    b = ScriptedMockBackend(responder=responder)
    split = ChatRequest(messages=(user_message(TextPart("a"), TextPart("b")),))
    joined = ChatRequest(messages=(user_message(TextPart("ab")),))
    run_sync(a.agenerate([split], SamplingConfig()))
    run_sync(b.agenerate([joined], SamplingConfig()))
    assert a.recorded_requests[0][0] == b.recorded_requests[0][0]  # text render collides...
    assert a.recorded_requests[0][1] != b.recorded_requests[0][1]  # ...the part counts do not


@pytest.mark.parametrize("strategy_name", _PARAM)
def test_resume_never_reissues_an_archived_eval_request(strategy_name: str) -> None:
    """Pin eval-store ↔ generation-store consistency: every archived
    outcome was originally produced by the backend, and ``_prime_restart``
    mirrors those outcomes into the record store, so the RESUMED stream must
    contain ZERO re-issued evaluate requests.

    The proof rebuilds each archived (prompt, sample, split) eval request with
    the same construction the Evaluator uses (task.build_request over the
    protocol's variants, prompt text from the checkpoint's texts map) and
    requires every rebuilt fingerprint to be ABSENT from the resumed backend's
    recorded stream. The invariant is framework-wide — whether or not a loop
    re-asks an archived prompt, the primed store must swallow it so it never
    reaches the backend. Sensitivity profile (mutation-verified): deleting
    ``_prime_restart`` fails this pin on SPO, the method whose pair-window and
    closing passes deliberately re-read archived windows; the other loops do
    not re-request archived prompts by construction, so for them the pin holds
    through the loop structure as much as through priming. The mid-phase
    equivalence harness separately pins the resumed values equal the reference
    run's, so absence + value identity together pin the two-store contract."""
    from eyas.engine.domain import Evaluation
    from eyas.tasks.base import Variant  # noqa: F401  (variant instances come from the protocol)

    cfg_kwargs, seeds, n_checkpoints, budget_kind = _harness_args(strategy_name)

    def archived_eval_requests_never_reissue(
        checkpoint: dict[str, Any], resumed_backend: Any
    ) -> None:
        task = ToyMCQ()
        protocol = IdentityProtocol()
        texts: dict[str, str] = checkpoint.get("texts", {})
        fingerprints: set[tuple[str, tuple[int, ...]]] = set()
        for item in checkpoint["archive"]:
            evaluation = Evaluation.from_json(item)
            text = texts.get(evaluation.prompt_id)
            assert text is not None, f"{strategy_name}: archived prompt lacks its text"
            samples = {s.id: s for s in task.load_split(evaluation.split)}
            for outcome in evaluation.per_sample:
                sample = samples[outcome.sample_id]
                for variant in protocol.variants(sample):
                    request = task.build_request(text, sample, variant)
                    fingerprints.add(
                        (
                            ScriptedMockBackend.render_request_text(request),
                            tuple(len(m.content) for m in request.messages),
                        )
                    )
        assert fingerprints, f"{strategy_name}: checkpoint carried no reconstructable eval request"
        reissued = fingerprints.intersection(resumed_backend.recorded_requests)
        assert not reissued, (
            f"{strategy_name}: {len(reissued)}/{len(fingerprints)} archived eval "
            "requests were RE-ISSUED on the resumed run — the record store failed to "
            "short-circuit them (eval-store/generation-store drift)"
        )

    resume_equivalence(
        strategy_name,
        cfg_kwargs,
        seeds,
        checkpoint_at_step=n_checkpoints,
        budget_kind=budget_kind,
        assert_after_resume=archived_eval_requests_never_reissue,
    )


# ---------------------------------------------------------------------------
# Checkpoint completeness: per-tier accounting + persisted artifacts (Y12)
# ---------------------------------------------------------------------------


def test_mid_phase_resume_preserves_cost_by_tier() -> None:
    """Per-tier request accounting survives a mid-phase crash: the resumed
    run's ``cost_by_tier`` equals the uninterrupted twin's (the harness compares
    it inside ``outcome``), and the checkpoint itself carries the tier ledger
    the restore reads. opro is generation-backed, so the pre-crash rounds hold
    a non-empty ledger a dropped restore would visibly amputate."""
    cfg_kwargs, seeds, n_checkpoints, budget_kind = _harness_args("opro")
    out = resume_equivalence(
        "opro",
        cfg_kwargs,
        seeds,
        checkpoint_at_step=n_checkpoints,
        budget_kind=budget_kind,
    )
    assert out["full"]["cost_by_tier"], "fixture must exercise generation tiers"
    assert out["checkpoint"]["tier_requests"], "checkpoint must carry the tier ledger"


def test_resume_reloads_artifacts_for_downstream_consumer(tmp_path) -> None:
    """An artifact-producing run interrupted at its checkpoint, then resumed on
    a FRESH store bound to the same directory: the persisted artifacts reload
    (manifest/get work again), the resumed result still reports them, and a
    downstream evoprompt soft-consumes ``ape.candidate_prompts`` after the
    restart — the cross-method handoff must not depend on the producer's
    process memory."""
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    art_dir = tmp_path / "artifacts"
    ape_cfg = ApeConfig(num_candidates=3, demos_per_prompt=2)
    seeds = ["Classify the animal."]

    store = ArtifactStore(path=art_dir)
    ctx = build_run_context(
        ApeStrategy,
        ape_cfg,
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        eval_model,
        artifacts=store,
        seed=42,
    )
    snapshots: list[dict[str, Any]] = []

    def interrupt(snapshot: dict[str, Any]) -> None:
        snapshots.append(snapshot)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        OptimizerLoop(ctx).run(ApeStrategy(ape_cfg, ctx), seeds, on_round=interrupt)
    manifest = dict(store.manifest())
    assert manifest, "ape must have emitted its artifact before the checkpoint"
    checkpoint = json.loads(json.dumps(snapshots[0]))
    assert checkpoint["artifacts"] == manifest  # the snapshot freezes the manifest

    # resume on a fresh process-shaped store: only the persisted files exist
    resumed_store = ArtifactStore(path=art_dir)
    resumed_store.load()
    assert resumed_store.manifest() == manifest
    assert "ape.candidate_prompts" in resumed_store
    ctx2 = build_run_context(
        ApeStrategy,
        ape_cfg,
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        eval_model,
        artifacts=resumed_store,
        seed=42,
    )
    resumed = OptimizerLoop(ctx2).run(ApeStrategy(ape_cfg, ctx2), seeds, resume=checkpoint)
    assert resumed.artifacts == manifest

    # downstream soft-consume after the restart (evoprompt init reads the store)
    evo_store = ArtifactStore(path=art_dir)
    evo_store.load()
    evo_cfg = EvopromptConfig(popsize=4, rounds=1)
    evo_ctx = build_run_context(
        EvopromptStrategy,
        evo_cfg,
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        eval_model,
        artifacts=evo_store,
        seed=42,
    )
    state = EvopromptStrategy(evo_cfg, evo_ctx).init([Prompt("Classify the animal one-shot.")])
    ape_texts = set(evo_store.get("ape.candidate_prompts").payload)
    assert {p.text for p in state.population} & ape_texts


@register_task("toy_resume_mcq")
class _ResumeToyMCQ(ToyMCQ):
    """Registry-visible twin of the contract toy task — the runtime resume
    tests assemble from a config dict, which resolves tasks by name."""


def _ape_runtime_cfg() -> dict[str, Any]:
    return {
        "run": {"name": "ape_artifact_resume", "seed": 42},
        "task": {"name": "toy_resume_mcq"},
        "protocol": {"name": "identity"},
        "backend": {"name": "mock", "params": {"responder": _toy_responder(_gen_mock)}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {"name": "ape", "params": {"num_candidates": 3, "demos_per_prompt": 2}},
    }


@pytest.fixture
def _isolated_logging():
    """run_experiment wires framework logging into module-global state; reset
    around the runtime-level tests (same shape as test_runtime)."""
    _reset()
    yield
    _reset()


def test_runtime_resume_reloads_artifacts_and_tier_ledger(
    tmp_path, monkeypatch, _isolated_logging
) -> None:
    """The runtime wiring pin: a run crashed right after its checkpoint write
    resumes with the tier ledger restored AND the persisted artifacts reloaded
    before assembly — the resumed result reports the same manifest and the same
    cost_by_tier the checkpoint froze, and a downstream evoprompt finds the
    upstream artifact through a fresh store over the run dir."""
    run_dir = tmp_path / "run"
    original = runtime_mod._StatusTracker.on_round

    def crash_after_checkpoint(self, snapshot: dict[str, Any]) -> None:
        original(self, snapshot)
        raise KeyboardInterrupt("synthetic crash right after the checkpoint write")

    with monkeypatch.context() as m:
        m.setattr(runtime_mod._StatusTracker, "on_round", crash_after_checkpoint)
        with pytest.raises(KeyboardInterrupt):
            run_experiment(_ape_runtime_cfg(), output_dir=run_dir)

    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["artifacts"], "checkpoint must freeze the artifact manifest"
    assert checkpoint["tier_requests"] == {"generate": 3}  # ape's init generation batch

    resumed = run_experiment(_ape_runtime_cfg(), resume=run_dir)
    assert resumed.artifacts == checkpoint["artifacts"]
    assert resumed.cost_by_tier == checkpoint["tier_requests"]

    store = ArtifactStore(path=run_dir / "artifacts")
    store.load()
    evo_cfg = EvopromptConfig(popsize=4, rounds=1)
    evo_ctx = build_run_context(
        EvopromptStrategy,
        evo_cfg,
        ToyMCQ(),
        IdentityProtocol(),
        _gen_mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        artifacts=store,
        seed=42,
    )
    state = EvopromptStrategy(evo_cfg, evo_ctx).init([Prompt("Classify the animal one-shot.")])
    assert {p.text for p in state.population} & set(store.get("ape.candidate_prompts").payload)


def test_runtime_resume_fails_fast_on_missing_artifact_file(tmp_path, _isolated_logging) -> None:
    """Integrity cross-check: the checkpoint froze an artifact manifest, but the
    persisted file is gone — resume must fail fast with a clear error, not
    silently continue with an amputated store a downstream consumer would read
    as 'no upstream output'."""
    run_dir = tmp_path / "run"
    run_experiment(_ape_runtime_cfg(), output_dir=run_dir)
    files = list((run_dir / "artifacts").glob("*.json"))
    assert files, "the completed ape run must have persisted its artifact"
    for file in files:
        file.unlink()
    with pytest.raises(RuntimeError, match="artifact"):
        run_experiment(_ape_runtime_cfg(), resume=run_dir)


# ---------------------------------------------------------------------------
# Resume config guard: fingerprint mismatch + checkpoint-less resume (Y13)
# ---------------------------------------------------------------------------


def test_runtime_resume_rejects_changed_non_path_config(tmp_path, _isolated_logging) -> None:
    """A resume whose config changed a non-path key must fail fast naming the
    changed top-level section — never silently continue (or restart) under a
    different experiment identity. The checkpoint freezes the fingerprint and
    the per-section companion hashes the diagnostic reads."""
    run_dir = tmp_path / "run"
    run_experiment(_ape_runtime_cfg(), output_dir=run_dir)
    ckpt_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert isinstance(checkpoint["config_fingerprint"], str)
    assert {"run", "task", "backend", "strategy"} <= set(checkpoint["config_sections"])

    changed = _ape_runtime_cfg()
    changed["run"]["seed"] = 99
    before = ckpt_path.read_bytes()
    with pytest.raises(RuntimeError, match=r"sections: run"):
        run_experiment(changed, resume=run_dir)
    assert ckpt_path.read_bytes() == before  # the guard fired before any write

    changed_strategy = _ape_runtime_cfg()
    changed_strategy["strategy"]["params"]["num_candidates"] = 5
    with pytest.raises(RuntimeError, match=r"sections: strategy"):
        run_experiment(changed_strategy, resume=run_dir)

    # Union logic pin: a top-level section present on only ONE side (added
    # now / removed now) is a changed section too, named in the diagnostic.
    added = _ape_runtime_cfg()
    added["extra_section"] = {"flag": True}
    with pytest.raises(RuntimeError, match=r"sections: extra_section"):
        run_experiment(added, resume=run_dir)

    removed = _ape_runtime_cfg()
    del removed["protocol"]
    with pytest.raises(RuntimeError, match=r"sections: protocol"):
        run_experiment(removed, resume=run_dir)


def test_runtime_resume_without_checkpoint_errors(tmp_path, _isolated_logging) -> None:
    """``--resume`` at a dir without checkpoint.json is an error, never a silent
    fresh start inside the resume dir (no status/result/checkpoint may appear)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="checkpoint"):
        run_experiment(_ape_runtime_cfg(), resume=run_dir)
    assert not (run_dir / "result.json").exists()
    assert not (run_dir / "status.json").exists()
    assert not (run_dir / "checkpoint.json").exists()

    # a resume target that does not exist at all is the same error shape
    with pytest.raises(RuntimeError, match="checkpoint"):
        run_experiment(_ape_runtime_cfg(), resume=tmp_path / "missing")


def test_runtime_resume_ignores_path_only_keys(tmp_path, _isolated_logging) -> None:
    """run.output_dir / run.output_root / run.cache_dir say WHERE the run lives,
    not WHAT it is: changing them (the legitimate resume escape hatches — a
    relocated run dir or outputs tree, an explicitly shared record cache) must
    not trip the fingerprint guard."""
    run_dir = tmp_path / "run"
    run_experiment(_ape_runtime_cfg(), output_dir=run_dir)
    changed = _ape_runtime_cfg()
    changed["run"]["output_dir"] = str(tmp_path / "elsewhere")
    changed["run"]["cache_dir"] = str(tmp_path / "shared" / "eyas.sqlite")
    resumed = run_experiment(changed, resume=run_dir)
    assert resumed.best is not None

    # output_root only feeds fresh-run resolve_run_dir; resume takes the dir
    # from --resume, so relocating the outputs tree must not reject a resume.
    relocated = _ape_runtime_cfg()
    relocated["run"]["output_root"] = str(tmp_path / "relocated_outputs")
    resumed = run_experiment(relocated, resume=run_dir)
    assert resumed.best is not None


def test_runtime_resume_rejects_pre_fingerprint_checkpoint(tmp_path, _isolated_logging) -> None:
    """Old checkpoints without the fingerprint field cannot be verified — the
    strict stance (no tolerant resume) rejects them with explicit re-run
    guidance instead of guessing that the config matches."""
    run_dir = tmp_path / "run"
    run_experiment(_ape_runtime_cfg(), output_dir=run_dir)
    ckpt_path = run_dir / "checkpoint.json"
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert isinstance(payload["config_fingerprint"], str)
    del payload["config_fingerprint"]
    payload.pop("config_sections", None)
    ckpt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="re-run"):
        run_experiment(_ape_runtime_cfg(), resume=run_dir)
