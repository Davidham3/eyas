"""Evaluation layer: pluggable evaluation semantics plus per-sample durability.

"Absolute scoring" and "relative comparison" are peer ``EvalModel``s sharing
one orchestration (same planner, same cost accounting, same bump); only the
scoring kernel differs. The absolute path is cache-first; the comparator
schedules fresh pairwise games through ``evaluator.planned_backend`` and never
gets a bare backend, so comparison calls keep trace / cost / watchdog coverage.
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ChatResponse,
    InferenceBackend,
    SamplingConfig,
    user_text,
)
from eyas.backends.scheduling import gather_bounded
from eyas.engine.budget import LoadPlan, WorkItem
from eyas.engine.capability import ABSOLUTE_EVAL, COMPARE_EVAL
from eyas.engine.domain import Cost, Evaluation, Prompt, SampleOutcome
from eyas.engine.judges import DefaultPairwiseJudge, PairwiseJudge
from eyas.engine.planner import PlannedBackend, PlannedRequest, Rejected, RequestPlanner
from eyas.judges.base import Judge, JudgeContext
from eyas.logging import get_logger
from eyas.tasks.base import Sample, Task, Variant
from eyas.tasks.protocol import VariantProtocol
from eyas.watchdog import bump

__all__ = [
    "AbsoluteEvalModel",
    "CompareEvalModel",
    "EvalModel",
    "Evaluator",
    "InMemoryRecordStore",
    "RecordStore",
    "SqliteRecordStore",
]

_log = get_logger(__name__)

# Record-store schema version. The key is split-qualified (prompt, sample, split)
# so a train-split record can never satisfy a test-split lookup, and the
# prediction column is type-tagged ("s:" str / "j:" JSON / NULL for None) so a
# free-text answer that merely looks like JSON round-trips as a str. Stores
# from an older schema are refused at open — no silent compat read.
SCHEMA_VERSION = 3


def _usage_pair(usage: Mapping[str, Any] | None) -> tuple[int, int]:
    """usage dict → (prompt_tokens, completion_tokens); missing shapes count as 0."""
    u = usage or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


@runtime_checkable
class RecordStore(Protocol):
    """Durable per-sample observation store: cache hits backfill the
    prediction as well. The key is split-qualified — a record stored under one
    split is invisible to lookups under another (no cross-split poisoning)."""

    def get(self, prompt_id: str, sample_id: str, split: str) -> SampleOutcome | None: ...

    def put(self, outcome: SampleOutcome, prompt_id: str, split: str) -> None: ...


class InMemoryRecordStore:
    """In-process implementation (contract tests / demos); the run path uses
    the sqlite store carrying a prediction column instead."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str], SampleOutcome] = {}

    def get(self, prompt_id: str, sample_id: str, split: str) -> SampleOutcome | None:
        return self._data.get((prompt_id, sample_id, split))

    def put(self, outcome: SampleOutcome, prompt_id: str, split: str) -> None:
        self._data[(prompt_id, outcome.sample_id, split)] = outcome

    def __len__(self) -> int:
        return len(self._data)


def _encode_prediction(prediction: Any) -> str | None:
    """Type-tag the prediction column: ``"s:" + text`` for str, ``"j:" + JSON``
    for non-str (KIE-style list[dict] cannot bind to sqlite directly), NULL for
    None. The tag — never a content guess — decides the read-side type, so a
    free-text answer that looks like JSON round-trips byte-identical."""
    if prediction is None:
        return None
    if isinstance(prediction, str):
        return "s:" + prediction
    return "j:" + json.dumps(prediction, ensure_ascii=False)


def _decode_prediction(encoded: str | None) -> Any:
    """Decode by type tag; an unknown tag (or invalid JSON under ``j:``) is a
    corrupt row — fail fast, no heuristic fallback."""
    if encoded is None:
        return None
    if encoded.startswith("s:"):
        return encoded[2:]
    if encoded.startswith("j:"):
        try:
            return json.loads(encoded[2:])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"record store prediction column is corrupt: 'j:' tag with invalid"
                f" JSON ({exc}). Re-run the experiment (no compat read)."
            ) from exc
    raise RuntimeError(
        f"record store prediction column is corrupt: unknown type tag in"
        f" {encoded[:16]!r}. Re-run the experiment (no compat read)."
    )


class SqliteRecordStore:
    """Durable per-sample observations: prediction/correct/error/scores
    stored as columns.

    Key design point: prediction is a first-class column (not only an aggregate
    metric) and is backfilled on cache hits too, so methods like protegi's
    gradient read per-sample errors directly.

    Schema stamp 3 (``meta.schema=3``): the outcomes key is split-qualified —
    ``PRIMARY KEY (namespace, prompt_id, sample_id, split)`` — so a record
    stored under one split can never satisfy a lookup under another — and the
    prediction column is type-tagged (``s:`` str / ``j:`` JSON / NULL), so a
    str answer that looks like JSON is never silently decoded into dict/list
    (resume byte-fidelity). The version is stamped at open; a store without
    that stamp (or stamped with an older version) is refused with an explicit
    RuntimeError — the framework decision is fail fast, no silent compat read.
    A brand-new database (no tables at all) initializes cleanly.
    """

    # Both parameters are keyword-only by contract: positionally, a path would
    # silently bind to ``namespace`` (SqliteRecordStore(Path(...)) → a str() of
    # the path as the namespace, store silently in-memory). TypeError at the
    # call site beats that.
    def __init__(self, *, namespace: str = "default", path: Path | None = None) -> None:
        self.namespace = namespace
        # lookup counters (observability surface: the DEBUG report reads them)
        self.hits = 0
        self.misses = 0
        target = str(path) if path is not None else ":memory:"
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(target, timeout=30.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(f"failed to open record store: {target} ({exc})") from exc
        self._validate_or_init_schema(target)

    def _validate_or_init_schema(self, target: str) -> None:
        """Version-gate the store at open (schema 3 is the only readable shape):
        no tables → initialize it and stamp it; a stamp other than 3, or an outcomes
        table missing its split column → fail fast. There is deliberately no
        migration path: older rows carry untagged predictions (and the oldest
        key was not split-qualified), so they are not trustworthy under the new
        semantics."""
        tables = {
            row[0]
            for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not tables:
            # brand-new database (no tables at all): a clean initialization
            self._conn.execute(
                "CREATE TABLE outcomes ("
                "namespace TEXT NOT NULL, prompt_id TEXT NOT NULL, sample_id TEXT NOT NULL,"
                " split TEXT NOT NULL DEFAULT '',"
                " prediction TEXT, correct INTEGER, scores_json TEXT NOT NULL, error TEXT,"
                " PRIMARY KEY (namespace, prompt_id, sample_id, split))"
            )
            self._conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema', ?)", (str(SCHEMA_VERSION),)
            )
            self._conn.commit()
            return
        if "meta" not in tables:
            raise RuntimeError(
                f"record store schema version mismatch at {target}: the meta table is"
                f" missing (unversioned store), expected schema {SCHEMA_VERSION}."
                " The run dir predates split-qualified records; re-run the experiment"
                " (no compat read)."
            )
        stamped = self._conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        stamped_version = int(stamped[0]) if stamped is not None else None
        if stamped_version != SCHEMA_VERSION:
            raise RuntimeError(
                f"record store schema version mismatch at {target}: "
                f"found {stamped_version if stamped_version is not None else 'none (unversioned)'},"
                f" expected {SCHEMA_VERSION}. The run dir predates the current record"
                " schema (split-qualified keys, type-tagged predictions); re-run the"
                " experiment (no compat read)."
            )
        if "outcomes" not in tables:
            raise RuntimeError(
                f"record store schema version mismatch at {target}: the outcomes table is"
                f" missing (found schema stamp {SCHEMA_VERSION}). The run dir predates"
                " split-qualified records; re-run the experiment (no compat read)."
            )
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(outcomes)")}
        if "split" not in columns:
            raise RuntimeError(
                f"record store schema version mismatch at {target}: the outcomes table"
                f" lacks its split column (an older 2-key table under schema stamp"
                f" {SCHEMA_VERSION}). The run dir predates split-qualified records;"
                " re-run the experiment (no compat read)."
            )

    def get(self, prompt_id: str, sample_id: str, split: str) -> SampleOutcome | None:
        row = self._conn.execute(
            "SELECT prediction, correct, scores_json, error FROM outcomes"
            " WHERE namespace=? AND prompt_id=? AND sample_id=? AND split=?",
            (self.namespace, prompt_id, sample_id, split),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        prediction, correct, scores_json, error = row
        return SampleOutcome(
            sample_id=sample_id,
            prediction=_decode_prediction(prediction),
            scores=json.loads(scores_json),
            correct=None if correct is None else bool(correct),
            error=error,
        )

    def put(self, outcome: SampleOutcome, prompt_id: str, split: str) -> None:
        correct = None if outcome.correct is None else int(outcome.correct)
        self._conn.execute(
            "INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.namespace,
                prompt_id,
                outcome.sample_id,
                split,
                _encode_prediction(outcome.prediction),
                correct,
                json.dumps(dict(outcome.scores), ensure_ascii=False),
                outcome.error,
            ),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM outcomes WHERE namespace=?", (self.namespace,)
            ).fetchone()[0]
        )

    def flush(self) -> None:
        """Explicit durability barrier + DEBUG breadcrumb (put already commits;
        the explicit commit makes ``flush`` independently durable)."""
        self._conn.commit()
        _log.debug("[store] flush namespace=%s records=%d", self.namespace, len(self))

    def close(self) -> None:
        _log.debug("[store] close namespace=%s records=%d", self.namespace, len(self))
        self._conn.close()


class EvalModel(ABC):
    """Evaluation semantics extension point: declares covered semantics +
    produced metrics; optionally overrides compare (relative semantics)."""

    name: str

    @abstractmethod
    def produces(self) -> frozenset[str]: ...

    @abstractmethod
    def semantics(self) -> frozenset[str]: ...

    async def compare(
        self,
        prompts: Sequence[Prompt],
        samples: Sequence[Sample],
        evaluator: Evaluator,
        split: str,
        round_idx: int,
    ) -> list[Evaluation]:
        raise NotImplementedError(f"{type(self).__name__} does not support compare semantics")


class AbsoluteEvalModel(EvalModel):
    """Absolute scoring: per-sample scores → protocol aggregation → cross-sample
    mean. produces is determined by the task metrics."""

    name = "absolute"

    def __init__(self, produces: frozenset[str]) -> None:
        self._produces = produces

    def produces(self) -> frozenset[str]:
        return self._produces

    def semantics(self) -> frozenset[str]:
        return frozenset({ABSOLUTE_EVAL})


@dataclass
class _ComparePass:
    """Mutable state threaded through the phases of one ``compare`` run:
    schedule → generate → tally → account."""

    requests: list[ChatRequest] = field(default_factory=list)
    meta: list[tuple[int, int, str]] = field(default_factory=list)
    responses: list[ChatResponse] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    games: list[int] = field(default_factory=list)
    # per-(prompt index, sample id) game outcomes: win 1.0 / tie 0.5 / loss 0.0
    sample_outcomes: dict[tuple[int, str], list[float]] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0


class CompareEvalModel(EvalModel):
    """Relative comparison: round-robin pairwise comparison over
    the candidate set → relative win-rate score.

    Goes through ``evaluator.planned_backend``: every pairwise game passes the
    planner chokepoint + cost accounting + watchdog bump. produces only
    compare_score — which keeps it off the accuracy leaderboard (assembly
    validation).
    """

    name = "compare"

    def __init__(self, pairwise_judge: PairwiseJudge | None = None) -> None:
        self.judge = pairwise_judge if pairwise_judge is not None else DefaultPairwiseJudge()

    def produces(self) -> frozenset[str]:
        return frozenset({"compare_score"})

    def semantics(self) -> frozenset[str]:
        return frozenset({COMPARE_EVAL})

    async def compare(
        self,
        prompts: Sequence[Prompt],
        samples: Sequence[Sample],
        evaluator: Evaluator,
        split: str,
        round_idx: int,
    ) -> list[Evaluation]:
        n = len(prompts)
        if n < 2:
            return _uncontested_evaluations(prompts, split, self.name, round_idx)
        stage = _ComparePass(scores=[0.0] * n, games=[0] * n)
        self._schedule_games(stage, evaluator, prompts, samples)
        _log.debug(
            "[compare] start split=%s pairs=%d prompts=%d samples=%d requests=%d",
            split,
            n * (n - 1),
            n,
            len(samples),
            len(stage.requests),
        )
        bump(f"[compare] start split={split} requests={len(stage.requests)}")
        t0 = time.monotonic()
        stage.responses = await evaluator.planned_backend.agenerate(
            stage.requests, evaluator.sampling
        )
        bump(f"[compare] done split={split} games={len(stage.requests)}")
        if len(stage.responses) != len(stage.requests):
            raise RuntimeError("compare backend returned a mismatched response count")
        self._tally_games(stage)
        _account_games(stage, t0)
        return _compare_evaluations(stage, prompts, samples, split, self.name, round_idx)

    def _schedule_games(
        self,
        stage: _ComparePass,
        evaluator: Evaluator,
        prompts: Sequence[Prompt],
        samples: Sequence[Sample],
    ) -> None:
        """Round-robin scheduling: every ordered pair of candidates plays
        every sample."""
        for i in range(len(prompts)):
            for j in range(len(prompts)):
                if i == j:
                    continue
                for sample in samples:
                    stage.requests.append(
                        self._pair_request(evaluator, prompts[i], prompts[j], sample)
                    )
                    stage.meta.append((i, j, sample.id))

    def _tally_games(self, stage: _ComparePass) -> None:
        """Scoring: parse each game's verdict into the running win tallies and
        the per-(prompt, sample) outcome lists (win 1.0 / tie 0.5 / loss 0.0)."""
        for (i, j, sample_id), response in zip(stage.meta, stage.responses, strict=True):
            choice = self.judge.parse(response.text)
            stage.games[i] += 1
            stage.games[j] += 1
            if choice == "A":
                outcome_i, outcome_j = 1.0, 0.0
            elif choice == "B":
                outcome_i, outcome_j = 0.0, 1.0
            else:
                outcome_i, outcome_j = 0.5, 0.5
            stage.scores[i] += outcome_i
            stage.scores[j] += outcome_j
            stage.sample_outcomes.setdefault((i, sample_id), []).append(outcome_i)
            stage.sample_outcomes.setdefault((j, sample_id), []).append(outcome_j)

    def _pair_request(
        self, evaluator: Evaluator, a: Prompt, b: Prompt, sample: Sample
    ) -> ChatRequest:
        question = user_text(evaluator.task.build_request(a.text, sample, Variant("identity")))
        return self.judge.build_request(question, a.text, b.text)


def _uncontested_evaluations(
    prompts: Sequence[Prompt], split: str, eval_model: str, round_idx: int
) -> list[Evaluation]:
    """Fewer than two candidates: nothing to compare, every prompt gets the
    neutral 0.5 with no games played."""
    return [
        Evaluation(
            prompt_id=p.id,
            split=split,
            eval_model=eval_model,
            scores={"compare_score": 0.5},
            per_sample=(),
            cost=Cost(),
            round_idx=round_idx,
        )
        for p in prompts
    ]


def _account_games(stage: _ComparePass, t0: float) -> None:
    """Cost accounting: sum the game-channel usage and close the wall-clock
    span of one ``compare`` run."""
    for response in stage.responses:
        p_in, p_out = _usage_pair(response.usage)
        stage.input_tokens += p_in
        stage.output_tokens += p_out
    stage.wall_seconds = time.monotonic() - t0


def _compare_evaluations(
    stage: _ComparePass,
    prompts: Sequence[Prompt],
    samples: Sequence[Sample],
    split: str,
    eval_model: str,
    round_idx: int,
) -> list[Evaluation]:
    """One Evaluation per prompt: win-rate score and the REAL per-sample view
    (mean game outcome of that prompt on that sample, always in [0,1]). The
    whole-batch game-channel cost is carried ONCE — only the first Evaluation
    bills it, the rest carry ``Cost()`` — so round-level cost summing counts
    every game exactly once (no n-fold inflation)."""
    batch_cost = Cost(
        requests=len(stage.requests),
        wall_seconds=stage.wall_seconds,
        input_tokens=stage.input_tokens,
        output_tokens=stage.output_tokens,
    )
    evaluations: list[Evaluation] = []
    for i, p in enumerate(prompts):
        per_sample: list[SampleOutcome] = []
        for s in samples:
            # a contested run plays every (prompt, sample) — a missing key or an
            # empty list is a scheduling/tally bug: fail fast, no neutral guess
            outcomes = stage.sample_outcomes[(i, s.id)]
            if not outcomes:
                raise RuntimeError(f"compare tally has no games for prompt {p.id} on sample {s.id}")
            per_sample.append(
                SampleOutcome(s.id, None, {"compare_score": sum(outcomes) / len(outcomes)})
            )
        evaluations.append(
            Evaluation(
                prompt_id=p.id,
                split=split,
                eval_model=eval_model,
                scores={"compare_score": stage.scores[i] / max(1, stage.games[i])},
                per_sample=tuple(per_sample),
                cost=batch_cost if i == 0 else Cost(),
                round_idx=round_idx,
            )
        )
    return evaluations


@dataclass
class _EvalPass:
    """Mutable state threaded through the phases of one ``_evaluate_absolute``
    run: collect → generate → judge → group → aggregate."""

    requests: list[ChatRequest] = field(default_factory=list)
    responses: list[ChatResponse] = field(default_factory=list)
    meta: list[tuple[Prompt, Sample, Variant, int, ChatRequest]] = field(default_factory=list)
    # cache keys are split-qualified (prompt, sample, split): a record under
    # one split must never satisfy a lookup under another
    hit_aggregated: dict[tuple[str, str, str], SampleOutcome] = field(default_factory=dict)
    rejected: set[tuple[str, str, str]] = field(default_factory=set)
    # first owning WorkItem index per (prompt, sample, split) key: the batch is
    # deduped before dispatch, and every pair is billed ONCE — to this owner
    owner_by_key: dict[tuple[str, str, str], int] = field(default_factory=dict)
    # billing is per WorkItem (not per prompt): two items sharing a prompt must
    # not each receive the whole prompt's bill
    requests_by_item: dict[int, int] = field(default_factory=dict)
    tokens_by_item: dict[int, tuple[int, int]] = field(default_factory=dict)
    grouped: dict[tuple[str, str, str], list[tuple[int, SampleOutcome]]] = field(
        default_factory=dict
    )
    # judge channel billed per WorkItem: [requests, input_tokens, output_tokens]
    judge_by_item: dict[int, list[int]] = field(default_factory=dict)


@dataclass
class Evaluator:
    """Orchestrator: cache-first + planner + cost accounting + durable
    observations + watchdog bump.

    Methods evaluate only through it and never get a bare backend — bypassing
    the evaluator to run inference directly is structurally impossible.
    """

    task: Task
    protocol: VariantProtocol
    backend: InferenceBackend
    planner: RequestPlanner
    model: EvalModel
    judge: Judge | None = None
    store: RecordStore = field(default_factory=InMemoryRecordStore)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    judge_concurrency: int = 64
    _planned_backend: PlannedBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # backend slot for code paths that hold a backend rather than the
        # evaluator (judges via JudgeContext, pairwise games via the compare
        # model): every request they send still passes the planner
        self._planned_backend = PlannedBackend(self.backend, self.planner)

    @property
    def planned_backend(self) -> PlannedBackend:
        return self._planned_backend

    async def aevaluate(
        self,
        prompts: Sequence[Prompt],
        load: LoadPlan,
        split: str = "train",
        round_idx: int = 0,
    ) -> list[Evaluation]:
        if COMPARE_EVAL in self.model.semantics():
            samples = list(self.task.load_split(split))
            return await self.model.compare(prompts, samples, self, split, round_idx)
        return await self._evaluate_absolute(load, split, round_idx)

    async def agenerate_prompts(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig | None = None
    ) -> list[str]:
        """Generation facility for LLM-backed propose: also goes through the
        planner (resource chokepoint) + backend + bump.

        Requests rejected by the planner return an empty string (the caller
        treats it as a parse failure); the rest of the batch is unaffected.
        """
        sampling = sampling or self.sampling
        idx_map: list[int] = []
        planned_reqs: list[ChatRequest] = []
        for i, request in enumerate(requests):
            planned = self.planner.plan(request)
            if isinstance(planned, PlannedRequest):
                planned_reqs.append(planned.request)
                idx_map.append(i)
        texts = [""] * len(requests)
        if planned_reqs:
            responses = await self.backend.agenerate(planned_reqs, sampling)
            bump(f"generate requests={len(planned_reqs)}")
            for i, response in zip(idx_map, responses, strict=True):
                texts[i] = response.text
        return texts

    async def _evaluate_absolute(
        self, load: LoadPlan, split: str, round_idx: int
    ) -> list[Evaluation]:
        samples_by_id = {s.id: s for s in self.task.load_split(split)}
        objective_keys = {obj.key for obj in self.task.objectives}
        judge_ctx = JudgeContext(
            # judging rides the planned backend
            backend=self.planned_backend,
            # the judging channel is deliberately PINNED greedy: a
            # verdict that varied with the run's seed/thinking setting would make
            # relative scoring itself irreproducible. Candidate generation and
            # evaluation rendering take the deployment layer; judging does not.
            sampling=SamplingConfig(temperature=0.0, max_new_tokens=4096),
        )
        acc = _EvalPass()
        _log.debug("[eval] start split=%s items=%d", split, len(load.items))
        bump(f"[eval] start split={split} items={len(load.items)}")
        t0 = time.monotonic()

        n_keys = self._collect_requests(load, split, samples_by_id, acc)
        _log.debug(
            "[eval] cache hit/miss hit=%d miss=%d requests=%d",
            len(acc.hit_aggregated),
            n_keys - len(acc.hit_aggregated),
            len(acc.requests),
        )

        await self._generate_and_account(acc, split, t0)
        # Judging fan-out runs concurrently: backend-backed judges (mcq_llm
        # etc.) issue their LLM calls in parallel — serial judging would drown
        # wall time; gather_bounded returns in order, so the aggregation-side
        # zip stays position-aligned.
        scored = await gather_bounded(
            (
                partial(self._score_one, prompt, sample, response, judge_ctx, variant, request)
                for (prompt, sample, variant, _v_idx, request), response in zip(
                    acc.meta, acc.responses, strict=True
                )
            ),
            self.judge_concurrency,
        )
        self._group_scored(acc, scored, split)

        wall = time.monotonic() - t0
        return [
            self._evaluation_for_item(
                item_idx, item, split, round_idx, samples_by_id, objective_keys, wall, acc
            )
            for item_idx, item in enumerate(load.items)
        ]

    def _collect_requests(
        self, load: LoadPlan, split: str, samples_by_id: dict[str, Sample], acc: _EvalPass
    ) -> int:
        """Cache-first request building: a store hit short-circuits the whole
        sample, everything else fans out over the protocol variants through the
        planner. (prompt, sample, split) keys are deduped across the WHOLE batch
        before dispatch — first occurrence wins (candidate-first order intact),
        and the first owning WorkItem is the one billed for the pair. Returns
        the unique key count (hit-rate log)."""
        seen: set[tuple[str, str, str]] = set()
        for item_idx, item in enumerate(load.items):
            ids = item.sample_ids or tuple(samples_by_id)
            for sample_id in ids:
                key = (item.prompt.id, sample_id, split)
                if key in seen:
                    # in-batch twin (within one item or across items sharing the
                    # prompt): the first occurrence already dispatched / hit /
                    # rejected this key — the store only fills after generation,
                    # so a second store.get would miss and double-dispatch
                    continue
                seen.add(key)
                acc.owner_by_key[key] = item_idx
                if (hit := self.store.get(*key)) is not None:
                    acc.hit_aggregated[key] = hit
                    continue
                sample = samples_by_id[sample_id]
                for v_idx, variant in enumerate(self.protocol.variants(sample)):
                    request = self.task.build_request(item.prompt.text, sample, variant)
                    planned = self.planner.plan(request)
                    if isinstance(planned, Rejected):
                        acc.rejected.add(key)
                        continue
                    assert isinstance(planned, PlannedRequest)
                    acc.requests.append(planned.request)
                    acc.meta.append((item.prompt, sample, variant, v_idx, planned.request))
                    acc.requests_by_item[item_idx] = acc.requests_by_item.get(item_idx, 0) + 1
        return len(seen)

    async def _generate_and_account(self, acc: _EvalPass, split: str, t0: float) -> None:
        """One backend pass over the collected requests + per-WorkItem usage
        accounting; fills acc.responses (nothing to do when all was cache-hit)."""
        if not acc.requests:
            return
        acc.responses = await self.backend.agenerate(acc.requests, self.sampling)
        bump(
            f"[eval] generated split={split} requests={len(acc.requests)} "
            f"elapsed={time.monotonic() - t0:.2f}s"
        )
        _log.debug(
            "[eval] generated split=%s requests=%d elapsed=%.2fs",
            split,
            len(acc.requests),
            time.monotonic() - t0,
        )
        if len(acc.responses) != len(acc.requests):
            raise RuntimeError(
                f"backend response count mismatch: sent {len(acc.requests)} got {len(acc.responses)}"
            )
        for (prompt, sample, _v, _i, _r), response in zip(acc.meta, acc.responses, strict=True):
            # a dispatched key always has an owner (KeyError = collect bug, fail fast)
            owner = acc.owner_by_key[(prompt.id, sample.id, split)]
            prev = acc.tokens_by_item.get(owner, (0, 0))
            p_in, p_out = _usage_pair(response.usage)
            acc.tokens_by_item[owner] = (prev[0] + p_in, prev[1] + p_out)

    def _group_scored(
        self,
        acc: _EvalPass,
        scored: Sequence[tuple[SampleOutcome, Mapping[str, Any] | None]],
        split: str,
    ) -> None:
        for (prompt, sample, _variant, v_idx, _request), (outcome, judge_usage) in zip(
            acc.meta, scored, strict=True
        ):
            acc.grouped.setdefault((prompt.id, sample.id, split), []).append((v_idx, outcome))
            if judge_usage is not None:  # a dict (even empty) = one billed request
                owner = acc.owner_by_key[(prompt.id, sample.id, split)]
                bill = acc.judge_by_item.setdefault(owner, [0, 0, 0])
                bill[0] += 1
                j_in, j_out = _usage_pair(judge_usage)
                bill[1] += j_in
                bill[2] += j_out

    def _evaluation_for_item(
        self,
        item_idx: int,
        item: WorkItem,
        split: str,
        round_idx: int,
        samples_by_id: dict[str, Sample],
        objective_keys: set[str],
        wall: float,
        acc: _EvalPass,
    ) -> Evaluation:
        """Item-level aggregate: cross-sample mean over the per-sample outcomes,
        cost totals include the judge channel (true inference bill); the
        judge_* fields keep the split visible for reporting. Billing is per
        WorkItem: only pairs this item owns (first occurrence in the batch) are
        charged here — a pair shared with a later item is billed once, to this
        item; the later item still scores it from the shared outcome."""
        outcomes = [
            self._outcome_for_sample(item, sample_id, split, acc)
            for sample_id in (item.sample_ids or tuple(samples_by_id))
        ]
        n = len(outcomes) or 1
        keys = objective_keys | {k for o in outcomes for k in o.scores}
        scores = {k: sum(o.scores.get(k, 0.0) for o in outcomes) / n for k in keys}
        tok = acc.tokens_by_item.get(item_idx, (0, 0))
        j_req, j_in, j_out = acc.judge_by_item.get(item_idx, [0, 0, 0])
        return Evaluation(
            prompt_id=item.prompt.id,
            split=split,
            eval_model=self.model.name,
            scores=scores,
            per_sample=tuple(outcomes),
            cost=Cost(
                requests=acc.requests_by_item.get(item_idx, 0) + j_req,
                wall_seconds=wall,
                input_tokens=tok[0] + j_in,
                output_tokens=tok[1] + j_out,
                judge_requests=j_req,
                judge_input_tokens=j_in,
                judge_output_tokens=j_out,
            ),
            round_idx=round_idx,
        )

    def _outcome_for_sample(
        self, item: WorkItem, sample_id: str, split: str, acc: _EvalPass
    ) -> SampleOutcome:
        """Per-sample outcome for one (prompt, sample, split) key: a store hit
        is taken as-is, a planner-rejected sample is marked parse_fail, and a
        freshly scored one has its variant scores protocol-aggregated + persisted."""
        key = (item.prompt.id, sample_id, split)
        if key in acc.hit_aggregated:
            return acc.hit_aggregated[key]
        if key in acc.rejected:
            return SampleOutcome(sample_id, None, {"parse_fail": 1.0})
        variants_out = sorted(acc.grouped.get(key, []), key=itemgetter(0))
        per_variant_scores = [dict(o.scores) for _, o in variants_out]
        canonical = _canonical_outcome(variants_out)
        if canonical is None:
            canonical = SampleOutcome(sample_id, None, {})
        aggregated = self.protocol.aggregate(per_variant_scores) if per_variant_scores else {}
        durable = SampleOutcome(
            sample_id=sample_id,
            prediction=canonical.prediction,
            scores=aggregated,
            correct=canonical.correct,
            error=canonical.error,
        )
        if aggregated:
            self.store.put(durable, item.prompt.id, split)
        return durable

    async def _score_one(
        self,
        prompt: Prompt,
        sample: Sample,
        response: ChatResponse,
        judge_ctx: JudgeContext,
        variant: Variant,
        request: ChatRequest,
    ) -> tuple[SampleOutcome, Mapping[str, Any] | None]:
        """(outcome, billed judge usage). The judge channel costs one extra
        backend call per judged sample (mcq_llm), so a verdict that consumed a
        backend call must carry its usage to the cost aggregation. Convention: None = the
        verdict consumed no request (no judge / pure-logic judge); a dict —
        possibly empty when the server omits usage — = one billed request."""
        if response.finish_reason == OVER_WINDOW_FINISH:
            return SampleOutcome(sample.id, None, {"parse_fail": 1.0}), None
        prediction = self.task.parse_response(response, sample)
        judge_usage: Mapping[str, Any] | None = None
        if self.judge is not None:
            # variant / instruction / user_text are filled in per verdict by the
            # evaluator — user_text comes from the actually rendered request
            # (variant included), not rebuilt
            ctx = JudgeContext(
                backend=judge_ctx.backend,
                sampling=judge_ctx.sampling,
                variant=variant,
                instruction=prompt.text,
                user_text=user_text(request),
            )
            verdict = await self.judge.ajudge(response, prediction, sample, ctx)
            # usage rides on the verdict regardless of whether this judge
            # decided: an undecided chain link still billed its request
            judge_usage = verdict.usage
            if verdict.decided:
                pred = prediction.value
                if pred is None and isinstance(verdict.note, str) and verdict.note:
                    pred = (
                        verdict.note
                    )  # parse failed but the judge recovered it (note = matched option)
                scores = {**verdict.scores, "parse_fail": float(pred is None)}
                return SampleOutcome(sample.id, pred, scores, _correct(scores), None), judge_usage
        scores = {
            **self.task.score(prediction, sample),
            "parse_fail": float(prediction.value is None),
        }
        return SampleOutcome(
            sample.id, prediction.value, scores, _correct(scores), None
        ), judge_usage


def _canonical_outcome(variants_out: Sequence[tuple[int, SampleOutcome]]) -> SampleOutcome | None:
    """Variant 0 is the canonical row (its prediction/correct/error are the
    durable ones); a sample that did not produce variant 0 falls back to the
    lowest variant (list is variant-sorted)."""
    for v_idx, outcome in variants_out:
        if v_idx == 0:
            return outcome
    return variants_out[0][1] if variants_out else None


def _correct(scores: Mapping[str, float]) -> bool | None:
    for key in ("accuracy", "field_acc", "strict_acc"):
        if key in scores:
            return scores[key] >= 1.0
    return None
