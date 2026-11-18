"""Domain model (no IO, pure data + in-memory containers).

Key design points:
- ``Prompt``'s sole identity is ``prompt_hash`` (identity rule): ``text``
  / ``metadata`` do not participate in ``__eq__`` / ``__hash__``, so labels can
  never leak into identity.
- ``Evaluation`` carries durable ``SampleOutcome``: per-sample predictions
  persist with the evaluation and are backfilled on cache hits too, so
  gradient/compare-style methods read per-sample errors directly instead of
  inferring them from aggregate metrics.
- Candidate containers are split into orthogonal ``Archive`` (append-only
  historical best). Selection uses a **single primary objective**
  (``objectives[0]``) — every published baseline is a single-objective
  optimizer, so no Pareto-front container is built (multi-objective search is
  not a goal of the current method set); where a method needs a position-
  aligned working set (evoprompt's DE), it lives in the method's typed state.
- ``OptimizerState`` is typed serializable state; serialization is the
  checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Archive",
    "Artifact",
    "Cost",
    "Evaluation",
    "OptimizerState",
    "Prompt",
    "SampleOutcome",
    "normalize_prompt",
    "prompt_hash",
    "rank_by_score",
]


def normalize_prompt(text: str) -> str:
    """Identity normalization: NFKC + strip leading/trailing whitespace. The
    sole identity convention; no path may bypass it."""
    return unicodedata.normalize("NFKC", text).strip()


def prompt_hash(text: str) -> str:
    """First 12 hex chars of sha256 after normalization — Prompt's sole identity."""
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Prompt:
    """A candidate prompt. Identity = ``id`` (prompt_hash); text/metadata are
    display-only and not part of equality."""

    text: str = field(compare=False)
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", prompt_hash(self.text))

    @property
    def label(self) -> str:
        return self.metadata.get("label", self.id)


def rank_by_score(
    prompts: Sequence[Prompt], score_of: Callable[[str], float | None]
) -> list[Prompt]:
    """Rank candidates by score: descending, ties broken ascending by ``prompt.id``.

    The framework-wide score-ranking convention. The tie-break on the stable
    identity (prompt_hash) makes the winner independent of insertion order —
    display-only metadata and list position never decide, so an interrupted run
    and its resume rank identically. ``score_of`` may return ``None`` for a
    never-evaluated prompt: those sort **last** through an explicit None check —
    a ``score or -inf`` conflation would let a missing score rank like a real one.
    """
    scored: list[tuple[float, str, Prompt]] = []
    missing: list[Prompt] = []
    for prompt in prompts:
        score = score_of(prompt.id)
        if score is None:
            missing.append(prompt)
        else:
            scored.append((score, prompt.id, prompt))
    scored.sort(key=lambda item: (-item[0], item[1]))
    missing.sort(key=lambda prompt: prompt.id)
    return [prompt for _, _, prompt in scored] + missing


@dataclass(frozen=True)
class Cost:
    """Resource bill of one evaluation / one run (cost profiling is a core
    selling point of the framework, so it lives in the domain model instead of
    being stitched on afterwards)."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    # Judge-channel breakdown (mcq_llm verdicts etc.): already included in the
    # totals above — these are diagnostic splits that make the judge channel's
    # extra call per judged sample visible in the cost profile. Convention: a verdict
    # is billed as one request when the backend reported usage for it (usage
    # missing shapes count as 0, same rule as the eval channel).
    judge_requests: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            judge_requests=self.judge_requests + other.judge_requests,
            judge_input_tokens=self.judge_input_tokens + other.judge_input_tokens,
            judge_output_tokens=self.judge_output_tokens + other.judge_output_tokens,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": self.wall_seconds,
            "judge_requests": self.judge_requests,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Cost:
        return cls(
            requests=int(data.get("requests", 0)),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            wall_seconds=float(data.get("wall_seconds", 0.0)),
            judge_requests=int(data.get("judge_requests", 0)),
            judge_input_tokens=int(data.get("judge_input_tokens", 0)),
            judge_output_tokens=int(data.get("judge_output_tokens", 0)),
        )


@dataclass(frozen=True)
class SampleOutcome:
    """Per-sample durable observation. correct=None means the judge did
    not decide (not "wrong")."""

    sample_id: str
    prediction: Any  # task parse output kept as-is (KIE is list[dict[str, str]], MCQ is str)
    scores: Mapping[str, float]
    correct: bool | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prediction": self.prediction,
            "scores": dict(self.scores),
            "correct": self.correct,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SampleOutcome:
        return cls(
            sample_id=str(data["sample_id"]),
            prediction=data.get("prediction"),
            scores=dict(data.get("scores", {})),
            correct=data.get("correct"),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class Evaluation:
    """Immutable result of one candidate in one evaluation. ``per_sample`` is
    durable."""

    prompt_id: str
    split: str
    eval_model: str
    scores: Mapping[str, float]
    per_sample: tuple[SampleOutcome, ...]
    cost: Cost
    round_idx: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "split": self.split,
            "eval_model": self.eval_model,
            "scores": dict(self.scores),
            "per_sample": [o.to_json() for o in self.per_sample],
            "cost": self.cost.to_json(),
            "round_idx": self.round_idx,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Evaluation:
        return cls(
            prompt_id=str(data["prompt_id"]),
            split=str(data.get("split", "train")),
            eval_model=str(data.get("eval_model", "absolute")),
            scores=dict(data.get("scores", {})),
            per_sample=tuple(SampleOutcome.from_json(o) for o in data.get("per_sample", [])),
            cost=Cost.from_json(data.get("cost", {})),
            round_idx=int(data.get("round_idx", 0)),
        )


@dataclass(frozen=True)
class Artifact:
    """Cross-method artifact (§9): typed, content-addressed. Replaces the
    implicit coupling of methods "reading each other's json paths" directly.

    Address = ``producer.kind`` (e.g. ``ape.candidate_prompts``); ``digest`` is
    the content hash of the payload, for sweep DAG awareness and idempotent
    persistence to disk.
    """

    kind: str
    payload: Any
    producer: str = ""

    def address(self) -> str:
        return f"{self.producer}.{self.kind}" if self.producer else self.kind

    @property
    def digest(self) -> str:
        canon = json.dumps(self.payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "producer": self.producer, "payload": self.payload}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Artifact:
        """Rebuild from ``to_json``'s shape (the persisted-artifact reload path).

        Persistence serializes with ``default=str``: JSON-native payloads (the
        artifact convention — lists of strings etc.) round-trip exactly; exotic
        payload members may come back as their ``str()`` form."""
        return cls(
            kind=str(data["kind"]),
            payload=data.get("payload"),
            producer=str(data.get("producer", "")),
        )


class Archive:
    """append-only history: prompt_id → the candidate's historical best
    Evaluation.

    Carries opro's unboundedly growing pool and hill_climb's reuse of evaluated
    candidates; ``score_of`` injects the ranking convention (objective
    orientation), decoupling the container from "which objective counts as
    best".
    """

    def __init__(self, score_of: Callable[[Evaluation], float]) -> None:
        self._score_of = score_of
        self._best: dict[str, Evaluation] = {}
        self._history: list[Evaluation] = []

    def add(self, evaluation: Evaluation) -> None:
        """Keep the candidate's best Evaluation: strictly higher score wins; on
        a score tie the **larger per-sample evidence set** wins (a subset round
        must not lock a prompt's reported entry below its full-split record —
        spo's closing pass overwrites its vote-subset record); equal score and
        equal size keep the first writer (deterministic, no insertion-order
        drift under resume)."""
        self._history.append(evaluation)
        current = self._best.get(evaluation.prompt_id)
        if current is None:
            self._best[evaluation.prompt_id] = evaluation
            return
        new_score = self._score_of(evaluation)
        old_score = self._score_of(current)
        if new_score > old_score or (
            new_score == old_score and len(evaluation.per_sample) > len(current.per_sample)
        ):
            self._best[evaluation.prompt_id] = evaluation

    def best(self, prompt_id: str) -> Evaluation | None:
        return self._best.get(prompt_id)

    def contains(self, prompt_id: str) -> bool:
        return prompt_id in self._best

    def ranked(self) -> list[Evaluation]:
        # same convention as rank_by_score: descending score, ties broken ascending
        # by the stable prompt identity — never by insertion order
        return sorted(self._best.values(), key=lambda e: (-self._score_of(e), e.prompt_id))

    def history(self) -> Iterator[Evaluation]:
        return iter(self._history)

    def __len__(self) -> int:
        return len(self._best)

    def __contains__(self, prompt_id: object) -> bool:
        return prompt_id in self._best


class OptimizerState(ABC):
    """Typed optimizer state: serializable = naturally checkpoint / resume."""

    @abstractmethod
    def to_json(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_json(cls, data: Mapping[str, Any]) -> OptimizerState: ...


@dataclass
class PromptListState(OptimizerState):
    """Baseline state: a candidate list + round counter. Starting point for
    raw / hill_climb."""

    candidates: list[Prompt] = field(default_factory=list)
    round_idx: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"candidates": [p.text for p in self.candidates], "round_idx": self.round_idx}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PromptListState:
        return cls(
            candidates=[Prompt(t) for t in data.get("candidates", [])],
            round_idx=int(data.get("round_idx", 0)),
        )
