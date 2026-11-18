"""Deterministic score ranking (rank_by_score): permutation-invariant ties, None last.

The tie-break is the stable prompt identity (``prompt.id`` = prompt_hash); display-only
metadata (label) must not influence order, and a ``None`` (never-evaluated) score sorts
last via an explicit None check — never ``or -inf`` conflation.
"""

from __future__ import annotations

from eyas.engine.domain import Prompt, rank_by_score


def test_equal_score_ranking_is_permutation_invariant() -> None:
    a = Prompt("alpha prompt text")
    b = Prompt("bravo prompt text")
    assert a.id != b.id
    scores = {a.id: 0.5, b.id: 0.5}
    score_of = scores.get
    forward = rank_by_score([a, b], score_of)
    backward = rank_by_score([b, a], score_of)
    assert forward == backward
    # winner is fully determined by the prompt-id tie-break, not insertion order
    assert forward[0].id == min(a.id, b.id)
    assert [p.id for p in forward] == sorted([a.id, b.id])


def test_none_score_sorts_last() -> None:
    a = Prompt("never evaluated")
    b = Prompt("evaluated prompt")
    scores: dict[str, float | None] = {b.id: 0.25}
    forward = rank_by_score([a, b], scores.get)
    backward = rank_by_score([b, a], scores.get)
    assert forward == backward
    assert [p.id for p in forward] == [b.id, a.id]


def test_explicit_none_beats_negative_infinity_value() -> None:
    # a real score of -inf is still a score: it must outrank a None (unevaluated)
    real = Prompt("scored negative infinity")
    none = Prompt("scored nothing")
    scores: dict[str, float | None] = {real.id: float("-inf")}
    ranked = rank_by_score([none, real], scores.get)
    assert [p.id for p in ranked] == [real.id, none.id]


def test_scores_are_ordered_descending() -> None:
    a = Prompt("prompt a")
    b = Prompt("prompt b")
    c = Prompt("prompt c")
    scores = {a.id: 0.1, b.id: 0.9, c.id: 0.5}
    assert [p.id for p in rank_by_score([a, b, c], scores.get)] == [b.id, c.id, a.id]


def test_label_metadata_does_not_affect_order() -> None:
    a = Prompt("same-ish text one", metadata={"label": "zzz"})
    b = Prompt("same-ish text two", metadata={"label": "aaa"})
    scores = {a.id: 1.0, b.id: 1.0}
    assert rank_by_score([a, b], scores.get)[0].id == min(a.id, b.id)
