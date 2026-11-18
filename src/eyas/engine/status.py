"""``status.json`` — the run dir's machine-readable status (schema v1).

One file answers "what is this run doing right now" for every external reader
(``eyas status``, a dashboard, an external monitor) without touching the
framework's internals. Contract:

* **Single writer** — the run loop (``run_experiment``) is the only writer,
  so no locking is needed; this module is pure functions over a payload dict
  (no threads, no module state).
* **Atomic refresh** — :func:`write_status` writes a temp file in the *same*
  directory and moves it with :func:`os.replace`, so a concurrent reader sees
  either the previous or the new document, never a torn one.
* **Write-only forward compatibility** — future schema versions may add keys;
  readers of unknown versions ignore what they do not know, so nothing here
  may rely on a reader understanding a new key.

Schema v1 (see :func:`merge_status` for the builder)::

    {"schema_version": 1, "run": {...}, "phase": str, "round": int,
     "budget": {...}, "cost": {...}, "cache": {"hits": int, "misses": int},
     "rejections": {reason: int}, "history": [{"round": int, "phase": str,
     "elapsed_sec": float}], "updated_at": iso8601}

``phase`` walks ``starting → optimizing → done``, with ``failed`` as the
terminal phase when the run dies before ``result.json`` exists (every started
run ends with either the result file or phase ``failed`` — the last phase alone
cannot tell "still running" from "dead").
``round`` is the loop's round cursor — the number of rounds whose checkpoint was
persisted, so it doubles as the next round index and the run-start payload is 0.
``cost`` mirrors result.json's ``total_cost`` keys, ``rejections`` is the
planner's tally by reason and ``cache`` the record store's hit/miss counters
(both in-process counters, so after a resume they count the
current attempt while ``cost`` stays cumulative). ``history`` keeps one entry
per refresh bounded to the last :data:`HISTORY_LIMIT`; an entry's
``elapsed_sec`` is the interval since the previous refresh — the span being
left when the entry is written, not seconds spent in the phase the entry
reports (the cumulative bill is ``cost.wall_seconds``); ``updated_at`` is
stamped by the writer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

__all__ = [
    "HISTORY_LIMIT",
    "SCHEMA_VERSION",
    "STATUS_FILENAME",
    "write_json_atomic",
    "budget_section",
    "cost_section",
    "merge_status",
    "read_status",
    "write_status",
]

SCHEMA_VERSION: Final = 1
STATUS_FILENAME: Final = "status.json"
#: A long run must not grow the file without bound; only the tail is interesting.
HISTORY_LIMIT: Final = 50

# Mirrors ``Cost.to_json`` / result.json's ``total_cost`` keys exactly — the
# status cost section is the same bill, not a second accounting.
_COST_KEYS: Final = (
    "requests",
    "input_tokens",
    "output_tokens",
    "wall_seconds",
    "judge_requests",
    "judge_input_tokens",
    "judge_output_tokens",
)


def write_json_atomic(path: Path, payload: str) -> None:
    """Same-directory temp + ``os.replace``: atomic on one filesystem, so a
    reader racing the write never observes a half-written document. Every
    durable run-dir JSON (status / checkpoint / result) goes through here —
    a torn ``checkpoint.json`` or ``result.json`` from a mid-write SIGKILL /
    OOM / watchdog ``os._exit`` would otherwise lose an otherwise-resumable run."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def write_status(run_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically (re)write ``run_dir/status.json``.

    The caller's payload is never mutated; the writer owns ``updated_at`` (every
    write is an observation) and enforces the :data:`HISTORY_LIMIT` bound.
    """
    doc = dict(payload)
    doc.setdefault("schema_version", SCHEMA_VERSION)
    doc["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    history = doc.get("history")
    if isinstance(history, list) and len(history) > HISTORY_LIMIT:
        doc["history"] = list(history[-HISTORY_LIMIT:])
    write_json_atomic(run_dir / STATUS_FILENAME, json.dumps(doc, ensure_ascii=False, indent=2))


def read_status(run_dir: Path) -> dict[str, Any] | None:
    """Read ``run_dir/status.json``; ``None`` when absent (pre-status run dir).

    A corrupt document raises :class:`json.JSONDecodeError` on purpose — a
    status file that cannot be parsed is a broken observation, not an empty one.
    """
    path = run_dir / STATUS_FILENAME
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(doc).__name__}")
    return doc


def _mapping_merge(
    default: Mapping[str, Any],
) -> Callable[[Any, Mapping[str, Any] | None], dict[str, Any]]:
    """Build the merge rule for one mapping field.

    A provided mapping **replaces wholesale**; ``None`` keeps the prior value —
    that is what keeps a resumed run from zeroing what the loop has not
    re-measured yet. *default* backs the first document of a lineage (and the
    cache counters, which must be present even before the first observation).
    """

    def rule(prior: Any, provided: Mapping[str, Any] | None) -> dict[str, Any]:
        if provided is not None:
            return dict(provided)
        return dict(prior or default)

    return rule


_EMPTY_DEFAULT: Final = _mapping_merge({})
_CACHE_DEFAULT: Final = _mapping_merge({"hits": 0, "misses": 0})

#: Per-field merge rules of schema v1 — the whole mapping-field ladder is this
#: table.
#: ``run``/``budget``/``cost``/``rejections`` start empty; the ``cache`` counters
#: start at zero hits/misses. ``phase``/``round``/``history`` are not mapping
#: fields: phase is always the writer's latest observation, ``round`` is
#: monotone (see :func:`merge_status`), and history is append-only.
_FIELD_MERGE_RULES: Final = (
    ("run", _EMPTY_DEFAULT),
    ("budget", _EMPTY_DEFAULT),
    ("cost", _EMPTY_DEFAULT),
    ("cache", _CACHE_DEFAULT),
    ("rejections", _EMPTY_DEFAULT),
)


def _carried_forward(doc: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """Unknown keys (a newer schema version, or a field this builder does not
    model) ride along; ``updated_at`` is excluded because the writer stamps it."""
    return {k: v for k, v in base.items() if k not in doc and k != "updated_at"}


def merge_status(
    prior: Mapping[str, Any] | None,
    *,
    phase: str,
    round: int,
    run: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
    cache: Mapping[str, Any] | None = None,
    rejections: Mapping[str, int] | None = None,
    elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    """Build the next schema-v1 document from the previous one.

    ``prior`` is the existing status (or ``None``): unknown keys and a history
    tail are carried forward, the known mapping fields follow their rule in
    :data:`_FIELD_MERGE_RULES` (replaced **when provided**, ``None`` keeps the
    prior value).

    ``elapsed_sec`` is the seconds since the writer's previous refresh (the
    span being left as this entry is written), as measured by the writer
    (``0.0`` when it does not keep a clock of its own).

    The ``round`` field is monotone within a status lineage: a round is only
    ever reported once its checkpoint was persisted, so a value from a shorter
    accounting (an interrupted attempt resuming from an older checkpoint) can
    never move an already-observed round backwards.
    """
    base = prior or {}
    history = [dict(entry) for entry in base.get("history") or [] if isinstance(entry, Mapping)]
    round_idx = max(int(round), int(base.get("round", 0)))
    history.append({"round": round_idx, "phase": phase, "elapsed_sec": float(elapsed_sec)})
    provided = {
        "run": run,
        "budget": budget,
        "cost": cost,
        "cache": cache,
        "rejections": rejections,
    }
    merged = {}
    for field, rule in _FIELD_MERGE_RULES:
        merged[field] = rule(base.get(field), provided[field])
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": merged["run"],
        "phase": phase,
        "round": round_idx,
        "budget": merged["budget"],
        "cost": merged["cost"],
        "cache": merged["cache"],
        "rejections": merged["rejections"],
        "history": history,
    }
    doc.update(_carried_forward(doc, base))
    return doc


def cost_section(cost: Mapping[str, Any] | None) -> dict[str, Any]:
    """Cost block from a ``Cost``-shaped mapping (zero-filled, mirrored keys)."""
    source = cost or {}
    return {key: source.get(key, 0.0 if key == "wall_seconds" else 0) for key in _COST_KEYS}


def budget_section(budget: Any, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Budget block: allocator name + its own state (opaque per allocator).

    Accepts the allocator instance or its class (the status writer only needs
    the declared ``name``); an allocator without one falls back to its class name.
    """
    name = getattr(budget, "name", None) or getattr(budget, "__name__", type(budget).__name__)
    return {"name": str(name), "state": dict(state or {})}
