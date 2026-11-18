"""Split-qualified record store (schema stamp 3) contract tests.

The record-store key is ``(prompt_id, sample_id, split)``: a train-split
outcome must never satisfy a test-split lookup (cache poisoning across
splits). Stores are version-stamped at open (``meta.schema=3``); a store
predating the current schema fails fast with an explicit schema/version-
mismatch message. No silent compat read.

The prediction column is type-tagged (``s:`` str / ``j:`` JSON / NULL for
None): a free-text str answer that merely looks like JSON must round-trip
as a str (resume byte-fidelity), never silently decoded into dict/list.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

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
from eyas.backends.scheduling import run_sync
from eyas.core.objective import Objective
from eyas.engine.budget import LoadPlan, WorkItem
from eyas.engine.capability import ResourceProfile
from eyas.engine.domain import Prompt, SampleOutcome
from eyas.engine.evaluation import (
    SCHEMA_VERSION,
    AbsoluteEvalModel,
    Evaluator,
    InMemoryRecordStore,
    SqliteRecordStore,
)
from eyas.engine.planner import RequestPlanner
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.protocol import IdentityProtocol

_LABELS = {"q1": "cat", "q2": "dog"}


def _toy_samples() -> list[Sample]:
    return [
        Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"}),
        Sample(id="q2", inputs={"question": "q2"}, references={"label": "dog"}),
    ]


class ToyMCQ(Task):
    name = "toy_mcq"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return _toy_samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _mock() -> ScriptedMockBackend:
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        return ChatResponse(text=_LABELS.get(user_text(request), ""))

    return ScriptedMockBackend(responder=responder)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SqliteRecordStore]:
    """File-backed store (real persistence, not :memory:) for round-trip pins."""
    s = SqliteRecordStore(path=tmp_path / "store.sqlite")
    try:
        yield s
    finally:
        s.close()


def test_get_is_split_isolated(tmp_path):
    """A test-split lookup must MISS a record stored under train."""
    store = SqliteRecordStore(path=tmp_path / "s.sqlite")
    store.put(SampleOutcome("q1", "cat", {"accuracy": 1.0}), "p1", "train")
    assert store.get("p1", "q1", "test") is None  # different split: no hit
    hit = store.get("p1", "q1", "train")
    assert hit is not None and hit.prediction == "cat" and hit.scores == {"accuracy": 1.0}
    assert (store.hits, store.misses) == (1, 1)  # the test-split lookup was a miss


def test_in_memory_store_is_split_isolated():
    """The in-memory store carries the same triple-key semantics."""
    store = InMemoryRecordStore()
    store.put(SampleOutcome("q1", "cat", {"accuracy": 1.0}), "p1", "train")
    assert store.get("p1", "q1", "test") is None
    assert store.get("p1", "q1", "train") is not None
    assert len(store) == 1


def test_new_empty_db_initializes_clean(tmp_path):
    """A brand-new database (no tables) opens cleanly and is stamped with the
    current schema version."""
    path = tmp_path / "fresh.sqlite"
    store = SqliteRecordStore(path=path)
    store.close()
    conn = sqlite3.connect(path)
    schema = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0]
    cols = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)")}
    conn.close()
    assert schema == str(SCHEMA_VERSION)
    assert "split" in cols
    # a valid store reopens and round-trips
    reopened = SqliteRecordStore(path=path)
    reopened.put(SampleOutcome("q1", "cat", {"accuracy": 1.0}), "p1", "train")
    assert reopened.get("p1", "q1", "train") is not None
    reopened.close()


def test_older_schema_store_fails_fast(tmp_path):
    """A store predating split qualification (3-column PK, no meta) is refused at open."""
    older = tmp_path / "older.sqlite"
    conn = sqlite3.connect(older)
    conn.execute(
        "CREATE TABLE outcomes ("
        "namespace TEXT NOT NULL, prompt_id TEXT NOT NULL, sample_id TEXT NOT NULL,"
        " prediction TEXT, correct INTEGER, scores_json TEXT NOT NULL, error TEXT,"
        " PRIMARY KEY (namespace, prompt_id, sample_id))"
    )
    conn.commit()
    conn.close()
    try:
        SqliteRecordStore(path=older)
    except RuntimeError as exc:
        msg = str(exc)
        assert "schema" in msg
        assert "split-qualified records" in msg
        assert "predates" in msg
    else:
        raise AssertionError("opening an older-schema store did not raise RuntimeError")


def test_old_schema_store_fails_fast(tmp_path):
    """A store stamped SCHEMA_VERSION-1 (split-qualified keys but untagged
    predictions) is refused at open — no compat read."""
    older = tmp_path / "old_schema.sqlite"
    conn = sqlite3.connect(older)
    conn.execute(
        "CREATE TABLE outcomes ("
        "namespace TEXT NOT NULL, prompt_id TEXT NOT NULL, sample_id TEXT NOT NULL,"
        " split TEXT NOT NULL DEFAULT '',"
        " prediction TEXT, correct INTEGER, scores_json TEXT NOT NULL, error TEXT,"
        " PRIMARY KEY (namespace, prompt_id, sample_id, split))"
    )
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema', ?)", (str(SCHEMA_VERSION - 1),))
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        SqliteRecordStore(path=older)


def test_json_looking_str_roundtrips_as_str(store):
    """A free-text answer that merely looks like JSON stays a str (resume
    byte-fidelity) — never silently decoded into a dict."""
    store.put(SampleOutcome("s1", '{"a":1}', {"acc": 1.0}), "p1", "train")
    out = store.get("p1", "s1", "train")
    assert out is not None
    assert out.prediction == '{"a":1}' and isinstance(out.prediction, str)


def test_list_prediction_roundtrips(store):
    pred = [{"field": "name", "value": "x"}]
    store.put(SampleOutcome("s1", pred, {}), "p1", "train")
    out = store.get("p1", "s1", "train")
    assert out is not None
    assert out.prediction == pred


def test_none_prediction_roundtrips_as_none(store):
    store.put(SampleOutcome("s1", None, {}), "p1", "train")
    out = store.get("p1", "s1", "train")
    assert out is not None and out.prediction is None


def test_empty_str_prediction_roundtrips(store: SqliteRecordStore) -> None:
    """Empty str is a str (tag "s:"), not None — round-trips byte-identical."""
    store.put(SampleOutcome("s1", "", {}), "p1", "train")
    out = store.get("p1", "s1", "train")
    assert out is not None
    assert out.prediction == "" and isinstance(out.prediction, str)


def test_tag_looking_str_roundtrips_as_str(store):
    """A str that itself starts like a type tag is payload, not a tag."""
    store.put(SampleOutcome("s1", 'j:{"a":1}', {"acc": 1.0}), "p1", "train")
    out = store.get("p1", "s1", "train")
    assert out is not None
    assert out.prediction == 'j:{"a":1}' and isinstance(out.prediction, str)


def test_json_looking_str_survives_reopen(tmp_path):
    """Resume shape: close the run-dir store, reopen it, the str prediction
    comes back byte-identical."""
    path = tmp_path / "reopen.sqlite"
    first = SqliteRecordStore(path=path)
    first.put(SampleOutcome("s1", "[1, 2]", {"acc": 1.0}), "p1", "train")
    first.close()
    second = SqliteRecordStore(path=path)
    try:
        out = second.get("p1", "s1", "train")
        assert out is not None
        assert out.prediction == "[1, 2]" and isinstance(out.prediction, str)
    finally:
        second.close()


def test_corrupt_prediction_tag_fails_fast(store, tmp_path):
    """An unknown type tag in the prediction column is a corrupt row: get
    raises RuntimeError instead of guessing."""
    store.put(SampleOutcome("s1", "cat", {"accuracy": 1.0}), "p1", "train")
    conn = sqlite3.connect(tmp_path / "store.sqlite")
    conn.execute("UPDATE outcomes SET prediction='x:cat'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="prediction"):
        store.get("p1", "s1", "train")


def test_evaluator_roundtrip_two_splits(tmp_path):
    """Two splits through the Evaluator: cross-split lookups stay cold."""
    store = SqliteRecordStore(path=tmp_path / "s.sqlite")
    evaluator = Evaluator(
        task=ToyMCQ(),
        protocol=IdentityProtocol(),
        backend=_mock(),
        planner=RequestPlanner(ResourceProfile(), context_window=None),
        model=AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        store=store,
    )
    prompt = Prompt("Classify the animal.")
    plan = LoadPlan(items=(WorkItem(prompt),))

    train_evals = run_sync(evaluator.aevaluate([prompt], plan, "train", 0))
    assert train_evals[0].scores["accuracy"] == 1.0
    assert (store.hits, store.misses) == (0, 2)  # cold split

    # a different split is a cold cache: the train records must NOT satisfy it
    test_evals = run_sync(evaluator.aevaluate([prompt], plan, "test", 1))
    assert test_evals[0].scores["accuracy"] == 1.0
    assert (store.hits, store.misses) == (0, 4)  # 4 misses, still 0 hits

    # replaying the train split is now a full hit
    train_again = run_sync(evaluator.aevaluate([prompt], plan, "train", 2))
    assert train_again[0].scores["accuracy"] == 1.0
    assert (store.hits, store.misses) == (2, 4)
