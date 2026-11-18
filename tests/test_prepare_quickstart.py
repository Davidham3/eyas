"""prepare_quickstart_data conversion and sampling contract tests (zero network).

Covers: SROIE BIO -> entity aggregation (multi-segment merge / orphan I- / all-O),
stratified sampling (fixed-seed determinism / disjoint exclude / per_label quota /
stable order), GSM8K final-answer extraction and seeded split derivation, and the
NExT-QA option/answer resolution plus per-video sampling.
"""

from __future__ import annotations

import json
import random

import pytest

import eyas.prepare as prep  # noqa: E402


def test_bio_to_entity_merges_segments_and_orders_fields():
    words = ["ZEUS", "STREET", "12", "TOTAL", "$", "3.50", "2018/11/17", "ZEUS", "LLC"]
    tags = [
        "B_company",
        "I_company",
        "I_company",
        "O",
        "B_total",
        "I_total",
        "B_date",
        "B_company",
        "I_company",
    ]
    entity = prep.bio_to_entity(words, tags)
    # multiple segments of one field merge with spaces; field order follows SROIE_FIELDS
    assert entity == {
        "company": "ZEUS STREET 12 ZEUS LLC",
        "date": "2018/11/17",
        "total": "$ 3.50",
    }
    assert list(entity) == [
        f for f in prep.SROIE_FIELDS if f in entity
    ]  # canonical-order projection


def test_bio_to_entity_orphan_i_tag_starts_segment():
    words = ["ACME", "CORP"]
    tags = ["I_company", "I_company"]  # no preceding B-: treated as a new segment start
    assert prep.bio_to_entity(words, tags) == {"company": "ACME CORP"}


def test_bio_to_entity_all_o_returns_empty():
    assert prep.bio_to_entity(["hello", "world"], ["O", "O"]) == {}
    assert prep.bio_to_entity([], []) == {}


def test_bio_to_entity_unknown_field_and_case_ignored():
    assert prep.bio_to_entity(["x"], ["B-phone"]) == {}
    # field names are case-insensitive
    assert prep.bio_to_entity(["ACME"], ["B-COMPANY"]) == {"company": "ACME"}


def _make_pool(n_per_cell: int, groups=("cat", "dog"), labels=("A", "B")):
    pool = []
    for group in groups:
        for label in labels:
            for i in range(n_per_cell):
                pool.append(
                    {
                        "filename": f"{group}_{label}_{i:02d}.jpg",
                        "group": group,
                        "label": label,
                    }
                )
    return pool


def test_stratified_take_deterministic_and_sorted():
    pool = _make_pool(5)
    taken1 = prep.stratified_take(pool, 2, random.Random(42))
    taken2 = prep.stratified_take(pool, 2, random.Random(42))
    assert taken1 == taken2  # fixed seed reproduces the exact same output
    assert [it["filename"] for it in taken1] == sorted(
        it["filename"] for it in taken1
    )  # returned sorted by filename
    assert len(taken1) == 8  # 2 groups x 2 labels x 2 items


def test_stratified_take_respects_per_label_quota():
    pool = _make_pool(5)
    taken = prep.stratified_take(pool, 3, random.Random(7))
    by_cell: dict[tuple[str, str], int] = {}
    for it in taken:
        by_cell[(it["group"], it["label"])] = by_cell.get((it["group"], it["label"]), 0) + 1
    assert set(by_cell.values()) == {3}  # exactly per_label per stratum


def test_stratified_take_exclude_keeps_val_disjoint_from_train():
    pool = _make_pool(5)
    rng = random.Random(42)
    train = prep.stratified_take(pool, 2, rng)
    val = prep.stratified_take(pool, 2, rng, exclude={it["filename"] for it in train})
    assert not {it["filename"] for it in train} & {it["filename"] for it in val}
    assert len(val) == 8


# ---------------------------------------------------------------------------
# GSM8K (numeric math QA)
# ---------------------------------------------------------------------------
def test_gsm8k_final_answer_takes_text_after_hash_marker():
    assert prep.gsm8k_final_answer("3 + 4 = 7 #### 7") == "7"


def test_derive_gsm8k_splits_disjoint_and_exact():
    rows = [{"question": f"q{i}", "answer": str(i)} for i in range(20)]
    out = prep.derive_gsm8k_splits(rows, train_n=5, val_n=3, test_n=4, seed=42)
    ids = [set(r["id"] for r in out[s]) for s in ("train", "val", "test")]
    assert [len(x) for x in ids] == [5, 3, 4]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


# ---------------------------------------------------------------------------
# NExT-QA (video MCQ)
# ---------------------------------------------------------------------------
def _nextqa_row(**overrides) -> dict:
    row = {
        "video": "v",
        "qid": "1",
        "question": "q?",
        "answer": "2",
        "option0": "a",
        "option1": "b",
        "option2": "c",
        "option3": "d",
        "option4": "e",
    }
    row.update(overrides)
    return row


def test_nextqa_answer_resolves_option_index_and_text():
    rec = prep.nextqa_rows_to_records([_nextqa_row()])[0]
    assert rec["answer"] == "c" and rec["choices"] == ["a", "b", "c", "d", "e"]
    assert prep.nextqa_rows_to_records([_nextqa_row(answer="d")])[0]["answer"] == "d"


def test_nextqa_unknown_answer_fails_fast():
    with pytest.raises(ValueError, match="answer"):
        prep.nextqa_rows_to_records([_nextqa_row(answer="z")])


def test_sample_nextqa_one_qa_per_video_deterministic():
    rows = [_nextqa_row(video=v, qid=str(i)) for v in ("v1", "v2") for i in range(3)]
    a = prep.sample_nextqa_per_video(rows, 2, 42)
    b = prep.sample_nextqa_per_video(rows, 2, 42)
    assert a == b and len({r["video"] for r in a}) == 2


def test_load_nextqa_rows_fails_fast_when_test_labels_are_withheld():
    header = "video,qid,question,answer,a0,a1,a2,a3,a4\n"

    def fetch(url: str) -> str:  # injected downloader: no network in this suite
        row = "v1,1,q?,,a,b,c,d,e\n" if url.endswith("test.csv") else "v1,1,q?,0,a,b,c,d,e\n"
        return header + row

    with pytest.raises(RuntimeError) as excinfo:
        prep.load_nextqa_rows(fetch)
    message = str(excinfo.value)
    assert "withheld" in message and "test.csv" in message


def test_load_nextqa_rows_reads_local_annotations_without_network(tmp_path):
    header = "video,qid,question,answer,a0,a1,a2,a3,a4\n"
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.csv").write_text(
            header + f"v_{split},1,q?,0,a,b,c,d,e\n", encoding="utf-8"
        )

    def fetch(url: str) -> str:  # must never run when a local directory is given
        raise AssertionError(f"unexpected download: {url}")

    rows = prep.load_nextqa_rows(fetch, annotations_dir=tmp_path)
    assert [r["split"] for r in rows] == ["train", "val", "test"]
    assert rows[0]["option0"] == "a"


def test_prepare_nextqa_samples_only_clips_the_pack_holds(tmp_path, capsys):
    """A partial clip pack (the public mirrors ship val+test clips only) must sample
    within the clips it can serve instead of failing on an unservable row."""
    header = "video,qid,question,answer,a0,a1,a2,a3,a4\n"
    ann = tmp_path / "ann"
    ann.mkdir()
    body = "".join(f"v{i},1,q{i}?,0,a,b,c,d,e\n" for i in range(6))
    for split in ("train", "val", "test"):
        (ann / f"{split}.csv").write_text(header + body, encoding="utf-8")
    pack = tmp_path / "pack"
    pack.mkdir()
    for i in range(4):
        (pack / f"v{i}.mp4").write_bytes(b"clip")

    info = prep.prepare_nextqa(
        tmp_path / "out", pack, train_n=2, val_n=1, test_n=1, seed=42, annotations=ann
    )

    assert info["stats"] == {"test": 1, "val": 1, "train": 2}
    served = set()
    for split in ("train", "val", "test"):
        rows = json.loads((tmp_path / "out" / "data" / f"{split}.json").read_text())
        served |= {row["video"] for row in rows}
    assert served <= {f"v{i}.mp4" for i in range(4)}
    assert "reference clips absent" in capsys.readouterr().out


def test_nextqa_dataset_requires_a_local_clip_pack(capsys):
    with pytest.raises(SystemExit) as excinfo:
        prep.main(["--dataset", "nextqa"])
    assert excinfo.value.code == 2  # argparse error, raised before any download
    assert "--nextqa-videos" in capsys.readouterr().err


def test_dataset_flag_is_required_no_silent_partial_build(capsys, monkeypatch):
    """No --dataset -> argparse error before any builder runs: the legacy
    `both` default silently built 2 of the 6 datasets, so a bench recipe that
    forgot the flag produced a partial tree instead of failing fast."""

    def _must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("a dataset builder ran without an explicit --dataset")

    for name in (
        "prepare_pets",
        "prepare_sroie",
        "prepare_cub",
        "prepare_textvqa",
        "prepare_gsm8k",
        "prepare_nextqa",
    ):
        monkeypatch.setattr(prep, name, _must_not_run)
    with pytest.raises(SystemExit) as excinfo:
        prep.main([])
    assert excinfo.value.code == 2  # argparse error, raised before any download
    assert "--dataset" in capsys.readouterr().err
