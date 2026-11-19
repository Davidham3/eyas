#!/usr/bin/env python3
"""Quickstart dataset preparation: Oxford-IIIT-Pets (grouped MCQ) + SROIE (KIE)
+ TextVQA (open-ended VQA) + GSM8K (numeric math QA) + NExT-QA (video MCQ) download
and conversion to the eyas format.

Outputs (default datasets/quickstart/, already gitignored)::

    pets/
    ├── data/{cat,dog}_{train,val,test}.json   # McqTask grouped format: [{"filename", "label"}]
    └── images/*.jpg
    sroie/
    ├── data/{train,val,test}.json             # KieTask format: [{"image", "entities", "id"}]
    └── images/*.jpg
    cub/
    ├── data/{family}_{train,val,test}.json    # McqTask grouped format; group = family (last word of the species name)
    └── images/*.jpg
    textvqa/
    ├── data/{train,val,test}.json             # VqaTask format: [{"image", "question", "answers", "id"}]
    └── images/*.jpg
    gsm8k/
    └── data/{train,val,test}.json             # MathQaTask format: [{"id", "question", "answer"}], answer = final number
    nextqa/
    ├── data/{train,val,test}.json             # VideoMcqTask format: [{"id", "video", "question", "choices", "answer"}]
    └── videos/*.mp4                           # only the sampled clips, never the 6.5GB source pack

Sources (--source auto falls back in order; all public mirrors):
    pets  HF ``timm/oxford-iiit-pet`` (direct -> hf-mirror) -> ModelScope ``modelscope/OxfordPets``
          (in practice integer labels only, no breed-name mapping -> degraded benchmark, last-resort fallback)
    sroie HF ``jinho8345/sroie-bio`` (direct -> hf-mirror; BIO sequences aggregated into
          single-entity records)
    cub   HF ``cassiekang/cub200_dataset`` (direct -> hf-mirror; single-pass streaming scan,
          the text column "A photo of a <species>" carries the class name; only the subset
          is persisted — the full 1.1GB is not)
    textvqa HF ``lmms-lab/TextVQA`` (direct -> hf-mirror; the labeled validation split is
          the only pool — 5001 questions x 10 human answers, embedded JPEGs; 3 parquet
          shards of ~310MB, measured ~7 min @ 2.2MB/s via hf-mirror)
    gsm8k HF ``openai/gsm8k`` config ``main`` (direct -> hf-mirror; 7473 train / 1319 test
          rows, no official val split -> val is carved out of the train split)
    nextqa GitHub ``doc-doc/NExT-QA`` annotations ``dataset/nextqa/{train,val,test}.csv``
          (fetched from raw.githubusercontent.com, or read from ``--nextqa-annotations
          <directory>`` where that host is unreachable; all three files ship the answer
          column and are pooled; --source does not apply) plus a user-supplied local clip
          pack ``--nextqa-videos <zip|directory>`` — this script never downloads the 6.5GB
          video pack, it is linked from the NExT-QA repository. Rows whose clip the pack
          lacks are excluded before sampling (the public mirrors carry the 1,570 val+test
          clips only, while the CSVs also label 3,870 train videos).

Dependencies: run with ``uv run --with "datasets>=4" --with "pyarrow>=17" python ...``
so the two reader libraries come from the environment, not the project's runtime deps;
``--source modelscope`` additionally needs ``--with modelscope``. gsm8k needs the same
two libraries; nextqa needs only the standard library (plus the local clip pack).

Sampling (fixed --seed, deterministic output):
    pets  stratified by (group, breed): train 2/breed, val 1/breed (same pool as train,
          disjoint from it), test 3/breed (official test pool); group = cat/dog (choices
          are the breeds within the group — labels within a group must be multi-valued,
          and grouping by breed would degenerate to a single option)
    cub   single-pass streaming scan (buffer only the first imgs-per-species images per
          species) -> group by the last word of the species name; keep the families_max
          richest groups with <= species-per-family species each; train/val/test split
          by a seeded shuffle within each species
    sroie shuffle the pooled upstream splits -> disjoint train/val/test slices
    textvqa image-group-aware slice of the labeled val split (~1.6 questions share one
          image): whole image-groups move into splits together, an overshooting group is
          truncated with its remainder discarded -> exact counts, zero image overlap
          across splits
    gsm8k one seeded shuffle of the official train split: the first val_n rows become
          val, the next train_n become train; test = the first test_n rows of the
          official test split (unshuffled, so it is identical across runs); ids name
          their upstream split (``train/<row>`` / ``test/<row>``)
    nextqa one seeded QA pick per video across the pooled official CSVs, then a seeded
          shuffle cut into disjoint train/val/test slices (test first, mirroring sroie);
          each video lands in exactly one split and only its sampled clip is copied out.
          The official test CSV is pooled like the other two (it ships labels), so these
          slices are this script's own seeded cut, not the official split

Usage::

    uv run --with "datasets>=4" --with "pyarrow>=17" python scripts/prepare_quickstart_data.py --dataset both       # pets + sroie (--dataset is required — no silent partial build)
    uv run --with "datasets>=4" --with "pyarrow>=17" python scripts/prepare_quickstart_data.py --dataset all        # all six; needs --nextqa-videos
    uv run --with "datasets>=4" --with "pyarrow>=17" python scripts/prepare_quickstart_data.py --dataset sroie
    uv run --with "datasets>=4" --with "pyarrow>=17" python scripts/prepare_quickstart_data.py --dataset cub
    uv run --with "datasets>=4" --with "pyarrow>=17" python scripts/prepare_quickstart_data.py --dataset textvqa
    uv run --with "datasets>=4" --with "pyarrow>=17" python scripts/prepare_quickstart_data.py --dataset gsm8k
    uv run --with "datasets>=4" --with "pyarrow>=17" python scripts/prepare_quickstart_data.py --dataset nextqa --nextqa-videos /path/to/NExT-QA.zip [--nextqa-annotations /path/to/nextqa/csvs]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

HF_MIRROR = "https://hf-mirror.com"
PETS_REPO = "timm/oxford-iiit-pet"
SROIE_REPO = "jinho8345/sroie-bio"
SROIE_FIELDS = ("company", "address", "date", "total")
CUB_REPO = "cassiekang/cub200_dataset"
CUB_TEXT_PREFIX = "A photo of a "

# Loader return contract:
#   pets  -> (pools, decode); pools[split] = [{"index","stem","filename","group","label"}]
#           decode(split, indices) -> returns PIL images ordered by indices
#   sroie -> (pool, decode); pool = [{"split","index","id","entities"}]
Loader = tuple[dict[str, list[dict[str, Any]]], Callable[[str, list[int]], list[Any]]]


def _diag(*args: Any, **kw: Any) -> None:
    """Diagnostics (source-fallback warnings, scan progress) go to stderr:
    stdout carries only the final summary + params block the user pastes
    (the package-wide "stdout = products, stderr = diagnostics" rule)."""
    print(*args, file=sys.stderr, **kw)


def _use_hf_mirror() -> None:
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    try:
        import huggingface_hub.constants as hh_constants
    except ImportError:
        return
    # The ENDPOINT constant is frozen at import time; the auto fallback must overwrite it in sync
    hh_constants.ENDPOINT = HF_MIRROR


def _load_hf(repo: str, name: str | None = None) -> Any:
    """``load_dataset(repo[, name])``; ``name`` selects the config of a multi-config repo
    (``openai/gsm8k`` ships ``main`` + ``socratic`` and refuses to load without one)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'the datasets library is required: rerun with uv run --with "datasets>=4" '
            '--with "pyarrow>=17" python ...'
        ) from exc
    return load_dataset(repo, name) if name else load_dataset(repo)


# ---------------------------------------------------------------------------
# SROIE: BIO sequences -> entity records
# ---------------------------------------------------------------------------
def bio_to_entity(words: list[str], tags: list[str]) -> dict[str, str]:
    """Aligned BIO word/tag sequence -> one entity record (SROIE has one record per
    receipt; multiple segments of the same field are joined with spaces).

    Tags look like ``B_company`` / ``I_company`` (underscore-separated; hyphens are
    handled too); orphan I- prefixes (no preceding B-) start a new segment; unknown
    fields and O tags are dropped; an all-O sequence returns {}.
    """
    segments: dict[str, list[str]] = {}
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None and buf:
            segments.setdefault(current, []).append(" ".join(buf))

    for word, tag in zip(words, tags, strict=True):
        prefix, _, field = str(tag).replace("-", "_").partition("_")
        field = field.lower()
        if prefix == "O" or prefix not in ("B", "I") or field not in SROIE_FIELDS:
            flush()
            current, buf = None, []
            continue
        if prefix == "B" or field != current:
            flush()
            buf = []
        current = field
        token = str(word).strip()
        if token:
            buf.append(token)
    flush()
    return {f: " ".join(segments[f]) for f in SROIE_FIELDS if f in segments}


JPEG_MAX_PIXELS = 2**21  # 2M px ≈ 2048 vision tokens/image (Qwen-family 32×32 px/token)


def _save_jpeg_bounded(image: Any, dest: Path, *, max_pixels: int = JPEG_MAX_PIXELS) -> None:
    """Persist an RGB JPEG, downscaling proportionally when the area exceeds the cap
    (shrink only, never upscale; LANCZOS).

    Raw SROIE scans reach 4961×7016 (34.8M px); under the Qwen3VL processor
    (max_pixels=16.7M) a single image can cost up to 16,264 vision tokens — a protegi
    gradient request bundling 4 oversized images measured 37,092 tok and blew the 32k
    context. The 2M px cap keeps 4 images worst-case ≤ ~8.6k tokens, fitting all
    multi-image methods and backends.
    """
    from PIL import Image

    rgb = image.convert("RGB")
    if rgb.width * rgb.height > max_pixels:
        scale = (max_pixels / (rgb.width * rgb.height)) ** 0.5
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.LANCZOS,
        )
    rgb.save(dest, "JPEG", quality=90)


def stratified_take(
    pool: list[dict[str, Any]], per_label: int, rng: random.Random, exclude: set[str] | None = None
) -> list[dict[str, Any]]:
    """Take per_label items stratified by (group, label); in-bucket shuffling keeps the
    output deterministic under a fixed seed.

    Filenames in exclude are skipped (train/val stay disjoint within the shared pool);
    returns items sorted by filename.
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in pool:
        if exclude and item["filename"] in exclude:
            continue
        buckets.setdefault((item["group"], item["label"]), []).append(item)
    picked: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        picked.extend(bucket[:per_label])
    return sorted(picked, key=lambda item: item["filename"])


# ---------------------------------------------------------------------------
# pets (grouped MCQ)
# ---------------------------------------------------------------------------
def _load_pets(source: str) -> Loader:
    if source in ("hf", "hf-mirror"):
        if source == "hf-mirror":
            _use_hf_mirror()
        return _load_pets_hf()
    if source == "modelscope":
        return _load_pets_modelscope()
    raise ValueError(f"unknown download source: {source!r}")


def _load_pets_hf() -> Loader:
    dsd = _load_hf(PETS_REPO)
    pools: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        dset = dsd[split]
        label_names = dset.features["label"].names
        species_names = dset.features["label_cat_dog"].names
        meta_cols = [c for c in ("label", "label_cat_dog", "image_id") if c in dset.column_names]
        if "label" not in meta_cols or "label_cat_dog" not in meta_cols:
            raise RuntimeError(
                f"{PETS_REPO}[{split}] is missing label/label_cat_dog columns; schema has changed"
            )
        items: list[dict[str, Any]] = []
        for idx, meta in enumerate(list(dset.select_columns(meta_cols))):
            stem = str(meta.get("image_id") or f"{split}_{idx:05d}")
            items.append(
                {
                    "index": idx,
                    "stem": stem,
                    "filename": f"{stem}.jpg",
                    "group": species_names[meta["label_cat_dog"]],
                    "label": str(label_names[meta["label"]]).replace("_", " ").title(),
                }
            )
        pools[split] = items

    def decode(split: str, indices: list[int]):
        # The val pool is sampled from the train split (same pool, disjoint); decode must fetch images from the source split
        src = "train" if split == "val" else split
        return [row["image"] for row in dsd[src].select(indices)]

    return pools, decode


def _load_pets_modelscope() -> Loader:
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is not installed: rerun with "
            'uv run --with "datasets>=4" --with "pyarrow>=17" --with modelscope '
            "python scripts/prepare_quickstart_data.py ..."
        ) from exc
    raw = MsDataset.load("modelscope/OxfordPets")
    pools: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        dset = raw[split] if hasattr(raw, "keys") and split in raw.keys() else raw
        items = []
        for idx, row in enumerate(list(dset)):
            image = _ms_row_image(row)
            label = _ms_row_label(row)
            if image is None or label is None:
                raise RuntimeError(
                    "unrecognized ModelScope OxfordPets row structure (expected "
                    "image:FILE / label:LABEL) — switch to --source hf-mirror"
                )
            items.append(
                {
                    "index": idx,
                    "stem": f"ms_{split}_{idx:05d}",
                    "filename": f"ms_{split}_{idx:05d}.jpg",
                    "group": _ms_row_group(row),
                    "label": label,
                    "image": image,
                }
            )
        pools[split] = items

    if all(str(it["label"]).isdigit() for pool in pools.values() for it in pool):
        _diag(
            "⚠ ModelScope OxfordPets only provides integer labels (no breed-name mapping), "
            "yielding a degraded benchmark (integer-string choices); prefer --source hf-mirror"
        )

    def decode(split: str, indices: list[int]):
        from PIL import Image

        # The val pool is sampled from the train split (same pool, disjoint); decode must fetch images from the source split
        pool = pools["train" if split == "val" else split]
        decoded = []
        for i in indices:
            value = pool[i]["image"]
            decoded.append(value if hasattr(value, "convert") else Image.open(value))
        return decoded

    return pools, decode


def _ms_row_image(row: dict[str, Any]) -> Any:
    """ModelScope row -> image payload (path str / PIL object / bytes dict); None if
    unrecognized.

    Probing found the real key is ``image:FILE`` (a local file path); the other keys are
    compatibility candidates. Opening is deferred to decode so pool construction does
    not hold thousands of file handles.
    """
    for key in ("image:FILE", "image", "img", "jpg", "pet_image"):
        value = row.get(key)
        if value is None:
            continue
        if hasattr(value, "convert") or isinstance(value, str):
            return value
        if isinstance(value, dict) and "bytes" in value:
            return value
    return None


def _ms_row_label(row: dict[str, Any]) -> str | None:
    """Row -> label string; in practice ModelScope only offers the ``label:LABEL``
    integer label (no breed-name mapping)."""
    for key in ("label:LABEL", "label", "breed", "category", "pet_type", "class_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None


def _ms_row_group(row: dict[str, Any]) -> str:
    """cat/dog grouping: use the species/cat_dog-like field on the MS row when present,
    otherwise fall into the single "both" group (choices stay multi-valued; only
    grouping granularity is lost)."""
    for key in ("species", "cat_dog", "label_cat_dog", "type"):
        value = row.get(key)
        if isinstance(value, str) and value.strip().lower() in ("cat", "dog"):
            return value.strip().lower()
    return "both"


def prepare_pets(
    out_root: Path,
    source: str,
    *,
    train_per_label: int,
    val_per_label: int,
    test_per_label: int,
    seed: int,
) -> dict[str, Any]:
    last: Exception | None = None
    sources = _resolve_sources(source, with_modelscope=True)
    for src in sources:
        try:
            pools, decode = _load_pets(src)
            break
        except Exception as exc:  # noqa: BLE001 - the fallback chain must swallow all load errors
            _diag(f"⚠ pets source {src} failed: {exc}")
            last = exc
    else:
        raise RuntimeError(
            f"all pets download sources failed ({sources}); retry with a single --source"
        ) from last

    rng = random.Random(seed)
    train_set = stratified_take(pools["train"], train_per_label, rng)
    val_set = stratified_take(
        pools["train"], val_per_label, rng, exclude={it["filename"] for it in train_set}
    )
    test_set = stratified_take(pools["test"], test_per_label, rng)

    all_filenames = [it["filename"] for it in (*train_set, *val_set, *test_set)]
    if len(set(all_filenames)) != len(all_filenames):
        raise RuntimeError("duplicate filenames in pets sampling — upstream image_id is not unique")

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[str, list[dict[str, Any]]] = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }
    for split, items in selected.items():
        chosen = sorted(items, key=lambda it: it["index"])
        for image, item in zip(decode(split, [it["index"] for it in chosen]), chosen, strict=True):
            dest = images_dir / item["filename"]
            if dest.exists():
                continue
            _save_jpeg_bounded(image, dest)
        by_group: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_group.setdefault(item["group"], []).append(item)
        for group, group_items in sorted(by_group.items()):
            payload = [
                {"filename": it["filename"], "label": it["label"]}
                for it in sorted(group_items, key=lambda it: it["filename"])
            ]
            (data_dir / f"{group}_{split}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
        for it in items:
            if not (images_dir / it["filename"]).is_file():
                raise RuntimeError(f"pets image not persisted to disk: {it['filename']}")

    breeds = {it["label"] for it in pools["train"]} | {it["label"] for it in pools["test"]}
    stats = {
        split: {
            group: sum(1 for it in items if it["group"] == group)
            for group in sorted({it["group"] for it in items})
        }
        for split, items in selected.items()
    }
    return {
        "stats": stats,
        "n_breeds": len(breeds),
        "data_dir": str(data_dir),
        "images_dir": str(images_dir),
        "source": src,
    }


# ---------------------------------------------------------------------------
# sroie (KIE)
# ---------------------------------------------------------------------------
def _load_sroie(source: str) -> Loader:
    if source not in ("hf", "hf-mirror"):
        raise ValueError(f"sroie has no {source!r} channel (only hf / hf-mirror)")
    if source == "hf-mirror":
        _use_hf_mirror()
    dsd = _load_hf(SROIE_REPO)
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for split in ("train", "val", "test"):
        if split not in dsd:
            continue
        for idx, row in enumerate(list(dsd[split].select_columns(["filename", "words", "labels"]))):
            stem = Path(str(row["filename"])).stem
            if stem in seen:
                continue
            seen.add(stem)
            try:
                entity = bio_to_entity(list(row["words"] or []), list(row["labels"] or []))
            except ValueError:  # dirty row with mismatched words/labels lengths; skip and count
                skipped += 1
                continue
            pool.append(
                {"split": split, "index": idx, "id": stem, "entities": [entity] if entity else []}
            )
    if skipped:
        _diag(f"⚠ sroie dirty rows (mismatched words/labels lengths) skipped: {skipped}")

    def decode(split: str, indices: list[int]):
        subset = dsd[split].select(indices)
        return [row["img"] for row in subset]

    return pool, decode


def prepare_sroie(
    out_root: Path,
    source: str,
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> dict[str, Any]:
    last: Exception | None = None
    sources = _resolve_sources(source, with_modelscope=False)
    for src in sources:
        try:
            pool, decode = _load_sroie(src)
            break
        except Exception as exc:  # noqa: BLE001
            _diag(f"⚠ sroie source {src} failed: {exc}")
            last = exc
    else:
        raise RuntimeError(
            f"all sroie download sources failed ({sources}); retry with a single --source"
        ) from last

    zero_entity = sum(1 for it in pool if not it["entities"])
    if pool and zero_entity / len(pool) >= 0.05:
        raise RuntimeError(
            f"abnormal zero-entity sroie receipt share: {zero_entity}/{len(pool)} (BIO aggregation likely failed; fail fast)"
        )
    if len(pool) < train_n + val_n + test_n:
        raise RuntimeError(
            f"sroie pool of {len(pool)} is below the requested {train_n + val_n + test_n} — lower --sroie-*-n"
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    slices = {
        "test": sorted(pool[:test_n], key=lambda it: it["id"]),
        "val": sorted(pool[test_n : test_n + val_n], key=lambda it: it["id"]),
        "train": sorted(pool[test_n + val_n : test_n + val_n + train_n], key=lambda it: it["id"]),
    }

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for split, items in slices.items():
        decode_map = {(it["split"], it["index"]): it for it in items}
        for (row_split, index), item in decode_map.items():
            dest = images_dir / f"{item['id']}.jpg"
            if dest.exists():
                continue
            (image,) = decode(row_split, [index])
            _save_jpeg_bounded(image, dest)
        payload = [
            {
                "image": str((images_dir / f"{it['id']}.jpg").absolute()),
                "entities": it["entities"],
                "id": it["id"],
            }
            for it in items
        ]
        (data_dir / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        for item in items:
            if not (images_dir / f"{item['id']}.jpg").is_file():
                raise RuntimeError(f"sroie image not persisted to disk: {item['id']}.jpg")

    fields = sorted({f for it in slices["train"] for e in it["entities"] for f in e})
    return {
        "stats": {split: len(items) for split, items in slices.items()},
        "fields": fields,
        "zero_entity": zero_entity,
        "data_dir": str(data_dir),
        "images_dir": str(images_dir),
        "source": src,
    }


def _resolve_sources(source: str, *, with_modelscope: bool) -> tuple[str, ...]:
    if source == "auto":
        chain = ["hf", "hf-mirror"] + (["modelscope"] if with_modelscope else [])
        return tuple(chain)
    return (source,)


# ---------------------------------------------------------------------------
# cub (CUB-200-2011 streaming subset: family-grouped MCQ; the fine-grained challenge = comparing species within a group)
# ---------------------------------------------------------------------------
CUB_SPLITS = ("train", "val", "test")


def _cub_species(text: str) -> str | None:
    """The text column ``A photo of a <species>`` -> species name (lowercase); None on mismatch."""
    if not text.startswith(CUB_TEXT_PREFIX):
        return None
    species = text[len(CUB_TEXT_PREFIX) :].strip().rstrip(".").strip()
    return species or None


def _cub_family(species: str) -> str:
    """Group = last word of the species name (family common name: laysan albatross -> albatross)."""
    return species.split()[-1]


def _stream_cub(source: str):
    if source not in ("hf", "hf-mirror"):
        raise ValueError(f"cub has no {source!r} channel (only hf / hf-mirror)")
    if source == "hf-mirror":
        _use_hf_mirror()
    from datasets import load_dataset

    dset = load_dataset(CUB_REPO, split="train", streaming=True)

    def gen():
        for idx, row in enumerate(dset):
            species = _cub_species(str(row.get("text") or ""))
            image = row.get("image")
            if species is None or image is None:
                continue
            yield idx, species, image

    return gen()


def prepare_cub(
    out_root: Path,
    source: str,
    *,
    families_max: int,
    species_per_family: int,
    imgs_per_species: int,
    test_per_species: int,
    train_per_species: int,
    val_per_species: int,
    seed: int,
) -> dict[str, Any]:
    need = test_per_species + train_per_species + val_per_species
    last: Exception | None = None
    for src in _resolve_sources(source, with_modelscope=False):
        try:
            rows = _stream_cub(src)
            break
        except Exception as exc:  # noqa: BLE001 - the fallback chain must swallow all load errors
            _diag(f"⚠ cub source {src} failed: {exc}")
            last = exc
    else:
        raise RuntimeError(
            "all cub download sources failed; retry with a single --source"
        ) from last

    images_dir = out_root / "images"
    data_dir = out_root / "data"
    scan_dir = out_root / "_scan"
    images_dir.mkdir(parents=True, exist_ok=True)
    scan_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Single-pass streaming scan: buffer only the first imgs_per_species images per
        # species (rows past quota are dropped as we scan)
        buffered: dict[str, list[str]] = {}
        scanned = bad = 0
        for _, species, image in rows:
            scanned += 1
            if scanned % 2000 == 0:
                _diag(f"   cub scan {scanned} rows / {len(buffered)} species ...")
            files = buffered.setdefault(species, [])
            if len(files) >= imgs_per_species:
                continue
            slug = species.replace(" ", "_")
            dest = scan_dir / f"{slug}_{len(files):02d}.jpg"
            try:
                _save_jpeg_bounded(image, dest)
            except Exception:  # noqa: BLE001 - skip and count individual bad images; fail fast past the cap
                bad += 1
                if bad > max(50, scanned * 0.02):
                    raise RuntimeError(
                        f"abnormal cub bad-image rate ({bad}/{scanned}); upstream data looks corrupted"
                    ) from None
                continue
            files.append(dest.name)
        if len(buffered) < 100:
            raise RuntimeError(
                f"cub scan saw only {len(buffered)} species (<100); stream looks truncated"
            )

        eligible: dict[str, list[str]] = {}
        for species, files in buffered.items():
            if len(files) >= need:
                eligible.setdefault(_cub_family(species), []).append(species)
        families = {f: sorted(ss) for f, ss in eligible.items() if len(ss) >= 4}
        if len(families) < 5:
            raise RuntimeError(
                f"not enough cub family groups (only {len(families)} with >=4 species); relax the per-family species floor or switch source"
            )

        # Greedy group selection: species count descending (lexicographic tiebreak);
        # groups with fewer than 4 species are skipped
        selected: dict[str, list[str]] = {}
        remaining = families_max * species_per_family
        for fam in sorted(families, key=lambda f: (-len(families[f]), f)):
            if remaining < 4:
                break
            take = families[fam][: min(species_per_family, remaining)]
            if len(take) < 4:
                continue
            selected[fam] = take
            remaining -= len(take)

        rng = random.Random(seed)
        split_of: dict[str, dict[str, list[str]]] = {}
        for _fam, species_list in selected.items():
            for species in species_list:
                files = list(buffered[species])
                rng.shuffle(files)
                split_of[species] = {
                    "test": files[:test_per_species],
                    "train": files[test_per_species : test_per_species + train_per_species],
                    "val": files[
                        test_per_species + train_per_species : test_per_species
                        + train_per_species
                        + val_per_species
                    ],
                }

        data_dir.mkdir(parents=True, exist_ok=True)
        for fam, species_list in selected.items():
            by_split: dict[str, list[dict[str, str]]] = {s: [] for s in CUB_SPLITS}
            for species in species_list:
                for split, filenames in split_of[species].items():
                    for name in filenames:
                        src_path = scan_dir / name
                        dest = images_dir / name
                        if not src_path.is_file():
                            raise RuntimeError(f"cub buffered image missing: {name}")
                        dest.write_bytes(src_path.read_bytes())
                        by_split[split].append({"filename": name, "label": species.title()})
            for split in CUB_SPLITS:
                payload = sorted(by_split[split], key=lambda it: it["filename"])
                (data_dir / f"{fam}_{split}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
                )
    finally:
        shutil.rmtree(scan_dir, ignore_errors=True)

    n_species = sum(len(ss) for ss in selected.values())
    stats = {
        split: {
            fam: sum(len(split_of[sp][split]) for sp in species_list)
            for fam, species_list in selected.items()
        }
        for split in CUB_SPLITS
    }
    return {
        "stats": stats,
        "n_families": len(selected),
        "n_species": n_species,
        "data_dir": str(data_dir),
        "images_dir": str(images_dir),
        "source": src,
    }


# ---------------------------------------------------------------------------
# textvqa (TextVQA labeled-val slice: open-ended VQA; scene-text reading keeps the
# ability-dominated shape and the 10-annotation answers feed the VQAv2 soft-accuracy
# metric directly — zero judging)
# ---------------------------------------------------------------------------
TEXTVQA_REPO = "lmms-lab/TextVQA"


def _load_textvqa(source: str) -> tuple[list[dict[str, Any]], Any]:
    """``(pool, validation-dataset)``; pool rows carry question metadata only — the
    embedded JPEGs stay in the arrow cache until a selected row is re-indexed at write
    time."""
    if source not in ("hf", "hf-mirror"):
        raise ValueError(f"textvqa has no {source!r} channel (only hf / hf-mirror)")
    if source == "hf-mirror":
        _use_hf_mirror()
    dsd = _load_hf(TEXTVQA_REPO)
    if "validation" not in dsd:
        raise RuntimeError(f"{TEXTVQA_REPO} has no 'validation' split: {list(dsd)}")
    val = dsd["validation"]
    pool = [
        {
            "image_id": str(row["image_id"]),
            "question_id": int(row["question_id"]),
            "question": str(row["question"]),
            "answers": [str(a) for a in row["answers"]],
        }
        for row in val.select_columns(["image_id", "question_id", "question", "answers"])
    ]
    if len(pool) < 4500:
        raise RuntimeError(
            f"textvqa validation pool is {len(pool)} rows (expected ~5001 labeled questions; packaging drift?)"
        )
    empty = sum(1 for it in pool if not any(a.strip() for a in it["answers"]))
    if pool and empty / len(pool) >= 0.05:
        raise RuntimeError(f"abnormal empty-answer textvqa share: {empty}/{len(pool)} (fail fast)")
    print(
        f"textvqa pool: {len(pool)} labeled questions over "
        f"{len({it['image_id'] for it in pool})} images"
    )
    return pool, val


def prepare_textvqa(
    out_root: Path,
    source: str,
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> dict[str, Any]:
    """Image-group-aware slice of the labeled validation split: whole image-groups move
    into splits together (test -> val -> train, the sroie slice order); a group
    overshooting its split target is truncated to the exact remainder and its leftover
    questions are discarded (never re-placed), giving exact counts with zero image
    overlap across splits — deterministic under a fixed seed."""
    last: Exception | None = None
    sources = _resolve_sources(source, with_modelscope=False)
    for src in sources:
        try:
            pool, val = _load_textvqa(src)
            break
        except Exception as exc:  # noqa: BLE001
            _diag(f"⚠ textvqa source {src} failed: {exc}")
            last = exc
    else:
        raise RuntimeError(
            f"all textvqa download sources failed ({sources}); retry with a single --source"
        ) from last

    groups: dict[str, list[dict[str, Any]]] = {}
    for it in pool:
        groups.setdefault(it["image_id"], []).append(it)
    group_list = [groups[iid] for iid in sorted(groups)]
    rng = random.Random(seed)
    rng.shuffle(group_list)

    slices: dict[str, list[dict[str, Any]]] = {"test": [], "val": [], "train": []}
    targets = {"test": test_n, "val": val_n, "train": train_n}
    queue = iter(group_list)
    for split, target in targets.items():
        while len(slices[split]) < target:
            group = next(queue, None)
            if group is None:
                raise RuntimeError(
                    f"textvqa pool exhausted filling {split}: "
                    f"{len(slices[split])}/{target} (lower --textvqa-*-n)"
                )
            need = target - len(slices[split])
            slices[split].extend(group[:need])  # group[need:] discarded on truncation

    from PIL import Image

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    row_index = {int(row["question_id"]): i for i, row in enumerate(val)}
    for split, items in slices.items():
        items.sort(key=lambda it: it["question_id"])
        for it in items:
            dest = images_dir / f"{it['image_id']}.jpg"
            if dest.exists():
                continue
            raw = val[row_index[it["question_id"]]]["image"]
            # three shapes in the wild: datasets decodes the Image feature to a PIL
            # object (normal), raw {"bytes": ...} parquet dicts, or plain bytes
            if isinstance(raw, dict):
                raw = Image.open(io.BytesIO(raw["bytes"]))
            elif isinstance(raw, (bytes, bytearray)):
                raw = Image.open(io.BytesIO(raw))
            _save_jpeg_bounded(raw, dest)
        payload = [
            {
                "id": f"textvqa_q{it['question_id']}",
                "image": f"{it['image_id']}.jpg",
                "question": it["question"],
                "answers": it["answers"],
            }
            for it in items
        ]
        (data_dir / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

    n_images = len({it["image_id"] for items in slices.values() for it in items})
    return {
        "stats": {split: len(items) for split, items in slices.items()},
        "n_images": n_images,
        "data_dir": str(data_dir),
        "images_dir": str(images_dir),
        "source": src,
    }


# ---------------------------------------------------------------------------
# gsm8k (numeric math QA, text only; the val split is carved out of the official
# train split because the corpus ships no official val)
# ---------------------------------------------------------------------------
GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"


def gsm8k_final_answer(text: str) -> str:
    """The text after the last ``####`` marker, stripped (``"3 + 4 = 7 #### 7"`` -> ``"7"``).

    GSM8K gold solutions end with ``#### <value>``, but the field is not guaranteed
    clean: a row without the marker returns the stripped answer text unchanged (the
    downstream numeric parser then takes the ``Answer:`` line or the last number)
    instead of raising, so one dirty row cannot abort the whole conversion.
    """
    head, marker, tail = str(text).rpartition("####")
    return (tail if marker else head).strip()


def derive_gsm8k_splits(
    rows: list[dict[str, Any]], train_n: int, val_n: int, test_n: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Carve disjoint ``train`` / ``val`` / ``test`` slices of exact sizes out of one
    ``rows`` pool with a single seeded shuffle.

    ``rows`` elements carry ``question`` / ``answer`` and may carry ``id``; a row
    without one is stamped ``train/<pool index>``, which is the true provenance in the
    real flow (the pool *is* the official train split; ``prepare_gsm8k`` pre-stamps the
    official test rows ``test/<row>`` and keeps them out of this pool). The draw order
    is val -> train -> test, mirroring which upstream split feeds each slice;
    ``test_n=0`` yields an empty test list (the real flow's call). Each slice keeps
    pool order, so equal seed + sizes produce byte-identical files.
    """
    need = train_n + val_n + test_n
    if len(rows) < need:
        raise ValueError(
            f"gsm8k pool of {len(rows)} rows is below the requested {need}; lower "
            "--gsm8k-train-n / --gsm8k-val-n / --gsm8k-test-n"
        )
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    bounds = {
        "val": (0, val_n),
        "train": (val_n, val_n + train_n),
        "test": (val_n + train_n, need),
    }
    return {
        split: [
            {
                "id": str(rows[i].get("id") or f"train/{i}"),
                "question": str(rows[i]["question"]),
                "answer": str(rows[i]["answer"]),
            }
            for i in sorted(order[start:stop])
        ]
        for split, (start, stop) in bounds.items()
    }


def _load_gsm8k(source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(official train pool, official test pool)`` of ``{"id", "question", "answer"}``
    records; ``answer`` is already the final numeric string."""
    if source not in ("hf", "hf-mirror"):
        raise ValueError(f"gsm8k has no {source!r} channel (only hf / hf-mirror)")
    if source == "hf-mirror":
        _use_hf_mirror()
    dsd = _load_hf(GSM8K_REPO, GSM8K_CONFIG)
    pools: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        if split not in dsd:
            raise RuntimeError(f"{GSM8K_REPO}[{GSM8K_CONFIG}] has no {split!r} split: {list(dsd)}")
        pool: list[dict[str, Any]] = []
        for idx, row in enumerate(dsd[split].select_columns(["question", "answer"])):
            answer = gsm8k_final_answer(str(row["answer"]))
            if not answer:
                raise RuntimeError(
                    f"{GSM8K_REPO}[{split}][{idx}] carries no final answer after '####'"
                )
            pool.append(
                {"id": f"{split}/{idx}", "question": str(row["question"]), "answer": answer}
            )
        pools[split] = pool
    if len(pools["train"]) < 7000:
        raise RuntimeError(
            f"gsm8k train pool is {len(pools['train'])} rows (expected ~7473; packaging drift?)"
        )
    if len(pools["test"]) < 1000:
        raise RuntimeError(
            f"gsm8k test pool is {len(pools['test'])} rows (expected ~1319; packaging drift?)"
        )
    return pools["train"], pools["test"]


def prepare_gsm8k(
    out_root: Path,
    source: str,
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> dict[str, Any]:
    """val + train from the official train split (seeded disjoint draw through
    ``derive_gsm8k_splits``); test = the first ``test_n`` rows of the official test
    split, so the test slice is identical across seeds and runs."""
    last: Exception | None = None
    sources = _resolve_sources(source, with_modelscope=False)
    for src in sources:
        try:
            train_pool, test_pool = _load_gsm8k(src)
            break
        except Exception as exc:  # noqa: BLE001 - the fallback chain must swallow all load errors
            _diag(f"⚠ gsm8k source {src} failed: {exc}")
            last = exc
    else:
        raise RuntimeError(
            f"all gsm8k download sources failed ({sources}); retry with a single --source"
        ) from last

    if len(test_pool) < test_n:
        raise RuntimeError(
            f"gsm8k official test pool of {len(test_pool)} rows is below the requested "
            f"{test_n} — lower --gsm8k-test-n"
        )
    slices = derive_gsm8k_splits(train_pool, train_n=train_n, val_n=val_n, test_n=0, seed=seed)
    # The official test rows already carry test/<row> ids; slicing them unshuffled keeps
    # the test split byte-identical across seeds
    slices["test"] = test_pool[:test_n]

    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, items in slices.items():
        (data_dir / f"{split}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    return {
        "stats": {split: len(items) for split, items in slices.items()},
        "data_dir": str(data_dir),
        "source": src,
    }


# ---------------------------------------------------------------------------
# nextqa (video MCQ: official CSVs + user-supplied clips; only the sampled clips
# are copied out, the 6.5GB pack is never downloaded here)
# ---------------------------------------------------------------------------
NEXTQA_ANNOTATION_BASE = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa"
NEXTQA_SPLITS = ("train", "val", "test")
NEXTQA_OPTIONS = 5
NEXTQA_VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".webm", ".mov")


def _nextqa_cell(row: dict[str, Any], key: str) -> str:
    """Row cell -> stripped string (``0`` stays ``"0"`` instead of collapsing to ``""``)."""
    value = row.get(key)
    return "" if value is None else str(value).strip()


def nextqa_rows_to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotation rows -> ``{"id", "video", "question", "choices", "answer"}`` records.

    ``answer`` is accepted as an option index ``"0".."4"`` (resolved to the matching
    option text) or as the option text itself; any other shape raises ``ValueError``
    naming the row, so a drifted CSV cannot silently produce unscoreable labels.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        video = _nextqa_cell(row, "video")
        qid = _nextqa_cell(row, "qid")
        question = _nextqa_cell(row, "question")
        if not video or not question:
            raise ValueError(
                f"NExT-QA row video={video!r} qid={qid!r} is missing its video/question column"
            )
        choices: list[str] = []
        for i in range(NEXTQA_OPTIONS):
            option = _nextqa_cell(row, f"option{i}")
            if not option:
                raise ValueError(
                    f"NExT-QA row video={video!r} qid={qid!r} has an empty option{i} "
                    f"(expected {NEXTQA_OPTIONS} options)"
                )
            choices.append(option)
        answer = _nextqa_cell(row, "answer")
        if answer.isdigit() and int(answer) < len(choices):
            resolved = choices[int(answer)]
        elif answer in choices:
            resolved = answer
        else:
            raise ValueError(
                f"NExT-QA row video={video!r} qid={qid!r}: answer {answer!r} is neither an option "
                f"index 0..{len(choices) - 1} nor one of the choices {choices}"
            )
        records.append(
            {
                "id": f"nextqa_{video}_{qid}",
                "video": video,
                "question": question,
                "choices": choices,
                "answer": resolved,
            }
        )
    return records


def sample_nextqa_per_video(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """One QA per video (seeded pick inside each video's bucket), then a seeded order
    capped at ``n`` rows.

    Videos are keyed in sorted order and the per-video + global shuffles draw from one
    seeded RNG, so the output is deterministic and never holds two rows of one video.
    """
    if n < 0:
        raise ValueError(f"sample size must be >= 0, got {n}")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_nextqa_cell(row, "video"), []).append(row)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for video in sorted(buckets):
        bucket = list(buckets[video])
        rng.shuffle(bucket)
        picked.append(bucket[0])
    rng.shuffle(picked)
    return picked[:n]


def nextqa_csv_rows(text: str, *, source: str) -> list[dict[str, Any]]:
    """CSV text -> normalized annotation rows.

    The official header is ``video,frame_count,width,height,question,answer,qid,type,a0..a4``;
    the option columns are normalized to ``option0..option4`` so every downstream step
    sees one shape (unknown extra columns are carried through untouched).
    """
    reader = csv.DictReader(io.StringIO(text))
    fields = list(reader.fieldnames or ())
    required = {"video", "qid", "question", "answer"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(
            f"NExT-QA annotation {source} is missing columns {missing} (header: {fields})"
        )
    rows: list[dict[str, Any]] = []
    for raw in reader:
        row = {str(k).strip(): _nextqa_cell(raw, k) for k in raw if k is not None}
        for i in range(NEXTQA_OPTIONS):
            name = f"option{i}" if f"option{i}" in row else f"a{i}"
            if name not in row:
                raise RuntimeError(
                    f"NExT-QA annotation {source} has no option column {name!r} (header: {fields})"
                )
            row[f"option{i}"] = row[name]
        rows.append(row)
    return rows


def _nextqa_download(url: str, timeout: float = 120.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"failed to download NExT-QA annotations from {url}: {exc} — check network access "
            "to GitHub (the annotations are fetched directly from raw.githubusercontent.com) "
            "or pass --nextqa-annotations <directory with train.csv/val.csv/test.csv>"
        ) from exc


def _nextqa_annotation_text(
    split: str, annotations_dir: Path | None, fetch: Callable[[str], str]
) -> tuple[str, str]:
    """``(source label, CSV text)`` for one split: a local ``<annotations_dir>/<split>.csv``
    when given (hosts where raw.githubusercontent.com is unreachable), else the official URL."""
    if annotations_dir is None:
        url = f"{NEXTQA_ANNOTATION_BASE}/{split}.csv"
        return url, fetch(url)
    path = annotations_dir / f"{split}.csv"
    try:
        return str(path), path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"failed to read NExT-QA annotation {path}: {exc}") from exc


def load_nextqa_rows(
    fetch: Callable[[str], str] = _nextqa_download,
    annotations_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """All labeled annotation rows of the three official CSV files.

    The official test CSV ships the answer column today; a distribution that withholds
    it raises ``RuntimeError`` naming the file instead of quietly producing an
    unscoreable test slice.
    """
    rows: list[dict[str, Any]] = []
    for split in NEXTQA_SPLITS:
        url, text = _nextqa_annotation_text(split, annotations_dir, fetch)
        parsed = nextqa_csv_rows(text, source=url)
        if not parsed:
            raise RuntimeError(f"NExT-QA annotation {url} has no rows")
        empty = [i for i, row in enumerate(parsed) if not _nextqa_cell(row, "answer")]
        if empty:
            if split == "test":
                raise RuntimeError(
                    f"NExT-QA test labels are withheld: {url} leaves the answer column empty for "
                    f"{len(empty)}/{len(parsed)} rows — supply a labeled test CSV before running "
                    "--dataset nextqa"
                )
            raise ValueError(f"NExT-QA annotation {url}[{empty[0]}] has an empty answer column")
        rows.extend({**row, "split": split} for row in parsed)
    return rows


def _nextqa_clip_index(source: Path) -> dict[str, tuple[Any, str]]:
    """video id -> (locator, suffix) for every clip reachable from ``source``.

    ``source`` is a directory of clips or a zip archive of them; clips are matched on
    the file stem because both the official bundle and hand-made directories nest them
    (``<group>/<video id>.mp4``).
    """
    index: dict[str, tuple[Any, str]] = {}
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in NEXTQA_VIDEO_SUFFIXES:
                index.setdefault(path.stem, (path, path.suffix.lower()))
        return index
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for info in zf.infolist():
                suffix = Path(info.filename).suffix.lower()
                if info.is_dir() or suffix not in NEXTQA_VIDEO_SUFFIXES:
                    continue
                index.setdefault(Path(info.filename).stem, (info.filename, suffix))
        return index
    raise RuntimeError(
        f"--nextqa-videos must be a zip archive or a directory of clips, got {source} "
        "(the official NExT-QA video pack is linked from https://github.com/doc-doc/NExT-QA)"
    )


def _copy_nextqa_clips(
    source: Path,
    index: dict[str, tuple[Any, str]],
    video_ids: list[str],
    clips_dir: Path,
) -> dict[str, str]:
    """Copy just the selected clips (zip members are streamed, never unpacked whole);
    returns video id -> on-disk file name, the value the split files reference."""
    missing = [vid for vid in video_ids if vid not in index]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected NExT-QA clips are not in {source} (first: "
            f"{sorted(missing)[:3]}) — the clip pack must contain every sampled video"
        )
    names: dict[str, str] = {}
    zip_file = None if source.is_dir() else zipfile.ZipFile(source)
    try:
        for vid in video_ids:
            locator, suffix = index[vid]
            dest = clips_dir / f"{vid}{suffix}"
            names[vid] = dest.name
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            if zip_file is None:
                shutil.copyfile(locator, dest)
            else:
                with zip_file.open(locator) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
    finally:
        if zip_file is not None:
            zip_file.close()
    return names


def prepare_nextqa(
    out_root: Path,
    videos: Path,
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    annotations: Path | None = None,
) -> dict[str, Any]:
    """Pooled official CSVs -> one QA per video -> a seeded disjoint train/val/test cut
    (test first, mirroring sroie); only the sampled clips are copied to ``videos/``.

    Rows whose clip is absent from the pack are excluded before sampling: the public
    mirrors of the clip pack (ModelScope ``AI-ModelScope/NExTQA``, HF
    ``VLM2Vec/nextqa-rawvideo``) carry the 1,570 val+test clips only, so a partial
    pack must sample within what it can actually serve."""
    index = _nextqa_clip_index(videos)
    rows = load_nextqa_rows(annotations_dir=annotations)
    available = [row for row in rows if _nextqa_cell(row, "video") in index]
    if len(available) < len(rows):
        _diag(
            f"⚠ {len(rows) - len(available)} NExT-QA annotation rows reference clips absent "
            f"from {videos}; sampling over the {len(index)} clips the pack holds"
        )
    rows = available
    total = train_n + val_n + test_n
    sampled = sample_nextqa_per_video(rows, total, seed)
    if len(sampled) < total:
        raise RuntimeError(
            f"NExT-QA has only {len(sampled)} labeled videos with a clip in {videos}, below "
            f"the requested {total}; lower --nextqa-train-n / --nextqa-val-n / --nextqa-test-n "
            "or supply a pack covering more of the corpus"
        )
    records = nextqa_rows_to_records(sampled)
    random.Random(seed).shuffle(records)
    slices = {
        "test": records[:test_n],
        "val": records[test_n : test_n + val_n],
        "train": records[test_n + val_n : total],
    }
    for split in slices:
        slices[split].sort(key=lambda rec: (rec["video"], rec["id"]))

    clips_dir = out_root / "videos"
    clips_dir.mkdir(parents=True, exist_ok=True)
    selected = [rec for split in NEXTQA_SPLITS for rec in slices[split]]
    names = _copy_nextqa_clips(videos, index, [rec["video"] for rec in selected], clips_dir)

    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, items in slices.items():
        payload = [
            {
                "id": rec["id"],
                "video": names[rec["video"]],
                "question": rec["question"],
                "choices": rec["choices"],
                "answer": rec["answer"],
            }
            for rec in items
        ]
        (data_dir / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    return {
        "stats": {split: len(items) for split, items in slices.items()},
        "n_videos": len(names),
        "videos_source": str(videos),
        "data_dir": str(data_dir),
        "videos_dir": str(clips_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=("pets", "sroie", "cub", "textvqa", "gsm8k", "nextqa", "both", "all"),
        required=True,
        help="which dataset(s) to build (required — no silent partial build); "
        "both = pets + sroie, all = all six (needs --nextqa-videos)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/quickstart"))
    parser.add_argument(
        "--source", choices=("auto", "hf", "hf-mirror", "modelscope"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pets-train-per-label", type=int, default=2)
    parser.add_argument("--pets-val-per-label", type=int, default=1)
    parser.add_argument("--pets-test-per-label", type=int, default=3)
    parser.add_argument("--sroie-train-n", type=int, default=100)
    parser.add_argument("--sroie-val-n", type=int, default=30)
    parser.add_argument("--sroie-test-n", type=int, default=100)
    parser.add_argument("--cub-families", type=int, default=8)
    parser.add_argument("--cub-species-per-family", type=int, default=8)
    parser.add_argument("--cub-imgs-per-species", type=int, default=10)
    parser.add_argument("--cub-train-per-species", type=int, default=2)
    parser.add_argument("--cub-val-per-species", type=int, default=1)
    parser.add_argument("--cub-test-per-species", type=int, default=4)
    parser.add_argument("--textvqa-train-n", type=int, default=100)
    parser.add_argument("--textvqa-val-n", type=int, default=30)
    parser.add_argument("--textvqa-test-n", type=int, default=100)
    parser.add_argument("--gsm8k-train-n", type=int, default=100)
    parser.add_argument("--gsm8k-val-n", type=int, default=30)
    parser.add_argument("--gsm8k-test-n", type=int, default=100)
    parser.add_argument("--nextqa-train-n", type=int, default=100)
    parser.add_argument("--nextqa-val-n", type=int, default=30)
    parser.add_argument("--nextqa-test-n", type=int, default=100)
    parser.add_argument(
        "--nextqa-videos",
        type=Path,
        default=None,
        help="local NExT-QA clip pack (zip archive or directory of clips); required by "
        "--dataset nextqa. The official 6.5GB video download is linked from "
        "https://github.com/doc-doc/NExT-QA and is never fetched by this script.",
    )
    parser.add_argument(
        "--nextqa-annotations",
        type=Path,
        default=None,
        help="directory holding train.csv / val.csv / test.csv; use it where "
        "raw.githubusercontent.com is unreachable (the official CSVs live under "
        "dataset/nextqa of https://github.com/doc-doc/NExT-QA)",
    )
    args = parser.parse_args(argv)
    if args.dataset in ("nextqa", "all") and args.nextqa_videos is None:
        parser.error(
            "--nextqa-videos is required for --dataset nextqa: pass the NExT-QA clip pack "
            "(zip archive or directory of clips; the official download is linked from "
            "https://github.com/doc-doc/NExT-QA)"
        )

    root: Path = args.output_root
    if args.dataset in ("pets", "both", "all"):
        info = prepare_pets(
            root / "pets",
            args.source,
            train_per_label=args.pets_train_per_label,
            val_per_label=args.pets_val_per_label,
            test_per_label=args.pets_test_per_label,
            seed=args.seed,
        )
        counts = {s: sum(v.values()) for s, v in info["stats"].items()}
        print(
            f"✅ pets done (source={info['source']}, {info['n_breeds']} breeds): "
            f"train {counts['train']} / val {counts['val']} / test {counts['test']}"
        )
        for split, per_group in info["stats"].items():
            print(f"   {split}: {per_group}")
        print(f"   McqTask params: data_dir={info['data_dir']}  images_dir={info['images_dir']}")

    if args.dataset in ("sroie", "both", "all"):
        info = prepare_sroie(
            root / "sroie",
            args.source,
            train_n=args.sroie_train_n,
            val_n=args.sroie_val_n,
            test_n=args.sroie_test_n,
            seed=args.seed,
        )
        print(f"✅ sroie done (source={info['source']}): {info['stats']}")
        print(f"   fields: {info['fields']}  zero-entity: {info['zero_entity']}")
        print(f"   KieTask params: data_dir={info['data_dir']}")

    if args.dataset in ("cub", "all"):
        info = prepare_cub(
            root / "cub",
            args.source,
            families_max=args.cub_families,
            species_per_family=args.cub_species_per_family,
            imgs_per_species=args.cub_imgs_per_species,
            test_per_species=args.cub_test_per_species,
            train_per_species=args.cub_train_per_species,
            val_per_species=args.cub_val_per_species,
            seed=args.seed,
        )
        counts = {s: sum(v.values()) for s, v in info["stats"].items()}
        print(
            f"✅ cub done (source={info['source']}, {info['n_families']} family groups / "
            f"{info['n_species']} species): train {counts['train']} / val {counts['val']} / "
            f"test {counts['test']}"
        )
        for split, per_family in info["stats"].items():
            print(f"   {split}: {per_family}")
        print(f"   McqTask params: data_dir={info['data_dir']}  images_dir={info['images_dir']}")

    if args.dataset in ("textvqa", "all"):
        info = prepare_textvqa(
            root / "textvqa",
            args.source,
            train_n=args.textvqa_train_n,
            val_n=args.textvqa_val_n,
            test_n=args.textvqa_test_n,
            seed=args.seed,
        )
        counts = info["stats"]
        print(
            f"✅ textvqa done (source={info['source']}, {info['n_images']} unique images): "
            f"train {counts['train']} / val {counts['val']} / test {counts['test']}"
        )
        print(f"   VqaTask params: data_dir={info['data_dir']}  images_dir={info['images_dir']}")

    if args.dataset in ("gsm8k", "all"):
        info = prepare_gsm8k(
            root / "gsm8k",
            args.source,
            train_n=args.gsm8k_train_n,
            val_n=args.gsm8k_val_n,
            test_n=args.gsm8k_test_n,
            seed=args.seed,
        )
        counts = info["stats"]
        print(
            f"✅ gsm8k done (source={info['source']}): "
            f"train {counts['train']} / val {counts['val']} / test {counts['test']}"
        )
        print(f"   MathQaTask params: data_dir={info['data_dir']}")

    if args.dataset in ("nextqa", "all"):
        videos = args.nextqa_videos
        assert videos is not None  # parser.error() above rejects --dataset nextqa without it
        info = prepare_nextqa(
            root / "nextqa",
            videos,
            train_n=args.nextqa_train_n,
            val_n=args.nextqa_val_n,
            test_n=args.nextqa_test_n,
            seed=args.seed,
            annotations=args.nextqa_annotations,
        )
        counts = info["stats"]
        print(
            f"✅ nextqa done ({info['n_videos']} clips from {info['videos_source']}): "
            f"train {counts['train']} / val {counts['val']} / test {counts['test']}"
        )
        print(
            f"   VideoMcqTask params: data_dir={info['data_dir']}  videos_dir={info['videos_dir']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
