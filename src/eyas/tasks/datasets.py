"""Group-based MCQ dataset loading + stratified sampling of the train split.

Data format (one directory per dataset; a group = a semantic subset such as
a bird family in CUB)::

    <Dataset>/
    ├── <group>_train.json / <group>_test.json   # JSON array: [{"filename","label"}, ...]
    └── images/<filename>                        # filename relative to images_dir

Ordering conventions (critical for parity: same seed, same output):
- groups sorted by filename; samples within a group keep JSON array order;
- a group's choice list = labels of the group's samples by descending
  frequency (Counter.most_common, ties keep first-occurrence order).

Sampling (stratified_sample_train) stratifies by (group, label) quotas:
with a fixed seed it produces a fully deterministic sample subset and order.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from eyas.tasks.base import Sample

__all__ = [
    "DATASET_CONFIGS",
    "DATASET_DOMAIN",
    "SUPPORTED_DATASETS",
    "extract_choices",
    "load_group_split",
    "build_mcq_samples",
    "load_math_qa_samples",
    "MIXED_COMBOS",
    "load_mcq_samples",
    "load_mixed_mcq_samples",
    "load_video_mcq_samples",
    "load_vqa_samples",
    "resolve_dataset_paths",
    "stratified_sample_train",
    "derive_sample_seed",
]

VALID_SPLITS = ("train", "test", "val")


def _dataset_root() -> str | None:
    """Dataset root dir (EYAS_DATASET_ROOT > DATASET_ROOT), read at call time
    so an environment set after this module's import still takes effect."""
    return os.environ.get("EYAS_DATASET_ROOT") or os.environ.get("DATASET_ROOT")


# Predefined dataset configs: data_dir (dir holding the JSONs) and images_dir (image root dir)
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "Aircraft": {
        "data_dir": "Aircraft",
        "images_dir": "Aircraft/images",
    },
    "CUB": {
        "data_dir": "CUB",
        "images_dir": "CUB-200-2011/CUB-200-2011/images",
    },
    "NABirds": {
        "data_dir": "NABirds",
        "images_dir": "NABirds/images",
    },
    "StanfordCars": {
        "data_dir": "StanfordCars",
        "images_dir": "StanfordCars/images",
    },
    "PlantVillage": {
        "data_dir": "PlantVillage",
        "images_dir": "PlantVillage/images",
    },
}

SUPPORTED_DATASETS = tuple(DATASET_CONFIGS.keys())

# prompt template {{domain}} placeholder -> domain word (unlisted datasets fall back to "target")
DATASET_DOMAIN: dict[str, str] = {
    "Aircraft": "aircraft",
}


def resolve_dataset_paths(
    dataset: str, data_dir: str | Path | None = None, images_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Resolve a dataset's JSON directory and image directory.

    Explicit arguments win (when both are given, the dataset name is
    informational only and the registry is not consulted); otherwise the
    registered config is looked up and resolved relative to the env root
    (EYAS_DATASET_ROOT > DATASET_ROOT, read at call time), failing fast when
    missing.
    """
    if data_dir is not None and images_dir is not None:
        return Path(data_dir), Path(images_dir)
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}', options: {list(DATASET_CONFIGS.keys())}")
    root = _dataset_root()
    if data_dir is None:
        if root is None:
            raise ValueError(
                "EYAS_DATASET_ROOT/DATASET_ROOT env var not set and data_dir not passed explicitly"
            )
        data_dir = Path(root) / DATASET_CONFIGS[dataset]["data_dir"]
    if images_dir is None:
        if root is None:
            raise ValueError(
                "EYAS_DATASET_ROOT/DATASET_ROOT env var not set and images_dir not passed explicitly"
            )
        images_dir = Path(root) / DATASET_CONFIGS[dataset]["images_dir"]
    return Path(data_dir), Path(images_dir)


def extract_choices(samples: list[dict]) -> list[str]:
    """A group's choice list: labels by descending frequency
    (Counter.most_common, ties keep first-occurrence order)."""
    label_counts = Counter(sample["label"] for sample in samples)
    return [label for label, _ in label_counts.most_common()]


def load_group_split(
    dataset: str,
    split: str,
    data_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> tuple[dict[str, list[dict]], Path]:
    """Load all groups of one split; returns ({group: [sample dicts]}, images_dir).

    Groups are sorted by filename; samples within a group keep JSON array
    order.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got: '{split}'")
    data_dir_resolved, images_dir_resolved = resolve_dataset_paths(dataset, data_dir, images_dir)
    datasets: dict[str, list[dict]] = {}
    for json_file in sorted(data_dir_resolved.glob(f"*_{split}.json")):
        # removesuffix, not replace: a group name containing "_train" (e.g.
        # foo_train_bar) must survive stem parsing intact
        group_name = json_file.stem.removesuffix(f"_{split}")
        with open(json_file, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(
                f"{json_file} top level must be a JSON array, got {type(items).__name__}"
            )
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{json_file}[{idx}] (group '{group_name}') must be a JSON object, "
                    f"got {type(item).__name__}"
                )
        datasets[group_name] = items
    if not datasets:
        raise FileNotFoundError(f"no *_{split}.json found under {data_dir_resolved}")
    return datasets, images_dir_resolved


def build_mcq_samples(datasets: dict[str, list[dict]], images_dir: str | Path) -> list[Sample]:
    """Turn the group structure into a list of Samples (choices computed per
    group; id = filename for cross-run stability). Malformed records raise with
    the group and record index (same contract as load_vqa_samples). A filename
    repeated across (or within) groups raises naming the id and both groups —
    the id is the evaluation-cache key, so a collision must never silently
    share one cached evaluation between two samples."""
    samples: list[Sample] = []
    seen: dict[str, str] = {}  # sample id -> first group that carried it
    for group_name, group_samples in datasets.items():
        for idx, item in enumerate(group_samples):
            for key in ("filename", "label"):
                if key not in item:
                    raise ValueError(
                        f"MCQ group '{group_name}'[{idx}] missing required field '{key}'"
                    )
        choices = extract_choices(group_samples)
        for item in group_samples:
            sample_id = item["filename"]
            if sample_id in seen:
                raise ValueError(
                    f"duplicate MCQ sample id {sample_id!r}: groups '{seen[sample_id]}' and "
                    f"'{group_name}' (ids are filenames and must be unique across groups — "
                    "rename the colliding files so each sample caches its own evaluation)"
                )
            seen[sample_id] = group_name
            samples.append(
                Sample(
                    id=sample_id,
                    inputs={
                        "image": str(Path(images_dir) / item["filename"]),
                        "choices": choices,
                        "group": group_name,
                    },
                    references={"label": item["label"]},
                )
            )
    return samples


def load_mcq_samples(
    dataset: str,
    split: str,
    data_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    *,
    per_group: int = 0,
    sample_seed: int = 42,
) -> list[Sample]:
    """One-step load: group-based MCQ split -> list of Samples.

    When per_group > 0, stratified sampling by (group, label) quotas is
    applied (two-level seed derivation: Random(sample_seed).randint(0,
    2**31-1) seeds stratified_sample_train); sampling happens on the group
    structure before Sample construction.
    """
    datasets, images_dir_resolved = load_group_split(dataset, split, data_dir, images_dir)
    if per_group > 0:
        datasets = stratified_sample_train(
            datasets, per_group=per_group, seed=derive_sample_seed(sample_seed)
        )
    return build_mcq_samples(datasets, images_dir_resolved)


def _label_quotas(
    buckets: dict[str, list[dict]], labels_sorted: list[str], per_group: int
) -> dict[str, int]:
    """Per-label quotas for one group when num_labels <= per_group: one label
    guarantees one slot, the rest is distributed proportionally to each label's
    spare records, and the integer-rounding deficit goes to the largest
    fractional shares (ties: more spare records first).

    Unreachability argument: callers only reach here when the group is
    strictly longer than per_group (else it passed through) and
    num_labels <= per_group, so spare_total = len(samples) - num_labels >= 1
    and every deficit slot lands on a label with a positive fractional share —
    those have spare >= 1 and raw share < spare, so a quota can never exceed
    its bucket size. No overflow redistribution is needed, so none is shipped.
    """
    quotas: dict[str, int] = {label: 1 for label in labels_sorted}
    remaining = per_group - len(labels_sorted)
    if remaining <= 0:
        return quotas

    available = {label: len(buckets[label]) - 1 for label in labels_sorted}
    available_total = sum(available.values())
    raw_share = {label: remaining * available[label] / available_total for label in labels_sorted}
    integer_share = {label: int(raw_share[label]) for label in labels_sorted}
    for label in labels_sorted:
        quotas[label] += integer_share[label]

    deficit = remaining - sum(integer_share.values())
    if deficit > 0:
        leftovers = sorted(
            labels_sorted,
            key=lambda label: (raw_share[label] - integer_share[label], available[label]),
            reverse=True,
        )
        for label in leftovers[:deficit]:
            quotas[label] += 1
    return quotas


def stratified_sample_train(
    datasets: dict[str, list[dict]], per_group: int, seed: int = 42
) -> dict[str, list[dict]]:
    """Stratified sampling of the train split by (group, label).

    With the same seed the subset and order are deterministic (one rng stream
    in group order), which resume and run-to-run comparison rely on; see
    _label_quotas for why no overflow redistribution is needed.
    """
    if per_group <= 0:
        return datasets

    rng = random.Random(seed)
    sampled: dict[str, list[dict]] = {}

    for group_name, samples in datasets.items():
        if len(samples) <= per_group:
            sampled[group_name] = list(samples)
            continue

        buckets: dict[str, list[dict]] = {}
        for sample in samples:
            buckets.setdefault(sample["label"], []).append(sample)
        labels_sorted = sorted(buckets.keys())

        if len(labels_sorted) > per_group:
            chosen_labels = rng.sample(labels_sorted, per_group)
            sampled[group_name] = [rng.choice(buckets[label]) for label in chosen_labels]
            continue

        quotas = _label_quotas(buckets, labels_sorted, per_group)
        picked: list[dict] = []
        for label in labels_sorted:
            picked.extend(rng.sample(buckets[label], quotas[label]))
        rng.shuffle(picked)
        sampled[group_name] = picked

    return sampled


def derive_sample_seed(train_sample_seed: int) -> int:
    """Two-level seed derivation: Random(train_sample_seed).randint(0, 2**31-1)."""
    return random.Random(train_sample_seed).randint(0, 2**31 - 1)


# ---------------------------------------------------------------------------
# VQA open-ended QA format (used by tasks/vqa.py)
# ---------------------------------------------------------------------------
def load_vqa_samples(data_dir: str | Path, split: str, images_dir: str | Path) -> list[Sample]:
    """Load a VQA split: ``data_dir/<split>.json`` (JSON array) -> list of Samples.

    Required fields per element: ``image`` (relative to images_dir),
    ``question``, ``answers`` (list of annotated answers, >=1, VQAv2 uses
    10); ``id`` is optional and defaults to ``<image name>#<index>`` (so
    multiple questions on one image do not collide). Missing files/fields
    raise immediately.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got: '{split}'")
    path = Path(data_dir) / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"VQA split file does not exist: {path}")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} top level must be a JSON array, got {type(items).__name__}")

    samples: list[Sample] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object, got {type(item).__name__}")
        for key in ("image", "question", "answers"):
            if key not in item:
                raise ValueError(f"{path}[{idx}] missing required field '{key}'")
        answers = item["answers"]
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{path}[{idx}] answers must be a non-empty list")
        sample_id = item.get("id") or f"{Path(item['image']).stem}#{idx}"
        samples.append(
            Sample(
                id=str(sample_id),
                inputs={
                    "image": str(Path(images_dir) / item["image"]),
                    "question": str(item["question"]),
                },
                references={"answers": [str(a) for a in answers]},
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Video MCQ format (used by tasks/video_mcq.py)
# ---------------------------------------------------------------------------
def load_video_mcq_samples(
    data_dir: str | Path, split: str, videos_dir: str | Path
) -> list[Sample]:
    """Load a video MCQ split: ``data_dir/<split>.json`` (JSON array) -> list of Samples.

    Required fields per element: ``video`` (relative to videos_dir),
    ``question``, ``choices`` (list, >=2) and ``answer`` (a member of
    ``choices``); ``id`` is optional and defaults to ``<video stem>#<index>``.
    The answer/choices relation is validated at load time so a broken label
    never silently scores 0 for a whole run. Missing files/fields raise
    immediately.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got: '{split}'")
    path = Path(data_dir) / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"video MCQ split file does not exist: {path}")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} top level must be a JSON array, got {type(items).__name__}")

    samples: list[Sample] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object, got {type(item).__name__}")
        for key in ("video", "question", "choices", "answer"):
            if key not in item:
                raise ValueError(f"{path}[{idx}] missing required field '{key}'")
        raw_choices = item["choices"]
        if not isinstance(raw_choices, list) or len(raw_choices) < 2:
            raise ValueError(f"{path}[{idx}] choices must be a list of at least 2 entries")
        choices = [str(choice) for choice in raw_choices]
        answer = str(item["answer"])
        sample_id = item.get("id") or f"{Path(item['video']).stem}#{idx}"
        if answer not in choices:
            raise ValueError(
                f"{path}[{idx}] sample {sample_id!r}: answer {answer!r} is not one of "
                f"the choices {choices}"
            )
        samples.append(
            Sample(
                id=str(sample_id),
                inputs={
                    "video": str(Path(videos_dir) / item["video"]),
                    "question": str(item["question"]),
                    "choices": choices,
                },
                references={"label": answer},
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Mixed-modality MCQ format (used by tasks/mixed_mcq.py)
# ---------------------------------------------------------------------------
# combo = the non-empty subset of {text, image, video} one sample carries; the
# canonical spelling is the components in this order joined by "_". "text" is
# the sample's own passage (instruction/question/choices are the query and are
# always present, so they never count as a modality).
MIXED_COMBOS = (
    "text",
    "image",
    "video",
    "text_image",
    "text_video",
    "image_video",
    "text_image_video",
)


def _combo_modalities(combo: str) -> tuple[bool, bool, bool]:
    """(has_text, has_image, has_video) declared by a mixed combo name."""
    parts = set(combo.split("_"))
    return ("text" in parts, "image" in parts, "video" in parts)


def _modality_flags(flags: tuple[bool, bool, bool]) -> str:
    return "/".join("yes" if flag else "no" for flag in flags)


def _resolve_media_paths(
    media_root: Path, entries: list[Any], kind: str, sample_id: str
) -> list[str]:
    """Absolute media paths for one modality (missing files fail immediately)."""
    paths: list[str] = []
    for entry in entries:
        path = Path(media_root) / str(entry)
        if not path.is_file():
            raise FileNotFoundError(f"mixed sample {sample_id!r}: {kind} file not found: {path}")
        paths.append(str(path))
    return paths


def load_mixed_mcq_samples(
    data_dir: str | Path, split: str, media_root: str | Path | None = None
) -> list[Sample]:
    """Load a mixed-modality MCQ split: ``data_dir/<split>.json`` (JSON array)
    -> list of Samples.

    Required fields per element: ``combo`` (one of MIXED_COMBOS), ``question``,
    ``choices`` (list, >=2) and ``answer`` (a member of ``choices``); the
    payload must match the declared combo: ``passage`` non-blank iff the combo
    names text, ``images`` / ``videos`` non-empty lists iff it names image /
    video. Media paths are relative to ``media_root`` (default:
    ``Path(data_dir).parent``) and every listed file must exist. ``id`` is
    optional and defaults to ``<split>#<index>``.

    Sample convention: inputs={"passage", "images", "videos", "question",
    "choices", "combo"}, references={"label": answer}. Every check raises with
    the sample id, so a malformed record identifies itself instead of scoring 0
    for a whole run.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got: '{split}'")
    path = Path(data_dir) / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"mixed MCQ split file does not exist: {path}")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} top level must be a JSON array, got {type(items).__name__}")

    root = Path(media_root) if media_root is not None else Path(data_dir).parent
    samples: list[Sample] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object, got {type(item).__name__}")
        for key in ("combo", "question", "choices", "answer"):
            if key not in item:
                raise ValueError(f"{path}[{idx}] missing required field '{key}'")
        sample_id = str(item.get("id") or f"{split}#{idx}")
        raw_choices = item["choices"]
        if not isinstance(raw_choices, list) or len(raw_choices) < 2:
            raise ValueError(
                f"mixed sample {sample_id!r}: choices must be a list of at least 2 entries"
            )
        choices = [str(choice) for choice in raw_choices]
        combo = str(item["combo"])
        if combo not in MIXED_COMBOS:
            raise ValueError(
                f"mixed sample {sample_id!r}: unknown combo {combo!r}, "
                f"expected one of {MIXED_COMBOS}"
            )
        passage = str(item.get("passage") or "")
        raw_images = item.get("images") or []
        raw_videos = item.get("videos") or []
        for key, value in (("images", raw_images), ("videos", raw_videos)):
            if not isinstance(value, list):
                raise ValueError(
                    f"mixed sample {sample_id!r}: {key} must be a list, got {type(value).__name__}"
                )
        declared = _combo_modalities(combo)
        present = (bool(passage.strip()), bool(raw_images), bool(raw_videos))
        if declared != present:
            raise ValueError(
                f"mixed sample {sample_id!r}: combo {combo!r} declares "
                f"text/image/video={_modality_flags(declared)} but payload has "
                f"{_modality_flags(present)}"
            )
        images = _resolve_media_paths(root, raw_images, "image", sample_id)
        videos = _resolve_media_paths(root, raw_videos, "video", sample_id)
        if len(videos) > 1:
            raise ValueError(
                f"mixed sample {sample_id!r}: at most one clip per sample, got {len(videos)} "
                "(the planner charges 16384 tok per clip, so two clips cannot fit a 32k window "
                "and would be rejected at plan time and scored 0)"
            )
        answer = str(item["answer"])
        if answer not in choices:
            raise ValueError(
                f"mixed sample {sample_id!r}: answer {answer!r} is not one of the choices {choices}"
            )
        samples.append(
            Sample(
                id=sample_id,
                inputs={
                    "passage": passage,
                    "images": images,
                    "videos": videos,
                    "question": str(item["question"]),
                    "choices": choices,
                    "combo": combo,
                },
                references={"label": answer},
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Numeric math QA format (used by tasks/math_qa.py)
# ---------------------------------------------------------------------------
def load_math_qa_samples(data_dir: str | Path, split: str) -> list[Sample]:
    """Load a math QA split: ``data_dir/<split>.json`` (JSON array) -> list of Samples.

    Required fields per element: ``question`` and ``answer``; ``answer`` is
    already the final numeric value (GSM8K's ``####`` marker is stripped by
    the preparation script). ``id`` is optional and defaults to
    ``<split>#<index>``. Missing files/fields raise immediately.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got: '{split}'")
    path = Path(data_dir) / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"math QA split file does not exist: {path}")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} top level must be a JSON array, got {type(items).__name__}")

    samples: list[Sample] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object, got {type(item).__name__}")
        for key in ("question", "answer"):
            if key not in item:
                raise ValueError(f"{path}[{idx}] missing required field '{key}'")
        sample_id = item.get("id") or f"{split}#{idx}"
        samples.append(
            Sample(
                id=str(sample_id),
                inputs={"question": str(item["question"])},
                references={"answer": str(item["answer"])},
            )
        )
    return samples
