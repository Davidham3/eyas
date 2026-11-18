"""KIE (Key Information Extraction) structured extraction task (framework-level general capability).

The extraction-family counterpart of McqTask (the classification-family
general form): image -> extract a list of entity records per the field spec
(``[{"name": ..., "phonenum": ...}, ...]``), with judge-free rule-based
scoring. Data format follows VqaTask's ``data_dir/<split>.json`` convention:
a JSON array whose elements are
``{"image": absolute path, "entities": [{field: value}, ...], "id": optional}``;
``entities: []`` is a valid annotation (zero-entity samples are kept, and
empty-vs-empty scores a perfect mark).

Semantic notes:
- **Three metrics**: ``field_acc`` (primary metric — after greedy alignment,
  hit field count / total reference field count + penalty fields from
  unaligned predictions; when counts differ, the symmetric denominator
  discounts both sides), ``strict_acc`` (counts match, each aligned pair has
  equal field sets, and every field hits), ``char_sim`` (difflib character
  similarity of aligned field pairs, zero-dependency stdlib — gives the
  optimizer a dense gradient beyond 0/1). All three satisfy the perfect-score
  consistency invariant: field_acc=1 ⟺ strict_acc=1 ⟺ char_sim=1
  (the protegi error-pool criterion "any maximize metric <1.0" relies on
  this invariant being self-consistent);
- **Normalization**: per-field config (``field_normalizers``; default
  phonenum -> digits keeps digits only, everything else -> text strips
  ASCII and Chinese punctuation and collapses whitespace/lowercase), applied
  the same way on both sides;
- **Parse tolerance**: whole JSON -> first ``[...]`` block -> first ``{...}``
  block (when the model drops the array shell, fall back to a single
  record); scalar values coerced to str; structurally invalid -> value=None
  (parse_fail is booked uniformly by the evaluator, scoring 0 without a
  judge);
- **Compatibility fields** (load-bearing, do not change semantics):
  ``references["label"]`` = serialized JSON of the entities — shared by the
  protegi gradient rendering of the ground-truth answer (protegi.py error
  pool) and the spo few-shot answers.
  ``inputs["group"]`` = entity-count bucket (n0/n1/n2+) — spo group_cycle
  sampling cycles on the first entry of each bucket, and 3 buckets keep
  few-shot diverse with short questions;
- build_request is a single forgiving renderer (the shape shared by the task family):
  ``<image>`` is stripped; ``{choices}`` is filled with the field-spec JSON;
  when an optimizer rewrite loses the placeholder, a spec line is appended
  at the end; ``{image}`` splits in place to insert ImagePart, and the image
  is prepended when missing.
"""

from __future__ import annotations

import difflib
import json
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from eyas.backends.base import ChatRequest, ChatResponse, ImagePart, TextPart, user_message
from eyas.core.objective import Objective
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.datasets import VALID_SPLITS

__all__ = [
    "KieTask",
    "load_kie_samples",
    "normalize_field_value",
    "parse_kie_entities",
    "score_entities",
]

# Strip ASCII and Chinese punctuation together (string.punctuation does not
# include Chinese marks; characters like the middle dot or full-width
# parentheses in store names/addresses do not affect semantic equivalence)
_PUNCT_CHARS = string.punctuation + "，。、；：？！“”‘’（）《》〈〉【】〔〕·…—～－·"
_PUNCT_TABLE = str.maketrans("", "", _PUNCT_CHARS)

NORMALIZER_KINDS = ("text", "digits")

# field_normalizers defaults: phone fields use digits (digits only — separator
# and country-code spelling differences do not affect equality), other fields
# use text. A task's `field_normalizers` param may override wholesale.
DEFAULT_NORMALIZERS: dict[str, str] = {"phonenum": "digits"}

# Placeholder value for fields in the {choices} spec (rendering generates the real spec from the split's field union)
_SPEC_PLACEHOLDER = "..."


def normalize_field_value(value: str, kind: str) -> str:
    """Field value normalization: digits keeps digits only (phone); text
    strips punctuation and collapses whitespace/lowercase (store name/address)."""
    if kind == "digits":
        return "".join(ch for ch in value if ch.isdigit())
    if kind == "text":
        return " ".join(value.lower().translate(_PUNCT_TABLE).split())
    raise ValueError(f"Unknown normalizer kind {kind!r} (options: {NORMALIZER_KINDS})")


def _coerce_scalar(value: Any) -> str | None:
    """Scalar value -> str (numbers coerced); non-scalars (dict/list/bool/None)
    return None.

    The parse side treats None as structurally invalid (tolerant, yields
    parse_fail); the loader side escalates None to fail fast — same check,
    two levels of semantics."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _coerce_record(item: Any, single_field: str | None) -> dict[str, str] | None:
    """One parsed JSON element -> entity record; non-coercible -> None
    (parse_fail is booked by the caller, parse_kie_entities).

    A bare string is wrapped as a record under the single field name — with
    a single-field spec the model often outputs `["address string"]` instead
    of `[{"field": "address string"}]`. A
    string is already a scalar, so it passes through verbatim; only dict
    values need scalar coercion. Keys are used as-is: every dict here comes
    from json.loads, whose keys are always str.
    """
    if isinstance(item, str):
        if single_field is None:
            return None
        return {single_field: item}
    if not isinstance(item, dict) or not item:
        return None
    record: dict[str, str] = {}
    for key, value in item.items():
        coerced = _coerce_scalar(value)
        if coerced is None:
            return None
        record[key] = coerced
    return record


def _coerce_records(parsed: Any, single_field: str | None = None) -> list[dict[str, str]] | None:
    """JSON parse result -> list[dict[str, str]]; a single object (or, with
    single_field given, a bare string) falls back to a one-record list;
    structurally invalid -> None, all-or-nothing.
    """
    if isinstance(parsed, dict):
        parsed = [parsed]
    elif isinstance(parsed, str):
        if single_field is None:
            return None
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None
    records: list[dict[str, str]] = []
    for item in parsed:
        record = _coerce_record(item, single_field)
        if record is None:
            return None
        records.append(record)
    return records


def _copy_string_literal(text: str, start: int, out: list[str]) -> int:
    """Copy the string literal opening at ``text[start]`` (a ``"``) into
    ``out`` through its closing quote (escape-aware), returning the index
    just past the closer — or the end of text when unterminated. Brackets
    inside the literal are copied verbatim: the repair never touches them.
    """
    out.append('"')
    idx = start + 1
    escaped = False
    while idx < len(text):
        ch = text[idx]
        out.append(ch)
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return idx + 1
        idx += 1
    return idx


def _emit_closer(ch: str, stack: list[str], out: list[str]) -> None:
    """Emit a ``}``/``]`` against the stack top: matching closer kept,
    wrong-type closer replaced by the expected one, extra closer (empty
    stack) dropped.
    """
    if stack and stack[-1] == ch:
        stack.pop()
        out.append(ch)
    elif stack:
        out.append(stack.pop())
    # extra closer with empty stack: drop it


def _balance_brackets(candidate: str) -> str:
    """JSON bracket-balance repair: wrong-type closers replaced per the
    stack-top expectation, extra closers dropped, tail closers appended.

    Handles the recurring malformation family of short-field records (an
    object closed with the wrong bracket type, e.g. `{"address": "…"]`);
    brackets inside string literals are left untouched. If unrepairable, the
    text is returned as-is and the caller's json.loads fails naturally into
    parse_fail.
    """
    out: list[str] = []
    stack: list[str] = []
    idx = 0
    while idx < len(candidate):
        ch = candidate[idx]
        if ch == '"':
            idx = _copy_string_literal(candidate, idx, out)
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
        elif ch in "}]":
            _emit_closer(ch, stack, out)
        else:
            out.append(ch)
        idx += 1
    while stack:
        out.append(stack.pop())
    return "".join(out)


def parse_kie_entities(
    text: str, *, single_field: str | None = None
) -> list[dict[str, str]] | None:
    """Model output -> list of entity records; tolerates code fences, leading/
    trailing noise, missing array shell, wrong-type closers, and bare strings
    under a single field; structurally invalid -> None."""
    candidates = [text.strip()]
    start, end = text.find("["), text.rfind("]")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_balance_brackets(candidate))
            except json.JSONDecodeError:
                continue
        records = _coerce_records(parsed, single_field)
        if records is not None:
            return records
    return None


def _normalized_fields(
    records: Sequence[Mapping[str, str]], normalizers: Mapping[str, str]
) -> list[dict[str, str]]:
    """One scoring side normalized per-field (default text) — applied the
    same way to reference and prediction."""
    return [
        {
            field: normalize_field_value(value, normalizers.get(field, "text"))
            for field, value in record.items()
        }
        for record in records
    ]


def _pair_agreement(ref: Mapping[str, str], pred: Mapping[str, str]) -> tuple[int, float]:
    """(exact hit field count, summed character similarity) for one
    normalized ref/pred record pair; a ref field the prediction omits
    contributes neither a hit nor similarity."""
    hits = 0
    similarity = 0.0
    for field, ref_value in ref.items():
        pred_value = pred.get(field)
        if pred_value is None:
            continue
        if pred_value == ref_value:
            hits += 1
            similarity += 1.0
        else:
            similarity += difflib.SequenceMatcher(None, ref_value, pred_value).ratio()
    return hits, similarity


def _align_records(
    ref_fields: list[dict[str, str]], pred_fields: list[dict[str, str]]
) -> list[tuple[int, int]]:
    """Greedy one-to-one alignment over all (ref_i, pred_j) pairs by
    descending "alignment score = (exact hit count + total char similarity) /
    reference field count" (ties break by stable (i, j) order); pairs with
    score > 0 enter the alignment — an approximately-correct OCR string
    (zero exact hits) still aligns and earns partial char_sim credit.
    """
    ratio_pairs: list[tuple[float, int, int]] = []
    for i in range(len(ref_fields)):
        for j in range(len(pred_fields)):
            hits, similarity = _pair_agreement(ref_fields[i], pred_fields[j])
            ratio = (hits + similarity) / len(ref_fields[i])
            if ratio > 0:
                ratio_pairs.append((-ratio, i, j))
    ratio_pairs.sort()
    used_i: set[int] = set()
    used_j: set[int] = set()
    aligned: list[tuple[int, int]] = []
    for _, i, j in ratio_pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        aligned.append((i, j))
    return aligned


def _metric_totals(
    ref_fields: list[dict[str, str]],
    pred_fields: list[dict[str, str]],
    aligned: list[tuple[int, int]],
) -> tuple[int, int, int, float]:
    """Single pass over the alignment: (matched hit fields, extra predicted
    fields within aligned pairs, all fields of unaligned predictions, summed
    character similarity)."""
    used_j = {j for _, j in aligned}
    pair_agreements = [_pair_agreement(ref_fields[i], pred_fields[j]) for i, j in aligned]
    matched = sum(agree[0] for agree in pair_agreements)
    # sum() over the list keeps a single accumulation order across the per-pair
    # similarities (an incremental 0.0-float += re-rounds the first addend and
    # can shift a score by 1 ULP)
    similarity = sum(agree[1] for agree in pair_agreements)
    extra_aligned = sum(len(set(pred_fields[j]) - set(ref_fields[i])) for i, j in aligned)
    unaligned_pred = sum(len(pred) for j, pred in enumerate(pred_fields) if j not in used_j)
    return matched, extra_aligned, unaligned_pred, similarity


def _strict_hit(
    ref_fields: list[dict[str, str]],
    pred_fields: list[dict[str, str]],
    aligned: list[tuple[int, int]],
) -> bool:
    """strict_acc: counts match, each aligned pair has equal field sets, and
    every reference field of the pair hits exactly."""
    if len(aligned) != len(ref_fields) or len(aligned) != len(pred_fields):
        return False
    return all(
        set(pred_fields[j]) == set(ref_fields[i])
        and _pair_agreement(ref_fields[i], pred_fields[j])[0] == len(ref_fields[i])
        for i, j in aligned
    )


def score_entities(
    pred: list[dict[str, str]] | None,
    ref: list[dict[str, str]],
    normalizers: Mapping[str, str],
) -> dict[str, float]:
    """Greedy-alignment scoring (module-level pure function, easy to unit test).

    Denominator = total reference fields + extra predicted fields within
    aligned pairs + all fields of unaligned predictions (symmetric penalty:
    both over- and under-reporting lose score). Alignment and metrics live
    in the helpers above; the perfect-score invariant (field_acc=1 ⟺
    strict_acc=1 ⟺ char_sim=1) holds by construction of those helpers.
    An empty entity record on either side raises ValueError naming the side
    and record index (fail fast — it has no scorable fields and would divide
    the alignment by zero).
    """
    zero = {"field_acc": 0.0, "strict_acc": 0.0, "char_sim": 0.0}
    if pred is None:
        return zero
    if not ref and not pred:
        return {"field_acc": 1.0, "strict_acc": 1.0, "char_sim": 1.0}
    for side, records in (("ref", ref), ("pred", pred)):
        for idx, record in enumerate(records):
            if not record:
                raise ValueError(
                    f"score_entities: {side} record [{idx}] is empty — entity records must "
                    "carry at least one field (an empty record has no scorable fields)"
                )

    ref_fields = _normalized_fields(ref, normalizers)
    pred_fields = _normalized_fields(pred, normalizers)
    aligned = _align_records(ref_fields, pred_fields)
    matched, extra_aligned, unaligned_pred, similarity = _metric_totals(
        ref_fields, pred_fields, aligned
    )
    denominator = sum(len(record) for record in ref_fields) + extra_aligned + unaligned_pred

    return {
        "field_acc": matched / denominator,
        "strict_acc": float(_strict_hit(ref_fields, pred_fields, aligned)),
        "char_sim": similarity / denominator,
    }


def _read_kie_records(path: Path) -> list[Any]:
    """File-read phase: split file -> raw record list; a non-array top level
    fails fast naming the file."""
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError(f"{path} must be a JSON array")
    return records


def _coerce_entity(entity: Any, origin: str) -> dict[str, str]:
    """One annotated entity object -> all-str record, escalating non-scalars
    to a fail fast naming ``origin``, the field and the offending type
    (the loader's stance where the parse side would yield parse_fail).

    Keys are used as-is: every dict here comes from json.load, whose keys
    are always str.
    """
    if not isinstance(entity, dict) or not entity:
        raise ValueError(f"{origin} must be a non-empty object")
    coerced_entity: dict[str, str] = {}
    for key, value in entity.items():
        coerced = _coerce_scalar(value)
        if coerced is None:
            raise ValueError(
                f"{origin} field {key!r} value must be a string/numeric scalar, "
                f"got {type(value).__name__}"
            )
        coerced_entity[key] = coerced
    return coerced_entity


def _parse_kie_record(record: Any, path: Path, idx: int) -> tuple[str, list[dict[str, str]]]:
    """Record-parse phase (parse once at the data boundary): one annotated
    record -> (image path, entity records), every corruption raising an
    actionable error that names ``path[idx]``."""
    origin = f"{path}[{idx}]"
    if not isinstance(record, dict):
        raise ValueError(f"{origin} must be an object")
    image = record.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"{origin} missing image (non-empty string path)")
    raw_entities = record.get("entities")
    if not isinstance(raw_entities, list):
        raise ValueError(f"{origin} missing entities (array, may be empty)")
    entities = [
        _coerce_entity(entity, f"{origin}.entities[{e_idx}]")
        for e_idx, entity in enumerate(raw_entities)
    ]
    if not Path(image).is_file():
        raise ValueError(f"{origin} image does not exist: {image}")
    return image, entities


def _assemble_kie_sample(
    split: str, idx: int, image: str, entities: list[dict[str, str]], fields: list[str]
) -> Sample:
    """Sample-assembly phase: stable cache-key id, entity-count bucket group
    (spo group_cycle cycles on it), JSON-serialized label (protegi error pool
    + spo few-shot answer)."""
    count = len(entities)
    group = "n0" if count == 0 else ("n1" if count == 1 else "n2+")
    return Sample(
        id=f"{split}_{idx:04d}",
        inputs={"image": image, "fields": fields, "group": group},
        references={"entities": entities, "label": json.dumps(entities, ensure_ascii=False)},
    )


def load_kie_samples(data_dir: str | Path, split: str) -> list[Sample]:
    """``data_dir/<split>.json`` array -> list of Samples, in three phases —
    file-read (``_read_kie_records``), record-parse (``_parse_kie_record``:
    the single boundary where annotation structure is checked and values
    coerced, failing fast with the row index), sample-assembly
    (``_assemble_kie_sample``).

    fields = union of all entity field names in the split (input for the
    {choices} spec rendering).
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got: '{split}'")
    path = Path(data_dir) / f"{split}.json"
    parsed = [
        _parse_kie_record(record, path, idx) for idx, record in enumerate(_read_kie_records(path))
    ]
    fields = sorted({f for _, entities in parsed for entity in entities for f in entity})
    samples = [
        _assemble_kie_sample(split, idx, image, entities, fields)
        for idx, (image, entities) in enumerate(parsed)
    ]
    if not samples:
        raise ValueError(f"{path} loaded no samples")
    return samples


@register_task("kie")
class KieTask(Task):
    """KIE structured extraction task (schema-driven + judge-free rule scoring).

    Sample convention: inputs={"image": path, "fields": field union,
    "group": entity-count bucket}, references={"entities": list of entity
    records, "label": serialized JSON}.
    """

    name = "kie"
    objectives: ClassVar[tuple[Objective, ...]] = (
        Objective(key="field_acc", maximize=True, display_name="field_acc"),
        Objective(key="strict_acc", maximize=True, display_name="strict_acc"),
        Objective(key="char_sim", maximize=True, display_name="char_sim"),
    )
    default_prompt = (
        "{image}Extract the following fields from this image: {choices}\n"
        "Respond with a JSON array of entity records. Output [] if none are present."
    )

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        field_normalizers: Mapping[str, str] | None = None,
        samples: list[Sample] | None = None,
    ) -> None:
        if samples is None and data_dir is None:
            raise ValueError("KieTask requires one of data_dir or explicit samples")
        normalizers = dict(DEFAULT_NORMALIZERS if field_normalizers is None else field_normalizers)
        for field, kind in normalizers.items():
            if kind not in NORMALIZER_KINDS:
                raise ValueError(
                    f"field_normalizers[{field!r}] must be one of {NORMALIZER_KINDS}, got {kind!r}"
                )
        self._data_dir = data_dir
        self._field_normalizers = normalizers
        self._samples = samples
        self._split_cache: dict[str, list[Sample]] = {}

    def load_split(self, split: str) -> list[Sample]:
        if self._samples is not None:
            return list(self._samples)
        if split not in self._split_cache:
            self._split_cache[split] = load_kie_samples(self._data_dir or "", split)
        return list(self._split_cache[split])

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        instruction = prompt.replace("<image>", "")
        fields: Sequence[str] = sample.inputs.get("fields") or ()
        spec = json.dumps([{f: _SPEC_PLACEHOLDER for f in fields}], ensure_ascii=False)
        if "{choices}" in instruction:
            instruction = instruction.replace("{choices}", spec)
        else:
            instruction = f"{instruction}\nOutput format: {spec}"
        if "{image}" in instruction:
            parts: list[ImagePart | TextPart] = []
            for idx, segment in enumerate(instruction.split("{image}")):
                if idx:
                    parts.append(ImagePart(image=sample.inputs["image"]))
                if segment:
                    parts.append(TextPart(segment))
        else:
            parts = [ImagePart(image=sample.inputs["image"]), TextPart(instruction)]
        return ChatRequest(
            messages=(user_message(*parts),),
            request_id=f"{sample.id}:{variant.id}",
        )

    def demo_media_parts(self, sample: Sample) -> list[ImagePart]:
        """A few-shot demo carries the sample image, like evaluation."""
        return [ImagePart(image=sample.inputs["image"])]

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        fields = list(sample.inputs.get("fields") or ())
        single_field = fields[0] if len(fields) == 1 else None
        return Prediction(
            value=parse_kie_entities(response.text, single_field=single_field),
            raw=response.text,
        )

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return score_entities(
            prediction.value, sample.references["entities"], self._field_normalizers
        )
