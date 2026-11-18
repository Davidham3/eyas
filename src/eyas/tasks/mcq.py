"""MCQ multiple-choice classification task.

Semantic notes (the rendering is byte-stable, pinned by the golden tests):
- the classification instruction (candidate prompt) + choice list render
  into the user turn with the media part first; choice format
  ``Choices: ['a', 'b']`` (``", ".join(f"'{c}'")``);
- the media part comes from the ``_media_part`` hook: the base task returns
  an ``ImagePart`` built from the ``media_key`` sample input (default
  ``"image"``), subclasses override the hook to emit another part type
  (``VideoMcqTask`` returns a ``VideoPart``);
- ``{image}`` / ``<image>`` markers in the candidate prompt are stripped
  before rendering;
- parse_response uses ``match_prediction``'s deterministic matching rules to
  map free text back to a choice name; the annotation-free judge (LLM
  fallback normalization) lives in judges/mcq_judge.py.

The choice list (descending within-group frequency) is computed at dataset
load time for the group-based image datasets; see tasks/datasets.py. Other
media datasets plug their own split loader in through ``loader`` (called as
``loader(data_dir, split, media_dir)``, where ``media_dir`` is ``videos_dir``
when given and ``images_dir`` otherwise), so a subclass needs no
``load_split`` override.

OPRO scorer assembly (enabled when ``instruction_pos`` is not None;
gen_scorer_prompt ported from the OPRO reference port (upstream:
google-deepmind/opro) and split into
per-position builders rendering byte-identical strings; unknown positions
raise instead of the port's silent fall-through to the A_begin layout): fixed
question line + instruction placed in one of 4 positions (optional Q/A
wrapping), always ending with the target label (``Target Image:``; the
``question`` / ``target`` keyword parameters let a subclass pin another text
and label, e.g. a per-sample video question and ``Target Video:``; their
defaults are today's image bytes). ``target_image_suffix`` has no effect in
this mode. Use the matching judge
``McqLLMJudge(variant="opro", classify_original_task=True)``.

SPO full-template assembly (enabled with ``raw_template=True``;
SPO reference-port semantics): the candidate prompt is a complete user-text
template (with ``{image}`` / ``{choices}`` placeholders; the optimizer may
rearrange the whole text). At render time ``{choices}`` is filled with the
sample's choices (same ``", ".join(f"'{c}'"`` format); the media marker is
replaced in place by the media part (original build_chat_prompt rule b), and
the media part is prepended when the template has none (rule c). Subclasses
may require extra placeholders (``{question}`` for video) and search their
own marker set. Use the matching judge
``McqLLMJudge(variant="cot", classify_original_task=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    Part,
    TextPart,
    assistant_message,
    user_message,
)
from eyas.core.objective import Objective
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.datasets import load_mcq_samples

__all__ = ["McqTask", "gen_scorer_prompt", "match_prediction"]

# Image placeholder markers that may appear in candidate prompts (stripped at
# render time; image placement is handled uniformly by the backend)
_IMAGE_MARKERS = ("{image}", "<image>")

_INSTRUCTION_POSITIONS = ("before_Q", "Q_begin", "Q_end", "A_begin")


# Per-position scorer-prompt builders. Each renders one instruction_pos; the
# decision-branch tests pin every byte (each builder's output is pinned
# byte-for-byte, so no rendered string can drift). A candidate prompt renders to
# head + question + tail, so each builder is one head/tail join.
def _qa_before_Q(instruction: str, question: str) -> str:
    """Instruction line above the Q: shell."""
    head = f"{instruction}\n" if instruction else ""
    return f"{head}Q: {question}\n\nA:"


def _qa_Q_begin(instruction: str, question: str) -> str:
    """Instruction inside the Q: shell, leading the question."""
    head = f"{instruction}\n" if instruction else ""
    return f"Q: {head}{question}\n\nA:"


def _qa_Q_end(instruction: str, question: str) -> str:
    """Instruction inside the Q: shell, trailing the question."""
    tail = f"\n{instruction}" if instruction else ""
    return f"Q: {question}{tail}\n\nA:"


def _qa_A_begin(instruction: str, question: str) -> str:
    """Instruction continues the answer line after ``A:``."""
    tail = f" {instruction}" if instruction else ""
    return f"Q: {question}\n\nA:{tail}"


_QA_BUILDERS: dict[str, Callable[[str, str], str]] = {
    "before_Q": _qa_before_Q,
    "Q_begin": _qa_Q_begin,
    "Q_end": _qa_Q_end,
    "A_begin": _qa_A_begin,
}


def _plain_instruction_leads(instruction: str, question: str) -> str:
    """No-QA rendering shared by before_Q/Q_begin: instruction line, then question."""
    return (f"{instruction}\n" if instruction else "") + question


def _plain_instruction_trails(instruction: str, question: str) -> str:
    """No-QA rendering shared by Q_end/A_begin: instruction trails ``Target Image:``."""
    return question + (f"\n{instruction}" if instruction else "")


def gen_scorer_prompt(
    instruction: str,
    choices: Sequence[str],
    instruction_pos: str = "Q_begin",
    include_qa: bool = False,
    *,
    question: str = "Classify the content of the target image.",
    target: str = "Target Image:",
) -> str:
    """OPRO scorer classify assembly (ported from the OPRO reference port, returns
    plain text; rendered verbatim-byte-identical through the per-position
    builders above).

    Fixed question line + choice list + target label with the instruction
    placed per instruction_pos; include_qa=True additionally wraps a
    ``Q: ... A:`` QA shell around it. ``question`` / ``target`` default to
    the image assembly's bytes; a subclass passes its own question line and
    target label (e.g. video).
    """
    if instruction_pos not in _INSTRUCTION_POSITIONS:
        raise ValueError(
            f"gen_scorer_prompt.instruction_pos must be one of {_INSTRUCTION_POSITIONS}, "
            f"got {instruction_pos!r}"
        )
    choices_str = ", ".join(f"'{c}'" for c in choices)
    question_block = f"{question}\nChoices: [{choices_str}]\n{target}"
    if include_qa:
        return _QA_BUILDERS[instruction_pos](instruction, question_block)
    if instruction_pos in {"before_Q", "Q_begin"}:
        return _plain_instruction_leads(instruction, question_block)
    return _plain_instruction_trails(instruction, question_block)


def _require_input(sample: Sample, key: str) -> Any:
    """Required sample input with an actionable error: a malformed record must
    identify itself (sample id + missing key), not leak a bare KeyError from
    the middle of a 10k-sample run."""
    try:
        return sample.inputs[key]
    except KeyError as exc:
        raise ValueError(
            f"MCQ sample {sample.id!r} is missing required input '{key}' "
            f"(present inputs: {sorted(sample.inputs)})"
        ) from exc


def _require_reference(sample: Sample, key: str) -> Any:
    """Required sample reference with an actionable error (the references-side
    twin of _require_input, same self-identifying contract)."""
    try:
        return sample.references[key]
    except KeyError as exc:
        raise ValueError(
            f"MCQ sample {sample.id!r} is missing required reference '{key}' "
            f"(present references: {sorted(sample.references)})"
        ) from exc


def _occurrences(text: str, sub: str) -> list[int]:
    hits: list[int] = []
    start = text.find(sub)
    while start != -1:
        hits.append(start)
        start = text.find(sub, start + 1)
    return hits


def _cluster_winner(text: str, cluster: list[str]) -> str:
    """Within a containment cluster, adjudicate by "independent last
    occurrence" (longest wins on ties).

    When choices are prefixes/suffixes of each other ("visible" ⊂
    "invisible"), the contained choice's many "hits" are mere side effects of
    the containing choice appearing; only occurrences not covered by any
    containing choice's span are real evidence. Take the choice whose
    independent last occurrence is latest (definitions are enumerated first,
    so longest-first matching systematically outranks the final true answer
    when one choice is a substring of another).
    """

    def independent(c: str) -> int:
        supers = [d for d in cluster if d != c and c in d]
        spans = [(s, s + len(d)) for d in supers for s in _occurrences(text, d)]
        pos = -1
        for s in _occurrences(text, c):
            if not any(a <= s < b for a, b in spans):
                pos = s
        return pos

    return max(cluster, key=lambda c: (independent(c), len(c)))


def _resolve_containment(text: str, matched: list[str]) -> list[str]:
    """Mutually-containing hit clusters (including chains a⊂b⊂c) are each
    adjudicated to one representative; the rest pass through unchanged.

    ``text`` and ``matched`` must both be lowercased so containment checks
    and occurrence spans agree with the case-insensitive substring match.

    Disjoint choices never take this path — the longest-first order has zero
    drift for CUB-like scenarios.
    """
    rel = {c: [d for d in matched if d != c and (c in d or d in c)] for c in matched}
    resolved: list[str] = []
    seen: set[str] = set()
    for c in matched:
        if c in seen:
            continue
        if not rel[c]:
            seen.add(c)
            resolved.append(c)
            continue
        cluster: list[str] = []
        stack = [c]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            cluster.append(x)
            stack.extend(rel[x])
        resolved.append(_cluster_winner(text, cluster))
    return resolved


def match_prediction(result: str, choices: list[str]) -> str | None:
    """Match a VLM's free-text output to one choice name (last-occurrence-wins
    + mutually-containing cluster adjudication).

    strip quotes/periods -> exact match -> case-insensitive substring match
    -> with multiple hits, prefer the latest occurrence in the original text,
    then the longest (last-occurrence-wins first, length as tiebreak — long
    CoT outputs often enumerate each choice's definitions first and only give
    the real answer at the end; a length tiebreak would systematically pick
    the longer choice from the definition section); one modification: when
    choices contain each other ("visible" ⊂ "invisible"), the cluster first
    elects a representative by "independent last occurrence" before joining
    the ranking — the contained choice's collateral hits are not evidence
    (_cluster_winner).
    """
    result = result.strip().strip('"').strip("'").strip(".").strip()

    if result in choices:
        return result

    result_lower = result.lower()
    matched = [c for c in choices if c.lower() in result_lower]

    if not matched:
        return None
    if len(matched) == 1:
        return matched[0]

    # Containment resolution runs entirely in lower-case (the text is already
    # lowered); map each lowered option back to its original label for the
    # return value (first occurrence wins, matching the stable-sort tie-break).
    original: dict[str, str] = {}
    for c in matched:
        original.setdefault(c.lower(), c)
    resolved = _resolve_containment(result_lower, list(original))
    if len(resolved) == 1:
        return original[resolved[0]]

    resolved.sort(key=lambda c: (-result_lower.rfind(c), -len(c)))
    return original[resolved[0]]


@register_task("mcq")
class McqTask(Task):
    """Image multiple-choice classification task.

    Sample convention: inputs={"image": path, "choices": [choice names],
    "group": group name}, references={"label": correct choice name}. Choice
    order affects the letter->label mapping and must match the dataset load
    order.
    """

    name = "mcq"
    objectives: ClassVar[tuple[Objective, ...]] = (
        Objective(key="accuracy", maximize=True, display_name="accuracy"),
    )
    default_prompt = "Classify the content of the target image."

    def __init__(
        self,
        dataset: str | None = None,
        *,
        data_dir: str | Path | None = None,
        images_dir: str | Path | None = None,
        videos_dir: str | Path | None = None,
        samples: list[Sample] | None = None,
        train_per_group: int = 50,
        train_sample_seed: int = 42,
        assistant_prefix: str | None = None,
        target_image_suffix: bool = False,
        instruction_pos: str | None = None,
        include_qa: bool = False,
        raw_template: bool = False,
        media_key: str = "image",
        loader: Callable[[Any, str, Any], list[Sample]] | None = None,
    ) -> None:
        if samples is None and dataset is None and data_dir is None:
            raise ValueError(
                "McqTask requires one of a dataset name, explicit samples or a data_dir"
            )
        if samples is None and loader is None and dataset is None:
            # config validation at construction (fail fast, survives -O):
            # data_dir/images_dir alone only override a registered dataset's
            # paths — without a dataset name or a loader no split can load
            raise ValueError(
                "McqTask requires a dataset name when neither samples nor loader is given "
                "(data_dir/images_dir alone only override a registered dataset's paths)"
            )
        if instruction_pos is not None and instruction_pos not in _INSTRUCTION_POSITIONS:
            raise ValueError(
                f"McqTask.instruction_pos must be one of {_INSTRUCTION_POSITIONS} or None, "
                f"got {instruction_pos!r}"
            )
        if raw_template and instruction_pos is not None:
            raise ValueError(
                "McqTask.raw_template and instruction_pos are mutually exclusive "
                "(full-template mode has no instruction position)"
            )
        self._dataset = dataset
        self._data_dir = data_dir
        self._images_dir = videos_dir if videos_dir is not None else images_dir
        self._samples = samples
        # sample input holding the media reference; _media_part builds the
        # ImagePart from it (subclasses override the hook for other media)
        self._media_key = media_key
        # alternative split loader: loader(data_dir, split, media_dir),
        # replacing the group-based image loader when given
        self._loader = loader
        # the train split uses a fixed stratified subsample as the evaluation
        # pool (default 50/group, seed 42); test/val use the full split.
        # 0 = no sampling.
        self._train_per_group = train_per_group
        self._train_sample_seed = train_sample_seed
        # CoT baseline continuation primer: when not None, append an assistant turn after the user turn
        self._assistant_prefix = assistant_prefix
        # EvoPrompt/CoT baseline classify assembly: append "\nTarget Image:"
        # after the choice line and rstrip the instruction
        self._target_image_suffix = target_image_suffix
        # OPRO scorer assembly: enabled when not None (target_image_suffix has no effect then)
        self._instruction_pos = instruction_pos
        self._include_qa = include_qa
        # SPO full-template assembly: the candidate prompt is the complete user-text template (with {image}/{choices})
        self._raw_template = raw_template
        self._split_cache: dict[str, list[Sample]] = {}

    @property
    def instruction_pos(self) -> str | None:
        """The OPRO scorer's instruction position; None = non-OPRO assembly
        (used by the optimizer for consistency checks)."""
        return self._instruction_pos

    @property
    def include_qa(self) -> bool:
        return self._include_qa

    def load_split(self, split: str) -> list[Sample]:
        if self._samples is not None:
            return list(self._samples)
        if split not in self._split_cache:
            if self._loader is not None:
                self._split_cache[split] = self._loader(self._data_dir, split, self._images_dir)
            else:
                assert self._dataset is not None, "dataset must be provided when samples is not"
                per_group = self._train_per_group if split == "train" else 0
                self._split_cache[split] = load_mcq_samples(
                    self._dataset,
                    split,
                    data_dir=self._data_dir,
                    images_dir=self._images_dir,
                    per_group=per_group,
                    sample_seed=self._train_sample_seed,
                )
        return list(self._split_cache[split])

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        choices = list(_require_input(sample, "choices"))
        media = self._media_part(sample)
        choices_str = ", ".join(f"'{c}'" for c in choices)
        if self._raw_template:
            parts = self._render_raw_template(prompt, sample, choices_str, media)
        else:
            text = self._render_classify_text(prompt, sample, choices, choices_str)
            parts = [media, TextPart(text)]
        messages = [user_message(*parts)]
        if self._assistant_prefix is not None:
            messages.append(assistant_message(self._assistant_prefix))
        return ChatRequest(
            messages=tuple(messages),
            request_id=f"{sample.id}:{variant.id}",
        )

    def _media_part(self, sample: Sample) -> Part:
        """Media hook: the base task renders the ``media_key`` input as an
        image; subclasses override for other media types (VideoMcqTask emits
        a VideoPart)."""
        return ImagePart(image=_require_input(sample, self._media_key))

    def demo_media_parts(self, sample: Sample) -> list[Part]:
        """A few-shot demo carries the same single media part as evaluation
        (subclasses override for other media, e.g. video demos carry none)."""
        return [self._media_part(sample)]

    def _render_raw_template(
        self, prompt: str, sample: Sample, choices_str: str, media_part: Part
    ) -> list[Part]:
        """SPO full template: {choices} filled per sample; the image marker
        replaced in place by the media part (media prepended when the marker
        is missing, empty segments dropped)."""
        if "{choices}" not in prompt:
            raise ValueError(
                f"raw_template mode requires the prompt to contain a {{choices}} placeholder, "
                f"got {prompt[:60]!r}"
            )
        text = prompt.replace("{choices}", choices_str)
        if "{image}" not in text:
            return [media_part, TextPart(text)]
        parts: list[Part] = []
        for idx, segment in enumerate(text.split("{image}")):
            if idx:
                parts.append(media_part)
            if segment:
                parts.append(TextPart(segment))
        return parts

    def _render_classify_text(
        self, prompt: str, sample: Sample, choices: list[str], choices_str: str
    ) -> str:
        """Compact user-text assembly: image markers stripped, then the OPRO
        scorer assembly, the target-image-suffix assembly, or the plain
        instruction + choices line."""
        instruction = prompt
        for marker in _IMAGE_MARKERS:
            instruction = instruction.replace(marker, "")
        if self._instruction_pos is not None:
            return gen_scorer_prompt(instruction, choices, self._instruction_pos, self._include_qa)
        if self._target_image_suffix:
            return f"{instruction.rstrip()}\nChoices: [{choices_str}]\nTarget Image:"
        return f"{instruction}\nChoices: [{choices_str}]"

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        value = match_prediction(response.text, list(_require_input(sample, "choices")))
        return Prediction(value=value, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": float(prediction.value == _require_reference(sample, "label"))}
