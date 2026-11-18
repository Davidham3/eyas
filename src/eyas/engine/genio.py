"""I/O conventions and candidate cleaning for generative propose (shared by
ape/opro/evoprompt/protegi/spo).

``GEN_MARKER`` appears in generation requests so mock/debug can tell generation
calls from evaluation calls; candidate parsing is unified through
``extract_candidates`` (a profile dispatch over ``clean_instruction`` /
``parse_instructions``), so every generative method shares one parser.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

GEN_MARKER = "###GEN###"
OUT_START = "###OUT###"
OUT_END = "###END###"

#: Extraction profiles — one per candidate-yield convention (see ``extract_candidates``).
ExtractProfile = Literal["lines", "single_bracket", "single_tag"]
LINES: ExtractProfile = "lines"
SINGLE_BRACKET: ExtractProfile = "single_bracket"
SINGLE_TAG: ExtractProfile = "single_tag"

#: Hard cap on a ``single_tag`` body (the per-line cap of the ``lines`` profile is ``_MAX_LEN``).
MAX_BODY_LEN = 2000

#: A completion that echoes this meta-prompt marker text back inside the bracket slice
#: is prompt echo, not a candidate (the official OPRO filter).
_ECHO_MARKER = "INS"

__all__ = [
    "FEEDBACK_END",
    "FEEDBACK_START",
    "GEN_MARKER",
    "ExtractProfile",
    "LINES",
    "MAX_BODY_LEN",
    "OUT_END",
    "OUT_START",
    "SINGLE_BRACKET",
    "SINGLE_TAG",
    "clean_instruction",
    "extract_candidates",
    "parse_gradients",
    "parse_instructions",
    "render_reference",
    "render_sample_reference",
]

_MAX_LEN = 500
_PREFIX_RE = re.compile(r"^(instruction|input|output|answer|response)\s*[:：]\s*", re.I)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s*")


def _strip_noise(text: str) -> str:
    text = text.strip().strip("`").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    text = _PREFIX_RE.sub("", text).strip()
    return text.strip().strip('"').strip("'").strip()


def clean_instruction(text: str) -> str:
    """Clean a single instruction: strip think/fences/prefixes/quotes → take
    the first segment → truncate. Empty string = parse failure."""
    if OUT_START in text:
        text = text.split(OUT_START, 1)[1].split(OUT_END, 1)[0]
    text = _strip_noise(text)
    for line in text.splitlines():
        candidate = _strip_noise(_NUMBERED_RE.sub("", line))
        if candidate:
            return candidate[:_MAX_LEN]
    return ""


def parse_instructions(text: str) -> list[str]:
    """One meta-prompt yields multiple candidates (OPRO convention): split by
    line, strip numbering, clean, drop empty and overlong entries."""
    if OUT_START in text:
        text = text.split(OUT_START, 1)[1].split(OUT_END, 1)[0]
    out: list[str] = []
    for line in text.splitlines():
        item = clean_instruction(line)
        if item and len(item) <= _MAX_LEN and item not in out:
            out.append(item)
    return out


def _first_bracket_slice(text: str) -> str:
    """Content of the first ``[...]`` slice, regex-free; empty when the pair is
    missing or inverted (an unparsable completion)."""
    start = text.find("[")
    end = text.find("]")
    if start < 0 or end < 0 or end <= start:
        return ""
    return text[start + 1 : end]


#: Opening/closing tokens of one critique reason (official ProTeGi ``_get_gradients``:
#: "Wrap each reason with <START> and <END>").
FEEDBACK_START = "<START>"
FEEDBACK_END = "<END>"

#: Fallback split for an untagged critique completion ("1. … 2. …" / one per line).
_REASON_SPLIT_RE = re.compile(r"\s*\d+\s*[.)]\s+")


def _tagged_gradients(text: str) -> list[str]:
    """Reasons from a tagged critique completion: the text between each
    ``<START>`` and the next ``<END>`` (an unclosed final segment runs to the
    end). Text before the first ``<START>`` is preamble and discarded."""
    out: list[str] = []
    for segment in text.split(FEEDBACK_START)[1:]:
        reason = clean_instruction(segment.split(FEEDBACK_END, 1)[0])
        if reason and reason not in out:
            out.append(reason)
    return out


def _untagged_gradients(text: str) -> list[str]:
    """Reasons from an untagged completion: number-split, then one reason per
    line. A whitespace-only numbered part yields no line and contributes
    nothing (``clean_instruction('')`` was always empty)."""
    out: list[str] = []
    for part in _REASON_SPLIT_RE.split(text):
        for line in part.strip().splitlines():
            reason = clean_instruction(line)
            if reason and reason not in out:
                out.append(reason)
    return out


def parse_gradients(text: str) -> list[str]:
    """Split a critique (∇) completion into individual gradient texts.

    The official feedback convention wraps every reason in ``<START>``/``<END>``;
    an untagged completion falls back to numbered/line splitting (a model that
    ignores the wrapping directive still contributes its reasons, cleaned via
    ``clean_instruction``). Empty or duplicate reasons are dropped."""
    if FEEDBACK_START in text:
        return _tagged_gradients(text)
    return _untagged_gradients(text)


def extract_candidates(text: str, profile: ExtractProfile) -> list[str]:
    """Parse the candidates out of one raw generation completion.

    The papers do not agree on how a completion carries candidates, so a strategy
    opts into the profile its method ships with instead of inheriting a
    one-size line-split:

    * ``lines`` — one candidate per line (``parse_instructions``, unchanged);
    * ``single_bracket`` — at most one candidate: the first ``[...]`` slice,
      cleaned and dropped when it echoes the generation marker or comes out empty;
    * ``single_tag`` — at most one candidate: the whole ``OUT_START``/``OUT_END``
      body verbatim (newlines preserved, no per-line cleaning), capped at
      ``MAX_BODY_LEN``. Untagged completions fall back to the full text.

    Empty output always means "this completion yielded nothing" — the caller keeps
    its previous candidates.
    """
    if profile == LINES:
        return parse_instructions(text)
    if profile == SINGLE_BRACKET:
        candidate = _strip_noise(_first_bracket_slice(text))
        if not candidate or _ECHO_MARKER in candidate:
            return []
        return [candidate]
    if profile == SINGLE_TAG:
        body = text.split(OUT_START, 1)[1] if OUT_START in text else text
        body = body.split(OUT_END, 1)[0].strip()
        return [body[:MAX_BODY_LEN]] if body else []
    raise ValueError(f"unknown extraction profile: {profile!r}")


def render_reference(value: object) -> str:
    """Render one sample reference as a natural-language demo answer.

    A list/tuple reference (the KIE-style multi-value answer) is joined with
    ``", "`` — dropping it through ``str`` verbatim would leak Python repr
    syntax into a natural-language demo line (``Output: ['a', 'b']`` instead
    of ``Output: a, b``). Scalars pass through ``str`` unchanged; the join is
    top-level only (a nested list is str'd as a unit, per the shipped tasks'
    single-level value shape)."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def render_sample_reference(references: Mapping[str, object]) -> str:
    """Render a sample's ground truth by the shared reference-key convention.

    ``label`` wins whenever the key exists (every pre-existing task), because
    position must not decide: KIE stores its ``entities`` list before
    ``label``. Tasks keyed differently (``math_qa``'s ``answer``) render their
    first reference value; no references render as the empty string."""
    if "label" in references:
        return render_reference(references["label"])
    return render_reference(next(iter(references.values()), ""))
