"""genio extraction profiles: each paper extracts candidates its own way, so the
raw completion is parsed through an explicitly named profile (single bracket /
single tag body / one-per-line) rather than one-size line-split."""

from __future__ import annotations

import pytest

from eyas.engine.genio import (
    LINES,
    MAX_BODY_LEN,
    OUT_END,
    OUT_START,
    SINGLE_BRACKET,
    SINGLE_TAG,
    extract_candidates,
    parse_gradients,
    parse_instructions,
    render_reference,
)


def test_single_bracket_yields_at_most_one_candidate() -> None:
    """OPRO convention: one instruction per raw output, taken from the first
    delimited slice; a second bracket block is not a second candidate."""
    text = "[ '''Raise the bar on visual detail''' ] and then [ '''second slice never read''' ]"
    assert extract_candidates(text, SINGLE_BRACKET) == ["Raise the bar on visual detail"]


def test_single_bracket_drops_echoed_marker() -> None:
    """Official OPRO filter: a completion that echoes the meta-prompt marker back
    inside the slice is not a candidate."""
    text = "[\nINS[\n'''Give a sharper answer'''\n]\n]"
    assert extract_candidates(text, SINGLE_BRACKET) == []


def test_single_bracket_drops_empty_and_unparsed() -> None:
    assert extract_candidates("[]", SINGLE_BRACKET) == []
    assert extract_candidates("no delimiters at all", SINGLE_BRACKET) == []
    assert extract_candidates("] closes before [ opens", SINGLE_BRACKET) == []


def test_single_tag_preserves_multiline_body() -> None:
    """SPO/EvoPrompt convention: the whole tagged body is the candidate — newlines
    stay, no line-split, no per-line cleaning."""
    body = "Look at the parcel label.\nReport the tracking number only."
    assert extract_candidates(f"{OUT_START}\n{body}\n{OUT_END}", SINGLE_TAG) == [body]


def test_single_tag_caps_body_at_max_len() -> None:
    body = "x" * (MAX_BODY_LEN + 500)
    out = extract_candidates(f"{OUT_START}{body}{OUT_END}", SINGLE_TAG)
    assert len(out) == 1
    assert MAX_BODY_LEN == 2000
    assert len(out[0]) == MAX_BODY_LEN


def test_single_tag_without_tags_falls_back_to_whole_text() -> None:
    assert extract_candidates("a plain untagged answer", SINGLE_TAG) == ["a plain untagged answer"]


def test_single_tag_drops_empty_body() -> None:
    assert extract_candidates(f"{OUT_START}\n   \n{OUT_END}", SINGLE_TAG) == []


def test_lines_delegates_to_parse_instructions() -> None:
    """The line-split profile IS the existing behavior (ape/protegi); delegation
    keeps the two paths from drifting apart."""
    text = "1. Read the label carefully\n2. Report the city name"
    assert extract_candidates(text, LINES) == parse_instructions(text)
    assert extract_candidates(text, LINES) == ["Read the label carefully", "Report the city name"]


def test_unknown_profile_fails_fast() -> None:
    with pytest.raises(ValueError, match="bogus"):
        extract_candidates("anything", "bogus")  # type: ignore[arg-type]


# --- parse_gradients (ProTeGi ∇ critique parsing, caller-level completions) ---


def test_gradients_tagged_reasons_extracted_in_order() -> None:
    """Official convention: the critique wraps each reason in <START>/<END>; the
    preamble before the first tag is not a reason."""
    text = (
        "My prompt fails because it is vague.\n"
        "<START>The prompt never mentions the shipping label.<END>\n"
        "<START>It omits the city name from the output spec.<END>"
    )
    assert parse_gradients(text) == [
        "The prompt never mentions the shipping label.",
        "It omits the city name from the output spec.",
    ]


def test_gradients_tagged_unclosed_final_segment_runs_to_end() -> None:
    """A truncated completion (last <END> never emitted) still contributes its
    final reason — the segment simply runs to the end of the text."""
    text = "<START>Check the parcel label twice.<END><START>Report the city only."
    assert parse_gradients(text) == ["Check the parcel label twice.", "Report the city only."]


def test_gradients_tagged_drops_empty_and_duplicate_reasons() -> None:
    text = (
        "<START>Think step by step.<END><START><END>"
        "<START>  Think step by step.<END><START>Think step by step.<END>"
    )
    assert parse_gradients(text) == ["Think step by step."]


def test_gradients_untagged_numbered_completion_falls_back() -> None:
    """A model ignoring the wrapping directive still contributes its reasons:
    the untagged completion splits on the 1. / 2) numbering."""
    text = "1. Mention the shipping label. 2) Add the city name to the output spec."
    assert parse_gradients(text) == [
        "Mention the shipping label.",
        "Add the city name to the output spec.",
    ]


def test_gradients_untagged_one_reason_per_line() -> None:
    text = "Look closer at the label.\nState the city, not the district."
    assert parse_gradients(text) == [
        "Look closer at the label.",
        "State the city, not the district.",
    ]


def test_gradients_untagged_blank_and_duplicate_lines_dropped() -> None:
    """Whitespace-only numbered parts and repeated/empty lines yield nothing
    beyond the distinct reasons."""
    text = "1. Be specific.  \n2.\nBe specific."
    assert parse_gradients(text) == ["Be specific."]


def test_gradients_empty_completion_contributes_nothing() -> None:
    assert parse_gradients("") == []


# --- render_reference (generative-propose reference rendering) ---


def test_render_reference_joins_top_level_list() -> None:
    """A list-valued reference (KIE-style multi-value answers) renders as
    natural text, never as Python ``['a', 'b']`` repr in a demo line."""
    assert render_reference(["a", "b"]) == "a, b"


def test_render_reference_joins_tuple() -> None:
    assert render_reference(("a", "b")) == "a, b"


def test_render_reference_passes_scalar_through() -> None:
    assert render_reference("cat") == "cat"
    assert render_reference(7) == "7"


def test_render_reference_top_level_only() -> None:
    """Top-level join only: a nested list is str()'d as a unit (the shipped
    tasks carry at most one nesting level of values)."""
    assert render_reference(["a", ["b", "c"]]) == "a, ['b', 'c']"
