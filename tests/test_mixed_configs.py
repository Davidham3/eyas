"""The mixed-modality board's config contract (spec-as-source): assembled via
the shared accessor, same assertions the per-cell YAMLs carried."""

import re

import pytest

from tests.support.spec_cells import cell, cell_id, shipped_cells

_MIXED = [(cell_id(ds, m, v), cell(ds, m, v)) for ds, m, v in shipped_cells() if ds == "mixed"]


def test_all_eight_methods_shipped():
    methods = {
        cell_id(ds, m, v).removeprefix("mixed_") for ds, m, v in shipped_cells() if ds == "mixed"
    }
    assert methods == {
        "raw",
        "cot",
        "evoprompt",
        "hill_climb",
        "opro",
        "ape",
        "protegi",
        "spo",
    }


@pytest.mark.parametrize("cell_id,cfg", _MIXED)
def test_task_section_targets_the_mixed_dataset(cell_id, cfg):
    assert cfg["task"]["name"] == "mixed_mcq"
    params = cfg["task"]["params"]
    assert params["data_dir"].endswith("mixed/data")
    assert "images_dir" not in params and "videos_dir" not in params


def test_spo_seeds_name_every_modality_marker():
    cfg = cell("mixed", "spo")
    assert cfg["task"]["params"]["raw_template"] is True
    for seed in cfg["strategy"]["seeds"]:
        for marker in ("{text}", "{image}", "{video}", "{question}", "{choices}"):
            assert marker in seed, f"spo seed missing {marker}: {seed!r}"


def test_spo_ships_a_single_seed():
    # SPO is single-incumbent: init warns and
    # keeps only seeds[0], so a second seed would be a config line that silently
    # never runs
    cfg = cell("mixed", "spo")
    assert len(cfg["strategy"]["seeds"]) == 1


_MODALITY_RE = re.compile(r"\b(?:video|clip|image|picture|frame|photo)s?\b", re.IGNORECASE)


def _modality_hits(text: str) -> list[str]:
    """Modality words in prompt prose, after removing the placeholders that
    legitimately name one. Word-bounded with an optional plural: an authoring
    verb like "eclipsed" must not trip the check, "images" must."""
    plain = text
    for marker in ("{text}", "{image}", "{video}", "<image>", "<video>"):
        plain = plain.replace(marker, " ")
    return [match.group(0) for match in _MODALITY_RE.finditer(plain)]


def test_modality_matcher_ignores_embedded_words_and_catches_plurals():
    assert _modality_hits("The framework eclipsed the prompt.") == []
    assert _modality_hits("Watch the clip.") == ["clip"]
    assert _modality_hits("Look at the images.") == ["images"]


@pytest.mark.parametrize("cell_id,cfg", _MIXED)
def test_prompt_prose_names_no_modality(cell_id, cfg):
    """The same config is evaluated on all seven combos, so no prompt prose may
    name a medium a sample can lack -- a text-only sample instructed to "watch
    the clip" turns a one-prompt benchmark into an ambiguous measurement."""
    strategy = cfg.get("strategy") or {}
    params = cfg["task"].get("params", {})
    prose = [params.get("prompt") or "", params.get("assistant_prefix") or ""]
    prose.append((strategy.get("params") or {}).get("base_prompt") or "")
    prose.extend(strategy.get("seeds") or [])
    for text in prose:
        hits = _modality_hits(text)
        assert not hits, f"{cell_id}: prompt names {hits}: {text!r}"
