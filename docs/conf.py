"""Sphinx build configuration for the Eyas documentation (MyST-NB).

Build:  uv sync --group docs && uv run sphinx-build -b html docs docs/_build/html

Notebook execution is OFF: every page is plain MyST markdown, so the site
builds on any CPU machine with no model, no GPU and no network. If executable
notebooks are added later, commit their saved outputs and keep execution off
in CI; refresh outputs manually on a GPU machine.
"""

project = "Eyas"
copyright = "2026, the Eyas authors"
author = "the Eyas authors"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Links into examples/ are legitimate on GitHub's web renderer but lie outside
# the sphinx source tree; their on-disk existence is gated by
# tests/docs/test_docs_links.py instead (same split as the build itself).
suppress_warnings = ["myst.xref_missing"]

# -- MyST -------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3

# -- notebook execution ------------------------------------------------------
nb_execution_mode = "off"

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "Eyas"
# Search/generative-engine surface: the landing description is what crawlers and
# LLM answer engines quote; keywords stay honest (every listed method ships).
# myst_html_meta, NOT Sphinx's html_meta: the toolchain (sphinx 9 + the book
# theme) consumes no html_meta anywhere — meta reaches <head> only as docutils
# meta nodes, which myst emits for every page (verified: the html_meta key
# renders nothing). Keep every docs page in myst (.md) or this surface goes dark.
myst_html_meta = {
    # Google Search Console ownership (HTML-tag method: a Pages project page
    # has no domain-DNS handle). Public by design — it ships in every page.
    "google-site-verification": "N7UCH_IISEJrpFFZ_3sC2SEUZAjKGvwWEJtZ3y9tbFI",
    "description": (
        "Eyas is a performance-first automatic prompt optimization (APO) framework "
        "for multimodal models: eight optimizers on one strategy contract "
        "(APE, OPRO, EvoPrompt, ProTeGi, SPO, hill climbing), backend-agnostic "
        "(in-process vLLM or any OpenAI-compatible endpoint), per-run cost "
        "accounting and a public 1008-run benchmark scoreboard."
    ),
    "keywords": (
        "prompt optimization, automatic prompt optimization, APO, prompt "
        "engineering, multimodal, vision-language model, APE, OPRO, EvoPrompt, "
        "ProTeGi, SPO, hill climbing, vLLM, LLM"
    ),
}
html_theme_options = {
    "use_edit_page_button": False,
    "navigation_with_keys": True,
    "show_toc_level": 2,
}
html_static_path = ["_static"]
html_logo = "_static/eyas.svg"
html_favicon = "_static/eyas.svg"