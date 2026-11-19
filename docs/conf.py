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
# tests/test_docs_links.py instead (same split as the build itself).
suppress_warnings = ["myst.xref_missing"]

# -- MyST -------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3

# -- notebook execution ------------------------------------------------------
nb_execution_mode = "off"

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "Eyas"
html_theme_options = {
    "use_edit_page_button": False,
    "navigation_with_keys": True,
    "show_toc_level": 2,
}
html_static_path = ["_static"]
html_logo = "_static/eyas.svg"
html_favicon = "_static/eyas.svg"