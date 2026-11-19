"""CI gate for the plugin-backend example: registration works for BACKENDS.

Proves the two things a third-party backend author needs, with zero GPU:

1. the config ``plugins`` key imports an unpackaged module and its
   ``@register_backend`` name resolves (mirroring exactly what built-ins obey);
2. a config that names that backend runs end to end through the real entry
   (``run_experiment``) and leaves ``result.json`` on disk.

Run from the repo root (the repo gate collects this file automatically —
``testpaths`` in pyproject.toml includes ``examples``):

    uv run pytest examples/plugin_backend -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Contract pin, not a skip: this import only exists in distributions built from
# a recent checkout. A stale installed ``eyas`` must fail LOUDLY at collection
# with the remedy named — a silent skip here would let CI pass having tested
# nothing (the one failure mode that looks like success).
try:
    from eyas.tasks.demo_shapes import generate_demo_samples  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "examples/plugin_backend: the installed eyas distribution is stale; "
        "run `uv sync` from the repo root"
    ) from exc

from eyas.backends.base import (
    ChatRequest,
    ImagePart,
    SamplingConfig,
    TextPart,
    assistant_message,
    system_message,
    user_message,
)
from eyas.engine.runtime import run_experiment
from eyas.registry import get
from eyas.runtime.config import load_plugins

# The package import below is absolute, so it needs the repo root on sys.path.
# pytest puts it there when you run `pytest examples/plugin_backend` from the
# root; when you enter this directory instead (the no-install command above),
# nothing else would resolve it — an installed `eyas` distribution does not make
# `examples` importable. Resolving it from this file's location works from
# anywhere and keeps the test runnable in both shapes.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.plugin_backend import echo_backend  # noqa: E402

_QUESTION = "\nWhat color is this square? Choices: [blue, green, red, yellow]\n"


def _request(request_id: str, text: str, *, turns: tuple[str, ...] | None = None) -> ChatRequest:
    """One canonical request: a system turn, optional filler turns, and a final
    user turn carrying an image part plus the text the echo should return."""
    messages = [system_message("You classify shapes.")]
    for turn in turns or ():
        messages.append(user_message(turn))
        messages.append(assistant_message("ok"))
    messages.append(
        user_message(ImagePart(image="/nonexistent/demo.png"), TextPart(text)),
    )
    return ChatRequest(messages=tuple(messages), request_id=request_id)


def test_plugin_route_registers_the_backend():
    """The config ``plugins`` key is the registration entry point for code that
    lives outside the framework package; after loading it, the registry resolves
    the decorated name to the class."""
    load_plugins({"plugins": ["examples.plugin_backend.echo_backend"]})
    assert get("backend", "echo_demo") is echo_backend.EchoBackend


def test_echo_is_deterministic_in_input_order():
    """Contract: one response per request, in input order; the text is the last
    user turn's text content truncated at 80 chars; ``request_id`` is carried."""
    backend = echo_backend.EchoBackend()
    long_text = "color " * 30  # 180 chars, truncates to 80
    requests = [
        _request("r1", _QUESTION),
        _request("r2", "What shape is this?", turns=("first turn", "second turn")),
        _request("r3", long_text),
    ]

    out = backend.generate(requests, SamplingConfig())  # the provided sync entry

    assert [r.request_id for r in out] == ["r1", "r2", "r3"]
    assert out[0].text == _QUESTION  # the image part contributes no text
    assert out[1].text == "What shape is this?"  # the LAST user turn wins
    assert out[2].text == long_text[:80] and len(out[2].text) == 80


def test_close_marks_the_backend_closed():
    """The optional close() hook the run calls after the loop (a sweep must
    not pay the previous engine's GC bill)."""
    backend = echo_backend.EchoBackend()
    assert backend.closed is False
    backend.close()
    assert backend.closed is True


def test_demo_config_runs_end_to_end_through_the_real_entry(tmp_path):
    """The documented run command, driven as data: demo_config.yaml's binding
    shape (plugins key + demo_shapes + raw + echo_demo) as a dict through
    ``run_experiment`` must complete and leave result.json on disk.

    image_dir moves the synthetic PNGs onto tmp (same override the framework's
    own demo run-entry test uses); the echo's metrics carry no algorithmic
    signal — they only pin that the whole assembly ran.
    """
    config = {
        "plugins": ["examples.plugin_backend.echo_backend"],
        "run": {"name": "plugin_echo_demo", "seed": 42},
        "task": {"name": "demo_shapes", "params": {"image_dir": str(tmp_path / "imgs")}},
        "protocol": {"name": "identity"},
        "backend": {"name": "echo_demo", "params": {}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {"name": "raw", "params": {}},
    }
    load_plugins(config)
    assert get("backend", "echo_demo") is echo_backend.EchoBackend

    run_dir = tmp_path / "run"
    result = run_experiment(config, output_dir=run_dir)

    assert (run_dir / "result.json").is_file()
    assert result.best is not None
    # The echo returns the question text; demo_shapes's parser picks the LAST
    # palette color appearing in it ("yellow" from the choices list), so the 8
    # train samples score 2/8 — deterministic, and proof the run evaluated.
    assert result.best.scores["accuracy"] == pytest.approx(0.25)
    assert result.total_cost.requests > 0
