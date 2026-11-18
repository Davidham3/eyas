"""Slow tier: template-owned golden byte anchors for the shipped Qwen3.5 template.

Re-derives every golden case through ``HfTemplateProfile`` with the real
tokenizer and diffs byte-for-byte against ``tests/golden/qwen35_template/goldens.json``
(the anchors were generated from the shipped chat template itself — regenerate
them with ``scripts/gen_renderer_golden.py``; this test never regenerates on
its own). The per-case digest equality is the gate. The stamp is provenance:
a drift in the template file's sha256 still FAILS the test with a clear
message (never a skip), because a template change silently redefines the
pinned bytes; a drift in ``transformers.__version__`` is a visible WARNING,
not a failure — the renders were measured byte-identical across the stamp's
5.19.0 and 5.15.1 (the vllm extra's locked resolution), 8/8 cases per
version, so the version stamp must not break the slow tier in the engine
environments it is meant to certify.

The case matrix (system/user/assistant history, thinking tri-state, single &
multi-image, no-system, whitespace-padded system) pins the shipped template's
divergences as truth: NO newline after the vision placeholder, and trimmed
system whitespace.

Run from the repo root with a venv carrying transformers (the fast tier
deselects this module via the ``slow`` marker):

    <venv>/bin/python -m pytest tests/test_renderer_goldens.py -m slow
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from eyas.backends.base import ImagePart, Message, TextPart
from eyas.backends.profiles import HfTemplateProfile

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "qwen35_template"
MODEL_PATH = os.environ.get("EYAS_QWEN35_MODEL_PATH", "")
THINKING_KWARG = "enable_thinking"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not (Path(MODEL_PATH) / "chat_template.jinja").is_file(),
    reason="set EYAS_QWEN35_MODEL_PATH to a local Qwen3.5 model dir "
    "(needs its shipped chat_template.jinja) to run the golden tier",
)


@dataclass(frozen=True)
class _Case:
    name: str
    messages: tuple[Message, ...]
    thinking: bool | None


def _load_golden_doc() -> dict[str, object]:
    return json.loads((GOLDEN_DIR / "goldens.json").read_text())


def _template_sha256() -> str:
    return hashlib.sha256((Path(MODEL_PATH) / "chat_template.jinja").read_bytes()).hexdigest()


def _render(case: _Case, tokenizer: object) -> str:
    # the real tokenizer is injected so the module fixture is not reloaded per
    # case; the rendering path itself is the profile's shipped-template one
    profile = HfTemplateProfile(model_path=MODEL_PATH, thinking_kwarg=THINKING_KWARG)
    profile._tokenizer = tokenizer
    return profile.render(case.messages, case.thinking)


def _expected_for(doc: dict[str, object], name: str) -> str:
    for case in doc["cases"]:
        if case["name"] == name:
            expected = case["expected"]
            assert isinstance(expected, str)
            return expected
    raise AssertionError(f"golden case {name!r} missing from goldens.json")


def _digest(text: str) -> str:
    # tag-safe diagnostics: raw rendered text carries the model's special-token
    # tags and can mangle display pipelines; the digest pair localizes drift
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _message(role: str, hf_content: list[dict[str, str]]) -> Message:
    """HF chat-format dict -> the repo's Message (the profile's own conversion)."""
    parts: list[TextPart | ImagePart] = []
    for part in hf_content:
        if part["type"] == "image":
            parts.append(ImagePart(image="x.png"))
        else:
            parts.append(TextPart(part["text"]))
    return Message(role=role, content=tuple(parts))


def _case_matrix() -> list[_Case]:
    """The HF chat-format case matrix (mirrors goldens.json; tag-safe)."""
    img = (
        {"type": "image"},
        {"type": "text", "text": "What color is the square?"},
    )
    sys_text = ({"type": "text", "text": "You are a helpful assistant."},)
    txt = ({"type": "text", "text": "Classify the animal."},)
    padded = ({"type": "text", "text": "\n\n  padded system  \n\n"},)
    txt2 = ({"type": "text", "text": "It is a cat."},)
    txt_q = ({"type": "text", "text": "And now?"},)

    return [
        _Case(
            "sys+user_text thinking=off",
            (_message("system", list(sys_text)), _message("user", list(txt))),
            False,
        ),
        _Case(
            "sys+user_text thinking=on",
            (_message("system", list(sys_text)), _message("user", list(txt))),
            True,
        ),
        _Case(
            "sys+user_text thinking=None",
            (_message("system", list(sys_text)), _message("user", list(txt))),
            None,
        ),
        _Case(
            "sys+user_img+text thinking=off",
            (_message("system", list(sys_text)), _message("user", list(img))),
            False,
        ),
        _Case("no_system user_img+text thinking=off", (_message("user", list(img)),), False),
        _Case(
            "history w assistant thinking=off",
            (
                _message("system", list(sys_text)),
                _message("user", list(txt)),
                _message("assistant", list(txt2)),
                _message("user", list(txt_q)),
            ),
            False,
        ),
        _Case(
            "padded system thinking=off",
            (_message("system", list(padded)), _message("user", list(txt))),
            False,
        ),
        _Case(
            "multi-image interleaved thinking=off",
            (
                _message("system", list(sys_text)),
                _message(
                    "user",
                    list(img) + [{"type": "image"}] + list(txt),
                ),
            ),
            False,
        ),
    ]


CASES = _case_matrix()


@pytest.fixture(scope="module")
def golden_doc() -> dict[str, object]:
    return _load_golden_doc()


@pytest.fixture(scope="module")
def real_tokenizer() -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.mark.slow
def test_stamp_matches(golden_doc: dict[str, object]) -> None:
    """Template-bytes drift is a FAILURE, never a skip; a transformers version
    drift is a provenance WARNING — the renders are byte-stable across the
    versions the extra locks (see the module docstring), and the per-case
    digest comparison is the real gate."""
    from transformers import __version__

    stamp = golden_doc["stamp"]
    assert isinstance(stamp, dict)
    if __version__ != stamp["transformers"]:
        warnings.warn(
            f"transformers version drift: installed {__version__!r} != golden stamp "
            f"{stamp['transformers']!r}; the stamp is provenance, not a gate — the digest "
            "comparison is the gate and renders byte-stably across versions",
            stacklevel=1,
        )
    assert _template_sha256() == stamp["template_sha256"], (
        f"chat_template.jinja sha256 drift: installed {_template_sha256()} != golden stamp "
        f"{stamp['template_sha256']}; regenerate the goldens or restore the template file"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_golden_bytes_match_shipped_template(
    case: _Case,
    golden_doc: dict[str, object],
    real_tokenizer: object,
) -> None:
    """Every case re-renders byte-for-byte through HfTemplateProfile's shipped-template path."""
    expected = _expected_for(golden_doc, case.name)
    actual = _render(case, real_tokenizer)
    assert actual == expected, (
        f"byte drift in {case.name!r}: rendered sha256[:16]={_digest(actual)} len={len(actual)} "
        f"!= golden sha256[:16]={_digest(expected)} len={len(expected)}; regenerate the goldens "
        "with scripts/gen_renderer_golden.py only after verifying the template change is intended"
    )


@pytest.mark.slow
def test_vision_placeholder_pinned_truth(golden_doc: dict[str, object]) -> None:
    """The shipped template emits NO newline after the vision placeholder and
    trims system whitespace; the anchors pin that byte shape (do not "fix" it)."""
    expected = _expected_for(golden_doc, "sys+user_img+text thinking=off")
    end_marker = "<|" + "vision_end" + "|>"
    assert end_marker in expected, "vision end marker missing from the image golden"
    after_end = expected.index(end_marker) + len(end_marker)
    assert expected[after_end] == "W", (
        "pinned truth violated: the shipped template emits no newline after the vision "
        f"placeholder; first char after the end marker is {expected[after_end]!r}"
    )
    padded = _expected_for(golden_doc, "padded system thinking=off")
    im_end = "<|" + "im_end" + "|>"
    assert "padded system" + im_end in padded, "pinned truth violated: system whitespace is trimmed"
