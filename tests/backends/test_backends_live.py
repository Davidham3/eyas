"""slow tier: local engine startup smoke (nightly, requires GPU).

Excluded by default via `-m 'not slow'`; see the README's GPU-smoke
line (`uv run pytest -m slow`) for how to run. Skip conditions: EYAS_SLOW_MODEL or
EYAS_SLOW_IMAGE unset, or the model path does not exist.

Consistency criteria (character-for-character equality across runs is
fragile): structural assertions at temperature=0 — the plain-text arithmetic
answer matches exactly; the image classification hits the expected class
(character-for-character equality not required); the CoT continuation is
non-empty. OpenAIBackend live consistency depends on an external server: when
EYAS_SLOW_OPENAI_BASE_URL is provided the OpenAI smoke runs, otherwise it is
skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    SamplingConfig,
    TextPart,
    assistant_message,
    user_message,
)

_MODEL = os.environ.get("EYAS_SLOW_MODEL", "")
_IMAGE = os.environ.get("EYAS_SLOW_IMAGE", "")
_INSTRUCTION = "Classify the bird species."
_CHOICES = "['Blue_Jay', 'Sooty_Albatross', 'Black_footed_Albatross']"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _MODEL or not _IMAGE or not Path(_MODEL).is_dir(),
        reason="EYAS_SLOW_MODEL/EYAS_SLOW_IMAGE unset or model directory does not exist",
    ),
]


def _sampling() -> SamplingConfig:
    return SamplingConfig(temperature=0.0, max_new_tokens=32)


def _text_request() -> ChatRequest:
    return ChatRequest(
        messages=(user_message("1+1=? Answer with the number only."),), request_id="t1"
    )


def _image_request() -> ChatRequest:
    return ChatRequest(
        messages=(
            user_message(ImagePart(image=_IMAGE), TextPart(f"{_INSTRUCTION}\nChoices: {_CHOICES}")),
        ),
        request_id="t2",
    )


def _cot_request() -> ChatRequest:
    return ChatRequest(
        messages=(
            user_message(ImagePart(image=_IMAGE), TextPart(f"{_INSTRUCTION}\nChoices: {_CHOICES}")),
            assistant_message("Let's think step by step."),
        ),
        request_id="t3",
    )


async def _probe(backend) -> dict[str, ChatResponse]:
    sampling = SamplingConfig(temperature=0.0, max_new_tokens=32)
    text = await backend.agenerate([_text_request()], sampling)
    sampling_cot = SamplingConfig(temperature=0.0, max_new_tokens=256)
    rest = await backend.agenerate([_image_request(), _cot_request()], sampling_cot)
    return {r.request_id: r for r in [text[0], *rest]}


def _classify_hit(text: str) -> bool:
    normalized = text.lower().replace("_", " ")
    return "blue jay" in normalized


async def test_vllm_backend_inprocess_smoke():
    from eyas.backends.vllm import VllmBackend

    backend = VllmBackend(
        _MODEL, mem_fraction=0.5, context_length=8192, max_running_requests=8, max_inflight=8
    )
    out = await _probe(backend)
    assert out["t1"].text.strip() == "2"
    assert _classify_hit(out["t2"].text)
    assert len(out["t3"].text.strip()) > 0


def _assert_consistent(*outputs: dict[str, ChatResponse]) -> None:
    """Consistency criteria shared by the slow smokes: plain text exact, expected class hit, CoT non-empty."""
    for out in outputs:
        assert out["t1"].text.strip() == "2"
        assert _classify_hit(out["t2"].text)
        assert len(out["t3"].text.strip()) > 0


@pytest.mark.skipif(
    not os.environ.get("EYAS_SLOW_OPENAI_BASE_URL"), reason="EYAS_SLOW_OPENAI_BASE_URL not set"
)
async def test_openai_backend_live_consistency():
    """OpenAIBackend vs a real server (vllm serve) runs the same structural assertions."""
    from eyas.backends.openai_api import OpenAIBackend

    backend = OpenAIBackend(
        base_url=os.environ["EYAS_SLOW_OPENAI_BASE_URL"], model=os.environ["EYAS_SLOW_MODEL_NAME"]
    )
    out = await _probe(backend)
    _assert_consistent(out)
