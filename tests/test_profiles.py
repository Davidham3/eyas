"""Contract tests for profile resolution and HF-template rendering (fast tier).

The tokenizer is lazy, so resolution and message-mapping contracts are all
assertable without transformers installed or a model on disk.
"""

from __future__ import annotations

import pytest

from eyas.backends.base import ImagePart, TextPart, VideoPart, user_message
from eyas.backends.profiles import HfTemplateProfile, resolve_profile


class _FakeTokenizer:
    """Stand-in for HF tokenizers: records the call, echoes a marker."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return "RENDERED"


def _hf_profile(**kwargs: object) -> tuple[HfTemplateProfile, _FakeTokenizer]:
    profile = HfTemplateProfile(model_path="/models/x", **kwargs)  # type: ignore[arg-type]
    fake = _FakeTokenizer()
    profile._tokenizer = fake
    return profile, fake


def test_resolve_default_is_hf_template() -> None:
    profile = resolve_profile(None, "/models/x")
    assert isinstance(profile, HfTemplateProfile)
    assert profile.model_path == "/models/x"
    assert profile._tokenizer is None  # default must not touch transformers


def test_resolve_retired_family_name_fails_fast() -> None:
    # An unregistered profile type must fail fast, never resolve silently.
    with pytest.raises(ValueError, match="unknown profile type"):
        resolve_profile("retired_profile_type", "/models/x")


def test_resolve_hf_template_injects_model_path_without_loading() -> None:
    profile = resolve_profile({"type": "hf_template", "trust_remote_code": True}, "/models/x")
    assert isinstance(profile, HfTemplateProfile)
    assert profile.model_path == "/models/x"
    assert profile.trust_remote_code is True
    assert profile._tokenizer is None  # resolve must not touch transformers


def test_resolve_instance_passthrough() -> None:
    instance = HfTemplateProfile(model_path="/models/x")
    assert resolve_profile(instance, "/models/x") is instance


def test_resolve_unknown_type_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown profile type"):
        resolve_profile({"type": "nope"}, "/models/x")


def test_hf_render_maps_structured_content() -> None:
    profile, fake = _hf_profile()
    out = profile.render(
        (user_message(ImagePart(image="a.jpg"), TextPart("what is this?")),),
        thinking=None,
    )
    assert out == "RENDERED"
    call = fake.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert call["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "what is this?"},
            ],
        }
    ]
    # the thinking tri-state is ignored unless the deployment names the kwarg
    assert "enable_thinking" not in call
    assert "thinking" not in call


@pytest.mark.parametrize(("thinking", "expected"), [(True, True), (False, False), (None, False)])
def test_declared_thinking_kwarg_always_forwarded(thinking: bool | None, expected: bool) -> None:
    # in-process engines have no server default to consult: a declared kwarg
    # is always forwarded, and None keeps the prefill rendered off
    profile, fake = _hf_profile(thinking_kwarg="enable_thinking")
    profile.render((user_message(TextPart("q")),), thinking)
    assert fake.calls[0]["enable_thinking"] is expected


def test_undeclared_thinking_kwarg_never_forwarded() -> None:
    profile, fake = _hf_profile()
    profile.render((user_message(TextPart("q")),), True)
    assert "enable_thinking" not in fake.calls[0]


def test_hf_extract_images_generic() -> None:
    profile, _ = _hf_profile()
    images = profile.extract_images(
        (user_message(ImagePart(image="a.jpg"), TextPart("q"), ImagePart(image="b.png")),)
    )
    assert images == ["a.jpg", "b.png"]


def test_hf_render_maps_video_content() -> None:
    profile, fake = _hf_profile()
    profile.render((user_message(VideoPart(video="clip.mp4"), TextPart("what happens?")),), None)
    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what happens?"}]}
    ]


def test_hf_extract_videos_and_images_are_separate() -> None:
    profile, _ = _hf_profile()
    messages = (
        user_message(VideoPart("a.mp4"), ImagePart("i.jpg"), TextPart("q"), VideoPart("b.mp4")),
    )
    assert profile.extract_videos(messages) == ["a.mp4", "b.mp4"]
    assert profile.extract_images(messages) == ["i.jpg"]
