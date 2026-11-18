"""Chat-template adaptation (canonical messages → each model's raw prompt).

Rendering is template-first: every family renders with the chat template
shipped alongside its weights (``tokenizer.apply_chat_template``), so a
model runs with nothing but its ``model_path`` and no per-family rendering
code exists:
- TextPart maps to ``{"type": "text"}``, ImagePart to ``{"type": "image"}``
  and VideoPart to ``{"type": "video"}`` in HF chat-format; the template owns
  all special tokens and placeholders;
- media are materialized out-of-band via ``extract_images`` /
  ``extract_videos`` and handed to the engine as multimodal input (images and
  videos may arrive prefetched);
- the thinking tri-state is a deployment-layer choice, forwarded to the
  template only when ``thinking_kwarg`` names its variable (families differ,
  e.g. ``enable_thinking``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eyas.backends.base import ImagePart, LoadedImage, LoadedVideo, Message, TextPart, VideoPart

__all__ = ["ModelProfile", "HfTemplateProfile", "resolve_profile"]


class ModelProfile(ABC):
    """Rendering protocol: messages + thinking → raw prompt string; media parts → template
    placeholders + the out-of-band image / video lists."""

    @abstractmethod
    def render(self, messages: tuple[Message, ...], thinking: bool | None) -> str: ...

    def extract_images(self, messages: tuple[Message, ...]) -> list[str | LoadedImage]:
        images: list[str | LoadedImage] = []
        for message in messages:
            for part in message.content:
                if isinstance(part, ImagePart):
                    images.append(part.image)
        return images

    def extract_videos(self, messages: tuple[Message, ...]) -> list[str | LoadedVideo]:
        return [
            part.video
            for message in messages
            for part in message.content
            if isinstance(part, VideoPart)
        ]


class HfTemplateProfile(ModelProfile):
    """The default profile: render with the model's own shipped HF chat template.

    Messages map to HF chat-format dicts (TextPart -> {"type": "text"},
    ImagePart -> {"type": "image"}, VideoPart -> {"type": "video"}), so any
    family whose shipped chat_template handles structured content renders
    exactly by construction -- no per-family rendering code. Templates that
    only accept string content (older minimal ChatML releases) are
    unsupported: they inject no vision placeholder, and image-/video-bearing
    requests fail fast at the engine's placeholder/media count check.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        thinking_kwarg: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        # thinking stays a deployment-layer choice: the tri-state is only
        # forwarded when the caller names the template variable (families
        # differ, e.g. enable_thinking); otherwise the template default wins.
        self.thinking_kwarg = thinking_kwarg
        self._tokenizer: Any | None = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # engine extras carry it

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
        return self._tokenizer

    def render(self, messages: tuple[Message, ...], thinking: bool | None) -> str:
        hf_messages: list[dict[str, object]] = []
        for message in messages:
            content: list[dict[str, str]] = []
            for part in message.content:
                if isinstance(part, ImagePart):
                    content.append({"type": "image"})
                elif isinstance(part, VideoPart):
                    content.append({"type": "video"})
                elif isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
            hf_messages.append({"role": message.role, "content": content})
        kwargs = dict(self.chat_template_kwargs)
        if self.thinking_kwarg is not None:
            # in-process engines have no server default to consult: a declared
            # kwarg is always forwarded, and None keeps the prefill rendered off
            kwargs[self.thinking_kwarg] = bool(thinking)
        return self._get_tokenizer().apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True, **kwargs
        )


def resolve_profile(
    spec: ModelProfile | str | dict[str, Any] | None, model_path: str
) -> ModelProfile:
    """Build a profile from the config knob backend.params.profile.

    None -> HfTemplateProfile rendering with the model's shipped chat
    template; an instance passes through (tests inject fakes); a string names
    a built-in; a dict carries ``{"type": <name>, **ctor_kwargs}`` --
    model_path is injected for specs that need it.
    """
    if spec is None:
        return HfTemplateProfile(model_path=model_path)
    if isinstance(spec, ModelProfile):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    name = spec.get("type")
    kwargs = {key: value for key, value in spec.items() if key != "type"}
    if name == "hf_template":
        kwargs.setdefault("model_path", model_path)
        return HfTemplateProfile(**kwargs)
    raise ValueError(f"unknown profile type: {name!r} (hf_template)")
