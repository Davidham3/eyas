"""Shared request preparation + sampling parameter mapping for the local
engine backend (VllmBackend).

The engine speaks the rendering protocol (ModelProfile renders the
prompt client-side, images and videos travel out-of-band), so the prepare
loop and the sampling-parameter dictionary live here. The token-cap key is a
class attribute (``sampling_tokens_key``): vLLM sets ``"max_tokens"``. The
backend keeps its contract-anchor comment where it sets that attribute; the
contract is pinned by ``test_sampling_params_pinned`` in its backend test
file.

``EngineBackendMixin`` carries the rendering-protocol prepare loop and the
sampling-params mapping, so the concrete engine class only owns engine wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from eyas.backends.base import ChatRequest, LoadedImage, LoadedVideo, SamplingConfig
from eyas.backends.profiles import ModelProfile

__all__ = ["EngineBackendMixin", "PreparedRequest", "common_sampling_params", "prepare_requests"]


@dataclass(frozen=True)
class PreparedRequest:
    """One request after client-side rendering: prompt + its out-of-band media.

    Named fields (not a tuple) so each backend picks the media it maps, and a
    later media kind adds a field instead of shifting a positional contract.
    """

    prompt: str
    images: tuple[str | LoadedImage, ...] = ()
    videos: tuple[str | LoadedVideo, ...] = ()


def prepare_requests(
    profile: ModelProfile, requests: Sequence[ChatRequest], sampling: SamplingConfig
) -> list[PreparedRequest]:
    """messages → prepared prompt + extracted media.

    The ``raw_prompt`` escape hatch skips template expansion, serving the
    radix-cache-friendly path; media are extracted from every request's
    messages regardless of which prompt path was taken.
    """
    prepared: list[PreparedRequest] = []
    for req in requests:
        prompt = (
            req.raw_prompt
            if req.raw_prompt is not None
            else profile.render(req.messages, sampling.thinking)
        )
        prepared.append(
            PreparedRequest(
                prompt=prompt,
                images=tuple(profile.extract_images(req.messages)),
                videos=tuple(profile.extract_videos(req.messages)),
            )
        )
    return prepared


def common_sampling_params(sampling: SamplingConfig, tokens_key: str) -> dict[str, Any]:
    """SamplingConfig → engine sampling params, pinned item by item.

    ``tokens_key`` names the token-cap entry — the only local-engine token cap
    key in this repository is ``"max_tokens"``; every other key and default is
    pinned item by item.
    """
    params: dict[str, Any] = {
        "temperature": sampling.temperature,
        "top_p": 1.0 if sampling.top_p is None else sampling.top_p,
        "top_k": -1 if sampling.top_k is None else sampling.top_k,
        "repetition_penalty": (
            1.0 if sampling.repetition_penalty is None else sampling.repetition_penalty
        ),
        tokens_key: sampling.max_new_tokens,
        "skip_special_tokens": True,
    }
    if sampling.seed is not None:
        params["seed"] = sampling.seed
    return params


class EngineBackendMixin:
    """prepare + sampling-params methods of the local engine backend.

    Mixed in ahead of ``InferenceBackend``; the concrete backend supplies the
    ``profile`` instance and its ``sampling_tokens_key``. ``prepare`` stays
    public as the zero-GPU test anchor on rendered output.
    """

    profile: ModelProfile

    #: Name of the token-cap key in this engine's sampling params — the only
    #: local-engine token cap key in this repository is ``"max_tokens"``.
    sampling_tokens_key: ClassVar[str]

    def prepare(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[PreparedRequest]:
        """messages → prepared prompt + extracted media. Public for zero-GPU test assertions on rendered output."""
        return prepare_requests(self.profile, requests, sampling)

    @classmethod
    def _sampling_params(cls, sampling: SamplingConfig) -> dict[str, Any]:
        # sampling_params pinned item by item (contract anchor); the token cap
        # travels under cls.sampling_tokens_key
        return common_sampling_params(sampling, cls.sampling_tokens_key)
