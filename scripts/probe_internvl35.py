"""Cross-family rendering probe: official InternVL3.5 on the vllm backend.

Adjudicates which rendering the native input mapper consumes, from one engine
load: A) the shipped chat template (HfTemplateProfile, literal image tag);
B) manual IMG-sequence expansion (raw_prompt escape hatch).

    uv sync --extra vllm
    uv run python scripts/probe_internvl35.py --model-path PATH --image-path IMG
"""

from __future__ import annotations

import argparse
import glob
import re

from eyas.backends.base import (
    ChatRequest,
    ImagePart,
    SamplingConfig,
    TextPart,
    user_message,
)
from eyas.backends.profiles import HfTemplateProfile

QUERY = "What animal is in this image? Answer with one word."


def _im(role: str) -> str:
    # assembled, never literal: the ChatML delimiters are special tokens
    return chr(60) + "|" + role + "|" + chr(62)


def first_shard(model_path: str) -> str:
    shards = sorted(glob.glob(model_path.rstrip("/") + "/model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {model_path}")
    return shards[0]


def sanitize(text: str) -> str:
    return re.sub(r"<\|([A-Za-z_]+)\|>", r"[\1]", text).replace("\n", "\\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image-path", required=True)
    ap.add_argument("--num-image-token", type=int, default=256)
    ap.add_argument("--model-len", type=int, default=8192)
    args = ap.parse_args()

    from eyas.backends.vllm import VllmBackend

    backend = VllmBackend(
        model_path=args.model_path,
        mem_fraction=0.85,
        context_length=args.model_len,
        engine_kwargs={"trust_remote_code": True, "enforce_eager": True},
        profile={"type": "hf_template", "trust_remote_code": True},
    )
    profile = backend.profile
    assert isinstance(profile, HfTemplateProfile)
    messages = (user_message(ImagePart(image=args.image_path), TextPart(QUERY)),)
    rendered = profile.render(messages, thinking=None)
    print("[probe] A_rendered:", sanitize(rendered))

    from safetensors import safe_open

    ctx_id = None
    with safe_open(first_shard(args.model_path), framework="pt") as f:
        for key in f.keys():
            if "img_context" in key.lower():
                ctx_id = int(f.get_tensor(key).item())
                break
    if ctx_id is None:
        for token, token_id in profile._get_tokenizer().get_vocab().items():
            if token.strip("<>").upper() == "IMG" + "_CONTEXT":
                ctx_id = token_id
                break
    print("[probe] IMG_CONTEXT id:", ctx_id)
    assert ctx_id is not None, "IMG_CONTEXT token not found"

    left = "<" + "IMG_START" + ">"
    ctx = "<" + "IMG_CONTEXT" + ">"
    right = "<" + "IMG_END" + ">"
    prompt_b = (
        _im("user")
        + "\n"
        + left
        + ctx * args.num_image_token
        + right
        + "\n"
        + QUERY
        + _im("end")
        + "\n"
        + _im("assistant")
        + "\n"
    )
    print("[probe] B_len:", len(prompt_b), "head:", sanitize(prompt_b[:48]))

    sampling = SamplingConfig(temperature=0.0, max_new_tokens=32, seed=42)
    images = profile.extract_images(messages) or None
    req_b = ChatRequest(messages=messages, raw_prompt=prompt_b)
    for label, request in (("A_template", ChatRequest(messages=messages)), ("B_manual", req_b)):
        try:
            response = backend.generate([request], sampling)[0]
            text = (response.text or "").strip()
            n_img = len(images) if images else 0
            print(f"[probe] {label}: images={n_img} reply={text!r}")
        except Exception as exc:  # fail loud per variant, keep the other signal
            print(f"[probe] {label}: ERROR {type(exc).__name__}: {str(exc)[:300]}")


if __name__ == "__main__":
    main()
