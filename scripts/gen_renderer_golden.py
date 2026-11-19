"""Generate byte-parity golden anchors for a model's shipped HF chat template.

The goldens are the template-owned rendering truth: every case is rendered ONCE
directly through ``AutoTokenizer.apply_chat_template`` with the chat template
shipped alongside the weights (the same path ``HfTemplateProfile`` uses at
inference time), and the exact rendered bytes are checked in. The slow-tier
test (tests/test_renderer_goldens.py) re-renders through the repo's profile
with the real tokenizer and diffs byte-for-byte: the digest equality is the
gate. The stamp (transformers version + chat_template.jinja sha256) is
provenance — template-bytes drift fails the test, a transformers version
drift only warns (renders were measured byte-identical across 5.19.0 /
5.15.1 / 4.57.1).

The case matrix mirrors the renderer parity experiment: system/user/assistant
history, thinking tri-state (True/False/None), single & multi-image, no-system,
whitespace-padded system. Pinned rendering truths that the anchors lock in
(don't "fix" them): the shipped template emits NO newline after the vision
placeholder, and it trims system-content whitespace.

Usage (needs ``transformers``, which the vllm extra pins):
    uv run --extra vllm python scripts/gen_renderer_golden.py \
        --model-path /path/to/Qwen3.5-9B \
        --out tests/golden/qwen35_template

The emitted file contains the exact rendered bytes, including the model's
special-token tags (ChatML envelope / vision placeholder markers); print it
through a sanitizer, never raw, when debugging (raw tags can mangle tool
output pipelines).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# HF chat-format content parts: the same conversion the repo's profile does at
# render time (TextPart -> {"type": "text"}, ImagePart -> {"type": "image"}).
IMG = [{"type": "image"}, {"type": "text", "text": "What color is the square?"}]
SYS = [{"type": "text", "text": "You are a helpful assistant."}]
TXT = [{"type": "text", "text": "Classify the animal."}]
PAD = [{"type": "text", "text": "\n\n  padded system  \n\n"}]
TXT2 = [{"type": "text", "text": "It is a cat."}]
TXTQ = [{"type": "text", "text": "And now?"}]


def _turns(*turns: tuple[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    return [{"role": role, "content": content} for role, content in turns]


# (name, messages, thinking) -- the parity case matrix. thinking is the
# tri-state the deployment layer would forward under a declared template
# variable: True/False pass through, None renders prefill with thinking off.
CASES: list[tuple[str, list[dict[str, Any]], bool | None]] = [
    (
        "sys+user_text thinking=off",
        _turns(("system", SYS), ("user", TXT)),
        False,
    ),
    (
        "sys+user_text thinking=on",
        _turns(("system", SYS), ("user", TXT)),
        True,
    ),
    (
        "sys+user_text thinking=None",
        _turns(("system", SYS), ("user", TXT)),
        None,
    ),
    (
        "sys+user_img+text thinking=off",
        _turns(("system", SYS), ("user", IMG)),
        False,
    ),
    (
        "no_system user_img+text thinking=off",
        _turns(("user", IMG)),
        False,
    ),
    (
        "history w assistant thinking=off",
        _turns(("system", SYS), ("user", TXT), ("assistant", TXT2), ("user", TXTQ)),
        False,
    ),
    (
        "padded system thinking=off",
        _turns(("system", PAD), ("user", TXT)),
        False,
    ),
    (
        "multi-image interleaved thinking=off",
        _turns(("system", SYS), ("user", IMG + [{"type": "image"}] + TXT)),
        False,
    ),
]


def _template_sha256(model_path: Path) -> str:
    return hashlib.sha256((model_path / "chat_template.jinja").read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="model dir carrying chat_template.jinja"
    )
    parser.add_argument("--out", required=True, help="output dir for goldens.json")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_dir():
        print(f"error: model path not found: {model_path}", file=sys.stderr)
        return 2

    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    template_sha = _template_sha256(model_path)
    transformers_version = transformers.__version__

    cases: list[dict[str, Any]] = []
    for name, messages, thinking in CASES:
        # the tri-state rule matches the profile's declared-kwarg contract:
        # the variable is always forwarded, and None keeps the prefill off
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(thinking),
        )
        cases.append(
            {"name": name, "messages": messages, "thinking": thinking, "expected": rendered}
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "stamp": {
            "transformers": transformers_version,
            "template_sha256": template_sha,
        },
        "cases": cases,
    }
    out_file = out_dir / "goldens.json"
    out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")

    # digest summary only: the rendered bytes contain special-token tags and
    # must never be printed raw
    for case in cases:
        digest = hashlib.sha256(case["expected"].encode("utf-8")).hexdigest()[:12]
        print(f"{case['name']}: len={len(case['expected'])} sha256[:12]={digest}")
    print(
        f"wrote {out_file} (stamp: transformers={transformers_version} template_sha256={template_sha[:12]}...)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
