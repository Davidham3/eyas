# Custom tasks

**What you'll learn:** how to check whether a built-in task already covers your data, and how to write, register, wire and test your own task when it does not.

This page assumes Eyas is installed (`uv pip install "eyas[api]"`; see
[Getting started](getting-started.md)) and that you run `eyas` from the directory holding
your module and config. No repository checkout is required.

## Check the built-ins first

Most multimodal evaluation fits one of the shapes the framework already ships:

| Task | Data shape | Scoring | Use when |
| --- | --- | --- | --- |
| `mcq` | Multiple choice with options per group | Rule-based option matching, optional LLM judge | Classification, including binary (2 options) |
| `vqa` | Open-ended question + reference answers | VQA accuracy (`min(1, matches/3)`) | Free-form Q&A |
| `kie` | Image -> list of entity records | `field_acc` / `strict_acc` / `char_sim`, zero judge | Structured extraction (name/phone/address, ...) |
| `mixed_mcq` | Per-sample modality composition: text passage / image / clip in any non-empty subset | Rule-based option matching | One dataset mixing text-only, image, video and combined samples |

If your task is "classify the image into one of N options", use `mcq`; a binary task is just N=2. Only write a custom task when your scoring semantics genuinely differ (a new metric, a different parsing convention, a domain-specific label space).

## The task contract

A task is a class implementing four methods, plus two class-level declarations:

```python
class Task(ABC):
    name: ClassVar[str]                                   # set by @register_task
    objectives: ClassVar[tuple[Objective, ...]] = ()      # ordered; first = primary
    default_prompt: str = "Answer the question about the image."

    def load_split(self, split: str) -> Sequence[Sample]: ...
    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest: ...
    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction: ...
    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]: ...
```

The three dataclasses you handle:

- **`Sample`**: `id`, `inputs` (open mapping: image path, question, choices, ...), `references` (labels). Tasks interpret the mappings; the framework never does.
- **`Variant`**: one evaluation-protocol variant of a sample (`id` like `"identity"` or `"shift_2"`). Under the default `identity` protocol you can ignore it beyond putting it in `request_id`.
- **`Prediction`**: `value` plus the raw response text. **`value=None` means "could not parse"**: with a judge configured (top-level `judge:` section) it falls through to the judge; without one it scores `0.0` (the framework's documented convention for missing metrics).

## A worked example: binary attribute classification

Suppose your data is a directory of JSON manifests: `<data_dir>/<split>.json`, each a JSON array of `{"id", "image", "label"}` with `label` in `{"yes", "no"}` and absolute image paths. That scoring is exact match after light normalization, which `mcq`'s option-list semantics do not cover, so a small custom task is warranted.

```python
# my_tasks.py  (your working directory; imported via the `plugins` config key)
from __future__ import annotations

import json
from pathlib import Path

from eyas.backends.base import ChatRequest, ChatResponse, ImagePart, TextPart, user_message
from eyas.core.objective import Objective
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant

_IMAGE_MARKERS = ("{image}", "<image>")


@register_task("polarity")
class PolarityTask(Task):
    """Binary image classification from JSON manifests."""

    objectives = (Objective(key="accuracy", maximize=True),)
    default_prompt = "Does this image show a storefront? Answer yes or no."

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache: dict[str, list[Sample]] = {}

    def load_split(self, split: str) -> list[Sample]:
        if split not in self._cache:
            records = json.loads((self.data_dir / f"{split}.json").read_text())
            self._cache[split] = [
                Sample(
                    id=r["id"],
                    inputs={"image": r["image"]},
                    references={"label": str(r["label"]).strip().lower()},
                )
                for r in records
            ]
        return list(self._cache[split])

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        # Optimizers may rewrite the prompt freely; strip image placeholder
        # markers and always put the image first, instruction after.
        instruction = prompt
        for marker in _IMAGE_MARKERS:
            instruction = instruction.replace(marker, "")
        instruction = instruction.rstrip() or self.default_prompt
        text = f"{instruction}\nAnswer with one word: yes or no."
        return ChatRequest(
            messages=(user_message(ImagePart(image=sample.inputs["image"]), TextPart(text)),),
            request_id=f"{sample.id}:{variant.id}",
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        first = next((ln.strip().lower() for ln in response.text.splitlines() if ln.strip()), "")
        value = first if first in ("yes", "no") else None
        return Prediction(value=value, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        if prediction.value is None:
            return {"accuracy": 0.0}
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}
```

Design notes worth copying:

- **Placeholders are defensive.** Optimization strategies rewrite prompts aggressively; a rewritten prompt may lose `{image}` or `<image>` markers entirely. Strip the markers wherever they appear, always inject the image yourself, and append the output spec when the instruction went empty.
- **Parse leniently, score strictly.** `parse_response` returns `value=None` on anything ambiguous (the parse-fail metric stays visible in run outputs); `score` never invents a value.
- **Normalize on load.** Labels and responses are normalized once, in one place.

## Wire it into a run

The `plugins` config key imports modules relative to the working directory you launch `eyas` from (put your module there or on `PYTHONPATH`) *after* built-ins are registered, making it the third-party path into the registry:

```yaml
plugins: [my_tasks]

run:
  name: polarity_hill_climb
  seed: 42

task:
  name: polarity
  params:
    data_dir: datasets/polarity

protocol:
  name: identity

backend:
  name: openai          # any OpenAI-compatible endpoint (see Inference backends)
  params:
    base_url: http://localhost:8000/v1
    model: qwen3.5-9b
    api_key_env: MY_ENDPOINT_KEY
    max_inflight: 16

eval_model:
  name: absolute

budget:
  name: flat

strategy:
  name: hill_climb
  params:
    rounds: 3
```

```bash
uv run eyas run --config my_config.yaml
```

Registration routes, from least to most invasive:

1. `plugins: [...]` in the config, for business adapters that stay out of the framework package (this tutorial's route).
2. `entry_points` group `eyas.plugins`, for packaged plugins; modules listed there are imported on first registry access.
3. An import in `src/eyas/builtin.py`, only for contributions that belong inside the framework itself.

## Test it

The framework's testing pillar: **every component must work end-to-end against the framework's scripted in-repo test backend, zero GPU**, the same fixture the `tests/` suite drives every shipped task and strategy through. A minimal contract test for the task:

```python
import pytest
from eyas.backends.base import ChatResponse
from eyas.tasks.base import Sample, Variant

from my_tasks import PolarityTask


def test_polarity_roundtrip(tmp_path):
    (tmp_path / "train.json").write_text(
        '[{"id": "a", "image": "/img/a.jpg", "label": "Yes"}]'
    )
    task = PolarityTask(data_dir=str(tmp_path))
    sample = task.load_split("train")[0]

    request = task.build_request(task.default_prompt, sample, Variant(id="identity"))
    assert request.request_id == "a:identity"
    prediction = task.parse_response(ChatResponse(text="  yes\n"), sample)
    assert prediction.value == "yes"
    assert task.score(prediction, sample) == {"accuracy": 1.0}
```

For the evaluator-level contract, a repository checkout ships `tests/support/engine.py`, whose `make_evaluator(task, backend, protocol, produces)` is the standard evaluator assembly (planner + absolute scoring) the framework's own task suites ride on. Then point the config's `backend` section at that scripted test backend (its registry name is in the `uv run eyas list` tour of [Getting started](getting-started.md)) and run it: if the loop completes and `result.json` appears, your task composes with evaluation, record caching and the strategy.

## Next steps

- [Custom optimization methods](custom-method.md): the other half of the extension story.
- [Inference backends](backends.md): swap the scripted test backend for a real model; nothing else changes.
