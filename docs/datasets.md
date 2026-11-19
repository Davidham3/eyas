# Data and models

Eyas ships code, not data and not weights. Every benchmark dataset is downloaded at
prepare time from its upstream host, and every model is referenced by a local
checkpoint path in the run config. Nothing here redistributes a dataset, a model, or a
derived copy of either: `datasets/` is gitignored and the published tables are
measurements of runs you can reproduce yourself.

**This repository's Apache-2.0 license covers the code.** Each dataset and each model
checkpoint keeps its own license and terms of use, independent of this repo's license,
and some benchmark sets are released for research use only. Check the upstream page
before downloading and cite the dataset authors in any work built on them.

## Datasets

| Dataset | Task | Upstream source | Prepared by |
| --- | --- | --- | --- |
| pets | fine-grained breed MCQ | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (VGG, Oxford), via HF `timm/oxford-iiit-pet` | `eyas prepare` |
| cub | bird-species MCQ (8 families) | [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), via HF `cassiekang/cub200_dataset` | same |
| sroie | key-information extraction | [SROIE, ICDAR 2019](https://rrc.cvc.uab.es/?ch=13), via HF `jinho8345/sroie-bio` | same |
| textvqa | open-ended scene-text QA | [TextVQA](https://textvqa.org/), via HF `lmms-lab/TextVQA` | same |
| gsm8k | numeric math QA | [GSM8K](https://github.com/openai/grade-school-math), via HF `openai/gsm8k` | same |
| nextqa | video next-action QA | [NExT-QA](https://github.com/doc-doc/NExT-QA) annotations; the 6.5 GB video pack is linked there and stays **user-supplied** — the script never downloads it | same |
| mixed | seven-combination mixed MCQ | none — synthesized locally (solid-color images and clips, no third-party bytes) | `scripts/prepare_mixed_data.py` |

The prepare scripts fetch through a fallback chain (Hugging Face direct → hf-mirror →
ModelScope for pets) and persist only the sampled subset the runs actually consume.
The mixed dataset is the exception: it downloads nothing and is synthesized locally;
its clip encoder needs PyAV, so run it as
`uv run --with av python scripts/prepare_mixed_data.py` (default output
`datasets/bench/mixed/`, the directory the `mixed/<method>` cells read).

## Models

Benchmark configs read `backend.params.model_path` (in-process vLLM) or
`backend.params.model` (OpenAI-compatible endpoint). Point them at a checkpoint you
obtained under its own license: Qwen and InternVL weights, for example, are governed by
their upstream model licenses, not by this repository. No weights are distributed here,
and every published score is tied to the checkpoint the run recorded.

## What the benchmark publishes

Accuracy, cost, prompts and per-sample evaluation records from runs, plus the code and
configs that reproduce them. Data itself never enters the repository or the package.
