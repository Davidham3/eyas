# Getting started

**What you'll learn:** what problem Eyas solves, how to install it, and how to walk the docs' flagship story end to end: optimize a TextVQA prompt with OPRO against any OpenAI-compatible endpoint and measure a real lift on a held-out test split.

## What is automatic prompt optimization?

Large multimodal models are prompt-sensitive: the wording of an instruction can move accuracy on a fine-grained classification, an entity-extraction task or a video question by several points, and hand-tuning prompts is slow, task-specific, and hard to reproduce. **Automatic prompt optimization (APO)** treats the prompt itself as the thing to be trained: start from a seed prompt, generate candidate rewrites, score them on a validation split, keep what works, and repeat. Eyas runs that loop on multimodal tasks, where "scoring a prompt" means rendering it into hundreds of text/image/video requests and applying task metrics. How the loop is organized internally is the [contract-driven design note](../10-architecture.md)'s job; here you just run it.

## 1. Install

Eyas requires **Python: core 3.12–3.14; vLLM extra verified on 3.12**. Install it into your
environment with [uv](https://docs.astral.sh/uv/) (`pip install` works too):

```bash
uv pip install "eyas[api,prepare]"   # the story path: HTTP client + dataset prep
```

The story on this page talks to an OpenAI-compatible endpoint and builds its data with
`uv run eyas prepare`, so it needs both extras. Extras compose; to instead serve the model
in-process:

| Install | What you get | Notes |
| --- | --- | --- |
| `uv pip install "eyas[vllm]"` | In-process vLLM engine | Recommended default; brings its own torch/transformers |
| `uv pip install "eyas[api]"` | OpenAI-compatible HTTP client | Lightest option; rendering happens server-side |
| `uv pip install "eyas[prepare]"` | `eyas prepare` dataset download/convert | Needed once per dataset; heavy reader stack (datasets, pyarrow) |

Every command below uses that `eyas` console script. Working on Eyas itself — editing source,
running `--cell` benchmark cells, or reading `configs/` and `examples/` — starts from a git
checkout with `uv sync`; see [CONTRIBUTING](https://github.com/Davidham3/eyas/blob/main/CONTRIBUTING.md).

## 2. One command to data

The story task is TextVQA, scene-text visual question answering, scored rule-based against ten human answers per question (no LLM judge). One command builds the exact splits every config on this page points at:

```bash
uv run eyas prepare --dataset textvqa
```

Honest size note: this streams the labeled validation split of `lmms-lab/TextVQA` (3 parquet shards of ~310 MB each with embedded images, measured ~7 min at 2.2 MB/s via the public mirror) and writes out only the sampled slice: seeded train 100 / val 30 / test 100 questions land under `datasets/quickstart/textvqa/`, with zero image overlap across splits.

## 3. The config — the whole secret is two lines

Save this as `quickstart_textvqa_opro.yaml`. It is the story cell (TextVQA × OPRO, rounds=4)
in inline form; a repository checkout can regenerate the same file from `configs/spec` with
`uv run eyas gen-configs --cell textvqa/opro --variant openai` (developer path — the spec stays the
source). `base_url`/`model` are placeholders: point them at your endpoint via
`--set backend.params.base_url=... --set backend.params.model=...` (API key via the
environment variable named in `api_key_env`).

```yaml
run:
  name: quickstart_textvqa_opro_openai
  seed: 42
  final_eval:
    split: test

task:
  name: vqa
  params:
    data_dir: datasets/quickstart/textvqa/data
    images_dir: datasets/quickstart/textvqa/images

backend:
  name: openai
  params:
    base_url: http://localhost:8000/v1
    model: qwen3.5-9b
    api_key_env: EYAS_DOC_STUB_KEY
    max_inflight: 16
  sampling:
    thinking: false

protocol:
  name: identity

eval_model:
  name: absolute

budget:
  name: flat
  params: {}  # FlatBudget is parameterless: every candidate, full split (explicit pin)

strategy:
  name: opro
  params:
    rounds: 4
```

Everything outside the `backend` section is the experiment setup, dissected line by line on the [Your first optimization](first-optimization.md) page. The connection to your model is two lines: `base_url` and `api_key_env`. Any OpenAI-compatible endpoint works: a self-hosted vLLM or SGLang server, or a hosted service whose URL ends in `/v1`. `api_key_env` names the *environment variable* holding your key (`export EYAS_DOC_STUB_KEY=<your-key>`); the key itself never enters configs or git, and the backend fails fast at startup if the variable is unset. The shipped file pins `http://localhost:8000/v1` and `qwen3.5-9b` as placeholders; point them at your endpoint without editing the file:

```text
--set backend.params.base_url=https://api.example.com/v1 --set backend.params.model=<your-served-model-name>
```

## 4. Run the story

Baseline first: `raw` scores the seed prompt once, so you know what you started from. Same
endpoint, same splits, one strategy swap (`--set strategy.params={}` clears OPRO's `rounds`
knob, which `raw` does not take):

```bash
uv run eyas run --config quickstart_textvqa_opro.yaml --set strategy.name=raw --set strategy.params={}
```

Then the optimization:

```bash
uv run eyas run --config quickstart_textvqa_opro.yaml
```

Each run writes a self-contained run directory under `outputs/<run.name>/<timestamp>/` (`result.json`, `checkpoint.json`, `status.json`, `framework.log`).

## 5. What the run prints

The optimization rounds stream as bare console lines. From the seed-42 run of the command above:

```text
round 1/400: 100 reqs in 10.94s eta≈43.74s
round 2/400: 400 reqs in 49.33s eta≈90.40s
round 3/400: 400 reqs in 42.16s eta≈68.28s
round 4/400: 400 reqs in 42.53s eta≈36.24s
round 5/400: 400 reqs in 42.68s eta≈0.00s
```

Five lines, one full loop: the first scores the seed prompt on the 100-sample train split; the next four are OPRO's propose-and-score rounds, each proposing 4 new candidates, and every candidate is scored on the full train split, which is where the 400 requests per round come from. Provenance: this excerpt, the prompts and the numbers below are captured from the seed-42 run of exactly the two commands above, against a self-hosted OpenAI-compatible endpoint serving Qwen3.5-9B (the same checkpoint and engine settings as the published study); the accuracy table quotes all three fixed seeds.

## 6. The ending

### What the optimizer changed

The seed prompt (input to the loop, verbatim):

```text
Answer the question about the target image with a short factual phrase (one to five words).
```

The winner OPRO returned on seed 42 (verbatim):

```text
Extract the precise fact from the image and reply with only that exact value, limited to one to five words.
```

Same contract, tightened: the winner forbids the sentence-framing that costs soft-match points and keeps the length band, so the model's answers get terse enough to match the human annotations the scorer compares against.

### The lift

Held-out test accuracy (n = 100), mean ± σ over the fixed seeds 42/43/44, Qwen3.5-9B, measured by re-running the two commands above on an OpenAI-compatible endpoint, run directories kept with the material page:

| Method | test accuracy | Δ vs raw (pp) |
| --- | --- | --- |
| raw | 0.736 ± 0.004 | — |
| opro | **0.788 ± 0.029** | +5.2 |

Read this honestly: every number here comes from Qwen3.5-9B, yours will vary by model. At n = 100 a single reading carries a ±4 to 5 pp noise band, which is why the table quotes three-seed means; every seed lifted (+1.3 / +9.0 / +5.3 pp), but the seed-to-seed spread (σ ≈ 2.9 pp) is the real run-to-run variance of a four-round search on a 100-question train split. The published in-process-vLLM wave of the same configs ([quickstart benchmark §E2, the three-seed effect table](../07-quickstart-benchmark.md)) measured 0.729 ± 0.002 → 0.816 ± 0.003; its three seeds landed unusually tight, so read that Δ alongside the spread shown here, not instead of it.

### The bill

A single-pass baseline costs 100 requests; the 4-round OPRO search cost 1,700 requests and 1,391,946 prompt tokens, in 153.6 s of loop wall time. There is no judge channel to bill on this task (TextVQA scoring is rule-based), and every line of the accounting, judge channel included wherever one exists, is in `result.json`: see [understanding results](understanding-results.md).

## 7. No endpoint yet?

You can still take the story exactly as written: the log, the prompts and the numbers are all above; come back with a key and type the two commands. To execute a complete loop today with zero downloads and zero keys, take the [guided tour](guided-tour.md), which runs the whole pipeline on a built-in synthetic task.

## Next steps

- [Your first optimization](first-optimization.md): dissect this page's config line by line and see what the strategy and the managed loop each do per round.
- [Guided tour](guided-tour.md): live state, every artifact, seeds and crash recovery.
- [Understanding results](understanding-results.md): what is inside `result.json`, how caching and resume work.
- [Methods](../methods.md): compare the eight shipped strategies and pick one.
- [Reproducing the benchmark](benchmark.md): the full published study.
