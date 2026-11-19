# Inference backends

**What you'll learn:** the canonical request form, the registered inference backends and their config knobs, deployment sampling vs. algorithm-tier sampling, the evaluation record store, and how to add a backend of your own.

## The canonical request form

Every backend speaks the same canonical form, an ordered list of messages whose content is a sequence of typed parts:

```python
from eyas.backends.base import ChatRequest, ImagePart, TextPart, system_message, user_message

request = ChatRequest(
    messages=(
        system_message("You are a helpful assistant."),
        user_message(ImagePart(image="/data/images/a.jpg"), TextPart("What color is the square?")),
    ),
    request_id="a:identity",
)
```

`ImagePart` accepts a local path, a URL, or a preloaded `LoadedImage` payload. Template rendering is deliberately **not** part of this layer: the in-process engine (vLLM) renders the messages with the model's chat template on the client side; the OpenAI-compatible backend sends the messages and lets the server render. That split is what keeps both shapes behaviorally identical for the same weights and template.

Responses are uniform too: `ChatResponse.text` is the clean answer (inline thinking is stripped into `reasoning` by the normalizing wrapper that sits in front of every engine), `finish_reason` records truncation/over-window events, and `usage` carries token counts.

## The two inference backends

| Name | Class | Install | Typical use |
| --- | --- | --- | --- |
| `vllm` | In-process vLLM engine | `uv pip install "eyas[vllm]"` | Throughput runs on your own GPU |
| `openai` | OpenAI-compatible HTTP client | `uv pip install "eyas[api]"` | Any remote endpoint serving `/v1/chat/completions` |

The registry carries one more entry beyond these two: the framework's scripted in-repo test backend, the zero-GPU fixture the whole `tests/` contract suite drives every task and strategy through (it is what the contract-test sections of the custom-task and custom-method tutorials point at).

### In-process engine (`vllm`)

```yaml
backend:
  name: vllm
  params:
    model_path: /path/to/Qwen3.5-9B     # local checkpoint directory
    max_inflight: 512                   # client-side concurrency window
    prefetch_workers: 8                 # multiprocess image prefetch (0 = off)
    engine_kwargs:                      # passed straight to the engine
      enforce_eager: true
      max_num_batched_tokens: 16384
  sampling:
    temperature: 0.0
    max_new_tokens: 1024
```

- `max_inflight` is the **client-side in-flight window**: requests slide in continuously, no batch barriers. Raise it until the engine saturates; the `EYAS_MAX_INFLIGHT` environment variable overrides it without touching the config (invalid values fail fast).
- `prefetch_workers > 0` enables the framework's image prefetch loader: images are decoded in worker processes and handed to the engine as loaded payloads, so the GPU never waits on disk.
- `engine_kwargs` are engine-native; tune them for your card and serving profile.

### OpenAI-compatible endpoints (`openai`)

```yaml
backend:
  name: openai
  params:
    base_url: https://your-endpoint.example.com/v1/
    model: qwen3.5-9b
    api_key_env: MY_ENDPOINT_KEY    # credentials come from the environment, never the config
    max_inflight: 16
    max_retries: 2
```

`api_key_env` names an environment variable holding the key: the backend fails fast at startup if it is unset, so secrets never land in configs or git. `extra_body` is the escape hatch for provider-specific request fields.

The fastest end-to-end path onto your own endpoint is the [Getting started](getting-started.md) config: save `quickstart_textvqa_opro.yaml`, keep its `openai` backend section, fill `base_url`/`model`, and export the key named in `api_key_env`. Per-machine values can ride `--set` (`--set backend.params.base_url=...`, `--set backend.params.model=...`). Repository checkouts also have the docs story's single-pass baseline pre-wired as the `textvqa/raw` cell's `openai` variant: `uv run eyas run --cell textvqa/raw --variant openai`, or `uv run eyas gen-configs --cell textvqa/raw --variant openai` to export it as a file (developer path).

## Sampling semantics: deployment layer vs. algorithm tiers

`backend.sampling` sets **deployment-level** defaults (temperature, `max_new_tokens`, `top_p`, `top_k`, `repetition_penalty`, `seed`). The special three-state `thinking` flag (`None` = leave the server default, `True`/`False` = force on/off) controls reasoning-mode rendering.

Strategies that make optimizer-side generation calls (asking a model to propose or rewrite prompts) never hand-construct sampling configs. They call through a **tier name** (`generate`, `evolve`, `gradient`, ...) and the framework merges:

1. the **tier profile** from the config's `sampling_profiles` section, algorithm intent only (temperature, `top_p`, `max_new_tokens`);
2. **deployment fields** (`thinking`, `seed`, ...) always inherited from `backend.sampling`.

```yaml
sampling_profiles:
  generate: { temperature: 0.9, max_new_tokens: 512 }
  gradient: { temperature: 0.7, max_new_tokens: 1024 }
```

This is why a tier that forgot to set anything still respects your deployment defaults instead of silently reverting to server defaults, and why `thinking`/`seed` can never be overridden by algorithm code.

## The evaluation record store

In a run, requests and scores flow through the **Evaluator**, whose record store persists one durable record per (prompt, sample, split) (prediction plus scores), keyed by the prompt's content hash. A store hit short-circuits inference entirely; that is the checkpoint mechanism described in [Understanding results](understanding-results.md).

## Watching throughput

During large batches the scheduler emits `[batch.progress]` heartbeats (rate + ETA) and a `[batch.stall]` warning if 600 seconds pass with zero completions. A giant batch that is silently grinding is **healthy**: check for the heartbeat, not for console chatter. A watchdog terminates the process (`exit 99`) after 20 minutes of no activity at all, on the principle that a hung GPU job is worse than a dead one. Every threshold involved is tunable through the environment:

## Environment-variable reference

All knobs below are read from the process environment; unset means the listed default. The liveness defaults suit a local GPU engine. **Slow remote endpoints need a raised watchdog threshold** (a minutes-long remote round-trip is legitimate silence, not a hang; `EYAS_WATCHDOG_STALL_SEC=3600` is the typical remote-endpoint setting).

| Variable | Default | Effect |
| --- | --- | --- |
| `EYAS_WATCHDOG_STALL_SEC` | `1200` | Activity-watchdog threshold in seconds: no activity (engine interaction / batch completion) past it → hang diagnostics + `exit 99`. `<=0` disables the watchdog. Raise it for slow remote endpoints (e.g. `3600`). |
| `EYAS_DIAG_DIR` | `/tmp` | Directory for the watchdog's hang-diagnostics file (`eyas-watchdog-<pid>.log`, includes a full-thread stack snapshot). Read at call time — an `os.environ` set after import still takes effect; point it elsewhere when `/tmp` is not writable or not persisted. |
| `EYAS_BATCH_STALL_SEC` | `600` | Per-batch zero-progress alert: a dispatched batch with no new completion for this many seconds logs `[batch.stall]` (a warning, never a kill). `<=0` disables the alert. |
| `EYAS_BATCH_STALL_REPEAT_SEC` | `600` | Minimum interval between repeated `[batch.stall]` alerts for the same stall. |
| `EYAS_BATCH_WATCH_INTERVAL_SEC` | `30` | Poll interval of the zero-progress guard. |
| `EYAS_BATCH_PROGRESS_MIN_INTERVAL_SEC` | `60` | Time backstop for `[batch.progress]` heartbeats on batches of ≥500 requests: besides every 500th completion, a heartbeat is forced when this many seconds passed since the last one. `<=0` disables the backstop (pure count throttling). |
| `EYAS_MAX_INFLIGHT` | unset (config value) | Overrides `backend.params.max_inflight` for the inference backends without editing the config. Must be an integer ≥ 1; invalid values fail fast. |
| `EYAS_LOG_LEVEL` | `INFO` | Framework log level: `DEBUG` / `INFO` / `WARNING` / `ERROR` (case-insensitive); invalid values fail fast. `DEBUG` opens the phase-breadcrumb inventory (`[loop]`, `[eval]`, `[planner]`, `[prefetch]`, ...). |
| `EYAS_VLLM_WARMUP_RESERVE_GIB` | `4.5` | GiB of GPU memory reserved for the vLLM warmup spike when the lifecycle manager folds `gpu_memory_utilization` to measured free memory. Accepts finite values in `[0, 1024]`; an invalid value warns and keeps the default. |
| `EYAS_DATASET_ROOT` (fallback `DATASET_ROOT`) | unset | Root directory for named-dataset path resolution when a task config omits explicit `data_dir` / `images_dir`; `EYAS_DATASET_ROOT` wins over `DATASET_ROOT`. Explicit config paths always win over both. |

Two more environment facts, not user knobs: the vllm backend sets `VLLM_USE_FLASHINFER_SAMPLER=0` at engine launch unless you already set it (the flashinfer sampler's JIT fails in CUDA < 12.9 containers), and prefetch worker processes force `CUDA_VISIBLE_DEVICES=""` internally so media decode stays on CPU.

The GPU timing probe helper (`scripts/probe_gpu_timing.py` in a repository checkout) reads three variables of its own: `EYAS_PROBE_MODEL_PATH` (model weights directory), `EYAS_PROBE_DATA_DIR` (dataset JSON directory), and `EYAS_PROBE_IMAGES_DIR` (image directory). Their defaults are `/path/to/...` placeholders, so set all three to real paths before running the probe; the framework runtime never reads them.

## Adding your own backend

Implement the interface and register it:

```python
from collections.abc import Sequence

from eyas.backends.base import ChatRequest, ChatResponse, InferenceBackend, SamplingConfig
from eyas.registry import register_backend


@register_backend("myengine")
class MyBackend(InferenceBackend):
    async def agenerate(
        self, requests: Sequence[ChatRequest], sampling: SamplingConfig
    ) -> list[ChatResponse]:
        ...  # fan out, return one ChatResponse per request, in order

    def close(self) -> None:
        ...  # release engines/processes; optional — the run closes the backend after the loop
```

Rules of the road:

- Return responses **in input order**, one per request; never raise for a single bad request if the batch can continue: encode per-request failure in the response (that is what `finish_reason` is for).
- Route sync calls through the provided `generate()` helper: it uses the framework's process-resident event loop; calling `asyncio.run` yourself will deadlock in-process engines.
- Bump the activity watchdog (`eyas.backends.vllm` shows the pattern) on every engine interaction if you implement long-blocking calls.
- Verify backend-agnosticism: the same config with `name` swapped should produce statistically identical scores (the published benchmark's backend-consistency experiment is exactly this check).

The worked example is [`examples/plugin_backend/`](../../examples/plugin_backend/README.md) in a repository checkout: a third-party backend registered via the config `plugins` key (or entry points, for packaged plugins), with a runnable config and a CI test pinning the registration.

## Next steps

- [Reproducing the benchmark](benchmark.md): both backend shapes exercised on public datasets.
- [Custom tasks](custom-task.md) covers the other side of the request: what gets asked.
