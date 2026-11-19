#!/usr/bin/env python
"""GPU timing breakdown probe (vLLM engine path).

Splits per-image latency into CPU preprocessing / vision tower / LLM prefill /
decode, so the dominant component of a slow VLM request is measured rather than
guessed. The request path mirrors the VllmBackend (prepare -> raw prompt ->
multi_modal_data) so the findings extrapolate.
Rendering builds the same hf_template profile the backends resolve, so this tool
runs where transformers is installed (a GPU tool; the vllm extra already carries
it).

Usage (inside a GPU container, repo root, requires `uv sync --extra vllm`):
    uv run python scripts/probe_gpu_timing.py --stage vllm

The model / data locations come from `EYAS_PROBE_MODEL_PATH`,
`EYAS_PROBE_DATA_DIR`, `EYAS_PROBE_IMAGES_DIR`.

stage=vllm (core + the vllm engine extra):
    V  per-request latency under the benchmark protocol: 288 cold images in one
       shot + max_new_tokens=4096 + text warmup, engine params pinned item by item
       (eager / no prefix cache / 16k chunked prefill / mem 0.85 / ctx 32768) ->
       the per-request figure the engine path actually delivers; GPU util sampling
       + py-spy on the engine subprocess for attribution

Output protocol: one `PROBE_RESULT <name> <json>` line per measurement point; a failed
step emits `PROBE_RESULT <name> {"error": ...}` and later steps continue.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from typing import Any

MODEL_PATH = os.environ.get("EYAS_PROBE_MODEL_PATH", "/path/to/Qwen3.5-9B")
DATA_DIR = os.environ.get("EYAS_PROBE_DATA_DIR", "/path/to/CUB")
IMAGES_DIR = os.environ.get("EYAS_PROBE_IMAGES_DIR", "/path/to/CUB/images")
INSTRUCTION = "Which bird species is shown in the image? Answer with the species name."


def emit(name: str, payload: Any) -> None:
    print(
        f"PROBE_RESULT {name} {json.dumps(payload, ensure_ascii=False, default=str)}",
        flush=True,
    )


def guarded(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:
        emit(
            name,
            {"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-2000:]},
        )


def load_eval_requests(n: int) -> list[Any]:
    """Build image-bearing requests via the McqTask path (full 200-class choices assembled; text fidelity preserved)."""
    from eyas.tasks.base import Variant
    from eyas.tasks.mcq import McqTask

    task = McqTask(dataset="CUB", data_dir=DATA_DIR, images_dir=IMAGES_DIR)
    samples = task.load_split("train")
    variant = Variant(id="identity", index=0)
    return [task.build_request(INSTRUCTION, s, variant) for s in samples[:n]]


def _probe_profile() -> Any:
    """The same template path the backends resolve."""
    from eyas.backends.profiles import resolve_profile

    return resolve_profile({"type": "hf_template"}, MODEL_PATH)


def text_only(req: Any) -> Any:
    """Same request minus the images (via the raw_prompt escape hatch); text token count preserved."""
    from eyas.backends.base import ChatRequest, Message, TextPart

    profile = _probe_profile()
    text_messages = tuple(
        Message(
            role=message.role,
            content=tuple(part for part in message.content if isinstance(part, TextPart)),
        )
        for message in req.messages
    )
    return ChatRequest(messages=req.messages, raw_prompt=profile.render(text_messages, False))


# ---------------------------------------------------------------------------
# attribution helpers: coarse GPU-util sampling + py-spy CPU stacks
# ---------------------------------------------------------------------------
def _sample_gpu_util(stop: threading.Event, samples: list[int]) -> None:
    """Background thread: samples GPU utilization every 200ms (coarse busy proxy when CUPTI deadlocks)."""
    while not stop.is_set():
        try:
            out = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                .stdout.strip()
                .splitlines()
            )
            if out:
                samples.append(int(out[0]))
        except Exception:
            pass
        stop.wait(0.2)


def _emit_pyspy_top(raw_path: str, name: str, top_n: int = 12) -> None:
    """py-spy --format raw (collapsed stacks) -> top-N heaviest stacks (summary of the 4 leaf frames)."""
    try:
        with open(raw_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        counter: Counter[str] = Counter()
        for ln in lines:
            stack, _, cnt = ln.rpartition(" ")
            try:
                counter[stack] += int(cnt)
            except ValueError:
                continue
        total = sum(counter.values())
        if not total:
            emit(name, {"error": "no samples (ptrace denied?)"})
            return
        top = [
            {"count": c, "leaf_frames": stack.split(";")[-4:]}
            for stack, c in counter.most_common(top_n)
        ]
        emit(name, {"total_samples": total, "top": top})
    except Exception as exc:
        emit(name, {"error": repr(exc)})


# ---------------------------------------------------------------------------
# stage=vllm (vLLM engine experiment: per-request latency under the benchmark protocol)
#
# Reproduces the 288-image / max_new_tokens=4096 workload with the engine params
# pinned item by item in the constructor below, so the per-request figure is the
# one the vLLM engine path actually delivers rather than a guess. GPU util
# sampling + py-spy on the engine subprocess do the attribution (cold image
# read+decode plus processor work explain only a small share of the per-request
# time; the remainder lands inside the engine).
# Rendering is also client-side single placeholder (the vLLM processor expands it at
# <|image_pad|> itself), multi_modal_data passed as PIL — consistent with the
# VllmBackend path. Note: run this stage in the vllm-extra venv (`uv sync --extra
# vllm`); the current extra already carries the qwen3_5-capable engine versions.
# ---------------------------------------------------------------------------
def stage_vllm() -> None:
    import vllm
    from vllm import LLM, SamplingParams

    from eyas.backends.base import SamplingConfig
    from eyas.backends.vllm import VllmBackend

    emit("v_env", {"vllm_version": vllm.__version__})

    backend = VllmBackend(model_path=MODEL_PATH, max_inflight=1024)
    sampling = SamplingConfig(temperature=0.0, max_new_tokens=4096)
    img_reqs = load_eval_requests(288)
    prepared = backend.prepare(img_reqs, sampling)

    # Watchdog: dumps all thread stacks every 15min on hang, preserving the scene
    faulthandler.dump_traceback_later(900, repeat=True)
    print("[vllm] constructing engine (weights from the configured model_path)", flush=True)
    # Engine params pinned item by item (the published benchmark protocol):
    #   gpu_memory_utilization=0.85 — 85% of device memory for one engine
    #   max_model_len=32768 — the 32k context the workload renders into
    #   enforce_eager=True — no CUDA graphs, so per-step time stays measurable
    #   enable_prefix_caching=False — a cold prefix for every request
    #   max_num_batched_tokens=16384 — 16k chunked prefill
    #   max_num_seqs=512 — 512 concurrent sequences, far above the 288-image batch
    engine = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        dtype="auto",
        disable_log_stats=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
    )
    print("[vllm] engine constructed", flush=True)

    entries: list[dict[str, Any]] = []
    for item in prepared:
        entry: dict[str, Any] = {"prompt": item.prompt}
        mm = backend._multi_modal_data(item)
        if mm is not None:
            entry["multi_modal_data"] = mm
        entries.append(entry)
    print(f"[vllm] prepare done: {len(entries)} entries", flush=True)

    # V0 text warmup — mirrors the main measurement's sampling (burns one-off init/JIT so it stays clean)
    warm_sampling = SamplingConfig(temperature=0.0, max_new_tokens=32)
    warm_entries = [
        {"prompt": item.prompt}
        for item in backend.prepare([text_only(r) for r in img_reqs[:32]], warm_sampling)
    ]
    t0 = time.perf_counter()
    engine.generate(warm_entries, SamplingParams(**backend._sampling_params(warm_sampling)))
    emit("v0_text_x32_warmup", {"n": 32, "wall_s": round(time.perf_counter() - t0, 2)})

    # py-spy samples the engine subprocess (EngineCore/worker, 50 Hz collapsed stacks)
    kids = subprocess.run(
        ["pgrep", "-P", str(os.getpid())], capture_output=True, text=True
    ).stdout.split()
    pyspy_bin = shutil.which("py-spy")
    pyspy_runs: list[tuple[str, str, subprocess.Popen[bytes]]] = []
    if pyspy_bin and kids:
        for kid in kids:
            out_path = os.path.join(tempfile.gettempdir(), f"pyspy_vllm_{kid}.raw")
            try:
                proc = subprocess.Popen(
                    [pyspy_bin, "record", "-p", kid, "-r", "50", "-o", out_path, "--format", "raw"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                pyspy_runs.append((kid, out_path, proc))
                print(f"[vllm] py-spy recording pid={kid}", flush=True)
            except OSError as exc:
                print(f"[vllm] py-spy launch failed pid={kid}: {exc!r}", flush=True)
    else:
        print(f"[vllm] py-spy unavailable (bin={pyspy_bin}, children={kids})", flush=True)

    stop_evt = threading.Event()
    util_samples: list[int] = []
    sampler = threading.Thread(target=_sample_gpu_util, args=(stop_evt, util_samples), daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    outs = engine.generate(entries, SamplingParams(**backend._sampling_params(sampling)))
    dt = time.perf_counter() - t0
    stop_evt.set()
    sampler.join(timeout=3)
    print(f"[vllm] generate done wall={dt:.1f}s", flush=True)
    if util_samples:
        srt = sorted(util_samples)
        emit(
            "v_gpu_util_coarse",
            {
                "n": len(util_samples),
                "mean": round(sum(util_samples) / len(util_samples), 1),
                "p50": srt[len(srt) // 2],
                "p90": srt[min(int(len(srt) * 0.9), len(srt) - 1)],
                "busy_frac_ge10": round(
                    sum(1 for v in util_samples if v >= 10) / len(util_samples), 3
                ),
            },
        )

    comp = sorted(len(o.outputs[0].token_ids) for o in outs)

    def pct(p: float) -> Any:
        return comp[min(int(p * len(comp)), len(comp) - 1)] if comp else None

    finish_types: dict[str, int] = {}
    for o in outs:
        key = o.outputs[0].finish_reason
        finish_types[str(key)] = finish_types.get(str(key), 0) + 1
    e2e = sorted(
        m.e2e_latency for o in outs if (m := o.metrics) is not None and m.e2e_latency is not None
    )
    emit(
        "v_oneshot_x288_4096",
        {
            "n": len(entries),
            "wall_s": round(dt, 2),
            "per_req_s": round(dt / max(len(entries), 1), 3),
            "completion_tokens": {
                "median": pct(0.5),
                "mean": round(sum(comp) / len(comp), 1) if comp else None,
                "p95": pct(0.95),
                "max": comp[-1] if comp else None,
                "sum": sum(comp) if comp else None,
            },
            "finish_types": finish_types,
            "e2e_min": round(e2e[0], 2) if e2e else None,
            "e2e_max": round(e2e[-1], 2) if e2e else None,
        },
    )

    for kid, out_path, proc in pyspy_runs:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        _emit_pyspy_top(out_path, f"v_pyspy_top_pid{kid}")
    faulthandler.cancel_dump_traceback_later()
    print("[vllm] all done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["vllm"], required=True)
    args = parser.parse_args()
    if args.stage == "vllm":
        guarded("stage_vllm", stage_vllm)


if __name__ == "__main__":
    main()
