"""vLLM backend fast tests: prepare rendering (client-side profile
rendering), sampling parameter mapping, concurrency safety and the
in-flight window on an AsyncLLM-shaped stub, VllmSlot lifecycle. All zero
GPU."""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ChatResponse,
    ImagePart,
    LoadedVideo,
    Message,
    SamplingConfig,
    TextPart,
    VideoPart,
    user_message,
)
from eyas.backends.prepare import PreparedRequest
from eyas.backends.profiles import ModelProfile
from eyas.backends.vllm import (
    _DEFAULT_WARMUP_RESERVE_GIB,
    _WARMUP_RESERVE_BYTES,
    VllmBackend,
    VllmSlot,
    _usage_from_output,
    _warmup_reserve_bytes,
)


@pytest.fixture(autouse=True)
def _fake_vllm_module(monkeypatch: pytest.MonkeyPatch):
    """agenerate lazily imports ``vllm.SamplingParams`` (V1 only accepts
    instances; a dict is misreported as pooling); the fast test environment
    has no vllm, so inject a minimal stand-in (a dict subclass keeps subscript
    assertions working)."""
    fake = ModuleType("vllm")

    class SamplingParams(dict):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    fake.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)


class _StubAsyncLLM:
    """AsyncLLM-shaped stub: per-request async generate stream (async
    generator, not a coroutine), recording every call and the in-flight
    concurrency peak."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.peak = 0

    async def generate(self, prompt: dict[str, Any], sampling_params: dict, request_id: str) -> Any:
        self.calls.append(
            {"prompt": prompt, "sampling_params": sampling_params, "request_id": request_id}
        )
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(text=f"resp:{request_id}:{prompt['prompt']}", finish_reason="stop")
            ],
            finished=True,
        )


def _backend_with_stub(
    max_inflight: int = 2, profile: ModelProfile | None = None
) -> tuple[VllmBackend, _StubAsyncLLM]:
    # the stub engine renders without transformers: the fixed profile is the default
    backend = VllmBackend(
        model_path="dummy", max_inflight=max_inflight, profile=profile or _FixedProfile()
    )
    stub = _StubAsyncLLM()
    backend.slot.engine = stub
    return backend, stub


# ---------------------------------------------------------------------------
# prepare / rendering (client-side profile rendering)
# ---------------------------------------------------------------------------
class _FixedProfile(ModelProfile):
    """Deterministic profile: the assertion anchor for what the backend renders."""

    def render(self, messages: tuple[Message, ...], thinking: bool | None) -> str:
        return "|".join(f"{message.role}:{index}" for index, message in enumerate(messages))


def test_prepare_renders_via_profile():
    profile = _FixedProfile()
    backend = VllmBackend(model_path="dummy", profile=profile)
    request = ChatRequest(messages=(user_message(ImagePart(image="a.jpg"), TextPart("q")),))
    prepared = backend.prepare([request], SamplingConfig())[0]
    # the backend renders exactly what the profile returns (no second renderer)
    assert prepared.prompt == profile.render(request.messages, thinking=False)
    assert prepared.images == ("a.jpg",)
    assert prepared.videos == ()


def test_prepare_uses_raw_prompt_escape_hatch():
    backend = VllmBackend(model_path="dummy")
    request = ChatRequest(messages=(user_message("hello"),), raw_prompt="<|im_start|>custom")
    assert backend.prepare([request], SamplingConfig()) == [
        PreparedRequest(prompt="<|im_start|>custom")
    ]


def test_prepare_renders_and_keeps_raw_prompt():
    """Mixed batch: raw_prompt passes through untouched, the other request is
    profile-rendered; images are extracted from both requests' messages."""
    profile = _FixedProfile()
    backend = VllmBackend(model_path="dummy", profile=profile)
    with_raw = ChatRequest(
        messages=(user_message(ImagePart(image="a.jpg"), TextPart("q")),),
        raw_prompt="<|im_start|>custom",
    )
    without_raw = ChatRequest(messages=(user_message(ImagePart(image="b.jpg"), TextPart("q")),))
    prepared = backend.prepare([with_raw, without_raw], SamplingConfig())
    assert prepared[0] == PreparedRequest(prompt="<|im_start|>custom", images=("a.jpg",))
    assert prepared[1] == PreparedRequest(
        prompt=profile.render(without_raw.messages, thinking=False), images=("b.jpg",)
    )


def test_prepare_returns_named_media_fields():
    backend = VllmBackend(model_path="dummy", profile=_FixedProfile())
    out = backend.prepare(
        [ChatRequest(messages=(user_message(VideoPart("clip.mp4"), "q"),))], SamplingConfig()
    )
    assert out[0].videos == ("clip.mp4",)
    assert out[0].images == ()
    assert out[0].prompt.startswith("user:")


# ---------------------------------------------------------------------------
# sampling parameter mapping (vllm SamplingParams naming)
# ---------------------------------------------------------------------------
def test_sampling_params_defaults():
    params = VllmBackend._sampling_params(SamplingConfig(temperature=0.0, max_new_tokens=4096))
    assert params == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": 4096,
        "skip_special_tokens": True,
    }


def test_sampling_params_explicit_overrides():
    params = VllmBackend._sampling_params(
        SamplingConfig(
            temperature=0.7, max_new_tokens=128, top_k=5, repetition_penalty=1.1, seed=42
        )
    )
    assert params["top_k"] == 5
    assert params["repetition_penalty"] == 1.1
    assert params["seed"] == 42
    assert params["max_tokens"] == 128


def test_sampling_params_pinned(vllm_cls):
    """SamplingConfig maps onto the vllm SamplingParams names item by item,
    pinned here so a silent mapping change fails."""
    sampling_with_defaults = SamplingConfig(
        temperature=0.7,
        max_new_tokens=128,
        top_p=0.9,
        top_k=5,
        repetition_penalty=1.1,
        seed=42,
    )
    assert vllm_cls._sampling_params(sampling_with_defaults) == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 5,
        "repetition_penalty": 1.1,
        "max_tokens": 128,
        "skip_special_tokens": True,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# agenerate: ordering, unique ids, in-flight window, multi_modal_data on an AsyncLLM-shaped stub
# ---------------------------------------------------------------------------
def test_agenerate_preserves_order_and_unique_ids():
    profile = _FixedProfile()
    backend, stub = _backend_with_stub(max_inflight=2, profile=profile)
    requests = [
        ChatRequest(messages=(user_message(f"q{i}"),), request_id=f"r{i}") for i in range(5)
    ]
    responses = asyncio.run(backend.agenerate(requests, SamplingConfig()))
    expected_prompts = [profile.render(r.messages, thinking=False) for r in requests]
    # result order = request order; engine calls submitted per request in input order, engine-side request_id globally unique
    assert [r.request_id for r in responses] == [f"r{i}" for i in range(5)]
    assert [c["prompt"]["prompt"] for c in stub.calls] == expected_prompts
    assert len({c["request_id"] for c in stub.calls}) == 5
    assert stub.peak <= 2  # max_inflight = client-side in-flight window


def test_agenerate_multi_modal_data_only_when_images(tmp_path: Any):
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", (2, 2)).save(tmp_path / "a.jpg")
    backend, stub = _backend_with_stub(max_inflight=10)
    requests = [
        ChatRequest(messages=(user_message("text only"),)),
        ChatRequest(
            messages=(
                user_message(ImagePart(image=str(tmp_path / "a.jpg")), TextPart("with image")),
            )
        ),
    ]
    asyncio.run(backend.agenerate(requests, SamplingConfig()))
    assert len(stub.calls) == 2
    assert "multi_modal_data" not in stub.calls[0]["prompt"]
    image = stub.calls[1]["prompt"]["multi_modal_data"]["image"]
    assert isinstance(image, Image.Image) and image.filename == str(tmp_path / "a.jpg")


def test_multi_modal_data_loads_pil(tmp_path: Any):
    pytest.importorskip("PIL")
    from PIL import Image

    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (2, 2)).save(tmp_path / name)
    backend = VllmBackend(model_path="dummy")
    single = backend._multi_modal_data(
        PreparedRequest(prompt="p", images=(str(tmp_path / "a.jpg"),))
    )
    multi = backend._multi_modal_data(
        PreparedRequest(prompt="p", images=(str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")))
    )
    assert isinstance(single["image"], Image.Image)
    assert isinstance(multi["image"], list) and len(multi["image"]) == 2
    assert backend._multi_modal_data(PreparedRequest(prompt="p")) is None


def test_multi_modal_data_uses_prefetched_video_payload(monkeypatch):
    from eyas.backends import vllm as vllm_backend

    monkeypatch.setattr(
        vllm_backend, "decode_video", lambda *a, **kw: pytest.fail("must not decode twice")
    )
    backend = VllmBackend(model_path="dummy")
    payload = LoadedVideo("a.mp4", ("frames", {"fps": 2}))
    one = backend._multi_modal_data(PreparedRequest(prompt="p", videos=(payload,)))
    two = backend._multi_modal_data(
        PreparedRequest(prompt="p", videos=(payload, LoadedVideo("b.mp4", ("frames2",))))
    )
    assert one == {"video": ("frames", {"fps": 2})}
    assert two == {"video": [("frames", {"fps": 2}), ("frames2",)]}


def test_multi_modal_data_decodes_bare_path_with_engine_video_kwargs(monkeypatch):
    from eyas.backends import vllm as vllm_backend

    seen = []
    monkeypatch.setattr(
        vllm_backend,
        "decode_video",
        lambda path, kwargs=None: seen.append((path, kwargs)) or LoadedVideo(path, ("frames",)),
    )
    backend = VllmBackend(
        model_path="dummy", engine_kwargs={"media_io_kwargs": {"video": {"num_frames": 8}}}
    )
    assert backend._multi_modal_data(PreparedRequest(prompt="p", videos=("a.mp4",))) == {
        "video": ("frames",)
    }
    assert seen == [("a.mp4", {"num_frames": 8})]


def test_video_loader_gets_decode_flag_and_engine_io_kwargs():
    backend = VllmBackend(
        model_path="dummy",
        prefetch_workers=2,
        engine_kwargs={"media_io_kwargs": {"video": {"num_frames": 8}}},
    )
    assert backend._loader.decode_videos is True
    assert backend._loader.video_io_kwargs == {"num_frames": 8}


class _OrderLoader:
    def __init__(self, order, futures=None, resolved=None):
        self.order, self.futures, self.resolved = order, futures or {}, resolved

    def begin(self, requests):
        self.order.append("begin")
        return self.futures

    async def resolve_image_async(self, image, futures):
        self.order.append("resolve.image")
        return image

    async def resolve_video_async(self, video, futures):
        self.order.append("resolve.video")
        return self.resolved if self.resolved is not None else video


def test_agenerate_submits_decode_before_engine_launch():
    order = []
    backend, stub = _backend_with_stub(max_inflight=4)
    backend._loader = _OrderLoader(order)
    backend.slot.ensure = lambda: order.append("ensure") or stub
    payload = LoadedVideo("a.mp4", ("frames",))
    req = ChatRequest(messages=(user_message(VideoPart(payload), "q"),), request_id="r0")
    asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert order == ["begin", "ensure", "resolve.video"]
    assert stub.calls[0]["prompt"]["multi_modal_data"] == {"video": ("frames",)}


def test_agenerate_resolves_future_inside_the_window():
    order = []
    resolved = LoadedVideo("a.mp4", ("decoded",))
    loader = _OrderLoader(order, futures={("video", "a.mp4"): Future()}, resolved=resolved)
    backend, stub = _backend_with_stub(max_inflight=4)
    backend._loader = loader
    stub.generate_orig = stub.generate

    async def generate(prompt, sampling_params, request_id):
        order.append("generate")
        async for item in stub.generate_orig(prompt, sampling_params, request_id):
            yield item

    stub.generate = generate
    req = ChatRequest(messages=(user_message(VideoPart("a.mp4"), "q"),), request_id="r0")
    asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert order == ["begin", "resolve.video", "generate"]
    assert stub.calls[0]["prompt"]["multi_modal_data"] == {"video": ("decoded",)}


@pytest.fixture
def eyas_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    eyas_root = logging.getLogger("eyas")
    old_propagate, old_level = eyas_root.propagate, eyas_root.level
    eyas_root.propagate = True
    caplog.set_level(logging.DEBUG, logger="eyas")
    yield caplog
    eyas_root.propagate = old_propagate
    eyas_root.setLevel(old_level)


def test_media_batch_logs_submit_and_done(eyas_logs):
    order = []
    backend, stub = _backend_with_stub(max_inflight=4)
    backend._loader = _OrderLoader(order, resolved=LoadedVideo("a.mp4", ("frames",)))
    req = ChatRequest(messages=(user_message(VideoPart("a.mp4"), "q"),), request_id="r0")
    asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert "[media] decode submitted requests=1 images=0 videos=1" in eyas_logs.text
    assert "[media] batch done" in eyas_logs.text and "wall=" in eyas_logs.text


def test_generate_failure_logs_request_context_first(eyas_logs):
    class _Boom:
        async def generate(self, prompt, sampling_params, request_id):
            raise RuntimeError("placeholder count mismatch")
            yield  # pragma: no cover

    backend, _ = _backend_with_stub()
    backend.slot.engine = _Boom()
    req = ChatRequest(messages=(user_message("q"),), request_id="r7")
    with pytest.raises(RuntimeError, match="placeholder count mismatch"):
        asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert "[media] generate failed request=r7" in eyas_logs.text
    assert "prompt_head=" in eyas_logs.text and "prompt_tail=" in eyas_logs.text


def test_decode_failure_degrades_to_sentinel_batch_survives(eyas_logs):
    """A missing clip is a per-REQUEST error: the failed request gets the
    over-window sentinel shape, the rest of the batch generates normally, and
    one WARNING summarizes the failed media count."""

    class _FailingLoader(_OrderLoader):
        async def resolve_video_async(self, video, futures):
            raise FileNotFoundError(f"missing clip {video}")

    backend, stub = _backend_with_stub(max_inflight=4)
    backend._loader = _FailingLoader([])
    requests = [
        ChatRequest(messages=(user_message("q0"),), request_id="r0"),
        ChatRequest(messages=(user_message(VideoPart("missing.mp4"), "q"),), request_id="r1"),
        ChatRequest(messages=(user_message("q2"),), request_id="r2"),
    ]
    responses = asyncio.run(backend.agenerate(requests, SamplingConfig()))
    assert [r.request_id for r in responses] == ["r0", "r1", "r2"]
    assert [r.text for r in responses] == [responses[0].text, "", responses[2].text]
    assert responses[0].text and responses[2].text  # the two healthy requests succeeded
    assert responses[1].finish_reason == OVER_WINDOW_FINISH
    assert len(stub.calls) == 2  # the failed request never reached the engine
    assert "[media] load failed request=r1" in eyas_logs.text
    assert "[media] 1/3 requests failed media load" in eyas_logs.text


def test_missing_image_degrades_to_sentinel_batch_survives(tmp_path: Path, eyas_logs):
    """No-prefetch path: the bare-path decode in _multi_modal_data hits the
    missing file — same per-request sentinel, batch survives (brief Step 1:
    batch of 3, middle image missing -> 2 successes + 1 error response)."""
    pytest.importorskip("PIL")
    from PIL import Image

    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (2, 2)).save(tmp_path / name)
    backend, stub = _backend_with_stub(max_inflight=4)
    requests = [
        ChatRequest(
            messages=(user_message(ImagePart(image=str(tmp_path / "a.jpg")), "q0"),),
            request_id="r0",
        ),
        ChatRequest(
            messages=(user_message(ImagePart(image=str(tmp_path / "missing.jpg")), "q1"),),
            request_id="r1",
        ),
        ChatRequest(
            messages=(user_message(ImagePart(image=str(tmp_path / "b.jpg")), "q2"),),
            request_id="r2",
        ),
    ]
    responses = asyncio.run(backend.agenerate(requests, SamplingConfig()))
    assert [r.request_id for r in responses] == ["r0", "r1", "r2"]
    assert responses[0].finish_reason == "stop" and responses[2].finish_reason == "stop"
    assert responses[1].text == "" and responses[1].finish_reason == OVER_WINDOW_FINISH
    assert len(stub.calls) == 2
    assert "[media] 1/3 requests failed media load" in eyas_logs.text


def test_corrupt_image_degrades_to_sentinel(tmp_path: Path) -> None:
    """Undecodable image (not just missing): the decode error degrades the same way."""
    pytest.importorskip("PIL")
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not-an-image")
    backend, stub = _backend_with_stub(max_inflight=4)
    req = ChatRequest(messages=(user_message(ImagePart(image=str(broken)), "q"),), request_id="r0")
    responses = asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert responses[0].text == "" and responses[0].finish_reason == OVER_WINDOW_FINISH
    assert stub.calls == []


def test_broken_prefetch_pool_still_raises():
    """Pool infrastructure death is NOT a per-file media failure: degrading it
    would silently null the whole batch — it must keep killing the batch."""
    from concurrent.futures.process import BrokenProcessPool

    class _BrokenLoader(_OrderLoader):
        async def resolve_image_async(self, image, futures):
            raise BrokenProcessPool("A child process terminated abruptly")

    backend, _ = _backend_with_stub()
    backend._loader = _BrokenLoader([])
    req = ChatRequest(messages=(user_message(ImagePart(image="a.jpg"), "q"),), request_id="r0")
    with pytest.raises(BrokenProcessPool):
        asyncio.run(backend.agenerate([req], SamplingConfig()))


class _FailOneEngine:
    """One request fails once every request has entered engine.generate; the
    others stream indefinitely until cancelled. Records abort calls (sync or
    coroutine shape, optionally broken) and engine-side cancellations."""

    def __init__(self, total: int, abort_shape: str) -> None:
        self.total = total
        self.entered: dict[str, str] = {}  # prompt -> engine request_id
        self.aborted: list[str] = []
        self.cancelled: set[str] = set()
        self.all_entered = asyncio.Event()
        if abort_shape == "async":

            async def abort(request_id: str) -> None:
                self.aborted.append(request_id)

        elif abort_shape == "sync":

            def abort(request_id: str) -> None:  # type: ignore[misc]
                self.aborted.append(request_id)

        else:  # "broken": abort itself fails; must not mask the original error

            def abort(request_id: str) -> None:  # type: ignore[misc]
                self.aborted.append(request_id)
                raise RuntimeError("abort transport dead")

        self.abort = abort

    async def generate(self, prompt: dict[str, Any], sampling_params: dict, request_id: str) -> Any:
        self.entered[prompt["prompt"]] = request_id
        if len(self.entered) == self.total:
            self.all_entered.set()
        if prompt["prompt"] == "boom":
            await self.all_entered.wait()  # the slow siblings are in-flight when the failure fires
            raise RuntimeError("engine boom")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled.add(request_id)
            raise
        yield SimpleNamespace(  # pragma: no cover
            outputs=[SimpleNamespace(text="never", finish_reason="stop")], finished=True
        )


@pytest.mark.parametrize("abort_shape", ["async", "sync", "broken"])
def test_agenerate_failure_aborts_inflight_and_cancels_siblings(abort_shape: str):
    """Batch failure: every still-in-flight engine request is aborted (GPU must
    not keep generating for a batch that already failed), sibling client tasks
    are cancelled, and the ORIGINAL exception propagates even when abort itself
    fails."""
    backend, _ = _backend_with_stub(max_inflight=8)
    engine = _FailOneEngine(total=3, abort_shape=abort_shape)
    backend.slot.engine = engine
    requests = [
        ChatRequest(messages=(user_message("q"),), raw_prompt=prompt, request_id=rid)
        for prompt, rid in [("boom", "r0"), ("slow-a", "r1"), ("slow-b", "r2")]
    ]
    with pytest.raises(RuntimeError, match="engine boom"):
        asyncio.run(backend.agenerate(requests, SamplingConfig()))
    slow_ids = {engine.entered["slow-a"], engine.entered["slow-b"]}
    assert slow_ids <= set(engine.aborted)  # in-flight requests aborted before propagation
    assert engine.entered["boom"] not in engine.aborted  # the failed request is no longer in-flight
    assert engine.cancelled == slow_ids  # sibling streams cancelled, not left running


def test_agenerate_success_never_aborts():
    backend, stub = _backend_with_stub(max_inflight=4)
    aborted: list[str] = []
    stub.abort = aborted.append  # type: ignore[attr-defined]
    requests = [ChatRequest(messages=(user_message(f"q{i}"),)) for i in range(5)]
    asyncio.run(backend.agenerate(requests, SamplingConfig()))
    assert aborted == []


def test_agenerate_empty_requests_no_engine_call():
    backend, stub = _backend_with_stub()
    assert asyncio.run(backend.agenerate([], SamplingConfig())) == []
    assert stub.calls == []


def test_agenerate_forwards_sampling_params():
    backend, stub = _backend_with_stub()
    asyncio.run(
        backend.agenerate(
            [ChatRequest(messages=(user_message("q"),))], SamplingConfig(max_new_tokens=77)
        )
    )
    assert stub.calls[0]["sampling_params"]["max_tokens"] == 77


def test_concurrent_agenerate_calls_coalesce_at_engine():
    """AsyncLLM contract regression: concurrent agenerate (the shape of
    evaluator judge_concurrency=128) safely enters the engine and gets batched
    by it (in-flight > 1) — this is exactly why AsyncLLM replaces the
    single-thread-pool serialization of synchronous LLM.generate."""
    backend, stub = _backend_with_stub(max_inflight=10)
    requests = [ChatRequest(messages=(user_message(f"q{i}"),)) for i in range(6)]

    async def fan_out() -> list[list[ChatResponse]]:
        return list(
            await asyncio.gather(*(backend.agenerate([r], SamplingConfig()) for r in requests))
        )

    results = asyncio.run(fan_out())
    assert len(results) == 6
    assert len(stub.calls) == 6
    assert stub.peak >= 2  # concurrent submissions overlap engine-side = batchable by the scheduler
    for result, call in zip(results, stub.calls, strict=True):
        assert result[0].text == f"resp:{call['request_id']}:{call['prompt']['prompt']}"


# ---------------------------------------------------------------------------
# VllmSlot: idempotent ensure / release (never touches real vllm)
# ---------------------------------------------------------------------------
def test_vllm_slot_ensure_idempotent():
    slot = VllmSlot(model_path="dummy")
    stub = _StubAsyncLLM()
    slot.engine = stub
    assert slot.ensure() is stub
    assert slot.ensure() is stub
    assert slot.is_active


def test_vllm_slot_ensure_launches_via_launcher(monkeypatch: pytest.MonkeyPatch):
    sentinel = _StubAsyncLLM()
    launched: list[str] = []
    monkeypatch.setattr(
        "eyas.backends.vllm.launch_vllm_engine",
        lambda model_path, **kwargs: (launched.append(model_path), sentinel)[1],
    )
    slot = VllmSlot(model_path="my-model", tp_size=2, mem_fraction=0.7)
    assert slot.ensure() is sentinel
    assert launched == ["my-model"]
    assert slot.ensure() is sentinel
    assert len(launched) == 1


def test_vllm_slot_release_calls_shutdown_and_idempotent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(VllmSlot, "_wait_gpu_memory_reclaimed", staticmethod(lambda *a, **kw: None))
    slot = VllmSlot(model_path="dummy")
    slot.release()  # no-op when inactive
    shutdowns: list[int] = []
    slot.engine = SimpleNamespace(shutdown=lambda: shutdowns.append(1))
    slot.release()
    assert slot.engine is None
    assert not slot.is_active
    assert shutdowns == [1]  # AsyncLLM.shutdown terminates EngineCore
    slot.release()  # second release is a no-op


def _fake_reclaim_env(monkeypatch: pytest.MonkeyPatch, mem_get_info: Any) -> dict[str, float]:
    """Inject fake torch + fake time: mem_get_info supplies a free-memory
    sample sequence, the clock accumulates sleep duration (replaces the
    module-level time binding, never touches real time.sleep)."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=mem_get_info)),
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "eyas.backends.vllm.time",
        SimpleNamespace(
            monotonic=lambda: clock["now"],
            sleep=lambda s: clock.__setitem__("now", clock["now"] + s),
        ),
    )
    return clock


def test_vllm_slot_release_waits_for_memory_reclaim(monkeypatch: pytest.MonkeyPatch):
    """EngineCore is a subprocess: shutdown returning != memory reclaimed —
    do not release early while free memory stays flat below the threshold
    (EngineCore teardown is not instantaneous, so a flat curve alone is not
    evidence of reclamation), so wait until the reclaim threshold is crossed."""
    samples = iter([100, 100, 100, 600_000])

    def mem_get_info() -> tuple[int, int]:
        return (next(samples, 600_000), 1_000_000)  # total=1e6 -> threshold 575000

    clock = _fake_reclaim_env(monkeypatch, mem_get_info)

    slot = VllmSlot(model_path="dummy")
    shutdowns: list[int] = []
    slot.engine = SimpleNamespace(shutdown=lambda: shutdowns.append(1))
    slot.release()

    assert shutdowns == [1]
    assert (
        clock["now"] == 4.0
    )  # flat below threshold for 2 rounds, released only when round 3 crosses it
    assert next(samples, None) is None  # 600_000 consumed (reclaim seen before release)


def test_vllm_slot_ensure_caps_fraction_by_free_memory(monkeypatch: pytest.MonkeyPatch):
    """When another process already holds memory on the same GPU, the f=0.85
    budget + that residency + the warmup spike overflows — ensure must narrow
    the budget by the measured free memory and write it back to the slot so
    release's reclaim check uses the same basis."""
    free, total = int(67.99 * 2**30), int(79.25 * 2**30)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (free, total))
        ),
    )
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "eyas.backends.vllm.launch_vllm_engine",
        lambda model_path, **kwargs: (seen.update(kwargs), _StubAsyncLLM())[1],
    )
    slot = VllmSlot(model_path="dummy", mem_fraction=0.85)
    slot.ensure()
    expected = (free - _WARMUP_RESERVE_BYTES) / total
    assert seen["mem_fraction"] == pytest.approx(expected)
    assert slot.mem_fraction == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["abc", "", "nan", "inf", "1e308", "-1"])
def test_warmup_reserve_env_garbage_keeps_default(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """The knob is read at import time, so a typo must not raise for every CLI verb."""
    monkeypatch.setenv("EYAS_VLLM_WARMUP_RESERVE_GIB", raw)
    assert _warmup_reserve_bytes() == int(_DEFAULT_WARMUP_RESERVE_GIB * 2**30)


@pytest.mark.parametrize(("raw", "gib"), [("0", 0.0), ("8", 8.0), ("0.5", 0.5)])
def test_warmup_reserve_env_accepts_values(
    monkeypatch: pytest.MonkeyPatch, raw: str, gib: float
) -> None:
    monkeypatch.setenv("EYAS_VLLM_WARMUP_RESERVE_GIB", raw)
    assert _warmup_reserve_bytes() == int(gib * 2**30)


def test_vllm_slot_ensure_keeps_fraction_when_free_abundant(monkeypatch: pytest.MonkeyPatch):
    """Cold start (no co-resident usage): the derived cap is above the configured value, behavior unchanged."""
    free, total = int(78.84 * 2**30), int(79.25 * 2**30)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (free, total))
        ),
    )
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "eyas.backends.vllm.launch_vllm_engine",
        lambda model_path, **kwargs: (seen.update(kwargs), _StubAsyncLLM())[1],
    )
    slot = VllmSlot(model_path="dummy", mem_fraction=0.85)
    slot.ensure()
    assert seen["mem_fraction"] == 0.85


def test_vllm_slot_ensure_fails_fast_when_free_tiny(monkeypatch: pytest.MonkeyPatch):
    """Co-resident usage so large the derived cap < 0.5: fail fast rather than silently shrinking KV."""
    free, total = int(5.0 * 2**30), int(79.25 * 2**30)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (free, total))
        ),
    )
    slot = VllmSlot(model_path="dummy", mem_fraction=0.85)
    with pytest.raises(RuntimeError, match="not enough free GPU memory"):
        slot.ensure()
    assert slot.engine is None


def test_vllm_slot_release_memory_wait_timeout_raises(monkeypatch: pytest.MonkeyPatch):
    """Free memory never reaches the threshold -> timeout fail fast rather
    than a silent release (continuing with the engine still holding memory is
    a guaranteed OOM; a best-effort release only postpones and blurs the
    cause)."""
    state = {"tick": 0}

    def mem_get_info() -> tuple[int, int]:
        state["tick"] += 1
        return (state["tick"] * 100, 1_000_000)  # always far below the 575000 threshold

    clock = _fake_reclaim_env(monkeypatch, mem_get_info)

    slot = VllmSlot(model_path="dummy")
    slot.engine = SimpleNamespace(shutdown=lambda: None)
    with pytest.raises(RuntimeError, match="not reclaimed"):
        slot.release()

    assert clock["now"] == 300.0  # raise at the deadline instead of looping forever


def test_launch_force_overrides_engine_kwargs_mem_fraction(monkeypatch: pytest.MonkeyPatch):
    """Regression anchor: gpu_memory_utilization in config
    engine_kwargs, expanded via **engine_kwargs, would override the
    lifecycle-derived value — the derived value must win forcibly, or the engine
    fills the configured budget regardless of the folded headroom."""
    seen: dict[str, Any] = {}

    engine_args_mod = ModuleType("vllm.engine.arg_utils")

    class AsyncEngineArgs:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    engine_args_mod.AsyncEngineArgs = AsyncEngineArgs  # type: ignore[attr-defined]
    async_llm_mod = ModuleType("vllm.v1.engine.async_llm")

    class AsyncLLM:
        @classmethod
        def from_engine_args(cls, args: Any) -> str:
            return "engine"

    async_llm_mod.AsyncLLM = AsyncLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.engine", ModuleType("vllm.engine"))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", engine_args_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", ModuleType("vllm.v1.engine"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", async_llm_mod)

    from eyas.backends.vllm import launch_vllm_engine

    launch_vllm_engine(
        model_path="dummy",
        mem_fraction=0.8064,
        # the default path is **self.engine_kwargs unpacking (VllmSlot.ensure);
        # only same-name keys actually enter the kwargs dict; the
        # engine_kwargs=... keyword form would nest as {'engine_kwargs': ...}
        **{"gpu_memory_utilization": 0.85, "enforce_eager": True},
    )
    assert seen["gpu_memory_utilization"] == 0.8064
    assert seen["enforce_eager"] is True


# ---------------------------------------------------------------------------
# declared resource limits (assembly-time intersection input)
# ---------------------------------------------------------------------------
def test_max_images_defaults_to_vllm_default():
    # vllm's own limit_mm_per_prompt default is 1 image per prompt; an unset
    # config must declare that, not None (None would skip the intersection).
    backend = VllmBackend(model_path="dummy")
    assert backend.max_images == 1


def test_max_images_reads_engine_image_limit():
    backend = VllmBackend(model_path="dummy", engine_kwargs={"limit_mm_per_prompt": {"image": 4}})
    assert backend.max_images == 4


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_vllm_backend_registered():
    import eyas.builtin  # noqa: F401
    from eyas.registry import get

    assert get("backend", "vllm") is VllmBackend


def test_usage_from_output_token_id_counts():
    out = SimpleNamespace(prompt_token_ids=[1, 2, 3], outputs=[SimpleNamespace(token_ids=[4, 5])])
    assert _usage_from_output(out) == {"prompt_tokens": 3, "completion_tokens": 2}


def test_usage_from_output_malformed_returns_none():
    assert _usage_from_output(SimpleNamespace(outputs=[])) is None  # no completion section
    assert _usage_from_output("raw string") is None  # no token id attributes
    assert _usage_from_output(SimpleNamespace(prompt_token_ids=None, outputs=[])) is None
