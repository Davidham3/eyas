"""MediaPrefetchLoader and the backends' prefetch seams, fast tests.

Covers materialization semantics (order preservation / cross-request dedup /
LoadedImage idempotence / URL pass-through / fail fast) and payload expansion
on the two consumer paths: vllm / openai. All zero GPU; image
decoding depends on PIL, so the whole module skips when the core environment
lacks it. The fixture PNG is generated with stdlib zlib+struct (no test image
assets imported)."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
import zlib
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from eyas import watchdog
from eyas.backends.base import (
    ChatRequest,
    ImagePart,
    LoadedImage,
    LoadedVideo,
    SamplingConfig,
    VideoPart,
    user_message,
)
from eyas.backends.prefetch import MediaPrefetchLoader

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

_RESPONSE_BODY = {
    "id": "cmpl-1",
    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


# ---------------------------------------------------------------------------
# fixture: minimal PNG (8-bit RGB, no auxiliary chunks)
# ---------------------------------------------------------------------------
def _png_bytes(width: int = 2, height: int = 2, rgb: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    idat = zlib.compress(row * height)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _write_png(path: Path, rgb: tuple[int, int, int] = (255, 0, 0)) -> str:
    path.write_bytes(_png_bytes(rgb=rgb))
    return str(path)


def _loaded(path: str) -> LoadedImage:
    raw = Path(path).read_bytes()
    pil = Image.open(BytesIO(raw)).convert("RGB")
    return LoadedImage(path=path, raw=raw, pil=pil)


def _req(*images: Any) -> ChatRequest:
    """Each image becomes an ImagePart (str / LoadedImage both accepted), with a text part appended at the end."""
    parts = tuple(ImagePart(image=img) for img in images)
    return ChatRequest(messages=(user_message(*parts, "q"),))


@pytest.fixture(scope="module")
def loader():
    ld = MediaPrefetchLoader(workers=2)
    yield ld
    ld.close()


# ---------------------------------------------------------------------------
# materialize: materialization semantics
# ---------------------------------------------------------------------------
def test_materialize_preserves_order_and_decodes(loader, tmp_path: Path):
    a = _write_png(tmp_path / "a.png", rgb=(255, 0, 0))
    b = _write_png(tmp_path / "b.png", rgb=(0, 255, 0))
    requests = [_req(a), _req(b, a), _req("https://example.com/x.jpg")]
    out = loader.materialize(requests)

    first = out[0].messages[0].content[0].image
    assert isinstance(first, LoadedImage)
    assert first.path == a
    assert first.pil.size == (2, 2)
    assert first.pil.mode == "RGB"
    assert first.raw == Path(a).read_bytes()

    # order preserved across requests: [b, a] order inside the second request is unchanged
    second = [p.image for p in out[1].messages[0].content if isinstance(p, ImagePart)]
    assert isinstance(second[0], LoadedImage)
    assert second[0].path == b
    assert second[1].path == a
    # URL pass-through
    assert out[2].messages[0].content[0].image == "https://example.com/x.jpg"


def test_materialize_dedups_same_path_across_requests(loader, tmp_path: Path):
    a = _write_png(tmp_path / "a.png")
    out = loader.materialize([_req(a), _req(a)])
    first = out[0].messages[0].content[0].image
    second = out[1].messages[0].content[0].image
    assert first is second  # same path shares one future -> one LoadedImage object


def test_materialize_idempotent_for_loaded_images(loader, tmp_path: Path):
    a = _write_png(tmp_path / "a.png")
    once = loader.materialize([_req(a)])[0].messages[0].content[0].image
    twice = loader.materialize([_req(once)])[0].messages[0].content[0].image
    assert twice is once


def test_materialize_url_schemes_passthrough(loader):
    urls = [
        "https://example.com/a.jpg",
        "http://example.com/b.jpg",
        "data:image/png;base64,AAAA",
        "file:///tmp/a.jpg",
    ]
    out = loader.materialize([_req(*urls)])
    assert [p.image for p in out[0].messages[0].content if isinstance(p, ImagePart)] == urls


def test_materialize_bumps_watchdog(loader, tmp_path: Path):
    """materialize harvesting is a long blocking point outside the scheduling
    layer: a slow filesystem read hang must be visible to the watchdog."""
    a = _write_png(tmp_path / "a.png")
    watchdog.bump("before")
    loader.materialize([_req(a)])
    assert watchdog._last_source.startswith("prefetch.materialize")
    assert watchdog.stalled_for() < 1.0


def test_missing_image_fails_fast(loader, tmp_path: Path):
    missing = str(tmp_path / "nope.png")
    with pytest.raises(FileNotFoundError):
        loader.materialize([_req(missing)])


def test_corrupt_image_fails_fast(loader, tmp_path: Path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not-an-image")
    with pytest.raises(Image.UnidentifiedImageError):
        loader.materialize([_req(str(broken))])


def test_invalid_workers_rejected():
    with pytest.raises(ValueError, match="workers"):
        MediaPrefetchLoader(workers=0)


def test_materialize_partial_degrades_per_request(loader, tmp_path: Path):
    """Per-file failures are reported by request index instead of raising;
    failed requests come back UNmaterialized, healthy ones fully loaded."""
    a = _write_png(tmp_path / "a.png")
    missing = str(tmp_path / "nope.png")
    requests = [_req(a), _req(missing), _req(a)]
    out, errors = loader.materialize_partial(requests)
    assert isinstance(out[0].messages[0].content[0].image, LoadedImage)
    assert out[1] is requests[1]  # failed request returned unchanged
    assert isinstance(out[2].messages[0].content[0].image, LoadedImage)
    assert list(errors) == [1]
    assert isinstance(errors[1], FileNotFoundError)


class _BrokenPool:
    """Pool stub whose every submission yields a BrokenProcessPool future.

    ``shutdown`` is a no-op so the loader can discard it without touching real
    worker processes; tests that need a PERMANENTLY broken pool patch
    ``_ensure_pool`` to keep returning one (rebuild cannot heal it).
    """

    def submit(self, fn, *args):
        from concurrent.futures.process import BrokenProcessPool

        fut = Future()
        fut.set_exception(BrokenProcessPool("A child process terminated abruptly"))
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_broken_pool_rebuilds_and_materialize_succeeds(tmp_path: Path):
    """Brief (b): ONE injected BrokenProcessPool is infrastructure, not a
    per-file failure — the loader discards it, rebuilds once, and the batch
    materializes successfully (success resets the consecutive-break count)."""
    a = _write_png(tmp_path / "a.png")
    ld = MediaPrefetchLoader(workers=2)
    try:
        ld._pool = _BrokenPool()
        out = ld.materialize([_req(a)])
        assert isinstance(out[0].messages[0].content[0].image, LoadedImage)
        assert not isinstance(ld._pool, _BrokenPool)  # rebuilt into a real pool
        assert ld._consecutive_breaks == 0
    finally:
        ld.close()


def test_broken_pool_rebuilds_and_materialize_partial_succeeds(tmp_path: Path):
    """openai harvest path: one breakage recovers within the batch; healthy
    requests come back materialized and NO per-file error is reported (pool
    death is never sentinelized)."""
    a = _write_png(tmp_path / "a.png")
    ld = MediaPrefetchLoader(workers=2)
    try:
        ld._pool = _BrokenPool()
        out, errors = ld.materialize_partial([_req(a), _req(a)])
        assert errors == {}
        assert isinstance(out[0].messages[0].content[0].image, LoadedImage)
        assert ld._consecutive_breaks == 0
    finally:
        ld.close()


def test_second_consecutive_break_raises_actionable(tmp_path: Path, monkeypatch):
    """A pool that breaks AGAIN after the rebuild (permanent stub) raises with
    an actionable message instead of looping — and the dead pool is discarded,
    never leaked."""
    from concurrent.futures.process import BrokenProcessPool

    a = _write_png(tmp_path / "a.png")
    ld = MediaPrefetchLoader(workers=2)
    broken = _BrokenPool()
    monkeypatch.setattr(ld, "_ensure_pool", lambda: broken)
    with pytest.raises(BrokenProcessPool, match="prefetch_workers=0"):
        ld.materialize([_req(a)])
    assert ld._pool is None
    assert ld._consecutive_breaks >= 2


def test_materialize_partial_second_break_raises_not_sentinel(tmp_path: Path, monkeypatch):
    """T19 contract adapted: pool death is infrastructure — a permanent break
    raises (actionable), it is NEVER degraded to a per-request error dict."""
    from concurrent.futures.process import BrokenProcessPool

    a = _write_png(tmp_path / "a.png")
    ld = MediaPrefetchLoader(workers=2)
    broken = _BrokenPool()
    monkeypatch.setattr(ld, "_ensure_pool", lambda: broken)
    with pytest.raises(BrokenProcessPool, match="prefetch_workers=0"):
        ld.materialize_partial([_req(a)])
    assert ld._pool is None


def test_begin_submit_break_rebuilds(tmp_path: Path):
    """Submit-time breakage (pool already dead between batches): begin discards
    it, rebuilds once, and the submissions land on the fresh pool."""
    from concurrent.futures.process import BrokenProcessPool

    class _SubmitBroken:
        def submit(self, fn, *args):
            raise BrokenProcessPool("dead at submit")

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    a = _write_png(tmp_path / "a.png")
    ld = MediaPrefetchLoader(workers=2)
    try:
        ld._pool = _SubmitBroken()
        futures = ld.begin([_req(a)])
        assert ("image", a) in futures
        assert isinstance(futures[("image", a)].result(), LoadedImage)
        assert not isinstance(ld._pool, _SubmitBroken)
    finally:
        ld.close()


def test_resolve_break_fails_fast_then_second_raises_actionable():
    """vllm mid-window contract: a resolve-time break is fail-fast (the batch
    dies, siblings cancelled / engine aborted upstream) and discards the pool
    so the NEXT begin rebuilds; a second consecutive break (no successful
    harvest between) raises the actionable message."""
    from concurrent.futures.process import BrokenProcessPool

    ld = MediaPrefetchLoader(workers=2)
    fut = Future()
    fut.set_exception(BrokenProcessPool("died"))
    with pytest.raises(BrokenProcessPool) as first:
        asyncio.run(ld.resolve_image_async("a.png", {("image", "a.png"): fut}))
    assert "prefetch_workers=0" not in str(first.value)  # first break: plain fail-fast
    assert ld._pool is None
    assert ld._consecutive_breaks == 1

    fut2 = Future()
    fut2.set_exception(BrokenProcessPool("died again"))
    with pytest.raises(BrokenProcessPool, match="prefetch_workers=0"):
        asyncio.run(ld.resolve_image_async("a.png", {("image", "a.png"): fut2}))


def test_close_then_reuse_spawns_new_pool(loader, tmp_path: Path):
    a = _write_png(tmp_path / "a.png")
    loader.close()
    assert loader._pool is None
    out = loader.materialize([_req(a)])  # reuse after close automatically respawns the pool
    assert isinstance(out[0].messages[0].content[0].image, LoadedImage)


# ---------------------------------------------------------------------------
# vllm seam: LoadedImage / str mixes go through _multi_modal_data
# ---------------------------------------------------------------------------
def test_vllm_multi_modal_data_accepts_loaded_image(tmp_path: Path):
    from eyas.backends.prepare import PreparedRequest
    from eyas.backends.vllm import VllmBackend

    a = _write_png(tmp_path / "a.png")
    li = _loaded(a)
    backend = VllmBackend(model_path="dummy")
    single = backend._multi_modal_data(PreparedRequest(prompt="p", images=(li,)))
    assert single == {"image": li.pil}

    pair = backend._multi_modal_data(PreparedRequest(prompt="p", images=(li, a)))
    assert isinstance(pair, dict) and isinstance(pair["image"], list)
    assert pair["image"][0] is li.pil
    assert isinstance(pair["image"][1], Image.Image)
    assert backend._multi_modal_data(PreparedRequest(prompt="p")) is None


# ---------------------------------------------------------------------------
# openai seam: LoadedImage -> data URL (raw bytes used directly, same as the str path)
# ---------------------------------------------------------------------------
def test_openai_image_url_from_loaded_image(tmp_path: Path):
    from eyas.backends.openai_api import OpenAIBackend

    a = _write_png(tmp_path / "a.png")
    url = OpenAIBackend._media_url(_loaded(a), "image/jpeg")
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == Path(a).read_bytes()
    # identical to the str path reading directly
    assert url == OpenAIBackend._media_url(a, "image/jpeg")


# ---------------------------------------------------------------------------
# video decode: connector-shaped fetch, pool submission, async resolution
# ---------------------------------------------------------------------------
def test_decode_video_forwards_file_uri_and_io_kwargs(monkeypatch, tmp_path):
    from eyas.backends import prefetch as prefetch_mod

    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"clip")
    seen = []

    utils = ModuleType("vllm.multimodal.utils")
    utils.fetch_video = lambda url, video_io_kwargs=None: (
        seen.append((url, video_io_kwargs)) or SimpleNamespace(media=("frames", {"fps": 2}))
    )
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.multimodal", ModuleType("vllm.multimodal"))
    monkeypatch.setitem(sys.modules, "vllm.multimodal.utils", utils)

    loaded = prefetch_mod.decode_video(str(clip), {"num_frames": 8})
    assert (loaded.path, loaded.media) == (str(clip), ("frames", {"fps": 2}))
    assert seen == [(clip.resolve().as_uri(), {"num_frames": 8})]

    remote = prefetch_mod.decode_video("https://example.com/a.mp4")
    assert remote.path == "https://example.com/a.mp4"
    assert seen[-1] == ("https://example.com/a.mp4", None)


class _FakePool:
    def __init__(self):
        self.submissions = []

    def submit(self, fn, *args):
        self.submissions.append((fn, args))
        fut = Future()
        fut.set_result(LoadedVideo(path=args[0], media=("frames",)))
        return fut


def test_materialize_decodes_videos_once_per_path(tmp_path):
    clip = str(tmp_path / "a.mp4")
    loader = MediaPrefetchLoader(workers=2, decode_videos=True, video_io_kwargs={"num_frames": 8})
    pool = _FakePool()
    loader._pool = pool
    video_req = ChatRequest(messages=(user_message(VideoPart(clip)),))
    out = loader.materialize([video_req, video_req])
    assert out[0].messages[0].content[0].video == LoadedVideo(path=clip, media=("frames",))
    assert out[1].messages[0].content[0].video == out[0].messages[0].content[0].video
    assert len(pool.submissions) == 1
    assert pool.submissions[0][1] == (clip, {"num_frames": 8})


def test_materialize_leaves_videos_alone_without_decode(tmp_path):
    clip = str(tmp_path / "a.mp4")
    loader = MediaPrefetchLoader(workers=2)
    out = loader.materialize([ChatRequest(messages=(user_message(VideoPart(clip)),))])
    assert out[0].messages[0].content[0].video == clip
    assert loader._pool is None  # no submissions -> no process pool spawned


def test_resolve_video_async_passthrough_and_failure():
    loader = MediaPrefetchLoader(workers=1, decode_videos=True)
    payload = LoadedVideo(path="a.mp4", media=("frames",))
    assert asyncio.run(loader.resolve_video_async(payload, {})) is payload
    assert asyncio.run(loader.resolve_video_async("https://e/a.mp4", {})) == "https://e/a.mp4"
    fut = Future()
    fut.set_exception(FileNotFoundError("missing clip"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(loader.resolve_video_async("/nope.mp4", {("video", "/nope.mp4"): fut}))


def test_resolve_releases_payload_after_the_last_consumer(loader, tmp_path: Path):
    a = _write_png(tmp_path / "a.png")
    b = _write_png(tmp_path / "b.png")
    futures = loader.begin([_req(a), _req(a), _req(b)])
    assert futures.remaining[("image", a)] == 2
    asyncio.run(loader.resolve_image_async(a, futures))
    assert ("image", a) in futures  # the second consumer still needs this payload
    asyncio.run(loader.resolve_image_async(a, futures))
    assert ("image", a) not in futures  # last consumer harvested it -> released
    asyncio.run(loader.resolve_image_async(b, futures))
    assert futures == {}  # nothing pinned past the batch


def test_openai_rejects_prefetched_video_payload():
    from eyas.backends.openai_api import OpenAIBackend

    backend = OpenAIBackend(base_url="http://server.test/v1", model="m")
    message = user_message(VideoPart(LoadedVideo("clip.mp4", ("frames", {}))))
    with pytest.raises(ValueError, match="does not decode videos"):
        backend._to_openai_messages((message,))


def test_openai_agenerate_with_prefetch_sends_data_url(tmp_path: Path):
    import httpx

    from eyas.backends.openai_api import OpenAIBackend

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_RESPONSE_BODY)

    a = _write_png(tmp_path / "a.png")
    backend = OpenAIBackend(
        base_url="http://server.test/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
        prefetch_workers=2,
    )
    responses = asyncio.run(backend.agenerate([_req(a)], SamplingConfig()))

    assert responses[0].text == "ok"
    content = bodies[0]["messages"][0]["content"]
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == Path(a).read_bytes()
