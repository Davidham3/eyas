"""OpenAI backend fast tests: payload construction, image data URLs,
sampling field mapping, retry semantics. All via httpx.MockTransport —
zero GPU, zero network."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eyas.backends.base import (
    OVER_WINDOW_FINISH,
    ChatRequest,
    ImagePart,
    SamplingConfig,
    TextPart,
    VideoPart,
    resolve_max_inflight,
    user_message,
)
from eyas.backends.openai_api import BackendTransientError, OpenAIBackend

_BASE_URL = "http://server.test/v1"
_CHAT_PATH = "/v1/chat/completions"

_RESPONSE_BODY = {
    "id": "cmpl-1",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def _make_backend(handler: Any, *, retry_backoff: float = 0.0, **kwargs: Any) -> OpenAIBackend:
    return OpenAIBackend(
        base_url=_BASE_URL,
        model="test-model",
        transport=httpx.MockTransport(handler),
        retry_backoff=retry_backoff,
        **kwargs,
    )


def _ok_handler(hits: list[httpx.Request], bodies: list[dict[str, Any]] | None = None) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        if bodies is not None:
            bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_RESPONSE_BODY)

    return handler


# ---------------------------------------------------------------------------
# prepare: payload construction
# ---------------------------------------------------------------------------
def test_prepare_text_only_content_is_plain_string():
    backend = _make_backend(None)
    payload = backend.prepare(
        [ChatRequest(messages=(user_message("hello"),), request_id="r1")], SamplingConfig()
    )[0]
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_prepare_preserves_part_order_and_encodes_local_image(tmp_path: Path):
    image = tmp_path / "img.jpg"
    image.write_bytes(b"fake-jpg-bytes")
    expected_url = "data:image/jpeg;base64," + base64.b64encode(b"fake-jpg-bytes").decode("ascii")
    backend = _make_backend(None)
    payload = backend.prepare(
        [
            ChatRequest(
                messages=(user_message(ImagePart(image=str(image)), TextPart("what is this?")),)
            )
        ],
        SamplingConfig(),
    )[0]
    content = payload["messages"][0]["content"]
    assert content == [
        {"type": "image_url", "image_url": {"url": expected_url}},
        {"type": "text", "text": "what is this?"},
    ]


def test_prepare_remote_and_data_urls_pass_through():
    backend = _make_backend(None)
    request = ChatRequest(
        messages=(
            user_message(
                ImagePart(image="https://example.com/a.jpg"),
                ImagePart(image="data:image/png;base64,AAAA"),
            ),
        )
    )
    content = backend.prepare([request], SamplingConfig())[0]["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == "https://example.com/a.jpg"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_prepare_file_url_image_reads_the_local_file(tmp_path: Path):
    """file:// is a local-file URL (the prefetch layer passes it through as a
    URL): the openai path strips the scheme and reads the file into a data URL
    exactly like a bare path — the server cannot fetch a client-local file://."""
    image = tmp_path / "img.png"
    image.write_bytes(b"fake-png-bytes")
    expected_url = "data:image/png;base64," + base64.b64encode(b"fake-png-bytes").decode("ascii")
    backend = _make_backend(None)
    payload = backend.prepare(
        [ChatRequest(messages=(user_message(ImagePart(image=f"file://{image}"), "q"),))],
        SamplingConfig(),
    )[0]
    assert payload["messages"][0]["content"][0]["image_url"]["url"] == expected_url


def test_agenerate_file_url_image_end_to_end(tmp_path: Path):
    image = tmp_path / "img.jpg"
    image.write_bytes(b"fake-jpg-bytes")
    bodies: list[dict[str, Any]] = []
    backend = _make_backend(_ok_handler([], bodies))
    req = ChatRequest(messages=(user_message(ImagePart(image=f"file://{image}"), "q"),))
    responses = asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert responses[0].text == "hi"
    url = bodies[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url == "data:image/jpeg;base64," + base64.b64encode(b"fake-jpg-bytes").decode("ascii")


def test_prepare_maps_video_to_video_url(tmp_path: Path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"fake-mp4")
    backend = OpenAIBackend(model="m", base_url="http://x")
    payload = backend.prepare(
        [ChatRequest(messages=(user_message(VideoPart(str(clip)), "q"),))], SamplingConfig()
    )
    part = payload[0]["messages"][0]["content"][0]
    assert part["type"] == "video_url"
    assert part["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_prepare_sampling_fields():
    backend = _make_backend(None)
    default_payload = backend.prepare(
        [ChatRequest(messages=(user_message("q"),))], SamplingConfig()
    )[0]
    assert default_payload["temperature"] == 0.0
    assert default_payload["max_tokens"] == 1024
    # extension fields defaulting to None are not sent (strict OpenAI endpoints work as-is)
    for absent in ("top_p", "top_k", "repetition_penalty", "seed", "chat_template_kwargs"):
        assert absent not in default_payload

    explicit = backend.prepare(
        [ChatRequest(messages=(user_message("q"),))],
        SamplingConfig(
            temperature=0.7,
            max_new_tokens=128,
            top_p=0.9,
            top_k=5,
            repetition_penalty=1.1,
            thinking=True,
            seed=42,
        ),
    )[0]
    assert explicit["top_p"] == 0.9
    assert explicit["top_k"] == 5
    assert explicit["repetition_penalty"] == 1.1
    assert explicit["seed"] == 42
    assert explicit["chat_template_kwargs"] == {"enable_thinking": True}


def test_prepare_thinking_false_is_explicit_disable():
    """thinking=False explicitly sends enable_thinking:false; only None (the default) omits it."""
    backend = _make_backend(None)
    payload = backend.prepare(
        [ChatRequest(messages=(user_message("q"),))], SamplingConfig(thinking=False)
    )[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_prepare_extra_body_merged_last_overrides_sampling():
    """extra_body is shallow-merged onto the payload last: same-name keys override sampling fields (escape hatch semantics)."""
    backend = _make_backend(None, extra_body={"top_k": 999, "custom_field": {"a": 1}})
    payload = backend.prepare(
        [ChatRequest(messages=(user_message("q"),))], SamplingConfig(top_k=5)
    )[0]
    assert payload["top_k"] == 999
    assert payload["custom_field"] == {"a": 1}


def test_api_key_env_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EYAS_TEST_INF_KEY", "sk-env-resolved")
    backend = OpenAIBackend(base_url=_BASE_URL, model="m", api_key_env="EYAS_TEST_INF_KEY")
    assert backend.api_key == "sk-env-resolved"


def test_api_key_env_missing_fails_fast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EYAS_TEST_INF_KEY", raising=False)
    with pytest.raises(ValueError, match="EYAS_TEST_INF_KEY"):
        OpenAIBackend(base_url=_BASE_URL, model="m", api_key_env="EYAS_TEST_INF_KEY")


def test_context_window_param_arms_the_declaration():
    """`backend.params.context_window` is the "telling" mechanism base.py's
    context_window doc promises for remote endpoints; it feeds the RequestPlanner
    guard (PlannedBackend reads inner.context_window)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_RESPONSE_BODY)

    assert _make_backend(handler).context_window is None  # un-told: guard stays disarmed
    told = _make_backend(handler, context_window=32_768)
    assert told.context_window == 32_768


def test_api_key_and_api_key_env_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot both be set"):
        OpenAIBackend(base_url=_BASE_URL, model="m", api_key="k", api_key_env="EYAS_TEST_INF_KEY")


def test_prepare_multi_turn_roles():
    from eyas.backends.base import assistant_message, system_message

    backend = _make_backend(None)
    request = ChatRequest(
        messages=(
            system_message("be brief"),
            user_message("hi"),
            assistant_message("Let's think step by step."),
        )
    )
    messages = backend.prepare([request], SamplingConfig())[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[2]["content"] == "Let's think step by step."


def test_prepare_rejects_raw_prompt_escape_hatch():
    backend = _make_backend(None)
    request = ChatRequest(messages=(user_message("hi"),), raw_prompt="<|im_start|>custom")
    with pytest.raises(ValueError, match="raw_prompt"):
        backend.prepare([request], SamplingConfig())


# ---------------------------------------------------------------------------
# agenerate: full pipeline through MockTransport
# ---------------------------------------------------------------------------
def test_agenerate_echoes_request_id_and_usage():
    hits: list[httpx.Request] = []
    bodies: list[dict[str, Any]] = []
    backend = _make_backend(_ok_handler(hits, bodies))
    responses = asyncio.run(
        backend.agenerate(
            [ChatRequest(messages=(user_message("q"),), request_id="r7")], SamplingConfig()
        )
    )
    assert len(responses) == 1
    assert responses[0].text == "hi"
    assert responses[0].request_id == "r7"
    assert responses[0].usage == _RESPONSE_BODY["usage"]
    assert hits[0].url.path == _CHAT_PATH
    assert hits[0].headers["Authorization"] == "Bearer EMPTY"
    assert bodies[0]["messages"] == [{"role": "user", "content": "q"}]


def test_agenerate_extracts_finish_reason_and_reasoning():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ans",
                            "reasoning_content": "think",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    backend = _make_backend(handler)
    (resp,) = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert resp.text == "ans"
    assert resp.reasoning == "think"
    assert resp.finish_reason == "stop"


def test_agenerate_valid_reasoning_string_passes_validation():
    """Pin: a str reasoning survives the 200-body shape validation untouched and
    lands in ChatResponse.reasoning; a None finish_reason is allowed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ans", "reasoning": "r3"},
                        "finish_reason": None,
                    }
                ]
            },
        )

    backend = _make_backend(handler)
    (resp,) = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert resp.reasoning == "r3"
    assert resp.finish_reason is None


def test_agenerate_reasoning_fallback_key_and_truncation():
    """reasoning (OpenAI-style) is the fallback key for reasoning_content; finish_reason=length passes through as-is."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ans", "reasoning": "r2"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    backend = _make_backend(handler)
    (resp,) = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert resp.reasoning == "r2"
    assert resp.finish_reason == "length"


def test_agenerate_empty_requests_no_http_call():
    backend = _make_backend(None)
    assert asyncio.run(backend.agenerate([], SamplingConfig())) == []


def test_agenerate_preserves_order_under_concurrency():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    backend = _make_backend(handler)
    requests = [
        ChatRequest(messages=(user_message(f"q{i}"),), request_id=f"r{i}") for i in range(8)
    ]
    responses = asyncio.run(backend.agenerate(requests, SamplingConfig()))
    assert [r.request_id for r in responses] == [f"r{i}" for i in range(8)]
    # echo-style check: each response pairs one-to-one with its request, proving gather keeps order
    assert [r.text for r in responses] == [f"q{i}" for i in range(8)]


def test_agenerate_image_encoding_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Local image encoding (file read + base64) runs synchronously inside
    prepare, so it must run on a thread pool: while encoding blocks, the event
    loop must stay schedulable (the ticker keeps counting), otherwise encoding
    a large image would freeze the other coroutines on the same loop."""
    import time

    (tmp_path / "big.jpg").write_bytes(b"\xff" * 1024)
    ticks = {"n": 0}

    def slow_encode(value: str, default_mime: str) -> str:
        time.sleep(0.3)  # simulate the base64 encoding cost of a large image
        return "data:image/jpeg;base64,QUJD"

    monkeypatch.setattr(OpenAIBackend, "_media_url", staticmethod(slow_encode))
    backend = _make_backend(_ok_handler([]))

    async def _ticker() -> None:
        while True:
            await asyncio.sleep(0.02)
            ticks["n"] += 1

    async def main() -> None:
        ticker = asyncio.create_task(_ticker())
        await backend.agenerate(
            [ChatRequest(messages=(user_message(ImagePart(image=str(tmp_path / "big.jpg"))),))],
            SamplingConfig(),
        )
        ticker.cancel()

    asyncio.run(main())
    # encoding takes 0.3s: with encoding on a thread pool the ticker accumulates >10 ticks; a wedged loop allows at most 1 catch-up
    assert ticks["n"] >= 5


# ---------------------------------------------------------------------------
# retry semantics
# ---------------------------------------------------------------------------
def test_agenerate_retries_then_succeeds():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        if len(hits) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=_RESPONSE_BODY)

    backend = _make_backend(handler, max_retries=2)
    responses = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert responses[0].text == "hi"
    assert len(hits) == 2


def test_agenerate_non_retryable_status_fails_fast():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(400, text="bad request body")

    backend = _make_backend(handler, max_retries=3)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        asyncio.run(
            backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
        )
    assert len(hits) == 1  # 4xx is not retried


def test_agenerate_retries_exhausted():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(500, text="boom")

    # batch-level recovery disabled: this test only checks that request-ladder exhaustion surfaces BackendTransientError
    backend = _make_backend(handler, max_retries=2, batch_retries=0, batch_retry_backoff=0.0)
    with pytest.raises(BackendTransientError, match="still failing after 2 retries"):
        asyncio.run(
            backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
        )
    assert len(hits) == 3  # first attempt + 2 retries


def test_agenerate_transport_error_retried(monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json=_RESPONSE_BODY)

    backend = _make_backend(flaky, max_retries=1)
    responses = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert responses[0].text == "hi"


# ---------------------------------------------------------------------------
# malformed 200 body: a broken completion payload is a server-side transient —
# it enters the retry ladder (and batch-level recovery), never crashes the run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_body",
    [
        httpx.Response(200, text="<html>gateway junk</html>"),  # not JSON at all
        httpx.Response(200, json={"id": "cmpl-1"}),  # choices missing
        httpx.Response(200, json={"choices": []}),  # choices empty
        httpx.Response(200, json={"choices": ["not an object"]}),  # entry not a dict
        httpx.Response(200, json=[1, 2, 3]),  # top level not an object
        # deeper shapes _one consumes: message must be an object, content /
        # reasoning_content / reasoning strings, finish_reason a string (or
        # None), usage an object — otherwise extraction leaks AttributeError
        # or a non-str text/reasoning/usage outside the retry classification
        httpx.Response(200, json={"choices": [{"message": "junk"}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": ["a", "b"]}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "junk"}}], "usage": [1, 2]}),
        httpx.Response(
            200, json={"choices": [{"message": {"content": "hi", "reasoning": ["junk"]}}]}
        ),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi", "reasoning_content": {"k": "v"}}}]},
        ),
        httpx.Response(200, json={"choices": [{"message": {"content": "hi"}, "finish_reason": 7}]}),
    ],
    ids=[
        "not-json",
        "no-choices",
        "empty-choices",
        "choice-not-object",
        "top-level-list",
        "message-not-object",
        "content-not-string",
        "usage-not-object",
        "reasoning-not-string",
        "reasoning-content-not-string",
        "finish-reason-not-string",
    ],
)
def test_agenerate_malformed_200_retried_then_succeeds(bad_body: httpx.Response):
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        if len(hits) == 1:
            return bad_body
        return httpx.Response(200, json=_RESPONSE_BODY)

    backend = _make_backend(handler, max_retries=2)
    responses = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert responses[0].text == "hi"
    assert len(hits) == 2  # the malformed body burned one ladder rung, then recovered


@pytest.mark.parametrize(
    "bad_body",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": [{"message": "junk"}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": ["a", "b"]}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "junk"}}], "usage": [1, 2]}),
        httpx.Response(
            200, json={"choices": [{"message": {"content": "hi", "reasoning": ["junk"]}}]}
        ),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi", "reasoning_content": {"k": "v"}}}]},
        ),
        httpx.Response(200, json={"choices": [{"message": {"content": "hi"}, "finish_reason": 7}]}),
    ],
    ids=[
        "not-json",
        "message-not-object",
        "content-not-string",
        "usage-not-object",
        "reasoning-not-string",
        "reasoning-content-not-string",
        "finish-reason-not-string",
    ],
)
def test_agenerate_malformed_200_exhaustion_raises_transient(bad_body: httpx.Response):
    """Ladder exhaustion on malformed bodies surfaces BackendTransientError (the
    batch-level recovery signal) — never a raw JSONDecodeError/KeyError, and
    never an AttributeError leaking out of the extraction."""
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return bad_body

    backend = _make_backend(handler, max_retries=1, batch_retries=0, batch_retry_backoff=0.0)
    with pytest.raises(BackendTransientError, match="still failing after 1 retries"):
        asyncio.run(
            backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
        )
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# over-window guard: detect -> thinking-off retry -> transient skip (sentinel response)
# ---------------------------------------------------------------------------
_OVER_WINDOW_BODY = (
    '{"error":{"message":"Engine internal server error!: ... You passed 32753 input tokens '
    "and requested 16 output tokens. However, the model's context length is only 32768 "
    "tokens, resulting in a maximum input length of 32752 tokens. Please reduce the length "
    'of the input prompt. (parameter=input_tokens, value=32753)","type":"EngineInternalError",'
    '"code":"514"}}'
)

# A vision request whose input alone exceeded the window: the engine clamps max_tokens negative
_OVER_WINDOW_VISION_BODY = (
    '{"error":{"message":"Engine internal server error!: Internal server error, error_msg '
    "is max_tokens must be at least 1, got -6221. (parameter=max_tokens, value=-6221), "
    'error_cause is None","type":"EngineInternalError","code":"514"}}'
)


def test_agenerate_over_window_vision_clamp_shape_is_guarded():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(514, text=_OVER_WINDOW_VISION_BODY)

    backend = _make_backend(handler)
    responses = asyncio.run(
        backend.agenerate(
            [ChatRequest(messages=(user_message("q"),), request_id="r1")],
            SamplingConfig(thinking=False),
        )
    )
    assert (
        len(hits) == 1
    )  # negative max_tokens clamp shape also counts as over-window: thinking off, skip directly
    assert responses[0].text == ""
    assert responses[0].finish_reason == OVER_WINDOW_FINISH


def test_agenerate_max_tokens_zero_misconfig_not_over_window():
    """max_tokens=0 misconfiguration (got 0, not negative) is not over-window — a persistent config error is not retried, staying fail fast."""
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(
            514,
            text='{"error":{"message":"max_tokens must be at least 1, got 0 '
            '(parameter=max_tokens, value=0)","type":"EngineInternalError","code":"514"}}',
        )

    backend = _make_backend(handler, max_retries=3)
    with pytest.raises(RuntimeError, match="HTTP 514"):
        asyncio.run(
            backend.agenerate(
                [ChatRequest(messages=(user_message("q"),))], SamplingConfig(thinking=False)
            )
        )
    assert len(hits) == 1  # retrying a config error is pointless: raise on the first hit


def test_agenerate_engine_514_generic_error_retried():
    """Engine 514 generic error (no over-window marker) is a server-side
    transient — it enters the retry ladder.

    A single "Internal server error, error_cause is None" under server overload
    jitter must not kill the whole batch: without retries the batch raises and
    the job dies.
    """
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        if len(hits) == 1:
            return httpx.Response(
                514,
                text='{"error":{"message":"Engine internal server error!: Internal server '
                'error, error_msg is Internal server error, error_cause is None",'
                '"type":"EngineInternalError","code":"514"}}',
            )
        return httpx.Response(200, json=_RESPONSE_BODY)

    backend = _make_backend(handler, max_retries=2)
    responses = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert responses[0].text == "hi"
    assert len(hits) == 2


def test_agenerate_engine_514_generic_error_exhausted_raises():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(
            514,
            text='{"error":{"message":"Engine internal server error!: Internal server '
            'error, error_msg is Internal server error, error_cause is None",'
            '"type":"EngineInternalError","code":"514"}}',
        )

    backend = _make_backend(handler, max_retries=2, batch_retries=0, batch_retry_backoff=0.0)
    with pytest.raises(BackendTransientError, match="still failing after 2 retries"):
        asyncio.run(
            backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
        )
    assert len(hits) == 3


def test_agenerate_over_window_skips_without_retry_when_thinking_off():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(514, text=_OVER_WINDOW_BODY)

    backend = _make_backend(handler)
    responses = asyncio.run(
        backend.agenerate(
            [ChatRequest(messages=(user_message("q"),), request_id="r1")],
            SamplingConfig(thinking=False),
        )
    )
    assert len(hits) == 1  # thinking explicitly off: no budget to yield, no retry
    assert responses[0].text == ""
    assert responses[0].request_id == "r1"
    assert responses[0].finish_reason == OVER_WINDOW_FINISH


def test_agenerate_over_window_retry_rescues():
    hits: list[httpx.Request] = []
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        bodies.append(json.loads(request.content))
        if len(hits) == 1:
            return httpx.Response(514, text=_OVER_WINDOW_BODY)
        return httpx.Response(200, json=_RESPONSE_BODY)

    backend = _make_backend(handler)
    responses = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert len(hits) == 2
    assert "chat_template_kwargs" not in bodies[0]
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert responses[0].text == "hi"
    assert responses[0].finish_reason == "stop"


def test_agenerate_over_window_retry_exhausted_returns_sentinel():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(514, text=_OVER_WINDOW_BODY)

    backend = _make_backend(handler)
    responses = asyncio.run(
        backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
    )
    assert len(hits) == 2  # first attempt + one thinking-off retry
    assert responses[0].finish_reason == OVER_WINDOW_FINISH


def test_agenerate_over_window_mixed_batch_survives():
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][0]["content"]
        if content == "huge":
            return httpx.Response(514, text=_OVER_WINDOW_BODY)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    backend = _make_backend(handler)
    responses = asyncio.run(
        backend.agenerate(
            [
                ChatRequest(messages=(user_message("ok1"),), request_id="r0"),
                ChatRequest(messages=(user_message("huge"),), request_id="r1"),
                ChatRequest(messages=(user_message("ok2"),), request_id="r2"),
            ],
            SamplingConfig(thinking=False),
        )
    )
    assert [r.request_id for r in responses] == ["r0", "r1", "r2"]
    assert [r.text for r in responses] == ["ok1", "", "ok2"]
    assert responses[1].finish_reason == OVER_WINDOW_FINISH
    assert responses[0].finish_reason is None


def test_agenerate_non_window_error_keeps_fail_fast():
    hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(400, text="invalid sampling parameter")

    backend = _make_backend(handler)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        asyncio.run(
            backend.agenerate([ChatRequest(messages=(user_message("q"),))], SamplingConfig())
        )
    assert len(hits) == 1


def test_agenerate_batch_retry_rescues_transient_ladder_exhaustion():
    """Batch-level recovery: after request-ladder exhaustion kills the
    batch, a batch-level backoff rerun rescues it — the direct countermeasure
    for long optimization rounds (compare/protegi) whose transport errors
    burn through the 8-run ladder. max_retries=0 makes the first
    transport error burn through the request ladder."""

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ConnectError("connection reset by peer")
        return httpx.Response(200, json=_RESPONSE_BODY)

    backend = _make_backend(handler, max_retries=0, batch_retries=1, batch_retry_backoff=0.0)
    responses = asyncio.run(
        backend.agenerate(
            [
                ChatRequest(messages=(user_message("q1"),), request_id="r1"),
                ChatRequest(messages=(user_message("q2"),), request_id="r2"),
            ],
            SamplingConfig(thinking=False),
        )
    )
    assert [r.text for r in responses] == ["hi", "hi"]
    assert calls["n"] == 4  # first batch: 2 failures + batch rerun: 2 successes


def test_agenerate_batch_retry_exhausted_raises_transient_error():
    """The batch rerun dies too: BackendTransientError (a RuntimeError
    subclass) propagates; total requests = batch attempts x batch size x
    request-ladder attempts."""

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    backend = _make_backend(handler, max_retries=0, batch_retries=2, batch_retry_backoff=0.0)
    with pytest.raises(BackendTransientError, match="still failing after 0 retries"):
        asyncio.run(
            backend.agenerate(
                [ChatRequest(messages=(user_message("q"),), request_id="r1")],
                SamplingConfig(thinking=False),
            )
        )
    assert calls["n"] == 3  # first batch 1 + two batch-level reruns


def test_agenerate_persistent_error_not_batch_retried():
    """Persistent failures (non-retryable codes) do not trigger a batch rerun — stay fail fast, no idle spinning."""

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="invalid sampling parameter")

    backend = _make_backend(handler, max_retries=0, batch_retries=2, batch_retry_backoff=0.0)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        asyncio.run(
            backend.agenerate(
                [ChatRequest(messages=(user_message("q"),), request_id="r1")],
                SamplingConfig(thinking=False),
            )
        )
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# media failures: per-request sentinel, batch survives (Y15)
# ---------------------------------------------------------------------------
@pytest.fixture
def eyas_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    import logging

    eyas_root = logging.getLogger("eyas")
    old_propagate, old_level = eyas_root.propagate, eyas_root.level
    eyas_root.propagate = True
    caplog.set_level(logging.DEBUG, logger="eyas")
    yield caplog
    eyas_root.propagate = old_propagate
    eyas_root.setLevel(old_level)


def _png_bytes() -> bytes:
    """Minimal 2x2 RGB PNG built with stdlib zlib+struct (decodable by the
    prefetch pool worker without any test image asset)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes((255, 0, 0)) * 2
    idat = zlib.compress(row * 2)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _image_batch(tmp_path: Path) -> list[ChatRequest]:
    """Batch of 3: two existing local images, the middle path missing."""
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(_png_bytes())
    return [
        ChatRequest(
            messages=(user_message(ImagePart(image=str(tmp_path / "a.png")), "q0"),),
            request_id="r0",
        ),
        ChatRequest(
            messages=(user_message(ImagePart(image=str(tmp_path / "missing.png")), "q1"),),
            request_id="r1",
        ),
        ChatRequest(
            messages=(user_message(ImagePart(image=str(tmp_path / "b.png")), "q2"),),
            request_id="r2",
        ),
    ]


@pytest.mark.parametrize("prefetch_workers", [0, 2])
def test_agenerate_missing_image_degrades_to_sentinel_batch_survives(
    tmp_path: Path, prefetch_workers: int, eyas_logs: pytest.LogCaptureFixture
):
    """Brief Step 1: batch of 3, middle image path missing -> 2 successes + 1
    error-carrying sentinel response, no exception — with and without the
    prefetch pool (both media paths degrade per request)."""
    hits: list[httpx.Request] = []
    backend = _make_backend(
        _ok_handler(hits), prefetch_workers=prefetch_workers, batch_retry_backoff=0.0
    )
    responses = asyncio.run(backend.agenerate(_image_batch(tmp_path), SamplingConfig()))
    assert [r.request_id for r in responses] == ["r0", "r1", "r2"]
    assert [r.text for r in responses] == ["hi", "", "hi"]
    assert responses[1].finish_reason == OVER_WINDOW_FINISH
    assert responses[0].finish_reason == "stop" and responses[2].finish_reason == "stop"
    assert len(hits) == 2  # the failed request never reached the server
    assert "[media] load failed request=r1" in eyas_logs.text
    assert "[media] 1/3 requests failed media load" in eyas_logs.text
    # separate accounting: a media failure is NOT an over-window skip
    assert "[over_window]" not in eyas_logs.text


def test_agenerate_corrupt_image_with_prefetch_degrades(
    tmp_path: Path, eyas_logs: pytest.LogCaptureFixture
):
    """Undecodable image: the pool worker's decode error degrades per request
    (the no-prefetch path sends raw bytes and leaves decoding to the server)."""
    (tmp_path / "broken.png").write_bytes(b"not-an-image")
    hits: list[httpx.Request] = []
    backend = _make_backend(_ok_handler(hits), prefetch_workers=2)
    req = ChatRequest(
        messages=(user_message(ImagePart(image=str(tmp_path / "broken.png")), "q"),),
        request_id="r0",
    )
    responses = asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert responses[0].text == "" and responses[0].finish_reason == OVER_WINDOW_FINISH
    assert hits == []
    assert "[media] 1/1 requests failed media load" in eyas_logs.text


def test_agenerate_media_failure_never_enters_over_window_ladder(tmp_path: Path):
    """thinking=None (ladder armed): a missing file must not trigger the
    thinking-off retry — no extra server call, no rescue accounting."""
    hits: list[httpx.Request] = []
    bodies: list[dict[str, Any]] = []
    backend = _make_backend(_ok_handler(hits, bodies), prefetch_workers=2)
    responses = asyncio.run(backend.agenerate(_image_batch(tmp_path), SamplingConfig()))
    assert len(hits) == 2
    assert all("chat_template_kwargs" not in body for body in bodies)
    assert responses[1].finish_reason == OVER_WINDOW_FINISH


def test_agenerate_raw_prompt_config_error_still_raises():
    """Config-level errors keep fail-fast even though media failures degrade."""
    backend = _make_backend(_ok_handler([]))
    req = ChatRequest(messages=(user_message("hi"),), raw_prompt="<|im_start|>custom")
    with pytest.raises(ValueError, match="raw_prompt"):
        asyncio.run(backend.agenerate([req], SamplingConfig()))


# ---------------------------------------------------------------------------
# off-loop materialize + lazy per-request payloads (Y16, Y20)
# ---------------------------------------------------------------------------
def test_agenerate_materialize_off_loop_keeps_ticker_alive() -> None:
    """Y16: media materialization (the prefetch harvest) is a blocking op and
    must run on a thread pool — while it blocks, a concurrent asyncio ticker
    must keep counting (the loop is not frozen). A stub loader blocks its
    materialize on a threading.Event until the test releases it."""
    import threading

    entered = threading.Event()
    release = threading.Event()

    class _BlockingLoader:
        def materialize_partial(self, requests: Any) -> Any:
            entered.set()
            assert release.wait(timeout=5), "test never released the materialize block"
            return list(requests), {}

    backend = _make_backend(_ok_handler([]), prefetch_workers=2)
    backend._loader = _BlockingLoader()  # type: ignore[assignment]
    ticks = {"n": 0}

    async def _ticker() -> None:
        while True:
            await asyncio.sleep(0.02)
            ticks["n"] += 1

    async def main() -> None:
        ticker = asyncio.create_task(_ticker())
        task = asyncio.create_task(
            backend.agenerate(
                [ChatRequest(messages=(user_message("q"),), request_id="r0")], SamplingConfig()
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        # materialize is now blocked in its worker thread; the loop must stay live
        await asyncio.sleep(0.2)
        release.set()
        await task
        ticker.cancel()

    asyncio.run(main())
    assert ticks["n"] >= 5  # a frozen loop allows at most 1 catch-up tick


def test_agenerate_payloads_built_per_request_not_whole_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Y20: with concurrency 1 and 3 requests, payload construction (base64
    encoding) for request N must happen only AFTER request N-1 completed — no
    whole-batch pre-encoding. The event log interleaves build/done per request;
    a whole-batch prepare would emit all three builds before any done."""
    events: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][0]["content"]
        events.append(("done", content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    backend = _make_backend(handler, max_inflight=1)
    orig_prepare = backend.prepare

    def spy_prepare(requests: Any, sampling: Any) -> Any:
        for req in requests:
            events.append(("build", req.messages[0].content[0].text))
        return orig_prepare(requests, sampling)

    monkeypatch.setattr(backend, "prepare", spy_prepare)
    reqs = [ChatRequest(messages=(user_message(f"q{i}"),), request_id=f"r{i}") for i in range(3)]
    asyncio.run(backend.agenerate(reqs, SamplingConfig()))
    assert events == [
        ("build", "q0"),
        ("done", "q0"),
        ("build", "q1"),
        ("done", "q1"),
        ("build", "q2"),
        ("done", "q2"),
    ]


def test_agenerate_batch_rerun_rereads_consumed_media(tmp_path: Path) -> None:
    """Consume-and-drop + batch re-run: after a request's payload is built its
    raw prefetched media is released, so a batch-level re-run cannot reuse it —
    it re-reads from the original path (files still exist; prefetch is an
    optimization). Both attempts must carry a freshly encoded data URL."""
    (tmp_path / "a.jpg").write_bytes(b"img-bytes")
    bodies: list[dict[str, Any]] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection reset by peer")
        return httpx.Response(200, json=_RESPONSE_BODY)

    backend = _make_backend(handler, max_retries=0, batch_retries=1, batch_retry_backoff=0.0)
    req = ChatRequest(
        messages=(user_message(ImagePart(image=str(tmp_path / "a.jpg")), "q"),), request_id="r0"
    )
    responses = asyncio.run(backend.agenerate([req], SamplingConfig()))
    assert responses[0].text == "hi"
    assert calls["n"] == 2  # first attempt failed, batch re-run succeeded
    assert len(bodies) == 2
    for body in bodies:
        url = body["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# close: the prefetch process pool must be released at run end (Y18)
# ---------------------------------------------------------------------------
def test_close_shuts_down_prefetch_pool(tmp_path: Path):
    """After close() the loader is gone and its resident executor is shut down
    (worker processes released) — mirrors VllmBackend.close's loader shutdown."""
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", (2, 2)).save(tmp_path / "a.png")
    backend = OpenAIBackend(base_url=_BASE_URL, model="m", prefetch_workers=2)
    assert backend._loader is not None
    req = ChatRequest(messages=(user_message(ImagePart(image=str(tmp_path / "a.png")), "q"),))
    backend._loader.materialize([req])  # spawns the resident pool
    pool = backend._loader._pool
    assert pool is not None
    backend.close()
    assert backend._loader is None
    assert pool._shutdown_thread  # executor shut down, workers released
    backend.close()  # idempotent: a second close is a no-op


def test_close_without_prefetch_is_noop():
    backend = OpenAIBackend(base_url=_BASE_URL, model="m")
    assert backend._loader is None
    backend.close()
    assert backend._loader is None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_openai_backend_registered():
    import eyas.builtin  # noqa: F401
    from eyas.registry import get

    assert get("backend", "openai") is OpenAIBackend


# ---------------------------------------------------------------------------
# EYAS_MAX_INFLIGHT env var override
# ---------------------------------------------------------------------------
class TestResolveMaxInflight:
    def test_unset_returns_configured(self, monkeypatch):
        monkeypatch.delenv("EYAS_MAX_INFLIGHT", raising=False)
        assert resolve_max_inflight(7) == 7

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("EYAS_MAX_INFLIGHT", "32")
        assert resolve_max_inflight(7) == 32

    def test_invalid_values_fail_fast(self, monkeypatch):
        for raw in ("0", "-3", "abc"):
            monkeypatch.setenv("EYAS_MAX_INFLIGHT", raw)
            with pytest.raises(RuntimeError, match="EYAS_MAX_INFLIGHT"):
                resolve_max_inflight(7)

    def test_empty_string_fails_fast(self, monkeypatch):
        """A set-but-empty export is a common slip — it must fail fast on the
        same path as other invalid values, never silently fall back to config."""
        monkeypatch.setenv("EYAS_MAX_INFLIGHT", "")
        with pytest.raises(RuntimeError, match="EYAS_MAX_INFLIGHT"):
            resolve_max_inflight(7)

    def test_backend_constructor_honors_env(self, monkeypatch):
        monkeypatch.setenv("EYAS_MAX_INFLIGHT", "24")
        backend = OpenAIBackend(base_url=_BASE_URL, model="test-model")
        assert backend.max_inflight == 24
