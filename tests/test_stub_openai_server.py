"""Contract tests for the stdlib stub OpenAI server the docs tutorials run against:
response shape, cross-instance determinism, localhost-only binding, 404 on unknown paths."""

from __future__ import annotations

import socket

import httpx
import pytest

from tests.support.stub_openai_server import StubOpenAIServer, stub_answer

MESSAGES = [
    {"role": "system", "content": "Answer the multiple-choice question."},
    {
        "role": "user",
        "content": "Which color is the sky?\nA) green\nB) blue\nC) red\n",
    },
]


@pytest.fixture
def server():
    srv = StubOpenAIServer()
    base = srv.start()
    yield srv, base
    srv.stop()


def _post(base: str, messages: list[dict]) -> dict:
    resp = httpx.post(f"{base}/chat/completions", json={"messages": messages}, timeout=10)
    assert resp.status_code == 200
    return resp.json()


def test_chat_completion_shape_and_determinism(server):
    srv, base = server
    body = _post(base, MESSAGES)
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert set(usage) >= {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert all(isinstance(usage[k], int) for k in usage)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert srv.request_count == 1

    again = _post(base, MESSAGES)
    assert again["choices"][0]["message"]["content"] == content
    assert srv.request_count == 2


def test_same_request_same_answer_across_servers():
    assert stub_answer(MESSAGES) == stub_answer(MESSAGES)
    srv1, srv2 = StubOpenAIServer(), StubOpenAIServer()
    base1, base2 = srv1.start(), srv2.start()
    try:
        c1 = _post(base1, MESSAGES)["choices"][0]["message"]["content"]
        c2 = _post(base2, MESSAGES)["choices"][0]["message"]["content"]
        assert c1 == c2 == stub_answer(MESSAGES)
    finally:
        srv1.stop()
        srv2.stop()


def test_server_serves_only_on_localhost(server):
    srv, base = server
    assert base.startswith("http://127.0.0.1:")
    # a non-loopback local address must refuse the connection
    local_ip = socket.gethostbyname(socket.gethostname())
    if local_ip != "127.0.0.1":
        port = srv.port
        with pytest.raises(OSError):
            with socket.create_connection((local_ip, port), timeout=2):
                pass


def test_unknown_path_is_404(server):
    _, base = server
    resp = httpx.get(f"{base}/nope", timeout=10)
    assert resp.status_code == 404


def test_non_dict_body_and_bad_content_length_are_400(server):
    _, base = server
    # a JSON body that parses to a list (not a dict) must 400, not raise in the handler
    resp = httpx.post(f"{base}/chat/completions", json=[1, 2], timeout=10)
    assert resp.status_code == 400
    # a non-numeric Content-Length header must 400 as well (raw socket: httpx forbids
    # overriding the header itself, so we frame the request by hand)
    host, port = (
        base[len("http://") :].split("/")[0].split(":")[0],
        int(base.rsplit(":", 1)[1].split("/")[0]),
    )
    raw = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Length: not-a-number\r\n"
        b"\r\n"
    )
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(raw)
        status_line = sock.makefile("rb").readline()
    assert b"400" in status_line
