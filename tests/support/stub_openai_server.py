"""Pure-stdlib stub of the OpenAI chat-completions endpoint.

The tutorials in ``docs/`` are executed verbatim by CI against this stub, so
answers must be deterministic across requests, server instances and processes
(no ``hash()``, no set-iteration order — only ``hashlib.sha256`` over
canonical JSON). Serves ``POST /v1/chat/completions`` on an ephemeral port
bound to 127.0.0.1 only; everything else is 404.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["StubOpenAIServer", "stub_answer"]

_OPTION_RE = re.compile(r"^\s*([A-Z])[.)]", re.MULTILINE)


def _flat_text(messages: list[dict]) -> str:
    """Concatenate the plain-text content of every message (multimodal
    parts' text chunks included) for option extraction."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def stub_answer(messages: list[dict]) -> str:
    """Deterministic reply for a chat-completions ``messages`` payload.

    Option letters (``A)`` / ``B.`` line starts) are extracted from the request
    text, sorted for a canonical order, and one is picked by the first byte of
    ``sha256`` over the canonical JSON of the messages — same messages always
    produce the same answer, even across processes. No options found → "red".
    """
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    options = sorted(set(_OPTION_RE.findall(_flat_text(messages))))
    if not options:
        return "red"
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return f"Answer: {options[digest[0] % len(options)]}"


class StubOpenAIServer:
    """A local OpenAI-compatible endpoint: ``start()`` returns the base URL
    (``http://127.0.0.1:<port>/v1``), ``stop()`` shuts it down."""

    def __init__(self) -> None:
        self.request_count = 0
        self.port = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # silence access log
                pass

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 — http.server naming contract
                self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):  # noqa: N802 — http.server naming contract
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    self._json(400, {"error": {"message": "bad Content-Length"}})
                    return
                raw = self.rfile.read(length)
                if self.path != "/v1/chat/completions":
                    self._json(404, {"error": {"message": "not found"}})
                    return
                try:
                    payload = json.loads(raw)
                    messages = payload["messages"]
                    assert isinstance(messages, list)
                except (ValueError, KeyError, AssertionError, TypeError):
                    # TypeError: the body parsed to a non-dict JSON value (list/str/int/None)
                    self._json(400, {"error": {"message": "bad chat payload"}})
                    return
                content = stub_answer(messages)
                canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
                prompt_tokens = len(canonical) // 4
                completion_tokens = max(1, len(content) // 4)
                outer.request_count += 1
                self._json(
                    200,
                    {
                        "id": "stub",
                        "object": "chat.completion",
                        "model": "stub",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    },
                )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
