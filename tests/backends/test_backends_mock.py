"""Scripted mock backend fast tests: plain-text request rendering (role|parts,
text verbatim, media as descriptors)."""

from __future__ import annotations

from eyas.backends.base import ChatRequest, VideoPart, user_message
from eyas.backends.mock import ScriptedMockBackend


def test_mock_render_text_describes_video():
    text = ScriptedMockBackend.render_request_text(
        ChatRequest(messages=(user_message(VideoPart("a.mp4"), "q"),))
    )
    assert text == "user|<video:a.mp4>q"
