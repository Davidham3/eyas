"""Shared pytest fixtures for the test suite.

Plain helpers live in ``tests/support/engine.py``; this file keeps fixtures
only — pytest collects conftest per-directory, so anything non-fixture here
would be imported by every test session regardless of topic.
"""

from __future__ import annotations

import pytest


# Engine backend class for the sampling-contract pins: the module imports without
# the extra (all engine imports are deferred), so exposing the class here is safe
# for every test session.
@pytest.fixture
def vllm_cls() -> type:
    from eyas.backends.vllm import VllmBackend

    return VllmBackend
