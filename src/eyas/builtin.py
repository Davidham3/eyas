"""Unified registration entry point for built-in components.

import eyas.builtin completes registration of all built-in components; third-party
plugins register via the entry_points group ``eyas.plugins`` and need no changes here.
"""

from __future__ import annotations

# Backends (vllm engine imports are deferred inside launch_vllm_engine, so the core
# package can register even without the extra installed)
from eyas.backends import mock, openai_api, vllm

# Judges
from eyas.judges import exact_match, mcq_judge

# Tasks and protocols
from eyas.tasks import demo_shapes, kie, math_qa, mcq, mixed_mcq, protocol, video_mcq
from eyas.tasks import vqa as vqa_task
