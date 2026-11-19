"""Structural contract for the two new quickstart boards (GSM8K text math QA,
NExT-QA video MCQ).

Each board ships the same 8 method cells as the image/text boards, so the cell
set, the task shape per board, the judge policy, the OPRO instruction position
and the seeded-pool conventions are pinned here rather than re-derived from the
spec (``tests/configs/test_configs.py`` covers binding for every cell).
"""

import pytest

from eyas.tasks.math_qa import MathQaTask
from eyas.tasks.video_mcq import VideoMcqTask
from tests.support.spec_cells import cell

_TASK = {"math_qa": MathQaTask, "video_mcq": VideoMcqTask}
BOARDS = {
    "gsm8k": (
        "math_qa",
        ("raw", "cot", "hill_climb", "evoprompt", "opro", "ape", "protegi", "spo"),
    ),
    "nextqa": (
        "video_mcq",
        ("raw", "cot", "hill_climb", "evoprompt", "opro", "ape", "protegi", "spo"),
    ),
}


@pytest.mark.parametrize(
    "dataset,shape,method", [(d, s, m) for d, (s, ms) in BOARDS.items() for m in ms]
)
def test_board_cell_binds(dataset, shape, method):
    cfg = cell(dataset, method)
    assert cfg["task"]["name"] == shape
    if dataset == "gsm8k":
        assert "judge" not in cfg
    if method == "opro" and dataset == "nextqa":
        assert cfg["task"]["params"]["instruction_pos"] == "Q_begin"
    if method == "spo":
        pool = cfg["strategy"]["seeds"]
        assert (
            all(("{question}" in s) for s in pool)
            if dataset == "gsm8k"
            else all("{video}" in s and "{choices}" in s for s in pool)
        )
        if dataset == "gsm8k":
            assert cfg["strategy"]["params"]["prompt_includes_choices"] is False
    if method == "evoprompt":
        assert len(cfg["strategy"]["seeds"]) == 10 and len(set(cfg["strategy"]["seeds"])) == 10
        assert cfg["strategy"]["seeds"][0] == _TASK[shape].default_prompt
