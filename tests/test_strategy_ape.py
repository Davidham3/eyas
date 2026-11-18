"""APE few-shot demo media: the task supplies the demo media (hook), not the strategy.

Image tasks keep the historical bytes (``Input: [Image k] <question>`` + one
image part per demo, in demo order); text tasks render plain
``Input: <question>`` lines with no media marker; video demos carry no clip
(the task overrides the hook to return no parts).
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random
from typing import Any

import pytest

from eyas.backends.base import ChatResponse, ImagePart, TextPart
from eyas.backends.mock import ScriptedMockBackend
from eyas.engine.evaluation import AbsoluteEvalModel
from eyas.engine.strategies.ape import ApeConfig, ApeStrategy, stratified_demo_groups
from eyas.engine.strategy import build_run_context
from eyas.tasks.base import Sample
from eyas.tasks.math_qa import MathQaTask
from eyas.tasks.mcq import McqTask
from eyas.tasks.mixed_mcq import MixedMcqTask
from eyas.tasks.protocol import IdentityProtocol


def _image_samples() -> list[Sample]:
    return [
        Sample(
            id=f"i{k}",
            inputs={"image": f"i{k}.png", "choices": ["cat", "dog"]},
            references={"label": "cat" if k % 2 else "dog"},
        )
        for k in range(4)
    ]


def _text_samples() -> list[Sample]:
    return [
        Sample(id=f"t{k}", inputs={"question": f"{k}+1?"}, references={"answer": str(k + 1)})
        for k in range(4)
    ]


def _ape_with_task(task) -> ApeStrategy:
    config = ApeConfig(num_candidates=2, demos_per_prompt=2)
    backend = ScriptedMockBackend(responder=lambda req, sampling, idx: ChatResponse(text="[x]"))
    ctx = build_run_context(
        ApeStrategy,
        config,
        task,
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=0,
    )
    return ApeStrategy(config, ctx)


def test_demo_lines_carry_image_markers_for_image_tasks():
    samples = _image_samples()
    strategy = _ape_with_task(McqTask(samples=samples))
    lines = strategy._demo_lines(strategy.ctx.task.default_prompt, samples)
    assert lines[0].startswith("Input: [Image 1] ")
    assert lines[1].startswith("Input: [Image 2] ")


def test_demo_lines_have_no_media_marker_for_text_tasks():
    samples = _text_samples()
    strategy = _ape_with_task(MathQaTask(samples=samples))
    lines = strategy._demo_lines(strategy.ctx.task.default_prompt, samples)
    assert lines[0].startswith("Input: ")
    assert "Question: " in lines[0]
    assert "[Image" not in "\n".join(lines)


def test_gen_request_attaches_one_image_per_demo_before_the_text_block():
    samples = _image_samples()
    strategy = _ape_with_task(McqTask(samples=samples))
    req = strategy._gen_request(strategy.ctx.task.default_prompt, samples[:2])
    assert [type(p) for p in req.messages[1].content] == [ImagePart, ImagePart, TextPart]


def test_gen_request_attaches_no_media_for_text_tasks():
    samples = _text_samples()
    strategy = _ape_with_task(MathQaTask(samples=samples))
    req = strategy._gen_request(strategy.ctx.task.default_prompt, samples[:2])
    assert [type(p) for p in req.messages[1].content] == [TextPart]


def _mixed_samples() -> list[Sample]:
    return [
        Sample(
            id="m_text",
            inputs={
                "passage": "P0.",
                "images": [],
                "videos": [],
                "question": "q0?",
                "choices": ["a", "b"],
                "combo": "text",
            },
            references={"label": "a"},
        ),
        Sample(
            id="m_image",
            inputs={
                "passage": "",
                "images": ["m_image.png"],
                "videos": [],
                "question": "q1?",
                "choices": ["a", "b"],
                "combo": "image",
            },
            references={"label": "b"},
        ),
        Sample(
            id="m_multi",
            inputs={
                "passage": "P2.",
                "images": ["x.png", "y.png"],
                "videos": ["v.mp4"],
                "question": "q2?",
                "choices": ["a", "b"],
                "combo": "text_image_video",
            },
            references={"label": "a"},
        ),
    ]


def test_mixed_demo_lines_number_only_media_bearing_demos():
    samples = _mixed_samples()
    strategy = _ape_with_task(MixedMcqTask(samples=samples))
    lines = strategy._demo_lines(strategy.ctx.task.default_prompt, samples[:2])
    assert lines[0].startswith("Input: ")
    assert "[Image" not in lines[0]
    assert lines[1].startswith("Input: [Image 1] ")


def test_mixed_gen_request_matches_each_demo_to_one_image():
    samples = _mixed_samples()
    strategy = _ape_with_task(MixedMcqTask(samples=samples))
    req = strategy._gen_request(strategy.ctx.task.default_prompt, samples)
    parts = req.messages[1].content
    # one image per media-bearing demo, in demo order (the text-only demo
    # contributes none); the multi-image demo ships only its first image --
    # its line addresses a single [Image 1] and APE caps images at the demo
    # count -- and clips stay out of demos entirely
    assert [p for p in parts if not isinstance(p, TextPart)] == [
        ImagePart("m_image.png"),
        ImagePart("x.png"),
    ]
    assert isinstance(parts[-1], TextPart)


def test_ape_rejects_nonpositive_demos_per_prompt_at_construction() -> None:
    """demos_per_prompt < 1 can never carry a demo (and would declare a
    zero-image resource profile) — the guard fires at strategy construction,
    before any request is built."""
    strategy = _ape_with_task(McqTask(samples=_image_samples()))
    for bad in (0, -2):
        with pytest.raises(ValueError, match="demos_per_prompt"):
            ApeStrategy(ApeConfig(num_candidates=2, demos_per_prompt=bad), strategy.ctx)


def test_demo_lines_render_the_label_reference_not_the_first_value() -> None:
    """KIE-style references store ``entities`` BEFORE ``label``: the demo
    answer must come from the shared ``render_sample_reference`` convention
    (label wins), never the positional first value (the entities list)."""
    strategy = _ape_with_task(McqTask(samples=_image_samples()))
    sample = Sample(
        id="k1",
        inputs={"image": "k1.png", "choices": ["a", "b"]},
        references={"entities": ["a", "b"], "label": "gold"},
    )
    lines = strategy._demo_lines(strategy.ctx.task.default_prompt, [sample])
    assert lines[0].endswith("Output: gold")
    assert "Output: a, b" not in lines[0]


def _labeled_samples(n_labels: int, per_label: int) -> list[Sample]:
    return [
        Sample(
            id=f"s{lab}_{k}",
            inputs={"question": f"q{lab}_{k}?"},
            references={"label": f"L{lab}"},
        )
        for lab in range(n_labels)
        for k in range(per_label)
    ]


class _BucketCollapsingRng(Random):
    """Adversarial deterministic rng for coverage-check placement probes.

    ``shuffle`` is the identity; ``choice`` collapses every single-sample label
    bucket onto ``stratified_pick`` while the full-sample backfill pool draws
    ``backfill_pick``. With two classes the stratified picks alone miss one
    class — only the backfilled (returned) group can restore coverage.
    """

    def __init__(self, stratified_pick: Sample, backfill_pick: Sample) -> None:
        super().__init__(0)
        self._stratified_pick = stratified_pick
        self._backfill_pick = backfill_pick

    def shuffle(self, x: list[Any], random: Any = None) -> None:
        """Identity shuffle — bucket order stays stable."""

    def choice(self, seq: Sequence[Any]) -> Any:
        return self._stratified_pick if len(seq) == 1 else self._backfill_pick


def test_stratified_groups_subset_regime_rotates_distinct_labels() -> None:
    """per_group < n_labels (the shipped 2-demo budget vs CUB/Pets label counts):
    each returned group carries exactly per_group demos with DISTINCT labels —
    a per-group-random subset of classes (the label order is reshuffled per
    group) — and the covered-label union rotates beyond per_group across
    groups. No coverage error: full coverage per prompt is not claimed in this
    regime."""
    samples = _labeled_samples(n_labels=5, per_label=3)
    groups = stratified_demo_groups(samples, groups=8, per_group=2, rng=Random(7))
    assert len(groups) == 8
    covered: set[Any] = set()
    for group in groups:
        assert len(group) == 2
        labels = [s.references["label"] for s in group]
        assert len(set(labels)) == 2  # distinct classes within the group
        covered.update(labels)
    assert len(covered) > 2  # rotation across groups, not one fixed pair


def test_stratified_groups_full_regime_covers_every_class() -> None:
    """per_group >= n_labels: every RETURNED group (post backfill + truncation)
    covers every class — both at exactly n_labels and with backfill above it."""
    samples = _labeled_samples(n_labels=3, per_label=3)
    for per_group in (3, 5):
        groups = stratified_demo_groups(samples, groups=4, per_group=per_group, rng=Random(11))
        assert len(groups) == 4
        for group in groups:
            assert len(group) == per_group
            assert {s.references["label"] for s in group} == {"L0", "L1", "L2"}


def test_coverage_check_inspects_the_returned_demos_after_backfill() -> None:
    """The coverage assurance must run on the RETURNED demos — after backfill
    and truncation. Here the stratified picks alone miss class 'b' (the rng
    collapses both label buckets onto the class-'a' sample) while the backfill
    pool restores it: the pre-backfill intermediate is uncovered but the
    returned group is covered, so this must build groups, not raise."""
    s_a = Sample(id="a", inputs={"question": "qa?"}, references={"label": "a"})
    s_b = Sample(id="b", inputs={"question": "qb?"}, references={"label": "b"})
    rng = _BucketCollapsingRng(stratified_pick=s_a, backfill_pick=s_b)
    groups = stratified_demo_groups([s_a, s_b], groups=1, per_group=4, rng=rng)
    assert len(groups) == 1
    assert len(groups[0]) == 4
    assert {s.references["label"] for s in groups[0]} == {"a", "b"}


def test_coverage_check_still_fires_when_returned_demos_miss_a_class() -> None:
    """Fail-fast preserved: when the RETURNED demos genuinely miss a class and
    full coverage is achievable (per_group >= n_labels), the RuntimeError still
    fires — computed on the truncated group itself."""
    s_a = Sample(id="a", inputs={"question": "qa?"}, references={"label": "a"})
    s_b = Sample(id="b", inputs={"question": "qb?"}, references={"label": "b"})
    rng = _BucketCollapsingRng(stratified_pick=s_a, backfill_pick=s_a)
    with pytest.raises(RuntimeError, match="do not cover classes"):
        stratified_demo_groups([s_a, s_b], groups=1, per_group=4, rng=rng)
