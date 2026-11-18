"""ape: one-shot forward induction — stratified demo sampling (the algorithm itself) +
LLM generation + clean/dedup + evaluate the whole pool, take top1.

Paper: Zhou et al., "Large Language Models Are Human-Level Prompt Engineers"
(arXiv:2211.01910, ICLR 2023).

The proposal distribution is the paper's forward mode (§3.1): the model INDUCES the
instruction from input→output demonstrations — the author repo's forward template
("I gave a friend an instruction ... The instruction was to [APE]", demos
``Input: [INPUT]\nOutput: [OUTPUT]``), not a blind prompt-engineering ask.

Ape has three structural pieces a stage skeleton cannot express: stratified demo
sampling is part of the algorithm, the "candidate cleaning" step between propose
and evaluate, and demo media sourced from the task (the ``Task.demo_media_parts``
hook — image tasks attach one image per demo, text tasks none, video tasks none
because a per-candidate request with N clips multiplies decode cost and trips
engine media limits; many media parts per request are routed through the planner
resource chokepoint).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, ClassVar

from eyas.backends.base import (
    ChatRequest,
    TextPart,
    system_message,
    user_message,
    user_text,
)
from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec, ResourceProfile
from eyas.engine.domain import OptimizerState, Prompt, rank_by_score
from eyas.engine.genio import (
    GEN_MARKER,
    OUT_END,
    OUT_START,
    clean_instruction,
    render_sample_reference,
)
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Decision, EvalView, RunContext, Strategy
from eyas.tasks.base import Variant

#: Forward-induction framing — the author repo's forward template (README memo +
#: experiments/run_instruction_induction.py): "I gave a friend a(n) instruction. Based
#: on the instruction they produced the following input-output pairs: [full_DEMO] The
#: instruction was to [APE]". The [APE] slot — the instruction itself — is what the
#: model completes (the OUT tags below carry that directive).
INDUCTION_INTRO = (
    "I gave a friend an instruction. Based on the instruction they produced "
    "the following input-output pairs:"
)
INDUCTION_SUFFIX = "The instruction was to"


@dataclass
class ApeConfig:
    num_candidates: int = 4
    demos_per_prompt: int = 2


@dataclass
class ApeState(OptimizerState):
    candidates: list[Prompt] = field(default_factory=list)
    round_idx: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"candidates": [p.text for p in self.candidates], "round_idx": self.round_idx}

    @classmethod
    def from_json(cls, data: Any) -> ApeState:
        return cls(
            candidates=[Prompt(t) for t in data.get("candidates", [])],
            round_idx=int(data.get("round_idx", 0)),
        )


def stratified_demo_groups(
    samples: Sequence[Any], groups: int, per_group: int, rng: Random
) -> list[list[Any]]:
    """One draw per class (label order shuffled per group) + backfill with
    replacement to ``per_group``, then truncate — coverage is enforced on the
    RETURNED demos, in two regimes:

    - ``per_group >= n_labels``: every returned group covers every class; the
      check runs after backfill/truncation and fails fast (RuntimeError) on any
      uncovered class.
    - ``per_group < n_labels``: each group covers a per-group-random subset of
      classes (the reshuffled label order rotates coverage across groups);
      full coverage per prompt is not claimed and no coverage error is raised.
    """
    by_label: dict[Any, list[Any]] = defaultdict(list)
    for s in samples:
        by_label[s.references.get("label", "?")].append(s)
    if not by_label:
        raise ValueError("ape stratified sampling: training set is empty")
    built: list[list[Any]] = []
    for _ in range(groups):
        labels = list(by_label)
        rng.shuffle(labels)
        chosen = [rng.choice(by_label[lab]) for lab in labels]
        while len(chosen) < per_group:
            chosen.append(rng.choice(list(samples)))
        group = chosen[:per_group]
        if per_group >= len(by_label):
            # the coverage read must carry the SAME default as the bucketing above --
            # without it, label-free samples (open-ended vqa) compute {None} vs {"?"}
            # and the invariant "covers every class" can never hold
            missing = {lab for lab in by_label} - {s.references.get("label", "?") for s in group}
            if missing:
                raise RuntimeError(f"ape demos do not cover classes: {missing}")
        built.append(group)
    return built


@register_strategy("ape")
class ApeStrategy(Strategy):
    config_cls: ClassVar[type] = ApeConfig
    state_cls: ClassVar[type] = ApeState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="ape",
        requires_eval=frozenset({ABSOLUTE_EVAL}),
        resource_profile=ResourceProfile(max_images=ApeConfig.demos_per_prompt),
        produces_artifacts=frozenset({"ape.candidate_prompts"}),
    )

    @classmethod
    def capabilities_for(cls, config: Any) -> CapSpec:
        # the demo images per generation request ARE config-driven; the class
        # CapSpec freezes the constructor default and would mis-enforce
        return replace(
            cls.capabilities, resource_profile=ResourceProfile(max_images=config.demos_per_prompt)
        )

    def __init__(self, config: Any, ctx: RunContext) -> None:
        # demos_per_prompt < 1 can never carry a demo (and would declare a
        # zero-image resource profile) — fail fast at construction, before
        # any request is built.
        if config.demos_per_prompt < 1:
            raise ValueError(
                f"ape demos_per_prompt must be >= 1, got {config.demos_per_prompt}; "
                "set strategy.params.demos_per_prompt to a positive demo count"
            )
        super().__init__(config, ctx)

    def _demo_lines(self, initial: str, demos: Sequence[Any]) -> list[str]:
        """One induction demo per sample as an input→output pair: the question is
        extracted with the same ``user_text`` path the evaluation renders (opro's
        exemplar ruling — never a reverse rebuild), the completion is the sample's
        ground truth under the shared reference-key convention
        (``render_sample_reference`` — label wins, never the positional first
        value: KIE stores its entities list before the label). Demos come from
        the selection split (``ctx.eval_split``) — the paper's "we sample five
        input-output pairs from the training data".

        The task's ``demo_media_parts`` hook decides the pairing: when a demo
        carries media its line leads with ``[Image k]`` (k counts only demos with
        media, in demo order — it addresses the k-th media part of the request);
        text-only demos render plain ``Input: <question>``."""
        lines: list[str] = []
        media_no = 0
        for sample in demos:
            question = user_text(self.ctx.task.build_request(initial, sample, Variant("identity")))
            answer = render_sample_reference(sample.references)
            if self.ctx.task.demo_media_parts(sample):
                media_no += 1
                lines.append(f"Input: [Image {media_no}] {question}\nOutput: {answer}")
            else:
                lines.append(f"Input: {question}\nOutput: {answer}")
        return lines

    def _gen_request(self, initial: str, demos: Sequence[Any]) -> ChatRequest:
        media: list[Any] = []
        for sample in demos:
            media.extend(self.ctx.task.demo_media_parts(sample))
        demos_block = "\n\n".join(self._demo_lines(initial, demos))
        parts: list[Any] = [
            *media,
            TextPart(f"{INDUCTION_INTRO}\n\n{demos_block}\n\n{INDUCTION_SUFFIX}"),
        ]
        system = (
            f"{GEN_MARKER} Complete the instruction after the cue above. "
            f"Wrap the instruction between {OUT_START} and {OUT_END}."
        )
        return ChatRequest(messages=(system_message(system), user_message(*parts)))

    def expected_rounds(self) -> int | None:
        return 1  # one evaluate round over the pool, then finish

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        cfg = self.config
        rng = self.derived_rng()
        samples = list(self.ctx.task.load_split(self.ctx.eval_split))
        initial = self.ctx.task.default_prompt
        groups = stratified_demo_groups(samples, cfg.num_candidates, cfg.demos_per_prompt, rng)
        raws = self.generate([self._gen_request(initial, g) for g in groups], "generate")
        # APE's published candidate set includes the original instruction; in this
        # framework it is also the monotonicity floor -- on label-free tasks the
        # generation channel can echo dataset text (questions/annotations) instead
        # of instructions, and without the floor selection would crown the best
        # garbage candidate.
        candidates: list[Prompt] = [Prompt(initial)]
        seen: set[str] = {candidates[0].id}
        for raw in raws:
            instruction = clean_instruction(raw)
            if not instruction:
                continue
            child = Prompt(instruction, metadata={"label": "ape"})
            if child.id not in seen:
                seen.add(child.id)
                candidates.append(child)
        self.emit_artifact("candidate_prompts", [c.text for c in candidates])
        return ApeState(candidates=candidates)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, ApeState)
        unevaluated = [p for p in state.candidates if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))
        ranked = rank_by_score(state.candidates, view.score)
        return Decision("finish", best_id=ranked[0].id)
