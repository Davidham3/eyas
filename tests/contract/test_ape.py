"""Contract tests (APE): candidate generation, induction floor, forward-induction
fingerprints (APE, ICLR 2023, arXiv:2211.01910), stratified demos.
"""

from __future__ import annotations

from random import Random

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.domain import (
    Prompt,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
)
from eyas.engine.genio import (
    GEN_MARKER,
    OUT_END,
    OUT_START,
)
from eyas.engine.strategies.ape import ApeConfig, ApeStrategy
from eyas.engine.strategy import build_run_context
from eyas.tasks.base import Sample
from eyas.tasks.protocol import IdentityProtocol
from tests.contract.helpers import _LABELS, ToyMCQ, _gen_mock, _run


def test_ape_generates_clean_candidate():
    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        ApeStrategy,
        ApeConfig(num_candidates=3, demos_per_prompt=2),
        ToyMCQ(),
        eval_model,
        ["seed"],
        backend=_gen_mock(),
    )
    assert result.best is not None
    assert (
        result.best.scores["accuracy"] == 1.0
    )  # the cleaned generated candidate contains step by step


def test_ape_floor_keeps_default_when_generation_echoes_dataset_text():
    """Label-free guard: on a label-free task the generation channel may echo
    dataset text (questions/annotations) instead of instructions. APE's published
    candidate set includes the original instruction; that is also the monotonicity
    floor -- selection must never end up worse than raw."""
    garbage = "what is the name of this magazine?"

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if GEN_MARKER in sys_text:
            return ChatResponse(text=f"{OUT_START}{garbage}{OUT_END}")
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if garbage not in sys_text else "")

    eval_model = AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))
    result = _run(
        ApeStrategy,
        ApeConfig(num_candidates=3, demos_per_prompt=2),
        ToyMCQ(),
        eval_model,
        ["seed"],
        backend=ScriptedMockBackend(responder=responder),
    )
    assert result.best is not None
    assert result.best.prompt_id == Prompt(ToyMCQ.default_prompt).id  # floor wins, not echo
    assert result.best.scores["accuracy"] == 1.0


# --- APE forward-induction fingerprints ---
#
# Paper (APE, ICLR 2023, arXiv:2211.01910 §3.1) forward-mode + author repo memo: the model
# induces the instruction from INPUT→OUTPUT demonstrations -- "I gave a friend an
# instruction. Based on the instruction they produced the following input-output
# pairs: [full_DEMO] The instruction was to [APE]", demos "Input: [INPUT]\nOutput:
# [OUTPUT]". Pinned at the RECORDED full request texts (the demo block carries
# Input/Output pairs and the meta-prompt the induction framing).


def _ape_ctx(config: ApeConfig, backend: ScriptedMockBackend, seed: int = 0):
    ctx = build_run_context(
        ApeStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        backend,
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
        seed=seed,
    )
    return ctx, ApeStrategy(config, ctx)


def test_ape_induction_conditions_on_input_output_pairs():
    """Every induction request's demo block pairs each ``Input:`` line with a
    nonempty ``Output:`` line (the paper's demos_template), and the completions
    are the selection-split gold answers (the toy split: cat/dog)."""
    backend = _gen_mock()
    ctx, strategy = _ape_ctx(ApeConfig(num_candidates=3, demos_per_prompt=2), backend)
    state = strategy.init([Prompt("seed")])
    assert state.candidates
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert len(gen) == 3  # one induction request per demo group
    for text in gen:
        lines = text.splitlines()
        input_idx = [i for i, line in enumerate(lines) if line.startswith("Input: ")]
        assert len(input_idx) == 2, "demo block must carry one Input line per demo"
        for i in input_idx:
            assert lines[i + 1].startswith("Output: ")
            assert lines[i + 1][len("Output: ") :].strip(), "gold completion must be nonempty"
        assert "Output: cat" in text and "Output: dog" in text
        # the question text rides the same user_text extraction path the
        # evaluation renders (opro's exemplar ruling), never a raw-input rebuild
        assert "q1" in text and "q2" in text
        # ToyMCQ renders text only, so the base Task hook attaches no media: demo
        # lines are plain "Input: <question>" (the [Image k] form is pinned in
        # tests/strategies/test_strategy_ape.py against a real McqTask).
        assert "[Image" not in text


def test_ape_induction_framing_is_the_paper_template():
    """The framing constants are the author-repo forward template verbatim: the
    induction ask ("I gave a friend an instruction...") before the demos, the
    completion cue ("The instruction was to") after them, and the [APE] slot is
    completed by the model itself."""
    from eyas.engine.strategies import ape as ape_mod

    assert (
        ape_mod.INDUCTION_INTRO
        == "I gave a friend an instruction. Based on the instruction they produced "
        "the following input-output pairs:"
    )
    assert ape_mod.INDUCTION_SUFFIX == "The instruction was to"
    backend = _gen_mock()
    ctx, strategy = _ape_ctx(ApeConfig(num_candidates=3, demos_per_prompt=2), backend)
    strategy.init([Prompt("seed")])
    gen = [t for t, _ in backend.recorded_requests if GEN_MARKER in t]
    assert gen
    for text in gen:
        assert ape_mod.INDUCTION_INTRO in text
        assert ape_mod.INDUCTION_SUFFIX in text
        assert text.index(ape_mod.INDUCTION_INTRO) < text.index("Input: ")
        assert text.index("Output: ") < text.index(ape_mod.INDUCTION_SUFFIX)


def test_ape_demo_renders_list_reference_as_text():
    """Fidelity doc ape D6: a list-valued reference (KIE-style
    multi-value answers) renders its demo ``Output:`` line as natural text
    (``Output: a, b``) — the Python repr (``['a', 'b']``) must never enter
    the induction demos as pseudo-syntax."""
    backend = _gen_mock()
    ctx, strategy = _ape_ctx(ApeConfig(num_candidates=1, demos_per_prompt=1), backend)
    sample = Sample(
        id="k1",
        inputs={"question": "extract the fields", "image": "k1.png"},
        references={"fields": ["a", "b"]},
    )
    lines = strategy._demo_lines(ToyMCQ.default_prompt, [sample])
    # ToyMCQ attaches no demo media, so the line carries no [Image k] marker
    assert lines == ["Input: extract the fields\nOutput: a, b"]
    assert "['a', 'b']" not in lines[0]


def test_ape_stratified_demos_labelless_single_bucket() -> None:
    """Label-free vqa families: the coverage check must use the
    ``"?"`` default that bucketing uses, not bare ``references.get("label")`` —
    label-free samples (open-ended vqa families) made the chosen-label set
    ``{None}`` vs bucket set ``{"?"}`` -- a guaranteed RuntimeError. Label-free
    samples must form one degenerate bucket and build demo groups."""

    from eyas.engine.strategies.ape import stratified_demo_groups

    samples = [
        Sample(
            id=f"q{i}",
            inputs={"image": f"i{i}.jpg", "question": f"what {i}?"},
            references={"answers": [f"a{i}"]},
        )
        for i in range(6)
    ]
    groups = stratified_demo_groups(samples, 4, 2, Random(0))
    assert len(groups) == 4 and all(len(g) == 2 for g in groups)
