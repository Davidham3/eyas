"""Shared fixtures for the contract suite (tests/contract/test_*.py).

The pure-toy task/backend pair the whole suite assembles strategies against;
also the compare-only method whose assembly-time rejection is itself a contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    system_message,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.core.objective import Objective
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.capability import COMPARE_EVAL, CapSpec
from eyas.engine.domain import (
    OptimizerState,
    Prompt,
    PromptListState,
    rank_by_score,
)
from eyas.engine.evaluation import (
    AbsoluteEvalModel,
)
from eyas.engine.genio import (
    GEN_MARKER,
    OUT_END,
    OUT_START,
)
from eyas.engine.strategies.cot import CotConfig, CotStrategy
from eyas.engine.strategy import Decision, EvalView, OptimizerLoop, Strategy, build_run_context
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.protocol import IdentityProtocol

_LABELS = {"q1": "cat", "q2": "dog"}


def _toy_samples() -> list[Sample]:
    return [
        Sample(id="q1", inputs={"question": "q1", "image": "q1.png"}, references={"label": "cat"}),
        Sample(id="q2", inputs={"question": "q2", "image": "q2.png"}, references={"label": "dog"}),
    ]


class ToyMCQ(Task):
    name = "toy_mcq"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return _toy_samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


class ToyCompare(Task):
    name = "toy_compare"
    objectives = (Objective(key="compare_score"),)
    default_prompt = "Compare."

    def load_split(self, split: str) -> Sequence[Sample]:
        return _toy_samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(messages=(user_message(sample.inputs["question"]),))

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"compare_score": 1.0}


def _mock() -> ScriptedMockBackend:
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        question = user_text(request)
        sys_msgs = [m for m in request.messages if m.role == "system"]
        prompt_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        label = _LABELS.get(question, "")
        answer = label if "step by step" in prompt_text.lower() else "wrong"
        return ChatResponse(text=answer)

    return ScriptedMockBackend(responder=responder)


def _gen_mock() -> ScriptedMockBackend:
    """Classification scores correct on step-by-step; generation requests carrying GEN_MARKER return an instruction containing step by step.

    The instruction is bracketed — the official OPRO output format — so the
    ``single_bracket`` yield channel produces a candidate; the OUT tags stay for the
    line/tag parsers of the other generative methods."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        sys_msgs = [m for m in request.messages if m.role == "system"]
        sys_text = "".join(p.text for p in sys_msgs[0].content) if sys_msgs else ""
        if GEN_MARKER in sys_text:
            return ChatResponse(
                text=f"{OUT_START}[Classify the animal carefully and think step by step.]{OUT_END}"
            )
        label = _LABELS.get(user_text(request), "")
        return ChatResponse(text=label if "step by step" in sys_text.lower() else "wrong")

    return ScriptedMockBackend(responder=responder)


def _run(strategy_cls, config, task, eval_model, seeds, backend=None, **kw):
    ctx = build_run_context(
        strategy_cls,
        config,
        task,
        IdentityProtocol(),
        backend or _mock(),
        eval_model,
        **kw,
    )
    return OptimizerLoop(ctx).run(strategy_cls(config, ctx), list(seeds))


@dataclass
class CompareOnlyConfig:
    """No parameter fields: candidate prompts are always injected via ``seeds``."""


class CompareOnlyStrategy(Strategy):
    """Synthetic compare-only fixture (test-local name ``_test_compare_only``).

    Declares ONLY compare_eval semantics and produces no absolute metric: mounted
    on an accuracy board it must fail at assembly, mounted on a compare task it
    runs one evaluate round over the seed pool and finishes with the top-ranked
    candidate. The assembly-time guarantee is a framework property and must not
    key on any single method, so the probe lives here."""

    config_cls: ClassVar[type] = CompareOnlyConfig
    state_cls: ClassVar[type] = PromptListState
    capabilities: ClassVar[CapSpec] = CapSpec(
        name="_test_compare_only", requires_eval=frozenset({COMPARE_EVAL})
    )

    def expected_rounds(self) -> int | None:
        return 1  # one evaluate round over the pool, then finish

    def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
        return PromptListState(candidates=list(seeds), round_idx=0)

    def step(self, state: OptimizerState, view: EvalView) -> Decision:
        assert isinstance(state, PromptListState)
        unevaluated = [p for p in state.candidates if not view.evaluated(p)]
        if unevaluated:
            return Decision("evaluate", prompts=tuple(unevaluated))
        ranked = rank_by_score(state.candidates, view.score)
        return Decision("finish", best_id=ranked[0].id)


def _cot_strategy(config: CotConfig) -> CotStrategy:
    ctx = build_run_context(
        CotStrategy,
        config,
        ToyMCQ(),
        IdentityProtocol(),
        _mock(),
        AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"})),
    )
    return CotStrategy(config, ctx)


def _capture_eval_trials(snapshots: list[dict]):
    def on_round(snapshot: dict) -> None:
        snapshots.append(snapshot)
        state = snapshot["state"]
        if state.get("phase") == "eval_trials" and state.get("trials"):
            raise KeyboardInterrupt("synthetic crash at the trial-eval boundary")

    return on_round


def _noop_state():
    return None  # BanditBudget does not read state
