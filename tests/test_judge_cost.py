"""Judge-channel cost accounting (E3 boundary closure): a verdict that consumed a
backend request must be billed into the run's cost, with the judge channel visible
as its own split (requests + tokens) instead of being silently free.

Contract under test:
1. backend-backed judge (mcq_llm): verdict usage rides Verdict.usage up to the
   evaluator, which bills one request per verdict and splits tokens per prompt;
2. pure-logic judge (exact_match): no backend call -> zero judge cost;
3. ChainedJudge accumulates usage across links (an undecided backend link is
   still billed when a later link decides);
4. run level: totals + report columns carry the judge channel end to end.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from conftest import full_split_plan, make_evaluator

from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    SamplingConfig,
    system_message,
    user_message,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.backends.scheduling import run_sync
from eyas.core.objective import Objective
from eyas.engine import builtin  # noqa: F401  trigger registration
from eyas.engine.domain import Cost, Prompt
from eyas.judges.base import ChainedJudge, JudgeContext
from eyas.judges.exact_match import ExactMatchJudge
from eyas.judges.mcq_judge import McqLLMJudge
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.mcq import match_prediction
from eyas.tasks.protocol import IdentityProtocol

_CHOICES = ["cat", "dog"]
_EVAL_USAGE = {"prompt_tokens": 10, "completion_tokens": 4}
_JUDGE_USAGE = {"prompt_tokens": 6, "completion_tokens": 2}


def _samples() -> list[Sample]:
    return [
        Sample(
            id="s1",
            inputs={"image": "/data/1.jpg", "choices": _CHOICES, "group": "g"},
            references={"label": "cat"},
        ),
        Sample(
            id="s2",
            inputs={"image": "/data/2.jpg", "choices": _CHOICES, "group": "g"},
            references={"label": "dog"},
        ),
    ]


@register_task("cost_toy_mcq")
class CostToyMCQ(Task):
    """Minimal MCQ shape: unparseable text -> value=None (so exact_match abstains
    and the LLM judge takes over)."""

    name = "cost_toy_mcq"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return _samples()

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["image"])),
            request_id=sample.id,
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        value = match_prediction(response.text, sample.inputs["choices"])
        return Prediction(value=value, raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _responder(eval_texts: dict[str, str]):
    """eval texts per sample id; judge requests always answer correctly and carry
    their own usage so the channels are distinguishable in the bill."""

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        if request.request_id.startswith("judge:"):
            return ChatResponse(text="cat", usage=dict(_JUDGE_USAGE))
        sample_id = request.request_id.split(":")[0]
        return ChatResponse(text=eval_texts[sample_id], usage=dict(_EVAL_USAGE))

    return responder


def _evaluator(judge, eval_texts: dict[str, str]):
    return make_evaluator(
        CostToyMCQ(),
        ScriptedMockBackend(responder=_responder(eval_texts)),
        protocol=IdentityProtocol(),
        produces=frozenset({"accuracy", "parse_fail"}),
        judge=judge,
    )


def test_mcq_llm_verdicts_are_billed_with_usage():
    # free-text eval answers force every sample through the LLM judge
    prompt = Prompt("Classify the animal.")
    evals = run_sync(
        _evaluator(McqLLMJudge(), {"s1": "hmm", "s2": "hmm"}).aevaluate(
            [prompt], full_split_plan(prompt), "train", 0
        )
    )
    cost = evals[0].cost
    # 2 eval + 2 judge requests; judge split carries exactly the judge channel
    assert cost.requests == 4
    assert cost.judge_requests == 2
    assert cost.input_tokens == 2 * _EVAL_USAGE["prompt_tokens"] + 2 * _JUDGE_USAGE["prompt_tokens"]
    assert cost.output_tokens == (
        2 * _EVAL_USAGE["completion_tokens"] + 2 * _JUDGE_USAGE["completion_tokens"]
    )
    assert cost.judge_input_tokens == 2 * _JUDGE_USAGE["prompt_tokens"]
    assert cost.judge_output_tokens == 2 * _JUDGE_USAGE["completion_tokens"]
    # judge recovered the correct labels -> accuracy stays a fair score
    assert evals[0].scores["accuracy"] == 0.5


def test_exact_match_judge_costs_nothing():
    prompt = Prompt("Classify the animal.")
    evals = run_sync(
        _evaluator(ExactMatchJudge(), {"s1": "cat", "s2": "dog"}).aevaluate(
            [prompt], full_split_plan(prompt), "train", 0
        )
    )
    cost = evals[0].cost
    assert cost.requests == 2
    assert cost.judge_requests == 0
    assert cost.judge_input_tokens == 0 and cost.judge_output_tokens == 0
    assert evals[0].scores["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# exact_match fail-fast: a missing reference key is a config/data error, not a
# silent 0 (present-but-None keeps the ordinary mismatch-scoring behavior)
# ---------------------------------------------------------------------------
def test_exact_match_missing_reference_key_fails_fast() -> None:
    sample = Sample(
        id="s1",
        inputs={"image": "/data/1.jpg", "choices": _CHOICES, "group": "g"},
        references={"answers": ["cat"]},
    )
    with pytest.raises(ValueError, match=r"sample 's1' has no reference key 'label'") as exc:
        ExactMatchJudge().judge(
            ChatResponse(text="cat"), Prediction(value="cat", raw="cat"), sample, JudgeContext()
        )
    assert "answers" in str(exc.value)  # the available keys are named for remediation


def test_exact_match_present_none_reference_still_scores_zero() -> None:
    sample = Sample(id="s1", inputs={}, references={"label": None})
    verdict = ExactMatchJudge().judge(
        ChatResponse(text="cat"), Prediction(value="cat", raw="cat"), sample, JudgeContext()
    )
    assert verdict.decided
    assert verdict.scores == {"accuracy": 0.0}


def test_chain_bills_the_backend_link_even_when_exact_match_decides():
    prompt = Prompt("Classify the animal.")
    # exact_match decides s1 (no request); s2 is unparseable -> mcq_llm link fires
    judge = ChainedJudge([ExactMatchJudge(), McqLLMJudge()])
    evals = run_sync(
        _evaluator(judge, {"s1": "cat", "s2": "hmm"}).aevaluate(
            [prompt], full_split_plan(prompt), "train", 0
        )
    )
    cost = evals[0].cost
    assert cost.requests == 3  # 2 eval + 1 judge (only s2 reached the backend link)
    assert cost.judge_requests == 1
    assert cost.judge_input_tokens == _JUDGE_USAGE["prompt_tokens"]
    assert cost.judge_output_tokens == _JUDGE_USAGE["completion_tokens"]


def test_cost_json_roundtrip_keeps_judge_split():
    cost = Cost(
        requests=3, input_tokens=30, output_tokens=6, judge_requests=1, judge_input_tokens=6
    )
    assert Cost.from_json(cost.to_json()) == cost


def test_run_totals_and_report_carry_judge_channel(tmp_path):
    from eyas.engine.report import report_runs
    from eyas.engine.runtime import run_experiment

    cfg = {
        "run": {"name": "judge_cost_run", "seed": 0},
        "task": {"name": "cost_toy_mcq"},
        "protocol": {"name": "identity"},
        "backend": {
            "name": "mock",
            "params": {"responses": ["hmm"], "cycle": True},
        },
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "judge": {"name": "mcq_llm"},
        "strategy": {"name": "raw"},
    }
    result = run_experiment(cfg, output_dir=tmp_path)
    # every verdict consumed a request -> half the bill is the judge channel
    assert result.total_cost.judge_requests > 0
    assert result.total_cost.judge_requests <= result.total_cost.requests
    md = report_runs([tmp_path], fmt="md")
    assert "judge_reqs" in md
