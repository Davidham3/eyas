"""Runtime end to end: segmented config -> assembly (capability validation) -> OptimizerLoop -> result.json -> report.

Exercises the full chain behind the CLI with a registry-registered toy task + mock(responses,cycle) (zero GPU).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from eyas.backends.base import ChatRequest, ChatResponse, system_message, user_message
from eyas.backends.mock import ScriptedMockBackend
from eyas.core.objective import Objective
from eyas.engine.runtime import load_config, resolve_run_dir, run_experiment, summarize_runs
from eyas.logging import _reset
from eyas.registry import register_backend, register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant


@pytest.fixture(autouse=True)
def _isolated_logging():
    """run_experiment wires framework logging into module-global state; reset around each test."""
    _reset()
    yield
    _reset()


@register_task("toy_it")
class ToyIT(Task):
    name = "toy_it"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"})]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


def _cfg() -> dict[str, object]:
    return {
        "run": {"name": "toy_it_run", "seed": 0},
        "task": {"name": "toy_it"},
        "protocol": {"name": "identity"},
        "backend": {"name": "mock", "params": {"responses": ["cat"], "cycle": True}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        "strategy": {
            "name": "cot",
            "params": {"base_prompt": "Classify the animal.", "cot_suffix": "Think step by step."},
        },
    }


def test_run_experiment_end_to_end(tmp_path):
    result = run_experiment(_cfg(), output_dir=tmp_path)
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "run_config.json").exists()
    # the export can be summarized by report
    table = summarize_runs([tmp_path])
    assert "cot" in table and "toy_it_run" in table


def test_run_experiment_starts_watchdog(tmp_path, monkeypatch):
    import eyas.watchdog as wd

    calls: list[object] = []
    monkeypatch.setattr(wd, "start_watchdog", lambda stall=None: calls.append(stall))
    cfg = _cfg()
    cfg["run"] = {**cfg["run"], "watchdog_stall_sec": 123}
    run_experiment(cfg, output_dir=tmp_path)
    assert calls == [
        123
    ]  # the anti-hang watchdog is wired up with the config threshold passed through


def test_run_experiment_writes_framework_log(tmp_path):
    from eyas import logging as eyas_logging

    run_experiment(_cfg(), output_dir=tmp_path)
    assert eyas_logging.drain()
    # the run entry wires framework logging (console + run-dir full-format dual write)
    assert (tmp_path / "framework.log").exists()


def test_run_experiment_accounts_response_tokens(tmp_path):
    cfg = _cfg()
    cfg["backend"] = {
        "name": "mock",
        "params": {
            "responses": [
                ChatResponse(text="cat", usage={"prompt_tokens": 10, "completion_tokens": 4}),
            ],
            "cycle": True,
        },
    }
    result = run_experiment(cfg, output_dir=tmp_path)
    assert result.total_cost.requests > 0
    # every billed request carries the scripted usage; token totals follow the request totals
    assert result.total_cost.input_tokens == 10 * result.total_cost.requests
    assert result.total_cost.output_tokens == 4 * result.total_cost.requests


def test_run_experiment_trace(tmp_path):
    cfg = _cfg()
    cfg["run"] = {**cfg["run"], "trace": True}
    run_experiment(cfg, output_dir=tmp_path)
    trace = tmp_path / "trace.jsonl"
    assert trace.exists()
    assert "llm_batch" in trace.read_text(encoding="utf-8")


def test_report_runs_markdown_and_csv(tmp_path):
    from eyas.engine.report import report_runs

    run_experiment(_cfg(), output_dir=tmp_path)
    md = report_runs([tmp_path], fmt="md")
    assert "| run | strategy |" in md
    assert "## Best prompts" in md and "Think step by step" in md
    csv_text = report_runs([tmp_path], fmt="csv")
    assert csv_text.splitlines()[0].startswith("run,strategy,task,seed,")
    assert "cot" in csv_text


def test_report_runs_rejects_unknown_format(tmp_path):
    import pytest

    from eyas.engine.report import report_runs

    run_experiment(_cfg(), output_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown format"):
        report_runs([tmp_path], fmt="json")


def test_run_sweep_sequential(tmp_path):
    from eyas.engine.report import report_runs
    from eyas.engine.runtime import run_sweep

    a, b = _cfg(), _cfg()
    a["run"] = {**a["run"], "name": "sweep_a"}
    b["run"] = {**b["run"], "name": "sweep_b", "seed": 43}
    dirs = run_sweep([a, b], tmp_path / "root")
    assert [d.name for d in dirs] == ["00-sweep_a", "01-sweep_b"]
    assert all((d / "result.json").is_file() for d in dirs)
    table = report_runs(dirs, fmt="md")
    assert "sweep_a" in table and "sweep_b" in table


def test_run_sweep_aborts_on_failure(tmp_path):
    import pytest

    from eyas.engine.runtime import run_sweep

    good = _cfg()
    good["run"] = {**good["run"], "name": "ok"}
    bad = _cfg()
    bad["strategy"] = {"name": "does_not_exist"}
    root = tmp_path / "root"
    with pytest.raises(Exception, match="does_not_exist"):
        run_sweep([good, bad], root)
    assert (root / "00-ok" / "result.json").is_file()  # the first run completed
    assert not (
        root / "01-toy_it_run" / "result.json"
    ).exists()  # the second run aborted, no result


CLOSE_CALLS: list[str] = []


@register_backend("close_spy")
class CloseSpyBackend(ScriptedMockBackend):
    """close-count spy: used by the engine-release regression anchor."""

    def close(self) -> None:
        CLOSE_CALLS.append("close")
        super().close()


def test_run_experiment_closes_backend_through_wrapper_chain(tmp_path):
    # The end of a run must explicitly close the backend, and close must penetrate the Traced/Normalizing
    # wrapper chain -- otherwise the next run in a sweep hits an unreleased engine: vllm EngineCore dying via
    # GC is mere luck, and the next run CUDA-OOMs on weight load.
    del CLOSE_CALLS[:]
    cfg = _cfg()
    cfg["run"] = {**cfg["run"], "trace": True}
    cfg["backend"] = {"name": "close_spy", "params": {"responses": ["cat"], "cycle": True}}
    run_experiment(cfg, output_dir=tmp_path)
    assert CLOSE_CALLS == ["close"]


def test_run_sweep_closes_backend_per_run(tmp_path):
    from eyas.engine.runtime import run_sweep

    del CLOSE_CALLS[:]
    a, b = _cfg(), _cfg()
    for c, n in ((a, "sweep_ca"), (b, "sweep_cb")):
        c["run"] = {**c["run"], "name": n}
        c["backend"] = {"name": "close_spy", "params": {"responses": ["cat"], "cycle": True}}
    run_sweep([a, b], tmp_path / "root")
    assert CLOSE_CALLS == ["close", "close"]


SEEN_API_KEYS: list[str | None] = []


@register_backend("api_key_spy")
class ApiKeySpyBackend(ScriptedMockBackend):
    """Credential-accepting spy: takes api_key / api_key_env params like the
    openai backend, recording the key the live backend was built with."""

    def __init__(
        self,
        responses: Sequence[str | ChatResponse] | None = None,
        cycle: bool = False,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ):
        SEEN_API_KEYS.append(api_key)
        self.api_key_env = api_key_env
        super().__init__(responses=responses, cycle=cycle)


def test_run_config_dump_redacts_literal_api_key(tmp_path):
    # A literal backend.params.api_key must never land in the run dir in
    # plaintext: the dump redacts a copy while the live cfg (resume fingerprint
    # + running backend) keeps the real key. api_key_env is a variable NAME,
    # not a secret — it stays visible.
    del SEEN_API_KEYS[:]
    cfg = _cfg()
    cfg["backend"] = {
        "name": "api_key_spy",
        "params": {
            "responses": ["cat"],
            "cycle": True,
            "api_key": "sk-secret",
            "api_key_env": "EYAS_TEST_API_KEY",
        },
    }
    run_experiment(cfg, output_dir=tmp_path)
    dumped = (tmp_path / "run_config.json").read_text(encoding="utf-8")
    assert "sk-secret" not in dumped
    params = json.loads(dumped)["backend"]["params"]
    assert params["api_key"] == "***REDACTED***"
    assert params["api_key_env"] == "EYAS_TEST_API_KEY"
    # in-memory cfg untouched; the live backend was built with the real key
    assert cfg["backend"]["params"]["api_key"] == "sk-secret"
    assert SEEN_API_KEYS == ["sk-secret"]


def test_run_normalizes_inline_think(tmp_path):
    # The outermost layer of the assembly chain must carry NormalizingBackend:
    # unstripped inline <think> pollutes parse and judge prompts.
    cfg = _cfg()
    cfg["backend"] = {
        "name": "mock",
        "params": {
            "responses": ["<think>reason it through first</think>\n\ncat"],
            "cycle": True,
        },
    }
    result = run_experiment(cfg, output_dir=tmp_path)
    assert result.best is not None
    assert result.best.scores["accuracy"] == 1.0


def test_resolve_run_dir_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = resolve_run_dir(_cfg())
    assert run_dir.parts[-3:] == ("outputs", "toy_it_run", run_dir.parts[-1])
    assert resolve_run_dir(_cfg(), output_dir=tmp_path / "x") == tmp_path / "x"


def test_load_config_requires_sections(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("task:\n  name: toy_it\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="backend"):
        load_config(bad)
