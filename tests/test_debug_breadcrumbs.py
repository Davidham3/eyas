"""DEBUG phase breadcrumbs (the minimum inventory EYAS_LOG_LEVEL=DEBUG makes visible).

Inventory under test (messages carry counts/elapsed only — never payloads):
- OptimizerLoop: round start / step done / end with elapsed sec (strategy.py)
- OptimizerLoop's default-level counterpart: the terminal progress contract
  ``round <i>/<max>: <n> reqs in <x>s eta≈<y>s``, INFO (strategy.py)
- RequestPlanner: every Rejected at DEBUG + ``reject_counts`` by reason (planner.py)
- Evaluator/SqliteRecordStore: per-batch cache hit/miss line, eval start/done,
  compare fan-out pair counts, store flush/close + ``hits``/``misses`` counters
  (evaluation.py)
- MediaPrefetchLoader: batch load start/done with counts (prefetch.py)
- The breadcrumbs bump the watchdog source on their own (zero-log ≠ hang proof)

Capture reuses the established pattern from tests/test_backends_scheduling.py:
the "eyas" logger is propagate=False with a queue handler, so the fixture
force-enables propagation and lowers the level to DEBUG.
"""

from __future__ import annotations

import logging
import re
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from eyas import watchdog
from eyas.backends.base import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    SamplingConfig,
    system_message,
    user_message,
    user_text,
)
from eyas.backends.mock import ScriptedMockBackend
from eyas.backends.prefetch import MediaPrefetchLoader
from eyas.backends.scheduling import run_sync
from eyas.core.objective import Objective
from eyas.engine import status as status_mod
from eyas.engine.budget import LoadPlan, WorkItem
from eyas.engine.capability import ABSOLUTE_EVAL, CapSpec, ResourceProfile
from eyas.engine.domain import OptimizerState, Prompt, PromptListState
from eyas.engine.evaluation import AbsoluteEvalModel, Evaluator, SqliteRecordStore
from eyas.engine.planner import RequestPlanner
from eyas.engine.registry import register_strategy
from eyas.engine.runtime import run_experiment
from eyas.engine.strategies.hill_climb import HillClimbConfig, HillClimbStrategy
from eyas.engine.strategies.raw import RawConfig, RawStrategy
from eyas.engine.strategy import (
    Decision,
    EvalView,
    OptimizerLoop,
    Strategy,
    _resumed_round_seconds,
    build_run_context,
)
from eyas.logging import _QUEUE_MAXSIZE, _reset, drain
from eyas.registry import register_task
from eyas.tasks.base import Prediction, Sample, Task, Variant
from eyas.tasks.protocol import IdentityProtocol

_LABELS = {"q1": "cat", "q2": "dog"}


@pytest.fixture
def eyas_debug_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture DEBUG records from the whole eyas namespace."""
    eyas_root = logging.getLogger("eyas")
    old_propagate, old_level = eyas_root.propagate, eyas_root.level
    eyas_root.propagate = True
    caplog.set_level(logging.DEBUG, logger="eyas")
    yield caplog
    eyas_root.propagate = old_propagate
    eyas_root.setLevel(old_level)


class ToyMCQ(Task):
    name = "toy_mcq"
    objectives = (Objective(key="accuracy"),)
    default_prompt = "Classify the animal."

    def load_split(self, split: str) -> Sequence[Sample]:
        return [
            Sample(id="q1", inputs={"question": "q1"}, references={"label": "cat"}),
            Sample(id="q2", inputs={"question": "q2"}, references={"label": "dog"}),
        ]

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(system_message(prompt), user_message(sample.inputs["question"]))
        )

    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction:
        return Prediction(value=response.text.strip(), raw=response.text)

    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]:
        return {"accuracy": 1.0 if prediction.value == sample.references["label"] else 0.0}


class TwoImageToyMCQ(ToyMCQ):
    """Every request carries 2 images — rejected by a max_images=1 resource cap."""

    name = "toy_two_image"

    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest:
        return ChatRequest(
            messages=(
                system_message(prompt),
                user_message(ImagePart("a.png"), ImagePart("b.png"), sample.inputs["question"]),
            )
        )


class ImageCappedRaw(RawStrategy):
    """raw with a 1-image cap: the capped plan forces planner rejections."""

    capabilities: ClassVar[CapSpec] = CapSpec(
        name="raw",
        requires_eval=frozenset({ABSOLUTE_EVAL}),
        resource_profile=ResourceProfile(max_images=1),
    )


def _mock() -> ScriptedMockBackend:
    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        return ChatResponse(text=_LABELS.get(user_text(request), ""))

    return ScriptedMockBackend(responder=responder)


def _absolute() -> AbsoluteEvalModel:
    return AbsoluteEvalModel(produces=frozenset({"accuracy", "parse_fail"}))


def test_queue_maxsize_absorbs_debug_bursts() -> None:
    # DEBUG runs emit far more records than INFO; the bounded
    # queue must not degrade to drops on a normal-sized run's breadcrumb burst.
    assert _QUEUE_MAXSIZE >= 8192


def test_round_breadcrumbs_and_reject_counts_end_to_end(eyas_debug_logs) -> None:
    ctx = build_run_context(
        ImageCappedRaw,
        RawConfig(),
        TwoImageToyMCQ(),
        IdentityProtocol(),
        _mock(),
        _absolute(),
    )
    result = OptimizerLoop(ctx).run(ImageCappedRaw(RawConfig(), ctx), ["Classify the animal."])
    assert result.history and result.history[0].n_requests == 0  # every request was rejected

    text = eyas_debug_logs.text
    # round boundaries with elapsed sec, bracketing strategy.step
    assert "[loop] round=0 start" in text
    assert "step done action=evaluate" in text  # before/after strategy.step
    assert "step done action=finish" in text
    assert "[loop] round=0 end" in text and "elapsed=" in text
    # the planner breadcrumb fires for every Rejected (2 samples × 1 variant)
    assert "[planner] rejected" in text
    counts = ctx.evaluator.planner.reject_counts
    assert sum(counts.values()) == 2
    assert any("images per request 2 > limit 1" == reason for reason in counts)
    # evaluation breadcrumbs surround the loop's evaluate call (the round was
    # fully rejected → no engine batch, so start fires but generated does not)
    assert "[eval] start" in text
    assert "[eval] generated" not in text
    # breadcrumbs bump on their own: the run's last activity echo is the
    # loop's own breadcrumb tag (silent console ≠ hang, once the level is open)
    assert watchdog._last_source.startswith("[loop]")


def test_round_progress_echo_on_console(eyas_debug_logs) -> None:
    """Terminal contract: ``round <i>/<max>: <n> reqs in <x>s eta≈<y>s`` at INFO.

    The echo derives from the round data that feeds status.json (the checkpoint's
    own history) plus the loop's round clock; the ``n/a``-until-measured rule is
    verbatim, but a numeric eta projects over
    ``min(max_steps, strategy.expected_rounds())`` -- the method's own declared
    horizon when it is smaller than the enforced ceiling (raw declares its single
    round, so after that round the run is over and the eta reads 0).
    """
    ctx = build_run_context(
        RawStrategy, RawConfig(), ToyMCQ(), IdentityProtocol(), _mock(), _absolute()
    )
    loop = OptimizerLoop(ctx)

    def progress() -> list[logging.LogRecord]:
        # the console-namespace records: strategy.py's INFO echo, nothing else
        return [
            r
            for r in eyas_debug_logs.records
            if r.name == "eyas.engine.strategy" and r.levelno == logging.INFO
        ]

    # before the first completed round there is no duration to average: n/a, verbatim
    loop._echo_round_progress([])
    assert len(progress()) == 1
    assert progress()[0].getMessage() == f"round 0/{ctx.max_steps}: 0 reqs in 0.00s eta≈n/a"

    result = loop.run(RawStrategy(RawConfig(), ctx), ["Classify the animal."])
    assert len(result.history) == 1  # the toy run completes exactly one round

    lines = [r.getMessage() for r in progress()]
    assert len(lines) == 2  # the loop echoes on its own: no caller callback involved
    echo = lines[1]
    # exact prefix + slash-form counter: completed-round count / enforced ceiling
    assert echo.startswith(f"round 1/{ctx.max_steps}: ")
    assert f"{result.history[0].n_requests} reqs in " in echo
    # raw declares its single round: with that round done nothing is left to project
    eta = re.search(r" eta≈([0-9.]+)s$", echo)
    assert eta is not None, echo  # a completed round replaces n/a with a number
    assert float(eta.group(1)) == pytest.approx(0.0, abs=0.01)


def test_eta_falls_back_to_the_ceiling_when_declaration_exceeds_it(eyas_debug_logs) -> None:
    """The declaration only ever lowers the eta horizon. An undeclared/over-declared
    method keeps the ceiling projection: this roundless strategy declares 10k rounds
    against max_steps=3, so after round 1 the eta projects the two ceiling steps."""

    @register_strategy("echo_unrolling")
    class EchoUnrollingStrategy(Strategy):
        config_cls: ClassVar[type] = RawConfig
        state_cls: ClassVar[type] = PromptListState
        capabilities: ClassVar[CapSpec] = CapSpec(
            name="echo_unrolling", requires_eval=frozenset({ABSOLUTE_EVAL})
        )

        def expected_rounds(self) -> int | None:
            return 10_000

        def init(self, seeds: Sequence[Prompt]) -> OptimizerState:
            return PromptListState(candidates=list(seeds), round_idx=0)

        def step(self, state: OptimizerState, view: EvalView) -> Decision:
            assert isinstance(state, PromptListState)
            seed = state.candidates[0]
            if not view.evaluated(seed):
                return Decision("evaluate", prompts=(seed,))
            return Decision("finish", best_id=seed.id)

    ctx = build_run_context(
        EchoUnrollingStrategy,
        RawConfig(),
        ToyMCQ(),
        IdentityProtocol(),
        _mock(),
        _absolute(),
        max_steps=3,
    )
    result = OptimizerLoop(ctx).run(
        EchoUnrollingStrategy(RawConfig(), ctx), ["Classify the animal."]
    )
    assert len(result.history) == 1  # the declaration is no floor: it stops after one round
    lines = [
        r.getMessage()
        for r in eyas_debug_logs.records
        if r.name == "eyas.engine.strategy" and r.levelno == logging.INFO
    ]
    assert len(lines) == 1  # the single completed round is the only echo
    eta = re.search(r" eta≈([0-9.]+)s$", lines[0])
    assert eta is not None, lines[0]
    assert float(eta.group(1)) == pytest.approx(ctx.round_seconds[0] * (3 - 1), abs=0.01)


def test_round_progress_eta_averages_every_measured_round(eyas_debug_logs) -> None:
    """A later round's line advances the counter and averages every round so far,
    projected over the strategy's DECLARED horizon (seed round + mutation rounds),
    not the 400-step ceiling."""
    config = HillClimbConfig(rounds=3)  # declares 4 rounds: the seed round + three
    ctx = build_run_context(
        HillClimbStrategy, config, ToyMCQ(), IdentityProtocol(), _mock(), _absolute()
    )
    result = OptimizerLoop(ctx).run(HillClimbStrategy(config, ctx), ["Classify the animal."])
    # the toy mock scores every candidate correct, so dedup can end the loop a
    # round before the declared horizon -- the declaration is an approximation,
    # the finish decision stays the truth
    assert len(result.history) >= 3

    lines = [
        r.getMessage()
        for r in eyas_debug_logs.records
        if r.name == "eyas.engine.strategy" and r.levelno == logging.INFO
    ]
    assert [line.split(":")[0] for line in lines] == [
        f"round {i}/{ctx.max_steps}" for i in range(1, len(lines) + 1)
    ]
    # at round 2 the clock held exactly two measured rounds; two of the four
    # declared rounds were still ahead
    mean = sum(ctx.round_seconds[:2]) / 2
    eta = re.search(r" eta≈([0-9.]+)s$", lines[1])
    assert eta is not None, lines[1]
    assert float(eta.group(1)) == pytest.approx(mean * (4 - 2), abs=0.01)


@register_task("toy_echo")
class ToyEchoTask(ToyMCQ):
    """ToyMCQ under its own registry name — config-driven runs resolve tasks by name."""

    name = "toy_echo"


@pytest.fixture
def _isolated_logging():
    """``run_experiment`` wires framework logging into module-global state."""
    _reset()
    yield
    _reset()


def test_progress_line_reaches_the_sink_and_survives_resume(
    _isolated_logging, tmp_path: Path, monkeypatch
) -> None:
    """Config-driven proof (the runtime plumbing a unit test cannot see): the INFO
    echo lands in the run dir's own log sink, the counter continues across
    ``--resume``, and the refilled clock keeps the eta numeric from the resumed
    attempt's very first round. The resume keeps the config IDENTICAL (the
    fingerprint guard rejects changed non-path keys): attempt 1 is interrupted
    after its second round instead of stopping on a lowered ceiling."""
    cfg: dict[str, Any] = {
        "run": {"name": "t5_resume", "seed": 0, "max_steps": 4},
        "task": {"name": "toy_echo"},
        "protocol": {"name": "identity"},
        "backend": {"name": "mock", "params": {"responses": ["cat", "dog"], "cycle": True}},
        "eval_model": {"name": "absolute"},
        "budget": {"name": "flat"},
        # rounds far above max_steps: the interrupted attempt leaves rounds
        # unfinished, so the resume really has work left and its eta clock has
        # to carry over
        "strategy": {"name": "hill_climb", "params": {"rounds": 50}},
    }
    run_dir = tmp_path / "run"
    real = status_mod.write_status
    calls: list[int] = []

    def interrupt_after_second_round(run_dir: Path, payload: dict[str, Any]) -> None:
        real(run_dir, payload)
        if payload["phase"] == "optimizing":
            calls.append(int(payload["round"]))
            if len(calls) == 2:
                # round 2's checkpoint already persisted (on_round writes it
                # before the status refresh); round 1's echo reached the sink,
                # round 2's ack dies with the crash — the durable moment
                raise KeyboardInterrupt

    monkeypatch.setattr(status_mod, "write_status", interrupt_after_second_round)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(cfg, output_dir=run_dir)
    monkeypatch.undo()
    drain()
    log = (run_dir / "framework.log").read_text(encoding="utf-8")
    assert re.search(r"INFO eyas\.engine\.strategy: round 1/4: \d+ reqs in [\d.]+s eta≈", log)

    # Fake the interrupted attempt's durations (mock rounds take microseconds, so
    # only an inflated number can tell "averaged over the resumed clock" apart
    # from "started over"): the resumed round's eta must carry it.
    prior = status_mod.read_status(run_dir)
    assert prior is not None
    for entry in prior["history"]:
        if entry["phase"] == "optimizing":
            entry["elapsed_sec"] = 1000.0
    status_mod.write_status(run_dir, prior)

    run_experiment(cfg, resume=run_dir)
    drain()
    log = (run_dir / "framework.log").read_text(encoding="utf-8")
    counters = re.findall(r"round (\d+)/(\d+): \d+ reqs in ([\d.]+)s eta≈(\S+)", log)
    # the counter continues where the checkpoint left off (never restarts at 1);
    # round 2 has no echo — its ack died with the crash after the checkpoint write
    assert [(r, m) for r, m, _, _ in counters] == [("1", "4"), ("3", "4"), ("4", "4")]
    assert all(eta.endswith("s") for *_, eta in counters)  # never n/a once rounds exist
    resumed_eta = float(counters[1][3][:-1])
    assert resumed_eta >= 100.0  # the pre-resume round's seconds fed this average


def test_resume_refills_the_eta_clock(tmp_path: Path) -> None:
    """The clock survives ``--resume``: it refills from status.json's round phases."""
    doc = status_mod.merge_status(None, phase="starting", round=0, elapsed_sec=9.0)
    doc = status_mod.merge_status(doc, phase="optimizing", round=1, elapsed_sec=1.5)
    doc = status_mod.merge_status(doc, phase="optimizing", round=2, elapsed_sec=2.5)
    status_mod.write_status(tmp_path, doc)

    assert _resumed_round_seconds(tmp_path) == [1.5, 2.5]  # the rounds, nothing else
    assert _resumed_round_seconds(None) == []  # library loop: no run dir
    assert _resumed_round_seconds(tmp_path / "no-such-dir") == []  # pre-status run dir


def test_reject_counts_is_read_only() -> None:
    planner = RequestPlanner(ResourceProfile(max_images=1))
    with pytest.raises(AttributeError):
        planner.reject_counts = {}  # type: ignore[misc]


def test_cache_hit_miss_counters_and_store_breadcrumbs(eyas_debug_logs, tmp_path: Path) -> None:
    store = SqliteRecordStore(namespace="t", path=tmp_path / "records.sqlite")
    evaluator = Evaluator(
        task=ToyMCQ(),
        protocol=IdentityProtocol(),
        backend=_mock(),
        planner=RequestPlanner(ResourceProfile(), context_window=None),
        model=_absolute(),
        store=store,
    )
    prompt = Prompt("Classify the animal.")
    plan = LoadPlan(items=(WorkItem(prompt),))
    evals = run_sync(evaluator.aevaluate([prompt], plan, "train", 0))
    assert evals[0].scores["accuracy"] == 1.0
    assert (store.hits, store.misses) == (0, 2)

    # replay the same work: the store answers every sample (cache-first)
    evals = run_sync(evaluator.aevaluate([prompt], plan, "train", 1))
    assert evals[0].scores["accuracy"] == 1.0
    assert (store.hits, store.misses) == (2, 2)

    text = eyas_debug_logs.text
    assert "[eval] cache hit/miss hit=0 miss=2 requests=2" in text
    assert "[eval] cache hit/miss hit=2 miss=0 requests=0" in text
    assert "[eval] generated" in text  # the cold round really sent the batch

    store.flush()
    store.close()
    text = eyas_debug_logs.text  # fresh snapshot: the store lifecycle lines
    assert "[store] flush" in text and "[store] close" in text
    # the eval-side breadcrumbs bump too (cold miss path): done carries the eval tag
    assert watchdog._last_source.startswith("[eval]")


async def test_compare_fanout_breadcrumb(eyas_debug_logs) -> None:
    from eyas.engine.evaluation import CompareEvalModel

    def responder(request: ChatRequest, sampling: SamplingConfig, idx: int) -> ChatResponse:
        return ChatResponse(text="A")

    task = ToyMCQ()
    evaluator = Evaluator(
        task=task,
        protocol=IdentityProtocol(),
        backend=ScriptedMockBackend(responder=responder),
        planner=RequestPlanner(ResourceProfile(), context_window=None),
        model=CompareEvalModel(),
    )
    prompts = [Prompt("Answer step by step."), Prompt("Answer instantly.")]
    plan = LoadPlan(items=tuple(WorkItem(p) for p in prompts))
    evals = await evaluator.aevaluate(prompts, plan, "train")
    assert len(evals) == 2
    # DEBUG line with the pair count before the compare batch: 2 prompts →
    # 2 ordered pairs × 2 samples = 4 game requests
    assert "[compare] start" in eyas_debug_logs.text
    assert "pairs=2 prompts=2 samples=2 requests=4" in eyas_debug_logs.text


# ---------------------------------------------------------------------------
# prefetch breadcrumbs (fast tier: stdlib-built PNG; same shape as
# tests/test_backends_prefetch.py)
# ---------------------------------------------------------------------------
def _png_bytes(width: int = 2, height: int = 2) -> bytes:
    import struct

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes((255, 0, 0)) * width
    idat = zlib.compress(row * height)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_prefetch_batch_breadcrumbs(eyas_debug_logs, tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    paths = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        p.write_bytes(_png_bytes())
        paths.append(str(p))
    loader = MediaPrefetchLoader(workers=2)
    try:
        requests = [
            ChatRequest(messages=(user_message(ImagePart(paths[0]), "q1"),)),
            ChatRequest(messages=(user_message(ImagePart(paths[1]), "q2"),)),
        ]
        loader.materialize(requests)
        text = eyas_debug_logs.text
        assert "[prefetch] batch start requests=2 images=2" in text
        assert "[prefetch] batch done" in text and "elapsed=" in text
    finally:
        loader.close()
