"""Experiment entry point: sectioned config → assembly (capability
validation) → OptimizerLoop → persist to disk.

Component factories build task/backend/judge/protocol from the base registry
and strategy/eval_model/budget from the engine registry. Capability validation
happens at assembly time (build_run_context).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import yaml

import eyas.builtin  # noqa: F401  registers reused components (task/backend/judge/protocol)
import eyas.engine.builtin  # noqa: F401  registers the engine components
from eyas import registry as base_registry
from eyas.backends.base import SamplingConfig
from eyas.engine import registry as engine_registry
from eyas.engine import status as status_mod
from eyas.engine.evaluation import SqliteRecordStore
from eyas.engine.strategy import OptimizerLoop, RunResult, build_run_context
from eyas.logging import get_logger, resolve_log_level
from eyas.logging import setup as setup_logging
from eyas.runtime.config import load_plugins

__all__ = [
    "BackendPool",
    "build_context_from_config",
    "load_config",
    "resolve_run_dir",
    "run_experiment",
    "summarize_runs",
    "run_plan",
    "run_sweep",
]

_log = get_logger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a sectioned YAML, validate the required sections
    (task/backend/strategy), and import the ``plugins`` modules (registration
    must precede any registry lookup)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    for section in ("task", "backend", "strategy"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(
                f"config is missing required section '{section}' (must be a mapping): {path}"
            )
    load_plugins(raw)
    return raw


def _run_name(cfg: dict[str, Any]) -> str:
    """Run name: ``run.name`` when set, else the strategy_task_seed default
    (shared by the run dir layout and status.json so both say the same thing)."""
    run_cfg = cfg.get("run", {})
    name = run_cfg.get("name")
    if name:
        return str(name)
    return "{}_{}_seed{}".format(
        cfg["strategy"].get("name", "unknown"),
        cfg["task"].get("name", "unknown"),
        run_cfg.get("seed", 42),
    )


def resolve_run_dir(cfg: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Run dir for one run: ``outputs/<strategy_task_seed>/<stamp>``."""
    run_cfg = cfg.get("run", {})
    name = _run_name(cfg)
    root = Path(run_cfg.get("output_root", "outputs"))
    if output_dir is not None:
        return Path(output_dir)
    explicit = run_cfg.get("output_dir")
    if explicit:
        return Path(explicit)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / name / stamp
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = root / name / f"{stamp}-{n}"
    return run_dir


def _build_backend(cfg: dict[str, Any]) -> Any:
    backend = base_registry.get("backend", cfg["name"])(**cfg.get("params", {}))
    return backend


class BackendPool:
    """Sweep/plan-level core-backend reuse keyed by the canonical backend config.

    A batch of configs that differ only in strategy/task re-pays the full
    engine weight load (minutes on large models) on every config. The pool dedups CORE backends; the
    per-run layers (Traced / Normalized / Planned and their recorders/planners)
    are still built fresh per run on top of the shared core, so the observation
    surface and planner semantics stay per-run.

    Close ownership moves to the pool: a run must not close a shared core
    (the per-run explicit close in run_experiment is precisely skipped when
    pooling — the pinned engine-release semantics survive, the release just happens once at
    pool end instead of once per run).
    """

    def __init__(self) -> None:
        self._pool: dict[str, Any] = {}

    @staticmethod
    def _key(backend_cfg: dict[str, Any]) -> str:
        return json.dumps(backend_cfg, sort_keys=True, default=str)

    def get(self, backend_cfg: dict[str, Any]) -> Any:
        key = self._key(backend_cfg)
        backend = self._pool.get(key)
        if backend is None:
            backend = _build_backend(backend_cfg)
            self._pool[key] = backend
        return backend

    def close_all(self) -> None:
        for backend in self._pool.values():
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 — one sick engine must not hide the rest
                get_logger(__name__).warning("pooled backend close failed: %r", exc)
        self._pool.clear()


def _build_task(cfg: dict[str, Any]) -> Any:
    return base_registry.get("task", cfg["name"])(**cfg.get("params", {}))


def _build_eval_model(cfg: dict[str, Any], task: Any) -> Any:
    name = cfg.get("name", "absolute")
    # registry lookup first: an unknown name fails fast here, never
    # silently degrades to the absolute model
    cls = engine_registry.get("eval_model", name)
    if name == "compare":
        judge_name = cfg.get("params", {}).get("pairwise_judge")
        pairwise_judge = engine_registry.get("pairwise_judge", judge_name)() if judge_name else None
        return cls(pairwise_judge=pairwise_judge)
    if name == "absolute":
        produces = frozenset({o.key for o in task.objectives} | {"parse_fail"})
        return cls(produces=produces)
    raise ValueError(
        f"eval_model {name!r} is registered but the runtime config builder "
        "cannot construct it from config yet (known limitation: construction "
        "is dispatch-coded for the built-ins; register as built-in or extend "
        "_build_eval_model)"
    )


def _build_budget(cfg: dict[str, Any]) -> Any:
    name = cfg.get("name", "flat")
    budget_cls = engine_registry.get("budget", name)
    return budget_cls(**cfg.get("params", {}))


def _build_judge(cfg: dict[str, Any] | None) -> Any:
    if not cfg:
        return None
    return base_registry.get("judge", cfg["name"])(**cfg.get("params", {}))


def build_context_from_config(
    cfg: dict[str, Any],
    *,
    trace_recorder: Any | None = None,
    artifacts: Any | None = None,
    run_dir: Path | None = None,
    pool: BackendPool | None = None,
) -> Any:
    run_cfg = cfg.get("run", {})
    eval_split = run_cfg.get("eval_split", "train")
    final_eval_split = (run_cfg.get("final_eval") or {}).get("split")
    task = _build_task(cfg["task"])
    protocol = base_registry.get("protocol", cfg.get("protocol", {}).get("name", "identity"))()
    backend = pool.get(cfg["backend"]) if pool is not None else _build_backend(cfg["backend"])
    if trace_recorder is not None:
        from eyas.runtime.trace import TracedBackend

        backend = TracedBackend(backend, trace_recorder)
    # Response normalization sits at the outermost layer (single chokepoint):
    # text is always the clean answer; unstripped inline think would pollute
    # parse and the judge prompt.
    from eyas.backends.normalize import NormalizingBackend

    backend = NormalizingBackend(backend)
    judge = _build_judge(cfg.get("judge"))
    eval_model = _build_eval_model(cfg.get("eval_model", {"name": "absolute"}), task)
    budget = _build_budget(cfg.get("budget", {"name": "flat"}))

    strategy_cfg = cfg["strategy"]
    strategy_cls = engine_registry.get("strategy", strategy_cfg["name"])
    strategy_config = strategy_cls.config_cls(**strategy_cfg.get("params", {}))

    sampling = SamplingConfig(**cfg["backend"].get("sampling", {}))
    sampling_profiles = {
        name: SamplingConfig(**params)
        for name, params in (cfg.get("sampling_profiles") or {}).items()
    }
    store_path = run_cfg.get("cache_dir")
    store = SqliteRecordStore(path=Path(store_path)) if store_path else None

    return build_run_context(
        strategy_cls,
        strategy_config,
        task,
        protocol,
        backend,
        eval_model,
        budget=budget,
        judge=judge,
        store=store,
        artifacts=artifacts,
        sampling=sampling,
        sampling_profiles=sampling_profiles,
        eval_split=eval_split,
        final_eval_split=final_eval_split,
        seed=int(run_cfg.get("seed", 42)),
        max_steps=int(run_cfg.get("max_steps", 400)),
        run_dir=run_dir,
    )


class _StatusCounters(TypedDict):
    """The live counter sections a status refresh reads off the context."""

    rejections: dict[str, int]
    cache: dict[str, int]


def _live_status_counters(ctx: Any) -> _StatusCounters:
    """Rejection tally + record-store hit/miss read off the assembled context
    (the in-process counter surface). ``ctx`` is None before assembly succeeds — the
    status layer then reports zeroes rather than pretending to have measurements.
    """
    if ctx is None:
        return {"rejections": {}, "cache": {"hits": 0, "misses": 0}}
    evaluator = ctx.evaluator
    store = evaluator.store
    return {
        "rejections": evaluator.planner.reject_counts,
        "cache": {
            "hits": int(getattr(store, "hits", 0)),
            "misses": int(getattr(store, "misses", 0)),
        },
    }


#: Run-section keys that only say WHERE the run lives, never WHAT it is:
#: ``output_dir`` picks the fresh run dir and ``output_root`` roots the fresh
#: run-dir layout (a resume pins the dir itself and reads neither), and
#: ``cache_dir`` is auto-defaulted into the run dir at startup. Exactly these
#: keys are excluded from the resume config fingerprint.
_PATH_ONLY_RUN_KEYS = frozenset({"output_dir", "output_root", "cache_dir"})


def _fingerprint_default(value: Any) -> str:
    """json.dumps fallback for non-JSON config values (mirrors the
    run_config.json dump's ``default=str``). Callables — mock responders in
    library/test configs — get a stable qualified-name identity: their repr
    carries a memory address, so two freshly built identical configs would
    otherwise read as a config change."""
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', type(value).__name__)}>"
    return str(value)


def _config_identity(cfg: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """(whole-config fingerprint, per-top-level-section hashes) of the run
    config minus the path-only run keys. An emptied run section drops out
    entirely, so the startup ``cache_dir`` default (which mutates cfg) can
    never shift the identity — fresh-run and resume sides compute the same
    values from the same user config regardless of when they call this."""
    canonical = dict(cfg)
    run_section = canonical.get("run")
    if isinstance(run_section, dict):
        trimmed = {k: v for k, v in run_section.items() if k not in _PATH_ONLY_RUN_KEYS}
        if trimmed:
            canonical["run"] = trimmed
        else:
            del canonical["run"]
    sections = {
        name: hashlib.sha256(
            json.dumps(value, sort_keys=True, default=_fingerprint_default).encode("utf-8")
        ).hexdigest()
        for name, value in canonical.items()
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=_fingerprint_default).encode("utf-8")
    ).hexdigest()
    return fingerprint, sections


def _check_resume_config(
    cfg: dict[str, Any], resume_payload: dict[str, Any], ckpt_path: Path
) -> None:
    """The resume config guard: the checkpoint's frozen fingerprint must equal
    the current config's (path-only keys excluded). A mismatch names the changed
    top-level sections via the companion per-section hashes; a checkpoint
    predating the fingerprint field is unverifiable and rejected with re-run
    guidance — never a tolerant resume, never a silent fresh start."""
    fingerprint, sections = _config_identity(cfg)
    stored = resume_payload.get("config_fingerprint")
    if stored == fingerprint:
        return
    if stored is None:
        detail = (
            "the checkpoint carries no config_fingerprint (it predates the resume "
            "config guard), so config identity cannot be verified"
        )
    else:
        stored_sections = resume_payload.get("config_sections")
        if not isinstance(stored_sections, dict):
            stored_sections = {}
        changed = sorted(
            name
            for name in set(sections) | set(stored_sections)
            if sections.get(name) != stored_sections.get(name)
        )
        detail = "changed top-level sections: " + (", ".join(changed) or "<unknown>")
    raise RuntimeError(
        f"resume config mismatch ({detail}) — {ckpt_path} was written under a different "
        "config; resume requires the identical config except the path-only keys "
        "run.output_dir / run.output_root / run.cache_dir; re-run with the original "
        "config, or start a fresh run dir without --resume"
    )


def _startup_run_dir(
    cfg: dict[str, Any], output_dir: str | Path | None, resume: str | Path | None
) -> tuple[Path, dict[str, Any] | None]:
    """Startup phase of a run: resolve the run dir (or the resume dir and its
    checkpoint), create the dir, wire the log file, and default the evaluation
    record cache into the run dir."""
    resume_payload: dict[str, Any] | None = None
    if resume is not None:
        run_dir = Path(resume)
        ckpt_path = run_dir / "checkpoint.json"
        if not ckpt_path.exists():
            raise RuntimeError(
                f"--resume target {run_dir} has no checkpoint.json — refusing a silent "
                "fresh start inside a resume dir (the run never reached a round-end "
                "checkpoint, or this is not a run dir); re-run without --resume to "
                "start fresh"
            )
        try:
            resume_payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"checkpoint at {ckpt_path} is corrupt (truncated/torn JSON: {exc}) — "
                "the evaluation records under the run dir's cache/ survive, but this "
                "checkpoint's loop position is lost (round-end writes are atomic; a "
                "torn file predates that or took external damage)"
            ) from exc
        _check_resume_config(cfg, resume_payload, ckpt_path)
    else:
        run_dir = resolve_run_dir(cfg, output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent: a sweep re-calls per run — the file handler follows each run's dir.
    setup_logging(file=run_dir / "framework.log", level=resolve_log_level())

    # Evaluation record cache defaults to living inside the run dir (the cache
    # is checkpointed with the run dir; resubmitting the same name / resuming
    # replays already-burned requests); an explicit run.cache_dir config takes
    # precedence.
    run_section = cfg.setdefault("run", {})
    if not run_section.get("cache_dir"):
        run_section["cache_dir"] = str(run_dir / "cache" / "eyas.sqlite")
    return run_dir, resume_payload


@dataclass
class _RunParts:
    """Mutable assembly state for one run, filled step by step so a failure
    mid-assembly still leaves the pieces the status/close paths must see
    (a half-built ctx must reach both the failed-phase counters and the
    engine release in ``finally``)."""

    recorder: Any | None = None
    ctx: Any = None
    strategy: Any = None
    seeds: list[str] = field(default_factory=list)


class _StatusTracker:
    """status.json (schema v1) bookkeeping for one run: this run loop is the
    file's only writer. Written before assembly so a slow start is never
    mistaken for a run dir that never began. A resume seeds the cumulative
    fields from the file the interrupted run left behind — observation
    continuity across resume (a resumed run never reports round/cost back
    at zero)."""

    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: Path,
        parts: _RunParts,
        resume_payload: dict[str, Any] | None,
        resume: str | Path | None,
    ) -> None:
        run_section = cfg.get("run", {})
        self._cfg = cfg
        self._run_dir = run_dir
        self._parts = parts
        self._state = status_mod.merge_status(
            status_mod.read_status(run_dir) if resume is not None else None,
            phase="starting",
            round=int(resume_payload.get("round_idx", 0)) if resume_payload else 0,
            run={
                "name": _run_name(cfg),
                "strategy": cfg["strategy"].get("name", "unknown"),
                "task": cfg["task"].get("name", "unknown"),
                "seed": int(run_section.get("seed", 42)),
                "run_dir": str(run_dir),
            },
            budget=self._budget_section(
                resume_payload.get("budget_state") if resume_payload else None
            ),
            cost=status_mod.cost_section(
                resume_payload.get("total_cost") if resume_payload else None
            ),
            cache={"hits": 0, "misses": 0},
            rejections={},
        )
        status_mod.write_status(run_dir, self._state)
        self._t = time.monotonic()

    def _budget_section(self, budget_state: Any) -> dict[str, Any]:
        return status_mod.budget_section(
            engine_registry.get("budget", (self._cfg.get("budget") or {}).get("name", "flat")),
            budget_state,
        )

    def advance(self, next_state: dict[str, Any]) -> None:
        # Rebind the run-local document so each refresh builds on the previous
        # one (history accumulates, keys this version does not know carry
        # forward); the entry just appended reports the interval since the
        # previous refresh — the span being left, not the time spent in the
        # phase being reported — measured on the run's own clock.
        now = time.monotonic()
        next_state["history"][-1]["elapsed_sec"] = round(now - self._t, 3)
        self._t = now
        self._state = next_state
        status_mod.write_status(self._run_dir, next_state)

    def mark_failed(self) -> None:
        # Closed-loop for the started-run invariant: every run that wrote its
        # run-start status ends with either result.json or phase "failed" —
        # the last phase alone cannot tell "still running" from "dead". The
        # caller always re-raises (exception semantics untouched); a failed
        # status write is a lost observation, never a second error.
        try:
            self.advance(
                status_mod.merge_status(
                    self._state,
                    phase="failed",
                    round=int(self._state.get("round", 0)),
                    **_live_status_counters(self._parts.ctx),
                )
            )
        except Exception:
            _log.exception("failed to record status phase=failed in %s", self._run_dir)

    def on_round(self, snapshot: dict[str, Any]) -> None:
        payload = json.dumps(snapshot, ensure_ascii=False)
        status_mod.write_json_atomic(self._run_dir / "checkpoint.json", payload)
        # Round echo → status refresh: the round / cost values the checkpoint
        # just froze are exactly what an observer should see; so does the
        # budget water level, taken from the checkpoint's frozen budget_state.
        self.advance(
            status_mod.merge_status(
                self._state,
                phase="optimizing",
                round=int(snapshot.get("round_idx", 0)),
                budget=self._budget_section(snapshot.get("budget_state")),
                cost=status_mod.cost_section(snapshot.get("total_cost")),
                **_live_status_counters(self._parts.ctx),
            )
        )

    def mark_done(self, result: RunResult) -> None:
        self.advance(
            status_mod.merge_status(
                self._state,
                phase="done",
                round=len(result.history),
                cost=status_mod.cost_section(result.total_cost.to_json()),
                **_live_status_counters(self._parts.ctx),
            )
        )


def _assemble_parts(
    cfg: dict[str, Any],
    run_dir: Path,
    parts: _RunParts,
    pool: BackendPool | None,
    resume_payload: dict[str, Any] | None = None,
) -> None:
    """Build the pieces the loop runs on — trace recorder, watchdog, artifacts,
    run context (backend/judge/eval/budget via build_context_from_config),
    strategy and seeds — filling ``parts`` as each lands."""
    # First statements of the failed-phase window: the recorder's eager
    # filesystem IO (mkdir + open) must sit inside the window, otherwise a
    # trace-path failure strands the run at phase "starting" with no
    # result.json (the started-run invariant's hole).
    if cfg.get("run", {}).get("trace"):
        from eyas.runtime.trace import TraceRecorder

        parts.recorder = TraceRecorder(
            run_dir / "trace.jsonl",
            labels={"strategy": cfg["strategy"].get("name", "unknown")},
        )

    from eyas.engine.artifacts import ArtifactStore
    from eyas.watchdog import start_watchdog

    # Hang protection: bump is already stamped in evaluation/loop; here we
    # start the watchdog thread consuming it.
    start_watchdog(cfg.get("run", {}).get("watchdog_stall_sec"))
    artifacts = ArtifactStore(path=run_dir / "artifacts")
    if resume_payload is not None:
        # Reload BEFORE assembly validation: a resumed downstream consumer must
        # find its upstream artifacts (consumes ⊆ available). The checkpoint's
        # frozen manifest is the integrity cross-check — an address it carries
        # that is missing or altered on disk is a genuine inconsistency between
        # the checkpoint and the run dir, so fail fast instead of continuing
        # with an amputated store.
        artifacts.load()
        reloaded = artifacts.manifest()
        for address, digest in (resume_payload.get("artifacts") or {}).items():
            if reloaded.get(address) != digest:
                raise RuntimeError(
                    f"checkpoint artifact {address!r} is missing or altered on disk "
                    f"(checkpoint digest {digest}, found {reloaded.get(address)!r}) — "
                    f"{run_dir / 'artifacts'} no longer matches the checkpoint"
                )
    parts.ctx = build_context_from_config(
        cfg, trace_recorder=parts.recorder, artifacts=artifacts, run_dir=run_dir, pool=pool
    )
    # the startup config identity rides into every checkpoint snapshot (the
    # resume guard compares against it); path-only keys are excluded, so the
    # startup cache_dir default (a cfg mutation) cannot shift it
    parts.ctx.config_fingerprint, parts.ctx.config_sections = _config_identity(cfg)
    strategy_cls = engine_registry.get("strategy", cfg["strategy"]["name"])
    strategy_config = strategy_cls.config_cls(**cfg["strategy"].get("params", {}))
    parts.strategy = strategy_cls(strategy_config, parts.ctx)
    seeds = cfg["strategy"].get("seeds") or [parts.ctx.task.default_prompt]
    parts.seeds = list(seeds)


def _redacted_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of the run config for the run_config.json dump: a literal
    ``backend.params.api_key`` is a credential and must never land in the run
    dir in plaintext. Only the dump copy is redacted — the live cfg (resume
    config fingerprint, running backend) keeps the real key. ``api_key_env``
    is an env-var NAME, not a secret, and stays visible."""
    redacted = copy.deepcopy(cfg)
    backend = redacted.get("backend")
    params = backend.get("params") if isinstance(backend, dict) else None
    if isinstance(params, dict) and "api_key" in params:
        params["api_key"] = "***REDACTED***"
    return redacted


def _run_and_finalize(
    cfg: dict[str, Any],
    run_dir: Path,
    parts: _RunParts,
    resume_payload: dict[str, Any] | None,
    tracker: _StatusTracker,
) -> RunResult:
    """Run the loop to completion, then finalize (result.json / run_config.json
    + the "done" status refresh)."""
    result = OptimizerLoop(parts.ctx).run(
        parts.strategy, list(parts.seeds), resume=resume_payload, on_round=tracker.on_round
    )
    # Finalization lives INSIDE the failed-phase window: a result.json /
    # run_config.json write failure (disk full, IO error) after a completed
    # loop is still a lost observation and must leave phase "failed" —
    # otherwise the run dies at "optimizing" with neither close. The live
    # counters are read before the finally chain releases the store/engine.
    result.export(run_dir / "result.json")
    (run_dir / "run_config.json").write_text(
        json.dumps(_redacted_config(cfg), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tracker.mark_done(result)
    return result


def _release_parts(parts: _RunParts, pool: BackendPool | None) -> None:
    """Explicitly release the engine at the end of the run:
    the next run in a sweep must not pay GC's bill — an unreleased engine
    can otherwise hit CUDA OOM while loading weights. Under a BackendPool the
    core is shared across runs: close ownership moves to the pool and the
    per-run release is precisely skipped (it would kill the shared engine)."""
    if parts.ctx is not None and pool is None:
        parts.ctx.evaluator.backend.close()
    if parts.recorder is not None:
        parts.recorder.close()


def run_experiment(
    cfg: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    pool: BackendPool | None = None,
) -> RunResult:
    """Run an experiment; when ``resume`` points at an existing run dir,
    resume from checkpoint.json.

    Phases: startup (run dir + checkpoint + logging), status bootstrap,
    assembly of the parts, the optimizer loop with finalization — the whole
    span from the run-start status write through finalization is the
    failed-phase window: assembly failures, mid-loop failures, and
    finalization-write failures all leave phase "failed" behind (propagation
    unchanged after; mark_failed treats a failed finalization as a lost
    observation, never as a second error).
    """
    run_dir, resume_payload = _startup_run_dir(cfg, output_dir, resume)
    parts = _RunParts()
    tracker = _StatusTracker(cfg, run_dir, parts, resume_payload, resume)
    try:
        _assemble_parts(cfg, run_dir, parts, pool, resume_payload)
        result = _run_and_finalize(cfg, run_dir, parts, resume_payload, tracker)
    except BaseException:
        tracker.mark_failed()
        raise
    finally:
        _release_parts(parts, pool)
    return result


def run_sweep(
    configs: list[dict[str, Any]], output_root: str | Path, *, share_backends: bool = False
) -> list[Path]:
    """Run multiple configs sequentially into ``output_root/<idx>-<name>/``;
    any failure aborts the whole sweep.

    ``share_backends=True`` reuses CORE backends across configs whose backend
    block is identical (one engine weight-load per sweep instead of one per
    config; see BackendPool). Default False keeps the fully isolated per-run
    assembly — including per-run close.
    """
    output_root = Path(output_root)
    entries: list[tuple[dict[str, Any], Path]] = []
    for index, cfg in enumerate(configs):
        name = str(cfg.get("run", {}).get("name", f"run{index}"))
        entries.append((cfg, output_root / f"{index:02d}-{name}"))
    pool = BackendPool() if share_backends else None
    run_dirs: list[Path] = []
    try:
        for cfg, run_dir in entries:
            run_experiment(cfg, output_dir=run_dir, pool=pool)  # failure propagates up
            run_dirs.append(run_dir)
    finally:
        if pool is not None:
            pool.close_all()
    return run_dirs


def run_plan(
    entries: list[tuple[dict[str, Any], Path]], *, continue_on_error: bool = True
) -> dict[str, str | None]:
    """Execute a sweep PLAN: explicit (config, run_dir) pairs, one process, one
    shared BackendPool — the engine weight-load amortizes across the whole plan
    (plans span datasets on purpose: the engine is dataset-independent, so a
    batch of datasets on one model tag pays exactly ONE weight load).

    Unlike run_sweep, a failed run is contained BY DEFAULT: runs are
    checkpoint-isolated (run dir = checkpoint), so one poisoned config must not
    sacrifice an otherwise healthy multi-hour plan. Outcomes map run_dir ->
    None (success) or the error repr; the CLI surfaces any failure as a
    non-zero exit. ``continue_on_error=False`` restores fail-fast.
    """
    log = get_logger(__name__)
    pool = BackendPool()
    outcomes: dict[str, str | None] = {}
    try:
        for index, (cfg, run_dir) in enumerate(entries):
            name = str(cfg.get("run", {}).get("name", f"entry{index}"))
            try:
                run_experiment(cfg, output_dir=run_dir, pool=pool)
                outcomes[str(run_dir)] = None
            except Exception as exc:  # noqa: BLE001 — contained, counted, re-surfaced by the caller
                outcomes[str(run_dir)] = repr(exc)
                log.error("[plan] run %s failed: %r (plan continues)", name, exc)
                if not continue_on_error:
                    raise
    finally:
        pool.close_all()
    return outcomes


def summarize_runs(run_dirs: list[Path]) -> str:
    """Cross-run summary markdown (delegates to :mod:`eyas.engine.report`)."""
    from eyas.engine.report import report_runs

    return report_runs(run_dirs, fmt="md")
