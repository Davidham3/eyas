"""Capability declarations and assembly-time compatibility validation
(contract-first).

Every registered piece declares not only a signature but capabilities: which
metrics a task exposes, which evaluation semantics an eval model covers and
which metrics it produces, which semantics a strategy needs and which
objectives it selects on, and backend resource limits. ``validate_assembly``
performs static checks when building the RunContext; an invalid assembly
returns a list of violations (the caller aggregates them into a ValueError).

Rejected assemblies: a compare-only method can only ever be assembled with
compare_eval semantics (producing compare_score), while the run's optimization
target is accuracy — no installed eval model can produce accuracy, so the
assembly is rejected at startup instead of surfacing ``—`` in the report after
the whole run has been paid for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ABSOLUTE_EVAL",
    "COMPARE_EVAL",
    "BackendLimits",
    "CapSpec",
    "EvalModelCap",
    "ResourceProfile",
    "TaskCap",
    "validate_assembly",
]

ABSOLUTE_EVAL = "absolute_eval"
COMPARE_EVAL = "compare_eval"


@dataclass(frozen=True)
class TaskCap:
    """Task capability: declares which metric keys it exposes."""

    name: str
    metrics: frozenset[str]
    optimize_key: str


@dataclass(frozen=True)
class ResourceProfile:
    """Resource profile: multi-image cap + context reserve. Validated at
    assembly time as the intersection with backend limits."""

    max_images: int | None = None
    ctx_reserve: int = 0


@dataclass(frozen=True)
class EvalModelCap:
    """Eval model capability: which evaluation semantics it covers, which
    metric keys it produces.

    Absolute scorer: semantics={absolute_eval}, produces includes the task
    objective key (e.g. accuracy).
    Compare model (compare-semantics methods): semantics={compare_eval}, produces
    only {compare_score}.
    """

    name: str
    semantics: frozenset[str]
    produces: frozenset[str]


@dataclass(frozen=True)
class CapSpec:
    """Capability declaration of a strategy (method)."""

    name: str
    requires_eval: frozenset[str] = frozenset({ABSOLUTE_EVAL})
    # Objective keys this method selects/reports on; every run optimize_key
    # must be covered by an installed eval model
    optimize_keys: frozenset[str] = frozenset()
    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)
    # Cross-method artifacts (§9): produces = addresses this method emits;
    # consumes = upstream addresses it hard-depends on
    produces_artifacts: frozenset[str] = frozenset()
    consumes_artifacts: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BackendLimits:
    """Backend resource limits (declared by the backend; at assembly time the
    strategy's resource_profile is checked against them)."""

    max_images: int | None = None
    context_window: int | None = None


@dataclass(frozen=True)
class _AssemblyFacts:
    """Facts derived once from the installed pieces that the rules check a
    strategy against: what the eval models produce/cover, the backend limits
    and the artifact addresses already available upstream."""

    task: TaskCap
    backend: BackendLimits
    produced: frozenset[str]
    semantics: frozenset[str]
    available_artifacts: frozenset[str]


def _check_semantics(cap: CapSpec, facts: _AssemblyFacts) -> list[str]:
    # Rule 1: the evaluation semantics the strategy requires must be covered by
    # an installed eval model
    missing_sem = cap.requires_eval - facts.semantics
    if missing_sem:
        return [
            f"strategy {cap.name} requires evaluation semantics {sorted(missing_sem)}, "
            f"but the installed eval models only provide {sorted(facts.semantics)}"
        ]
    return []


def _check_objectives(cap: CapSpec, facts: _AssemblyFacts) -> list[str]:
    # Rule 2 (compare-only fix): every optimization objective key of the run must be
    # produced by some installed eval model. When the strategy declares no
    # optimize_keys, fall back to the task's primary objective (degenerate
    # methods such as raw/cot get their metrics via final_eval; their strategy
    # must still declare it can produce the primary objective or rely on the
    # closing evaluation — see the evaluate-side convention).
    keys = cap.optimize_keys or {facts.task.optimize_key}
    unproducible = keys - facts.produced
    if unproducible:
        return [
            f"strategy {cap.name} cannot produce optimization objectives {sorted(unproducible)} "
            f"for task {facts.task.name}: the installed eval models only produce "
            f"{sorted(facts.produced)}; "
            "this method must not enter the leaderboard for these objectives (use an eval model or method covering the metric)"
        ]
    return []


def _check_resources(cap: CapSpec, facts: _AssemblyFacts) -> list[str]:
    # Rule 3: the strategy's resource profile must not exceed backend limits
    violations: list[str] = []
    rp = cap.resource_profile
    backend = facts.backend
    if rp.max_images is not None and backend.max_images is not None:
        if rp.max_images > backend.max_images:
            violations.append(
                f"strategy {cap.name} requests {rp.max_images} images per request, "
                f"exceeding the backend limit of {backend.max_images}"
            )
    if backend.context_window is not None and rp.ctx_reserve >= backend.context_window:
        violations.append(
            f"strategy {cap.name} context reserve {rp.ctx_reserve} >= backend window {backend.context_window}"
        )
    return violations


def _check_artifacts(cap: CapSpec, facts: _AssemblyFacts) -> list[str]:
    # Rule 4 (§9): cross-method artifact addresses the strategy hard-consumes
    # must already be produced upstream (assembly-time dependency validation);
    # an address the strategy produces itself never blocks its own assembly
    if not cap.consumes_artifacts:
        return []
    available = facts.available_artifacts | frozenset(cap.produces_artifacts)
    missing_art = frozenset(cap.consumes_artifacts) - available
    if missing_art:
        return [
            f"strategy {cap.name} depends on cross-method artifacts {sorted(missing_art)}, "
            f"but the available artifacts are only {sorted(available) or 'none'} (missing upstream producer)"
        ]
    return []


# One pure validator per assembly rule; the parent only derives the facts and
# concatenates what the rules report (empty list = rule passes).
_VALIDATION_RULES = (
    _check_semantics,
    _check_objectives,
    _check_resources,
    _check_artifacts,
)


def validate_assembly(
    cap: CapSpec,
    task: TaskCap,
    eval_models: list[EvalModelCap],
    backend: BackendLimits,
    available_artifacts: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the list of assembly violations (empty = valid assembly). Does
    not raise; the caller aggregates."""
    produced: set[str] = set()
    available_semantics: set[str] = set()
    for model in eval_models:
        produced |= model.produces
        available_semantics |= model.semantics
    facts = _AssemblyFacts(
        task=task,
        backend=backend,
        produced=frozenset(produced),
        semantics=frozenset(available_semantics),
        available_artifacts=frozenset(available_artifacts),
    )
    return [violation for rule in _VALIDATION_RULES for violation in rule(cap, facts)]
