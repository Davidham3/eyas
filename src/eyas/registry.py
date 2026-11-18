"""Centralized component registries.

Extension points: task / backend / judge / protocol (engine-side strategy / eval_model /
budget / pairwise_judge live in eyas.engine.registry). Registering a DIFFERENT class under
an already-registered name raises immediately (fail fast); re-registering the SAME class
object is intentional idempotency (see register). Third-party packages provide plugin
modules via the entry_points group ``eyas.plugins``, loaded on first get().
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

_GROUPS = (
    "task",
    "backend",
    "judge",
    "protocol",
)

_REGISTRY: dict[str, dict[str, type]] = {group: {} for group in _GROUPS}
_plugins_loaded = False


def register(group: str, name: str) -> Callable[[type], type]:
    """Build the ``@register_<group>(name)`` decorator.

    Duplicate-name semantics: a DIFFERENT class under an already-registered
    name raises ValueError immediately (fail fast — two components must never
    silently share one name). Re-registering the SAME class object under the
    same name is intentional idempotency, not an error: the entry is
    re-assigned and ``cls.name`` re-stamped, so a module imported/registered
    twice (plugin reload, re-executed registration cell) is a no-op rather
    than a crash. The identity check is ``existing is not cls`` — equality or
    subclass relationships do not qualify.
    """
    group = group.lower()
    if group not in _GROUPS:
        raise ValueError(f"unknown registry group: {group!r} (valid: {', '.join(_GROUPS)})")

    def decorator(cls: type) -> type:
        bucket = _REGISTRY[group]
        existing = bucket.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"duplicate registration: {group}/{name} (already registered: {existing.__module__}.{existing.__qualname__})"
            )
        bucket[name] = cls
        cls.name = name  # type: ignore[attr-defined]  # component name overwritten by decorator
        return cls

    return decorator


def register_task(name: str) -> Callable[[type], type]:
    return register("task", name)


def register_backend(name: str) -> Callable[[type], type]:
    return register("backend", name)


def register_judge(name: str) -> Callable[[type], type]:
    return register("judge", name)


def register_protocol(name: str) -> Callable[[type], type]:
    return register("protocol", name)


def get(group: str, name: str) -> type:
    _load_plugins()
    bucket = _REGISTRY.get(group, {})
    if name not in bucket:
        raise KeyError(f"unregistered {group}: {name!r} (registered: {sorted(bucket)})")
    return bucket[name]


def registered(group: str) -> list[str]:
    _load_plugins()
    return sorted(_REGISTRY.get(group, {}))


def groups() -> tuple[str, ...]:
    return _GROUPS


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    for ep in entry_points(group="eyas.plugins"):
        ep.load()  # let failures propagate: a broken plugin should break early (fail fast)
    # latch only after a fully successful load: a failed load must raise again on
    # the next get(), never silently skip the broken plugin for the whole process
    _plugins_loaded = True
