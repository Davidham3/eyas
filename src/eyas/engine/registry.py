"""Registry for the strategy / eval_model / budget / pairwise_judge
extension points, separate from the task / backend / judge / protocol registry.

Duplicate names are an error (fail fast).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_GROUPS = ("strategy", "eval_model", "budget", "pairwise_judge")
_REGISTRY: dict[str, dict[str, type]] = {group: {} for group in _GROUPS}


def _register(group: str, name: str, cls: type) -> None:
    if group not in _GROUPS:
        raise ValueError(
            f"unknown engine registry group: {group!r} (options: {', '.join(_GROUPS)})"
        )
    bucket = _REGISTRY[group]
    existing = bucket.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"duplicate registration: engine/{group}/{name} (existing: {existing.__module__}.{existing.__qualname__})"
        )
    bucket[name] = cls


def _registrar(kind: str, *, sets_name: bool) -> Callable[[str], Callable[[type], type]]:
    """Build the registrar for one extension-point group.

    ``register(name)`` returns a class decorator that records the class under
    ``kind``; ``sets_name`` controls whether ``cls.name`` is assigned (strategy
    and pairwise_judge do, eval_model and budget do not).
    """

    def register(name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            if sets_name:
                cls.name = name  # type: ignore[attr-defined]
            _register(kind, name, cls)
            return cls

        return decorator

    return register


register_strategy = _registrar("strategy", sets_name=True)
register_eval_model = _registrar("eval_model", sets_name=False)
register_budget = _registrar("budget", sets_name=False)
register_pairwise_judge = _registrar("pairwise_judge", sets_name=True)


def get(group: str, name: str) -> type[Any]:
    if group not in _REGISTRY:
        raise KeyError(f"unknown engine group: {group!r} (options: {_GROUPS})")
    bucket = _REGISTRY[group]
    if name not in bucket:
        raise KeyError(f"unregistered {group}: {name!r} (registered: {sorted(bucket)})")
    return bucket[name]


def registered(group: str) -> list[str]:
    return sorted(_REGISTRY.get(group, {}))


def groups() -> tuple[str, ...]:
    return _GROUPS
