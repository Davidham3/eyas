from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from eyas.registry import register_protocol
from eyas.tasks.base import Sample, Variant

# register_protocol is re-exported so a custom-protocol author gets the base
# classes and the decorator from one module; the registry itself stays eyas.registry.
__all__ = ["IdentityProtocol", "VariantProtocol", "register_protocol"]


class VariantProtocol(ABC):
    """Evaluation protocol = variant enumerator + per-sample aggregation of
    metrics across variants.

    This is the extension point for variant-evaluation protocols; the shipped
    protocol is ``identity`` (one variant per sample). ``aggregate`` takes one
    sample's score dicts across variants (same schema as Task.score).
    Cross-sample averaging is the Evaluator's job.
    """

    name: ClassVar[str]

    @abstractmethod
    def variants(self, sample: Sample) -> list[Variant]: ...

    @abstractmethod
    def aggregate(self, per_variant: Sequence[dict[str, float]]) -> dict[str, float]: ...


@register_protocol("identity")
class IdentityProtocol(VariantProtocol):
    """No-variant protocol: one evaluation per sample; cross-variant
    aggregation is just the mean."""

    name = "identity"

    def variants(self, sample: Sample) -> list[Variant]:
        return [Variant(id="identity")]

    def aggregate(self, per_variant: Sequence[dict[str, float]]) -> dict[str, float]:
        if not per_variant:
            return {}
        keys = {k for d in per_variant for k in d}
        n = len(per_variant)
        return {k: sum(d.get(k, 0.0) for d in per_variant) / n for k in keys}
