from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from eyas.backends.base import ChatRequest, ChatResponse, Part
from eyas.core.objective import Objective


@dataclass(frozen=True)
class Sample:
    """A single evaluation sample. inputs / references are open mappings,
    interpreted by the concrete Task."""

    id: str
    inputs: Mapping[str, Any]  # {"question": ..., "choices": [...], "image": path, ...}
    references: Mapping[str, Any]  # {"label": ...}


@dataclass(frozen=True)
class Variant:
    """A variant enumerated by the evaluation protocol. The id follows the
    protocol's convention (e.g. "identity" / "shift_2")."""

    id: str
    index: int = 0


@dataclass
class Prediction:
    """Result of the Task's deterministic parsing. value=None means parsing
    failed; the judge takes over as fallback."""

    value: Any | None
    raw: str


class Task(ABC):
    """Task abstraction: data loading + request building + response parsing + scoring.

    The Task is unaware of inference backends and caching; the evaluation
    protocol (variant enumeration) is handled by VariantProtocol, while
    Task.build_request only renders (prompt, sample, variant) into a request.
    """

    name: ClassVar[str]
    objectives: ClassVar[tuple[Objective, ...]] = ()
    # Not a ClassVar: tasks may override it per instance via constructor args (e.g. a per-instance category template)
    default_prompt: str = "Answer the question about the image."

    @abstractmethod
    def load_split(self, split: str) -> Sequence[Sample]: ...

    @abstractmethod
    def build_request(self, prompt: str, sample: Sample, variant: Variant) -> ChatRequest: ...

    @abstractmethod
    def parse_response(self, response: ChatResponse, sample: Sample) -> Prediction: ...

    @abstractmethod
    def score(self, prediction: Prediction, sample: Sample) -> dict[str, float]: ...

    def demo_media_parts(self, sample: Sample) -> Sequence[Part]:
        """Media a few-shot demo may attach for this sample (few-shot induction)."""
        return ()
