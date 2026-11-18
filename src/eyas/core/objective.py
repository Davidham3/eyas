from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    """One optimization objective. key corresponds to a key in the evaluated
    sample's metrics dict."""

    key: str
    maximize: bool = True
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            object.__setattr__(self, "display_name", self.key)

    def oriented(self, value: float) -> float:
        """Reorient into the "bigger is better" view for convenient sorting."""
        return value if self.maximize else -value
