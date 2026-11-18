"""Built-in strategies: importing this package triggers registration (all 8 algorithms)."""

from __future__ import annotations

from eyas.engine.strategies import ape, cot, evoprompt, hill_climb, opro, protegi, raw, spo

__all__ = ["ape", "cot", "evoprompt", "hill_climb", "opro", "protegi", "raw", "spo"]
