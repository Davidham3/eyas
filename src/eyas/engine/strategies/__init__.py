"""Built-in strategies: importing this package triggers registration.

TODO(WIREUP): temporary trimmed import (6 of 8 algorithms); restore the baseline
full-8-algorithm import (add protegi, evoprompt) at Task 9.4.
"""

from __future__ import annotations

from eyas.engine.strategies import ape, cot, hill_climb, opro, raw, spo

__all__ = ["ape", "cot", "hill_climb", "opro", "raw", "spo"]
