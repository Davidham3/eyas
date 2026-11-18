"""Built-in strategies: importing this package triggers registration.

TODO(WIREUP): temporary trimmed import (raw + hill_climb only); restore the
baseline full-8-algorithm import at Task 9.4.
"""

from __future__ import annotations

from eyas.engine.strategies import hill_climb, raw

__all__ = ["hill_climb", "raw"]
