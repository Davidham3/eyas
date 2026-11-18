"""Built-in strategies: importing this package triggers registration.

TODO(WIREUP): temporary trimmed import (ape, hill_climb, opro, raw only);
restore the baseline full-8-algorithm import at Task 9.4.
"""

from __future__ import annotations

from eyas.engine.strategies import ape, hill_climb, opro, raw

__all__ = ["ape", "hill_climb", "opro", "raw"]
