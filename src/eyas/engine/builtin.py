"""import eyas.engine.builtin triggers registration of the engine built-in components
(strategies / eval models / budgets / pairwise judges)."""

from __future__ import annotations

from eyas.engine import judges as _judges  # noqa: F401 — triggers DefaultPairwiseJudge registration
from eyas.engine.budget import BanditBudget, EvenSplitBudget, FlatBudget
from eyas.engine.evaluation import AbsoluteEvalModel, CompareEvalModel
from eyas.engine.registry import register_budget, register_eval_model, registered
from eyas.engine.strategies import ape, cot, evoprompt, hill_climb, opro, protegi, raw, spo

register_eval_model("absolute")(AbsoluteEvalModel)
register_eval_model("compare")(CompareEvalModel)
register_budget("flat")(FlatBudget)
register_budget("even_split")(EvenSplitBudget)
register_budget("bandit")(BanditBudget)

STRATEGIES = (raw, cot, hill_climb, evoprompt, opro, ape, protegi, spo)
__all__ = ["BanditBudget", "EvenSplitBudget", "FlatBudget", "STRATEGIES", "registered"]
