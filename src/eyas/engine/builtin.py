"""import eyas.engine.builtin triggers registration of the engine built-in components
(strategies / eval models / budgets / pairwise judges)."""

from __future__ import annotations

# Defensive import: loading this module must register both pairwise judges
# regardless of what evaluation's own import chain happens to pull in.
from eyas.engine import judges as _judges
from eyas.engine.budget import BanditBudget, EvenSplitBudget, FlatBudget
from eyas.engine.evaluation import AbsoluteEvalModel, CompareEvalModel
from eyas.engine.registry import register_budget, register_eval_model
from eyas.engine.strategies import ape, cot, evoprompt, hill_climb, opro, protegi, raw, spo

register_eval_model("absolute")(AbsoluteEvalModel)
register_eval_model("compare")(CompareEvalModel)
register_budget("flat")(FlatBudget)
register_budget("even_split")(EvenSplitBudget)
register_budget("bandit")(BanditBudget)

STRATEGIES = (raw, cot, hill_climb, evoprompt, opro, ape, protegi, spo)
#: the module's own product (the published strategy tuple); budgets and eval
#: models are importable from their home modules, not re-exported here.
__all__ = ["STRATEGIES"]
