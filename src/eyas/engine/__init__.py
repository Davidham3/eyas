"""Eyas core runtime (contract-driven).

Sits above the stable lower layers (backends / tasks / judges / registry /
logging / watchdog). The abstraction unit is: Strategy (one-step decision) +
BudgetAllocator (stateful budget) + EvalModel (pluggable absolute/compare
semantics) + typed OptimizerState, all with capability declarations;
compatibility is validated at assembly time.
See docs/10-architecture.md.
"""
