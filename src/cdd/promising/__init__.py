"""Customer-order promising: a second downstream decision on the same compiled state.

See docs/research/icores-revision-scope.md, Section 5. Physical supply is fixed to the
replenishment planner's gold-state orders; readers differ only in the ŝ the promise decision
reads.
"""

from cdd.promising.execution import OrderOutcome, PromisingResult, PromisingSimulator
from cdd.promising.orders import CustomerOrder, generate_orders
from cdd.promising.params import PromisingParams, margins, unit_rates
from cdd.promising.policy import Commitment, PromiseDecision, promise
from cdd.promising.supply import FixedPlanDecider, FixedSupply, reference_supply

__all__ = [
    "Commitment",
    "CustomerOrder",
    "FixedPlanDecider",
    "FixedSupply",
    "OrderOutcome",
    "PromiseDecision",
    "PromisingParams",
    "PromisingResult",
    "PromisingSimulator",
    "generate_orders",
    "margins",
    "promise",
    "reference_supply",
    "unit_rates",
]
