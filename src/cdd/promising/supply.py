"""Fixed physical supply for the promising study (scope 5.2).

Every promising condition on a stream gets the same supply: the order lines the replenishment
planner placed when it read the gold deltas (the run behind ``StreamContext.planner_cost``). A
promising reader's misreading therefore cannot change what arrives; it changes only which
customer orders are accepted and when they are promised. This is an experimental control, not a
deployable policy: the reference planner reads every message correctly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cdd.interpreters.replay import OracleInterpreter
from cdd.orders import InventoryPosition, OrderLine, Plan
from cdd.simulator import SimulationResult
from cdd.solver import MILPSolver
from cdd.state import StateView

if TYPE_CHECKING:
    from cdd.experiments import StreamContext


@dataclass(frozen=True)
class FixedSupply:
    # The lines the reference planner placed each week, and the lines the true plants accepted.
    plans: Mapping[int, Plan]
    accepted: Mapping[int, tuple[OrderLine, ...]]
    # The reference run's replenishment cost; replaying ``plans`` must reproduce it.
    cost: float

    @classmethod
    def from_result(cls, result: SimulationResult) -> FixedSupply:
        return cls(
            plans={r.period: Plan(period=r.period, orders=r.ordered) for r in result.periods},
            accepted={r.period: r.accepted for r in result.periods},
            cost=result.total_cost,
        )


class FixedPlanDecider:
    """Places the fixed lines whatever ŝ says. It lets the unchanged compiling front end
    (parser, SOP, update) run each week while supply stays fixed."""

    name = "fixed-supply"

    def __init__(self, plans: Mapping[int, Plan]) -> None:
        self._plans = plans

    def plan(self, view: StateView, position: InventoryPosition) -> Plan:
        return self._plans.get(position.period, Plan(period=position.period))


def reference_supply(context: StreamContext) -> FixedSupply:
    """The configured planner on the gold deltas, through the same front end as every reader."""
    solver = context.config.solver
    planner = MILPSolver(
        horizon=solver.horizon, time_limit=solver.time_limit, safety_factor=solver.safety_factor
    )
    policy = context.compiling(OracleInterpreter(context.stream.gold), planner)
    return FixedSupply.from_result(context.simulator.run(policy))
