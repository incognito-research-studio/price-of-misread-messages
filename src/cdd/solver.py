"""Deterministic solver (L5): a rolling-horizon MILP over the interpreted state.

Each period the solver chooses order quantities ``x[p, k, τ]`` for every orderable (plant, SKU)
line over a short horizon and executes the first period. The model encodes capacity, supplier
status (including status notices from refused orders), quality holds (through usable stock),
minimum order quantities and a per-period budget, and minimises purchase, holding and backlog
cost. The solver reads only the interpreted state;
the interpreter never sees this objective.

Duals come from the LP obtained by fixing the integer variables at their MILP values. They are
exact local sensitivities only for right-hand-side changes inside the optimal basis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_array, vstack

from cdd.orders import InventoryPosition, OrderLine, Plan, capacity_available, lead_periods
from cdd.schema import Attribute, HoldStatus
from cdd.state import StateView

_EPS = 1e-6

DemandFn = Callable[[str, int], float]
_Row = tuple[dict[int, float], float, float, str]


class _Model:
    """A small sparse LP/MILP builder on top of scipy's HiGHS interface."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.cost: list[float] = []
        self.upper: list[float] = []
        self.integer: list[int] = []
        self.rows: list[_Row] = []

    def var(
        self, name: str, *, cost: float = 0.0, upper: float = math.inf, integer: bool = False
    ) -> int:
        self.names.append(name)
        self.cost.append(cost)
        self.upper.append(upper)
        self.integer.append(int(integer))
        return len(self.names) - 1

    def row(self, coeffs: Mapping[int, float], lower: float, upper: float, name: str) -> None:
        self.rows.append((dict(coeffs), lower, upper, name))

    def _matrix(self, rows: Sequence[_Row], sign: float = 1.0) -> csr_array:
        data: list[float] = []
        row_idx: list[int] = []
        col_idx: list[int] = []
        for i, (coeffs, _, _, _) in enumerate(rows):
            for j, value in coeffs.items():
                data.append(sign * value)
                row_idx.append(i)
                col_idx.append(j)
        return csr_array((data, (row_idx, col_idx)), shape=(len(rows), len(self.names)))

    def solve(
        self, time_limit: float, mip_rel_gap: float | None = None
    ) -> tuple[np.ndarray | None, float | None, str]:
        """``mip_rel_gap`` overrides HiGHS's relative optimality gap (default 1e-4)."""
        options: dict[str, Any] = {"time_limit": time_limit, "disp": False}
        if mip_rel_gap is not None:
            options["mip_rel_gap"] = mip_rel_gap
        constraints = LinearConstraint(
            self._matrix(self.rows), [r[1] for r in self.rows], [r[2] for r in self.rows]
        )
        result = milp(
            c=np.asarray(self.cost),
            integrality=np.asarray(self.integer),
            bounds=Bounds(np.zeros(len(self.names)), np.asarray(self.upper)),
            constraints=constraints,
            options=options,
        )
        status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(
            result.status, "error"
        )
        if result.x is None:
            return None, None, status
        return np.asarray(result.x), float(result.fun), status

    def duals(self, solution: np.ndarray) -> dict[str, float]:
        """Shadow prices of the LP with every integer variable fixed at its MILP value."""
        upper: list[float | None] = [None if math.isinf(u) else u for u in self.upper]
        lower: list[float] = [0.0] * len(self.names)
        for j, is_integer in enumerate(self.integer):
            if is_integer:
                lower[j] = float(round(solution[j]))
                upper[j] = lower[j]
        eq = [r for r in self.rows if r[1] == r[2]]
        le = [r for r in self.rows if r[1] != r[2] and math.isfinite(r[2])]
        ge = [r for r in self.rows if r[1] != r[2] and math.isfinite(r[1])]
        inequalities = le + ge
        result = linprog(
            c=np.asarray(self.cost),
            A_ub=self._stack(le, ge) if inequalities else None,
            b_ub=[r[2] for r in le] + [-r[1] for r in ge] if inequalities else None,
            A_eq=self._matrix(eq) if eq else None,
            b_eq=[r[1] for r in eq] if eq else None,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        if result.status != 0:
            return {}
        duals: dict[str, float] = {}
        for i, r in enumerate(eq):
            duals[r[3]] = float(result.eqlin.marginals[i])
        for i, r in enumerate(le):
            duals[r[3]] = float(result.ineqlin.marginals[i])
        for i, r in enumerate(ge):
            duals[r[3]] = -float(result.ineqlin.marginals[len(le) + i])
        return duals

    def _stack(self, le: Sequence[_Row], ge: Sequence[_Row]) -> csr_array:
        return csr_array(vstack([self._matrix(le), self._matrix(ge, sign=-1.0)]))


@dataclass(frozen=True)
class Schedule:
    orders: Mapping[int, tuple[OrderLine, ...]]
    objective: float | None
    status: str
    duals: Mapping[str, float] | None = None


@dataclass(frozen=True)
class MILPSolver:
    """Rolling-horizon MILP. ``safety_factor`` inflates the demand forecast it plans on
    (0.1 plans for 10% more than forecast); the tuned benchmark x* searches over it."""

    horizon: int = 6
    time_limit: float = 10.0
    compute_duals: bool = False
    safety_factor: float = 0.0
    name: str = "milp"

    def plan(self, view: StateView, position: InventoryPosition) -> Plan:
        t = position.period
        periods = [p for p in range(t, t + self.horizon) if p <= view.world.end_period]
        if not periods:
            return Plan(period=t, status="beyond_horizon")
        schedule = solve_orders(
            view,
            position,
            periods,
            demand=lambda sku, p: view.number(sku, Attribute.DEMAND, p) * (1 + self.safety_factor),
            time_limit=self.time_limit,
            compute_duals=self.compute_duals,
        )
        return Plan(
            period=t,
            orders=schedule.orders.get(t, ()),
            objective=schedule.objective,
            status=schedule.status,
            duals=schedule.duals,
        )


def hindsight_plans(
    view: StateView,
    position: InventoryPosition,
    demand: Mapping[tuple[str, int], float],
    *,
    time_limit: float = 60.0,
    yields: Mapping[tuple[str, int], float] | None = None,
) -> dict[int, Plan]:
    """Perfect-information benchmark: one MILP over the whole horizon, solved with the true
    parameters, the realised demand path and any out-of-schema yield shortfalls. Pass a view
    of the oracle state with ``as_of=None``."""
    periods = list(view.world.periods)
    schedule = solve_orders(
        view,
        position,
        periods,
        demand=lambda sku, p: demand[(sku, p)],
        time_limit=time_limit,
        yields=yields,
    )
    return {
        p: Plan(period=p, orders=schedule.orders.get(p, ()), status=schedule.status)
        for p in periods
    }


def solve_orders(
    view: StateView,
    position: InventoryPosition,
    periods: Sequence[int],
    demand: DemandFn,
    *,
    time_limit: float,
    compute_duals: bool = False,
    yields: Mapping[tuple[str, int], float] | None = None,
) -> Schedule:
    world = view.world
    shares = yields or {}
    model = _Model()
    period_set = set(periods)
    x: dict[tuple[str, str, int], int] = {}
    arriving: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)

    for plant_id, sku_id in world.lines():
        offer = world.offer_for(plant_id, sku_id)
        if offer is None:  # pragma: no cover - lines() only returns lines with offers
            continue
        for p in periods:
            cap = capacity_available(view, plant_id, p, position.status_notices)
            price = view.number(offer.id, Attribute.PRICE, p)
            j = model.var(f"x[{plant_id},{sku_id},{p}]", cost=price, upper=cap)
            x[(plant_id, sku_id, p)] = j
            arrival = p + lead_periods(view, plant_id, p)
            if arrival in period_set:
                arriving[(sku_id, arrival)].append((j, shares.get((plant_id, p), 1.0)))
            if offer.moq > 0 and cap > 0:
                on = model.var(f"y[{plant_id},{sku_id},{p}]", upper=1.0, integer=True)
                key = f"{plant_id},{sku_id},{p}"
                model.row({j: 1.0, on: -cap}, -math.inf, 0.0, f"moq_on[{key}]")
                model.row({j: 1.0, on: -offer.moq}, 0.0, math.inf, f"moq_min[{key}]")

    for plant in world.plants:
        for p in periods:
            coeffs = {x[(plant.id, k, p)]: 1.0 for k in plant.skus if (plant.id, k, p) in x}
            if coeffs:
                cap = capacity_available(view, plant.id, p, position.status_notices)
                model.row(coeffs, -math.inf, cap, f"capacity[{plant.id},{p}]")

    if world.budget_per_period is not None:
        for p in periods:
            spend = {j: model.cost[j] for (_, _, q), j in x.items() if q == p}
            if spend:
                model.row(spend, -math.inf, world.budget_per_period, f"budget[{p}]")

    draws = _lot_draws(model, view, position, periods)
    inflow = _expected_arrivals(view, position, periods)
    for sku in world.skus:
        net_start = position.on_hand.get(sku.id, 0.0) - position.backlog.get(sku.id, 0.0)
        previous: tuple[int, int] | None = None
        for p in periods:
            inv = model.var(f"inventory[{sku.id},{p}]", cost=sku.holding_cost)
            back = model.var(f"backlog[{sku.id},{p}]", cost=sku.backlog_cost)
            balance: dict[int, float] = {inv: 1.0, back: -1.0}
            for j, share in arriving[(sku.id, p)]:
                balance[j] = balance.get(j, 0.0) - share
            for j in draws[(sku.id, p)]:
                balance[j] = -1.0
            need = demand(sku.id, p)
            rhs = inflow.get((sku.id, p), 0.0) - need
            if previous is None:
                rhs += net_start
                need += position.backlog.get(sku.id, 0.0)
            else:
                balance[previous[0]] = -1.0
                balance[previous[1]] = 1.0
            model.row(balance, rhs, rhs, f"balance[{sku.id},{p}]")
            if draws[(sku.id, p)]:
                # Lot stock only serves this period's demand and backlog, as in the simulator.
                cap_draw = dict.fromkeys(draws[(sku.id, p)], 1.0)
                if previous is not None:
                    cap_draw[previous[1]] = -1.0
                model.row(cap_draw, -math.inf, need, f"lot_draw[{sku.id},{p}]")
            previous = (inv, back)

    solution, objective, status = model.solve(time_limit)
    if solution is None:
        return Schedule(orders={}, objective=None, status=status)
    orders: dict[int, list[OrderLine]] = defaultdict(list)
    for (plant_id, sku_id, p), j in sorted(x.items()):
        quantity = float(solution[j])
        if quantity > _EPS:
            orders[p].append(OrderLine(plant_id, sku_id, round(quantity, 6)))
    duals = model.duals(solution) if compute_duals else None
    return Schedule({p: tuple(lines) for p, lines in orders.items()}, objective, status, duals)


def _released(view: StateView, lot_id: str, period: int) -> bool:
    return view.label(lot_id, Attribute.HOLD_STATUS, period) == HoldStatus.RELEASED


def _expected_arrivals(
    view: StateView, position: InventoryPosition, periods: Sequence[int]
) -> dict[tuple[str, int], float]:
    """Open orders, expected after the lead time in force when they were placed (per ŝ)."""
    first, last = periods[0], periods[-1]
    inflow: dict[tuple[str, int], float] = defaultdict(float)
    for order in position.outstanding:
        expected = max(first, order.placed + lead_periods(view, order.plant, order.placed))
        if expected <= last:
            inflow[(order.sku, expected)] += order.quantity
    return inflow


def _lot_draws(
    model: _Model, view: StateView, position: InventoryPosition, periods: Sequence[int]
) -> dict[tuple[str, int], list[int]]:
    """Per-lot stock and draw variables. A lot can be drawn only in periods where it has
    arrived and ŝ shows it released, so a hold freezes whatever is left of it, as it does in the
    simulator. Stock is charged holding cost only once it has arrived."""
    draws: dict[tuple[str, int], list[int]] = defaultdict(list)
    for lot in view.world.lots:
        remaining = position.lot_remaining.get(lot.id, 0.0)
        if remaining <= _EPS:
            continue
        holding = view.world.sku(lot.sku).holding_cost
        previous: int | None = None
        for p in periods:
            stock = model.var(f"lot_stock[{lot.id},{p}]", cost=holding if lot.arrived(p) else 0.0)
            flow: dict[int, float] = {stock: 1.0}
            if lot.arrived(p) and _released(view, lot.id, p):
                draw = model.var(f"lot_draw[{lot.id},{p}]", upper=remaining)
                draws[(lot.sku, p)].append(draw)
                flow[draw] = 1.0
            if previous is not None:
                flow[previous] = -1.0
            rhs = remaining if previous is None else 0.0
            model.row(flow, rhs, rhs, f"lot_balance[{lot.id},{p}]")
            previous = stock
    return draws
