"""The weekly promise decision (scope 5.6): accept or reject each new customer order and choose
its promised week, against the supply ŝ projects (available-to-promise, D1).

Projected supply for SKU k from week t to ``end_period``: free stock on hand, lots that ŝ shows
arrived and released in a week (a held lot is frozen, as in the planner), and open replenishment
orders at the arrival ŝ's lead time implies (``solver._expected_arrivals``). Existing
commitments are fixed requirements at their promised weeks; when the projection cannot cover
them, a penalised slack keeps the model solvable and the shortfall is reported as at risk.
Promises are binding and never revised (D5).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cdd.orders import InventoryPosition
from cdd.promising.orders import CustomerOrder
from cdd.promising.params import PromisingParams
from cdd.schema import Attribute, HoldStatus
from cdd.solver import _expected_arrivals, _Model
from cdd.state import StateView
from cdd.world import Lot

_EPS = 1e-6
# Relative optimality gap for the promise MILP: tight, so that near-ties between orders are
# decided by the objective and the tie-breaks, not by where the solver stopped.
_GAP = 1e-9


@dataclass(frozen=True)
class Commitment:
    order: CustomerOrder
    promised: int
    remaining: float


@dataclass(frozen=True)
class PromiseDecision:
    # New order id -> promised week, or None when rejected.
    promised: Mapping[str, int | None]
    status: str
    # True when the solver gave no solution and every new order of the week was rejected.
    fallback: bool = False
    # Units of existing commitments the projection cannot cover.
    at_risk: float = 0.0


def promise(
    view: StateView,
    position: InventoryPosition,
    commitments: Sequence[Commitment],
    new_orders: Sequence[CustomerOrder],
    rates: Mapping[str, Mapping[str, float]],
    params: PromisingParams,
    hold_notices: Mapping[str, int] | None = None,
) -> PromiseDecision:
    """Promise this week's new orders. ``rates`` holds per-SKU ``margin`` and ``lateness``.

    A new order may be promised for any week from its requested week (or now, if later) to
    ``max_late_weeks`` after it, never after ``end_period``; a later week costs lateness per
    unit-week, so an order is promised as early as projected supply allows. ``hold_notices``
    (lot -> week a failed delivery revealed its hold) override ŝ's hold status until ŝ records
    a hold-status delta for that lot after the notice (the hold-notice sensitivity, D10).
    """
    t = position.period
    end = view.world.end_period
    if not new_orders:
        return PromiseDecision({}, "no_orders")
    periods = list(range(t, end + 1))
    model = _Model()

    # Existing commitments: requirements at their promised week, overdue ones now.
    due: dict[tuple[str, int], float] = defaultdict(float)
    for commitment in commitments:
        if commitment.remaining > _EPS:
            due[(commitment.order.sku, max(commitment.promised, t))] += commitment.remaining
    largest = max((rates[order.sku]["margin"] for order in new_orders), default=1.0)
    slack_cost = 100.0 * largest + 1.0

    # New orders: one binary per candidate week; at most one week per order.
    choices: dict[tuple[str, int], int] = {}
    requirement: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for index, order in enumerate(new_orders):
        rate = rates[order.sku]
        weeks = range(
            max(order.requested, t), min(order.requested + params.max_late_weeks, end) + 1
        )
        picks = []
        for week in weeks:
            value = (rate["margin"] - rate["lateness"] * (week - order.requested)) * order.quantity
            # Tie-breaks: earlier orders, then earlier weeks.
            cost = -value + 1e-6 * (index + 1) + 1e-7 * (week - t)
            var = model.var(f"y[{order.id},{week}]", cost=cost, upper=1.0, integer=True)
            choices[(order.id, week)] = var
            requirement[(order.sku, week)][var] = order.quantity
            picks.append(var)
        if picks:
            model.row(dict.fromkeys(picks, 1.0), 0.0, 1.0, f"one_week[{order.id}]")

    # Supply per SKU and week: free stock carried forward, lots drawn only when usable.
    arrivals = _expected_arrivals(view, position, periods)
    draws: dict[tuple[str, int], list[int]] = defaultdict(list)
    for lot in view.world.lots:
        remaining = position.lot_remaining.get(lot.id, 0.0)
        if remaining <= _EPS:
            continue
        mine = []
        for week in periods:
            if _usable(view, lot, week, hold_notices):
                var = model.var(f"draw[{lot.id},{week}]", upper=remaining)
                draws[(lot.sku, week)].append(var)
                mine.append(var)
        if mine:
            model.row(dict.fromkeys(mine, 1.0), 0.0, remaining, f"lot[{lot.id}]")

    slacks: list[int] = []
    for sku in view.world.skus:
        previous: int | None = None
        for week in periods:
            stock = model.var(f"stock[{sku.id},{week}]")
            use = model.var(f"use[{sku.id},{week}]")
            inflow = arrivals.get((sku.id, week), 0.0)
            if previous is None:
                inflow += position.on_hand.get(sku.id, 0.0)
                model.row({stock: 1.0, use: 1.0}, inflow, inflow, f"free[{sku.id},{week}]")
            else:
                model.row(
                    {stock: 1.0, previous: -1.0, use: 1.0}, inflow, inflow, f"free[{sku.id},{week}]"
                )
            previous = stock
            existing = due.get((sku.id, week), 0.0)
            deliver: dict[int, float] = {use: 1.0}
            deliver.update(dict.fromkeys(draws.get((sku.id, week), ()), 1.0))
            if existing > _EPS:
                slack = model.var(f"slack[{sku.id},{week}]", cost=slack_cost, upper=existing)
                deliver[slack] = 1.0
                slacks.append(slack)
            for var, quantity in requirement.get((sku.id, week), {}).items():
                deliver[var] = -quantity
            model.row(deliver, existing, existing, f"deliver[{sku.id},{week}]")

    solution, _, status = model.solve(params.time_limit, mip_rel_gap=_GAP)
    if solution is None:
        return PromiseDecision(dict.fromkeys((o.id for o in new_orders), None), status, True)
    promised: dict[str, int | None] = {}
    for order in new_orders:
        chosen = [w for (oid, w), var in choices.items() if oid == order.id and solution[var] > 0.5]
        promised[order.id] = chosen[0] if chosen else None
    at_risk = float(sum(solution[var] for var in slacks))
    return PromiseDecision(promised, status, False, at_risk if at_risk > _EPS else 0.0)


def _usable(view: StateView, lot: Lot, week: int, hold_notices: Mapping[str, int] | None) -> bool:
    """Whether the projection may draw on ``lot`` in ``week``: arrived, and released in ŝ unless
    a hold notice newer than ŝ's latest hold-status record says otherwise."""
    if not lot.arrived(week):
        return False
    notice = (hold_notices or {}).get(lot.id)
    if notice is not None:
        recorded = view.last_recorded(lot.id, Attribute.HOLD_STATUS)
        if recorded is None or recorded <= notice:
            return False
    return view.label(lot.id, Attribute.HOLD_STATUS, week) == HoldStatus.RELEASED
