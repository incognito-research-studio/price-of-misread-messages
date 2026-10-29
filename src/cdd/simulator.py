"""Inventory simulator: executes plans against the true state and scores them with cost ``C``.

The simulator owns the physics: true capacities, supplier status, lead times, holds, prices and
realised demand. A policy sees only an :class:`~cdd.orders.InventoryPosition` each period.
Demand noise is drawn once per stream seed (common random numbers), so every condition faces
the same demand path.

v0.1 models one distribution centre supplied by several plants (single echelon, several
sources). Multi-echelon networks and an adapter for gym-invmgmt are planned.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from cdd.orders import (
    InventoryPosition,
    OrderLine,
    OutstandingOrder,
    Plan,
    capacity_available,
    lead_periods,
)
from cdd.schema import Attribute, HoldStatus, SupplierStatus
from cdd.state import StateView

_EPS = 1e-6

Policy = Callable[[InventoryPosition], Plan]


@dataclass(frozen=True)
class Violation:
    """An executed order that broke a true constraint (a constraint of ``X(s*)``)."""

    period: int
    kind: str
    entity: str
    amount: float


@dataclass(frozen=True)
class PeriodRecord:
    period: int
    plan_status: str
    ordered: tuple[OrderLine, ...]
    accepted: tuple[OrderLine, ...]
    demand: Mapping[str, float]
    filled: Mapping[str, float]
    backlog: Mapping[str, float]
    purchase_cost: float
    holding_cost: float
    backlog_cost: float
    invalid_lines: int = 0


@dataclass(frozen=True)
class SimulationResult:
    periods: tuple[PeriodRecord, ...]
    violations: tuple[Violation, ...]

    @property
    def purchase_cost(self) -> float:
        return sum(record.purchase_cost for record in self.periods)

    @property
    def holding_cost(self) -> float:
        return sum(record.holding_cost for record in self.periods)

    @property
    def backlog_cost(self) -> float:
        return sum(record.backlog_cost for record in self.periods)

    @property
    def invalid_lines(self) -> int:
        """Order lines a model decider wrote for a plant that does not make the SKU."""
        return sum(record.invalid_lines for record in self.periods)

    @property
    def total_cost(self) -> float:
        return self.purchase_cost + self.holding_cost + self.backlog_cost

    @property
    def fill_rate(self) -> float:
        """Share of each period's demand filled in that period."""
        demand = sum(sum(record.demand.values()) for record in self.periods)
        filled = sum(sum(record.filled.values()) for record in self.periods)
        return 1.0 if demand <= _EPS else filled / demand

    def order_vector(self) -> list[float]:
        """Ordered quantities in a fixed (period, plant, SKU) order, for consistency metrics."""
        quantities: dict[tuple[int, str, str], float] = defaultdict(float)
        for record in self.periods:
            for line in record.ordered:
                quantities[(record.period, line.plant, line.sku)] += line.quantity
        return [quantities[key] for key in sorted(quantities)]


class InventorySimulator:
    def __init__(
        self,
        truth: StateView,
        demand: Mapping[tuple[str, int], float],
        yields: Mapping[tuple[str, int], float] | None = None,
    ) -> None:
        """``yields`` holds out-of-schema shortfalls: the share of an order accepted by a
        plant in a period that actually arrives (1 when absent)."""
        self.truth = truth
        self.world = truth.world
        self.demand = demand
        self.yields = dict(yields or {})

    @staticmethod
    def sample_demand(truth: StateView, seed: int, cv: float = 0.2) -> dict[tuple[str, int], float]:
        """Realised demand: the true signal times mean-one lognormal noise.

        The draws depend only on ``seed`` and the world's shape, never on any condition, which
        gives common random numbers across conditions.
        """
        world = truth.world
        rng = np.random.default_rng(seed)
        sigma = math.sqrt(math.log1p(cv * cv))
        noise = rng.lognormal(-sigma * sigma / 2, sigma, size=(len(world.skus), len(world.periods)))
        return {
            (sku.id, period): max(0.0, truth.number(sku.id, Attribute.DEMAND, period))
            * float(noise[i, j])
            for i, sku in enumerate(world.skus)
            for j, period in enumerate(world.periods)
        }

    def initial_position(self) -> InventoryPosition:
        return InventoryPosition(
            period=self.world.start_period,
            on_hand={sku.id: sku.initial_stock for sku in self.world.skus},
            lot_remaining={lot.id: lot.quantity for lot in self.world.lots},
            backlog={sku.id: 0.0 for sku in self.world.skus},
        )

    def run(self, policy: Policy) -> SimulationResult:
        world, truth = self.world, self.truth
        start = self.initial_position()
        on_hand = dict(start.on_hand)
        lots = dict(start.lot_remaining)
        backlog = dict(start.backlog)
        # (arrival period, the order as the planner sees it, the quantity that will arrive)
        in_transit: list[tuple[int, OutstandingOrder, float]] = []
        notices: dict[str, tuple[int, str]] = {}
        records: list[PeriodRecord] = []
        violations: list[Violation] = []

        for t in world.periods:
            still_moving = []
            for arrival, order, quantity in in_transit:
                if arrival <= t:
                    on_hand[order.sku] += quantity
                else:
                    still_moving.append((arrival, order, quantity))
            in_transit = still_moving

            plan = policy(
                InventoryPosition(
                    period=t,
                    on_hand=dict(on_hand),
                    lot_remaining=dict(lots),
                    backlog=dict(backlog),
                    outstanding=tuple(order for _, order, _ in in_transit),
                    status_notices=dict(notices),
                )
            )

            accepted: list[OrderLine] = []
            used: dict[str, float] = defaultdict(float)
            purchase = 0.0
            for line in plan.orders:
                quantity = max(0.0, line.quantity)
                offer = world.offer_for(line.plant, line.sku)
                if quantity <= _EPS or offer is None:
                    continue
                supplier = world.plant(line.plant).supplier
                status = truth.label(supplier, Attribute.SUPPLIER_STATUS, t)
                room = max(0.0, capacity_available(truth, line.plant, t) - used[line.plant])
                taken = min(quantity, room)
                if taken < quantity - _EPS:
                    kind = "supplier_status" if status == SupplierStatus.SUSPENDED else "capacity"
                    violations.append(Violation(t, kind, line.plant, quantity - taken))
                    if status != SupplierStatus.ACTIVE:
                        notices[supplier] = (t, status)
                if taken <= _EPS:
                    continue
                used[line.plant] += taken
                purchase += truth.number(offer.id, Attribute.PRICE, t) * taken
                accepted.append(OrderLine(line.plant, line.sku, taken))
                arrival = t + lead_periods(truth, line.plant, t)
                delivered = taken * self.yields.get((line.plant, t), 1.0)
                if arrival <= t:
                    on_hand[line.sku] += delivered
                else:
                    # The planner sees the accepted quantity; a shortfall shows on arrival.
                    order = OutstandingOrder(t, line.plant, line.sku, taken)
                    in_transit.append((arrival, order, delivered))
            budget = world.budget_per_period
            if budget is not None and purchase > budget + _EPS:
                violations.append(Violation(t, "budget", "", purchase - budget))

            demand: dict[str, float] = {}
            filled: dict[str, float] = {}
            holding = backlog_cost = 0.0
            for sku in world.skus:
                usable_lots = [
                    lot.id
                    for lot in world.lots
                    if lot.sku == sku.id
                    and lot.arrived(t)
                    and lots[lot.id] > _EPS
                    and truth.label(lot.id, Attribute.HOLD_STATUS, t) == HoldStatus.RELEASED
                ]
                available = on_hand[sku.id] + sum(lots[lot_id] for lot_id in usable_lots)
                demand[sku.id] = self.demand[(sku.id, t)]
                need = backlog[sku.id] + demand[sku.id]
                served = min(available, need)
                filled[sku.id] = min(demand[sku.id], max(0.0, available - backlog[sku.id]))
                backlog[sku.id] = need - served

                # Released lots go first: a later hold would freeze whatever is left of them.
                remaining = served
                for lot_id in sorted(usable_lots):
                    take = min(lots[lot_id], remaining)
                    lots[lot_id] -= take
                    remaining -= take
                on_hand[sku.id] -= remaining

                stock = on_hand[sku.id] + sum(
                    lots[lot.id] for lot in world.lots if lot.sku == sku.id and lot.arrived(t)
                )
                holding += sku.holding_cost * stock
                backlog_cost += sku.backlog_cost * backlog[sku.id]

            records.append(
                PeriodRecord(
                    period=t,
                    plan_status=plan.status,
                    ordered=plan.orders,
                    accepted=tuple(accepted),
                    demand=demand,
                    filled=filled,
                    backlog=dict(backlog),
                    purchase_cost=purchase,
                    holding_cost=holding,
                    backlog_cost=backlog_cost,
                    invalid_lines=plan.invalid_lines,
                )
            )

        return SimulationResult(tuple(records), tuple(violations))
