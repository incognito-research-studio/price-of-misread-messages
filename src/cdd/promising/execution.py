"""True execution of customer promises on fixed supply (scope 5.3, 5.7, 5.8).

Each week t, in this order:

1. replenishment shipments whose true arrival is t or earlier enter free stock;
2. the compiling front end reads week t's messages (``CompilingPolicy`` with a
   :class:`~cdd.promising.supply.FixedPlanDecider`), which updates ŝ;
3. the fixed replenishment lines of week t are placed; the true plants' accepted lines join the
   open orders, arriving after the true lead time in force now;
4. customer orders arriving in week t are revealed and the promise decision runs on ŝ, the
   observed position (free stock, lot remainders, open orders) and the commitments;
5. commitments due by t are served from true usable stock (free stock plus arrived lots the truth
   shows released; released lots first, by id), by promised week, then arrival, then id. Partial
   deliveries are allowed and nothing is delivered before its promised week.

Realised value: Π = Σ margin x units delivered − lateness x unit-weeks after the requested week
− broken x units not delivered by their promised week − terminal x accepted units undelivered at
the end. Units never delivered count late through ``end_period``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cdd.orders import InventoryPosition, OutstandingOrder, lead_periods
from cdd.pipeline import CompilingPolicy
from cdd.promising.orders import CustomerOrder
from cdd.promising.params import PromisingParams, unit_rates
from cdd.promising.policy import Commitment, promise
from cdd.promising.supply import FixedSupply
from cdd.schema import Attribute, HoldStatus
from cdd.state import StateView

_EPS = 1e-6


@dataclass
class OrderOutcome:
    order: CustomerOrder
    promised: int | None = None
    deliveries: list[tuple[int, float]] = field(default_factory=list)

    @property
    def delivered(self) -> float:
        return sum(quantity for _, quantity in self.deliveries)


@dataclass(frozen=True)
class PromisingResult:
    outcomes: tuple[OrderOutcome, ...]
    value: float
    components: Mapping[str, float]
    metrics: dict[str, Any]


class PromisingSimulator:
    def __init__(
        self,
        truth: StateView,
        supply: FixedSupply,
        orders: Sequence[CustomerOrder],
        params: PromisingParams,
        margin: Mapping[str, float],
        yields: Mapping[tuple[str, int], float] | None = None,
    ) -> None:
        self.truth = truth
        self.world = truth.world
        self.supply = supply
        self.orders = tuple(orders)
        self.params = params
        self.margin = dict(margin)
        self.yields = dict(yields or {})

    def run(self, policy: CompilingPolicy) -> PromisingResult:
        """Run every week with ``policy`` as the compiling front end (its decider must place the
        fixed lines). The front end only updates ŝ; the promise decision reads it."""
        world, truth, params = self.world, self.truth, self.params
        rates = unit_rates(self.margin, params)
        on_hand = {sku.id: sku.initial_stock for sku in world.skus}
        lots = {lot.id: lot.quantity for lot in world.lots}
        backlog = {sku.id: 0.0 for sku in world.skus}
        # (arrival week, the order as the planner sees it, the quantity that will arrive)
        in_transit: list[tuple[int, OutstandingOrder, float]] = []
        outcomes = {order.id: OrderOutcome(order) for order in self.orders}
        by_arrival: dict[int, list[CustomerOrder]] = {}
        for order in self.orders:
            by_arrival.setdefault(order.arrival, []).append(order)
        notices: dict[str, int] = {}
        fallbacks = 0
        at_risk = 0.0
        received = 0.0
        statuses: dict[str, int] = {}

        for t in world.periods:
            moving = []
            for arrival, open_order, quantity in in_transit:
                if arrival <= t:
                    on_hand[open_order.sku] += quantity
                    received += quantity
                else:
                    moving.append((arrival, open_order, quantity))
            in_transit = moving

            policy(self._position(t, on_hand, lots, backlog, in_transit))
            view = policy.state.view(as_of=t)

            for line in self.supply.accepted.get(t, ()):
                arrival = t + lead_periods(truth, line.plant, t)
                delivered = line.quantity * self.yields.get((line.plant, t), 1.0)
                if arrival <= t:
                    on_hand[line.sku] += delivered
                    received += delivered
                else:
                    placed = OutstandingOrder(t, line.plant, line.sku, line.quantity)
                    in_transit.append((arrival, placed, delivered))

            open_commitments = [
                Commitment(o.order, o.promised, o.order.quantity - o.delivered)
                for o in outcomes.values()
                if o.promised is not None and o.order.quantity - o.delivered > _EPS
            ]
            decision = promise(
                view,
                self._position(t, on_hand, lots, backlog, in_transit),
                open_commitments,
                by_arrival.get(t, []),
                rates,
                params,
                notices if params.reveal_holds else None,
            )
            for order_id, week in decision.promised.items():
                outcomes[order_id].promised = week
            fallbacks += decision.fallback
            at_risk += decision.at_risk
            statuses[decision.status] = statuses.get(decision.status, 0) + 1

            self._execute(t, outcomes, on_hand, lots, notices)

        result = self._result(tuple(outcomes.values()), rates, fallbacks, at_risk, statuses)
        result.metrics["received_units"] = received
        return result

    def _position(
        self,
        t: int,
        on_hand: Mapping[str, float],
        lots: Mapping[str, float],
        backlog: Mapping[str, float],
        in_transit: Sequence[tuple[int, OutstandingOrder, float]],
    ) -> InventoryPosition:
        return InventoryPosition(
            period=t,
            on_hand=dict(on_hand),
            lot_remaining=dict(lots),
            backlog=dict(backlog),
            outstanding=tuple(order for _, order, _ in in_transit),
        )

    def _execute(
        self,
        t: int,
        outcomes: Mapping[str, OrderOutcome],
        on_hand: dict[str, float],
        lots: dict[str, float],
        notices: dict[str, int],
    ) -> None:
        """Serve commitments due by ``t`` from true usable stock, released lots first."""
        due = sorted(
            (
                o
                for o in outcomes.values()
                if o.promised is not None
                and o.promised <= t
                and o.order.quantity - o.delivered > _EPS
            ),
            key=lambda o: (o.promised, o.order.arrival, o.order.id),
        )
        for outcome in due:
            sku = outcome.order.sku
            need = outcome.order.quantity - outcome.delivered
            taken = 0.0
            for lot in sorted(self.world.lots, key=lambda lot: lot.id):
                if lot.sku != sku or not lot.arrived(t) or lots[lot.id] <= _EPS:
                    continue
                if self.truth.label(lot.id, Attribute.HOLD_STATUS, t) != HoldStatus.RELEASED:
                    continue
                take = min(lots[lot.id], need - taken)
                lots[lot.id] -= take
                taken += take
            take = min(max(0.0, on_hand[sku]), need - taken)
            on_hand[sku] -= take
            taken += take
            if taken > _EPS:
                outcome.deliveries.append((t, taken))
            if need - taken > _EPS and self.params.reveal_holds:
                for lot in self.world.lots:
                    held = self.truth.label(lot.id, Attribute.HOLD_STATUS, t) == HoldStatus.HELD
                    if lot.sku == sku and lot.arrived(t) and held and lots[lot.id] > _EPS:
                        notices.setdefault(lot.id, t)

    def _result(
        self,
        outcomes: tuple[OrderOutcome, ...],
        rates: Mapping[str, Mapping[str, float]],
        fallbacks: int,
        at_risk: float,
        statuses: Mapping[str, int],
    ) -> PromisingResult:
        end = self.world.end_period
        components = {"margin": 0.0, "lateness": 0.0, "broken": 0.0, "terminal": 0.0}
        accepted = [o for o in outcomes if o.promised is not None]
        on_time = broken = late = undelivered_orders = 0
        undelivered = 0.0
        for o in accepted:
            assert o.promised is not None
            rate = rates[o.order.sku]
            requested = o.order.requested
            components["margin"] += rate["margin"] * o.delivered
            weeks_late = sum(q * max(0, week - requested) for week, q in o.deliveries)
            missing = max(0.0, o.order.quantity - o.delivered)
            weeks_late += missing * max(0, end - requested)
            components["lateness"] += rate["lateness"] * weeks_late
            by_promise = sum(q for week, q in o.deliveries if week <= o.promised)
            short = max(0.0, o.order.quantity - by_promise)
            components["broken"] += rate["broken"] * short
            components["terminal"] += rate["terminal"] * missing
            on_time += short <= _EPS
            broken += short > _EPS
            late += any(week > o.promised for week, _ in o.deliveries)
            undelivered_orders += missing > _EPS
            undelivered += missing
        value = (
            components["margin"]
            - components["lateness"]
            - components["broken"]
            - components["terminal"]
        )
        rejected = [o for o in outcomes if o.promised is None]
        metrics: dict[str, Any] = {
            "orders": len(outcomes),
            "units": sum(o.order.quantity for o in outcomes),
            "accepted_orders": len(accepted),
            "accepted_units": sum(o.order.quantity for o in accepted),
            "rejected_orders": len(rejected),
            "rejected_units": sum(o.order.quantity for o in rejected),
            "promised_after_request": sum(
                o.promised > o.order.requested for o in accepted if o.promised is not None
            ),
            "on_time_orders": on_time,
            "broken_orders": broken,
            "late_orders": late,
            "undelivered_orders": undelivered_orders,
            "delivered_units": sum(o.delivered for o in accepted),
            "undelivered_units": undelivered,
            "fallbacks": fallbacks,
            "at_risk_units": at_risk,
            "solver_statuses": dict(statuses),
            "value": value,
        }
        return PromisingResult(outcomes, value, components, metrics)
