from __future__ import annotations

import pytest

from cdd.orders import InventoryPosition, Plan
from cdd.schema import Delta, DeltaType, Op, Window
from cdd.solver import MILPSolver, hindsight_plans
from cdd.state import BitemporalState
from cdd.update import apply_update
from cdd.world import Lot, World
from helpers import build_world, make_delta


def _position(world: World, period: int = 40) -> InventoryPosition:
    return InventoryPosition(
        period=period,
        on_hand={"SKU-01": 0.0},
        lot_remaining={lot.id: lot.quantity for lot in world.lots},
        backlog={"SKU-01": 0.0},
    )


def _ordered(plan: Plan) -> dict[str, float]:
    return {line.plant: line.quantity for line in plan.orders}


def test_orders_from_the_cheapest_plant(world: World, state: BitemporalState) -> None:
    plan = MILPSolver(horizon=4).plan(state.view(40), _position(world))
    assert plan.status == "optimal"
    # Demand is 100 a week and orders take a week, so week 40 ends 50 short (the lot covers 50).
    # The week-40 order covers week 41 plus that backlog, all from P2, the cheaper plant.
    assert _ordered(plan) == {"P2": pytest.approx(150.0)}


def test_suspended_supplier_gets_no_orders(world: World, state: BitemporalState) -> None:
    suspension = make_delta(
        type=DeltaType.SUPPLIER_STATUS, entity="SUP-C", op=Op.SET, value="suspended", unit=None,
        window=Window(start=40),
    )  # fmt: skip
    apply_update(state, [suspension], recorded_at=40)
    plan = MILPSolver(horizon=4).plan(state.view(40), _position(world))
    assert set(_ordered(plan)) == {"P3"}


def test_capacity_cut_is_respected(world: World, state: BitemporalState) -> None:
    apply_update(state, [make_delta(value=0.05, window=Window(start=40, end=45))], recorded_at=40)
    plan = MILPSolver(horizon=4).plan(state.view(40), _position(world))
    assert _ordered(plan)["P2"] <= 50.0 + 1e-6


def test_duals_are_reported(world: World, state: BitemporalState) -> None:
    plan = MILPSolver(horizon=3, compute_duals=True).plan(state.view(40), _position(world))
    assert plan.duals is not None
    assert "capacity[P2,40]" in plan.duals and "balance[SKU-01,41]" in plan.duals


def test_hindsight_plans_cover_every_period(world: World, state: BitemporalState) -> None:
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    plans = hindsight_plans(state.view(None), _position(world), demand)
    assert sorted(plans) == list(world.periods)
    assert all(plan.status == "optimal" for plan in plans.values())


def _hold(start: int) -> Delta:
    return make_delta(
        type=DeltaType.QUALITY_HOLD,
        entity="LOT-7781",
        op=Op.SET,
        value="held",
        unit=None,
        window=Window(start=start),
    )


def test_planner_does_not_count_on_a_lot_that_goes_on_hold() -> None:
    world = build_world(lots=(Lot(id="LOT-7781", name="Lot 7781", sku="SKU-01", quantity=300.0),))
    free = BitemporalState(world)
    assert _ordered(MILPSolver(horizon=3).plan(free.view(40), _position(world))) == {}
    held = BitemporalState(world)
    apply_update(held, [_hold(41)], recorded_at=40)
    # Week 40 draws 100 from the lot; the hold freezes the other 200, so week 41 needs an order.
    plan = MILPSolver(horizon=3).plan(held.view(40), _position(world))
    assert _ordered(plan) == {"P2": pytest.approx(100.0)}
