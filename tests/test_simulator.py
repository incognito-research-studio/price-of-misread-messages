from __future__ import annotations

from collections.abc import Callable

import pytest

from cdd.orders import InventoryPosition, OrderLine, Plan
from cdd.schema import DeltaType, Op, Window
from cdd.simulator import InventorySimulator
from cdd.state import BitemporalState
from cdd.update import apply_update
from cdd.world import Sku, World
from helpers import build_world, make_delta


def _fixed(quantity: float, plant: str = "P2") -> Callable[[InventoryPosition], Plan]:
    def policy(position: InventoryPosition) -> Plan:
        return Plan(period=position.period, orders=(OrderLine(plant, "SKU-01", quantity),))

    return policy


def test_demand_uses_common_random_numbers(world: World, state: BitemporalState) -> None:
    truth = state.view(None)
    assert InventorySimulator.sample_demand(truth, 1) == InventorySimulator.sample_demand(truth, 1)
    assert InventorySimulator.sample_demand(truth, 1) != InventorySimulator.sample_demand(truth, 2)
    flat = InventorySimulator.sample_demand(truth, 1, cv=0.0)
    assert all(value == pytest.approx(100.0) for value in flat.values())


def test_costs_and_fill_rate(world: World, state: BitemporalState) -> None:
    truth = state.view(None)
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    result = InventorySimulator(truth, demand).run(_fixed(100.0))
    assert result.violations == ()
    assert result.purchase_cost == pytest.approx(10.0 * 100.0 * len(world.periods))
    assert result.total_cost == pytest.approx(
        result.purchase_cost + result.holding_cost + result.backlog_cost
    )
    assert 0.0 < result.fill_rate < 1.0  # the first week has only the lot
    assert len(result.order_vector()) == len(world.periods)


def test_orders_beyond_true_capacity_are_violations(world: World, state: BitemporalState) -> None:
    apply_update(state, [make_delta(value=0.1, window=Window(start=40, end=45))], recorded_at=40)
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    result = InventorySimulator(state.view(None), demand).run(_fixed(300.0))
    assert {v.kind for v in result.violations} == {"capacity"}
    assert result.periods[0].accepted[0].quantity == pytest.approx(100.0)


def test_suspended_supplier_ships_nothing(world: World, state: BitemporalState) -> None:
    suspension = make_delta(
        type=DeltaType.SUPPLIER_STATUS, entity="SUP-C", op=Op.SET, value="suspended", unit=None,
        window=Window(start=40),
    )  # fmt: skip
    apply_update(state, [suspension], recorded_at=40)
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    result = InventorySimulator(state.view(None), demand).run(_fixed(100.0))
    assert {v.kind for v in result.violations} == {"supplier_status"}
    assert result.purchase_cost == 0.0


def test_released_lots_are_used_before_free_stock() -> None:
    world = build_world(
        skus=(Sku(id="SKU-01", name="SKU-01", base_demand=100.0, initial_stock=100.0),)
    )
    state = BitemporalState(world)
    hold = make_delta(
        type=DeltaType.QUALITY_HOLD, entity="LOT-7781", op=Op.SET, value="held", unit=None,
        window=Window(start=41),
    )  # fmt: skip
    apply_update(state, [hold], recorded_at=40)
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    result = InventorySimulator(state.view(None), demand).run(_fixed(0.0))
    # Week 40 uses the 50-unit lot before the hold freezes it, so 50 free units remain for week 41.
    assert result.periods[1].backlog["SKU-01"] == pytest.approx(50.0)


def test_a_refused_order_teaches_the_planner_the_supplier_status(
    world: World, state: BitemporalState
) -> None:
    from cdd.interpreters import ReplayInterpreter
    from cdd.pipeline import CompilingPolicy
    from cdd.solver import MILPSolver

    suspension = make_delta(
        type=DeltaType.SUPPLIER_STATUS, entity="SUP-C", op=Op.SET, value="suspended", unit=None,
        window=Window(start=40),
    )  # fmt: skip
    apply_update(state, [suspension], recorded_at=40)
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    simulator = InventorySimulator(state.view(None), demand)
    # A reader that missed the suspension: the planner first orders from the cheaper Plant 2.
    policy = CompilingPolicy(world, (), ReplayInterpreter(()), MILPSolver(horizon=3))
    result = simulator.run(policy)
    refused = [v for v in result.violations if v.kind == "supplier_status"]
    assert [v.period for v in refused] == [40]
    assert all(line.plant == "P3" for r in result.periods[1:] for line in r.ordered)
