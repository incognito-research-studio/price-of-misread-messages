"""Order-promising study (docs/research/icores-revision-scope.md, Section 5).

The five hand-worked cases below were written before the promise and execution code existed.
Their expected values are worked out by hand in the comments. They were drafted without author
review (2026-10-27): an author must check each expectation before they count as the scope's
author-checked cases (Section 6). Do not change an expectation to match the implementation; if
they disagree, work the case again by hand.

Shared setup for the cases: ``helpers.build_world`` (weeks 40-45, one SKU, Plant 2 behind Port X
with a one-week lead time, Lot 7781 with 50 units on hand from the start and released). Unit
margin 10, so with the default fractions lateness = 2 per unit-week after the requested week,
a broken promise = 5 per unit missing at its promised week, terminal = 5 per accepted unit never
delivered. A promise may be at most two weeks after the requested week.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from cdd.benchmark import GeneratorConfig, generate
from cdd.experiments import ExperimentConfig, StreamContext
from cdd.interpreters import Interpreter, OracleInterpreter, ReplayInterpreter
from cdd.orders import OrderLine, Plan
from cdd.pipeline import CompilingPolicy, PlannedPolicy, build_truth
from cdd.promising import (
    CustomerOrder,
    FixedPlanDecider,
    FixedSupply,
    PromisingParams,
    PromisingResult,
    PromisingSimulator,
    generate_orders,
    margins,
    reference_supply,
)
from cdd.schema import Delta, DeltaType, Message, Op, Span, Window
from cdd.world import World
from helpers import build_world, make_delta, make_message

PARAMS = PromisingParams()
MARGIN = {"SKU-01": 10.0}


def _order(id: str, quantity: float, arrival: int, requested: int) -> CustomerOrder:
    return CustomerOrder(
        id=id, sku="SKU-01", quantity=quantity, arrival=arrival, requested=requested
    )


def _run(
    world: World,
    orders: Sequence[CustomerOrder],
    *,
    accepted: Mapping[int, tuple[OrderLine, ...]] | None = None,
    messages: Sequence[Message] = (),
    gold: Sequence[Delta] = (),
    read: Sequence[Delta] | None = None,
) -> PromisingResult:
    """Truth from ``gold``; the reader is the oracle, or replays ``read`` when given."""
    accepted = dict(accepted or {})
    plans = {t: Plan(period=t, orders=lines) for t, lines in accepted.items()}
    supply = FixedSupply(plans=plans, accepted=accepted, cost=0.0)
    truth = build_truth(world, messages, gold).view(as_of=None)
    interpreter: Interpreter = OracleInterpreter(gold) if read is None else ReplayInterpreter(read)
    policy = CompilingPolicy(world, messages, interpreter, FixedPlanDecider(plans))
    return PromisingSimulator(truth, supply, orders, PARAMS, MARGIN).run(policy)


def _promised(result: PromisingResult) -> dict[str, int | None]:
    return {o.order.id: o.promised for o in result.outcomes}


def _deliveries(result: PromisingResult) -> dict[str, list[tuple[int, float]]]:
    return {o.order.id: o.deliveries for o in result.outcomes}


def _span(message: Message) -> Span:
    return Span(message_id=message.id, start=0, end=len(message.text))


# --- Hand-worked cases (drafted without author review; an author must check each expectation) --


def test_case_1_excess_supply_accepts_everything_on_time() -> None:
    # Supply: 100 units from Plant 2 ordered in week 40, arriving in week 41; the lot's 50 units
    # are on hand from week 40. Orders: o1 60 units, arrives 40, requested 41; o2 40 units,
    # arrives 41, requested 42. Projected supply 150 covers both, so each is promised at its
    # requested week (later weeks only add lateness).
    # Week 41: o1 is due; released lot first (50), then free stock (10). Week 42: o2 takes 40 of
    # the remaining 90 free units. Nothing is delivered before its promised week.
    # Π = 10 x (60 + 40) = 1,000; no lateness, broken or terminal penalties.
    result = _run(
        build_world(),
        [_order("o1", 60, 40, 41), _order("o2", 40, 41, 42)],
        accepted={40: (OrderLine("P2", "SKU-01", 100.0),)},
    )
    assert _promised(result) == {"o1": 41, "o2": 42}
    assert _deliveries(result) == {"o1": [(41, 60.0)], "o2": [(42, 40.0)]}
    assert result.value == pytest.approx(1000.0)
    assert result.components["lateness"] == result.components["broken"] == 0
    assert result.metrics["on_time_orders"] == 2


def test_case_2_scarce_supply_accepts_the_more_valuable_order() -> None:
    # No replenishment; only the lot's 50 units. Both orders arrive in week 40 for week 41:
    # o1 40 units (worth 400), o2 30 units (worth 300). 70 > 50, and no later supply makes a
    # late promise feasible, so exactly one fits: o1, worth more. o2 is rejected.
    # Week 41: o1 takes 40 units from the lot. Π = 400.
    result = _run(build_world(), [_order("o1", 40, 40, 41), _order("o2", 30, 40, 41)])
    assert _promised(result) == {"o1": 41, "o2": None}
    assert _deliveries(result) == {"o1": [(41, 40.0)], "o2": []}
    assert result.value == pytest.approx(400.0)
    assert result.metrics["rejected_orders"] == 1


def _delay_case() -> tuple[World, Message, Delta]:
    world = build_world(lots=())
    message = make_message(
        "Carrier update: shipments on Port X delayed by 7 days starting week 40.",
        period=40,
        sender_class="carrier",
    )
    delay = make_delta(
        id="m1#0",
        type=DeltaType.LEAD_TIME_CHANGE,
        entity="LANE-X",
        op=Op.ADD,
        value=7.0,
        unit="days",
        window=Window(start=40),
        src=_span(message),
    )
    return world, message, delay


def test_case_3_delayed_receipt_costs_a_broken_promise_when_misread() -> None:
    # No lot. 100 units from Plant 2 ordered in week 40. The true lead time becomes 14 days in
    # week 40, so they arrive in week 42. Order o1: 80 units, arrives 40, requested 41.
    # Gold reader: ŝ says 14 days, the shipment is expected in week 42, so week 41 is
    # infeasible and o1 is promised for week 42 (value (10 - 2 x 1) x 80 = 640 > 0).
    # Delivered in 42: Π = 800 - 2 x 80 x 1 = 640.
    # Reader that misses the delay: expects the shipment in 41, promises 41. Week 41 has no
    # stock: broken. Week 42: 80 delivered, one week after the request.
    # Π = 800 - 2 x 80 x 1 - 5 x 80 = 240. Interpretation loss 640 - 240 = 400 (the broken
    # promise; the lateness is the same).
    world, message, delay = _delay_case()
    accepted = {40: (OrderLine("P2", "SKU-01", 100.0),)}
    orders = [_order("o1", 80, 40, 41)]
    gold = _run(world, orders, accepted=accepted, messages=[message], gold=[delay])
    assert _promised(gold) == {"o1": 42}
    assert _deliveries(gold) == {"o1": [(42, 80.0)]}
    assert gold.value == pytest.approx(640.0)

    missed = _run(world, orders, accepted=accepted, messages=[message], gold=[delay], read=[])
    assert _promised(missed) == {"o1": 41}
    assert _deliveries(missed) == {"o1": [(42, 80.0)]}
    assert missed.value == pytest.approx(240.0)
    assert missed.components["broken"] == pytest.approx(400.0)
    assert missed.metrics["broken_orders"] == 1


def _hold_release_case() -> tuple[list[Message], Delta, Delta]:
    hold_message = make_message(
        "QA has placed Lot 7781 on hold after a failed inspection.",
        id="m1",
        period=40,
        sender_class="supplier_quality",
    )
    release_message = make_message(
        "QA has released Lot 7781.", id="m2", period=42, sender_class="supplier_quality"
    )
    hold = make_delta(
        id="m1#0",
        type=DeltaType.QUALITY_HOLD,
        entity="LOT-7781",
        op=Op.SET,
        value="held",
        unit=None,
        window=Window(start=40),
        src=_span(hold_message),
    )
    release = make_delta(
        id="m2#0",
        type=DeltaType.QUALITY_RELEASE,
        entity="LOT-7781",
        op=Op.SET,
        value="released",
        unit=None,
        window=Window(start=42),
        src=_span(release_message),
    )
    return [hold_message, release_message], hold, release


def test_case_4_hold_then_release() -> None:
    # The lot (50 units) is truly held from week 40 and released from week 42. No replenishment.
    # Orders: o1 30 units, arrives 40, requested 41; o2 20 units, arrives 42, requested 43.
    # Gold reader: in week 40 ŝ holds the lot until further notice, so no week in 41-43 has
    # supply and o1 is rejected. In week 42 the release is read; o2 is promised for 43 and
    # delivered then. Π = 10 x 20 = 200.
    # Reader that misses the hold (it reads only the release, which SOP-003 rejects because ŝ
    # does not hold the lot): ŝ shows the lot released throughout. Week 40: o1 promised 41.
    # Week 41: the lot is truly held, so o1 is broken. Week 42: the projection has the lot's 50
    # units against o1's overdue 30 (due now) and o2's 20 in week 43, so o2 is promised 43;
    # execution delivers o1's 30 (one week after the request). Week 43: o2's 20.
    # Π = 10 x 50 - 2 x 30 x 1 - 5 x 30 = 290. The misreading gains 90 here: a negative
    # interpretation loss, because the gold reader rejected an order it could have served late.
    messages, hold, release = _hold_release_case()
    orders = [_order("o1", 30, 40, 41), _order("o2", 20, 42, 43)]
    gold = _run(build_world(), orders, messages=messages, gold=[hold, release])
    assert _promised(gold) == {"o1": None, "o2": 43}
    assert _deliveries(gold) == {"o1": [], "o2": [(43, 20.0)]}
    assert gold.value == pytest.approx(200.0)

    missed = _run(build_world(), orders, messages=messages, gold=[hold, release], read=[release])
    assert _promised(missed) == {"o1": 41, "o2": 43}
    assert _deliveries(missed) == {"o1": [(42, 30.0)], "o2": [(43, 20.0)]}
    assert missed.value == pytest.approx(290.0)
    assert missed.components["broken"] == pytest.approx(150.0)
    assert missed.components["lateness"] == pytest.approx(60.0)


def test_case_5_a_spurious_hold_rejects_a_servable_order() -> None:
    # The truth has no hold. A note in week 40 mentions Lot 7781 without changing anything; the
    # reader reads it as a hold from week 40. Order o1: 30 units, arrives 40, requested 41.
    # Gold reader: o1 promised 41 and delivered from the lot. Π = 300.
    # Misled reader: ŝ holds the lot until further notice, so o1 is rejected. Π = 0.
    message = make_message(
        "QA audit of Lot 7781 is planned for next week.", period=40, sender_class="internal"
    )
    spurious = make_delta(
        id="m1#0",
        type=DeltaType.QUALITY_HOLD,
        entity="LOT-7781",
        op=Op.SET,
        value="held",
        unit=None,
        window=Window(start=40),
        src=_span(message),
    )
    orders = [_order("o1", 30, 40, 41)]
    gold = _run(build_world(), orders, messages=[message])
    assert _promised(gold) == {"o1": 41}
    assert gold.value == pytest.approx(300.0)

    misled = _run(build_world(), orders, messages=[message], read=[spurious])
    assert _promised(misled) == {"o1": None}
    assert misled.value == 0
    assert misled.metrics["accepted_orders"] == 0


# --- Customer orders and fixed supply ---------------------------------------------------------


def test_orders_split_demand_exactly_and_arrive_before_their_week() -> None:
    world = build_world(start_period=36, end_period=52)
    rng = np.random.default_rng(3)
    demand = {("SKU-01", week): float(rng.uniform(50, 150)) for week in world.periods}
    orders = generate_orders(world, demand, seed=7, params=PARAMS)

    first = world.start_period + PARAMS.first_requested_offset
    for week in world.periods:
        mine = [o for o in orders if o.requested == week]
        if week < first:
            assert mine == []
            continue
        assert len(mine) == PARAMS.orders_per_week
        assert sum(o.quantity for o in mine) == pytest.approx(demand[("SKU-01", week)], abs=1e-9)
        assert all(o.quantity >= 0 for o in mine)
        assert all(week - o.arrival in PARAMS.customer_lead_weeks for o in mine)
    assert all(o.arrival >= world.start_period for o in orders)
    assert [(o.arrival, o.id) for o in orders] == sorted((o.arrival, o.id) for o in orders)
    assert generate_orders(world, demand, seed=7, params=PARAMS) == orders
    assert generate_orders(world, demand, seed=8, params=PARAMS) != orders


def test_surge_setting_scales_every_week_of_orders() -> None:
    world = build_world(start_period=36, end_period=52)
    demand = {("SKU-01", week): 100.0 for week in world.periods}
    surge = PARAMS.model_copy(update={"demand_scale": 1.3})
    base = generate_orders(world, demand, seed=7, params=PARAMS)
    scaled = generate_orders(world, demand, seed=7, params=surge)
    for week in range(world.start_period + PARAMS.first_requested_offset, world.end_period + 1):
        assert sum(o.quantity for o in scaled if o.requested == week) == pytest.approx(130.0)
    # Same draws: only the quantities scale.
    assert [(o.id, o.arrival) for o in scaled] == [(o.id, o.arrival) for o in base]


def test_params_validate_the_order_horizon() -> None:
    with pytest.raises(ValueError, match="first_requested_offset"):
        PromisingParams(customer_lead_weeks=(1, 4), first_requested_offset=3)
    assert margins(build_world(), PARAMS) == {"SKU-01": pytest.approx(0.5 * (10.0 + 12.0) / 2)}


def test_reference_supply_replays_the_planner_on_the_gold_state() -> None:
    stream = generate(GeneratorConfig.model_validate({"name": "t", "n_streams": 1, "seed": 5}))[0]
    config = ExperimentConfig.model_validate(
        {"name": "t", "streams": ".", "conditions": [{"name": "oracle"}]}
    )
    context = StreamContext(stream, config)
    supply = reference_supply(context)
    assert supply.cost == pytest.approx(context.planner_cost())
    replay = context.simulator.run(PlannedPolicy(supply.plans))
    assert replay.total_cost == pytest.approx(supply.cost)
    assert {r.period: r.accepted for r in replay.periods} == dict(supply.accepted)


# --- Execution properties on tiny worlds (also worked by hand) --------------------------------


def test_obligations_open_at_the_horizon_end_are_counted() -> None:
    # Case 3 with the horizon cut at week 41: the delayed shipment (week 42) never arrives in
    # time. The reader that misses the delay promises o1 (80 units) for 41; nothing is ever
    # delivered. Lateness counts through week 41 = the requested week, so 0; broken 5 x 80;
    # terminal 5 x 80. Π = -800.
    world, message, delay = _delay_case()
    world = world.model_copy(update={"end_period": 41})
    result = _run(
        world,
        [_order("o1", 80, 40, 41)],
        accepted={40: (OrderLine("P2", "SKU-01", 100.0),)},
        messages=[message],
        gold=[delay],
        read=[],
    )
    assert _promised(result) == {"o1": 41}
    assert result.value == pytest.approx(-800.0)
    assert result.metrics["undelivered_units"] == pytest.approx(80.0)
    assert result.components["terminal"] == pytest.approx(400.0)


def _held_throughout() -> tuple[list[Message], Delta]:
    messages, hold, _ = _hold_release_case()
    return messages[:1], hold


def test_held_stock_is_never_delivered() -> None:
    # The lot is truly held for the whole horizon and is the only stock. A reader that misses
    # the hold promises o1; execution must not touch the held lot.
    messages, hold = _held_throughout()
    result = _run(
        build_world(), [_order("o1", 30, 40, 41)], messages=messages, gold=[hold], read=[]
    )
    assert _promised(result) == {"o1": 41}
    assert _deliveries(result) == {"o1": []}
    assert result.metrics["delivered_units"] == 0


def test_hold_notice_sensitivity_stops_promising_held_stock() -> None:
    # Lot held throughout; the reader misses the hold. o1 (30, arrives 40, for 41) is promised
    # and fails in week 41. Without notices, in week 42 ŝ still shows 50 units against o1's
    # overdue 30, so o2 (20, arrives 42, for 43) is promised too and fails. With the hold-notice
    # sensitivity, week 41's failure reveals the hold from week 42, so o2 is rejected.
    messages, hold = _held_throughout()
    orders = [_order("o1", 30, 40, 41), _order("o2", 20, 42, 43)]
    plain = _run(build_world(), orders, messages=messages, gold=[hold], read=[])
    assert _promised(plain) == {"o1": 41, "o2": 43}

    world = build_world()
    truth = build_truth(world, messages, [hold]).view(as_of=None)
    supply = FixedSupply(plans={}, accepted={}, cost=0.0)
    policy = CompilingPolicy(world, messages, ReplayInterpreter([]), FixedPlanDecider({}))
    revealed = PromisingSimulator(
        truth, supply, orders, PARAMS.model_copy(update={"reveal_holds": True}), MARGIN
    ).run(policy)
    assert _promised(revealed) == {"o1": 41, "o2": None}
