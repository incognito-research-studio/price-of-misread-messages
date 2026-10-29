"""Layer ablations, the hybrid interpreter, model deciders over state, and calibration."""

from __future__ import annotations

import json

from cdd.deciders import LLMStateDecider
from cdd.evaluation import align, expected_calibration_error
from cdd.interpreters import HybridInterpreter, ReplayInterpreter
from cdd.interpreters.base import Interpretation
from cdd.orders import InventoryPosition
from cdd.pipeline import CompilingPolicy, build_truth
from cdd.schema import Attribute, DeltaType, Message, Op, Proposal, Span, Window
from cdd.solver import MILPSolver
from cdd.state import BitemporalState, StateView
from cdd.update import apply_overwrite, apply_update
from cdd.world import World
from fakes import ScriptedLLMClient, ScriptedTypedClient
from helpers import make_delta, make_message


def _position(world: World) -> InventoryPosition:
    return InventoryPosition(
        period=40, on_hand={"SKU-01": 0.0}, lot_remaining={"LOT-7781": 50.0}, backlog={}
    )


def test_overwrite_drops_earlier_deltas_on_the_same_attribute(state: BitemporalState) -> None:
    first = make_delta(id="m1#0", value=0.5, window=Window(start=41, end=41))
    second = make_delta(
        id="m2#0",
        value=0.8,
        window=Window(start=44, end=44),
        src=Span(message_id="m2", start=0, end=5),
    )
    apply_overwrite(state, [first], recorded_at=40)
    result = apply_overwrite(state, [second], recorded_at=41)
    assert result.retracted == ("m1#0",)
    assert state.value("P2", Attribute.CAPACITY, 41) == 1000.0  # the first cut is gone
    versioned = BitemporalState(state.world)
    apply_update(versioned, [first], recorded_at=40)
    apply_update(versioned, [second], recorded_at=41)
    assert versioned.value("P2", Attribute.CAPACITY, 41) == 500.0


def test_without_the_sop_gate_a_release_without_hold_is_admitted(world: World) -> None:
    text = "Lot 7781 released."
    message = make_message(text, id="m1", period=40)
    release = make_delta(
        id="m1#0", type=DeltaType.QUALITY_RELEASE, entity="LOT-7781", op=Op.SET, value="released",
        unit=None, window=Window(start=40), src=Span(message_id="m1", start=0, end=len(text)),
    )  # fmt: skip
    gated = CompilingPolicy(world, [message], ReplayInterpreter([release]), MILPSolver(horizon=2))
    open_gate = CompilingPolicy(
        world, [message], ReplayInterpreter([release]), MILPSolver(horizon=2), sop_gate=False
    )
    gated(_position(world))
    open_gate(_position(world))
    assert gated.state.active() == []
    assert [d.id for d in open_gate.state.active()] == ["m1#0"]


class _Fixed:
    def __init__(self, name: str, interpretation: Interpretation) -> None:
        self.name = name
        self.interpretation = interpretation
        self.calls = 0

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        self.calls += 1
        return self.interpretation


def _proposal(kind: DeltaType, confidence: float | None) -> Proposal:
    return Proposal(
        type=kind, entity="P2", span=Span(message_id="m1", start=0, end=5), confidence=confidence
    )


def test_hybrid_escalates_low_confidence_failures_and_listed_types(
    world: World, state: BitemporalState
) -> None:
    fallback = _Fixed("llm", Interpretation())
    message, view = make_message("x"), state.view(40)

    def run(primary: Interpretation, **kwargs: object) -> bool:
        hybrid = HybridInterpreter(_Fixed("typed", primary), fallback, **kwargs)  # type: ignore[arg-type]
        hybrid.interpret(message, view)
        return hybrid.escalation_share == 1.0

    confident = Interpretation((_proposal(DeltaType.CAPACITY_CHANGE, 0.95),))
    assert not run(confident, threshold=0.8)
    assert run(Interpretation((_proposal(DeltaType.CAPACITY_CHANGE, 0.5),)), threshold=0.8)
    assert run(Interpretation(failures=("bad answer",)))
    assert run(confident, escalate_types=(DeltaType.CAPACITY_CHANGE,))
    assert not run(Interpretation())  # a clean abstention stands


def test_llm_decides_over_the_compiled_state(world: World) -> None:
    reply = json.dumps({"orders": [{"plant": "P3", "sku": "SKU-01", "quantity": 80}]})
    client = ScriptedLLMClient(lambda system, user: reply)
    decider = LLMStateDecider(client, constrained=True)
    truth = build_truth(world, [], []).view(None)
    plan = decider.plan(truth, _position(world))
    assert [(o.plant, o.quantity) for o in plan.orders] == [("P3", 80.0)]
    assert client.schemas[0] is not None and "Demand forecast" in client.last_user


def test_safety_factor_orders_more(world: World, state: BitemporalState) -> None:
    def total(factor: float) -> float:
        plan = MILPSolver(horizon=3, safety_factor=factor).plan(state.view(40), _position(world))
        return sum(o.quantity for o in plan.orders)

    assert total(0.3) > total(0.0)


def test_calibration_error() -> None:
    gold = [
        make_delta(id="m1#g0"),
        make_delta(id="m2#g0", src=Span(message_id="m2", start=0, end=5)),
    ]
    right = make_delta(confidence=0.9)
    wrong = make_delta(
        id="m2#0", value=0.5, confidence=0.9, src=Span(message_id="m2", start=0, end=5)
    )
    alignment = align([right, wrong], gold)
    assert expected_calibration_error(alignment) == 0.4  # 90% confident, 50% right
    assert expected_calibration_error(align([make_delta()], gold[:1])) is None
    assert ScriptedTypedClient().probability == 0.9


def test_model_deciders_are_told_the_objective_and_the_master_data(world: World) -> None:
    from cdd.deciders import LLMStreamDecider, orders_schema, parse_orders
    from cdd.schema import Message
    from fakes import ScriptedLLMClient

    reply = (
        '{"orders": [{"plant": "P2", "sku": "SKU-01", "quantity": 50},'
        ' {"plant": "P2", "sku": "SKU-09", "quantity": 20}]}'
    )
    client = ScriptedLLMClient(lambda system, user: reply)
    decider = LLMStreamDecider(client, world, constrained=True)
    message = Message(
        id="m1", period=40, sender="x", sender_class="procurement", channel="email", text="Hi."
    )
    plan = decider.decide((message,), _position(world))
    assert "Objective:" in client.last_user and "backlog cost" in client.last_user
    assert "Master data as of week 40" in client.last_user
    assert "capacity 1000 units per week; makes SKU-01" in client.last_user
    assert (len(plan.orders), plan.invalid_lines) == (1, 1)
    schema = orders_schema(world)
    allowed = schema["properties"]["orders"]["items"]["anyOf"]
    assert [
        (a["properties"]["plant"]["const"], a["properties"]["sku"]["const"]) for a in allowed
    ] == [
        ("P2", "SKU-01"),
        ("P3", "SKU-01"),
    ]
    assert parse_orders("not json", world, 40).status == "malformed"


def test_deciders_are_told_the_horizon_and_see_only_live_notices(world: World) -> None:
    from dataclasses import replace

    from cdd.prompts import render_objective, render_state
    from cdd.schema import DeltaType, Op

    assert f"ends after week {world.end_period}" in render_objective(world)
    # A state decider can link each plant to its supplier's prices.
    rendered = render_state(BitemporalState(world).view(40), _position(world), 3)
    assert "Plant 2 (P2), Supplier C (SUP-C): makes SKU-01" in rendered
    state = BitemporalState(world)
    position = replace(_position(world), period=42, status_notices={"SUP-C": (40, "suspended")})
    assert "refused in wk40" in render_state(state.view(42), position, 3)
    reinstated = make_delta(
        id="r#0", type=DeltaType.SUPPLIER_STATUS, entity="SUP-C", op=Op.SET, value="active",
        unit=None, window=Window(start=41),
    )  # fmt: skip
    apply_update(state, [reinstated], recorded_at=41)
    assert "refused in wk40" not in render_state(state.view(42), position, 3)
