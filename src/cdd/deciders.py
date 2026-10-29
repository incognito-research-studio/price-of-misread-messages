"""Deciders: the component that turns information into orders.

A :class:`Decider` reads an interpreted state; the MILP solver is the default, and a typed
model deciding over ŝ is the strongest rival to it. A :class:`StreamDecider` reads raw
messages with no state at all: condition (a) and its typed variant. Together they let an
experiment cross the state source (raw stream or compiled ŝ) with the decider (model or
solver), which separates the effect of compiling from the effect of not deciding.

Every model decider is told what the solver optimises: the objective, the holding and backlog
costs and the physical rules (``render_objective``). Stream deciders also get the master data as
of the first week (capacities, demand, lead times, prices, which plant makes which SKU, lots),
so the messages are their only source of change, as they are for the readers. Order lines for a
plant that does not make the SKU are dropped and counted; with ``constrained`` decoding the
schema allows only real supply lines.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from cdd.interpreters.llm import LLMClient
from cdd.interpreters.typed import MAX_OPTIONS, TypedModelClient, TypedQuestion, ask_all
from cdd.orders import InventoryPosition, OrderLine, Plan
from cdd.prompts import (
    render_entities,
    render_master,
    render_messages,
    render_objective,
    render_position,
    render_state,
)
from cdd.schema import Message
from cdd.state import StateView
from cdd.world import World


class Decider(Protocol):
    @property
    def name(self) -> str: ...

    def plan(self, view: StateView, position: InventoryPosition) -> Plan: ...


class StreamDecider(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, history: Sequence[Message], position: InventoryPosition) -> Plan: ...


def select_history(history: Sequence[Message], context: str) -> Sequence[Message]:
    """``"full"`` keeps every message seen so far; ``"last:N"`` keeps the latest N."""
    if context == "full":
        return history
    if context.startswith("last:"):
        count = int(context.removeprefix("last:"))
        return history[-count:] if count > 0 else ()
    raise ValueError(f"unknown context policy {context!r}")


def quantity_options(world: World, plant_id: str, bucket: float) -> tuple[str, ...]:
    """Order quantities from zero up to the plant's base capacity, in steps of ``bucket``."""
    count = min(MAX_OPTIONS, int(world.plant(plant_id).capacity // bucket) + 1)
    return tuple(f"{i * bucket:g}" for i in range(count))


def _typed_orders(
    client: TypedModelClient, context: str, world: World, period: int, bucket: float
) -> Plan:
    questions = [
        TypedQuestion(
            f"{plant}|{sku}",
            f"How many units of {world.sku(sku).name} should be ordered from "
            f"{world.plant(plant).name} this week?",
            quantity_options(world, plant, bucket),
        )
        for plant, sku in world.lines()
    ]
    try:
        answers = ask_all(client, context, questions)
    except ValueError:
        return Plan(period=period, status="malformed")
    orders = []
    for plant, sku in world.lines():
        quantity = float(answers[f"{plant}|{sku}"].choice)
        if quantity > 0:
            orders.append(OrderLine(plant, sku, quantity))
    return Plan(period=period, orders=tuple(orders), status="model")


class TypedStateDecider:
    """A typed model picks an order bucket for every line, reading the compiled state ŝ."""

    def __init__(
        self,
        client: TypedModelClient,
        *,
        bucket: float = 25.0,
        horizon: int = 6,
        name: str = "typed-state",
    ) -> None:
        self.client = client
        self.bucket = bucket
        self.horizon = horizon
        self.name = name

    def plan(self, view: StateView, position: InventoryPosition) -> Plan:
        context = f"{render_objective(view.world)}\n\n{render_state(view, position, self.horizon)}"
        return _typed_orders(self.client, context, view.world, position.period, self.bucket)


class TypedStreamDecider:
    """A typed model picks order buckets from the raw message history (no ŝ)."""

    def __init__(
        self,
        client: TypedModelClient,
        world: World,
        *,
        bucket: float = 25.0,
        context: str = "full",
        name: str = "typed-stream",
    ) -> None:
        self.client = client
        self.world = world
        self.bucket = bucket
        self.context = context
        self.name = name

    def decide(self, history: Sequence[Message], position: InventoryPosition) -> Plan:
        context = (
            f"{render_objective(self.world)}\n\n{render_master(self.world)}\n\n"
            f"{render_messages(select_history(history, self.context))}\n\n"
            f"{render_position(self.world, position)}"
        )
        return _typed_orders(self.client, context, self.world, position.period, self.bucket)


DECIDER_SYSTEM_PROMPT = """\
You plan replenishment orders for one distribution centre. Read the objective, the master data
(the network as of the first week), the messages received so far (they report changes since
then) and the inventory position, then decide how many units to order from each plant this
week. Order a SKU only from a plant that makes it.
Answer with JSON only:
{"orders": [{"plant": "<plant id>", "sku": "<sku id>", "quantity": <number>}]}"""

STATE_DECIDER_SYSTEM_PROMPT = """\
You plan replenishment orders for one distribution centre. Read the objective and the planning
state (which SKUs each plant makes, capacity, lead times, demand forecast, prices, lot holds
and the inventory position), then decide how many units to order from each plant this week.
Order a SKU only from a plant that makes it.
Answer with JSON only:
{"orders": [{"plant": "<plant id>", "sku": "<sku id>", "quantity": <number>}]}"""


def orders_schema(world: World) -> dict[str, Any]:
    """JSON Schema for order replies: each line names one of this world's supply lines (a
    plant and a SKU it makes) and a non-negative quantity."""
    lines = [
        {
            "type": "object",
            "properties": {
                "plant": {"const": plant},
                "sku": {"const": sku},
                "quantity": {"type": "number", "minimum": 0},
            },
            "required": ["plant", "sku", "quantity"],
        }
        for plant, sku in world.lines()
    ]
    return {
        "type": "object",
        "properties": {"orders": {"type": "array", "items": {"anyOf": lines}}},
        "required": ["orders"],
    }


def parse_orders(raw: str, world: World, period: int) -> Plan:
    """Orders from a model reply. Lines for a plant that does not make the SKU are dropped and
    counted in ``invalid_lines``; a reply that does not parse becomes an empty plan with
    status ``malformed``."""
    try:
        start, end = raw.find("{"), raw.rfind("}")
        items = json.loads(raw[start : end + 1])["orders"]
        lines = world.lines()
        parsed = [
            OrderLine(str(item["plant"]), str(item["sku"]), float(item["quantity"]))
            for item in items
        ]
    except (ValueError, KeyError, TypeError):
        return Plan(period=period, status="malformed")
    valid = [line for line in parsed if (line.plant, line.sku) in lines]
    orders = tuple(line for line in valid if line.quantity > 0)
    return Plan(
        period=period, orders=orders, status="model", invalid_lines=len(parsed) - len(valid)
    )


class LLMStreamDecider:
    """Condition (a): an LLM reads the message history and returns orders directly."""

    def __init__(
        self,
        client: LLMClient,
        world: World,
        *,
        context: str = "full",
        constrained: bool = False,
        name: str = "llm-stream",
    ) -> None:
        self.client = client
        self.world = world
        self.context = context
        self.constrained = constrained
        self.name = name

    def decide(self, history: Sequence[Message], position: InventoryPosition) -> Plan:
        user = (
            f"{render_objective(self.world)}\n\n{render_master(self.world)}\n\n"
            f"Entities:\n{render_entities(self.world)}\n\n"
            f"{render_messages(select_history(history, self.context))}\n\n"
            f"{render_position(self.world, position)}"
        )
        schema = orders_schema(self.world) if self.constrained else None
        raw = self.client.complete(DECIDER_SYSTEM_PROMPT, user, schema)
        return parse_orders(raw, self.world, position.period)


class LLMStateDecider:
    """Condition (i), LLM version: an LLM reads the compiled state and returns orders."""

    def __init__(
        self,
        client: LLMClient,
        *,
        horizon: int = 6,
        constrained: bool = False,
        name: str = "llm-state",
    ) -> None:
        self.client = client
        self.horizon = horizon
        self.constrained = constrained
        self.name = name

    def plan(self, view: StateView, position: InventoryPosition) -> Plan:
        user = (
            f"{render_objective(view.world)}\n\n"
            f"Entities:\n{render_entities(view.world)}\n\n"
            f"{render_state(view, position, self.horizon)}"
        )
        schema = orders_schema(view.world) if self.constrained else None
        raw = self.client.complete(STATE_DECIDER_SYSTEM_PROMPT, user, schema)
        return parse_orders(raw, view.world, position.period)
