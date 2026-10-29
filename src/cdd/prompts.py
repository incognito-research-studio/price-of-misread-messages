"""Plain-text renderings of messages, entities and state for model-backed components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from cdd.orders import InventoryPosition, capacity_available, lead_periods
from cdd.schema import Attribute, DeltaType, EntityKind, Message
from cdd.state import StateView
from cdd.world import World

# Descriptions with the example messages of Table 1 in the paper, renamed to entities that no
# generated world contains (Riverside plant, Port Q, Lot 1234, Acme Corp). With the Table 1
# names, a model would quote an example as its evidence when the example named a real entity:
# "Supplier C suspended pending audit" for a differently worded message about Supplier C.
TYPE_GUIDE: dict[DeltaType, str] = {
    DeltaType.CAPACITY_CHANGE: (
        "a plant's output capacity changes ('Riverside plant output reduced ~15% wk42-44')"
    ),
    DeltaType.LEAD_TIME_CHANGE: (
        "a lane's transit time changes ('Transit via Port Q +6 days through Nov')"
    ),
    DeltaType.QUALITY_HOLD: "a lot goes on quality hold ('Lot 1234 on hold pending CoA')",
    DeltaType.QUALITY_RELEASE: "a held lot is released ('Lot 1234 released')",
    DeltaType.SUPPLIER_STATUS: (
        "a supplier is suspended, restricted or reinstated ('Acme Corp suspended pending audit')"
    ),
    DeltaType.DEMAND_SIGNAL: ("demand for a SKU changes ('Key account orders up 20% in wk40')"),
    DeltaType.PRICE_CHANGE: (
        "a supplier's price for a SKU changes ('Surcharge 4% effective Oct 1')"
    ),
}


def render_types() -> str:
    return "\n".join(f"- {kind.value}: {text}" for kind, text in TYPE_GUIDE.items())


def render_message(message: Message) -> str:
    return (
        f"Message {message.id} (week {message.period}, {message.channel}, "
        f"from {message.sender} [{message.sender_class}]):\n{message.text}"
    )


def render_messages(messages: Sequence[Message]) -> str:
    return "\n\n".join(render_message(message) for message in messages) or "(no messages)"


def render_entities(world: World) -> str:
    lines = []
    for kind in EntityKind:
        entries = ", ".join(f"{e.id} = {e.name}" for e in world.entities(kind))
        if entries:
            lines.append(f"- {kind.value}: {entries}")
    return "\n".join(lines)


def render_deltas(view: StateView) -> str:
    deltas = view.active()
    if not deltas:
        return "(none)"
    return "\n".join(
        f"- {d.id}: {d.type.value} {d.entity} {d.op.value} {d.value} "
        f"wk{d.window.start}-{'open' if d.window.end is None else f'wk{d.window.end}'}"
        for d in deltas
    )


def _costs(values: Sequence[float], world: World, unit: str) -> str:
    if len(set(values)) == 1:
        return f"{values[0]:g} {unit}"
    return ", ".join(f"{sku.id} {value:g}" for sku, value in zip(world.skus, values, strict=True))


def render_objective(world: World) -> str:
    """The planner's objective and the physical rules, as every model decider is told them."""
    holding = _costs([sku.holding_cost for sku in world.skus], world, "per unit per week")
    backlog = _costs([sku.backlog_cost for sku in world.skus], world, "per unit per week")
    return (
        "Objective: keep the total cost over the remaining weeks as low as possible. The "
        f"simulation ends after week {world.end_period}; units that arrive later are paid for "
        "but never used. Actual weekly demand varies randomly around the forecast, typically "
        "within about 20%. Total cost "
        "is the purchase price of every unit accepted, plus holding cost on every unit in stock "
        f"at the end of each week ({holding}), plus backlog cost on every unit of demand not "
        f"yet served at the end of each week ({backlog}). Unserved demand stays as backlog "
        "until it is served. An order arrives after its lane's lead time, rounded up to whole "
        "weeks. A plant ships at most its weekly capacity, shared across the SKUs it makes; a "
        "suspended supplier ships nothing, and a restricted supplier ships "
        f"{world.restricted_capacity_factor:.0%} of its plants' capacity. A plant refuses what "
        "it cannot ship. Lots on quality hold cannot be used. Orders can only be placed for a "
        "SKU at a plant that makes it."
    )


def render_master(world: World) -> str:
    """Master data at the first week: what the network looked like before any message."""
    lines = [
        f"Master data as of week {world.start_period} (the messages report changes since then):",
        "Plants:",
    ]
    for plant in world.plants:
        supplier = world.entity(plant.supplier)
        lane = world.lane_of(plant.id)
        lines.append(
            f"- {plant.name} ({plant.id}), {supplier.name} ({supplier.id}): capacity "
            f"{plant.capacity:g} units per week; makes {', '.join(plant.skus)}; inbound lane "
            f"{lane.name} ({lane.id}), lead time {lane.lead_time_days:g} days"
        )
    lines.append("SKUs:")
    for sku in world.skus:
        lines.append(
            f"- {sku.name} ({sku.id}): demand {sku.base_demand:g} units per week; "
            f"starting stock {sku.initial_stock:g}"
        )
    lines.append("Prices per unit:")
    for offer in world.offers:
        lines.append(f"- {offer.name} ({offer.id}): {offer.price:.2f}")
    if world.lots:
        lines.append("Lots:")
        for lot in world.lots:
            arrival = "" if lot.arrival is None else f", arrives wk{lot.arrival}"
            lines.append(
                f"- {lot.name} ({lot.id}): {lot.quantity:g} units of {lot.sku}{arrival}, "
                f"{lot.hold_status}"
            )
    return "\n".join(lines)


def render_position(world: World, position: InventoryPosition) -> str:
    lines = [f"Week {position.period}. Inventory position:"]
    for sku in world.skus:
        lines.append(
            f"- {sku.name}: on hand {position.on_hand.get(sku.id, 0.0):.0f}, "
            f"backlog {position.backlog.get(sku.id, 0.0):.0f}"
        )
    for lot in world.lots:
        arriving = "" if lot.arrived(position.period) else f", arrives wk{lot.arrival}"
        lines.append(
            f"- {lot.name} ({lot.sku}): {position.lot_remaining.get(lot.id, 0.0):.0f} left"
            f"{arriving}"
        )
    for order in position.outstanding:
        lines.append(
            f"- open order: {order.quantity:.0f} {order.sku} from {order.plant}, "
            f"placed wk{order.placed}"
        )
    for supplier_id, (week, status) in sorted(position.status_notices.items()):
        supplier = world.entity(supplier_id)
        lines.append(
            f"- an order to {supplier.name} ({supplier_id}) was refused in wk{week}: "
            f"supplier {status}"
        )
    return "\n".join(lines)


def render_state(view: StateView, position: InventoryPosition, horizon: int) -> str:
    """The interpreted state over the planning horizon, as a typed decider reads it."""
    world = view.world
    periods = [p for p in range(position.period, position.period + horizon) if p in world.periods]
    notices = position.status_notices
    # Show only the notices the planner still uses: a status delta recorded after a refusal
    # replaces it (cdd.orders.supplier_status).
    current = {
        supplier: notice
        for supplier, notice in notices.items()
        if (recorded := view.last_recorded(supplier, Attribute.SUPPLIER_STATUS)) is None
        or recorded <= notice[0]
    }
    lines = [
        render_position(world, replace(position, status_notices=current)),
        "",
        "Plants (supplier, SKUs made, capacity per week, lead time):",
    ]
    for plant in world.plants:
        supplier = world.entity(plant.supplier)
        caps = ", ".join(
            f"wk{p} {capacity_available(view, plant.id, p, notices):.0f}" for p in periods
        )
        lead = lead_periods(view, plant.id, position.period)
        lines.append(
            f"- {plant.name} ({plant.id}), {supplier.name} ({supplier.id}): makes "
            f"{', '.join(plant.skus)}; {caps}; lead time {lead} week(s)"
        )
    lines.append("Demand forecast:")
    for sku in world.skus:
        demand = ", ".join(f"wk{p} {view.number(sku.id, Attribute.DEMAND, p):.0f}" for p in periods)
        lines.append(f"- {sku.name} ({sku.id}): {demand}")
    lines.append("Prices this week:")
    for offer in world.offers:
        price = view.number(offer.id, Attribute.PRICE, position.period)
        lines.append(f"- {offer.name} ({offer.id}): {price:.2f}")
    lines.append("Lot status this week:")
    for lot in world.lots:
        status = view.label(lot.id, Attribute.HOLD_STATUS, position.period)
        lines.append(f"- {lot.name} ({lot.id}): {status}")
    return "\n".join(lines)
