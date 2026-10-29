"""Orders, plans and inventory positions, plus the two physical rules that the planner and the
simulator share: how much a plant can ship in a period, and how long a lane takes.

When a plant refuses an order because its supplier is suspended or restricted, the refusal
says so: the position carries a *status notice* (the week and the status) from then on. A
planner reading the interpreted state uses the notice in place of the state's supplier status
until a status delta for that supplier is recorded after the refusal. Capacity shortfalls
of an active supplier are refused without a notice."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from cdd.schema import Attribute, SupplierStatus
from cdd.state import StateView


@dataclass(frozen=True)
class OrderLine:
    plant: str
    sku: str
    quantity: float


@dataclass(frozen=True)
class Plan:
    """Orders to place in ``period``. ``duals`` maps constraint names to shadow prices;
    ``invalid_lines`` counts order lines a model decider wrote for a plant that does not make
    the SKU, which are dropped."""

    period: int
    orders: tuple[OrderLine, ...] = ()
    objective: float | None = None
    status: str = "optimal"
    duals: Mapping[str, float] | None = None
    invalid_lines: int = 0


@dataclass(frozen=True)
class OutstandingOrder:
    placed: int
    plant: str
    sku: str
    quantity: float


@dataclass(frozen=True)
class InventoryPosition:
    """What the planner may observe at the start of a period (ERP data, not messages)."""

    period: int
    on_hand: Mapping[str, float]
    lot_remaining: Mapping[str, float]
    backlog: Mapping[str, float]
    outstanding: tuple[OutstandingOrder, ...] = field(default=())
    # supplier id -> (week of the refused order, status the refusal reported)
    status_notices: Mapping[str, tuple[int, str]] = field(default_factory=dict)


def supplier_status(
    view: StateView,
    supplier_id: str,
    period: int,
    notices: Mapping[str, tuple[int, str]] | None = None,
) -> str:
    """The supplier status to plan with: the state's, unless a refusal reported a status after
    the state's latest status delta for this supplier."""
    notice = (notices or {}).get(supplier_id)
    if notice is not None:
        recorded = view.last_recorded(supplier_id, Attribute.SUPPLIER_STATUS)
        if recorded is None or recorded <= notice[0]:
            return notice[1]
    return view.label(supplier_id, Attribute.SUPPLIER_STATUS, period)


def capacity_available(
    view: StateView,
    plant_id: str,
    period: int,
    notices: Mapping[str, tuple[int, str]] | None = None,
) -> float:
    plant = view.world.plant(plant_id)
    status = supplier_status(view, plant.supplier, period, notices)
    if status == SupplierStatus.SUSPENDED:
        return 0.0
    capacity = max(0.0, view.number(plant_id, Attribute.CAPACITY, period))
    if status == SupplierStatus.RESTRICTED:
        capacity *= view.world.restricted_capacity_factor
    return capacity


def lead_periods(view: StateView, plant_id: str, period: int) -> int:
    lane = view.world.lane_of(plant_id)
    days = max(0.0, view.number(lane.id, Attribute.LEAD_TIME, period))
    return math.ceil(days / view.world.days_per_period)
