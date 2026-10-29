"""Delta-first stream generation (Section 5).

Ground truth comes first and text second: the generator samples gold delta sequences
(including hold→release, suspension→reinstatement and revision patterns), then renders each
delta through a surface template. Messages that change nothing are mixed in at a set rate.
Every stream is a pure function of the config and the stream index.

Benchmark version b2 calibrates the worlds so that misreading any delta type can cost
something (each setting below defaults to its b1 value):

- ``capacity_ratio``: each plant's capacity is a multiple of its share of the demand for the
  SKUs it makes, so a capacity cut can bind (b1 capacities were 5-10 times demand).
- ``min_suppliers_per_sku``: plants make SKUs assigned at random, with every SKU available
  from at least this many suppliers (b1 left SKU-04 with Supplier B alone).
- ``price_spread``: a SKU's offers lie within this share of one base price, so a 2-8% price
  change can reorder its sources.
- ``lot_weeks`` and ``lot_arrival``: lots hold several weeks of demand and reach the
  distribution centre during the horizon; holds go on lots that have just arrived or are
  about to.
- ``cross_week_boundary``: lead-time changes always change the lead time in whole weeks.
- ``plain_spikes``: a key-account spike is a plain one-week increase ("orders ... up 20% in
  wk40"); b1 said "pulls 20% forward", which also implies a later decrease the gold lacked.
- ``last_message_week``: no message after this week, so every change can still affect an
  order (orders placed in the last weeks arrive after the horizon).
- ``exclusive_windows``: two numeric changes to the same entity and attribute never overlap
  unless one revises the other. In b1, "demand for SKU-03 down 15% in weeks 41-43" followed by
  "up 15% in weeks 42-43" compounded in the gold, though a careful reader could take the second
  as a correction of the first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from cdd import weeks
from cdd.benchmark import templates as tpl
from cdd.benchmark.stream import Shock, Stream
from cdd.benchmark.templates import Template
from cdd.schema import (
    DELTA_SPECS,
    Delta,
    DeltaType,
    HoldStatus,
    Message,
    Op,
    Span,
    SupplierStatus,
    Value,
    Window,
    percent_to_scale,
)
from cdd.world import Lane, Lot, Offer, Plant, Sku, Supplier, World

T = TypeVar("T")

_LANE_NAMES = ("Port X", "Port Y", "Port Z", "Port W", "Port V", "Port U", "Port T", "Port S")


class WorldSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_period: int = 36
    end_period: int = 52
    calendar_year: int = 2026
    n_suppliers: int = 3
    plants_per_supplier: int = 2
    n_skus: int = 4
    skus_per_plant: int = 2
    lots_per_sku: int = 1
    capacity: tuple[float, float] = (250.0, 500.0)
    base_demand: tuple[float, float] = (60.0, 140.0)
    price: tuple[float, float] = (8.0, 20.0)
    lead_time_days: tuple[int, int] = (7, 21)
    lot_quantity: tuple[float, float] = (40.0, 120.0)
    initial_stock_weeks: float = 2.0
    holding_cost: float = 0.5
    backlog_cost: float = 25.0
    moq: float = 0.0
    budget_per_period: float | None = None
    # Calibration (benchmark b2); None or the default keeps the b1 behaviour.
    capacity_ratio: tuple[float, float] | None = None
    min_suppliers_per_sku: int = 1
    price_spread: float | None = None
    lot_weeks: tuple[float, float] | None = None
    lot_arrival: tuple[int, int] | None = None


class EventSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    rate: float = 0.9
    weights: dict[DeltaType, float] = {
        DeltaType.CAPACITY_CHANGE: 1.0,
        DeltaType.LEAD_TIME_CHANGE: 1.0,
        DeltaType.QUALITY_HOLD: 0.8,
        DeltaType.SUPPLIER_STATUS: 0.4,
        DeltaType.DEMAND_SIGNAL: 1.0,
        DeltaType.PRICE_CHANGE: 0.6,
    }
    increase_prob: float = 0.25
    revise_prob: float = 0.15
    release_prob: float = 0.8
    reinstate_prob: float = 0.7
    cross_week_boundary: bool = False
    plain_spikes: bool = False
    capacity_weeks: tuple[int, int] = (1, 3)  # length of a capacity change, in weeks
    exclusive_windows: bool = False


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "pilot"
    version: str = "b1"
    n_streams: int = 8
    seed: int = 0
    world: WorldSpec = WorldSpec()
    events: EventSpec = EventSpec()
    no_change_rate: float = 0.25
    distractor_rate: float = 0.2
    out_of_schema_rate: float = 0.0  # share of weeks with an out-of-schema yield shortfall
    last_message_week: int | None = None  # b2: no message after this week

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeneratorConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})


def _pick(rng: np.random.Generator, items: Sequence[T]) -> T:
    return items[int(rng.integers(len(items)))]


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return round(float(rng.uniform(*bounds)), 1)


def _assign_skus(
    spec: WorldSpec, rng: np.random.Generator, skus: Sequence[Sku]
) -> list[tuple[str, ...]]:
    """The SKUs each plant makes. b1: a fixed rotation. With ``min_suppliers_per_sku`` above
    one: a random draw, repeated until every SKU has plants of that many suppliers."""
    n_plants = spec.n_suppliers * spec.plants_per_supplier
    if spec.min_suppliers_per_sku <= 1:
        return [
            tuple(skus[(index + j) % len(skus)].id for j in range(spec.skus_per_plant))
            for index in range(n_plants)
        ]
    for _ in range(1000):
        made = [
            tuple(
                skus[int(i)].id
                for i in sorted(rng.choice(len(skus), size=spec.skus_per_plant, replace=False))
            )
            for _ in range(n_plants)
        ]
        suppliers: dict[str, set[int]] = {sku.id: set() for sku in skus}
        for index, made_here in enumerate(made):
            for sku_id in made_here:
                suppliers[sku_id].add(index // spec.plants_per_supplier)
        if all(len(s) >= spec.min_suppliers_per_sku for s in suppliers.values()):
            return made
    raise ValueError("no SKU assignment gives every SKU enough suppliers")


def make_world(spec: WorldSpec, rng: np.random.Generator) -> World:
    letters = [chr(ord("A") + i) for i in range(spec.n_suppliers)]
    suppliers = tuple(Supplier(id=f"SUP-{c}", name=f"Supplier {c}") for c in letters)
    sku_list = []
    for i in range(spec.n_skus):
        demand = _uniform(rng, spec.base_demand)
        sku_list.append(
            Sku(
                id=f"SKU-{i + 1:02d}",
                name=f"SKU-{i + 1:02d}",
                base_demand=demand,
                holding_cost=spec.holding_cost,
                backlog_cost=spec.backlog_cost,
                initial_stock=round(demand * spec.initial_stock_weeks, 1),
            )
        )
    skus = tuple(sku_list)
    assignment = _assign_skus(spec, rng, skus)
    makers = {sku.id: sum(sku.id in made for made in assignment) for sku in skus}
    demand_of = {sku.id: sku.base_demand for sku in skus}
    plants, lanes = [], []
    for s, supplier in enumerate(suppliers):
        for k in range(spec.plants_per_supplier):
            index = s * spec.plants_per_supplier + k
            made = assignment[index]
            if spec.capacity_ratio is None:
                capacity = _uniform(rng, spec.capacity)
            else:
                share = sum(demand_of[sku_id] / makers[sku_id] for sku_id in made)
                capacity = round(share * float(rng.uniform(*spec.capacity_ratio)), 1)
            plants.append(
                Plant(
                    id=f"P{index + 1}",
                    name=f"Plant {index + 1}",
                    supplier=supplier.id,
                    capacity=capacity,
                    skus=made,
                )
            )
            port = _LANE_NAMES[index] if index < len(_LANE_NAMES) else f"Port {index + 1}"
            lanes.append(
                Lane(
                    id=f"LANE-{port.split()[-1]}",
                    name=port,
                    plant=f"P{index + 1}",
                    lead_time_days=float(
                        rng.integers(spec.lead_time_days[0], spec.lead_time_days[1] + 1)
                    ),
                )
            )
    base_price = (
        {sku.id: _uniform(rng, spec.price) for sku in skus} if spec.price_spread is not None else {}
    )

    def price(sku_id: str) -> float:
        if spec.price_spread is None:
            return _uniform(rng, spec.price)
        spread = spec.price_spread
        return round(base_price[sku_id] * float(rng.uniform(1 - spread, 1 + spread)), 2)

    offers = tuple(
        Offer(
            id=f"{supplier.id}/{sku_id}",
            name=f"{supplier.name} {sku_id}",
            supplier=supplier.id,
            sku=sku_id,
            price=price(sku_id),
            moq=spec.moq,
        )
        for supplier in suppliers
        for sku_id in sorted({sku for p in plants if p.supplier == supplier.id for sku in p.skus})
    )
    lots = tuple(
        _lot(spec, rng, sku, 7701 + i * spec.lots_per_sku + j, j)
        for i, sku in enumerate(skus)
        for j in range(spec.lots_per_sku)
    )
    return World(
        start_period=spec.start_period,
        end_period=spec.end_period,
        calendar_year=spec.calendar_year,
        budget_per_period=spec.budget_per_period,
        suppliers=suppliers,
        plants=tuple(plants),
        lanes=tuple(lanes),
        skus=skus,
        offers=offers,
        lots=lots,
    )


def _lot(spec: WorldSpec, rng: np.random.Generator, sku: Sku, number: int, j: int) -> Lot:
    """b1: a small lot on hand from the start. With ``lot_weeks`` and ``lot_arrival``: several
    weeks of demand, arriving in the j-th part of the arrival range."""
    if spec.lot_weeks is None or spec.lot_arrival is None:
        quantity, arrival = _uniform(rng, spec.lot_quantity), None
    else:
        quantity = round(sku.base_demand * float(rng.uniform(*spec.lot_weeks)), 1)
        first, last = spec.lot_arrival
        width = (last - first + 1) / spec.lots_per_sku
        low = first + math.floor(j * width)
        high = max(low, first + math.floor((j + 1) * width) - 1)
        arrival = int(rng.integers(low, high + 1))
    return Lot(
        id=f"LOT-{number}",
        name=f"Lot {number}",
        sku=sku.id,
        quantity=quantity,
        hold_status=HoldStatus.RELEASED,
        arrival=arrival,
    )


@dataclass
class _Event:
    period: int
    template: Template
    fields: dict[str, str]
    sender: str
    sender_class: str
    type: DeltaType | None = None
    entity: str = ""
    op: Op = Op.SET
    value: Value = 0.0
    window: Window | None = None
    revises: _Event | None = None
    shock: dict[str, Any] | None = None
    order: int = 0
    delta_id: str = field(default="", init=False)


class _StreamBuilder:
    def __init__(self, config: GeneratorConfig, world: World, rng: np.random.Generator) -> None:
        self.config = config
        self.world = world
        self.rng = rng
        self.events: list[_Event] = []
        self.keys: set[tuple[str, str, int, int | None]] = set()
        self.windows: dict[tuple[str, str], list[Window]] = {}
        self.busy_until: dict[str, float] = {}
        # The last week in which any message (including a follow-up) may arrive.
        self.last_week = config.last_message_week or world.end_period

    # -- helpers ---------------------------------------------------------------------------

    def add(self, event: _Event) -> bool:
        """Queue an event unless it would contradict one already queued for the same key."""
        if event.type is not None and event.window is not None and event.revises is None:
            spec = DELTA_SPECS[event.type]
            key = (event.entity, spec.attribute.value, event.window.start, event.window.end)
            if key in self.keys:
                return False
            cell = (event.entity, spec.attribute.value)
            if self.config.events.exclusive_windows and not spec.categorical:
                if any(event.window.overlaps(w) for w in self.windows.get(cell, ())):
                    return False
                self.windows.setdefault(cell, []).append(event.window)
            self.keys.add(key)
        event.order = len(self.events)
        self.events.append(event)
        return True

    def range_phrase(self, start: int, stop: int) -> str:
        if start == stop:
            return f"wk{start}"
        return _pick(self.rng, tpl.RANGE_FORMS).format(a=start, b=stop)

    def from_phrase(self, start: int) -> str:
        return _pick(self.rng, tpl.FROM_FORMS).format(a=start)

    def window_ahead(self, t: int, lead: tuple[int, int], length: tuple[int, int]) -> Window | None:
        start = t + int(self.rng.integers(lead[0], lead[1] + 1))
        if start > self.world.end_period:
            return None
        stop = min(
            start + int(self.rng.integers(length[0], length[1] + 1)) - 1, self.world.end_period
        )
        return Window(start=start, end=stop)

    def free(self, entity: str, t: int) -> bool:
        return self.busy_until.get(entity, -1) < t

    # -- event samplers --------------------------------------------------------------------

    def capacity(self, t: int) -> None:
        plant = _pick(self.rng, self.world.plants)
        window = self.window_ahead(t, (1, 3), self.config.events.capacity_weeks)
        if window is None or window.end is None:
            return
        percent = _pick(self.rng, (10, 15, 20, 25, 30, 40))
        up = self.rng.random() < self.config.events.increase_prob
        supplier = self.world.entity(plant.supplier).name
        event = _Event(
            period=t,
            template=_pick(self.rng, tpl.CAPACITY_UP if up else tpl.CAPACITY_DOWN),
            fields={
                "plant": plant.name,
                "pct": str(percent),
                "window": self.range_phrase(window.start, window.end),
            },
            sender=f"{supplier} planning",
            sender_class="supplier_planning",
            type=DeltaType.CAPACITY_CHANGE,
            entity=plant.id,
            op=Op.SCALE,
            value=percent_to_scale(percent if up else -percent),
            window=window,
        )
        if not self.add(event) or up or t + 1 >= window.start or t + 1 > self.last_week:
            return
        if self.rng.random() < self.config.events.revise_prob:
            revised = _pick(self.rng, [p for p in (10, 15, 20, 25, 30, 40) if p != percent])
            self.add(
                _Event(
                    period=t + 1,
                    template=_pick(self.rng, tpl.CAPACITY_REVISION),
                    fields={**event.fields, "pct": str(revised)},
                    sender=event.sender,
                    sender_class=event.sender_class,
                    type=DeltaType.CAPACITY_CHANGE,
                    entity=plant.id,
                    op=Op.SCALE,
                    value=percent_to_scale(-revised),
                    window=window,
                    revises=event,
                )
            )

    def lead_time(self, t: int) -> None:
        lane = _pick(self.rng, self.world.lanes)
        down = self.rng.random() < 0.2
        options: tuple[int, ...] = (3, 5) if down else (3, 5, 6, 7, 10, 14)
        if self.config.events.cross_week_boundary:
            # Only changes that move the lead time into another whole week can matter.
            base, week = lane.lead_time_days, self.world.days_per_period
            sign = -1 if down else 1
            crossing = tuple(
                d for d in options if math.ceil((base + sign * d) / week) != math.ceil(base / week)
            )
            options = crossing or (7,)
        days = _pick(self.rng, options)
        form = _pick(self.rng, ("range", "from", "until_month"))
        year = self.world.calendar_year
        if form == "range":
            window = self.window_ahead(t, (1, 3), (2, 4))
            if window is None or window.end is None:
                return
            phrase = self.range_phrase(window.start, window.end)
        elif form == "from":
            window = Window(start=min(t + int(self.rng.integers(0, 3)), self.world.end_period))
            phrase = self.from_phrase(window.start)
        else:
            target = min(t + int(self.rng.integers(2, 7)), self.world.end_period)
            month = weeks.month_of_week(year, target)
            window = Window(start=t, end=max(t, weeks.last_week_of_month(year, month)))
            phrase = f"through {weeks.month_abbr(month)}"
        self.add(
            _Event(
                period=t,
                template=_pick(self.rng, tpl.LEAD_TIME_DOWN if down else tpl.LEAD_TIME_UP),
                fields={"lane": lane.name, "days": str(days), "window": phrase},
                sender="Carrier ops",
                sender_class="carrier",
                type=DeltaType.LEAD_TIME_CHANGE,
                entity=lane.id,
                op=Op.ADD,
                value=float(-days if down else days),
                window=window,
            )
        )

    def hold(self, t: int) -> None:
        # Lots that arrive during the horizon are held only around their arrival, when the
        # stock is still there to freeze.
        lots = [
            lot
            for lot in self.world.lots
            if self.free(lot.id, t) and (lot.arrival is None or t - 1 <= lot.arrival <= t + 3)
        ]
        if not lots:
            return
        lot = _pick(self.rng, lots)
        if not self.add(
            _Event(
                period=t,
                template=_pick(self.rng, tpl.HOLD),
                fields={"lot": lot.name},
                sender="QA",
                sender_class="supplier_quality",
                type=DeltaType.QUALITY_HOLD,
                entity=lot.id,
                value=HoldStatus.HELD.value,
                window=Window(start=t),
            )
        ):
            return
        self.busy_until[lot.id] = float("inf")
        release = t + int(self.rng.integers(1, 5))
        if self.rng.random() < self.config.events.release_prob and release <= self.last_week:
            self.busy_until[lot.id] = release
            self.add(
                _Event(
                    period=release,
                    template=_pick(self.rng, tpl.RELEASE),
                    fields={"lot": lot.name},
                    sender="QA",
                    sender_class="supplier_quality",
                    type=DeltaType.QUALITY_RELEASE,
                    entity=lot.id,
                    value=HoldStatus.RELEASED.value,
                    window=Window(start=release),
                )
            )

    def supplier_status(self, t: int) -> None:
        suppliers = [s for s in self.world.suppliers if self.free(s.id, t)]
        if not suppliers:
            return
        supplier = _pick(self.rng, suppliers)
        if self.rng.random() < 0.4:
            start = min(t + int(self.rng.integers(0, 3)), self.world.end_period)
            event = _Event(
                period=t,
                template=_pick(self.rng, tpl.RESTRICT),
                fields={"supplier": supplier.name, "window": self.from_phrase(start)},
                sender="Procurement",
                sender_class="procurement",
                type=DeltaType.SUPPLIER_STATUS,
                entity=supplier.id,
                value=SupplierStatus.RESTRICTED.value,
                window=Window(start=start),
            )
        else:
            start = t
            event = _Event(
                period=t,
                template=_pick(self.rng, tpl.SUSPEND),
                fields={"supplier": supplier.name},
                sender="Procurement",
                sender_class="procurement",
                type=DeltaType.SUPPLIER_STATUS,
                entity=supplier.id,
                value=SupplierStatus.SUSPENDED.value,
                window=Window(start=t),
            )
        if not self.add(event):
            return
        self.busy_until[supplier.id] = float("inf")
        back = start + int(self.rng.integers(2, 6))
        if self.rng.random() < self.config.events.reinstate_prob and back <= self.last_week:
            self.busy_until[supplier.id] = back
            self.add(
                _Event(
                    period=back,
                    template=_pick(self.rng, tpl.REINSTATE),
                    fields={"supplier": supplier.name},
                    sender="Procurement",
                    sender_class="procurement",
                    type=DeltaType.SUPPLIER_STATUS,
                    entity=supplier.id,
                    value=SupplierStatus.ACTIVE.value,
                    window=Window(start=back),
                )
            )

    def demand(self, t: int) -> None:
        sku = _pick(self.rng, self.world.skus)
        percent = _pick(self.rng, (10, 15, 20, 30))
        up = self.rng.random() < 0.6
        if up and self.rng.random() < 0.5:
            week = t + int(self.rng.integers(1, 4))
            if week > self.world.end_period:
                return
            window = Window(start=week, end=week)
            spikes = tpl.DEMAND_SPIKE if self.config.events.plain_spikes else tpl.DEMAND_PULL
            template = _pick(self.rng, spikes)
            fields = {"sku": sku.name, "pct": str(percent), "week": str(week)}
        else:
            ahead = self.window_ahead(t, (1, 3), (1, 3))
            if ahead is None or ahead.end is None:
                return
            window = ahead
            template = _pick(self.rng, tpl.DEMAND_UP if up else tpl.DEMAND_DOWN)
            fields = {
                "sku": sku.name,
                "pct": str(percent),
                "window": self.range_phrase(window.start, ahead.end),
            }
        self.add(
            _Event(
                period=t,
                template=template,
                fields=fields,
                sender="Sales ops",
                sender_class="sales",
                type=DeltaType.DEMAND_SIGNAL,
                entity=sku.id,
                op=Op.SCALE,
                value=percent_to_scale(percent if up else -percent),
                window=window,
            )
        )

    def price(self, t: int) -> None:
        offer = _pick(self.rng, self.world.offers)
        percent = _pick(self.rng, (2, 3, 4, 5, 6, 8))
        up = self.rng.random() < 0.75
        start = t + int(self.rng.integers(1, 4))
        if start > self.world.end_period:
            return
        if self.rng.random() < 0.5:
            thursday = date.fromisocalendar(self.world.calendar_year, start, 4)
            phrase = f"effective {weeks.month_abbr(thursday.month)} {thursday.day}"
        else:
            phrase = self.from_phrase(start)
        supplier = self.world.entity(offer.supplier).name
        self.add(
            _Event(
                period=t,
                template=_pick(self.rng, tpl.PRICE_UP if up else tpl.PRICE_DOWN),
                fields={
                    "supplier": supplier,
                    "sku": offer.sku,
                    "pct": str(percent),
                    "window": phrase,
                },
                sender=f"{supplier} billing",
                sender_class="supplier_finance",
                type=DeltaType.PRICE_CHANGE,
                entity=offer.id,
                op=Op.SCALE,
                value=percent_to_scale(percent if up else -percent),
                window=Window(start=start),
            )
        )

    def yield_shortfall(self, t: int) -> None:
        plant = _pick(self.rng, self.world.plants)
        window = self.window_ahead(t, (1, 2), (2, 3))
        if window is None or window.end is None:
            return
        percent = _pick(self.rng, (15, 20, 30, 40))
        supplier = self.world.entity(plant.supplier).name
        self.add(
            _Event(
                period=t,
                template=_pick(self.rng, tpl.OUT_OF_SCHEMA),
                fields={
                    "plant": plant.name,
                    "pct": str(percent),
                    "window": self.range_phrase(window.start, window.end),
                },
                sender=f"{supplier} logistics",
                sender_class="supplier_logistics",
                shock={
                    "kind": "yield_shortfall",
                    "plant": plant.id,
                    "window": window,
                    "factor": 1 - percent / 100,
                },
            )
        )

    def chatter(self, t: int) -> None:
        self.add(
            _Event(
                period=t,
                template=_pick(self.rng, tpl.NO_CHANGE),
                fields={
                    "plant": _pick(self.rng, self.world.plants).name,
                    "supplier": _pick(self.rng, self.world.suppliers).name,
                },
                sender="Ops desk",
                sender_class="internal",
            )
        )


def generate_stream(config: GeneratorConfig, index: int) -> Stream:
    rng = np.random.default_rng([config.seed, index])
    world = make_world(config.world, rng)
    builder = _StreamBuilder(config, world, rng)
    samplers = {
        DeltaType.CAPACITY_CHANGE: builder.capacity,
        DeltaType.LEAD_TIME_CHANGE: builder.lead_time,
        DeltaType.QUALITY_HOLD: builder.hold,
        DeltaType.SUPPLIER_STATUS: builder.supplier_status,
        DeltaType.DEMAND_SIGNAL: builder.demand,
        DeltaType.PRICE_CHANGE: builder.price,
    }
    kinds = [kind for kind in samplers if config.events.weights.get(kind, 0.0) > 0]
    weights = np.array([config.events.weights[kind] for kind in kinds], dtype=float)
    weights /= weights.sum()
    # Follow-ups (releases, reinstatements, revisions) come up to two weeks after their event.
    last_event_week = world.end_period - 1
    if config.last_message_week is not None:
        last_event_week = min(last_event_week, config.last_message_week - 2)
    for t in world.periods:
        if t > last_event_week:
            break
        for _ in range(int(rng.poisson(config.events.rate))):
            samplers[kinds[int(rng.choice(len(kinds), p=weights))]](t)
        if rng.random() < config.no_change_rate:
            builder.chatter(t)
        # Draw only when the slice is on, so streams without it stay identical.
        if config.out_of_schema_rate > 0 and rng.random() < config.out_of_schema_rate:
            builder.yield_shortfall(t)

    stream_id = f"{config.name}-{index:03d}"
    messages: list[Message] = []
    gold: list[Delta] = []
    shocks: list[Shock] = []
    for number, event in enumerate(sorted(builder.events, key=lambda e: (e.period, e.order))):
        message_id = f"{stream_id}-m{number:04d}"
        clause = event.template.clause.format(**event.fields)
        suffix = _pick(rng, tpl.DISTRACTORS) if rng.random() < config.distractor_rate else ""
        text = event.template.prefix + clause + suffix
        messages.append(
            Message(
                id=message_id,
                period=event.period,
                sender=event.sender,
                sender_class=event.sender_class,
                channel=event.template.channel,
                text=text,
            )
        )
        if event.shock is not None:
            shocks.append(Shock(message_id=message_id, **event.shock))
        if event.type is None or event.window is None:
            continue
        event.delta_id = f"{message_id}#g0"
        start = len(event.template.prefix)
        spec = DELTA_SPECS[event.type]
        gold.append(
            Delta(
                id=event.delta_id,
                type=event.type,
                entity=event.entity,
                attribute=spec.attribute,
                op=event.op,
                value=event.value,
                unit=None if spec.categorical else ("ratio" if event.op is Op.SCALE else spec.unit),
                window=event.window,
                src=Span(message_id=message_id, start=start, end=start + len(clause)),
                supersedes=(event.revises.delta_id,) if event.revises else (),
            )
        )
    return Stream(
        id=stream_id,
        seed=int(np.random.default_rng([config.seed, index, 1]).integers(2**31)),
        world=world,
        messages=tuple(messages),
        gold=tuple(gold),
        shocks=tuple(shocks),
        provenance={"benchmark": config.name, "benchmark_version": config.version},
    )


def generate(config: GeneratorConfig) -> list[Stream]:
    return [generate_stream(config, index) for index in range(config.n_streams)]
