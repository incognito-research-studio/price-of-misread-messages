"""Static entities, their relations and initial attribute values (the state ``s_0``).

In schema v1 the graph structure (entities ``V`` and relations ``E``) is fixed for a stream;
only attribute values change over time, through deltas.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from cdd.schema import Attribute, EntityKind, HoldStatus, SupplierStatus, Value


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    aliases: tuple[str, ...] = ()

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


class Supplier(Entity):
    status: SupplierStatus = SupplierStatus.ACTIVE


class Plant(Entity):
    supplier: str
    capacity: float
    skus: tuple[str, ...]


class Lane(Entity):
    """Inbound lane from one plant to the distribution centre."""

    plant: str
    lead_time_days: float


class Sku(Entity):
    base_demand: float
    holding_cost: float = 1.0
    backlog_cost: float = 10.0
    initial_stock: float = 0.0


class Offer(Entity):
    """The price line of one supplier for one SKU."""

    supplier: str
    sku: str
    price: float
    moq: float = 0.0


class Lot(Entity):
    """A lot of finished stock for one SKU. ``arrival`` is the period it reaches the
    distribution centre (None: on hand from the start); before that it cannot be used."""

    sku: str
    quantity: float
    hold_status: HoldStatus = HoldStatus.RELEASED
    arrival: int | None = None

    def arrived(self, period: int) -> bool:
        return self.arrival is None or self.arrival <= period


class World(BaseModel):
    """Entities, relations and planning parameters for one stream. Periods are ISO weeks."""

    model_config = ConfigDict(frozen=True)

    start_period: int
    end_period: int
    calendar_year: int = 2026
    days_per_period: int = 7
    budget_per_period: float | None = None
    restricted_capacity_factor: float = 0.5
    suppliers: tuple[Supplier, ...]
    plants: tuple[Plant, ...]
    lanes: tuple[Lane, ...]
    skus: tuple[Sku, ...]
    offers: tuple[Offer, ...]
    lots: tuple[Lot, ...] = ()

    _kind: dict[str, EntityKind] = PrivateAttr(default_factory=dict)
    _by_id: dict[str, Entity] = PrivateAttr(default_factory=dict)
    _lane_of: dict[str, Lane] = PrivateAttr(default_factory=dict)
    _offer_of: dict[tuple[str, str], Offer] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.end_period < self.start_period:
            raise ValueError("end_period precedes start_period")
        seen: set[str] = set()
        for _, group in self.groups():
            for entity in group:
                if entity.id in seen:
                    raise ValueError(f"duplicate entity id {entity.id!r}")
                seen.add(entity.id)
        suppliers = {s.id for s in self.suppliers}
        skus = {s.id for s in self.skus}
        plants = {p.id for p in self.plants}
        for plant in self.plants:
            _require(plant.supplier in suppliers, f"plant {plant.id} has unknown supplier")
            _require(set(plant.skus) <= skus, f"plant {plant.id} lists an unknown SKU")
        lane_plants = [lane.plant for lane in self.lanes]
        _require(set(lane_plants) <= plants, "a lane references an unknown plant")
        _require(
            sorted(lane_plants) == sorted(plants), "every plant needs exactly one inbound lane"
        )
        for offer in self.offers:
            _require(offer.supplier in suppliers, f"offer {offer.id} has unknown supplier")
            _require(offer.sku in skus, f"offer {offer.id} has unknown SKU")
        for lot in self.lots:
            _require(lot.sku in skus, f"lot {lot.id} has unknown SKU")
        return self

    def model_post_init(self, context: Any, /) -> None:
        for kind, group in self.groups():
            for entity in group:
                self._kind[entity.id] = kind
                self._by_id[entity.id] = entity
        self._lane_of = {lane.plant: lane for lane in self.lanes}
        self._offer_of = {(offer.supplier, offer.sku): offer for offer in self.offers}

    def groups(self) -> tuple[tuple[EntityKind, tuple[Entity, ...]], ...]:
        return (
            (EntityKind.SUPPLIER, self.suppliers),
            (EntityKind.PLANT, self.plants),
            (EntityKind.LANE, self.lanes),
            (EntityKind.SKU, self.skus),
            (EntityKind.OFFER, self.offers),
            (EntityKind.LOT, self.lots),
        )

    def entities(self, kind: EntityKind | None = None) -> Iterator[Entity]:
        for group_kind, group in self.groups():
            if kind is None or kind is group_kind:
                yield from group

    @property
    def periods(self) -> range:
        return range(self.start_period, self.end_period + 1)

    def kind_of(self, entity_id: str) -> EntityKind | None:
        return self._kind.get(entity_id)

    def entity(self, entity_id: str) -> Entity:
        return self._by_id[entity_id]

    def plant(self, plant_id: str) -> Plant:
        return _typed(self._by_id[plant_id], Plant)

    def sku(self, sku_id: str) -> Sku:
        return _typed(self._by_id[sku_id], Sku)

    def lot(self, lot_id: str) -> Lot:
        return _typed(self._by_id[lot_id], Lot)

    def lane_of(self, plant_id: str) -> Lane:
        return self._lane_of[plant_id]

    def offer_for(self, plant_id: str, sku_id: str) -> Offer | None:
        return self._offer_of.get((self.plant(plant_id).supplier, sku_id))

    def offer_by_parts(self, supplier_id: str, sku_id: str) -> Offer | None:
        return self._offer_of.get((supplier_id, sku_id))

    def lines(self) -> tuple[tuple[str, str], ...]:
        """Every (plant, SKU) pair that can be ordered: the plant makes it and has a price line."""
        return tuple(
            (plant.id, sku)
            for plant in self.plants
            for sku in plant.skus
            if self.offer_for(plant.id, sku) is not None
        )

    def base_value(self, entity_id: str, attribute: Attribute) -> Value:
        entity = self._by_id[entity_id]
        if attribute is Attribute.CAPACITY and isinstance(entity, Plant):
            return entity.capacity
        if attribute is Attribute.LEAD_TIME and isinstance(entity, Lane):
            return entity.lead_time_days
        if attribute is Attribute.HOLD_STATUS and isinstance(entity, Lot):
            return entity.hold_status.value
        if attribute is Attribute.SUPPLIER_STATUS and isinstance(entity, Supplier):
            return entity.status.value
        if attribute is Attribute.DEMAND and isinstance(entity, Sku):
            return entity.base_demand
        if attribute is Attribute.PRICE and isinstance(entity, Offer):
            return entity.price
        raise KeyError(f"{entity_id} has no attribute {attribute}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


E = TypeVar("E", bound=Entity)


def _typed(entity: Entity, cls: type[E]) -> E:
    if not isinstance(entity, cls):
        raise KeyError(f"{entity.id} is not a {cls.__name__}")
    return entity
