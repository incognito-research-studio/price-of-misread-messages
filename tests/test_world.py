from __future__ import annotations

import pytest
from pydantic import ValidationError

from cdd.schema import Attribute, EntityKind
from cdd.world import Lane, Supplier, World
from helpers import build_world


def test_lookups(world: World) -> None:
    assert world.kind_of("P2") is EntityKind.PLANT
    assert world.kind_of("nope") is None
    assert world.lane_of("P3").id == "LANE-Y"
    assert world.lines() == (("P2", "SKU-01"), ("P3", "SKU-01"))
    assert world.offer_for("P2", "SKU-01") is not None
    assert list(world.periods) == [40, 41, 42, 43, 44, 45]


def test_base_values(world: World) -> None:
    assert world.base_value("P2", Attribute.CAPACITY) == 1000.0
    assert world.base_value("LOT-7781", Attribute.HOLD_STATUS) == "released"
    assert world.base_value("SUP-C", Attribute.SUPPLIER_STATUS) == "active"
    with pytest.raises(KeyError):
        world.base_value("P2", Attribute.PRICE)


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        build_world(suppliers=(Supplier(id="P2", name="Clash"), Supplier(id="SUP-C", name="C")))


def test_every_plant_needs_one_lane() -> None:
    with pytest.raises(ValidationError, match="lane"):
        build_world(lanes=(Lane(id="LANE-X", name="Port X", plant="P2", lead_time_days=7.0),))


def test_round_trips_through_json(world: World) -> None:
    copy = World.model_validate_json(world.model_dump_json())
    assert copy.kind_of("LOT-7781") is EntityKind.LOT
