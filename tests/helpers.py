"""Builders shared by the tests."""

from __future__ import annotations

from typing import Any

from cdd.schema import DELTA_SPECS, Attribute, Delta, DeltaType, Message, Op, Span, Window
from cdd.world import Lane, Lot, Offer, Plant, Sku, Supplier, World


def build_world(**overrides: Any) -> World:
    """Two suppliers with one plant each, one SKU and one lot. Every lead time is one week."""
    fields: dict[str, Any] = {
        "start_period": 40,
        "end_period": 45,
        "calendar_year": 2026,
        "suppliers": (
            Supplier(id="SUP-C", name="Supplier C"),
            Supplier(id="SUP-D", name="Supplier D"),
        ),
        "plants": (
            Plant(id="P2", name="Plant 2", supplier="SUP-C", capacity=1000.0, skus=("SKU-01",)),
            Plant(id="P3", name="Plant 3", supplier="SUP-D", capacity=500.0, skus=("SKU-01",)),
        ),
        "lanes": (
            Lane(id="LANE-X", name="Port X", plant="P2", lead_time_days=7.0),
            Lane(id="LANE-Y", name="Port Y", plant="P3", lead_time_days=7.0),
        ),
        "skus": (
            Sku(
                id="SKU-01",
                name="SKU-01",
                base_demand=100.0,
                holding_cost=0.5,
                backlog_cost=25.0,
                initial_stock=0.0,
            ),
        ),
        "offers": (
            Offer(
                id="SUP-C/SKU-01",
                name="Supplier C SKU-01",
                supplier="SUP-C",
                sku="SKU-01",
                price=10.0,
            ),
            Offer(
                id="SUP-D/SKU-01",
                name="Supplier D SKU-01",
                supplier="SUP-D",
                sku="SKU-01",
                price=12.0,
            ),
        ),
        "lots": (Lot(id="LOT-7781", name="Lot 7781", sku="SKU-01", quantity=50.0),),
    }
    fields.update(overrides)
    return World(**fields)


def make_message(
    text: str, *, id: str = "m1", period: int = 40, sender_class: str = "procurement"
) -> Message:
    return Message(
        id=id, period=period, sender="Tester", sender_class=sender_class, channel="email", text=text
    )


def make_delta(**overrides: Any) -> Delta:
    """A 15% capacity cut on Plant 2 in weeks 42-44, unless overridden."""
    fields: dict[str, Any] = {
        "id": "m1#0",
        "type": DeltaType.CAPACITY_CHANGE,
        "entity": "P2",
        "attribute": Attribute.CAPACITY,
        "op": Op.SCALE,
        "value": 0.85,
        "unit": "ratio",
        "window": Window(start=42, end=44),
        "src": Span(message_id="m1", start=0, end=20),
    }
    fields.update(overrides)
    if "type" in overrides and "attribute" not in overrides:
        fields["attribute"] = DELTA_SPECS[fields["type"]].attribute
    return Delta(**fields)
