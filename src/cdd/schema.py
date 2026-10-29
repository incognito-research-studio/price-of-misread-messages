"""Typed delta schema, version 1 (Table 1 of the paper).

The interpreter (L1) emits :class:`Proposal` objects: a delta type, an entity and a supporting
span. The span parser (L4) completes each proposal into a :class:`Delta` by filling the
operation, value, unit and validity window from the span. A message that changes nothing maps
to an empty set of proposals.

Delta ids follow the convention ``"<message_id>#<suffix>"``, so the source message of any delta
can be recovered from its id alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "v1"

Value = float | str


class DeltaType(StrEnum):
    """The closed set of delta types. Abstaining is an empty set, not a type."""

    CAPACITY_CHANGE = "capacity_change"
    LEAD_TIME_CHANGE = "lead_time_change"
    QUALITY_HOLD = "quality_hold"
    QUALITY_RELEASE = "quality_release"
    SUPPLIER_STATUS = "supplier_status"
    DEMAND_SIGNAL = "demand_signal"
    PRICE_CHANGE = "price_change"


class Op(StrEnum):
    SET = "set"
    ADD = "add"
    SCALE = "scale"


class EntityKind(StrEnum):
    SUPPLIER = "supplier"
    PLANT = "plant"
    LANE = "lane"
    SKU = "sku"
    OFFER = "offer"
    LOT = "lot"


class Attribute(StrEnum):
    CAPACITY = "capacity"
    LEAD_TIME = "lead_time"
    HOLD_STATUS = "hold_status"
    SUPPLIER_STATUS = "supplier_status"
    DEMAND = "demand"
    PRICE = "price"


class HoldStatus(StrEnum):
    HELD = "held"
    RELEASED = "released"


class SupplierStatus(StrEnum):
    ACTIVE = "active"
    RESTRICTED = "restricted"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class DeltaSpec:
    """What a delta type may touch: one attribute of one entity kind, through the listed ops."""

    attribute: Attribute
    entity_kind: EntityKind
    ops: frozenset[Op]
    unit: str | None = None
    categories: tuple[str, ...] = ()

    @property
    def categorical(self) -> bool:
        return bool(self.categories)


DELTA_SPECS: dict[DeltaType, DeltaSpec] = {
    DeltaType.CAPACITY_CHANGE: DeltaSpec(
        Attribute.CAPACITY, EntityKind.PLANT, frozenset(Op), unit="units/period"
    ),
    DeltaType.LEAD_TIME_CHANGE: DeltaSpec(
        Attribute.LEAD_TIME, EntityKind.LANE, frozenset({Op.SET, Op.ADD}), unit="days"
    ),
    DeltaType.QUALITY_HOLD: DeltaSpec(
        Attribute.HOLD_STATUS, EntityKind.LOT, frozenset({Op.SET}), categories=(HoldStatus.HELD,)
    ),
    DeltaType.QUALITY_RELEASE: DeltaSpec(
        Attribute.HOLD_STATUS,
        EntityKind.LOT,
        frozenset({Op.SET}),
        categories=(HoldStatus.RELEASED,),
    ),
    DeltaType.SUPPLIER_STATUS: DeltaSpec(
        Attribute.SUPPLIER_STATUS,
        EntityKind.SUPPLIER,
        frozenset({Op.SET}),
        categories=tuple(SupplierStatus),
    ),
    DeltaType.DEMAND_SIGNAL: DeltaSpec(
        Attribute.DEMAND, EntityKind.SKU, frozenset(Op), unit="units/period"
    ),
    DeltaType.PRICE_CHANGE: DeltaSpec(
        Attribute.PRICE, EntityKind.OFFER, frozenset(Op), unit="currency/unit"
    ),
}


def percent_to_scale(percent: float) -> float:
    """Convert a signed percentage change into a scale factor (``-15`` becomes ``0.85``)."""
    return 1.0 + percent / 100.0


class Window(BaseModel):
    """Validity window in periods, inclusive at both ends. ``end=None`` means open-ended."""

    model_config = ConfigDict(frozen=True)

    start: int
    end: int | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end is not None and self.end < self.start:
            raise ValueError(f"window ends ({self.end}) before it starts ({self.start})")
        return self

    def contains(self, period: int) -> bool:
        return self.start <= period and (self.end is None or period <= self.end)

    def overlaps(self, other: Window) -> bool:
        ends = [end for end in (self.end, other.end) if end is not None]
        return not ends or max(self.start, other.start) <= min(ends)


class Span(BaseModel):
    """Character offsets ``[start, end)`` into the text of one message."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"empty span [{self.start}, {self.end})")
        return self

    def overlaps(self, other: Span) -> bool:
        return (
            self.message_id == other.message_id
            and self.start < other.end
            and other.start < self.end
        )


class Message(BaseModel):
    """One unstructured operational message: an email, portal notice, carrier update or note."""

    model_config = ConfigDict(frozen=True)

    id: str
    period: int
    sender: str
    sender_class: str
    channel: str
    text: str

    def excerpt(self, span: Span) -> str:
        return self.text[span.start : span.end]


class Proposal(BaseModel):
    """L1 output: a delta type, an entity and the span that supports them.

    The remaining slots are optional. The span parser fills any that are missing, so an error in
    the span is attributable separately from an error in the type or entity. ``delta_id`` is set
    only by interpreters that replay known deltas (the oracle and attribution replays); otherwise
    the pipeline assigns ``"<message_id>#<index>"``.
    """

    model_config = ConfigDict(frozen=True)

    type: DeltaType
    entity: str
    span: Span
    op: Op | None = None
    value: Value | None = None
    unit: str | None = None
    window: Window | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    supersedes: tuple[str, ...] = ()
    hints: dict[str, str] = Field(default_factory=dict)
    delta_id: str | None = None

    @classmethod
    def from_delta(cls, delta: Delta) -> Proposal:
        return cls(
            type=delta.type,
            entity=delta.entity,
            span=delta.src,
            op=delta.op,
            value=delta.value,
            unit=delta.unit,
            window=delta.window,
            confidence=delta.confidence,
            supersedes=delta.supersedes,
            delta_id=delta.id,
        )


class Delta(BaseModel):
    """A complete typed state delta ``(id, τ, e, a, op, v, u, w, src, κ, sup)``."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: DeltaType
    entity: str
    attribute: Attribute
    op: Op
    value: Value
    unit: str | None = None
    window: Window
    src: Span
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    supersedes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _conforms_to_spec(self) -> Self:
        spec = DELTA_SPECS[self.type]
        if self.attribute != spec.attribute:
            raise ValueError(f"{self.type} writes {spec.attribute}, not {self.attribute}")
        if self.op not in spec.ops:
            raise ValueError(f"{self.type} does not allow op {self.op}")
        if spec.categorical:
            if not isinstance(self.value, str) or self.value not in spec.categories:
                raise ValueError(f"{self.type} value must be one of {spec.categories}")
        else:
            if isinstance(self.value, str):
                raise ValueError(f"{self.type} needs a numeric value, got {self.value!r}")
            if self.op is Op.SCALE and self.value <= 0:
                raise ValueError("scale factors must be positive")
        return self

    @property
    def message_id(self) -> str:
        return self.src.message_id

    @property
    def effect(self) -> tuple[Any, ...]:
        """Everything that determines how the delta changes state. Equal effects are duplicates."""
        return (
            self.type,
            self.entity,
            self.attribute,
            self.op,
            self.value,
            self.window.start,
            self.window.end,
        )


def message_of(delta_id: str) -> str:
    """Recover the source message id from a delta id (see the module docstring)."""
    return delta_id.split("#", 1)[0]


def proposal_json_schema() -> dict[str, Any]:
    """JSON Schema for interpreter output, for structured-output and typed-model adapters."""
    schema = Proposal.model_json_schema()
    schema["$comment"] = f"compile-dont-decide delta schema {SCHEMA_VERSION}"
    return schema
