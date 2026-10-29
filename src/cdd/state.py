"""Bitemporal state (L3): an append-only history of accepted deltas.

Valid time is the period a fact describes. Transaction time (``recorded_at``) is the period in
which the pipeline wrote it. Nothing is overwritten or deleted. :meth:`BitemporalState.value`
starts from the base value in ``s_0`` and applies, in recording order, every delta that was in
force at transaction time ``as_of`` and whose window covers the requested period.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from cdd.schema import Attribute, Delta, Op, Span, Value
from cdd.world import World


@dataclass(frozen=True)
class Assertion:
    seq: int
    recorded_at: int
    delta: Delta


@dataclass(frozen=True)
class Retraction:
    seq: int
    recorded_at: int
    delta_id: str
    by: str


@dataclass(frozen=True)
class Merge:
    """A duplicate delta whose provenance was added to an existing one instead of asserted."""

    seq: int
    recorded_at: int
    delta_id: str
    into: str
    source: Span


@dataclass(frozen=True)
class ConflictRecord:
    seq: int
    recorded_at: int
    delta_ids: tuple[str, ...]


HistoryEntry = Assertion | Retraction | Merge | ConflictRecord


def apply_op(op: Op, current: Value, value: Value) -> Value:
    if op is Op.SET:
        return value
    if isinstance(current, str) or isinstance(value, str):
        raise TypeError(f"{op} needs numeric operands, got {current!r} and {value!r}")
    if op is Op.ADD:
        return current + value
    return current * value


def _cutoff(as_of: int | None) -> float:
    return math.inf if as_of is None else as_of


class BitemporalState:
    def __init__(self, world: World) -> None:
        self.world = world
        self._history: list[HistoryEntry] = []
        self._assertions: dict[str, Assertion] = {}
        self._by_key: dict[tuple[str, Attribute], list[Assertion]] = defaultdict(list)
        self._retracted_at: dict[str, int] = {}
        self._merged_sources: dict[str, list[Span]] = defaultdict(list)

    @property
    def history(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._history)

    @property
    def version(self) -> int:
        return len(self._history)

    def assert_delta(self, delta: Delta, recorded_at: int) -> Assertion:
        if delta.id in self._assertions:
            raise ValueError(f"delta {delta.id!r} is already recorded")
        entry = Assertion(self.version, recorded_at, delta)
        self._history.append(entry)
        self._assertions[delta.id] = entry
        self._by_key[(delta.entity, delta.attribute)].append(entry)
        return entry

    def retract(self, delta_id: str, by: str, recorded_at: int) -> Retraction | None:
        if not self.is_active(delta_id, as_of=recorded_at):
            return None
        entry = Retraction(self.version, recorded_at, delta_id, by)
        self._history.append(entry)
        self._retracted_at[delta_id] = recorded_at
        return entry

    def merge(self, delta_id: str, into: str, source: Span, recorded_at: int) -> Merge:
        entry = Merge(self.version, recorded_at, delta_id, into, source)
        self._history.append(entry)
        self._merged_sources[into].append(source)
        return entry

    def record_conflict(self, delta_ids: Sequence[str], recorded_at: int) -> ConflictRecord:
        entry = ConflictRecord(self.version, recorded_at, tuple(delta_ids))
        self._history.append(entry)
        return entry

    def assertion(self, delta_id: str) -> Assertion | None:
        return self._assertions.get(delta_id)

    def sources(self, delta_id: str) -> tuple[Span, ...]:
        entry = self._assertions[delta_id]
        return (entry.delta.src, *self._merged_sources.get(delta_id, ()))

    def is_active(self, delta_id: str, as_of: int | None = None) -> bool:
        entry = self._assertions.get(delta_id)
        if entry is None or entry.recorded_at > _cutoff(as_of):
            return False
        retracted = self._retracted_at.get(delta_id)
        return retracted is None or retracted > _cutoff(as_of)

    def active(self, as_of: int | None = None) -> list[Delta]:
        return [
            entry.delta
            for entry in self._history
            if isinstance(entry, Assertion) and self.is_active(entry.delta.id, as_of)
        ]

    def value(
        self, entity: str, attribute: Attribute, period: int, as_of: int | None = None
    ) -> Value:
        value = self.world.base_value(entity, attribute)
        for entry in self._by_key.get((entity, attribute), ()):
            delta = entry.delta
            if delta.window.contains(period) and self.is_active(delta.id, as_of):
                value = apply_op(delta.op, value, delta.value)
        return value

    def last_recorded(
        self, entity: str, attribute: Attribute, as_of: int | None = None
    ) -> int | None:
        """The latest period in which a delta on this cell was recorded (None if never)."""
        periods = [
            entry.recorded_at
            for entry in self._by_key.get((entity, attribute), ())
            if entry.recorded_at <= _cutoff(as_of)
        ]
        return max(periods, default=None)

    def view(self, as_of: int | None = None) -> StateView:
        return StateView(self, as_of)


@dataclass(frozen=True)
class StateView:
    """Read-only access to the state as known at transaction time ``as_of`` (None: all time)."""

    state: BitemporalState
    as_of: int | None

    @property
    def world(self) -> World:
        return self.state.world

    def value(self, entity: str, attribute: Attribute, period: int) -> Value:
        return self.state.value(entity, attribute, period, self.as_of)

    def number(self, entity: str, attribute: Attribute, period: int) -> float:
        value = self.value(entity, attribute, period)
        if isinstance(value, str):
            raise TypeError(f"{entity}.{attribute} is categorical ({value!r})")
        return float(value)

    def label(self, entity: str, attribute: Attribute, period: int) -> str:
        value = self.value(entity, attribute, period)
        if not isinstance(value, str):
            raise TypeError(f"{entity}.{attribute} is numeric ({value!r})")
        return value

    def active(self) -> list[Delta]:
        return self.state.active(self.as_of)

    def last_recorded(self, entity: str, attribute: Attribute) -> int | None:
        return self.state.last_recorded(entity, attribute, self.as_of)
