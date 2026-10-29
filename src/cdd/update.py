"""The deterministic update operator ``s_{t+1} = s_t ⊕ D_t`` (Section 3.5).

The update runs in four steps: canonicalise and sort the batch, resolve duplicates and
contradictions, write new versions (never deleting), and log the diff. Because every step is
deterministic, the interpreted state ŝ is a pure function of ``s_0`` and the interpreter outputs.

A ``supersedes`` reference retracts the delta it names only when that delta sets the same
entity and attribute; a reference to another entity's delta is recorded as unresolved, so a
misread link cannot delete an unrelated fact. A delta whose ``supersedes`` references retract
nothing (there are none, or none names a delta on its cell) and that sets the same entity,
attribute and window as a delta recorded in an earlier period, with a different effect,
restates it and replaces it (an unlinked revision), instead of compounding with it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from cdd.schema import Attribute, Delta, Value, Window
from cdd.state import BitemporalState


@dataclass(frozen=True)
class Conflict:
    """Deltas that set the same attribute over the same window to different effects.

    None of them is applied. The pipeline routes the conflict to the approver.
    """

    entity: str
    attribute: Attribute
    window: Window
    deltas: tuple[Delta, ...]


@dataclass(frozen=True)
class DiffEntry:
    entity: str
    attribute: Attribute
    period: int
    before: Value
    after: Value


@dataclass(frozen=True)
class UpdateResult:
    asserted: tuple[str, ...] = ()
    merged: tuple[tuple[str, str], ...] = ()
    retracted: tuple[str, ...] = ()
    unresolved_supersedes: tuple[str, ...] = ()
    restated: tuple[tuple[str, str], ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    diff: tuple[DiffEntry, ...] = ()


def canonical_order(deltas: Iterable[Delta]) -> list[Delta]:
    def key(delta: Delta) -> tuple[Any, ...]:
        end = math.inf if delta.window.end is None else delta.window.end
        return (
            delta.entity,
            delta.attribute,
            delta.window.start,
            end,
            delta.type,
            delta.op,
            str(delta.value),
            delta.id,
        )

    return sorted(deltas, key=key)


def apply_update(state: BitemporalState, deltas: Iterable[Delta], recorded_at: int) -> UpdateResult:
    batch = canonical_order(deltas)
    earlier = state.active(as_of=recorded_at)

    # Duplicates of each other, or of a delta already in force, are merged into the first one.
    in_force = {delta.effect: delta.id for delta in earlier}
    unique: list[Delta] = []
    merges: list[tuple[Delta, str]] = []
    for delta in batch:
        kept = in_force.get(delta.effect)
        if kept is None:
            in_force[delta.effect] = delta.id
            unique.append(delta)
        else:
            merges.append((delta, kept))

    # Same entity, attribute and window with different effects, and no supersession between
    # them, is an unreconciled contradiction.
    groups: dict[tuple[Any, ...], list[Delta]] = defaultdict(list)
    for delta in unique:
        groups[(delta.entity, delta.attribute, delta.window.start, delta.window.end)].append(delta)
    conflicts: list[Conflict] = []
    conflicted: set[str] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        ids = {member.id for member in members}
        if any(ids.intersection(member.supersedes) for member in members):
            continue
        first = members[0]
        conflicts.append(Conflict(first.entity, first.attribute, first.window, tuple(members)))
        conflicted |= ids
    writable = [delta for delta in unique if delta.id not in conflicted]

    touched = _touched(state, writable)
    before = {key: state.value(*key, as_of=recorded_at) for key in touched}

    for delta in writable:
        state.assert_delta(delta, recorded_at)
    for delta, kept in merges:
        state.merge(delta.id, kept, delta.src, recorded_at)
    retracted: list[str] = []
    unresolved: list[str] = []
    restated: list[tuple[str, str]] = []
    for delta in writable:
        linked = False
        for old in delta.supersedes:
            entry = state.assertion(old)
            same_cell = entry is not None and _same_cell(entry.delta, delta)
            if same_cell and state.retract(old, by=delta.id, recorded_at=recorded_at) is not None:
                retracted.append(old)
                linked = True
            else:
                unresolved.append(old)
        if linked:
            continue
        for old_delta in earlier:
            same = _same_cell(old_delta, delta) and old_delta.window == delta.window
            if same and state.retract(old_delta.id, by=delta.id, recorded_at=recorded_at):
                restated.append((old_delta.id, delta.id))
    for conflict in conflicts:
        state.record_conflict([delta.id for delta in conflict.deltas], recorded_at)

    diff = []
    for key in sorted(touched):
        after = state.value(*key, as_of=recorded_at)
        if after != before[key]:
            diff.append(DiffEntry(key[0], key[1], key[2], before[key], after))

    return UpdateResult(
        asserted=tuple(delta.id for delta in writable),
        merged=tuple((delta.id, kept) for delta, kept in merges),
        retracted=tuple(retracted),
        unresolved_supersedes=tuple(unresolved),
        restated=tuple(restated),
        conflicts=tuple(conflicts),
        diff=tuple(diff),
    )


def _same_cell(a: Delta, b: Delta) -> bool:
    return (a.entity, a.attribute) == (b.entity, b.attribute)


def _touched(state: BitemporalState, deltas: list[Delta]) -> set[tuple[str, Attribute, int]]:
    periods = state.world.periods
    touched: set[tuple[str, Attribute, int]] = set()

    def add(delta: Delta) -> None:
        for period in periods:
            if delta.window.contains(period):
                touched.add((delta.entity, delta.attribute, period))

    for delta in deltas:
        add(delta)
        for old in delta.supersedes:
            entry = state.assertion(old)
            if entry is not None:
                add(entry.delta)
    return touched


def apply_overwrite(
    state: BitemporalState, deltas: Iterable[Delta], recorded_at: int
) -> UpdateResult:
    """Ablation without versioning: each delta replaces every delta in force on its
    (entity, attribute), whatever their windows. There is no duplicate merging, no
    supersession and no conflict detection; the later message wins."""
    batch = sorted(deltas, key=lambda d: (d.message_id, d.id))
    touched = _touched(state, batch)
    for delta in batch:
        for entry in state.active(as_of=recorded_at):
            if (entry.entity, entry.attribute) == (delta.entity, delta.attribute):
                touched |= _touched(state, [entry])
    before = {key: state.value(*key, as_of=recorded_at) for key in touched}

    retracted: list[str] = []
    for delta in batch:
        for old in state.active(as_of=recorded_at):
            if (old.entity, old.attribute) == (delta.entity, delta.attribute):
                state.retract(old.id, by=delta.id, recorded_at=recorded_at)
                retracted.append(old.id)
        state.assert_delta(delta, recorded_at)

    diff = []
    for key in sorted(touched):
        after = state.value(*key, as_of=recorded_at)
        if after != before[key]:
            diff.append(DiffEntry(key[0], key[1], key[2], before[key], after))
    return UpdateResult(
        asserted=tuple(delta.id for delta in batch),
        retracted=tuple(retracted),
        diff=tuple(diff),
    )
