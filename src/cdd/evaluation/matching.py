"""Alignment of predicted and gold deltas, and the seven kinds of delta error (Table 2).

Section 3.8 aligns deltas by bipartite matching on ``(τ, e, a)`` with window overlap. That
alone cannot produce *wrong type* or *mis-scoped* errors, because those pairs differ in ``τ`` or
``e``. The alignment therefore runs in three stages:

1. Match on type, entity and attribute with overlapping windows. A matched pair can be exact,
   mis-timed, wrong in value, or wrong in provenance.
2. Pair the remaining deltas that come from the same source message. These pairs are wrong in
   type, mis-scoped, or (same key, disjoint windows) mis-timed.
3. Whatever is left is missed (gold) or spurious (predicted).

A wrong or missing supersession reference counts as a wrong value only when it changes a state
cell at or after the week of the message: a release that also names the hold it ends, or a
reference to another entity's delta (which the update leaves unresolved), changes nothing a
decision can read and is not an error. An unlinked restatement of the same cell and window,
which the update treats as a revision, counts as a reference. References are compared by
source message, because predicted and gold deltas carry different ids.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from cdd.schema import Delta, DeltaType, Op, Window, message_of

_INFEASIBLE = 1e9


class ErrorKind(StrEnum):
    WRONG_VALUE = "wrong_value"
    MISSED = "missed"
    SPURIOUS = "spurious"
    MIS_SCOPED = "mis_scoped"
    MIS_TIMED = "mis_timed"
    WRONG_TYPE = "wrong_type"
    PROVENANCE = "provenance"


PRIORITY = (
    ErrorKind.WRONG_TYPE,
    ErrorKind.MIS_SCOPED,
    ErrorKind.MIS_TIMED,
    ErrorKind.WRONG_VALUE,
    ErrorKind.PROVENANCE,
)


@dataclass(frozen=True)
class DeltaError:
    """One error. ``kind`` is the primary kind; ``kinds`` lists every discrepancy found."""

    kind: ErrorKind
    kinds: tuple[ErrorKind, ...]
    predicted: Delta | None
    gold: Delta | None

    @property
    def delta_type(self) -> DeltaType:
        """The gold type when there is one, so S(τ, k) is indexed by what should have happened."""
        if self.gold is not None:
            return self.gold.type
        assert self.predicted is not None
        return self.predicted.type


@dataclass(frozen=True)
class Alignment:
    exact: tuple[tuple[Delta, Delta], ...]
    errors: tuple[DeltaError, ...]


def same_value(a: Delta, b: Delta, rel_tol: float = 1e-6) -> bool:
    if a.op != b.op:
        return False
    if isinstance(a.value, str) or isinstance(b.value, str):
        return a.value == b.value
    return math.isclose(a.value, b.value, rel_tol=rel_tol, abs_tol=1e-9)


Horizon = tuple[int, int] | None


def same_supersession(a: Delta, b: Delta) -> bool:
    """Raw comparison of the references by source message (no state effect considered)."""
    return {message_of(i) for i in a.supersedes} == {message_of(i) for i in b.supersedes}


def retraction_matters(old: Delta, new: Delta, period: int, horizon: Horizon = None) -> bool:
    """Whether retracting ``old`` when ``new`` is recorded in ``period`` changes any state cell
    at or after ``period`` (inside ``horizon``). It does not when ``new`` sets every such cell
    of ``old`` itself, since the later delta wins."""
    first = max(old.window.start, period)
    last = old.window.end
    if horizon is not None:
        first = max(first, horizon[0])
        last = horizon[1] if last is None else min(last, horizon[1])
    if last is not None and last < first:
        return False
    if new.op is not Op.SET or new.window.start > first:
        return True
    if new.window.end is None:
        return False
    return last is None or new.window.end < last


@dataclass(frozen=True)
class _Pool:
    """The deltas of one side (predicted or gold), to resolve supersession references."""

    by_id: Mapping[str, Delta]
    periods: Mapping[str, int] | None
    horizon: Horizon

    def period(self, delta: Delta) -> int:
        if self.periods is not None and delta.message_id in self.periods:
            return self.periods[delta.message_id]
        return delta.window.start

    def replaced(self, delta: Delta) -> frozenset[str]:
        """Source messages of the deltas ``delta`` replaces in a way a decision can see."""

        def same_cell(other: Delta) -> bool:
            return (other.entity, other.attribute) == (delta.entity, delta.attribute)

        # As in the update: explicit references on the same cell, else a restatement.
        old = [self.by_id[i] for i in delta.supersedes if i in self.by_id]
        old = [o for o in old if same_cell(o)]
        if not old and self.periods is not None:
            old = [
                o
                for o in self.by_id.values()
                if same_cell(o) and o.window == delta.window and self.period(o) < self.period(delta)
            ]
        period = self.period(delta)
        return frozenset(
            o.message_id for o in old if retraction_matters(o, delta, period, self.horizon)
        )


def same_window(a: Window, b: Window, horizon: Horizon = None) -> bool:
    """Windows are equal if they cover the same periods inside ``horizon`` (all time if None).

    Inside a horizon that ends in week 52, ``wk50-52`` and ``from wk50`` change the same cells.
    """
    if horizon is None:
        return a == b
    first, last = horizon

    def clip(w: Window) -> tuple[int, int]:
        return max(w.start, first), last if w.end is None else min(w.end, last)

    return clip(a) == clip(b)


def _discrepancies(
    pred: Delta,
    gold: Delta,
    horizon: Horizon = None,
    pools: tuple[_Pool, _Pool] | None = None,
) -> list[ErrorKind]:
    kinds = []
    if pred.type != gold.type:
        kinds.append(ErrorKind.WRONG_TYPE)
    if pred.entity != gold.entity:
        kinds.append(ErrorKind.MIS_SCOPED)
    if pred.type == gold.type and pred.entity == gold.entity:
        if not same_window(pred.window, gold.window, horizon):
            kinds.append(ErrorKind.MIS_TIMED)
        if pools is None:
            linked = same_supersession(pred, gold)
        else:
            linked = pools[0].replaced(pred) == pools[1].replaced(gold)
        if not same_value(pred, gold) or not linked:
            kinds.append(ErrorKind.WRONG_VALUE)
    if not pred.src.overlaps(gold.src):
        kinds.append(ErrorKind.PROVENANCE)
    return kinds


def _assign(
    preds: Sequence[Delta], golds: Sequence[Delta], cost: Callable[[Delta, Delta], float]
) -> list[tuple[int, int]]:
    if not preds or not golds:
        return []
    matrix = np.array([[cost(p, g) for g in golds] for p in preds], dtype=float)
    rows, cols = linear_sum_assignment(matrix)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if matrix[i, j] < _INFEASIBLE]


def align(
    predicted: Sequence[Delta],
    gold: Sequence[Delta],
    horizon: Horizon = None,
    periods: Mapping[str, int] | None = None,
) -> Alignment:
    """Align predictions with gold. Pass the world's ``(start_period, end_period)`` as
    ``horizon`` to compare windows only where they can affect a decision, and the period of
    each message (by id) as ``periods`` to judge supersession by its effect from the message's
    week (without it, a delta's window start stands in for its message's week)."""
    preds, golds = list(predicted), list(gold)
    pools = (
        _Pool({d.id: d for d in preds}, periods, horizon),
        _Pool({d.id: d for d in golds}, periods, horizon),
    )
    exact: list[tuple[Delta, Delta]] = []
    errors: list[DeltaError] = []

    def stage_one(pred: Delta, gold: Delta) -> float:
        if (pred.type, pred.entity, pred.attribute) != (gold.type, gold.entity, gold.attribute):
            return _INFEASIBLE
        if not pred.window.overlaps(gold.window):
            return _INFEASIBLE
        return float(len(_discrepancies(pred, gold, horizon, pools)))

    def stage_two(pred: Delta, gold: Delta) -> float:
        if pred.message_id != gold.message_id:
            return _INFEASIBLE
        return float(len(_discrepancies(pred, gold, horizon, pools)))

    for stage_cost in (stage_one, stage_two):
        pairs = _assign(preds, golds, stage_cost)
        for i, j in pairs:
            kinds = _discrepancies(preds[i], golds[j], horizon, pools)
            if not kinds:
                exact.append((preds[i], golds[j]))
            else:
                primary = next(kind for kind in PRIORITY if kind in kinds)
                errors.append(DeltaError(primary, tuple(kinds), preds[i], golds[j]))
        used_p = {i for i, _ in pairs}
        used_g = {j for _, j in pairs}
        preds = [p for i, p in enumerate(preds) if i not in used_p]
        golds = [g for j, g in enumerate(golds) if j not in used_g]

    errors += [DeltaError(ErrorKind.MISSED, (ErrorKind.MISSED,), None, g) for g in golds]
    errors += [DeltaError(ErrorKind.SPURIOUS, (ErrorKind.SPURIOUS,), p, None) for p in preds]
    return Alignment(tuple(exact), tuple(errors))
