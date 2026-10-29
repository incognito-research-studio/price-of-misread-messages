"""Extraction metrics (precision, recall, F1), error tables and decision consistency."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from cdd.evaluation.matching import Alignment, ErrorKind
from cdd.schema import DeltaType


@dataclass(frozen=True)
class PRF:
    """Strict scores: a predicted delta counts as correct only if it matches gold exactly."""

    tp: int
    n_predicted: int
    n_gold: int

    @property
    def precision(self) -> float:
        if self.n_predicted == 0:
            return 1.0 if self.n_gold == 0 else 0.0
        return self.tp / self.n_predicted

    @property
    def recall(self) -> float:
        if self.n_gold == 0:
            return 1.0 if self.n_predicted == 0 else 0.0
        return self.tp / self.n_gold

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def prf(alignment: Alignment) -> PRF:
    tp = len(alignment.exact)
    n_predicted = tp + sum(e.predicted is not None for e in alignment.errors)
    n_gold = tp + sum(e.gold is not None for e in alignment.errors)
    return PRF(tp, n_predicted, n_gold)


def prf_by_type(alignment: Alignment) -> dict[DeltaType, PRF]:
    tp: Counter[DeltaType] = Counter(gold.type for _, gold in alignment.exact)
    predicted: Counter[DeltaType] = Counter(pred.type for pred, _ in alignment.exact)
    gold: Counter[DeltaType] = Counter(tp)
    for error in alignment.errors:
        if error.predicted is not None:
            predicted[error.predicted.type] += 1
        if error.gold is not None:
            gold[error.gold.type] += 1
    return {kind: PRF(tp[kind], predicted[kind], gold[kind]) for kind in DeltaType}


def error_table(alignment: Alignment) -> Counter[tuple[DeltaType, ErrorKind]]:
    """Counts of errors by (gold delta type, error kind): the cells of S(τ, k)."""
    return Counter((error.delta_type, error.kind) for error in alignment.errors)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Sample standard deviation over the mean; 0 for constant input, inf for a zero mean."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if variance == 0:
        return 0.0
    return math.inf if mean == 0 else math.sqrt(variance) / abs(mean)


def expected_calibration_error(alignment: Alignment, bins: int = 10) -> float | None:
    """ECE of the predicted deltas' confidence κ against exact correctness.

    Predictions without a confidence are skipped; None when no prediction has one. Each bin
    contributes |accuracy − mean confidence| weighted by its share of predictions.
    """
    scored = [(pred.confidence, True) for pred, _ in alignment.exact if pred.confidence is not None]
    scored += [
        (error.predicted.confidence, False)
        for error in alignment.errors
        if error.predicted is not None and error.predicted.confidence is not None
    ]
    if not scored:
        return None
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (conf, ok)
            for conf, ok in scored
            if low <= conf < high or (b == bins - 1 and conf == 1.0)
        ]
        if members:
            accuracy = sum(ok for _, ok in members) / len(members)
            confidence = sum(conf for conf, _ in members) / len(members)
            total += len(members) / len(scored) * abs(accuracy - confidence)
    return total
