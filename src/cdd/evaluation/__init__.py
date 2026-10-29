"""Evaluation: delta alignment, extraction metrics, regret decomposition and attribution."""

from cdd.evaluation.matching import Alignment, DeltaError, ErrorKind, align
from cdd.evaluation.metrics import (
    PRF,
    coefficient_of_variation,
    error_table,
    expected_calibration_error,
    prf,
    prf_by_type,
)
from cdd.evaluation.regret import (
    CachedCost,
    RegretDecomposition,
    Repair,
    apply_repairs,
    decompose,
    repairs_for,
    shapley_permutation,
    single_repair_effects,
)

__all__ = [
    "PRF",
    "Alignment",
    "CachedCost",
    "DeltaError",
    "ErrorKind",
    "RegretDecomposition",
    "Repair",
    "align",
    "apply_repairs",
    "coefficient_of_variation",
    "decompose",
    "error_table",
    "expected_calibration_error",
    "prf",
    "prf_by_type",
    "repairs_for",
    "shapley_permutation",
    "single_repair_effects",
]
