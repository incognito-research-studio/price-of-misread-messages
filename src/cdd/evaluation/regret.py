"""Regret decomposition (Section 3.7) and attribution of regret to individual delta errors.

``R_total = C(x^π(ŝ)) − C(x*)`` splits by telescoping into interpretation regret
``R_interp = C(x^π(ŝ)) − C(x^π(s*))`` and decision regret ``R_dec = C(x^π(s*)) − C(x*)``.
The split is an accounting identity; the empirical content is in how large each part is and
which delta errors produce ``R_interp``. ``R_interp`` can be negative.

Attribution repairs errors and re-runs the stream. A *repair* removes the predicted delta and
inserts the gold one. A gold delta's ``supersedes`` references name gold ids, which the
interpreted state does not hold, so the repair also names the predicted deltas aligned with
those gold deltas; otherwise repairing a revision could never retract anything. ``ΔC_j`` is the
cost reduction from repairing error ``j`` alone; Shapley values average its marginal reduction
over random repair orders (Castro et al., 2009).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from cdd.evaluation.matching import Alignment
from cdd.schema import Delta

CostFn = Callable[[frozenset[str]], float]


@dataclass(frozen=True)
class RegretDecomposition:
    total: float
    interpretation: float
    decision: float

    @property
    def interpretation_share(self) -> float | None:
        """``R_interp / R_total``; undefined (None) when total regret is zero."""
        return None if self.total == 0 else self.interpretation / self.total


def decompose(
    cost_interpreted: float, cost_oracle: float, cost_benchmark: float
) -> RegretDecomposition:
    return RegretDecomposition(
        total=cost_interpreted - cost_benchmark,
        interpretation=cost_interpreted - cost_oracle,
        decision=cost_oracle - cost_benchmark,
    )


@dataclass(frozen=True)
class Repair:
    error_id: str
    remove: frozenset[str]
    add: tuple[Delta, ...]


def repairs_for(alignment: Alignment) -> dict[str, Repair]:
    aligned = {gold.id: pred.id for pred, gold in alignment.exact}
    aligned |= {
        error.gold.id: error.predicted.id
        for error in alignment.errors
        if error.gold is not None and error.predicted is not None
    }
    repairs = {}
    for index, error in enumerate(alignment.errors):
        error_id = f"e{index:04d}"
        remove = frozenset() if error.predicted is None else frozenset({error.predicted.id})
        add: tuple[Delta, ...] = ()
        if error.gold is not None:
            refs = [aligned[i] for i in error.gold.supersedes if i in aligned]
            supersedes = tuple(dict.fromkeys([*refs, *error.gold.supersedes]))
            add = (error.gold.model_copy(update={"supersedes": supersedes}),)
        repairs[error_id] = Repair(error_id, remove, add)
    return repairs


def apply_repairs(predicted: Sequence[Delta], repairs: Iterable[Repair]) -> list[Delta]:
    """The predicted deltas with the chosen repairs applied.

    A reader that keeps gold ids (the gold-span reader) can hold its own copy of a gold delta
    while a repair of another error inserts that gold delta; the inserted copy then gets a fresh
    id, so the state never records one id twice. Readers that mint their own ids never collide.
    """
    chosen = list(repairs)
    removed = set().union(*(repair.remove for repair in chosen)) if chosen else set()
    kept = [delta for delta in predicted if delta.id not in removed]
    taken = {delta.id for delta in kept}
    added = []
    for delta in (delta for repair in chosen for delta in repair.add):
        if delta.id in taken:
            delta = delta.model_copy(update={"id": f"{delta.id}~repaired"})
        taken.add(delta.id)
        added.append(delta)
    return kept + added


class CachedCost:
    """Memoises a cost function over sets of repaired error ids."""

    def __init__(self, cost: CostFn) -> None:
        self._cost = cost
        self._cache: dict[frozenset[str], float] = {}

    def __call__(self, repaired: frozenset[str]) -> float:
        if repaired not in self._cache:
            self._cache[repaired] = self._cost(repaired)
        return self._cache[repaired]

    @property
    def evaluations(self) -> int:
        return len(self._cache)


def single_repair_effects(players: Sequence[str], cost: CostFn) -> dict[str, float]:
    """``ΔC_j = C(no repairs) − C(repair j only)``. Positive values mean the error was costly."""
    base = cost(frozenset())
    return {player: base - cost(frozenset({player})) for player in players}


def shapley_permutation(
    players: Sequence[str], cost: CostFn, n_permutations: int, rng: np.random.Generator
) -> dict[str, float]:
    """Monte Carlo Shapley values of the cost reduction from repairing each error.

    The values sum to ``C(no repairs) − C(all repaired)`` for every sample (efficiency).
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    totals = dict.fromkeys(players, 0.0)
    for _ in range(n_permutations):
        coalition: frozenset[str] = frozenset()
        previous = cost(coalition)
        for index in rng.permutation(len(players)):
            player = players[int(index)]
            coalition = coalition | {player}
            current = cost(coalition)
            totals[player] += previous - current
            previous = current
    return {player: total / n_permutations for player, total in totals.items()}
