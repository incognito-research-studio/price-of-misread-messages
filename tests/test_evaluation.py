from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from cdd.evaluation import (
    CachedCost,
    ErrorKind,
    Repair,
    align,
    apply_repairs,
    coefficient_of_variation,
    decompose,
    error_table,
    prf,
    prf_by_type,
    repairs_for,
    shapley_permutation,
    single_repair_effects,
)
from cdd.schema import Delta, DeltaType, Op, Span, Window
from helpers import make_delta

GOLD = make_delta(id="m1#g0")


def _kind(predicted: list[Delta], gold: list[Delta]) -> list[ErrorKind]:
    alignment = align(predicted, gold)
    return [error.kind for error in alignment.errors]


def test_exact_match() -> None:
    alignment = align([make_delta()], [GOLD])
    assert len(alignment.exact) == 1 and alignment.errors == ()


@pytest.mark.parametrize(
    ("predicted", "kind"),
    [
        (make_delta(value=1.15), ErrorKind.WRONG_VALUE),
        (make_delta(window=Window(start=42, end=43)), ErrorKind.MIS_TIMED),
        (make_delta(window=Window(start=46, end=47)), ErrorKind.MIS_TIMED),
        (make_delta(entity="P3"), ErrorKind.MIS_SCOPED),
        (
            make_delta(
                type=DeltaType.LEAD_TIME_CHANGE, entity="LANE-X", op=Op.ADD, value=6.0, unit="days"
            ),
            ErrorKind.WRONG_TYPE,
        ),
        (make_delta(src=Span(message_id="m1", start=30, end=40)), ErrorKind.PROVENANCE),
    ],
)
def test_each_error_kind(predicted: Delta, kind: ErrorKind) -> None:
    assert _kind([predicted], [GOLD]) == [kind]


def test_a_supersession_counts_only_when_it_changes_state() -> None:
    earlier = make_delta(id="m0#0", value=0.9, src=Span(message_id="m0", start=0, end=5))
    gold_earlier = earlier.model_copy(update={"id": "m0#g0"})
    linked = make_delta(supersedes=("m0#0",))
    errors = align([earlier, linked], [gold_earlier, GOLD]).errors
    assert [e.kind for e in errors] == [ErrorKind.WRONG_VALUE]
    # A link to a delta the reader never wrote retracts nothing.
    assert _kind([make_delta(supersedes=("m0#0",))], [GOLD]) == []


def test_right_content_cited_to_the_wrong_message_is_provenance() -> None:
    cited_elsewhere = make_delta(id="m9#0", src=Span(message_id="m9", start=0, end=5))
    assert _kind([cited_elsewhere], [GOLD]) == [ErrorKind.PROVENANCE]


def test_missed_and_spurious() -> None:
    unrelated = make_delta(id="m9#0", entity="P3", src=Span(message_id="m9", start=0, end=5))
    assert sorted(_kind([unrelated], [GOLD])) == [ErrorKind.MISSED, ErrorKind.SPURIOUS]


def test_scores_and_tables() -> None:
    alignment = align([make_delta(), make_delta(id="m1#1", entity="P3")], [GOLD])
    scores = prf(alignment)
    assert (scores.tp, scores.n_predicted, scores.n_gold) == (1, 2, 1)
    assert scores.precision == 0.5 and scores.recall == 1.0
    assert prf_by_type(alignment)[DeltaType.CAPACITY_CHANGE].precision == 0.5
    # The spurious P3 delta is a mis-scope candidate only if no exact match took the gold.
    assert error_table(alignment) == {(DeltaType.CAPACITY_CHANGE, ErrorKind.SPURIOUS): 1}


def test_coefficient_of_variation() -> None:
    assert coefficient_of_variation([5.0, 5.0, 5.0]) == 0.0
    assert coefficient_of_variation([1.0, 3.0]) == pytest.approx(math.sqrt(2) / 2)
    assert coefficient_of_variation([1.0]) == 0.0


def test_regret_decomposition_telescopes() -> None:
    regret = decompose(cost_interpreted=130.0, cost_oracle=110.0, cost_benchmark=100.0)
    assert (regret.total, regret.interpretation, regret.decision) == (30.0, 20.0, 10.0)
    assert regret.interpretation_share == pytest.approx(2 / 3)
    assert decompose(100.0, 100.0, 100.0).interpretation_share is None


def test_repairs_swap_predicted_for_gold() -> None:
    wrong = make_delta(value=1.15)
    alignment = align([wrong], [GOLD])
    repairs = repairs_for(alignment)
    assert apply_repairs([wrong], repairs.values()) == [GOLD]
    assert apply_repairs([wrong], []) == [wrong]


def test_repair_that_reinserts_a_kept_id_gets_a_fresh_id() -> None:
    # The gold-span reader keeps gold ids: its misparsed copy of m2's delta can be aligned to
    # m1's gold delta while it also holds its own copy of m1's delta under the gold id.
    other = make_delta(id="m2#g0", value=0.9)
    own_copy = make_delta(id="m1#g0", value=1.15)
    repair = Repair("e0000", frozenset({"m2#g0"}), (GOLD,))
    repaired = apply_repairs([other, own_copy], [repair])
    assert [d.id for d in repaired] == ["m1#g0", "m1#g0~repaired"]
    assert repaired[0] == own_copy


def test_shapley_matches_single_repairs_for_additive_costs() -> None:
    weights = {"a": 3.0, "b": 5.0, "c": 0.0}
    cost = CachedCost(lambda repaired: 100.0 - sum(weights[p] for p in repaired))
    players = list(weights)
    single = single_repair_effects(players, cost)
    shapley = shapley_permutation(players, cost, 20, np.random.default_rng(0))
    assert single == pytest.approx(weights)
    assert shapley == pytest.approx(weights)
    assert cost.evaluations == 2 ** len(players)


def test_shapley_is_efficient_with_interactions() -> None:
    def cost(repaired: frozenset[str]) -> float:
        return 0.0 if {"a", "b"} <= repaired else 10.0  # only fixing both helps

    shapley = shapley_permutation(["a", "b"], cost, 50, np.random.default_rng(1))
    assert sum(shapley.values()) == pytest.approx(10.0)
    assert shapley["a"] == pytest.approx(5.0, abs=2.5)
    assert single_repair_effects(["a", "b"], cost) == {"a": 0.0, "b": 0.0}


def test_windows_are_compared_inside_the_horizon() -> None:
    open_ended = make_delta(id="m1#g0", window=Window(start=50))
    to_end = make_delta(window=Window(start=50, end=52))
    assert _kind([to_end], [open_ended]) == [ErrorKind.MIS_TIMED]
    assert align([to_end], [open_ended], horizon=(36, 52)).errors == ()
    assert len(align([to_end], [open_ended], horizon=(36, 60)).errors) == 1


def _lot_delta(id: str, kind: DeltaType, value: str, start: int, **extra: Any) -> Delta:
    return make_delta(
        id=id, type=kind, entity="LOT-7781", op=Op.SET, value=value, unit=None,
        window=Window(start=start), src=Span(message_id=id.split("#")[0], start=0, end=5),
        **extra,
    )  # fmt: skip


def test_a_release_that_names_its_hold_is_not_an_error() -> None:
    hold = _lot_delta("h#g0", DeltaType.QUALITY_HOLD, "held", 41)
    release = _lot_delta("r#g0", DeltaType.QUALITY_RELEASE, "released", 43)
    read_hold = _lot_delta("h#0", DeltaType.QUALITY_HOLD, "held", 41)
    read_release = _lot_delta("r#0", DeltaType.QUALITY_RELEASE, "released", 43, supersedes=("h#0",))
    periods = {"h": 41, "r": 43}
    alignment = align([read_hold, read_release], [hold, release], (40, 45), periods)
    assert alignment.errors == ()


def test_a_missing_or_unlinked_revision() -> None:
    original = make_delta(id="a#g0", value=0.8, src=Span(message_id="a", start=0, end=5))
    revision = make_delta(
        id="b#g0", value=0.6, supersedes=("a#g0",), src=Span(message_id="b", start=0, end=5)
    )
    read_original = original.model_copy(update={"id": "a#0"})
    unlinked = revision.model_copy(update={"id": "b#0", "supersedes": ()})
    periods = {"a": 40, "b": 41}
    # The update treats a later delta on the same cell and window as a revision.
    assert align([read_original, unlinked], [original, revision], (40, 45), periods).errors == ()
    # A revision with another window does not restate the original, so its link is missing.
    shifted = unlinked.model_copy(update={"window": Window(start=42, end=45)})
    errors = align([read_original, shifted], [original, revision], (40, 45), periods).errors
    assert [e.kinds for e in errors] == [(ErrorKind.MIS_TIMED, ErrorKind.WRONG_VALUE)]


def test_repairing_a_revision_names_the_predicted_original() -> None:
    original = make_delta(id="a#g0", value=0.8, src=Span(message_id="a", start=0, end=5))
    revision = make_delta(
        id="b#g0", value=0.6, supersedes=("a#g0",), src=Span(message_id="b", start=0, end=5)
    )
    read_original = original.model_copy(update={"id": "a#0"})
    wrong = revision.model_copy(update={"id": "b#0", "value": 0.7, "supersedes": ()})
    alignment = align([read_original, wrong], [original, revision], (40, 45), {"a": 40, "b": 41})
    (repair,) = repairs_for(alignment).values()
    assert repair.add[0].supersedes == ("a#0", "a#g0")
