from __future__ import annotations

import pytest

from cdd.schema import Attribute, Delta, DeltaType, Op, Span, Window
from cdd.state import BitemporalState
from cdd.update import apply_update, canonical_order
from helpers import make_delta


def _lead(id: str, days: float, start: int = 41) -> Delta:
    return make_delta(
        id=id,
        type=DeltaType.LEAD_TIME_CHANGE,
        entity="LANE-X",
        op=Op.ADD,
        value=days,
        unit="days",
        window=Window(start=start),
        src=Span(message_id=id.split("#")[0], start=0, end=10),
    )


def test_canonical_order_is_independent_of_input_order() -> None:
    a, b = make_delta(id="a#0"), make_delta(id="b#0", window=Window(start=41, end=41))
    assert canonical_order([a, b]) == canonical_order([b, a])


def test_duplicates_merge_and_keep_provenance(state: BitemporalState) -> None:
    result = apply_update(state, [_lead("a#0", 6.0), _lead("b#0", 6.0)], recorded_at=40)
    assert result.asserted == ("a#0",)
    assert result.merged == (("b#0", "a#0"),)
    assert len(state.sources("a#0")) == 2
    assert state.value("LANE-X", Attribute.LEAD_TIME, 42) == 13.0


def test_repeated_message_in_a_later_period_is_not_applied_twice(state: BitemporalState) -> None:
    apply_update(state, [_lead("a#0", 6.0)], recorded_at=40)
    result = apply_update(state, [_lead("b#0", 6.0)], recorded_at=41)
    assert result.asserted == ()
    assert state.value("LANE-X", Attribute.LEAD_TIME, 42) == 13.0


def test_contradictions_are_withheld_as_conflicts(state: BitemporalState) -> None:
    result = apply_update(
        state, [make_delta(id="a#0", value=0.85), make_delta(id="b#0", value=0.7)], recorded_at=40
    )
    assert result.asserted == ()
    assert len(result.conflicts) == 1
    assert state.value("P2", Attribute.CAPACITY, 42) == 1000.0


def test_supersession_retracts_the_old_delta(state: BitemporalState) -> None:
    apply_update(state, [make_delta(id="a#0", value=0.8)], recorded_at=40)
    result = apply_update(
        state, [make_delta(id="b#0", value=0.9, supersedes=("a#0",))], recorded_at=41
    )
    assert result.retracted == ("a#0",)
    assert state.value("P2", Attribute.CAPACITY, 43, as_of=41) == pytest.approx(900.0)
    assert state.value("P2", Attribute.CAPACITY, 43, as_of=40) == pytest.approx(800.0)


def test_unknown_supersession_is_reported(state: BitemporalState) -> None:
    result = apply_update(state, [make_delta(supersedes=("ghost#0",))], recorded_at=40)
    assert result.unresolved_supersedes == ("ghost#0",)


def test_release_closes_hold(state: BitemporalState) -> None:
    hold = make_delta(
        id="h#0", type=DeltaType.QUALITY_HOLD, entity="LOT-7781", op=Op.SET, value="held",
        unit=None, window=Window(start=40),
    )  # fmt: skip
    release = hold.model_copy(
        update={"id": "r#0", "type": DeltaType.QUALITY_RELEASE, "value": "released",
                "window": Window(start=42)}
    )  # fmt: skip
    apply_update(state, [hold], recorded_at=40)
    apply_update(state, [release], recorded_at=42)
    assert state.value("LOT-7781", Attribute.HOLD_STATUS, 41) == "held"
    assert state.value("LOT-7781", Attribute.HOLD_STATUS, 42) == "released"


def test_diff_lists_changed_cells(state: BitemporalState) -> None:
    result = apply_update(state, [make_delta()], recorded_at=40)
    assert [(d.period, d.before, d.after) for d in result.diff] == [
        (42, 1000.0, pytest.approx(850.0)),
        (43, 1000.0, pytest.approx(850.0)),
        (44, 1000.0, pytest.approx(850.0)),
    ]


def test_supersedes_link_to_another_entity_is_unresolved(state: BitemporalState) -> None:
    apply_update(state, [_lead("a#0", 14.0)], recorded_at=40)
    result = apply_update(
        state, [make_delta(id="b#0", value=0.9, supersedes=("a#0",))], recorded_at=41
    )
    assert result.retracted == ()
    assert result.unresolved_supersedes == ("a#0",)
    assert state.value("LANE-X", Attribute.LEAD_TIME, 42, as_of=41) == 21.0


def test_unlinked_revision_of_the_same_window_replaces_the_original(
    state: BitemporalState,
) -> None:
    apply_update(state, [make_delta(id="a#0", value=0.8)], recorded_at=40)
    result = apply_update(state, [make_delta(id="b#0", value=0.6)], recorded_at=41)
    assert result.restated == (("a#0", "b#0"),)
    assert state.value("P2", Attribute.CAPACITY, 43, as_of=41) == pytest.approx(600.0)


def test_a_change_with_another_window_still_compounds(state: BitemporalState) -> None:
    apply_update(state, [make_delta(id="a#0", value=0.8)], recorded_at=40)
    later = make_delta(id="b#0", value=0.5, window=Window(start=43, end=45))
    result = apply_update(state, [later], recorded_at=41)
    assert result.restated == ()
    assert state.value("P2", Attribute.CAPACITY, 43, as_of=41) == pytest.approx(400.0)


def test_a_revision_whose_only_link_names_another_entity_still_restates(
    state: BitemporalState,
) -> None:
    apply_update(state, [_lead("x#0", 14.0)], recorded_at=40)
    apply_update(state, [make_delta(id="a#0", value=0.8)], recorded_at=40)
    revision = make_delta(id="b#0", value=0.6, supersedes=("x#0",))
    result = apply_update(state, [revision], recorded_at=41)
    assert result.unresolved_supersedes == ("x#0",) and result.restated == (("a#0", "b#0"),)
    assert state.value("P2", Attribute.CAPACITY, 43, as_of=41) == pytest.approx(600.0)
