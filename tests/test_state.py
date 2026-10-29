from __future__ import annotations

import pytest

from cdd.schema import Attribute, Op, Window
from cdd.state import BitemporalState
from helpers import make_delta


def test_value_folds_deltas_in_recording_order(state: BitemporalState) -> None:
    state.assert_delta(make_delta(id="a#0", value=0.5), recorded_at=40)
    state.assert_delta(make_delta(id="b#0", op=Op.SET, value=800.0, unit=None), recorded_at=41)
    state.assert_delta(make_delta(id="c#0", value=0.5), recorded_at=41)
    assert state.value("P2", Attribute.CAPACITY, 43) == pytest.approx(400.0)
    assert state.value("P2", Attribute.CAPACITY, 45) == 1000.0  # outside the window


def test_transaction_time_hides_later_records(state: BitemporalState) -> None:
    state.assert_delta(make_delta(), recorded_at=42)
    assert state.value("P2", Attribute.CAPACITY, 43, as_of=41) == 1000.0
    assert state.value("P2", Attribute.CAPACITY, 43, as_of=42) == pytest.approx(850.0)


def test_retraction_is_bitemporal(state: BitemporalState) -> None:
    state.assert_delta(make_delta(), recorded_at=40)
    assert state.retract("m1#0", by="m2#0", recorded_at=42) is not None
    assert state.is_active("m1#0", as_of=41)
    assert not state.is_active("m1#0", as_of=42)
    assert state.retract("m1#0", by="m3#0", recorded_at=43) is None
    assert len(state.history) == 2  # nothing is ever deleted


def test_assertions_are_unique(state: BitemporalState) -> None:
    state.assert_delta(make_delta(), recorded_at=40)
    with pytest.raises(ValueError, match="already recorded"):
        state.assert_delta(make_delta(), recorded_at=41)


def test_view_types(state: BitemporalState) -> None:
    view = state.view(as_of=40)
    assert view.label("LOT-7781", Attribute.HOLD_STATUS, 40) == "released"
    with pytest.raises(TypeError):
        view.number("LOT-7781", Attribute.HOLD_STATUS, 40)
    with pytest.raises(TypeError):
        view.label("P2", Attribute.CAPACITY, 40)
    assert make_delta(window=Window(start=41)).window.end is None
