from __future__ import annotations

import pytest
from pydantic import ValidationError

from cdd.schema import (
    DeltaType,
    Op,
    Proposal,
    Span,
    Window,
    message_of,
    percent_to_scale,
    proposal_json_schema,
)
from helpers import make_delta


def test_valid_delta_round_trips_through_json() -> None:
    delta = make_delta()
    assert type(delta).model_validate_json(delta.model_dump_json()) == delta


@pytest.mark.parametrize(
    "overrides",
    [
        {"op": Op.SCALE, "type": DeltaType.LEAD_TIME_CHANGE, "entity": "LANE-X"},
        {"value": "held"},
        {"value": 0.0},
        {"type": DeltaType.QUALITY_HOLD, "op": Op.SET, "value": "released", "entity": "LOT-7781"},
    ],
)
def test_delta_rejects_values_outside_its_spec(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_delta(**overrides)


def test_window_semantics() -> None:
    closed, open_ended = Window(start=42, end=44), Window(start=44)
    assert closed.contains(42) and closed.contains(44) and not closed.contains(45)
    assert open_ended.contains(100)
    assert closed.overlaps(open_ended)
    assert not Window(start=40, end=41).overlaps(closed)
    with pytest.raises(ValidationError):
        Window(start=44, end=42)


def test_span_overlap_requires_same_message() -> None:
    a = Span(message_id="m1", start=0, end=10)
    assert a.overlaps(Span(message_id="m1", start=9, end=12))
    assert not a.overlaps(Span(message_id="m1", start=10, end=12))
    assert not a.overlaps(Span(message_id="m2", start=0, end=10))


def test_ids_and_helpers() -> None:
    assert message_of("stream-m0003#g0") == "stream-m0003"
    assert percent_to_scale(-15) == pytest.approx(0.85)
    proposal = Proposal.from_delta(make_delta())
    assert proposal.delta_id == "m1#0" and proposal.value == pytest.approx(0.85)


def test_json_schema_names_the_interpreter_slots() -> None:
    schema = proposal_json_schema()
    assert {"type", "entity", "span"} <= set(schema["required"])
    assert "v1" in schema["$comment"]
