from __future__ import annotations

import pytest

from cdd import weeks
from cdd.parser import ParseError, SpanParser
from cdd.schema import Delta, DeltaType, Op, Proposal, Span, Window
from helpers import make_message

PARSER = SpanParser(calendar_year=2026)


def _complete(text: str, kind: DeltaType, entity: str = "P2", **slots: object) -> Delta:
    message = make_message(text, period=38)
    proposal = Proposal(
        type=kind, entity=entity, span=Span(message_id="m1", start=0, end=len(text)), **slots
    )
    return PARSER.complete(proposal, message, "m1#0")


# The example messages of Table 1 in the paper.
@pytest.mark.parametrize(
    ("text", "kind", "op", "value", "window"),
    [
        ("Plant 2 output reduced ~15% wk42–44", DeltaType.CAPACITY_CHANGE, Op.SCALE, 0.85,
         Window(start=42, end=44)),
        ("Transit via Port X +6 days through Nov", DeltaType.LEAD_TIME_CHANGE, Op.ADD, 6.0,
         Window(start=38, end=weeks.last_week_of_month(2026, 11))),
        ("Lot 7781 on hold pending CoA", DeltaType.QUALITY_HOLD, Op.SET, "held", Window(start=38)),
        ("Lot 7781 released", DeltaType.QUALITY_RELEASE, Op.SET, "released", Window(start=38)),
        ("Supplier C suspended pending audit", DeltaType.SUPPLIER_STATUS, Op.SET, "suspended",
         Window(start=38)),
        ("Key account pulls 20% forward into wk40", DeltaType.DEMAND_SIGNAL, Op.SCALE, 1.2,
         Window(start=40, end=40)),
        ("Surcharge 4% effective Oct 1", DeltaType.PRICE_CHANGE, Op.SCALE, 1.04,
         Window(start=40)),
    ],
)  # fmt: skip
def test_table_one_examples(
    text: str, kind: DeltaType, op: Op, value: object, window: Window
) -> None:
    delta = _complete(text, kind)
    assert delta.op is op
    assert delta.value == (pytest.approx(value) if isinstance(value, float) else value)
    assert delta.window == window


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Transit via Port Y improved by 3 days from wk44", -3.0),
        ("Shipments on Port X delayed by 10 days wk41 to wk44", 10.0),
    ],
)
def test_lead_time_direction(text: str, value: float) -> None:
    assert _complete(text, DeltaType.LEAD_TIME_CHANGE).value == value


@pytest.mark.parametrize(
    ("text", "window"),
    [
        ("Starting in week 38, transit via Port W is 5 days shorter", Window(start=38)),
        ("Transit via Port W 5 days shorter from week 41 onwards", Window(start=41)),
        ("Transit via Port W 5 days shorter, week 41 onward", Window(start=41)),
        ("Transit via Port W 5 days shorter during weeks 39 through 41", Window(start=39, end=41)),
        ("Transit via Port W 5 days shorter in week 39", Window(start=39, end=39)),
    ],
)
def test_window_wordings(text: str, window: Window) -> None:
    assert _complete(text, DeltaType.LEAD_TIME_CHANGE).window == window


def test_dates_with_ordinals() -> None:
    plain = _complete("Surcharge 4% effective Nov 5", DeltaType.PRICE_CHANGE)
    ordinal = _complete("Starting from November 5th, a 4% surcharge", DeltaType.PRICE_CHANGE)
    assert ordinal.window == plain.window == Window(start=weeks.week_of(2026, 11, 5))


def test_slots_from_the_interpreter_are_kept() -> None:
    delta = _complete(
        "Plant 2 output reduced ~15% wk42-44", DeltaType.CAPACITY_CHANGE,
        op=Op.SCALE, value=0.5, window=Window(start=41, end=41),
    )  # fmt: skip
    assert (delta.value, delta.window) == (0.5, Window(start=41, end=41))


def test_missing_direction_is_a_parse_failure() -> None:
    with pytest.raises(ParseError) as info:
        _complete("Plant 2 output 15% wk42-44", DeltaType.CAPACITY_CHANGE)
    assert info.value.slot == "value"


def test_magnitude_hint_fills_a_span_without_numbers() -> None:
    delta = _complete(
        "Plant 2 output reduced wk42-44", DeltaType.CAPACITY_CHANGE, hints={"magnitude": "-20%"}
    )
    assert delta.value == pytest.approx(0.8)


def test_span_must_point_into_the_message() -> None:
    message = make_message("short")
    proposal = Proposal(
        type=DeltaType.QUALITY_HOLD, entity="LOT-7781", span=Span(message_id="m1", start=0, end=99)
    )
    with pytest.raises(ParseError) as info:
        PARSER.complete(proposal, message, "m1#0")
    assert info.value.slot == "span"


def test_backwards_window_is_a_parse_failure() -> None:
    with pytest.raises(ParseError) as info:
        _complete("Plant 2 output reduced 15% wk44-42", DeltaType.CAPACITY_CHANGE)
    assert info.value.slot == "window"


def test_unknown_entity_is_a_schema_failure() -> None:
    parser = SpanParser(calendar_year=2026, known_entities={"P2"})
    message = make_message("Plant 9 output reduced ~15% wk42-44")
    proposal = Proposal(
        type=DeltaType.CAPACITY_CHANGE,
        entity="Plant 9",
        span=Span(message_id="m1", start=0, end=10),
    )
    with pytest.raises(ParseError) as info:
        parser.complete(proposal, message, "m1#0")
    assert info.value.slot == "entity"


def test_window_hints_apply_only_when_the_span_states_no_window() -> None:
    hints = {"window_start": "wk45", "window_end": "wk46"}
    stated = _complete("Plant 2 output reduced 15% wk42-44", DeltaType.CAPACITY_CHANGE, hints=hints)
    assert stated.window == Window(start=42, end=44)
    unstated = _complete("Plant 2 output reduced 15%", DeltaType.CAPACITY_CHANGE, hints=hints)
    assert unstated.window == Window(start=45, end=46)
    open_ended = _complete(
        "Plant 2 output reduced 15%",
        DeltaType.CAPACITY_CHANGE,
        hints={"window_start": "wk45", "window_end": "open-ended"},
    )
    assert open_ended.window == Window(start=45)


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("Supplier B suspended pending audit.", "suspended"),
        ("Supplier B suspended until the audit is cleared.", "suspended"),
        ("Orders with Supplier B are paused pending the audit.", "suspended"),
        ("Supplier B limited to priority orders from wk40.", "restricted"),
        ("Supplier B may ship priority orders only from wk40.", "restricted"),
        ("Supplier B reinstated after a successful audit.", "active"),
        ("The suspension of Supplier B has been lifted.", "active"),
        ("Supplier B is no longer suspended.", "active"),
    ],
)
def test_supplier_status_words(text: str, status: str) -> None:
    assert _complete(text, DeltaType.SUPPLIER_STATUS).value == status


def test_on_hold_is_not_a_supplier_status() -> None:
    with pytest.raises(ParseError):
        _complete("Lot 7702 is on hold pending CoA.", DeltaType.SUPPLIER_STATUS)
