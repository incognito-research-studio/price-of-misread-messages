from __future__ import annotations

import pytest

from cdd.schema import Delta, DeltaType, Op, Span, Window
from cdd.sop import (
    AlwaysApprove,
    MagnitudeCapApprover,
    SOPConfig,
    SOPGraph,
    Verdict,
    relative_magnitude,
)
from cdd.state import BitemporalState
from cdd.update import Conflict, apply_update
from helpers import make_delta, make_message

SOP = SOPGraph(SOPConfig())


def _status(value: str) -> Delta:
    return make_delta(
        type=DeltaType.SUPPLIER_STATUS, entity="SUP-C", op=Op.SET, value=value, unit=None,
        window=Window(start=40),
    )  # fmt: skip


def _lot(kind: DeltaType, value: str, id: str = "m1#0") -> Delta:
    return make_delta(
        id=id, type=kind, entity="LOT-7781", op=Op.SET, value=value, unit=None,
        window=Window(start=40), src=Span(message_id=id.split("#")[0], start=0, end=5),
    )  # fmt: skip


def test_large_capacity_change_needs_approval(state: BitemporalState) -> None:
    view = state.view(40)
    big = SOP.check(make_delta(value=0.7), make_message("x"), view)
    small = SOP.check(make_delta(value=0.9), make_message("x"), view)
    assert (big.verdict, big.rule_id) == (Verdict.APPROVE, "SOP-002")
    assert small.verdict is Verdict.ACCEPT


def test_release_needs_a_matching_hold(state: BitemporalState) -> None:
    release = _lot(DeltaType.QUALITY_RELEASE, "released", id="m2#0")
    rejected = SOP.check(release, make_message("x"), state.view(40))
    assert (rejected.verdict, rejected.rule_id) == (Verdict.REJECT, "SOP-003")
    apply_update(state, [_lot(DeltaType.QUALITY_HOLD, "held")], recorded_at=40)
    accepted = SOP.check(release, make_message("x"), state.view(40))
    assert accepted.verdict is Verdict.ACCEPT


def test_only_authorised_senders_can_suspend(state: BitemporalState) -> None:
    view = state.view(40)
    suspension = _status("suspended")
    carrier = SOP.check(suspension, make_message("x", sender_class="carrier"), view)
    procurement = SOP.check(suspension, make_message("x"), view)
    assert (carrier.verdict, carrier.rule_id) == (Verdict.REJECT, "SOP-004")
    assert procurement.verdict is Verdict.ACCEPT


def test_entity_must_have_the_right_kind(state: BitemporalState) -> None:
    verdict = SOP.check(make_delta(entity="LANE-X"), make_message("x"), state.view(40))
    assert (verdict.verdict, verdict.rule_id) == (Verdict.REJECT, "SOP-001")


def test_approvers(state: BitemporalState) -> None:
    view = state.view(40)
    assert relative_magnitude(make_delta(value=0.7), view) == pytest.approx(0.3)
    cap = MagnitudeCapApprover(cap=0.5)
    assert cap.approve(make_delta(value=0.7), "SOP-002", view)
    assert not cap.approve(make_delta(value=0.4), "SOP-002", view)
    early = make_delta(id="m1#0", value=0.8)
    late = make_delta(id="m2#0", value=0.7, src=Span(message_id="m2", start=0, end=5))
    conflict = Conflict("P2", early.attribute, early.window, (early, late))
    assert AlwaysApprove().resolve(conflict, view) == late
    assert cap.resolve(conflict, view) is None


SOP_V2 = SOPGraph(SOPConfig.from_yaml("configs/sop/sop-v2.yaml"))


def test_sop_v2_checks_the_sender_of_every_status_change(state: BitemporalState) -> None:
    view = state.view(40)
    internal = make_message("x", sender_class="internal")
    for value in ("active", "restricted", "suspended"):
        verdict = SOP_V2.check(_status(value), internal, view)
        assert (verdict.verdict, verdict.rule_id) == (Verdict.REJECT, "SOP-004")
        named = make_message("Supplier C status update from procurement.")
        assert SOP_V2.check(_status(value), named, view).verdict is Verdict.ACCEPT
    # sop-v1 checks suspensions only.
    assert SOP.check(_status("active"), internal, view).verdict is Verdict.ACCEPT


def test_sop_v2_resolves_a_conflict_by_sender_authority() -> None:
    true = make_delta(id="m1#0", value=0.85)
    misread = make_delta(id="m2#0", value=0.7, src=Span(message_id="m2", start=0, end=5))
    conflict = Conflict("P2", true.attribute, true.window, (true, misread))
    senders = {"m1#0": "supplier_planning", "m2#0": "sales"}
    assert SOP_V2.resolve(conflict, senders) == true
    assert SOP_V2.resolve(conflict, dict.fromkeys(senders, "supplier_planning")) is None
    assert SOP.resolve(conflict, senders) is None


def test_sop_v2_status_change_must_name_its_supplier(state: BitemporalState) -> None:
    view = state.view(40)
    named = make_message("Supplier C suspended pending audit.")
    unnamed = make_message("QA has suspended Lot 7781 pending tests.")
    delta = _status("suspended").model_copy(update={"src": Span(message_id="m1", start=0, end=30)})
    assert SOP_V2.check(delta, named, view).verdict is Verdict.ACCEPT
    verdict = SOP_V2.check(delta, unnamed, view)
    assert (verdict.verdict, verdict.rule_id) == (Verdict.REJECT, "SOP-005")
    assert SOP.check(delta, unnamed, view).verdict is Verdict.ACCEPT


def test_sop_v2_reads_the_sentence_around_a_quoted_clause(state: BitemporalState) -> None:
    text = "Supplier C: Suspension of procurement is in effect until audit completion."
    clause = text.index("Suspension")
    delta = _status("suspended").model_copy(
        update={"src": Span(message_id="m1", start=clause, end=len(text) - 1)}
    )
    assert SOP_V2.check(delta, make_message(text), state.view(40)).verdict is Verdict.ACCEPT
