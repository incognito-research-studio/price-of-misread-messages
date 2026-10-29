"""SOP admissibility (L2) and the simulated approver.

``adm_P : Δ × S → {accept, approve(ℓ), reject}``. Rules run in order: the first rejection
wins; otherwise the first approval requirement wins; otherwise the delta is accepted. Every
verdict carries the id of the rule that produced it, so the audit log can cite it.

Approvers see the delta, the rule id and the interpreted state only. They never see gold
deltas, so approval cannot act as a channel for ground truth.

SOP-005 (sop-v2, supplier status only) admits a delta only if the sentences of its supporting
span name its entity. A reader that takes "QA has suspended Lot 7702" for a status change, and
must then name a supplier, cannot suspend one the message never mentions.

A conflict (two deltas setting the same cell to different effects in one period) is first
resolved by sender authority: when exactly one of the deltas comes from a sender class that
normally issues its type (``type_owners``), that delta stands. Otherwise the approver decides.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict

from cdd.schema import DELTA_SPECS, Delta, DeltaType, HoldStatus, Message, Op, SupplierStatus
from cdd.state import StateView
from cdd.text import sentence_spans
from cdd.update import Conflict


class Verdict(StrEnum):
    ACCEPT = "accept"
    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True)
class Admissibility:
    verdict: Verdict
    rule_id: str | None = None
    reason: str = ""


ACCEPT = Admissibility(Verdict.ACCEPT)


class SOPConfig(BaseModel):
    """Pre-registered SOP thresholds. Keep one file per registered version."""

    model_config = ConfigDict(frozen=True)

    version: str = "sop-v1"
    capacity_approval_threshold: float = 0.25
    authorised_status_senders: tuple[str, ...] = ("procurement", "supplier_quality")
    # Supplier statuses that only authorised senders may set (sop-v1: suspensions only).
    sender_checked_statuses: tuple[SupplierStatus, ...] = (SupplierStatus.SUSPENDED,)
    # Sender classes that normally issue each delta type, for resolving conflicts (sop-v2).
    type_owners: dict[DeltaType, tuple[str, ...]] = {}
    # Delta types whose span must name their entity (SOP-005, sop-v2).
    grounded_types: tuple[DeltaType, ...] = ()

    @classmethod
    def from_yaml(cls, path: str | Path) -> SOPConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})


def relative_magnitude(delta: Delta, view: StateView) -> float | None:
    """Size of a numeric change relative to the value in force at the window start."""
    if isinstance(delta.value, str):
        return None
    if delta.op is Op.SCALE:
        return abs(delta.value - 1.0)
    base = view.number(delta.entity, delta.attribute, delta.window.start)
    change = abs(delta.value) if delta.op is Op.ADD else abs(delta.value - base)
    return math.inf if base == 0 else change / abs(base)


Rule = Callable[[Delta, Message, StateView], Admissibility | None]


class SOPGraph:
    def __init__(self, config: SOPConfig | None = None) -> None:
        self.config = config or SOPConfig()
        self.rules: tuple[tuple[str, Rule], ...] = (
            ("SOP-001", self._entity_kind),
            ("SOP-002", self._capacity_magnitude),
            ("SOP-003", self._release_needs_hold),
            ("SOP-004", self._status_sender),
            ("SOP-005", self._names_entity),
        )

    def check(self, delta: Delta, message: Message, view: StateView) -> Admissibility:
        approval: Admissibility | None = None
        for _, rule in self.rules:
            verdict = rule(delta, message, view)
            if verdict is None:
                continue
            if verdict.verdict is Verdict.REJECT:
                return verdict
            if verdict.verdict is Verdict.APPROVE and approval is None:
                approval = verdict
        return approval or ACCEPT

    def _entity_kind(self, delta: Delta, message: Message, view: StateView) -> Admissibility | None:
        expected = DELTA_SPECS[delta.type].entity_kind
        actual = view.world.kind_of(delta.entity)
        if actual is not expected:
            return Admissibility(
                Verdict.REJECT, "SOP-001", f"{delta.type} needs a {expected}, got {actual}"
            )
        return None

    def _capacity_magnitude(
        self, delta: Delta, message: Message, view: StateView
    ) -> Admissibility | None:
        if delta.type is not DeltaType.CAPACITY_CHANGE:
            return None
        magnitude = relative_magnitude(delta, view)
        threshold = self.config.capacity_approval_threshold
        if magnitude is not None and magnitude > threshold:
            return Admissibility(
                Verdict.APPROVE, "SOP-002", f"capacity change {magnitude:.0%} > {threshold:.0%}"
            )
        return None

    def _release_needs_hold(
        self, delta: Delta, message: Message, view: StateView
    ) -> Admissibility | None:
        if delta.type is not DeltaType.QUALITY_RELEASE:
            return None
        status = view.label(delta.entity, DELTA_SPECS[delta.type].attribute, delta.window.start)
        if status != HoldStatus.HELD:
            return Admissibility(Verdict.REJECT, "SOP-003", "release without a matching hold")
        return None

    def _status_sender(
        self, delta: Delta, message: Message, view: StateView
    ) -> Admissibility | None:
        if delta.type is not DeltaType.SUPPLIER_STATUS:
            return None
        if delta.value not in self.config.sender_checked_statuses:
            return None
        if message.sender_class not in self.config.authorised_status_senders:
            return Admissibility(
                Verdict.REJECT,
                "SOP-004",
                f"sender class {message.sender_class!r} cannot set supplier status {delta.value}",
            )
        return None

    def _names_entity(
        self, delta: Delta, message: Message, view: StateView
    ) -> Admissibility | None:
        if delta.type not in self.config.grounded_types:
            return None
        # The sentences around the span: a reader may quote a clause ("Suspension of
        # procurement is in effect") of a sentence that names the supplier.
        span = ""
        if delta.message_id == message.id:
            covering = [
                (start, end)
                for start, end in sentence_spans(message.text)
                if start < delta.src.end and delta.src.start < end
            ]
            first = min([delta.src.start, *(s for s, _ in covering)])
            last = max([delta.src.end, *(e for _, e in covering)])
            span = message.text[first:last]
        forms = view.world.entity(delta.entity).surface_forms
        if any(re.search(rf"(?<!\w){re.escape(f)}(?!\w)", span, re.IGNORECASE) for f in forms):
            return None
        return Admissibility(Verdict.REJECT, "SOP-005", f"the span does not name {delta.entity}")

    def resolve(self, conflict: Conflict, sender_class: Mapping[str, str]) -> Delta | None:
        """The one conflicting delta whose sender normally issues its type, if exactly one
        is; ``sender_class`` maps delta ids to the sender class of their message."""
        owned = [
            delta
            for delta in conflict.deltas
            if sender_class.get(delta.id) in self.config.type_owners.get(delta.type, ())
        ]
        return owned[0] if len(owned) == 1 else None


class Approver(Protocol):
    @property
    def name(self) -> str: ...

    def approve(self, delta: Delta, rule_id: str, view: StateView) -> bool: ...

    def resolve(self, conflict: Conflict, view: StateView) -> Delta | None: ...


class AlwaysApprove:
    """Approves everything. A conflict resolves to the delta from the latest message."""

    name = "always_approve"

    def approve(self, delta: Delta, rule_id: str, view: StateView) -> bool:
        return True

    def resolve(self, conflict: Conflict, view: StateView) -> Delta | None:
        return max(conflict.deltas, key=lambda delta: (delta.message_id, delta.id))


class AlwaysReject:
    """Rejects every approval request and leaves conflicts unresolved (the prior state stands)."""

    name = "always_reject"

    def approve(self, delta: Delta, rule_id: str, view: StateView) -> bool:
        return False

    def resolve(self, conflict: Conflict, view: StateView) -> Delta | None:
        return None


@dataclass(frozen=True)
class MagnitudeCapApprover:
    """Approves changes up to ``cap`` (relative) and escalates, i.e. rejects, anything larger."""

    cap: float = 0.5
    name: str = "magnitude_cap"

    def approve(self, delta: Delta, rule_id: str, view: StateView) -> bool:
        magnitude = relative_magnitude(delta, view)
        return magnitude is None or magnitude <= self.cap

    def resolve(self, conflict: Conflict, view: StateView) -> Delta | None:
        return None


def make_approver(name: str, cap: float = 0.5) -> Approver:
    if name == "always_approve":
        return AlwaysApprove()
    if name == "always_reject":
        return AlwaysReject()
    if name == "magnitude_cap":
        return MagnitudeCapApprover(cap=cap)
    raise ValueError(f"unknown approver {name!r}")
