"""Deterministic span parser (L4): ``(span, slot) → (v, u, w)`` or failure.

Completes an interpreter :class:`~cdd.schema.Proposal` into a :class:`~cdd.schema.Delta` by
reading the operation, value and validity window from the supporting span. Slots the
interpreter already filled are kept. When the span states no magnitude or no window, the parser
falls back to the ``magnitude``, ``window_start`` and ``window_end`` hints, which typed models
supply by answering atomic questions (the parser-first rule of Section 3.3).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from pydantic import ValidationError

from cdd import weeks
from cdd.schema import (
    DELTA_SPECS,
    Delta,
    DeltaSpec,
    DeltaType,
    HoldStatus,
    Message,
    Op,
    Proposal,
    SupplierStatus,
    Value,
    Window,
    percent_to_scale,
)


class ParseError(ValueError):
    """The span does not support a complete delta. ``slot`` names what could not be filled."""

    def __init__(self, slot: str, reason: str) -> None:
        super().__init__(f"{slot}: {reason}")
        self.slot = slot


_MONTH = (
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
)
_WEEK = r"(?:wk|weeks?)\s*"
_TO = r"(?:-|–|—|to|through|thru)"
_FROM = r"(?:from|effective|starting|as of|beginning)(?:\s+(?:in|on|with|from))?"
_UNTIL = r"(?:through|thru|until|till)"

_RANGE = re.compile(
    rf"\b{_WEEK}(?P<a>\d{{1,2}})\s*{_TO}\s*(?:{_WEEK})?(?P<b>\d{{1,2}})\b", re.IGNORECASE
)
_FROM_WEEK = re.compile(rf"\b{_FROM}\s+{_WEEK}(?P<a>\d{{1,2}})\b", re.IGNORECASE)
_UNTIL_WEEK = re.compile(rf"\b{_UNTIL}\s+{_WEEK}(?P<b>\d{{1,2}})\b", re.IGNORECASE)
_UNTIL_MONTH = re.compile(rf"\b{_UNTIL}\s+(?:end of\s+)?{_MONTH}", re.IGNORECASE)
_FROM_DATE = re.compile(
    rf"\b{_FROM}\s+{_MONTH}\s*(?P<day>\d{{1,2}})?(?:st|nd|rd|th)?\b", re.IGNORECASE
)
_ONWARD = re.compile(rf"\b{_WEEK}(?P<a>\d{{1,2}})\s*(?:onwards?|and (?:after|beyond))\b", re.I)
_ONE_WEEK = re.compile(rf"\b{_WEEK}(?P<a>\d{{1,2}})\b", re.IGNORECASE)

_SIGN = r"(?P<sign>[+\-−])?"
_PERCENT = re.compile(
    rf"{_SIGN}\s*~?\s*(?P<num>\d+(?:\.\d+)?)\s*(?:%|percent\b|pct\b)", re.IGNORECASE
)
_DAYS = re.compile(rf"{_SIGN}\s*(?P<num>\d+(?:\.\d+)?)\s*(?:days?|d)\b", re.IGNORECASE)
_ABSOLUTE = re.compile(r"\bto\s+(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?:units?|u)\b", re.IGNORECASE)

_DOWN = re.compile(
    r"\b(?:reduc\w*|cut\w*|down|drops?|dropped|lower\w*|decreas\w*|fall\w*|discount\w*"
    r"|shorter|earlier|faster|improv\w*|less)\b",
    re.IGNORECASE,
)
_UP = re.compile(
    r"\b(?:increas\w*|up|rais\w*|boost\w*|higher|surcharge\w*|delay\w*|longer|later|slower"
    r"|forward|extra|added)\b",
    re.IGNORECASE,
)

# Supplier status words, read in this order: an explicit end of a suspension or restriction,
# then the specific statuses, then the weak words "lifted", "cleared" and "active". So
# "suspended until the audit is cleared" reads as suspended, and "the suspension has been
# lifted" as active.
_ACTIVE_STRONG = re.compile(
    r"\b(?:reinstat\w*|resum\w*|reactivat\w*|back to normal|active again|good standing"
    r"|no longer (?:suspended|restricted|limited|paused|on hold)"
    r"|(?:suspension|restriction|limitation|hold|pause)s?\s+(?:\w+\s+){0,6}?"
    r"(?:lifted|ended|removed|revoked|over|cleared))\b",
    re.IGNORECASE,
)
# Not "on hold": that is how lot holds are worded, and a hold misread as a status change would
# then suspend a supplier.
_SUSPENDED = re.compile(r"\b(?:suspen\w*|paus\w*|halt\w*|frozen|blocked|barred)", re.IGNORECASE)
_RESTRICTED = re.compile(r"\b(?:restrict\w*|limit\w*|priority orders?|capped)", re.IGNORECASE)
_ACTIVE = re.compile(r"\b(?:lifted|cleared|active)\b", re.IGNORECASE)
STATUS_PATTERNS = (
    (_ACTIVE_STRONG, SupplierStatus.ACTIVE),
    (_SUSPENDED, SupplierStatus.SUSPENDED),
    (_RESTRICTED, SupplierStatus.RESTRICTED),
    (_ACTIVE, SupplierStatus.ACTIVE),
)


class SpanParser:
    """``known_entities``, when given, closes the schema over one world: a proposal naming any
    other entity fails with slot ``entity`` (a schema failure, not a mis-scoped delta)."""

    def __init__(
        self, calendar_year: int = 2026, known_entities: Collection[str] | None = None
    ) -> None:
        self.calendar_year = calendar_year
        self.known_entities = None if known_entities is None else frozenset(known_entities)

    def complete(self, proposal: Proposal, message: Message, delta_id: str) -> Delta:
        if self.known_entities is not None and proposal.entity not in self.known_entities:
            raise ParseError("entity", f"unknown entity id {proposal.entity!r}")
        span = proposal.span
        if span.message_id != message.id or span.end > len(message.text):
            raise ParseError("span", "span does not point into this message")
        text = message.excerpt(span)
        spec = DELTA_SPECS[proposal.type]

        if (proposal.op is None) != (proposal.value is None):
            raise ParseError("value", "op and value must be given together")
        if proposal.op is not None and proposal.value is not None:
            op, value = proposal.op, proposal.value
        else:
            op, value = self._value_or_hint(proposal, text)

        try:
            window = (
                proposal.window
                or self.find_window(text, message)
                or _window_from_hints(proposal.hints)
                or Window(start=message.period)
            )
        except ValidationError as exc:
            raise ParseError("window", _first_error(exc)) from exc

        try:
            return Delta(
                id=delta_id,
                type=proposal.type,
                entity=proposal.entity,
                attribute=spec.attribute,
                op=op,
                value=value,
                unit=proposal.unit if proposal.unit is not None else _default_unit(spec, op),
                window=window,
                src=span,
                confidence=proposal.confidence,
                supersedes=proposal.supersedes,
            )
        except ValidationError as exc:
            raise ParseError("schema", _first_error(exc)) from exc

    def _value_or_hint(self, proposal: Proposal, text: str) -> tuple[Op, Value]:
        try:
            return self.read_value(proposal.type, text)
        except ParseError:
            hint = proposal.hints.get("magnitude")
            if not hint:
                raise
            return self.read_value(proposal.type, hint)

    def read_value(self, kind: DeltaType, text: str) -> tuple[Op, Value]:
        if kind is DeltaType.QUALITY_HOLD:
            return Op.SET, HoldStatus.HELD.value
        if kind is DeltaType.QUALITY_RELEASE:
            return Op.SET, HoldStatus.RELEASED.value
        if kind is DeltaType.SUPPLIER_STATUS:
            status = read_status(text)
            if status is None:
                raise ParseError("value", "no supplier status word")
            return Op.SET, status.value
        if kind is DeltaType.LEAD_TIME_CHANGE:
            days = _DAYS.search(text)
            if days is None:
                raise ParseError("value", "no day count")
            return Op.ADD, _signed(days, text, default=1)
        absolute = _ABSOLUTE.search(text)
        if absolute is not None:
            return Op.SET, float(absolute["num"].replace(",", ""))
        percent = _PERCENT.search(text)
        if percent is None:
            raise ParseError("value", "no percentage")
        default = 1 if kind is DeltaType.PRICE_CHANGE else None
        return Op.SCALE, percent_to_scale(_signed(percent, text, default=default))

    def read_window(self, text: str, message: Message) -> Window:
        """The window the text states, or open-ended from the message's period."""
        return self.find_window(text, message) or Window(start=message.period)

    def find_window(self, text: str, message: Message) -> Window | None:
        """The window the text states, or None when it states none."""
        year = self.calendar_year
        if match := _RANGE.search(text):
            return Window(start=int(match["a"]), end=int(match["b"]))
        if match := _FROM_WEEK.search(text):
            return Window(start=int(match["a"]))
        if match := _UNTIL_WEEK.search(text):
            return Window(start=message.period, end=int(match["b"]))
        if match := _UNTIL_MONTH.search(text):
            month = weeks.month_number(match["mon"])
            return Window(start=message.period, end=weeks.last_week_of_month(year, month))
        if match := _FROM_DATE.search(text):
            month = weeks.month_number(match["mon"])
            return Window(start=weeks.week_of(year, month, int(match["day"] or 1)))
        if match := _ONWARD.search(text):
            return Window(start=int(match["a"]))
        if match := _ONE_WEEK.search(text):
            week = int(match["a"])
            return Window(start=week, end=week)
        return None


def read_status(text: str) -> SupplierStatus | None:
    """The supplier status a text states, or None if it names none."""
    for pattern, status in STATUS_PATTERNS:
        if pattern.search(text):
            return status
    return None


def change_words(text: str) -> set[str]:
    """Words that state a status or a change (status words, up and down words), lowercased."""
    found: set[str] = set()
    for pattern in (*(p for p, _ in STATUS_PATTERNS), _UP, _DOWN):
        found |= {match.group().lower() for match in pattern.finditer(text)}
    return found


def _signed(match: re.Match[str], text: str, default: int | None) -> float:
    number = float(match["num"])
    sign = match["sign"]
    if sign:
        return number if sign == "+" else -number
    down, up = _DOWN.search(text), _UP.search(text)
    if down and (up is None or down.start() < up.start()):
        return -number
    if up:
        return number
    if default is None:
        raise ParseError("value", "no direction")
    return default * number


_HINT_WEEK = re.compile(r"^wk(\d{1,2})$")


def _window_from_hints(hints: Mapping[str, str]) -> Window | None:
    """A window from typed answers such as ``wk42`` / ``open-ended``; None if not stated."""
    start = _HINT_WEEK.match(hints.get("window_start", ""))
    if start is None:
        return None
    end = _HINT_WEEK.match(hints.get("window_end", ""))
    if end is None:
        return Window(start=int(start[1]))
    first, last = int(start[1]), int(end[1])
    return Window(start=first, end=last) if last >= first else Window(start=first)


def _default_unit(spec: DeltaSpec, op: Op) -> str | None:
    if spec.categorical:
        return None
    return "ratio" if op is Op.SCALE else spec.unit


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    return str(errors[0]["msg"]) if errors else str(exc)
