"""Condition (f): a regex and gazetteer baseline with no learned components.

Each sentence gets at most one proposal: the first rule whose keyword appears and whose entity
kind is mentioned. Values and windows are left to the span parser. A sentence that announces a
revision supersedes the latest delta in force with the same type and entity.
"""

from __future__ import annotations

import re

from cdd.interpreters.base import Interpretation
from cdd.schema import DELTA_SPECS, DeltaType, EntityKind, Message, Proposal, Span
from cdd.state import StateView
from cdd.text import Gazetteer, sentence_spans
from cdd.world import World

_RULES: tuple[tuple[DeltaType, re.Pattern[str]], ...] = (
    (DeltaType.QUALITY_RELEASE, re.compile(r"\breleas\w*", re.IGNORECASE)),
    (DeltaType.QUALITY_HOLD, re.compile(r"\bhold\b", re.IGNORECASE)),
    (
        DeltaType.SUPPLIER_STATUS,
        re.compile(r"\b(?:suspen\w*|restrict\w*|reinstat\w*)", re.IGNORECASE),
    ),
    (
        DeltaType.LEAD_TIME_CHANGE,
        re.compile(r"\b(?:transit|lead[- ]?times?|delay\w*|eta|shipments?)\b", re.IGNORECASE),
    ),
    (DeltaType.CAPACITY_CHANGE, re.compile(r"\b(?:output|capacity|production)\b", re.IGNORECASE)),
    (
        DeltaType.DEMAND_SIGNAL,
        re.compile(r"\b(?:forecast|demand|pulls?|key account)\b", re.IGNORECASE),
    ),
    (DeltaType.PRICE_CHANGE, re.compile(r"\b(?:surcharge|price|discount)\w*", re.IGNORECASE)),
)
_REVISION = re.compile(r"\b(?:revis\w*|correct\w*|update to earlier)\b", re.IGNORECASE)


class RegexInterpreter:
    name = "regex"

    def __init__(self, world: World) -> None:
        self.gazetteer = Gazetteer(world)

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        proposals = []
        for start, end in sentence_spans(message.text):
            sentence = message.text[start:end]
            for kind, keyword in _RULES:
                if not keyword.search(sentence):
                    continue
                entity = self._entity(kind, sentence)
                if entity is None:
                    continue
                revision = _REVISION.search(sentence)
                proposals.append(
                    Proposal(
                        type=kind,
                        entity=entity,
                        span=Span(message_id=message.id, start=start, end=end),
                        supersedes=self._latest(kind, entity, view) if revision else (),
                    )
                )
                break
        return Interpretation(tuple(proposals))

    def _entity(self, kind: DeltaType, sentence: str) -> str | None:
        entity_kind = DELTA_SPECS[kind].entity_kind
        if entity_kind is EntityKind.OFFER:
            return self.gazetteer.find_offer(sentence)
        return self.gazetteer.find(sentence, entity_kind)

    @staticmethod
    def _latest(kind: DeltaType, entity: str, view: StateView) -> tuple[str, ...]:
        matches = [d.id for d in view.active() if d.type is kind and d.entity == entity]
        return (matches[-1],) if matches else ()
