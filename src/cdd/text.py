"""Text helpers shared by interpreters: sentence spans and entity lookup by surface form."""

from __future__ import annotations

import re

from cdd.schema import EntityKind
from cdd.world import World

_SENTENCE = re.compile(r"(?:[^.!?\n]|(?<=\d)\.(?=\d))+(?:[.!?]+|$)")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in ``text``, without surrounding whitespace."""
    spans = []
    for match in _SENTENCE.finditer(text):
        chunk = match.group()
        stripped = chunk.strip()
        if stripped:
            start = match.start() + chunk.index(stripped)
            spans.append((start, start + len(stripped)))
    return spans


class Gazetteer:
    """Finds entity mentions by name or alias, matching whole words and ignoring case."""

    def __init__(self, world: World) -> None:
        self.world = world
        self._patterns: dict[EntityKind, list[tuple[re.Pattern[str], str]]] = {}
        for kind in EntityKind:
            forms = [
                (form, entity.id)
                for entity in world.entities(kind)
                for form in entity.surface_forms
            ]
            forms.sort(key=lambda pair: -len(pair[0]))
            self._patterns[kind] = [
                (re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.IGNORECASE), entity_id)
                for form, entity_id in forms
            ]

    def find(self, text: str, kind: EntityKind) -> str | None:
        """The earliest mention of an entity of ``kind``; the longest name wins a tie."""
        best: tuple[int, str] | None = None
        for pattern, entity_id in self._patterns[kind]:
            match = pattern.search(text)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), entity_id)
        return None if best is None else best[1]

    def find_offer(self, text: str) -> str | None:
        supplier = self.find(text, EntityKind.SUPPLIER)
        sku = self.find(text, EntityKind.SKU)
        if supplier is None or sku is None:
            return None
        offer = self.world.offer_by_parts(supplier, sku)
        return None if offer is None else offer.id

    def surface_forms(self, kind: EntityKind) -> dict[str, str]:
        """Display name to entity id, for prompts and typed-question options."""
        return {entity.name: entity.id for entity in self.world.entities(kind)}
