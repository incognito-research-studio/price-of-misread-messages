"""Conditions (b) and (c): an LLM returns proposals as JSON.

The module is provider-agnostic: pass any object with ``complete(system, user, schema)``
(see ``docs/adapters.md``). Condition (b) leaves the output free-form; condition (c) passes a
JSON Schema for clients that support constrained decoding. The schema restricts the type to the
delta types, the entity to this world's ids of the kind that type needs (a demand signal names
a SKU, never an offer), the evidence to one of the message's sentences and ``supersedes`` to the
ids of deltas in force, so (c) cannot misquote or cite a notice that does not exist. Typed
readers have the same restriction, since they choose the entity after the type. The model
chooses the delta type, the entity and a verbatim evidence string. The interpreter turns the
evidence into a span and the span parser fills the value and window, so a model cannot write a
value the message does not support. Output that does not parse is recorded as a failure, not
repaired.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from cdd.interpreters.base import Interpretation
from cdd.prompts import render_deltas, render_entities, render_message, render_types
from cdd.schema import DELTA_SPECS, DeltaType, Message, Proposal, Span
from cdd.state import StateView
from cdd.text import sentence_spans
from cdd.world import World


class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        """Return the model's reply. ``schema``, when given, is a JSON Schema the reply must
        satisfy; clients without constrained decoding may ignore it."""
        ...


def deltas_schema(
    world: World, message: Message | None = None, view: StateView | None = None
) -> dict[str, Any]:
    """JSON Schema for interpreter replies, restricted to the schema's types and this world;
    with ``message``, evidence must be one of its sentences, and with ``view``, ``supersedes``
    may list only the deltas in force."""
    evidence: dict[str, Any] = {"type": "string"}
    if message is not None:
        sentences = list(dict.fromkeys(message.text[s:e] for s, e in sentence_spans(message.text)))
        if sentences:
            evidence = {"type": "string", "enum": sentences}
    supersedes: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    if view is not None:
        in_force = [delta.id for delta in view.active()]
        supersedes = (
            {"type": "array", "items": {"type": "string", "enum": in_force}}
            if in_force
            else {"type": "array", "maxItems": 0}
        )
    items = [
        {
            "type": "object",
            "properties": {
                "type": {"const": kind.value},
                "entity": {
                    "type": "string",
                    "enum": [e.id for e in world.entities(DELTA_SPECS[kind].entity_kind)],
                },
                "evidence": evidence,
                "supersedes": supersedes,
            },
            "required": ["type", "entity", "evidence"],
        }
        for kind in DeltaType
    ]
    return {
        "type": "object",
        "properties": {"deltas": {"type": "array", "items": {"anyOf": items}}},
        "required": ["deltas"],
    }


SYSTEM_PROMPT = f"""\
You read one operational message and report the changes it makes to the planning state.
Use only these delta types (the quoted examples illustrate each type; they are not messages):
{render_types()}

"entity" is the id exactly as listed (for example P2 or LOT-7781), never the name. Copy the
evidence verbatim from the message below, never from the examples or the deltas in force: the
shortest sentence or clause that states the change.
Fill "supersedes" only when the message explicitly revises, corrects or replaces an earlier
notice; then list the id of that notice's delta. If the message changes nothing, return no
deltas.
Answer with JSON only, in this shape:
{{"deltas": [{{"type": "<type>", "entity": "<entity id>",
              "evidence": "<verbatim text>", "supersedes": []}}]}}"""


def render_user_prompt(message: Message, view: StateView) -> str:
    return (
        f"Entities:\n{render_entities(view.world)}\n\n"
        f"Deltas in force:\n{render_deltas(view)}\n\n"
        f"{render_message(message)}"
    )


class JsonLLMInterpreter:
    def __init__(
        self, client: LLMClient, name: str = "llm-json", *, constrained: bool = False
    ) -> None:
        self.client = client
        self.name = name
        self.constrained = constrained

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        schema = deltas_schema(view.world, message, view) if self.constrained else None
        raw = self.client.complete(SYSTEM_PROMPT, render_user_prompt(message, view), schema)
        try:
            items = parse_deltas(raw)
        except ValueError as exc:
            return Interpretation(failures=(f"malformed output: {exc}",))
        proposals = []
        failures = []
        for item in items:
            try:
                proposals.append(_proposal(item, message))
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(f"unusable delta {item!r}: {exc}")
        return Interpretation(tuple(proposals), tuple(failures))


def parse_deltas(raw: str) -> list[Any]:
    """Extract the ``deltas`` list from a model reply, tolerating code fences around it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    payload = json.loads(raw[start : end + 1])
    deltas = payload.get("deltas") if isinstance(payload, dict) else None
    if not isinstance(deltas, list):
        raise ValueError('reply has no "deltas" list')
    return deltas


def _proposal(item: Any, message: Message) -> Proposal:
    evidence = str(item["evidence"]).strip()
    start = message.text.find(evidence) if evidence else -1
    if start < 0:
        raise ValueError("evidence is not a substring of the message")
    return Proposal(
        type=item["type"],
        entity=item["entity"],
        span=Span(message_id=message.id, start=start, end=start + len(evidence)),
        supersedes=tuple(item.get("supersedes") or ()),
    )
