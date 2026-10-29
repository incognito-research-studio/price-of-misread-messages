"""Paraphrase layer (Section 5, step 3): rewrite template messages so interpreters must read
varied wording, while every gold delta stays valid.

Use a paraphrasing model from a family disjoint from every interpreter under test, so no
interpreter reads text written by its own family. A paraphrase is accepted only when

1. entity names, percentages and day counts survive verbatim (the model is told to keep them),
   the same weeks and dates are named (their wording may change: ``wk39–41`` may become
   ``weeks 39 to 41``), and no new name, number, week or date appears;
2. the span parser, given only the gold type and entity and the sentences that now carry the
   delta's names, numbers and dates, recovers the gold operation, value and window; and
3. two cues the parser does not read survive: a hold must not read as a release (and a release
   must say it is one), and a revision must still say it revises an earlier notice.

Check 2 keeps the L4 contract (the span supports the delta) and catches flipped directions.
What varies is the wording an interpreter must read to choose the type, the entity and the
span. A rejected paraphrase is retried with the next client (for example at a higher
temperature); after the last attempt the message keeps its template text and is listed as a
fallback in the stream's provenance. Messages with no gold delta get check 1 and one more: the
rewrite may not gain a word that states a status or a change (``parser.change_words``, plus
hold, release and revision words). "All open orders are on schedule" rewritten as "all active
orders..." reads as a reinstatement, and a reader that takes it so is not wrong about the text.

Clients with a ``seed`` attribute (such as the Ollama client) are reseeded for every request
from their own seed, the message id and the attempt number. Without this, a sampling model
with a fixed seed gives one rewrite per distinct template text, so a message repeated across
streams would read the same everywhere.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cdd import weeks
from cdd.benchmark.stream import Stream
from cdd.interpreters.llm import LLMClient
from cdd.parser import ParseError, SpanParser, change_words
from cdd.schema import Delta, DeltaType, Message, Proposal, Span
from cdd.text import sentence_spans
from cdd.world import World

PROMPT_VERSION = "p2"
CHECKS_VERSION = "c3"

SYSTEM_PROMPT = """\
You rewrite short operational messages (supplier emails, portal notices, carrier updates,
quality notes) so that they read as if a different person wrote them. Keep the meaning exactly:
the same change, the same direction, the same amounts and the same weeks and dates. Do not add
or remove information. Keep every string listed under KEEP exactly as written, character for
character. Week numbers and dates may be reworded (for example wk42 as week 42) but must name
the same weeks and dates. Do not add greetings, sign-offs or placeholders. Return only the
rewritten message, with no quotes, labels or commentary."""

_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")
_DAYS = re.compile(r"\d+\s*days?\b", re.IGNORECASE)
_WEEKS = re.compile(
    r"\b(?:wk|weeks?)\s*\d{1,2}"
    r"(?:\s*(?:-|–|—|to|through|thru|and)\s*(?:wk|weeks?)?\s*\d{1,2}\b)?",
    re.IGNORECASE,
)
# "May" counts only with a day number, so a sentence that opens with the verb is not a date.
_DATES = re.compile(
    r"\b(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May(?=\s+\d)|June?|July?"
    r"|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?"
    r"(?:\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\b)?"
)
_RELEASE_CUE = re.compile(r"\b(?:releas\w*|lift\w*|clear\w*|off hold)\b", re.IGNORECASE)
_REVISION_CUE = re.compile(
    r"\b(?:revis\w*|correct\w*|amend\w*|updat\w*|supersed\w*|replac\w*|earlier|previous\w*)\b",
    re.IGNORECASE,
)
_HOLD_CUE = re.compile(r"\b(?:hold|quarantin\w*|block\w*)\b", re.IGNORECASE)


def cue_words(text: str) -> set[str]:
    """Words that would make a message read as a change: status, direction, hold, release
    and revision words, lowercased."""
    found = change_words(text)
    for pattern in (_RELEASE_CUE, _REVISION_CUE, _HOLD_CUE):
        found |= {match.group().lower() for match in pattern.finditer(text)}
    return found


def invariants(text: str, world: World) -> list[str]:
    """Entity names, percentages and day counts that must survive verbatim, in order."""
    found: list[tuple[int, str]] = []
    for entity in world.entities():
        for form in entity.surface_forms:
            name = rf"(?<!\w){re.escape(form)}(?!\w)"
            found += [(m.start(), m.group()) for m in re.finditer(name, text, re.IGNORECASE)]
    for pattern in (_PERCENT, _DAYS):
        found += [(m.start(), m.group()) for m in pattern.finditer(text)]
    ordered: list[str] = []
    for _, value in sorted(found):
        if value not in ordered:
            ordered.append(value)
    return ordered


def time_mentions(text: str) -> list[tuple[int, int, str]]:
    """Week and date references as ``(start, end, key)``; the key ignores wording
    (``wk39–41`` and ``weeks 39 to 41`` both give ``wk39`` and ``wk41``)."""
    found = []
    for match in _WEEKS.finditer(text):
        for number in re.findall(r"\d+", match.group()):
            found.append((match.start(), match.end(), f"wk{int(number)}"))
    for match in _DATES.finditer(text):
        key = f"m{weeks.month_number(match['mon'])}"
        if match["day"]:
            key += f"d{int(match['day'])}"
        found.append((match.start(), match.end(), key))
    return found


def time_refs(text: str) -> set[str]:
    return {key for _, _, key in time_mentions(text)}


def clean_reply(reply: str) -> str:
    """Strip quotes and labels models sometimes add around the rewritten message."""
    text = reply.strip()
    text = re.sub(r"^(?:rewritten message|rewrite|message)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\s*(?:-{3,}|\*{3,})$", "", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return " ".join(text.split())


def evidence_span(text: str, keep: Sequence[str], dates: set[str], message_id: str) -> Span | None:
    """The sentences of ``text`` that carry ``keep`` and the ``dates`` keys, as one span;
    None if a ``keep`` string is missing."""
    positions = []
    for value in keep:
        start = text.find(value)
        if start < 0:
            return None
        positions.append((start, start + len(value)))
    positions += [(start, end) for start, end, key in time_mentions(text) if key in dates]
    sentences = sentence_spans(text) or [(0, len(text))]
    covered = [
        (s, e) for s, e in sentences if any(s <= a < e or s < b <= e for a, b in positions)
    ] or [(0, len(text))]
    return Span(message_id=message_id, start=covered[0][0], end=covered[-1][1])


def supports(delta: Delta, message: Message, span: Span, parser: SpanParser) -> bool:
    """Whether ``span`` still states ``delta``: the parser recovers its op, value and window,
    and the cues the parser does not read (hold or release, revision) survive."""
    text = message.excerpt(span)
    if delta.type is DeltaType.QUALITY_HOLD and _RELEASE_CUE.search(text):
        return False
    if delta.type is DeltaType.QUALITY_RELEASE and not _RELEASE_CUE.search(text):
        return False
    if delta.supersedes and not _REVISION_CUE.search(text):
        return False
    proposal = Proposal(
        type=delta.type, entity=delta.entity, span=span, supersedes=delta.supersedes
    )
    try:
        parsed = parser.complete(proposal, message, delta.id)
    except ParseError:
        return False
    if parsed.op != delta.op or parsed.window != delta.window:
        return False
    if isinstance(delta.value, str) or isinstance(parsed.value, str):
        return parsed.value == delta.value
    return math.isclose(parsed.value, delta.value, rel_tol=1e-6)


@runtime_checkable
class Seeded(Protocol):
    seed: int


def request_seed(base: int, message_id: str, attempt: int) -> int:
    """A seed that differs per message and attempt and does not depend on run order."""
    return zlib.crc32(f"{base}:{message_id}:{attempt}".encode())


@dataclass
class ParaphraseStats:
    messages: int = 0
    accepted: int = 0
    unchanged: int = 0
    attempts: int = 0
    fallback: list[str] = field(default_factory=list)


class Paraphraser:
    """Rewrites a stream's messages; ``clients`` are tried in order for each message."""

    def __init__(self, clients: Sequence[LLMClient], model: str = "") -> None:
        if not clients:
            raise ValueError("at least one client")
        self.clients = list(clients)
        self.model = model
        self.base_seeds = [c.seed if isinstance(c, Seeded) else 0 for c in self.clients]

    def stream(self, stream: Stream) -> Stream:
        world = stream.world
        parser = SpanParser(world.calendar_year, [entity.id for entity in world.entities()])
        stats = ParaphraseStats()
        messages: list[Message] = []
        gold: list[Delta] = []
        for message in stream.messages:
            stats.messages += 1
            deltas = stream.gold_for(message.id)
            new_message, new_gold = self._message(message, deltas, world, parser, stats)
            messages.append(new_message)
            gold.extend(new_gold)
        provenance = dict(stream.provenance)
        provenance["paraphrase"] = {
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "checks_version": CHECKS_VERSION,
            "messages": stats.messages,
            "accepted": stats.accepted,
            "unchanged": stats.unchanged,
            "attempts": stats.attempts,
            "fallback": stats.fallback,
        }
        return stream.model_copy(
            update={"messages": tuple(messages), "gold": tuple(gold), "provenance": provenance}
        )

    def _message(
        self,
        message: Message,
        deltas: Sequence[Delta],
        world: World,
        parser: SpanParser,
        stats: ParaphraseStats,
    ) -> tuple[Message, list[Delta]]:
        keep = invariants(message.text, world)
        user = f"KEEP: {json.dumps(keep)}\n\nMESSAGE:\n{message.text}"
        for attempt, (client, base) in enumerate(zip(self.clients, self.base_seeds, strict=True)):
            if isinstance(client, Seeded):
                client.seed = request_seed(base, message.id, attempt)
            stats.attempts += 1
            text = clean_reply(client.complete(SYSTEM_PROMPT, user))
            accepted = _accept(text, message, deltas, keep, world, parser)
            if accepted is not None:
                stats.accepted += 1
                stats.unchanged += text == message.text
                return accepted
        stats.fallback.append(message.id)
        return message, list(deltas)


def _accept(
    text: str,
    message: Message,
    deltas: Sequence[Delta],
    keep: Sequence[str],
    world: World,
    parser: SpanParser,
) -> tuple[Message, list[Delta]] | None:
    # A reply that stops mid-sentence (often an echoed KEEP string) is not a message.
    if not re.search(r"[.!?)]$", text) or any(value not in text for value in keep):
        return None
    allowed = {value.lower() for value in keep}
    if any(value.lower() not in allowed for value in invariants(text, world)):
        return None
    if time_refs(text) != time_refs(message.text):
        return None
    if not deltas and cue_words(text) - cue_words(message.text):
        return None
    candidate = message.model_copy(update={"text": text})
    gold = []
    for delta in deltas:
        excerpt = message.excerpt(delta.src)
        span = evidence_span(text, invariants(excerpt, world), time_refs(excerpt), message.id)
        if span is None or not supports(delta, candidate, span, parser):
            return None
        gold.append(delta.model_copy(update={"src": span}))
    return candidate, gold


def provenance_summary(streams: Sequence[Stream]) -> dict[str, Any]:
    """Totals across streams, and fallbacks per delta type as ``"fallbacks/messages"``."""
    totals: dict[str, Any] = dict.fromkeys(
        ("messages", "accepted", "unchanged", "attempts", "fallback"), 0
    )
    fallback: dict[str, int] = {}
    messages: dict[str, int] = {}
    for stream in streams:
        info = stream.provenance.get("paraphrase", {})
        for key in ("messages", "accepted", "unchanged", "attempts"):
            totals[key] += int(info.get(key, 0))
        totals["fallback"] += len(info.get("fallback", []))
        fell_back = set(info.get("fallback", []))
        for message in stream.messages:
            gold = stream.gold_for(message.id)
            kind = gold[0].type.value if gold else "no_change"
            messages[kind] = messages.get(kind, 0) + 1
            fallback[kind] = fallback.get(kind, 0) + (message.id in fell_back)
    totals["fallback_by_type"] = {k: f"{fallback[k]}/{messages[k]}" for k in sorted(messages)}
    return totals
