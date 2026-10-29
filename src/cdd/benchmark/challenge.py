"""The unfiltered language challenge (WP2): hand-written message texts, imported unchanged.

Writers rewrite each message of a few pilot streams from a fact sheet; a verifier checks each
rewrite against its facts and marks the evidence, the exact part of the text that states the
change. :func:`import_challenge` swaps the rewrites into the streams and re-anchors each gold
delta's span to its evidence. Gold types, entities, values and windows are not touched, and no
message is dropped or tested against the parser: every verified rewrite is kept, including those
the parser cannot read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cdd.benchmark.stream import Stream
from cdd.schema import Span


@dataclass(frozen=True)
class Rewrite:
    id: str
    writer: str
    text: str
    evidence: str
    verified_by: str
    verified: bool
    notes: str = ""


class ChallengeError(ValueError):
    """Raised with every problem found, so the verifier can fix them in one pass."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = tuple(problems)


def load_rewrites(path: str | Path) -> tuple[list[Rewrite], str]:
    """The entries of a rewrites file and the file's sha256."""
    raw = Path(path).read_bytes()
    entries: list[dict[str, Any]] = yaml.safe_load(raw) or []
    rewrites = [
        Rewrite(
            id=str(entry.get("id", "")),
            writer=str(entry.get("writer") or "").strip(),
            text=str(entry.get("text") or "").strip(),
            evidence=str(entry.get("evidence") or "").strip(),
            verified_by=str(entry.get("verified_by") or "").strip(),
            verified=entry.get("verified") is True,
            notes=str(entry.get("notes") or "").strip(),
        )
        for entry in entries
    ]
    return rewrites, hashlib.sha256(raw).hexdigest()


def import_challenge(
    streams: Sequence[Stream], rewrites: Sequence[Rewrite], rewrites_sha256: str = ""
) -> list[Stream]:
    """Streams with every message text replaced by its verified rewrite.

    Refuses (``ChallengeError``) unless every message of ``streams`` has exactly one entry, every
    entry is verified and names its writer and verifier, a message with a change has evidence
    that occurs exactly once in its text, and a message without a change has no evidence.
    """
    by_id: dict[str, Rewrite] = {}
    problems: list[str] = []
    for rewrite in rewrites:
        if rewrite.id in by_id:
            problems.append(f"{rewrite.id}: two entries")
        by_id[rewrite.id] = rewrite
    known = {message.id for stream in streams for message in stream.messages}
    problems += [f"{rid}: no such message in these streams" for rid in sorted(set(by_id) - known)]

    imported: list[Stream] = []
    for stream in streams:
        messages, gold = [], list(stream.gold)
        for message in stream.messages:
            entry = by_id.get(message.id)
            if entry is None:
                problems.append(f"{message.id}: no rewrite")
                continue
            changes = stream.gold_for(message.id)
            problems += _check(entry, has_change=bool(changes))
            messages.append(message.model_copy(update={"text": entry.text}))
            start = entry.text.find(entry.evidence) if entry.evidence else -1
            if changes and start >= 0:
                span = Span(message_id=message.id, start=start, end=start + len(entry.evidence))
                gold = [
                    delta.model_copy(update={"src": span})
                    if delta.message_id == message.id
                    else delta
                    for delta in gold
                ]
        mine = [by_id[m.id] for m in stream.messages if m.id in by_id]
        provenance = dict(stream.provenance)
        provenance["challenge"] = {
            "rewrites_sha256": rewrites_sha256,
            "writers": sorted({r.writer for r in mine}),
            "verifiers": sorted({r.verified_by for r in mine}),
            "unfiltered": True,
        }
        imported.append(
            stream.model_copy(
                update={"messages": tuple(messages), "gold": tuple(gold), "provenance": provenance}
            )
        )
    if problems:
        raise ChallengeError(problems)
    return imported


def _check(rewrite: Rewrite, *, has_change: bool) -> list[str]:
    problems = []
    if not rewrite.text:
        problems.append(f"{rewrite.id}: empty text")
    if not rewrite.verified:
        problems.append(f"{rewrite.id}: not verified")
    if not rewrite.writer or not rewrite.verified_by:
        problems.append(f"{rewrite.id}: writer or verifier missing")
    if has_change:
        if not rewrite.evidence:
            problems.append(f"{rewrite.id}: a change but no evidence")
        elif rewrite.text.count(rewrite.evidence) != 1:
            found = rewrite.text.count(rewrite.evidence)
            problems.append(
                f"{rewrite.id}: evidence occurs {found} times in the text (must be exactly once)"
            )
    elif rewrite.evidence:
        problems.append(f"{rewrite.id}: no change, so evidence must be empty")
    return problems
