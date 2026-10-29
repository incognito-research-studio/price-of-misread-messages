"""Fact sheets for the WP2 challenge set (docs/research/icores-revision-scope.md, Section 4).

usage: python scripts/challenge_fact_sheets.py STREAMS_DIR OUT_DIR STREAM_ID [STREAM_ID ...]

For every message of the chosen streams, writes what a writer needs to rewrite it without seeing
the template wording: week and date, sender, the change in plain words from the gold deltas (or
"no change"), and an assigned language feature, so the set covers negation, cancellation,
corrections, calendar dates, distractor numbers and unfamiliar structure. OUT_DIR gets:

- fact-sheets.md: the sheets, one section per stream, for the writers;
- manifest.json: streams, message ids, assigned features and a hash of each stream's messages
  and gold deltas, so a regenerated copy (e.g. on another machine) can be compared;
- rewrites.yaml: one empty entry per message for the writer's text and the verifier's check.

The template text is never written to OUT_DIR. Features are assigned in a fixed rotation by
message order, not by any reader outcome.
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

from cdd.benchmark.stream import Stream, load_streams
from cdd.schema import Delta, DeltaType, Op
from cdd.world import Lane, Lot, Offer, Plant, World

streams_dir, out = sys.argv[1], Path(sys.argv[2])
chosen = sys.argv[3:]

CHANGE_FEATURES = (
    "write the date as a calendar date (e.g. 'from 5 October') instead of a week number",
    "add an irrelevant number (an order count, a phone extension, a reference number)",
    "unfamiliar structure: put the change at the end of a longer sentence, or use passive voice",
    "negation: also say what is NOT changing (another site, lane or product, by name)",
    "informal or abbreviated wording, as in a quick chat message",
)
CANCEL_FEATURE = "cancellation wording: say the earlier restriction is lifted, withdrawn or ended"
REVISION_FEATURE = "correction: say that this corrects or replaces the earlier notice"
NO_CHANGE_FEATURES = (
    "mention a named site, lane, lot or product and a number, but change nothing",
    "use a change word in a negated or routine sense (e.g. 'no delay expected at Port X', "
    "'order plan unchanged') and change nothing",
)


def monday(world: World, week: int) -> datetime.date:
    return datetime.date.fromisocalendar(world.calendar_year, week, 1)


def when(world: World, delta: Delta, period: int) -> str:
    start, end = delta.window.start, delta.window.end
    first = f"week {start} (Mon {monday(world, start):%d %b %Y})"
    if end is None:
        note = " (the week of the message; may be left implicit)" if start == period else ""
        return f"from {first}{note} until further notice"
    if end == start:
        return f"{first} only"
    last = monday(world, end) + datetime.timedelta(days=6)
    return f"from {first} through week {end} (Sun {last:%d %b %Y})"


def entity(world: World, entity_id: str) -> str:
    e = world.entity(entity_id)
    if isinstance(e, Plant):
        return f"{e.name} (a plant of {world.entity(e.supplier).name})"
    if isinstance(e, Lane):
        return f"the lane through {e.name} (shipments from {world.entity(e.plant).name})"
    if isinstance(e, Offer):
        return f"{world.entity(e.supplier).name}'s price for {world.entity(e.sku).name}"
    if isinstance(e, Lot):
        return f"{e.name} (stock of {world.entity(e.sku).name})"
    return e.name


def change(delta: Delta) -> str:
    v, op = delta.value, delta.op
    if delta.type is DeltaType.QUALITY_HOLD:
        return "placed on quality hold"
    if delta.type is DeltaType.QUALITY_RELEASE:
        return "released from quality hold"
    if delta.type is DeltaType.SUPPLIER_STATUS:
        meaning = {
            "active": "active again (reinstated, orders accepted normally)",
            "restricted": "restricted (accepts only part of its normal volume)",
            "suspended": "suspended (no new orders accepted)",
        }
        return f"supplier status becomes {meaning[str(v)]}"
    assert isinstance(v, int | float)
    what, unit = {
        DeltaType.CAPACITY_CHANGE: ("capacity", "units per week"),
        DeltaType.DEMAND_SIGNAL: ("customer orders", "units per week"),
        DeltaType.PRICE_CHANGE: ("price", "per unit"),
        DeltaType.LEAD_TIME_CHANGE: ("transit time", "days"),
    }[delta.type]
    if op is Op.SCALE:
        percent = round((v - 1) * 100, 6)
        return f"{what} {'up' if percent > 0 else 'down'} {abs(percent):g}%"
    if op is Op.ADD:
        if delta.type is DeltaType.LEAD_TIME_CHANGE:
            return f"{what} {'longer' if v > 0 else 'shorter'} by {abs(v):g} {unit}"
        return f"{what} {'up' if v > 0 else 'down'} by {abs(v):g} {unit}"
    return f"{what} set to {v:g} {unit}"


def stream_hash(stream: Stream) -> str:
    payload = {
        "messages": [m.model_dump(mode="json") for m in stream.messages],
        "gold": [d.model_dump(mode="json") for d in stream.gold],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


streams = {s.id: s for s in load_streams(streams_dir)}
missing = [sid for sid in chosen if sid not in streams]
if missing:
    sys.exit(f"streams not found in {streams_dir}: {missing}")

out.mkdir(parents=True, exist_ok=True)
lines = [
    "# WP2 challenge: fact sheets",
    "",
    "Rewrite each message in your own words so that it states exactly the facts below and nothing",
    "more. Keep the sender. Use the assigned language feature. Do not look at the generated",
    "message texts, the parser or any reader output, and do not test your text with them.",
    "Name every site, lane, lot, product or supplier the facts name. A no-change message must",
    "not imply any change. One message, one change: never add a second change.",
    "",
]
manifest: dict = {"streams": [], "messages": []}
rewrites = [
    "# WP2 challenge rewrites. Writer fills text; verifier fills evidence and verified.",
    "",
]
counter = {"change": 0, "nochange": 0}
for sid in chosen:
    stream = streams[sid]
    world = stream.world
    gold_by_id = {d.id: d for d in stream.gold}
    period_of = {m.id: m.period for m in stream.messages}
    manifest["streams"].append(
        {"id": sid, "messages": len(stream.messages), "sha256": stream_hash(stream)}
    )
    lines += [f"## Stream {sid}", ""]
    for message in stream.messages:
        gold = stream.gold_for(message.id)
        if gold:
            delta = gold[0]
            if delta.supersedes:
                feature = REVISION_FEATURE
            elif delta.type is DeltaType.QUALITY_RELEASE or (
                delta.type is DeltaType.SUPPLIER_STATUS and delta.value == "active"
            ):
                feature = CANCEL_FEATURE
            else:
                feature = CHANGE_FEATURES[counter["change"] % len(CHANGE_FEATURES)]
                counter["change"] += 1
            facts = [
                f"{entity(world, delta.entity)}: {change(delta)}, "
                f"{when(world, delta, message.period)}."
            ]
            for old_id in delta.supersedes:
                old = gold_by_id[old_id]
                facts.append(
                    f"This corrects the notice of week {period_of[old.message_id]} "
                    f"({old.message_id}), which said: {change(old)}, "
                    f"{when(world, old, period_of[old.message_id])}."
                )
            if delta.type is DeltaType.SUPPLIER_STATUS:
                facts.append("Name the supplier in the sentence that states the status.")
        else:
            feature = NO_CHANGE_FEATURES[counter["nochange"] % len(NO_CHANGE_FEATURES)]
            counter["nochange"] += 1
            facts = [
                "No change to any capacity, transit time, hold, supplier status, demand or price."
            ]
        lines += [
            f"### {message.id}",
            "",
            f"- Week {message.period} (week of Mon {monday(world, message.period):%d %b %Y}); "
            f"sender: {message.sender} ({message.sender_class}); channel: {message.channel}",
            *[f"- {fact}" for fact in facts],
            f"- Language feature: {feature}",
            "",
        ]
        manifest["messages"].append(
            {"id": message.id, "stream": sid, "changes": len(gold), "feature": feature}
        )
        rewrites += [
            f"- id: {message.id}",
            "  writer: ''",
            "  text: ''",
            "  evidence: ''       # verifier: exact sentence(s) of text that state the change",
            "  verified_by: ''",
            "  verified: false    # true only if the text states exactly the facts on the sheet",
            "  notes: ''",
            "",
        ]

(out / "fact-sheets.md").write_text("\n".join(lines))
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(out / "rewrites.yaml").write_text("\n".join(rewrites))
n = len(manifest["messages"])
changes = sum(m["changes"] > 0 for m in manifest["messages"])
print(f"{n} messages in {len(chosen)} streams: {changes} with a change, {n - changes} without")
