"""Surface templates for delta-first generation.

Each template renders one clause that states one delta. ``prefix`` is text before the clause
(a greeting or channel label) and is not part of the gold span. Every template must round-trip:
the span parser, given only the gold type, entity and span, recovers the gold value and window.
``tests/test_generator.py`` checks this for every template.

The paraphrase layer (``paraphrase.py``) rewrites rendered messages with a held-out LLM family
and keeps only rewrites that still round-trip. Vague and conflicting messages and adversarial
approval injections are later difficulty factors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    channel: str
    clause: str
    prefix: str = ""


CAPACITY_DOWN = (
    Template("portal", "{plant} output reduced ~{pct}% {window}."),
    Template(
        "email",
        "{plant} will run at reduced capacity, down {pct}% {window}, due to line maintenance.",
        prefix="Hi team, ",
    ),
)
CAPACITY_UP = (
    Template("portal", "{plant} capacity increased {pct}% {window} after a shift change."),
)
CAPACITY_REVISION = (
    Template(
        "portal", "Update to earlier notice: {plant} capacity reduction revised to {pct}% {window}."
    ),
)

LEAD_TIME_UP = (
    Template("carrier", "Transit via {lane} +{days} days {window}."),
    Template(
        "carrier", "Shipments on {lane} delayed by {days} days {window}.", prefix="Carrier update: "
    ),
)
LEAD_TIME_DOWN = (Template("carrier", "Transit via {lane} improved by {days} days {window}."),)

HOLD = (
    Template("quality", "{lot} on hold pending CoA.", prefix="QA note: "),
    Template("email", "QA has placed {lot} on hold after a failed inspection."),
)
RELEASE = (
    Template("quality", "{lot} released.", prefix="QA note: "),
    Template("email", "Good news: {lot} has been released by QA."),
)

SUSPEND = (Template("email", "{supplier} suspended pending audit.", prefix="Procurement: "),)
RESTRICT = (Template("email", "{supplier} restricted to priority orders {window}."),)
REINSTATE = (Template("email", "{supplier} reinstated after a successful audit."),)

# b1 only: "pulls forward" also implies a later decrease, which the gold delta lacked.
DEMAND_PULL = (Template("email", "Key account pulls {pct}% forward into wk{week} for {sku}."),)
# b2: the same one-week increase, stated plainly.
DEMAND_SPIKE = (Template("email", "Key account orders for {sku} up {pct}% in wk{week}."),)
DEMAND_UP = (Template("email", "Forecast for {sku} up {pct}% {window}."),)
DEMAND_DOWN = (Template("email", "Forecast for {sku} down {pct}% {window}."),)

PRICE_UP = (Template("email", "{supplier}: surcharge {pct}% on {sku} {window}."),)
PRICE_DOWN = (Template("email", "{supplier} offers a {pct}% discount on {sku} {window}."),)

# Out-of-schema: a yield shortfall changes what arrives, and no delta type can say so.
OUT_OF_SCHEMA = (
    Template(
        "email",
        "Heads-up: shipments from {plant} will arrive about {pct}% short {window} because of "
        "packaging defects.",
    ),
)

NO_CHANGE = (
    Template("email", "Reminder: {plant} audit visit scheduled for next month."),
    Template("email", "{supplier} confirms all open orders are on schedule."),
    Template("portal", "Portal maintenance on Saturday; expect brief downtime."),
    Template("portal", "{plant} output unchanged this week."),
    Template("email", "Please send updated contact details for {supplier}."),
)

DISTRACTORS = (" Please confirm receipt.", " Thanks.", " Reply to this thread with questions.")

RANGE_FORMS = ("wk{a}–{b}", "weeks {a}-{b}", "wk{a} to wk{b}")
FROM_FORMS = ("from wk{a}", "effective wk{a}", "starting week {a}")
