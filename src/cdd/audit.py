"""Audit log (L6): lineage from each decision to deltas, spans and SOP rules.

Records are plain JSON-serialisable dictionaries, written one per line. *Audit completeness* is
the share of decisions whose lineage can be reconstructed: every delta the decision relied on
has an admissibility record, and its source message is in the log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cdd.orders import Plan
from cdd.schema import Delta, Message
from cdd.sop import Admissibility
from cdd.update import Conflict, UpdateResult


class AuditLog:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def message(self, message: Message) -> None:
        self.records.append({"kind": "message", **message.model_dump(mode="json")})

    def failure(self, message_id: str, slot: str, reason: str) -> None:
        self.records.append(
            {"kind": "failure", "message_id": message_id, "slot": slot, "reason": reason}
        )

    def admissibility(self, delta: Delta, verdict: Admissibility, admitted: bool) -> None:
        self.records.append(
            {
                "kind": "sop",
                "delta": delta.model_dump(mode="json"),
                "verdict": verdict.verdict.value,
                "rule_id": verdict.rule_id,
                "reason": verdict.reason,
                "admitted": admitted,
            }
        )

    def update(self, period: int, result: UpdateResult) -> None:
        self.records.append(
            {
                "kind": "update",
                "period": period,
                "asserted": list(result.asserted),
                "merged": [list(pair) for pair in result.merged],
                "retracted": list(result.retracted),
                "unresolved_supersedes": list(result.unresolved_supersedes),
                "restated": [list(pair) for pair in result.restated],
                "conflicts": [[d.id for d in c.deltas] for c in result.conflicts],
                "diff": [
                    {
                        "entity": d.entity,
                        "attribute": d.attribute.value,
                        "period": d.period,
                        "before": d.before,
                        "after": d.after,
                    }
                    for d in result.diff
                ],
            }
        )

    def resolution(self, period: int, conflict: Conflict, chosen: Delta | None) -> None:
        self.records.append(
            {
                "kind": "resolution",
                "period": period,
                "conflict": [d.id for d in conflict.deltas],
                "chosen": None if chosen is None else chosen.id,
            }
        )

    def decision(self, plan: Plan, lineage: list[str]) -> None:
        self.records.append(
            {
                "kind": "decision",
                "period": plan.period,
                "status": plan.status,
                "orders": [
                    {"plant": o.plant, "sku": o.sku, "quantity": o.quantity} for o in plan.orders
                ],
                "lineage": lineage,
            }
        )

    def completeness(self) -> float:
        """Share of decisions with fully reconstructable lineage (1.0 when there are none)."""
        messages = {r["id"] for r in self.records if r["kind"] == "message"}
        admitted = {
            r["delta"]["id"]: r["delta"]["src"]["message_id"]
            for r in self.records
            if r["kind"] == "sop" and r["admitted"]
        }
        decisions = [r for r in self.records if r["kind"] == "decision"]
        if not decisions:
            return 1.0
        complete = sum(
            all(admitted.get(delta_id) in messages for delta_id in record["lineage"])
            for record in decisions
        )
        return complete / len(decisions)

    def write_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
