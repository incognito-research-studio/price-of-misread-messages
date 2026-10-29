"""Recorded reader output: what one reader wrote on one stream, kept for offline replay.

The experiment runner appends one record per compiled (stream, condition) pair to
``proposals.jsonl``: the completed deltas the span parser produced (before the SOP layer),
the parse and interpreter failures, and the admissibility verdicts. The front end never sees a
decision, so replaying the completed deltas through the same front end rebuilds the same
interpreted state ŝ under any downstream policy, without asking the reader again. Repairs
already work this way (:func:`cdd.experiments.score_compiled`).

Parse failures produce no delta, so a replay reproduces cost, alignment and errors but reports
no parse failures of its own; the record keeps them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cdd.interpreters.replay import ReplayInterpreter
from cdd.schema import Delta

if TYPE_CHECKING:
    from cdd.pipeline import CompilingPolicy

RECORDS_FILE = "proposals.jsonl"


@dataclass(frozen=True)
class ReaderRecord:
    stream: str
    condition: str
    interpreter: str
    proposed: tuple[Delta, ...]
    # (message id, slot, reason), as in CompilingPolicy.failures
    failures: tuple[tuple[str, str, str], ...] = ()
    # one entry per parsed delta: delta id, verdict, rule id, admitted
    admissibility: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_policy(
        cls, stream: str, condition: str, interpreter: str, policy: CompilingPolicy
    ) -> ReaderRecord:
        return cls(
            stream=stream,
            condition=condition,
            interpreter=interpreter,
            proposed=tuple(policy.proposed),
            failures=tuple(policy.failures),
            admissibility=tuple(
                {
                    "delta_id": record["delta"]["id"],
                    "verdict": record["verdict"],
                    "rule_id": record["rule_id"],
                    "admitted": record["admitted"],
                }
                for record in policy.audit.records
                if record["kind"] == "sop"
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "condition": self.condition,
            "interpreter": self.interpreter,
            "proposed": [delta.model_dump(mode="json") for delta in self.proposed],
            "failures": [list(failure) for failure in self.failures],
            "admissibility": list(self.admissibility),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ReaderRecord:
        return cls(
            stream=data["stream"],
            condition=data["condition"],
            interpreter=data["interpreter"],
            proposed=tuple(Delta.model_validate(delta) for delta in data["proposed"]),
            failures=tuple((m, slot, reason) for m, slot, reason in data["failures"]),
            admissibility=tuple(data["admissibility"]),
        )

    def replay(self, name: str | None = None) -> ReplayInterpreter:
        """An interpreter that returns the recorded completed deltas, keeping their ids."""
        return ReplayInterpreter(self.proposed, name=name or self.condition)


def read_records(path: str | Path) -> dict[tuple[str, str], ReaderRecord]:
    """Records by (stream, condition), from a run folder or a ``proposals.jsonl`` file."""
    path = Path(path)
    if path.is_dir():
        path = path / RECORDS_FILE
    records: dict[tuple[str, str], ReaderRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = ReaderRecord.from_json(json.loads(line))
            key = (record.stream, record.condition)
            if key in records:
                raise ValueError(f"{path}: two records for {key}")
            records[key] = record
    return records
