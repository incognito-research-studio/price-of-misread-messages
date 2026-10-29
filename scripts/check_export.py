"""Export gate: does a re-run that records reader output reproduce the recorded run?

usage: python scripts/check_export.py EXPORT_RUN RECORDED_RUN

EXPORT_RUN is a run of the recorded run's conditions with the reply cache and attribution off,
which writes proposals.jsonl. The gate passes when the export made no model request, recorded
every compiled pair, and reproduced every pair's cost, F1 and error identity (kind, delta type,
predicted id, gold id, in order). Single repairs and Shapley values are rechecked separately
from the exported deltas (scripts/replay_check.py). Ends with a list of problems, or "none".
"""

import collections
import json
import sys
from pathlib import Path

export, recorded = Path(sys.argv[1]), Path(sys.argv[2])
IDENTITY = ("kind", "delta_type", "predicted_id", "gold_id")
problems: list[str] = []


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_pair(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["stream"], row["condition"]): row for row in rows}


def error_ids(path: Path) -> dict[tuple[str, str], list[tuple]]:
    out: dict[tuple[str, str], list[tuple]] = collections.defaultdict(list)
    for e in jsonl(path):
        out[(e["stream"], e["condition"])].append(tuple(e[f] for f in IDENTITY))
    return out


def same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


new, old = by_pair(jsonl(export / "results.jsonl")), by_pair(jsonl(recorded / "results.jsonl"))
new_errors, old_errors = error_ids(export / "errors.jsonl"), error_ids(recorded / "errors.jsonl")
records = (
    {(r["stream"], r["condition"]) for r in jsonl(export / "proposals.jsonl")}
    if (export / "proposals.jsonl").exists()
    else set()
)

print(f"== export {export.name} against {recorded.name}")
failed = json.loads((export / "failed.json").read_text())
problems += [f"failed pair: {f['stream']} {f['condition']}: {f['error']}" for f in failed]
skipped = json.loads((export / "skipped.json").read_text())
problems += [f"skipped condition: {name}: {why}" for name, why in skipped.items()]

requests = sum(int(row.get("model_requests") or 0) for row in new.values())
hits = sum(int(row.get("model_cache_hits") or 0) for row in new.values())
print(f"pairs: {len(new)}  model requests: {requests}  cache hits: {hits}")
if requests:
    by_condition = collections.Counter()
    for row in new.values():
        by_condition[row["condition"]] += int(row.get("model_requests") or 0)
    problems.append(f"{requests} model requests reached a server: {dict(by_condition)}")

conditions = {condition for _, condition in new}
for pair in sorted(new):
    if pair not in old:
        problems.append(f"{pair}: not in the recorded run")
        continue
    if pair not in records:
        problems.append(f"{pair}: no record in proposals.jsonl")
    a, b = new[pair], old[pair]
    for field in ("cost", "f1"):
        if not same(a.get(field), b.get(field)):
            problems.append(f"{pair}: {field} {a.get(field)} != recorded {b.get(field)}")
    if new_errors.get(pair, []) != old_errors.get(pair, []):
        problems.append(
            f"{pair}: errors differ ({len(new_errors.get(pair, []))} against "
            f"{len(old_errors.get(pair, []))} recorded)"
        )
absent = sorted(pair for pair in old if pair[1] in conditions and pair not in new)
problems += [f"{pair}: recorded but not exported" for pair in absent]
others = sorted({condition for _, condition in old} - conditions)
if others:
    print(f"recorded conditions not in this export (not checked): {others}")

print("\n== problems")
print("\n".join(problems) if problems else "none")
sys.exit(1 if problems else 0)
