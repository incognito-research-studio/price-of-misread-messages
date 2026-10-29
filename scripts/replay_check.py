"""Recheck a recorded run from exported reader output, with no model call (cloud side of the gate).

usage: python scripts/replay_check.py RECORDED_RUN EXPORT_RUN STREAMS_DIR

Replays every solver-decided condition of RECORDED_RUN's config from EXPORT_RUN/proposals.jsonl
(cdd.experiments.replay_experiment) and compares cost, R_interp, F1, and each error's identity,
single-repair effect and Shapley value with the recorded run. Pairs with no exported record
(e.g. a reader that was not re-run) are listed, not checked. Solver results can differ between
machines when a solve hits its time limit or the HiGHS version differs, so a mismatch is a
finding to inspect, not proof of a replay defect. Ends with a list of problems, or "none".
"""

import collections
import json
import sys
import time
from pathlib import Path

from cdd.experiments import ExperimentConfig, replay_experiment
from cdd.recorded import read_records

recorded, export, streams = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
FIELDS = ("kind", "delta_type", "predicted_id", "gold_id")
problems: list[str] = []


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


config = ExperimentConfig.model_validate_json((recorded / "config.json").read_text())
started = time.perf_counter()
replay = replay_experiment(config, read_records(export), streams=streams)
print(
    f"== replay of {recorded.name} from {export.name}: {len(replay.rows)} pairs in "
    f"{time.perf_counter() - started:,.0f} s"
)
if replay.missing:
    by_condition = collections.Counter(condition for _, condition in replay.missing)
    print(f"not exported, not checked: {dict(by_condition)}")

old = {(r["stream"], r["condition"]): r for r in jsonl(recorded / "results.jsonl")}
for row in replay.rows:
    pair = (row["stream"], row["condition"])
    if pair not in old:
        problems.append(f"{pair}: not in the recorded run")
        continue
    for field in ("cost", "r_interp", "f1"):
        if not same(row[field], old[pair][field]):
            problems.append(f"{pair}: {field} {row[field]} != recorded {old[pair][field]}")

old_errors: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
for e in jsonl(recorded / "errors.jsonl"):
    old_errors[(e["stream"], e["condition"])].append(e)
new_errors: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
for e in replay.errors:
    new_errors[(e["stream"], e["condition"])].append(e)
compared = attributed = 0
for pair in sorted({(r["stream"], r["condition"]) for r in replay.rows}):
    a, b = new_errors.get(pair, []), old_errors.get(pair, [])
    if [tuple(e[f] for f in FIELDS) for e in a] != [tuple(e[f] for f in FIELDS) for e in b]:
        problems.append(f"{pair}: error identities differ ({len(a)} against {len(b)} recorded)")
        continue
    for x, y in zip(a, b, strict=True):
        compared += 1
        for field in ("delta_c", "shapley"):
            if y.get(field) is None:
                continue
            attributed += field == "shapley"
            if not same(x.get(field), y.get(field)):
                problems.append(
                    f"{pair} {y['error_id']}: {field} {x.get(field)} != recorded {y[field]}"
                )
print(f"errors compared: {compared}  with recorded Shapley values: {attributed}")

print("\n== problems")
print("\n".join(problems) if problems else "none")
sys.exit(1 if problems else 0)
