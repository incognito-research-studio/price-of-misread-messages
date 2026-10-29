"""WP2 analysis: reading, parsing, applying and decision cost, reported separately.

usage: python scripts/challenge_analysis.py CHALLENGE_RUN [PARA_RUN TEMPLATE_RUN]

For each reader on the hand-written challenge texts, against the gold deltas:

1. identification: gold deltas whose type and entity the reader got right (an error of primary
   kind missed, wrong_type or mis_scoped counts against it);
2. parser completion: of those, the ones whose value and window are also right, plus the parse
   failures by slot from proposals.jsonl (a failed parse shows as a missed delta in 1);
3. applied: exact reads the SOP layer admitted (exact matches minus rejected_correct);
4. decision cost: R_interp per stream.

Spurious deltas (a change read where none was written) are counted separately. The gold-span
reader isolates the parser: its losses are the deterministic layer's alone. With PARA_RUN and
TEMPLATE_RUN, the same numbers are printed for those runs on the same four streams.
"""

import collections
import json
import statistics
import sys
from pathlib import Path

READ_ERRORS = {"missed", "wrong_type", "mis_scoped"}
VALUE_ERRORS = {"wrong_value", "mis_timed"}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(run: Path, streams: set[str] | None = None) -> set[str]:
    rows = [r for r in jsonl(run / "results.jsonl") if streams is None or r["stream"] in streams]
    errors = [e for e in jsonl(run / "errors.jsonl") if streams is None or e["stream"] in streams]
    records = [
        r for r in jsonl(run / "proposals.jsonl") if streams is None or r["stream"] in streams
    ]
    shown = {r["stream"] for r in rows}
    print(f"== {run.name}: {len(shown)} streams")
    by_condition: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    header = (
        f"   {'reader':24s} {'gold':>5s} {'identified':>11s} {'complete':>9s} "
        f"{'applied':>8s} {'spurious':>9s}  parse failures by slot      R_interp mean / median"
    )
    print(header)
    for condition, mine in by_condition.items():
        if condition == "oracle":
            continue
        errs = [e for e in errors if e["condition"] == condition]
        gold = sum(r["n_gold"] for r in mine)
        read = sum(e["gold_id"] is not None and e["kind"] in READ_ERRORS for e in errs)
        wrong_value = sum(e["gold_id"] is not None and e["kind"] in VALUE_ERRORS for e in errs)
        spurious = sum(e["kind"] == "spurious" for e in errs)
        identified = gold - read
        complete = identified - wrong_value
        exact = gold - sum(e["gold_id"] is not None for e in errs)
        applied = exact - sum(int(r.get("rejected_correct") or 0) for r in mine)
        slots = collections.Counter(
            slot
            for rec in records
            if rec["condition"] == condition
            for _, slot, _ in rec["failures"]
        )
        regret = [r["r_interp"] for r in mine]
        print(
            f"   {condition:24s} {gold:5d} {identified:5d} {identified / gold:5.0%} "
            f"{complete:4d} {complete / max(identified, 1):4.0%} {applied:8d} {spurious:9d}  "
            f"{dict(slots) or '-'!s:27s} {statistics.fmean(regret):10,.0f} / "
            f"{statistics.median(regret):,.0f}"
        )
    return shown


def main() -> None:
    challenge = Path(sys.argv[1])
    streams = report(challenge)
    for other in sys.argv[2:]:
        print()
        report(Path(other), streams)


if __name__ == "__main__":
    main()
