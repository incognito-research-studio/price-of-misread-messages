"""Scan a run folder for crashes, silent failures and suspicious model output.

usage: python scripts/scan_run.py RUN_DIR STREAMS_DIR
"""

import collections
import csv
import json
import re
import sys
from pathlib import Path

from cdd.benchmark.stream import load_streams
from cdd.interpreters.llm import parse_deltas

run, streams_dir = Path(sys.argv[1]), sys.argv[2]
streams = {s.id: s for s in load_streams(streams_dir)}
texts = {m.id: m.text for s in streams.values() for m in s.messages}
gold = {m.id: s.gold_for(m.id) for s in streams.values() for m in s.messages}
REVISION = re.compile(r"\b(revis|correct|amend|updat|supersed|replac|earlier|previous)", re.I)
problems: list[str] = []

print("== run")
failed = json.loads((run / "failed.json").read_text())
skipped = json.loads((run / "skipped.json").read_text())
print(f"failed pairs: {len(failed)}  skipped conditions: {list(skipped)}")
problems += [f"failed: {f}" for f in failed]
rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
for row in rows:
    if row["condition"] == "oracle" and abs(row["r_interp"]) > 1e-6:
        problems.append(f"oracle r_interp {row['r_interp']} on {row['stream']}")
    if row["condition"] == "oracle" and (row.get("rejected") or row.get("conflicts_dropped")):
        problems.append(
            f"oracle blocked gold deltas on {row['stream']}: "
            f"rejected={row.get('rejected')} conflicts={row.get('conflicts_dropped')}"
        )
    if row.get("interpreter") == "oracle" and abs(row.get("r_interp") or 0) > 1e-6:
        problems.append(
            f"{row['condition']}: oracle-reader r_interp {row['r_interp']:.1f} on {row['stream']}"
        )
    if row.get("r_dec", 0) < -1e-6:
        problems.append(f"negative r_dec {row['r_dec']:.1f} on {row['stream']}/{row['condition']}")
by_condition = collections.defaultdict(list)
for row in rows:
    by_condition[row["condition"]].append(row)
for condition, group in by_condition.items():
    sums = {
        k: sum(r.get(k) or 0 for r in group)
        for k in (
            "parse_failures",
            "model_requests",
            "model_cache_hits",
            "model_retries",
            "model_truncations",
            "model_capped_replies",
            "rejected",
            "rejected_correct",
            "conflicts_dropped",
            "invalid_order_lines",
            "seconds",
        )
    }
    print(
        f"  {condition:24s} streams={len(group)} "
        + " ".join(f"{k}={v:.0f}" for k, v in sums.items())
    )
    for key in ("model_retries", "model_truncations", "model_capped_replies"):
        if sums[key]:
            problems.append(f"{condition}: {key}={sums[key]:.0f}")

print("== request logs")
for log in sorted(run.glob("requests-*.jsonl")):
    records = [json.loads(line) for line in log.read_text().splitlines()]
    stats: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    slow = max(records, key=lambda r: r["seconds"])
    longest_prompt = max(len(json.dumps(r["request"]["messages"])) for r in records)
    longest_reply = max(len(r["reply"]) for r in records)
    for r in records:
        condition = (r.get("tag") or "?/?").split("/", 1)[1]
        c = stats[condition]
        c["requests"] += 1
        c["cached"] += bool(r.get("cached"))
        if r.get("done_reason") == "length":
            c["capped"] += 1
        system, user = (
            r["request"]["messages"][0]["content"],
            r["request"]["messages"][1]["content"],
        )
        if system.startswith("You read one operational message"):
            match = re.search(r"Message (\S+) \(", user)
            message_id = match[1] if match else "?"
            text = texts.get(message_id, "")
            in_force = set(re.findall(r"^- (\S+): ", user.split("Deltas in force:")[1], re.M))
            entities = set(re.findall(r"(\S+) = ", user.split("Deltas in force:")[0]))
            try:
                items = parse_deltas(r["reply"])
            except ValueError:
                c["malformed"] += 1
                continue
            c[f"{len(items)} deltas"] += 1
            for item in items:
                if not isinstance(item, dict):
                    c["non-object item"] += 1
                    continue
                evidence = str(item.get("evidence", "")).strip()
                if not evidence or evidence not in text:
                    c["evidence not in message"] += 1
                if item.get("entity") not in entities:
                    c["unknown entity"] += 1
                for ref in item.get("supersedes") or []:
                    c["supersedes filled"] += 1
                    if ref not in in_force:
                        c["supersedes unknown id"] += 1
                    if not REVISION.search(text):
                        c["supersedes without revision cue"] += 1
            if not gold.get(message_id) and items:
                c["deltas on a no-change message"] += 1
        elif system.startswith("You plan replenishment"):
            try:
                orders = json.loads(r["reply"][r["reply"].find("{") : r["reply"].rfind("}") + 1])
                c["orders ok"] += isinstance(orders.get("orders"), list)
            except (ValueError, AttributeError):
                c["orders malformed"] += 1
        elif system.startswith("You answer multiple-choice"):
            try:
                answers = json.loads(r["reply"])
                c["typed ok"] += isinstance(answers, dict)
            except ValueError:
                c["typed malformed"] += 1
    print(
        f"{log.name}: {len(records)} requests; slowest {slow['seconds']:.1f} s ({slow.get('tag')}); "
        f"longest prompt {longest_prompt} chars; longest reply {longest_reply} chars"
    )
    for condition, c in stats.items():
        print(f"  {condition:24s} {dict(c)}")
        for key in (
            "malformed",
            "typed malformed",
            "orders malformed",
            "capped",
            "unknown entity",
            "supersedes unknown id",
        ):
            if c[key]:
                problems.append(f"{log.name} {condition}: {key}={c[key]}")

print("== error kinds")
errors = list(csv.DictReader((run / "errors.csv").open()))
kinds = collections.defaultdict(collections.Counter)
for e in errors:
    kinds[e["condition"]][e["kind"]] += 1
for condition, c in kinds.items():
    print(f"  {condition:24s} {dict(c)}")

print("== problems")
print("\n".join(problems) if problems else "none")
