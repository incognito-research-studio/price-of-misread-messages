"""Snapping ablation: free JSON with its choices snapped to valid options after generation.

usage: python scripts/snap_ablation.py RUN_DIR [--streams DIR] [--free qwen-json]
                                       [--constrained qwen-json-constrained] [--out DIR]

Post hoc and exploratory: not part of any frozen protocol. It asks how much of the gap between
free and constrained JSON comes from the choices the schema restricts. Constrained decoding
restricts the type to the schema, the entity to identifiers of the kind the type needs, the
evidence to one of the message's sentences and ``supersedes`` to deltas in force. This script
replays the free reader's own replies from RUN_DIR's request log, applies the same restrictions
after generation, and scores the result through the unchanged front end and planner, with no
model call.

Variants, each a separate replay:
- ``none``: the replies as recorded. A gate: it must reproduce the recorded free-JSON cost on
  every stream, or the replay is not faithful and the script stops.
- ``evidence``: evidence that is not a substring of the message is snapped to the most similar
  sentence (difflib ratio); evidence already in the message is kept.
- ``evidence-sentence``: every evidence string becomes a full sentence (the sentence containing
  it, or the most similar one), as the schema's sentence enum forces.
- ``entity``: an invalid type is snapped to the most similar delta type, and an entity that is not
  an identifier of the required kind to the identifier whose id or name is most similar.
- ``supersedes``: ids that are not deltas in force are dropped.
- ``all``: ``evidence-sentence``, ``entity`` and ``supersedes`` together.

Replies are held fixed per message (open loop): a reader that saw a snapped state might have
answered later messages differently, so, as with repairs, this prices correcting the recorded
output, not a different reader. Attribution is off. For each variant the script reports mean and
median R_interp, F1, parse failures and the paired difference against the recorded constrained
reader (95% percentile bootstrap over streams, 10,000 resamples, seed 0, as in
replication_analysis.py), and the share of the free-minus-constrained gap that snapping closes.
Writes per-stream rows to OUT (default: runs/<run>-snap/).
"""

import argparse
import collections
import csv
import difflib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from cdd.benchmark.stream import load_streams
from cdd.experiments import (
    AttributionConfig,
    ExperimentConfig,
    StreamContext,
    _Clients,
    run_condition,
)
from cdd.interpreters.base import Interpretation
from cdd.interpreters.llm import _proposal, parse_deltas
from cdd.schema import DELTA_SPECS, DeltaType, Message
from cdd.state import StateView
from cdd.text import sentence_spans

VARIANTS = ("none", "evidence", "evidence-sentence", "entity", "supersedes", "all")
RESAMPLES = 10_000
MESSAGE_LINE = re.compile(r"^Message (\S+) \(week \d+", re.MULTILINE)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best(target: str, options: list[str]) -> str:
    """The most similar option; ties go to the first."""
    return max(options, key=lambda option: similarity(target, option))


class Snapper:
    """Applies the schema's restrictions to one parsed reply item, counting each change."""

    def __init__(self, evidence: str, entity: bool, supersedes: bool) -> None:
        self.evidence, self.entity, self.supersedes = evidence, entity, supersedes
        self.counts: collections.Counter[str] = collections.Counter()

    def __call__(self, item: Any, message: Message, view: StateView) -> Any:
        if not isinstance(item, dict):
            return item
        item = dict(item)
        if self.entity:
            self._entity(item, view)
        if self.evidence != "off":
            self._evidence(item, message)
        if self.supersedes and item.get("supersedes"):
            in_force = {delta.id for delta in view.active()}
            kept = [ref for ref in item["supersedes"] if ref in in_force]
            if len(kept) != len(item["supersedes"]):
                self.counts["supersedes dropped"] += len(item["supersedes"]) - len(kept)
                item["supersedes"] = kept
        return item

    def _entity(self, item: dict[str, Any], view: StateView) -> None:
        types = [kind.value for kind in DeltaType]
        if item.get("type") not in types:
            item["type"] = best(str(item.get("type", "")), types)
            self.counts["type snapped"] += 1
        entities = list(view.world.entities(DELTA_SPECS[DeltaType(item["type"])].entity_kind))
        ids = [e.id for e in entities]
        given = str(item.get("entity", ""))
        if given in ids:
            return
        exact = {e.id.lower(): e.id for e in entities} | {e.name.lower(): e.id for e in entities}
        if given.lower() in exact:
            item["entity"] = exact[given.lower()]
        else:
            item["entity"] = max(
                entities, key=lambda e: max(similarity(given, e.id), similarity(given, e.name))
            ).id
        self.counts["entity snapped"] += 1

    def _evidence(self, item: dict[str, Any], message: Message) -> None:
        evidence = str(item.get("evidence", "")).strip()
        sentences = [message.text[s:e] for s, e in sentence_spans(message.text)]
        if not evidence or not sentences:
            return
        if evidence in message.text:
            if self.evidence == "invalid":
                return
            containing = [s for s in sentences if evidence in s]
            snapped = containing[0] if containing else best(evidence, sentences)
        else:
            snapped = best(evidence, sentences)
        if snapped != evidence:
            item["evidence"] = snapped
            self.counts["evidence snapped"] += 1


class SnappedReplay:
    """Replays recorded raw replies by message, snapped, through the free reader's own parsing."""

    def __init__(self, replies: dict[str, str], snapper: Snapper, name: str) -> None:
        self.replies, self.snapper, self.name = replies, snapper, name

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        raw = self.replies.get(message.id)
        if raw is None:
            return Interpretation(failures=("no recorded reply",))
        try:
            items = parse_deltas(raw)
        except ValueError as exc:
            return Interpretation(failures=(f"malformed output: {exc}",))
        proposals, failures = [], []
        for item in items:
            try:
                proposals.append(_proposal(self.snapper(item, message, view), message))
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(f"unusable delta {item!r}: {exc}")
        return Interpretation(tuple(proposals), tuple(failures))


def snapper_for(variant: str) -> Snapper:
    return Snapper(
        evidence={"evidence": "invalid", "evidence-sentence": "sentence", "all": "sentence"}.get(
            variant, "off"
        ),
        entity=variant in ("entity", "all"),
        supersedes=variant in ("supersedes", "all"),
    )


def recorded_replies(run: Path, condition: str) -> dict[str, dict[str, str]]:
    """Raw replies of ``condition`` by stream and message id, from every request log in RUN."""
    replies: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for log in sorted(run.glob("requests-*.jsonl")):
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            stream, _, tag_condition = str(entry.get("tag") or "").partition("/")
            if tag_condition != condition:
                continue
            user = next(m["content"] for m in entry["request"]["messages"] if m["role"] == "user")
            found = MESSAGE_LINE.findall(user)
            if not found:
                sys.exit(f"{log.name}: no message id in a {condition} request on {stream}")
            message_id = found[-1]
            if message_id in replies[stream]:
                sys.exit(f"{log.name}: two {condition} replies for {stream}/{message_id}")
            replies[stream][message_id] = entry["reply"]
    return replies


def interval(values: list[float], seed: int = 0) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(RESAMPLES, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--streams")
    parser.add_argument("--free", default="qwen-json")
    parser.add_argument("--constrained", default="qwen-json-constrained")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run
    config = ExperimentConfig.model_validate_json((run / "config.json").read_text())
    config = config.model_copy(
        update={"attribution": AttributionConfig(single_repair=False, shapley_permutations=0)}
    )
    names = [c.name for c in config.conditions]
    condition_index = names.index(args.free)
    condition = config.conditions[condition_index]
    recorded = {
        (r["stream"], r["condition"]): r
        for r in map(json.loads, (run / "results.jsonl").read_text().splitlines())
        if r["condition"] in (args.free, args.constrained)
    }
    replies = recorded_replies(run, args.free)
    streams = [
        (i, s)
        for i, s in enumerate(load_streams(args.streams or config.streams))
        if (s.id, args.free) in recorded and (s.id, args.constrained) in recorded
    ]
    print(f"== {run.name}: {len(streams)} streams with both {args.free} and {args.constrained}")
    print(f"   replies recorded: {sum(len(v) for v in replies.values())} (open-loop replay)")

    rows: list[dict[str, Any]] = []
    totals: dict[str, collections.Counter[str]] = {}
    clients = _Clients()
    started = time.perf_counter()
    for index, stream in streams:
        context = StreamContext(stream, config)
        for variant in VARIANTS:
            snapper = snapper_for(variant)
            interpreter = SnappedReplay(replies.get(stream.id, {}), snapper, args.free)
            rng = np.random.default_rng([config.seed, index, condition_index])
            row, _, _ = run_condition(context, condition, clients, rng, interpreter=interpreter)
            totals.setdefault(variant, collections.Counter()).update(snapper.counts)
            rows.append(
                {
                    "stream": stream.id,
                    "variant": variant,
                    "cost": row["cost"],
                    "r_interp": row["r_interp"],
                    "f1": row["f1"],
                    "parse_failures": row["parse_failures"],
                    "n_errors": row["n_errors"],
                    "constrained_cost": recorded[(stream.id, args.constrained)]["cost"],
                    **{f"snap_{k.replace(' ', '_')}": v for k, v in snapper.counts.items()},
                }
            )
            if variant == "none":
                want = recorded[(stream.id, args.free)]["cost"]
                if not same(row["cost"], want):
                    sys.exit(
                        f"GATE FAILED: {stream.id} unsnapped replay cost {row['cost']:,.1f} "
                        f"!= recorded {args.free} {want:,.1f}; the replay is not faithful"
                    )
    print(f"   gate passed: unsnapped replay reproduces {args.free} on every stream")
    print(f"   {len(rows)} replays in {time.perf_counter() - started:,.0f} s")

    by_variant: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    gap = statistics.fmean(r["cost"] - r["constrained_cost"] for r in by_variant["none"])
    print(f"\n   recorded gap, {args.free} minus {args.constrained}: mean {gap:,.1f} per stream")
    print(
        f"\n   {'variant':18s} {'R_interp':>10s} {'median':>9s} {'F1':>6s} {'fails':>6s}  "
        f"{'minus constrained [95% interval]':36s} {'gap closed':>10s}  snaps"
    )
    for variant in VARIANTS:
        vrows = by_variant[variant]
        diffs = [r["cost"] - r["constrained_cost"] for r in vrows]
        low, high = interval(diffs)
        closed = 1 - statistics.fmean(diffs) / gap if gap else float("nan")
        snaps = ", ".join(f"{k} {v}" for k, v in sorted(totals[variant].items())) or "-"
        print(
            f"   {variant:18s} {statistics.fmean(r['r_interp'] for r in vrows):10,.1f} "
            f"{statistics.median(r['r_interp'] for r in vrows):9,.1f} "
            f"{statistics.fmean(r['f1'] for r in vrows):6.3f} "
            f"{sum(r['parse_failures'] for r in vrows):6d}  "
            f"{statistics.fmean(diffs):10,.1f} [{low:10,.1f}, {high:10,.1f}]  "
            f"{closed:9.0%}  {snaps}"
        )

    out = args.out or Path(config.output_dir) / f"{run.name}-snap"
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}, key=lambda k: (k not in rows[0], k))
    with (out / "snap_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n   per-stream rows: {out / 'snap_ablation.csv'}")


if __name__ == "__main__":
    main()
