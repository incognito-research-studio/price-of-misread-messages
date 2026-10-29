"""Buffer ablation: which misreads cost money when the replenishment safety buffer shrinks.

usage: python scripts/buffer_ablation.py RUN_DIR [--streams DIR] [--buffers 0 0.15 0.3]
                                         [--workers 4] [--out DIR]

Post hoc and exploratory: not part of any frozen protocol. With the tuned buffer (beta = 0.3),
missed capacity and demand changes rarely cost anything, so the finding that the cost of reading
depends on the decision could be an artefact of one planner setting. This script reprices the
same recorded reader output (RUN_DIR/proposals.jsonl) under smaller buffers, with no model call:
for each beta it replays every solver-decided condition through the unchanged front end
(cdd.experiments.replay_experiment) with single-repair attribution, and runs the model-free check
(cdd.benchmark.calibration) on the same streams. The benchmark x* is the planner at that beta on
the oracle state, so each beta is its own reference.

A gate: at the run's own beta the replay must reproduce every recorded cost, or the script stops.
For each beta it reports each reader's mean R_interp, the no-update cost, and by delta type the
share of reader errors whose single repair saves more than 100 (the paper's threshold) and the
share of the summed positive repair effects, plus the model-free costly-miss share per type.
Writes JSON to OUT (default: runs/<run>-buffers/).
"""

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from cdd.benchmark.calibration import calibration_report
from cdd.benchmark.stream import load_streams
from cdd.experiments import AttributionConfig, ExperimentConfig, replay_experiment
from cdd.recorded import read_records

COSTLY = 100.0
TYPES = (
    "supplier_status",
    "quality_hold",
    "quality_release",
    "lead_time_change",
    "capacity_change",
    "demand_signal",
    "price_change",
)


def same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--streams")
    parser.add_argument("--buffers", type=float, nargs="+", default=[0.0, 0.15, 0.3])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run
    base = ExperimentConfig.model_validate_json((run / "config.json").read_text())
    streams_dir = args.streams or base.streams
    records = read_records(run)
    recorded = {
        (r["stream"], r["condition"]): r
        for r in map(json.loads, (run / "results.jsonl").read_text().splitlines())
    }
    own = base.solver.safety_factor
    if own not in args.buffers:
        args.buffers.append(own)
    print(f"== {run.name}: {len(records)} recorded pairs; run's own buffer {own}")

    report: dict[str, Any] = {}
    for beta in sorted(args.buffers):
        config = base.model_copy(
            update={
                "solver": base.solver.model_copy(update={"safety_factor": beta}),
                "attribution": AttributionConfig(single_repair=True, shapley_permutations=0),
            }
        )
        started = time.perf_counter()
        replay = replay_experiment(config, records, streams=streams_dir)
        if beta == own:
            for row in replay.rows:
                want = recorded.get((row["stream"], row["condition"]))
                if want is not None and not same(row["cost"], want["cost"]):
                    sys.exit(
                        f"GATE FAILED at beta {beta}: {row['stream']}/{row['condition']} "
                        f"replay {row['cost']:,.1f} != recorded {want['cost']:,.1f}"
                    )
            print(f"   gate passed: beta {beta} reproduces every recorded cost")
        calibration = calibration_report(
            list(load_streams(streams_dir)), config, workers=args.workers, costly=COSTLY
        )
        rows_by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in replay.rows:
            rows_by[row["condition"]].append(row)
        readers = [c for c in rows_by if c not in ("oracle", "no-update")]
        errors = [
            e for e in replay.errors if e["condition"] in readers and e["delta_c"] is not None
        ]
        by_type: dict[str, list[float]] = collections.defaultdict(list)
        for e in errors:
            by_type[e["delta_type"]].append(float(e["delta_c"]))
        positive = sum(max(0.0, c) for costs in by_type.values() for c in costs) or 1.0
        oracle_cost = statistics.fmean(r["cost"] for r in rows_by["oracle"])
        entry = {
            "oracle_cost": oracle_cost,
            "fill_rate_oracle": statistics.fmean(r["fill_rate"] for r in rows_by["oracle"]),
            "r_interp": {c: statistics.fmean(r["r_interp"] for r in rows_by[c]) for c in rows_by},
            "reader_errors": {
                t: {
                    "errors": len(by_type.get(t, [])),
                    "costly_share": (
                        sum(c > COSTLY for c in by_type[t]) / len(by_type[t])
                        if by_type.get(t)
                        else None
                    ),
                    "share_of_cost": sum(max(0.0, c) for c in by_type.get(t, [])) / positive,
                }
                for t in TYPES
            },
            "model_free": {
                t: calibration["by_type"][t]["costly_share"]
                for t in TYPES
                if t in calibration.get("by_type", {})
            },
            "seconds": time.perf_counter() - started,
        }
        report[str(beta)] = entry

        print(f"\n== beta {beta}  (oracle cost {oracle_cost:,.0f}; {entry['seconds']:,.0f} s)")
        print(
            "   mean R_interp: " + ", ".join(f"{c} {v:,.0f}" for c, v in entry["r_interp"].items())
        )
        print(
            f"   {'delta type':18s} {'reader errors':>13s} {'costly':>7s} "
            f"{'cost share':>10s} {'model-free costly':>18s}"
        )
        for t in TYPES:
            e = entry["reader_errors"][t]
            share = "-" if e["costly_share"] is None else f"{e['costly_share']:.0%}"
            mf = entry["model_free"].get(t)
            print(
                f"   {t:18s} {e['errors']:13d} {share:>7s} {e['share_of_cost']:10.0%} "
                f"{'-' if mf is None else f'{mf:.0%}':>18s}"
            )

    out = args.out or Path(base.output_dir) / f"{run.name}-buffers"
    out.mkdir(parents=True, exist_ok=True)
    (out / "buffer_ablation.json").write_text(json.dumps(report, indent=1))
    print(f"\n   report: {out / 'buffer_ablation.json'}")


if __name__ == "__main__":
    main()
