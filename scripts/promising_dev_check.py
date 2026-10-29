"""Model-free check of the promising parameters on development streams (scope D7, Section 5.1).

usage: python scripts/promising_dev_check.py EXPERIMENT_CONFIG STREAMS_DIR PARAMS [OUT_JSON]

Uses no reader output. For every stream it runs the gold-update reference and reports the D7
criteria (share of arriving units accepted, share of accepted orders on time, share broken), the
fixed-supply gate (replaying the reference supply reproduces the planner cost), and two checks:

- calibration: the promising loss of missing each gold delta alone, by delta type, as
  `cdd calibrate` does for replenishment (a miss is costly above 100). Types the promising
  decision cannot see under the fixed supply should show no loss (scope 5.4);
- repair gate: the regex reader (no model), with single repairs; repairing every error must
  reproduce the reference exactly.
"""

import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cdd.benchmark.stream import Stream, load_streams
from cdd.experiments import AttributionConfig, ExperimentConfig, StreamContext
from cdd.interpreters import RegexInterpreter, ReplayInterpreter
from cdd.pipeline import PlannedPolicy
from cdd.promising import PromisingParams
from cdd.promising.run import PromisingContext, score_promising

COSTLY = 100.0


def check(job: tuple[Stream, ExperimentConfig, PromisingParams]) -> dict:
    stream, config, params = job
    pc = PromisingContext(StreamContext(stream, config), params)
    ref = pc.reference()
    m = ref.metrics
    replay_cost = pc.context.simulator.run(PlannedPolicy(pc.supply.plans)).total_cost
    misses = []
    for delta in stream.gold:
        kept = [d for d in stream.gold if d.id != delta.id]
        value = pc.run(ReplayInterpreter(kept, name="miss"))[0].value
        misses.append({"type": delta.type.value, "loss": ref.value - value})
    row, _ = score_promising(
        pc,
        RegexInterpreter(stream.world),
        condition="regex",
        attribution=AttributionConfig(single_repair=True),
    )
    return {
        "stream": stream.id,
        "value": ref.value,
        "accepted_share": m["accepted_units"] / m["units"] if m["units"] else None,
        "on_time_share": m["on_time_orders"] / m["accepted_orders"]
        if m["accepted_orders"]
        else None,
        "broken_share": m["broken_orders"] / m["accepted_orders"] if m["accepted_orders"] else None,
        "fallbacks": m["fallbacks"],
        "at_risk_units": m["at_risk_units"],
        "supply_gap": replay_cost - pc.supply.cost,
        "misses": misses,
        "regex_loss": row["loss"],
        "regex_errors": row["n_errors"],
        "regex_repaired_gap": row["repaired_gap"],
    }


def main() -> None:
    config = ExperimentConfig.from_yaml(sys.argv[1])
    streams = load_streams(sys.argv[2])
    params = PromisingParams.from_yaml(sys.argv[3])
    with ProcessPoolExecutor(os.cpu_count() or 1) as pool:
        results = list(pool.map(check, [(s, config, params) for s in streams]))

    def share(key: str) -> str:
        values = [r[key] for r in results if r[key] is not None]
        return f"mean {statistics.fmean(values):.1%}, min {min(values):.1%}, max {max(values):.1%}"

    print(f"== gold-update reference on {len(results)} streams ({sys.argv[2]})")
    print(f"   accepted units:  {share('accepted_share')}   (criterion: >= 80%)")
    print(f"   on time:         {share('on_time_share')}   (criterion: >= 90%)")
    print(f"   broken:          {share('broken_share')}   (criterion: <= 10%)")
    print(
        f"   solver fallbacks: {sum(r['fallbacks'] for r in results)}; "
        f"units at risk: {sum(r['at_risk_units'] for r in results):,.1f}"
    )
    print(
        f"   supply gate: max |replay - planner cost| = "
        f"{max(abs(r['supply_gap']) for r in results):.2g}"
    )

    print(f"\n== calibration: loss of missing each gold delta (costly above {COSTLY:g})")
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for miss in r["misses"]:
            by_type[miss["type"]].append(miss["loss"])
    total = sum(max(0.0, x) for xs in by_type.values() for x in xs)
    for kind, losses in sorted(by_type.items()):
        costly = sum(x > COSTLY for x in losses)
        positive = sum(max(0.0, x) for x in losses)
        print(
            f"   {kind:18s} {len(losses):4d} misses; costly {costly / len(losses):5.0%}; "
            f"any loss {sum(x > 1e-6 for x in losses) / len(losses):5.0%}; "
            f"negative {sum(x < -1e-6 for x in losses):3d}; "
            f"share of loss {positive / total if total else 0:5.0%}"
        )

    gaps = [
        r
        for r in results
        if r["regex_repaired_gap"] not in (None, 0.0) and abs(r["regex_repaired_gap"]) > 1e-6
    ]
    print("\n== regex repair gate")
    print(
        f"   regex errors {sum(r['regex_errors'] for r in results)}, mean loss "
        f"{statistics.fmean(r['regex_loss'] for r in results):,.1f}; gate failures {len(gaps)}"
    )
    criteria = (
        min(r["accepted_share"] for r in results if r["accepted_share"] is not None) >= 0.0
        and statistics.fmean(r["accepted_share"] for r in results) >= 0.8
        and statistics.fmean(r["on_time_share"] for r in results) >= 0.9
        and statistics.fmean(r["broken_share"] for r in results) <= 0.1
    )
    print(f"\n== D7 criteria on stream means: {'met' if criteria else 'NOT met'}")
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
