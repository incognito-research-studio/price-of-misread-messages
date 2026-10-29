"""Model-free calibration check: what would each missed delta cost?

For every gold delta of a stream, the check removes that delta alone (a simulated miss), runs
the oracle pipeline with the configured planner, and records the cost increase
``ΔC = C(miss) - C(oracle)``. A benchmark where most misses of a type cost nothing cannot show
whether reading that type matters, whatever the readers do; one where a single type carries
most of the cost makes "cost concentrates in a few types" a property of the settings. Run it
before any model run, on template streams (the oracle does not read text).

The oracle run itself must be clean: no SOP rejection and no conflict drop, or gold deltas are
being blocked and the reference is wrong.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from cdd.benchmark.stream import Stream


def miss_effects(stream: Stream, config: Any) -> dict[str, Any]:
    """ΔC of missing each gold delta of ``stream``, and the oracle run's SOP and conflict
    counts. ``config`` is an :class:`~cdd.experiments.ExperimentConfig`."""
    from cdd.experiments import StreamContext, _solver
    from cdd.interpreters import OracleInterpreter, ReplayInterpreter

    context = StreamContext(stream, config)
    planner = _solver(config)
    policy = context.compiling(OracleInterpreter(stream.gold), planner)
    base = context.simulator.run(policy).total_cost
    sop = [r for r in policy.audit.records if r["kind"] == "sop"]
    misses = []
    for delta in stream.gold:
        kept = [d for d in stream.gold if d.id != delta.id]
        replay = context.compiling(ReplayInterpreter(kept, name="miss"), planner)
        cost = context.simulator.run(replay).total_cost
        misses.append({"stream": stream.id, "delta": delta.id, "type": delta.type.value,
                       "delta_c": cost - base})  # fmt: skip
    return {
        "stream": stream.id,
        "oracle_cost": base,
        "hindsight_cost": context.hindsight_cost(),
        "rejected": sum(r["verdict"] == "reject" for r in sop),
        "unapproved": sum(not r["admitted"] for r in sop),
        "conflicts_dropped": policy.conflicts_dropped,
        "misses": misses,
    }


def calibration_report(
    streams: Sequence[Stream], config: Any, *, workers: int = 1, costly: float = 100.0
) -> dict[str, Any]:
    """Miss effects for every stream, summarised by delta type. A miss counts as costly when
    ΔC exceeds ``costly``."""
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(miss_effects, streams, [config] * len(streams)))
    else:
        results = [miss_effects(stream, config) for stream in streams]
    by_type: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for miss in result["misses"]:
            by_type[miss["type"]].append(miss["delta_c"])
    positive = sum(max(0.0, c) for costs in by_type.values() for c in costs)
    summary = {
        kind: {
            "misses": len(costs),
            "costly_share": sum(c > costly for c in costs) / len(costs),
            "free_share": sum(abs(c) <= 1e-6 for c in costs) / len(costs),
            "mean_delta_c": sum(costs) / len(costs),
            "share_of_cost": (sum(max(0.0, c) for c in costs) / positive) if positive else 0.0,
        }
        for kind, costs in sorted(by_type.items())
    }
    return {
        "streams": len(results),
        "oracle_rejected": sum(r["rejected"] for r in results),
        "oracle_unapproved": sum(r["unapproved"] for r in results),
        "oracle_conflicts_dropped": sum(r["conflicts_dropped"] for r in results),
        "mean_oracle_cost": sum(r["oracle_cost"] for r in results) / len(results),
        "mean_value_of_perfect_information": sum(
            r["oracle_cost"] - r["hindsight_cost"] for r in results
        )
        / len(results),
        "by_type": summary,
        "results": results,
    }
