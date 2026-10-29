"""Base-stock baseline for RQ1: a standard order-up-to policy deciding from the oracle state.

usage: python scripts/basestock_baseline.py DECIDERS_RUN [--dev data/v2/dev]
                                            [--grid 0 0.1 0.2 0.3 0.4 0.5] [--out DIR]

Post hoc: added after the pilot's model-decider results were seen. It anchors decision regret to
a textbook heuristic, so the model deciders can be judged as worse than trivial or merely not
optimal. The policy is periodic review, order-up-to, one review per week. For each SKU:

- the line it orders from first is the cheapest plant line with capacity in the oracle state
  (zero while the supplier is suspended, halved while restricted, as in the planner);
- the target is (1 + s) times the forecast over that line's lead time plus the review week;
- the inventory position is stock on hand minus backlog, plus open orders, plus lot stock that
  the state shows released and that has arrived or arrives within the target window;
- it orders the shortfall from the cheapest lines first, within each plant's capacity this week.

It reads no messages: like the "oracle state" model deciders, it decides from the true state, so
its regret against the planner is decision regret. Its safety factor s is tuned on the
development split over the planner's grid (mean cost, oracle state), as the planner's was, then
fixed. It runs on DECIDERS_RUN's streams with that run's config, priced in the same simulator;
the planner reference must reproduce the recorded oracle cost on every stream, or the script
stops. Reports the per-stream cost ratio to the planner, decision regret with a 95% percentile
bootstrap interval (10,000 resamples, seed 0) and fill rate, beside the recorded model deciders.
"""

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cdd.benchmark.stream import load_streams
from cdd.experiments import ExperimentConfig, StreamContext
from cdd.interpreters.replay import OracleInterpreter
from cdd.orders import InventoryPosition, OrderLine, Plan, capacity_available, lead_periods
from cdd.schema import Attribute, HoldStatus
from cdd.state import StateView

RESAMPLES = 10_000
EPS = 1e-9


class BaseStockDecider:
    def __init__(self, safety: float) -> None:
        self.safety = safety
        self.name = f"base-stock-{safety}"

    def plan(self, view: StateView, position: InventoryPosition) -> Plan:
        world, t = view.world, position.period
        notices = position.status_notices
        remaining = {p.id: capacity_available(view, p.id, t, notices) for p in world.plants}
        orders: list[OrderLine] = []
        for sku in world.skus:
            lines = []
            for plant_id, sku_id in world.lines():
                offer = world.offer_for(plant_id, sku_id)
                if sku_id != sku.id or offer is None or remaining[plant_id] <= EPS:
                    continue
                lines.append((view.number(offer.id, Attribute.PRICE, t), plant_id))
            if not lines:
                continue
            lines.sort()
            lead = lead_periods(view, lines[0][1], t)
            window = [p for p in range(t, t + lead + 1) if p <= world.end_period]
            target = (1 + self.safety) * sum(
                view.number(sku.id, Attribute.DEMAND, p) for p in window
            )
            on_order = sum(o.quantity for o in position.outstanding if o.sku == sku.id)
            lots = sum(
                position.lot_remaining.get(lot.id, 0.0)
                for lot in world.lots
                if lot.sku == sku.id
                and (lot.arrival is None or lot.arrival <= window[-1] if window else False)
                and view.label(lot.id, Attribute.HOLD_STATUS, t) == HoldStatus.RELEASED
            )
            level = (
                position.on_hand.get(sku.id, 0.0)
                - position.backlog.get(sku.id, 0.0)
                + on_order
                + lots
            )
            need = target - level
            for _, plant_id in lines:
                if need <= EPS:
                    break
                quantity = min(need, remaining[plant_id])
                if quantity > EPS:
                    orders.append(OrderLine(plant_id, sku.id, round(quantity, 6)))
                    remaining[plant_id] -= quantity
                    need -= quantity
        return Plan(period=t, orders=tuple(orders), status="base-stock")


def run(context: StreamContext, safety: float) -> Any:
    policy = context.compiling(OracleInterpreter(context.stream.gold), BaseStockDecider(safety))
    return context.simulator.run(policy)


def interval(values: list[float], seed: int = 0) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(RESAMPLES, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--dev", default="data/v2/dev")
    parser.add_argument("--grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.model_validate_json((args.run / "config.json").read_text())

    dev = list(load_streams(args.dev))
    tuning = {}
    for safety in args.grid:
        tuning[safety] = statistics.fmean(
            run(StreamContext(s, config), safety).total_cost for s in dev
        )
    best = min(tuning, key=tuning.__getitem__)
    print(f"== tuning on {args.dev} ({len(dev)} streams), mean cost by safety factor:")
    print("   " + ", ".join(f"{s}: {c:,.0f}" for s, c in tuning.items()) + f"  -> s = {best}")

    recorded = collections.defaultdict(dict)
    for row in map(json.loads, (args.run / "results.jsonl").read_text().splitlines()):
        recorded[row["condition"]][row["stream"]] = row
    rows = []
    for stream in load_streams(config.streams):
        context = StreamContext(stream, config)
        planner = context.planner_cost()
        want = recorded["oracle"][stream.id]["cost"]
        if abs(planner - want) > 1e-6 * max(1.0, abs(want)):
            sys.exit(f"GATE FAILED: {stream.id} planner {planner:,.1f} != recorded {want:,.1f}")
        result = run(context, best)
        rows.append(
            {
                "stream": stream.id,
                "cost": result.total_cost,
                "planner_cost": planner,
                "ratio": result.total_cost / planner,
                "r_dec": result.total_cost - planner,
                "fill_rate": result.fill_rate,
                "planner_fill_rate": recorded["oracle"][stream.id]["fill_rate"],
            }
        )
    print(
        f"   gate passed: planner reference reproduces the recorded oracle cost"
        f" on {len(rows)} streams"
    )

    def describe(name: str, ratios: list[float], regrets: list[float], fills: list[float]) -> None:
        low, high = interval(regrets)
        print(
            f"   {name:32s} ratio {statistics.fmean(ratios):5.2f}  "
            f"regret {statistics.fmean(regrets):10,.0f} "
            f"[{low:,.0f}, {high:,.0f}]  fill {statistics.fmean(fills):5.1%}  "
            f"costlier on {sum(r > 0 for r in regrets)}/{len(regrets)}"
        )

    print(
        f"\n== {args.run.name}: {len(rows)} streams,"
        " all deciding from the oracle state unless named"
    )
    describe(
        "planner (x*)", [1.0] * len(rows), [0.0] * len(rows), [r["planner_fill_rate"] for r in rows]
    )
    describe(
        f"base-stock (s = {best})",
        [r["ratio"] for r in rows],
        [r["r_dec"] for r in rows],
        [r["fill_rate"] for r in rows],
    )
    planner_by = {r["stream"]: r["planner_cost"] for r in rows}
    for condition, by_stream in recorded.items():
        if condition == "oracle":
            continue
        streams = [s for s in by_stream if s in planner_by]
        describe(
            condition,
            [by_stream[s]["cost"] / planner_by[s] for s in streams],
            [by_stream[s]["cost"] - planner_by[s] for s in streams],
            [by_stream[s]["fill_rate"] for s in streams],
        )
    base = {r["stream"]: r["cost"] for r in rows}
    for condition, by_stream in recorded.items():
        if condition == "oracle":
            continue
        diffs = [by_stream[s]["cost"] - base[s] for s in by_stream if s in base]
        low, high = interval(diffs)
        print(
            f"   {condition} minus base-stock: {statistics.fmean(diffs):,.0f}"
            f" [{low:,.0f}, {high:,.0f}]"
            f", worse on {sum(d > 0 for d in diffs)}/{len(diffs)}"
        )

    out = args.out or Path(config.output_dir) / f"{args.run.name}-basestock"
    out.mkdir(parents=True, exist_ok=True)
    (out / "basestock.json").write_text(
        json.dumps({"tuning": tuning, "safety": best, "rows": rows}, indent=1)
    )
    print(f"\n   report: {out / 'basestock.json'}")


if __name__ == "__main__":
    main()
