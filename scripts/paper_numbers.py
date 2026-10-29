"""Numbers for the ICORES paper, read from the run folders. Run from the CDD repo root.

usage: python scripts/paper_numbers.py READERS_PARA READERS_TEMPLATE DECIDERS

Under the planner benchmark, a reader's regret is its interpretation regret (the solver decides
in every reader condition), and a model decider's regret on the oracle state is its decision
regret. Intervals are 95% bootstrap intervals over streams (10,000 resamples, seed 0); a
difference between two conditions is resampled by stream, so it is paired.
"""

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

runs = Path("runs")

NAMES = {
    "oracle": "Oracle",
    "regex": "Regex",
    "qwen-json": "Qwen, free JSON",
    "qwen-json-constrained": "Qwen, constrained JSON",
    "llama-json-constrained": "Llama, constrained JSON",
    "qwen-atomic": "Qwen, typed questions",
    "laya-atomic": "Laya, typed questions (zero-shot)",
    "laya-ft-atomic": "Laya, typed questions (fine-tuned)",
}
RNG = np.random.default_rng(0)


def rows(run: Path) -> list[dict]:
    path = run / "results.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else []


def errors(run: Path) -> list[dict]:
    path = run / "errors.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else []


def by_condition(run: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for row in rows(run):
        out[row["condition"]][row["stream"]] = row
    return out


def ci(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return (float("nan"), float("nan"))
    means = data[RNG.integers(0, len(data), size=(10_000, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mean(values) -> float:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else float("nan")


def describe(values: list[float]) -> str:
    low, high = ci(values)
    nonzero = sum(abs(v) > 1e-6 for v in values)
    return (
        f"mean {mean(values):9,.0f} [{low:,.0f}, {high:,.0f}]  median "
        f"{statistics.median(values):8,.0f}  non-zero {nonzero}/{len(values)}"
    )


def main() -> None:
    para, template, deciders = (runs / name for name in sys.argv[1:4])
    P, T = by_condition(para), by_condition(template)
    streams = sorted(P.get("oracle", {}))
    print(
        f"== readers: {para.name} ({len(rows(para))} rows), {template.name} ({len(rows(template))} rows)"
    )
    print(
        f"   streams: {len(streams)}; VPI mean {mean(P['oracle'][s]['value_of_perfect_information'] for s in streams):,.0f}; "
        f"oracle cost mean {mean(P['oracle'][s]['cost'] for s in streams):,.0f}"
    )
    for key, name in NAMES.items():
        p, t = P.get(key, {}), T.get(key, {})
        if not p and not t:
            continue
        print(f"-- {name}")
        for label, group, other in (
            ("template", t, T.get("oracle", {})),
            ("paraphrased", p, P.get("oracle", {})),
        ):
            if not group:
                continue
            common = sorted(set(group) & set(other))
            regret = [group[s]["r_total"] for s in common]
            print(
                f"   {label:11s} F1 {mean(group[s]['f1'] for s in common):.3f}  regret {describe(regret)}  "
                f"cost ratio {mean(group[s]['cost'] / other[s]['cost'] for s in common):.3f}  "
                f"parse failures {mean(group[s]['parse_failures'] for s in common):.1f}  "
                f"rejected {mean(group[s]['rejected'] for s in common):.1f}  "
                f"conflicts dropped {sum(group[s].get('conflicts_dropped', 0) for s in common)}  "
                f"capped {sum(group[s].get('model_capped_replies') or 0 for s in common)}"
            )
        if p and t:
            common = sorted(set(p) & set(t))
            diff = [p[s]["r_total"] - t[s]["r_total"] for s in common]
            low, high = ci(diff)
            print(
                f"   paraphrase minus template regret: {mean(diff):,.0f} [{low:,.0f}, {high:,.0f}] (paired)"
            )

    print(f"\n== deciders: {deciders.name} ({len(rows(deciders))} rows)")
    D = by_condition(deciders)
    for key, group in D.items():
        if key == "oracle":
            continue
        common = sorted(set(group) & set(D.get("oracle", {})))
        regret = [group[s]["cost"] - group[s]["planner_cost"] for s in common]
        ratio = mean(group[s]["cost"] / group[s]["planner_cost"] for s in common)
        print(
            f"   {key:32s} decision regret {describe(regret)}  cost ratio {ratio:.2f}  "
            f"violations/stream {mean(group[s]['violations_true'] for s in common):.1f}  "
            f"invalid lines {sum(group[s].get('invalid_order_lines', 0) for s in common)}"
        )

    print("\n== cost concentration (paraphrased readers, single-repair Delta C)")
    errs = errors(para)
    for key, name in NAMES.items():
        mine = [e for e in errs if e["condition"] == key]
        if not mine:
            continue
        costs = sorted((float(e["delta_c"] or 0) for e in mine), reverse=True)
        positive = sum(c for c in costs if c > 0)
        free = sum(c <= 1e-6 for c in costs)
        if positive <= 0:
            print(f"   {name}: {len(mine)} errors; none raised cost")
            continue
        top = costs[: max(1, round(0.1 * len(costs)))]
        by_type = collections.Counter()
        for e in mine:
            by_type[e["delta_type"]] += max(0.0, float(e["delta_c"] or 0))
        lead = ", ".join(f"{t} {v / positive:.0%}" for t, v in by_type.most_common(3))
        print(
            f"   {name}: {len(mine)} errors; {free / len(mine):.0%} cost nothing; top 10% "
            f"({len(top)}) carry {sum(top) / positive:.0%} of positive Delta C; by type: {lead}"
        )


if __name__ == "__main__":
    main()
