"""Numbers added in the 2026-10-24 revision of the ICORES paper. Run from the CDD root.

usage: python scripts/revision_numbers.py READERS_SHAPLEY NO_UPDATE DECIDERS [BUFFER_RUN ...]

Prints the no-update baseline, each reader's worst stream and positive/negative regret, Shapley
values against single repairs, the two cost-ratio statistics of the decider table, and the buffer
sensitivity runs. Intervals are 95% bootstrap intervals over streams (10,000 resamples, seed 0),
as in paper_numbers.py; a pooled share's interval resamples streams, not errors.
"""

import collections
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paper_numbers import NAMES, by_condition, ci, describe, errors, mean  # noqa: E402

runs = Path("runs")
shap_run, no_update_run, decider_run = (runs / n for n in sys.argv[1:4])
buffer_runs = [runs / n for n in sys.argv[4:]]
RNG = np.random.default_rng(0)

S, N = by_condition(shap_run), by_condition(no_update_run)
streams = sorted(S["oracle"])
oracle = {s: S["oracle"][s]["cost"] for s in streams}
print(f"== no-update baseline ({no_update_run.name}); oracle mean {mean(oracle.values()):,.0f}")
excess = [N["no-update"][s]["cost"] - N["oracle"][s]["cost"] for s in streams]
print(
    f"   excess {describe(excess)}  positive {sum(v > 1e-6 for v in excess)}/{len(excess)}  "
    f"max {max(excess):,.0f}  share of oracle {mean(excess) / mean(oracle.values()):.1%}  "
    f"fill {mean(N['no-update'][s]['fill_rate'] for s in streams):.3f} vs "
    f"{mean(N['oracle'][s]['fill_rate'] for s in streams):.3f}"
)
gold_value = [e for e in errors(no_update_run) if e["condition"] == "no-update"]
if gold_value:
    worth = sorted((float(e["delta_c"] or 0) for e in gold_value), reverse=True)
    print(f"   reading one message alone: {sum(w > 100 for w in worth)}/{len(worth)} worth more than 100")

print(f"\n== readers ({shap_run.name}): spread, and attribution")
errs = errors(shap_run)
for key, name in NAMES.items():
    if key not in S or key == "oracle":
        continue
    regret = [S[key][s]["r_interp"] for s in streams]
    mine = [e for e in errs if e["condition"] == key]
    single = sum(float(e["delta_c"] or 0) for e in mine)
    shap = sum(float(e["shapley"] or 0) for e in mine)
    print(
        f"-- {name}: R_interp {describe(regret)}  positive streams {sum(v > 1e-6 for v in regret)}  "
        f"worst {max(regret):,.0f}  best {min(regret):,.0f}  sum of positive {sum(v for v in regret if v > 0):,.0f}"
    )
    # Shapley values sum to C(no repair) - C(all repaired) on every sample, so a zero gap to
    # R_interp on every stream means that repairing every error reproduces the oracle cost.
    gap = max(
        abs(sum(float(e["shapley"] or 0) for e in mine if e["stream"] == s) - S[key][s]["r_interp"])
        for s in streams
    )
    print(
        f"   errors {len(mine)}; sum R_interp {sum(regret):,.0f}; sum single repairs {single:,.0f} "
        f"({single / sum(regret) if sum(regret) else float('nan'):.2f}x); sum Shapley {shap:,.0f}; "
        f"max per-stream |Shapley sum - R_interp| {gap:.6f}"
    )
    for field in ("delta_c", "shapley"):
        costs = sorted((float(e[field] or 0) for e in mine), reverse=True)
        positive = sum(c for c in costs if c > 0)
        if positive <= 0:
            print(f"   {field:8s} none positive")
            continue
        top = costs[: max(1, round(0.1 * len(costs)))]
        by_type, by_kind = collections.Counter(), collections.Counter()
        for e in mine:
            by_type[e["delta_type"]] += max(0.0, float(e[field] or 0))
            by_kind[e["kind"]] += max(0.0, float(e[field] or 0))
        print(
            f"   {field:8s} cost nothing or less {sum(c <= 1e-6 for c in costs)}/{len(costs)}; top 10% "
            f"({len(top)}) carry {sum(top) / positive:.0%}; types "
            + ", ".join(f"{t} {v / positive:.0%}" for t, v in by_type.most_common(4))
            + "; kinds "
            + ", ".join(f"{k} {v / positive:.0%}" for k, v in by_kind.most_common(3))
        )

print("\n== pooled positive shares across readers, with stream-bootstrap intervals")
readers = [e for e in errs if e["condition"] != "oracle"]
for field in ("delta_c", "shapley"):
    per_stream = collections.defaultdict(collections.Counter)
    for e in readers:
        value = max(0.0, float(e[field] or 0))
        per_stream[e["stream"]][e["delta_type"]] += value
        per_stream[e["stream"]]["kind:" + e["kind"]] += value
        per_stream[e["stream"]]["total"] += value
    keys = ["supplier_status", "quality_hold", "lead_time_change", "kind:missed"]
    draws = RNG.integers(0, len(streams), size=(10_000, len(streams)))
    out = []
    for k in keys:
        num = np.array([per_stream[s][k] for s in streams])
        den = np.array([per_stream[s]["total"] for s in streams])
        shares = num[draws].sum(axis=1) / den[draws].sum(axis=1)
        out.append(f"{k} {num.sum() / den.sum():.0%} [{np.percentile(shares, 2.5):.0%}, {np.percentile(shares, 97.5):.0%}]")
    print(f"   {field:8s} " + "; ".join(out))

print(f"\n== decider cost ratios ({decider_run.name})")
D = by_condition(decider_run)
for key, group in D.items():
    if key == "oracle":
        continue
    common = sorted(group)
    ratios = [group[s]["cost"] / group[s]["planner_cost"] for s in common]
    print(
        f"   {key:32s} mean of per-stream ratios {mean(ratios):.2f}  ratio of means "
        f"{mean(group[s]['cost'] for s in common) / mean(group[s]['planner_cost'] for s in common):.2f}  "
        f"min {min(ratios):.2f}"
    )

for run in buffer_runs:
    B = by_condition(run)
    if not B:
        continue
    common = sorted(B["oracle"])
    factor = B["oracle"][common[0]].get("planner_safety_factor")
    print(
        f"\n== buffer {factor} ({run.name}): oracle mean cost {mean(B['oracle'][s]['cost'] for s in common):,.0f}, "
        f"fill {mean(B['oracle'][s]['fill_rate'] for s in common):.3f}"
    )
    for key in ["no-update", *NAMES]:
        if key in B and key != "oracle":
            regret = [B[key][s]["cost"] - B["oracle"][s]["cost"] for s in common]
            low, high = ci(regret)
            print(
                f"   {NAMES.get(key, key):34s} mean {mean(regret):9,.0f} [{low:,.0f}, {high:,.0f}]  "
                f"median {statistics.median(regret):7,.0f}  positive {sum(v > 1e-6 for v in regret)}"
            )
