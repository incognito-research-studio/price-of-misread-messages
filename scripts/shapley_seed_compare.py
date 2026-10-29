"""Compare a recorded run's Shapley values with a replay under another seed.

usage: python scripts/shapley_seed_compare.py RECORDED_ERRORS REPLAY_ERRORS

Matches errors by stream, condition, kind, delta type and ids, and prints per-error agreement;
for each condition, the share of positive attributed cost carried by its costliest tenth (rounded
to whole errors, at least one) and the share of errors costing nothing or less under both seeds;
and how often the costliest error is the same, per stream-condition pair and per condition.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

Key = tuple[str, str, str, str, str | None, str | None]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(e: dict) -> Key:
    return (
        e["stream"],
        e["condition"],
        e["kind"],
        e["delta_type"],
        e.get("predicted_id"),
        e.get("gold_id"),
    )


def attributed(path: Path) -> dict[Key, float]:
    return {key(e): e["shapley"] for e in jsonl(path) if e.get("shapley") is not None}


def shares(values: dict[Key, float]) -> dict[str, tuple[int, float, float]]:
    """Per condition: error count, costliest-tenth share of positive cost, share at or below 0."""
    by_condition: dict[str, list[float]] = collections.defaultdict(list)
    for k, v in values.items():
        by_condition[k[1]].append(v)
    out = {}
    for condition, v in by_condition.items():
        v = sorted(v, reverse=True)
        positive = sum(s for s in v if s > 0)
        if positive <= 0:
            continue
        top = v[: max(1, round(0.1 * len(v)))]
        out[condition] = (len(v), sum(top) / positive, sum(s <= 0 for s in v) / len(v))
    return out


a, b = attributed(Path(sys.argv[1])), attributed(Path(sys.argv[2]))
common = sorted(set(a) & set(b), key=str)
print(f"recorded {len(a)}, replay {len(b)}, common {len(common)}")
x = np.array([a[k] for k in common])
y = np.array([b[k] for k in common])
d = np.abs(x - y)
print(f"max |diff| {d.max():,.1f}; median {np.median(d):,.2f}; corr {np.corrcoef(x, y)[0, 1]:.5f}")
for i in np.argsort(-d)[:5]:
    print("  ", common[i][:2], f"{x[i]:,.1f} vs {y[i]:,.1f}")

sa = shares({k: a[k] for k in common})
sb = shares({k: b[k] for k in common})
for c in sorted(sa):
    print(
        f"{c:40s} n={sa[c][0]:3d} top10 {sa[c][1]:.3f} vs {sb[c][1]:.3f}  "
        f"<=0 share {sa[c][2]:.2f} vs {sb[c][2]:.2f}"
    )

for label, group in (("pair", lambda k: k[:2]), ("condition", lambda k: k[1])):
    top_a: dict = {}
    top_b: dict = {}
    for k in common:
        g = group(k)
        if g not in top_a or a[k] > a[top_a[g]]:
            top_a[g] = k
        if g not in top_b or b[k] > b[top_b[g]]:
            top_b[g] = k
    same = sum(top_a[g] == top_b[g] for g in top_a)
    print(f"costliest error same per {label}: {same}/{len(top_a)}")
