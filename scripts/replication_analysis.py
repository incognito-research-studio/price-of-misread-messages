"""The WP1 analysis, fixed before the fresh split was generated (replication-protocol.md).

usage: python scripts/replication_analysis.py RUN_DIR [STREAMS_DIR] [--first N]

Primary (the one confirmatory comparison): per stream, cost of Qwen free JSON minus cost of Qwen
constrained JSON (equal to the difference in R_interp, since both share the oracle reference).
Reported: mean, median, streams where free costs more, and the 95% percentile bootstrap interval
over streams (10,000 resamples, seed 0); "replicated" when the lower bound is above 0. Only
streams where both conditions finished are used, and their number is printed.

Secondary (descriptive, unadjusted 95% intervals, no confirmatory claim): each reader's R_interp;
each reader's cost minus the no-update baseline's; Kendall's tau between readers' mean F1 and
mean cost (higher F1 against lower cost), with a stream-bootstrap interval; cost concentration
for readers with at least 30 errors (manuscript definition: errors pooled over streams, sorted by
Shapley value, top max(1, round(n/10)), share of the summed positive Shapley value); positive
Shapley shares by delta type and error kind, pooled over readers; and, with STREAMS_DIR, the
paraphrase acceptance and fallback counts from each stream's provenance.

--first N restricts everything to the first N streams of the split in order (the ids in
STREAMS_DIR, else in the run, sorted): the protocol's time fallback analyses the first 24 of 48.
"""

import argparse
import collections
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
from paper_numbers import by_condition, errors

FREE, CONSTRAINED = "qwen-json", "qwen-json-constrained"
READERS = (
    "regex",
    "qwen-json",
    "qwen-json-constrained",
    "qwen-atomic",
    "llama-json-constrained",
    "laya-atomic",
)
RESAMPLES = 10_000


def interval(values: list[float], seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean, over the given per-stream values."""
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(RESAMPLES, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(runs: dict, a: str, b: str, field: str = "cost") -> tuple[list[str], list[float]]:
    streams = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    return streams, [runs[a][s][field] - runs[b][s][field] for s in streams]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("streams", nargs="?")
    parser.add_argument("--first", type=int)
    args = parser.parse_args()
    run = args.run
    runs = by_condition(run)
    split = None
    if args.streams:
        from cdd.benchmark.stream import load_streams

        split = load_streams(args.streams)
    keep = None
    if args.first is not None:
        ids = [s.id for s in split] if split else [s for rows in runs.values() for s in rows]
        keep = set(sorted(set(ids))[: args.first])
        runs = {c: {s: r for s, r in rows.items() if s in keep} for c, rows in runs.items()}
        print(f"== restricted to the first {args.first} streams of the split (--first)")
    all_streams = sorted({s for rows in runs.values() for s in rows})
    print(f"== {run.name}: {len(all_streams)} streams, conditions {sorted(runs)}")
    for condition in sorted(runs):
        missing = len(all_streams) - len(runs[condition])
        if missing:
            print(f"   {condition}: {missing} streams missing")

    print("\n== PRIMARY: cost of Qwen free JSON minus Qwen constrained JSON, per stream")
    _, diffs = paired(runs, FREE, CONSTRAINED)
    if len(diffs) < 2:
        print("   not enough paired streams")
    else:
        low, high = interval(diffs)
        print(
            f"   paired streams {len(diffs)}; mean {statistics.fmean(diffs):,.1f}; "
            f"median {statistics.median(diffs):,.1f}; free costs more on "
            f"{sum(d > 1e-6 for d in diffs)}, less on {sum(d < -1e-6 for d in diffs)}"
        )
        print(
            f"   95% interval [{low:,.1f}, {high:,.1f}]  ->  "
            f"{'REPLICATED' if low > 0 else 'NOT replicated'} (lower bound above 0)"
        )

    print("\n== SECONDARY (descriptive, unadjusted)")
    print("   R_interp per reader:")
    for reader in ("no-update", *READERS):
        values = [row["r_interp"] for row in runs.get(reader, {}).values()]
        if values:
            low, high = interval(values)
            print(
                f"     {reader:24s} mean {statistics.fmean(values):10,.1f} [{low:,.1f}, "
                f"{high:,.1f}]  median {statistics.median(values):9,.1f}  "
                f"non-zero {sum(abs(v) > 1e-6 for v in values)}/{len(values)}"
            )
    print("   cost minus no-update (negative: the reader beats ignoring messages):")
    for reader in READERS:
        _, diffs = paired(runs, reader, "no-update")
        if len(diffs) > 1:
            low, high = interval(diffs)
            print(
                f"     {reader:24s} mean {statistics.fmean(diffs):10,.1f} [{low:,.1f}, "
                f"{high:,.1f}] over {len(diffs)} streams"
            )

    present = [r for r in READERS if r in runs]
    common = sorted(set.intersection(*(set(runs[r]) for r in present))) if present else []
    if len(present) >= 3 and len(common) >= 2:

        def tau(sample: list[str]) -> float:
            f1 = [statistics.fmean(runs[r][s]["f1"] for s in sample) for r in present]
            cost = [-statistics.fmean(runs[r][s]["cost"] for s in sample) for r in present]
            return float(kendalltau(f1, cost).statistic)

        rng = np.random.default_rng(0)
        boot = [
            tau([common[i] for i in rng.integers(0, len(common), len(common))])
            for _ in range(2_000)
        ]
        boot = [b for b in boot if not np.isnan(b)]
        print(
            f"   Kendall tau, mean F1 against lower mean cost, {len(present)} readers, "
            f"{len(common)} streams: {tau(common):.2f} "
            f"[{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]"
        )

    print("   cost concentration (Shapley, readers with at least 30 errors):")
    errs = [e for e in errors(run) if keep is None or e["stream"] in keep]
    for reader in READERS:
        mine = [e for e in errs if e["condition"] == reader]
        values = sorted((float(e["shapley"] or 0) for e in mine), reverse=True)
        positive = sum(v for v in values if v > 0)
        if len(values) < 30:
            print(f"     {reader:24s} {len(values)} errors (fewer than 30, not reported)")
        elif positive <= 0:
            print(f"     {reader:24s} {len(values)} errors, no costly errors")
        else:
            top = values[: max(1, round(0.1 * len(values)))]
            print(
                f"     {reader:24s} {len(values)} errors; top {len(top)} carry "
                f"{sum(top) / positive:.0%} of the positive Shapley value"
            )

    pooled = [e for e in errs if e["condition"] in READERS and float(e["shapley"] or 0) > 0]
    total = sum(float(e["shapley"]) for e in pooled)
    for field in ("delta_type", "kind"):
        shares: collections.Counter[str] = collections.Counter()
        for e in pooled:
            shares[e[field]] += float(e["shapley"])
        if total > 0:
            print(
                f"   positive Shapley share by {field}, pooled over readers: "
                + ", ".join(f"{k} {v / total:.0%}" for k, v in shares.most_common())
            )

    if split is not None:
        counts: collections.Counter[str] = collections.Counter()
        for stream in split:
            if keep is not None and stream.id not in keep:
                continue
            info = stream.provenance.get("paraphrase", {})
            for key in ("messages", "accepted", "unchanged", "fallback"):
                value = info.get(key, 0)
                # fallback is the list of message ids kept as template text
                counts[key] += len(value) if isinstance(value, list) else int(value)
        print(f"   paraphrase ({args.streams}): {dict(counts)}")


if __name__ == "__main__":
    main()
