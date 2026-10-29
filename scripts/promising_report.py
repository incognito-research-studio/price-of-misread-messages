"""WP3 report: reader losses under order promising next to replenishment (scope 5.8, 5.10, 5.11).

usage: python scripts/promising_report.py REPLENISHMENT_RUN PROMISING_RUN [PROMISING_RUN ...]

REPLENISHMENT_RUN is the recorded reader run (its R_interp is the replenishment loss);
each PROMISING_RUN is a promising_run.py output folder (promising-v1, promising-v1-surge, or a
second Shapley seed). Runs are labelled by their folder names. Only streams present in every run
are used, and their number is printed. All intervals are 95% stream-bootstrap percentile
intervals (10,000 resamples, seed 0), descriptive and unadjusted.

Per policy: each reader's loss (mean, median, non-zero and negative streams, interval);
volume against the gold-update reference (the oracle row of the same run); solver fallbacks and
at-risk units; the checks of 5.10 (repaired gap, max_errors); cost concentration; positive
Shapley shares by delta type and error kind, pooled over readers.

Across policies: Kendall's tau between replenishment and each promising run's reader means,
with an interval; and the paired error table, joined on (stream, condition, error_id), of single
repair effects under both policies, with the largest disagreements listed.
"""

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RESAMPLES = 10_000
MAX_ERRORS = 200
EPS = 1e-6
NOT_READERS = {"oracle"}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def interval(values: list[float], seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean, over per-stream values."""
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(RESAMPLES, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class Run:
    """One run folder: rows by (condition, stream), errors, and each row's loss."""

    def __init__(self, path: Path, *, promising: bool) -> None:
        self.path, self.name, self.promising = path, path.name, promising
        self.rows: dict[str, dict[str, dict]] = collections.defaultdict(dict)
        for row in jsonl(path / "results.jsonl"):
            if not promising and row.get("r_interp") is None:
                continue  # direct-decision conditions have no interpretation loss
            self.rows[row["condition"]][row["stream"]] = row
        self.errors = jsonl(path / "errors.jsonl")
        config = json.loads((path / "config.json").read_text())
        conditions = config.get("experiment", config)["conditions"]
        self.layers = {
            c["name"]: (c.get("sop_gate", True), c.get("versioning", True)) for c in conditions
        }

    @property
    def streams(self) -> set[str]:
        return {s for rows in self.rows.values() for s in rows}

    def loss(self, row: dict) -> float:
        return float(row["loss"] if self.promising else row["r_interp"])

    def readers(self) -> list[str]:
        return sorted(c for c in self.rows if c not in NOT_READERS)

    def restrict(self, streams: set[str]) -> None:
        for condition in self.rows:
            self.rows[condition] = {s: r for s, r in self.rows[condition].items() if s in streams}
        self.errors = [e for e in self.errors if e["stream"] in streams]


def losses(run: Run) -> None:
    print("   loss per reader (positive: worse than the gold-update reference):")
    for reader in run.readers():
        values = [run.loss(r) for r in run.rows[reader].values()]
        if not values:
            continue
        low, high = interval(values)
        print(
            f"     {reader:24s} mean {statistics.fmean(values):10,.1f} [{low:,.1f}, {high:,.1f}]"
            f"  median {statistics.median(values):9,.1f}"
            f"  non-zero {sum(abs(v) > EPS for v in values)}/{len(values)}"
            f"  negative {sum(v < -EPS for v in values)}"
        )
    pairs = [run.loss(r) for c in run.readers() for r in run.rows[c].values()]
    print(f"   stream-reader pairs with negative loss: {sum(v < -EPS for v in pairs)}/{len(pairs)}")


def promising_details(run: Run) -> None:
    reference = run.rows.get("oracle", {})
    if reference:
        # The oracle row is the gold-update reference only for readers with the same layers.
        print("   volume against the gold-update reference (mean per stream, reader - reference):")
        for reader in run.readers():
            mine = run.rows[reader]
            if run.layers.get(reader) != run.layers.get("oracle"):
                print(f"     {reader:24s} other layers than the oracle row; not compared")
                continue
            shared = [s for s in mine if s in reference]
            if not shared:
                continue
            diff = {
                field: statistics.fmean(mine[s][field] - reference[s][field] for s in shared)
                for field in ("accepted_units", "rejected_units", "broken_orders")
            }
            print(
                f"     {reader:24s} accepted units {diff['accepted_units']:+9,.1f}  rejected units "
                f"{diff['rejected_units']:+9,.1f}  broken orders {diff['broken_orders']:+6.2f}"
            )
    rows = [r for c in run.rows.values() for r in c.values()]
    fallbacks = [r for r in rows if r.get("fallbacks")]
    statuses: collections.Counter[str] = collections.Counter()
    for row in rows:
        statuses.update(row.get("solver_statuses") or {})
    print(
        f"   solver: fallbacks {sum(r['fallbacks'] for r in fallbacks)} in {len(fallbacks)} pairs;"
        f" at-risk units {sum(float(r.get('at_risk_units') or 0) for r in rows):,.1f};"
        f" statuses {dict(statuses)}"
    )
    gaps = [r for r in rows if r.get("repaired_gap") is not None and abs(r["repaired_gap"]) > EPS]
    unchecked = [r for r in rows if r.get("repaired_gap") is None]
    print(f"   repaired-gap gate: {len(gaps)} failures, {len(unchecked)} pairs not checked")
    for row in gaps:
        print(f"     FAILED {row['stream']} {row['condition']}: {row['repaired_gap']}")


def attribution(run: Run) -> None:
    over = [r for c in run.rows.values() for r in c.values() if r["n_errors"] > MAX_ERRORS]
    print(f"   pairs over max_errors ({MAX_ERRORS}): {len(over)}")
    print("   cost concentration (Shapley, readers with at least 30 errors):")
    readers = run.readers()
    for reader in readers:
        values = sorted(
            (float(e["shapley"] or 0) for e in run.errors if e["condition"] == reader),
            reverse=True,
        )
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
    pooled = [e for e in run.errors if e["condition"] in readers and float(e["shapley"] or 0) > 0]
    total = sum(float(e["shapley"]) for e in pooled)
    if total <= 0:
        print("   positive Shapley shares: no costly errors")
        return
    for field in ("delta_type", "kind"):
        shares: collections.Counter[str] = collections.Counter()
        for e in pooled:
            shares[e[field]] += float(e["shapley"])
        print(
            f"   positive Shapley share by {field}, pooled over readers: "
            + ", ".join(f"{k} {v / total:.0%}" for k, v in shares.most_common())
        )


def rank_association(base: Run, other: Run) -> None:
    readers = sorted(set(base.readers()) & set(other.readers()))
    streams = sorted(
        set.intersection(*(set(base.rows[r]) & set(other.rows[r]) for r in readers))
        if readers
        else set()
    )
    if len(readers) < 3 or len(streams) < 2:
        print(f"   {other.name}: too few readers or streams for Kendall's tau")
        return

    def tau(sample: list[str]) -> float:
        a = [statistics.fmean(base.loss(base.rows[r][s]) for s in sample) for r in readers]
        b = [statistics.fmean(other.loss(other.rows[r][s]) for s in sample) for r in readers]
        return float(kendalltau(a, b).statistic)

    rng = np.random.default_rng(0)
    boot = [
        tau([streams[i] for i in rng.integers(0, len(streams), len(streams))]) for _ in range(2_000)
    ]
    boot = [b for b in boot if not np.isnan(b)]
    point = tau(streams)
    label = (
        f"   Kendall tau, {base.name} against {other.name} reader mean losses, {len(readers)} "
        f"readers, {len(streams)} streams:"
    )
    if np.isnan(point):
        print(f"{label} undefined (one policy's reader means are all tied)")
        return
    low, high = (np.percentile(boot, 2.5), np.percentile(boot, 97.5)) if boot else (np.nan,) * 2
    print(f"{label} {point:.2f} [{low:.2f}, {high:.2f}]")


def paired_errors(base: Run, other: Run, show: int = 10) -> None:
    def key(e: dict) -> tuple[str, str, str]:
        return e["stream"], e["condition"], e["error_id"]

    mine = {key(e): e for e in base.errors if e["condition"] not in NOT_READERS}
    theirs = {key(e): e for e in other.errors if e["condition"] not in NOT_READERS}
    shared = sorted(set(mine) & set(theirs))
    print(
        f"   paired errors {base.name} / {other.name}: {len(shared)} joined, "
        f"{len(set(mine) - set(theirs))} only in {base.name}, "
        f"{len(set(theirs) - set(mine))} only in {other.name}"
    )
    differ = [k for k in shared if mine[k]["kind"] != theirs[k]["kind"]]
    if differ:
        print(f"     WARNING: {len(differ)} joined errors differ in kind (error ids should match)")
    priced = [
        (k, float(mine[k]["delta_c"]), float(theirs[k]["delta_c"]))
        for k in shared
        if mine[k]["delta_c"] is not None and theirs[k]["delta_c"] is not None
    ]
    if not priced:
        print("     no single repair effects in both runs")
        return
    table = collections.Counter(
        (a > EPS, b > EPS) for _, a, b in priced
    )  # costly under replenishment, costly under promising
    print(
        f"     single repair effects, {len(priced)} errors: costly under both {table[True, True]},"
        f" replenishment only {table[True, False]}, promising only {table[False, True]},"
        f" neither {table[False, False]}"
    )
    print(f"     largest disagreements (by |rank difference|), top {show}:")
    rank_a = _ranks([a for _, a, _ in priced])
    rank_b = _ranks([b for _, _, b in priced])
    order = sorted(range(len(priced)), key=lambda i: -abs(rank_a[i] - rank_b[i]))
    for i in order[:show]:
        (stream, condition, error_id), a, b = priced[i]
        error = mine[stream, condition, error_id]
        print(
            f"       {stream} {condition:22s} {error_id:28s} {error['delta_type']:16s} "
            f"{error['kind']:14s} {base.name} {a:10,.1f}  {other.name} {b:10,.1f}"
        )


def _ranks(values: list[float]) -> list[float]:
    """Ranks in [0, 1], ties averaged, so policies with different monetary scales compare."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 / max(len(values) - 1, 1)
        i = j + 1
    return ranks


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base = Run(Path(sys.argv[1]), promising=False)
    others = [Run(Path(p), promising=True) for p in sys.argv[2:]]
    streams = set.intersection(base.streams, *(o.streams for o in others))
    for run in (base, *others):
        dropped = len(run.streams - streams)
        run.restrict(streams)
        print(
            f"== {run.name}: {len(streams)} shared streams"
            + (f" ({dropped} of its streams not in every run)" if dropped else "")
        )
        losses(run)
        if run.promising:
            promising_details(run)
        attribution(run)
        print()
    print("== ACROSS POLICIES (descriptive; monetary values are never pooled)")
    for other in others:
        rank_association(base, other)
        paired_errors(base, other)


if __name__ == "__main__":
    main()
