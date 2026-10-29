"""WP1 planning numbers: model digests, run time per stream, and 24 or 48 streams.

usage: python scripts/wp1_budget.py READERS_RUN SHAPLEY_RUN PARA_STREAMS_DIR [OLLAMA_URL]

READERS_RUN is the reader run whose model replies were generated fresh (icores-final-readers-
para-r2); SHAPLEY_RUN the same readers again with single repairs and 200 Shapley permutations
(icores-final-readers-para-r3-shapley); PARA_STREAMS_DIR the paraphrased pilot streams.
Reads only; see docs/research/icores-revision-scope.md, Section 3.

1. Digests: each Ollama model in the runs' models.json against what the server now serves. A
   changed digest means the reply cache will not hit and the export gate cannot pass.
2. Run time: the logs carry no timestamps, so a run's wall time is meta.json's start to the last
   write of results.jsonl (unreliable for resumed runs, which are flagged). Fresh inference per
   stream = replies per stream x mean seconds of uncached requests (READERS_RUN request logs).
   Everything else (solver, repairs, Shapley, Laya, which has no cache) is SHAPLEY_RUN's wall
   time minus the model seconds it spent. Paraphrase time per stream comes from the spacing of
   the paraphrased stream files' modification times.
3. Stream count: per-stream cost differences, Qwen free JSON minus constrained JSON. For n = 24
   and 48, resample n streams from the pilot differences (treating the pilot effect as real),
   compute the 95% percentile bootstrap lower bound, and report how often it clears 0. The inner
   bootstrap uses 2,000 resamples to keep this quick; the analysis itself uses 10,000.
"""

import collections
import json
import statistics
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

readers, shapley, para = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
url = sys.argv[4] if len(sys.argv) > 4 else "http://localhost:11434"
FREE, CONSTRAINED = "qwen-json", "qwen-json-constrained"
OUTER, INNER, SEED = 2000, 2000, 0


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wall_seconds(run: Path) -> tuple[float, bool]:
    meta = json.loads((run / "meta.json").read_text())
    start = datetime.fromisoformat(meta["created_utc"])
    end = datetime.fromtimestamp((run / "results.jsonl").stat().st_mtime, tz=UTC)
    return (end - start).total_seconds(), bool(meta.get("resumed"))


print("== 1. model digests")
recorded = {}
for run in (readers, shapley):
    path = run / "models.json"
    for client in json.loads(path.read_text()) if path.exists() else []:
        if client.get("client") == "ollama":
            recorded.setdefault(client["model"], set()).add(client.get("digest"))
try:
    with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as response:
        served = {m["name"]: m.get("digest") for m in json.load(response).get("models", [])}
except OSError as exc:
    served = None
    print(f"   Ollama not reachable at {url} ({exc}); start it and run again")
for model, digests in sorted(recorded.items()):
    if len(digests) > 1:
        print(f"   {model}: the two runs recorded different digests {sorted(digests)}")
    if served is not None:
        now = served.get(model) or served.get(f"{model}:latest")
        status = "match" if now in digests else f"MISMATCH (serving {now})"
        print(f"   {model}: recorded {sorted(digests)[0][:12]}  {status}")

print("\n== 2. run time per stream")
rows = jsonl(shapley / "results.jsonl")
streams = sorted({row["stream"] for row in rows})
n_streams = len(streams)
uncached: dict[str, list[float]] = collections.defaultdict(list)
for log in sorted(readers.glob("requests-*.jsonl")):
    for entry in jsonl(log):
        if not entry.get("cached"):
            uncached[entry["model"]].append(float(entry["seconds"]))
replies: dict[str, int] = collections.Counter()
for log in sorted(shapley.glob("requests-*.jsonl")):
    for entry in jsonl(log):
        replies[entry["model"]] += 1
inference = 0.0
for model, count in sorted(replies.items()):
    if not uncached.get(model):
        print(f"   {model}: no uncached requests in {readers.name}; cannot price fresh replies")
        continue
    mean = statistics.fmean(uncached[model])
    inference += count * mean
    print(f"   {model}: {count / n_streams:,.0f} replies per stream x {mean:.2f} s uncached")
spent = sum(float(row.get("model_seconds") or 0) for row in rows)
wall, resumed = wall_seconds(shapley)
other = wall - spent
print(
    f"   {shapley.name}: wall {wall / 3600:.2f} h over {n_streams} streams"
    + ("  (RESUMED: wall time unreliable)" if resumed else "")
)
times = sorted(path.stat().st_mtime for path in para.glob("*.json"))
paraphrase = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else float("nan")
per_stream = (inference + other) / n_streams + paraphrase
print(
    f"   per stream: inference {inference / n_streams / 60:.1f} min, other (solver, repairs, "
    f"Shapley, Laya) {other / n_streams / 60:.1f} min, paraphrase {paraphrase / 60:.1f} min"
)
for n in (24, 48):
    print(f"   n = {n}: about {n * per_stream / 3600:.1f} h of run time")

print("\n== 3. stream count (free minus constrained JSON)")
cost = {(row["stream"], row["condition"]): row["cost"] for row in rows}
diffs = np.array(
    [
        cost[(s, FREE)] - cost[(s, CONSTRAINED)]
        for s in streams
        if (s, FREE) in cost and (s, CONSTRAINED) in cost
    ]
)
rng = np.random.default_rng(SEED)
pilot_means = diffs[rng.integers(0, len(diffs), size=(10_000, len(diffs)))].mean(axis=1)
print(
    f"   pilot: {len(diffs)} streams, mean {diffs.mean():,.0f}, median {np.median(diffs):,.0f}, "
    f"free costs more on {(diffs > 0).sum()}; 95% interval "
    f"{np.percentile(pilot_means, 2.5):,.0f} to {np.percentile(pilot_means, 97.5):,.0f}"
)
for n in (24, 48):
    cleared = 0
    for _ in range(OUTER):
        sample = diffs[rng.integers(0, len(diffs), size=n)]
        means = sample[rng.integers(0, n, size=(INNER, n))].mean(axis=1)
        cleared += np.percentile(means, 2.5) > 0
    print(f"   n = {n}: lower bound above 0 in {cleared / OUTER:.0%} of {OUTER} simulated splits")
print("   (an upper bound: the pilot estimate was chosen after inspection)")
