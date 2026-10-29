# WP1 results: fresh-stream replication

Protocol: [replication-protocol.md](replication-protocol.md), frozen by tag
`icores-replication-freeze-v2` (commit bb45c0f). This file records what the frozen procedure
produced. Nothing in the protocol, code or configs changed after the split was generated.

## Run

- **Run on one local machine, 2026-10-26 22:41 to 2026-10-27 00:46 ET** (`scripts/wp1-overnight.sh`, about
  2 h 05 min). The 48 streams of `configs/benchmark/confirm.yaml` (seed 2027) were generated once
  and paraphrased once with Phi-4.
- **Run folder:** `icores-replication-para`.
  - `meta.json` records commit bb45c0f with `git_dirty: false`.
  - 384 rows (48 streams × 8 conditions); `failed.json` and `skipped.json` are empty; no resume
    was needed.
  - The recorded config is identical to `configs/experiments/icores-replication.yaml`.
- **Models:** Ollama 0.40.3 (`ollama.txt`, `models.json`), with the same digests as the pilot:
  - qwen2.5:7b-instruct 845dbda0ea48
  - llama3.1:8b 46e0c10c039e
  - phi4 ac896e5b8b34
  - laya 0.4.1

  The reply cache was new, so every reply was generated for this split (0 cache hits).
- **Checked independently on a second machine** from the uploaded run folder and streams: the
  metadata, the row and failure counts, and the config comparison above. Re-running
  `scripts/replication_analysis.py` gives output identical to the original log.
- **Before the run,** a smoke test of the frozen tag on 2 dev streams passed every check
  (protocol, "Pre-run check"). The time fallback was not needed.

## Primary (confirmatory)

Cost of Qwen free JSON minus Qwen constrained JSON, per stream:

| Paired streams | Mean | Median | Free costs more / less | 95% interval |
| --- | --- | --- | --- | --- |
| 48 | 7,916.1 | 304.8 | 26 / 14 | [3,863.1, 12,548.8] |

The lower bound is above 0, so **the pilot direction replicated.** (Pilot, 24 streams: mean
6,874.6, interval [381.6, 17,021.1].)

## Secondary (descriptive, unadjusted 95% intervals)

**R_interp per reader:**

| Reader | Mean | 95% interval | Median | Non-zero |
| --- | --- | --- | --- | --- |
| no-update | 22,866.9 | [14,688.8, 33,624.4] | 13,795.7 | 48/48 |
| regex | 2,520.2 | [620.9, 4,951.4] | 0.0 | 42/48 |
| qwen-json | 8,109.8 | [4,074.5, 12,696.0] | 905.2 | 42/48 |
| qwen-json-constrained | 193.7 | [-36.5, 505.6] | 0.0 | 9/48 |
| qwen-atomic | 82.3 | [0.0, 226.7] | 0.0 | 2/48 |
| llama-json-constrained | 798.7 | [-1.8, 2,270.0] | 0.0 | 10/48 |
| laya-atomic | 5,059.3 | [2,728.6, 7,718.3] | 244.5 | 42/48 |

**Cost minus the no-update baseline** (negative means the reader beats ignoring the messages).
Every reader beats it:

| Reader | Mean | 95% interval |
| --- | --- | --- |
| regex | -20,346.7 | [-31,059.2, -12,382.5] |
| qwen-json | -14,757.1 | [-25,143.4, -7,378.7] |
| qwen-json-constrained | -22,673.2 | [-33,386.0, -14,528.9] |
| qwen-atomic | -22,784.6 | [-33,531.8, -14,603.5] |
| llama-json-constrained | -22,068.2 | [-32,671.2, -14,020.2] |
| laya-atomic | -17,807.6 | [-28,258.4, -9,875.4] |

**Rank agreement:** Kendall's τ between readers' mean F1 and lower mean cost, over 6 readers and
48 streams, is 0.73 [0.47, 0.87].

**Cost concentration** (Shapley values; readers with at least 30 errors):

| Reader | Errors | Top 10% | Share of positive Shapley value |
| --- | --- | --- | --- |
| regex | 114 | 11 | 81% |
| qwen-json | 121 | 12 | 75% |
| laya-atomic | 156 | 16 | 81% |

qwen-json-constrained (25 errors), llama-json-constrained (19) and qwen-atomic (4) have fewer
than 30 errors, so they are not reported.

**Positive Shapley shares, pooled over readers:**
- **By delta type:** supplier_status 70%, quality_hold 16%, demand_signal 6%, lead_time_change
  5%, capacity_change 2%, quality_release 1%, price_change 1%.
- **By error kind:** missed 72%, wrong_type 22%, mis_timed 5%, wrong_value 1%; spurious,
  mis_scoped and provenance under 1%.

**Paraphrase:** 710 messages, 666 accepted, 2 unchanged, and 44 fell back to template text. The
fallbacks are kept as written, per the protocol.

## Scan notes (`scripts/scan_run.py`)

The only harness flags are reader output, which the protocol counts as such:
- llama-json-constrained: 1 reply hit the length cap, and 1 was malformed.
- qwen-json: 13 unknown entities and 1 `supersedes` naming an unknown id.

There were no failed pairs, retries or truncations.

## Laya export for WP3

The run script then re-ran Laya on the 24 pilot streams to export its completed deltas
(`icores-export-laya-para`). `scripts/check_export.py` against
`icores-final-readers-para-r3-shapley` found no mismatch in any pair's cost, F1 or error
identity. Its only listed problem was:

```
828 model requests reached a server: {'oracle': 0, 'laya-atomic': 828}
```

That check exists for the cached readers, whose export must make no request. Laya has no reply
cache, so it always asks the server. The rule fixed for Laya beforehand was:
include it in WP3 only if the re-run reproduces its recorded outputs exactly. That is met, so
**Laya is included in WP3**. An earlier checklist had asked for a gate ending in `none`, which was
stricter than that rule by mistake. The gate was re-run on a second machine from the uploaded
files, with the same output. The pilot reader export gate also reproduces here, ending in `none`.
