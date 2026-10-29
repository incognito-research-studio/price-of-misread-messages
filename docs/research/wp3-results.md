# WP3 results: customer-order promising as a second decision

Parameters:
`configs/promising/promising-v1.yaml` and `promising-v1-surge.yaml`, frozen before any pilot
promising run ([promising-dev-check.md](promising-dev-check.md)). The readers' recorded pilot
outputs were replayed with no model call. Promising and replenishment values are never pooled.

## Inputs and gates

- **Recorded run:** `icores-final-readers-para-r3-shapley`, the paper's paraphrased pilot, with 24
  streams, 6 readers plus the oracle, single repairs and 200 Shapley permutations.
- **Reader output:**
  - the export gate run `icores-export-readers-para` (5 readers; gate `none`, reproduced on a
    second machine);
  - the Laya export `icores-export-laya-para`, which reproduces every recorded Laya pair and
    is included under the rule in [wp1-results.md](wp1-results.md).

  The two `proposals.jsonl` files were merged, and their oracle records are identical.
- **Replay check** (`scripts/replay_check.py`): all 168 pairs replayed from the exported
  deltas reproduce the recorded cost, R_interp, F1, error identity, single-repair effects and
  Shapley values for all 182 errors. Problems: `none`.
- **Promising runs** (`scripts/promising_run.py`), 168 pairs each:
  - nothing missing from the export;
  - 0 repaired-gap failures: repairing every error reproduces the gold-update reference exactly
    in every pair;
  - no pair over 200 errors;
  - 0 solver fallbacks; all 2,688 weekly promise solves optimal.
- **Second Shapley seed:** surge re-run with run seed 1.
  - Losses are identical.
  - The largest change in any error's Shapley value is 21.8.
  - Pooled positive shares by type are lead_time_change 62 / quality_release 34 / quality_hold 4,
    against 61 / 34 / 5 with seed 0.

## Reader losses within each policy

Losses are against the gold-update reference of the same policy. Positive means worse. Means are
per stream over 24 streams, with stream-bootstrap 95% intervals.

| Reader | Replenishment R_interp | promising-v1 | promising-v1-surge |
| --- | --- | --- | --- |
| regex | 1,042.0 [-296.8, 2,913.9] | 98.8 [0.0, 294.7] | 77.6 [0.0, 203.2] |
| qwen-json | 6,817.7 [323.2, 16,998.0] | 0.0 | -3.8 [-33.2, 22.0] |
| qwen-json-constrained | -57.0 [-165.6, 14.1] | 0.0 | 0.0 |
| qwen-atomic | -461.7 [-1,106.6, 0.0] | 14.8 [0.0, 44.4] | 37.7 [0.0, 113.2] |
| llama-json-constrained | -470.3 [-1,612.9, 217.8] | 1.3 [0.0, 3.9] | 5.9 [0.0, 17.6] |
| laya-atomic | 1,086.9 [484.2, 1,815.3] | 0.0 | 31.8 [0.0, 80.0] |
| **Pairs with a non-zero loss** | 72 / 144 | 4 / 144 | 9 / 144 |
| **Pairs with a negative loss** | 28 / 144 | 0 / 144 | 1 / 144 |

**Volume against the gold-update reference** (mean per stream, reader minus reference):
- **v1:** regex accepts 10.2 fewer units; qwen-atomic adds 0.29 broken orders; the rest are
  unchanged.
- **Surge:**
  - qwen-json accepts 4.0 more units, with 0.12 more broken orders;
  - regex accepts 5.7 fewer, laya-atomic 2.4 fewer;
  - qwen-atomic has 0.58 more broken orders.

**At-risk units** (commitments the policy could not cover from projected supply), summed over all
pairs: 311.2 in v1 and 5,944.5 in surge.

**Concentration and shares** (Shapley values; readers with at least 30 errors):
- **v1:** regex's top 6 errors carry 100% of its positive value. Qwen JSON and Laya have no costly
  errors. Pooled positive shares: quality_release 84%, lead_time_change 16%.
- **Surge:** in each of regex, qwen-json and laya, the top 10% of errors carry 100% of the
  positive value. Pooled shares by type are lead_time_change 61%, quality_release 34% and
  quality_hold 5%. By kind they are missed 72%, mis_timed 24% and wrong_value 4%.
- **Replenishment (same streams):**
  - qwen-json's top 4 of 38 errors carry 95%, regex's top 6 of 57 carry 87%, and laya's top 7 of
    70 carry 66%.
  - Pooled shares by type: supplier_status 55%, quality_hold 18%, lead_time_change 12%,
    capacity_change 7%, demand_signal 5%.

## Across the two decisions (RQ4)

**Reader rankings do not carry over.** Kendall's τ between the replenishment and promising reader
mean losses (6 readers, 24 streams) is -0.30 [-0.58, 0.35] for v1 and -0.20 [-0.55, 0.75] for
surge. The intervals are wide, as expected with 6 readers, and many readers tie at 0 under
promising.

**The same reading error is priced differently.** The error ids are identical across policies
(182 joined, none unmatched). Single-repair effects:

| Errors costly under | v1 | Surge |
| --- | --- | --- |
| both | 2 | 5 |
| replenishment only | 80 | 77 |
| promising only | 2 | 2 |
| neither | 98 | 98 |

**The delta type explains it.** Errors costly under replenishment / costly under surge promising,
by type:

| Delta type | Errors | Costly, replenishment | Costly, surge promising |
| --- | --- | --- | --- |
| demand_signal | 53 | 23 | 0 |
| price_change | 29 | 16 | 0 |
| capacity_change | 27 | 9 | 0 |
| supplier_status | 24 | 11 | 0 |
| lead_time_change | 23 | 14 | 6 |
| quality_release | 13 | 6 | 1 |
| quality_hold | 13 | 3 | 0 |

Promising holds physical supply fixed at the planner's gold-state orders (Section 4.5 of the paper). A misread
demand, price, capacity or supplier-status message changes what replenishment orders, but cannot
change a promise. Only messages that change projected availability can: lead times (receipt
timing), holds and releases. This follows from the design (Section 4.5 of the paper) and is
not a separate finding.

**Examples:**
- The costliest replenishment error, a missed supplier-status message (pilot-018, qwen-json),
  costs 110,410 under replenishment and nothing under promising.
- A missed quality release (pilot-021, regex) costs nothing under replenishment and 2,329 under
  v1 (1,309 under surge): the policy keeps rejecting orders against stock it believes is held.
- A missed lead-time change (pilot-018, qwen-atomic) lowers replenishment cost by 6,370 but costs
  355 under v1 (905 under surge).

## Notes for the manuscript

- **v1 shows almost no reader loss.** That matches the dev check: with orders equal to demand,
  supply is rarely binding. Surge is the informative setting; report both, as frozen.
- **Monetary scales differ** (thousands under replenishment, tens under promising). Compare
  rankings, shares and the paired table, never the amounts.
- **No hold error is costly under surge,** and quality_hold carries 5% of the positive Shapley
  value. The hold-reveal sensitivity is below.

## Predetermined sensitivities (surge; losses only)

Run with `--shapley 0 --no-single`, so the repaired-gap gate is checked only for pairs with no
errors. 75 pairs with errors were not checked here; the main surge run checks all of them.
Parameter files: `configs/promising/promising-v1-surge-{penalties-half,penalties-double,reveal-holds}.yaml`.

| Reader | Surge | Penalties ×0.5 | Penalties ×2 | Hold revealed |
| --- | --- | --- | --- | --- |
| regex | 77.6 | 66.1 | 99.3 | 77.6 |
| qwen-json | -3.8 | -11.1 | 10.9 | -3.8 |
| qwen-json-constrained | 0.0 | 0.0 | 0.0 | 0.0 |
| qwen-atomic | 37.7 | 18.9 | 87.8 | 37.7 |
| llama-json-constrained | 5.9 | 2.9 | 11.7 | 5.9 |
| laya-atomic | 31.8 | 23.5 | 48.3 | 31.8 |
| **Kendall τ against replenishment** | -0.20 | -0.07 | -0.07 | -0.20 |

**Penalties.** Losses scale with the penalties, as expected; which pairs are non-zero does not
change. Every τ interval is wide and includes 0.

**Hold revealed.**
- Every reader loss is unchanged. The notice does engage: at-risk units rise from 5,944.5 to
  5,969.4.
- No hold error was costly in the main surge run either.
- So the hold-error promising costs need no upper-bound label: revealing the hold changes no
  reader's loss.

All three runs have 0 fallbacks, all solves optimal, and nothing missing from the export.
