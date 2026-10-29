# Promising: development-split check (2026-10-27)

Model-free: gold deltas only, no reader output, dev split (`data/v2/dev`, seed 101, 24 template
streams), `configs/experiments/icores-readers-shapley.yaml` (sop-v2, planner safety factor 0.3).
Command: `python scripts/promising_dev_check.py configs/experiments/icores-readers-shapley.yaml
data/v2/dev configs/promising/promising-v1.yaml`.

## promising-v1 as fixed

| Check | Result | Criterion |
| --- | --- | --- |
| Arriving units accepted (gold reference) | mean 99.5%, min 92.8% | >= 80% |
| Accepted orders on time | 100% on every stream | >= 90% |
| Broken promises | 0% | <= 10% |
| Solver fallbacks, units at risk | 0, 0 | reported |
| Fixed-supply gate (replay vs planner cost) | exact | exact |
| Regex repair gate | 0 failures, but regex makes 0 errors on dev template text | exact |

The criteria fixed beforehand (last column) are met. But **missing any single gold delta costs nothing under promising**: 0%
of 270 misses (all seven types) change Π. The planner's supply (three weeks of starting stock,
30% buffer) leaves so much slack that available-to-promise rarely binds. Run on the pilot as is,
WP3 would show near-zero interpretation loss for every reader, by construction of the settings.

## Model-free exploration (not adopted; development split only)

Share of single-delta misses costing more than 100 under promising, by type (hold, release,
lead time; the other four types are 0% in every variant, by design):

| Variant | Accepted | Hold | Release | Lead time |
| --- | --- | --- | --- | --- |
| v1 (orders = demand, customer lead 1-3 weeks) | 99% | 0% | 0% | 0% |
| planner buffer 0 instead of 0.3 (fixed supply only) | 99% | 3% | 7% | 0% |
| orders = 1.15 x demand | 99% | 3% | 4% | 0% |
| orders = 1.3 x demand | 95% | 8% | 15% | 13% |
| orders = 1.3 x demand, customer lead 2-5 weeks | 98% | 3% | 15% | 0% |

## Decision (2026-10-27, authors)

Keep v1 as the baseline and add one predeclared supply-constrained setting,
`configs/promising/promising-v1-surge.yaml`: identical to v1 except `demand_scale: 1.3` (customer
orders total 1.3 x the realised demand the fixed supply was planned for). Both are reported. The
setting was chosen on dev streams with gold deltas only, before any pilot promising run; nothing
else was tried beyond the table above. **Both parameter files are frozen from here on.**

## promising-v1-surge, through the real code path

| Check | Result |
| --- | --- |
| Arriving units accepted | mean 94.8%, min 78.6% |
| Accepted orders on time | mean 99.9%, min 98.8% |
| Broken promises | mean 0.1%, max 1.2% |
| Solver fallbacks, units at risk | 0, 167 units in total |
| Fixed-supply gate | exact |

The same criteria hold on stream means (one stream accepts 78.6% of units).

| Missed type | Misses | Costly (> 100) | Any loss | Negative | Share of loss |
| --- | --- | --- | --- | --- | --- |
| lead_time_change | 39 | 13% | 23% | 6 | 43% |
| quality_hold | 36 | 8% | 14% | 5 | 32% |
| quality_release | 27 | 15% | 26% | 0 | 25% |
| capacity, demand, price, supplier status | 168 | 0% | 0% | 0 | 0% |

The four zero types are zero by design; this is not an empirical result.

## Trajectories inspected by hand (surge setting)

- `dev-001`, missed hold on Lot 7703 (week 38), loss 435. Without the hold, ŝ counts the held lot
  as available: the reader accepts four SKU-03 orders the gold reference rejects, earns 274 more
  margin, but breaks promises later in weeks 49-52 (broken 413, terminal 220, extra lateness 77).
- `dev-003`, missed 5-day delay on Port W (weeks 43-49), loss 404. ŝ expects the open orders a week
  early; three SKU-04 orders are promised one week earlier than the gold reference promises them
  and are delivered partly a week late (broken 458), with the same margin.

Both match the intended mechanism: a misread hold or lead time changes the projected supply the
promise is made against.

## Caveats

- The regex repair gate is vacuous on dev template text (regex makes no errors there). The repair
  gate is exercised in `tests/test_promising_run.py` with a scripted reader that makes errors, and
  is checked on every pilot pair by `scripts/promising_run.py`.
- Negative single-miss losses exist (11 under surge): the weekly decision is myopic, so the gold
  reference is not optimal.
