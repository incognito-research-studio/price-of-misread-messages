# Evaluation

## Regret decomposition

For each stream the runner computes three costs with the same simulator and the same demand
draws:

| Cost | Policy |
| --- | --- |
| `C(x^π(ŝ))` | the condition: its interpreter, the SOP layer, ⊕ and its decider |
| `C(x^π(s*))` | the oracle condition with the same decider: gold deltas through the same pipeline |
| `C(x*)` | the benchmark (below) |

`R_total = R_interp + R_dec` follows by telescoping; it is an accounting identity. `R_interp`
can be negative. The oracle reference is computed per decider (and per model client), so a
model deciding over ŝ gets its own `R_dec` and can be compared with the solver's directly. When
a condition's reader is the oracle, its own run is the reference, so `R_interp` is exactly zero
for it even if a model decider is not deterministic. Stream deciders (condition (a) and its
typed variant) have no ŝ; for them the runner reports `R_total` only.

Gold deltas that the SOP layer blocks in the oracle condition would show up in `R_dec`, not
`R_interp`, because the oracle condition is SOP-gated and the physical truth is not. On the b2
splits the oracle condition has no rejection and no conflict drop (`cdd calibrate` checks it).

### The benchmark x*

The default (`benchmark: planner`) fixes x* before any reported run: it is the configured
solver deciding from the oracle state, with a safety factor (the share by which it inflates the
demand forecast) tuned once on the dev split and fixed in the configs before any evaluation run. So:

- for every condition whose decider is the solver, `R_dec = 0` by construction and all regret
  is interpretation regret;
- for a model decider on the oracle state, `R_dec` is its extra cost over the solver on the
  same state;
- the gap from x* to the hindsight optimum x^H (one MILP over the whole horizon with the true
  parameters and the realised demand) is reported separately as
  `value_of_perfect_information`: foresight of demand noise and of events not yet announced,
  which no reader and no decider working week by week can reduce.

Nothing is chosen after seeing the evaluation streams. The earlier default, `benchmark: tuned`,
picked the best of several safety factors for each stream after seeing its realised cost; the
streams that picked a large buffer were exactly the ones with a supplier suspension, so it
partly anticipated events. It is kept to reproduce earlier runs, as is `benchmark: hindsight`.

Tuning, on the b2 dev split:

```bash
cdd generate --config configs/benchmark/dev.yaml --out data/v2/dev
cdd tune-solver --config configs/experiments/local-readers.yaml --streams data/v2/dev \
  --factors 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5
```

A 30% buffer was best (mean oracle cost 84,967 against 108,745 with none, 85,632 at 25% and
85,770 at 35%); the configs carry `safety_factor: 0.3` from 2026-10-23.

## Delta alignment and error kinds

[`matching.py`](../src/cdd/evaluation/matching.py) aligns predicted with gold deltas per
stream in three stages:

1. Bipartite matching on (type, entity, attribute) with overlapping windows. A matched pair is
   exact, **mis-timed** (window differs), **wrong value** (op or value differs) or
   **provenance** (the spans do not overlap, e.g. cited to the wrong message).
2. Bipartite matching of the remaining deltas from the same source message: **wrong type**,
   **mis-scoped** (entity differs), or mis-timed when the windows do not overlap at all.
3. Leftover gold deltas are **missed**; leftover predictions are **spurious**.

Windows are compared inside the simulated horizon: with a horizon ending in week 52,
`wk50-52` and `from wk50` are the same window, because no decision can tell them apart. An
entity id that does not exist in the world never reaches alignment; the parser records it as
a schema failure, so the gold delta it should have produced counts as missed.

When a pair has several discrepancies, the primary kind follows the order wrong type,
mis-scoped, mis-timed, wrong value, provenance; all of them are kept in `kinds`.

**Supersession references.** A wrong or missing reference is not one of the seven kinds. It
counts as a wrong value, but only when it changes a state cell at or after the week of the
message: a release that also names the hold it ends changes nothing a decision can read, and a
reference to another entity's delta is left unresolved by the update, so neither is an error.
A delta with no reference that restates the same cell and window as an earlier one replaces it
(the update treats it as a revision), and alignment counts it as a reference. References are
compared by source message, because predicted and gold ids differ. Section 3.8 aligns on
(τ, e, a) with window overlap, which cannot produce wrong-type or mis-scoped pairs on its own;
stage 2 is the implementation's answer.

## Metrics

- **Precision, recall, F1** (strict): a prediction counts only if it matches gold exactly.
  Reported overall and by type.
- **Error table**: counts by (gold delta type, error kind), the cells of S(τ, k).
- **Parse failures** and interpreter failures, counted per condition.
- **True-constraint violations** of executed orders, and fill rate.
- **Audit completeness** (1.0 by construction for compiling conditions).
- **Decision consistency**: `coefficient_of_variation` over order vectors from repeated or
  paraphrased runs (`SimulationResult.order_vector()`).
- **Calibration** (`calibration_ece`): expected calibration error of the deltas' confidence κ
  against exact correctness, when the interpreter reports κ.
- **Model usage** (`model_requests`, `model_seconds`): requests and model time for the main run
  of a condition, for the cost–latency frontier; replays and references are excluded.
- **Escalation share** for the hybrid condition (j).
- **SOP and update counts**: `rejected` (SOP rejections), `approvals_requested`, and
  `conflicts_dropped` (deltas of a same-week conflict that neither the sender-authority rule
  nor the approver kept).
- **Invalid order lines** (`invalid_order_lines`): lines a model decider wrote for a plant that
  does not make the SKU; they are dropped.
- **Reply cache hits** (`model_cache_hits`): requests answered from the disk cache.

`summary.csv` holds per condition the mean of each metric, the median of cost and regret, the
number of streams with non-zero total regret (`n_nonzero_r_total`), and the interpretation share
as a ratio of sums (`ratio_interp_share`, ΣR_interp / ΣR_total). Report the ratio, not the mean
of per-stream shares: when a few streams carry most of the regret, the mean understates it
(0.037 against 0.182 for typed Qwen in the first ICORES run). A warning is logged when
conditions cover different numbers of streams.

## Ablations

| Flag | Effect |
| --- | --- |
| `sop_gate: false` | every parsed delta is admitted (no L2) |
| `versioning: false` | each delta overwrites every delta in force on its attribute (no bitemporal L3) |
| `window_rule: model_first` | the typed model's window answer wins over the span |
| `labels: neutral` / `bare` | typed type-question options become letters with definitions, or names without definitions |

The oracle reference for interpretation regret runs through the same layers as the condition,
so an ablation's `R_interp` still isolates interpretation.

## Attribution

A *repair* removes the predicted delta behind an error and inserts the gold one. A gold
delta's `supersedes` references name gold ids, so the repair also names the predicted deltas
aligned with them; without that, repairing a revision could not retract the reader's original. The runner
replays the condition's own parsed deltas through the full pipeline with a set of repairs
applied, so SOP checks, conflicts and supersessions are re-evaluated. With no repairs, the
replay reproduces the original cost exactly (tested).

- `ΔC_j = C(no repairs) − C(repair j only)`: positive values are costly errors.
- Shapley values average each repair's marginal cost reduction over random repair orders
  (Castro et al., 2009). They sum to `C(no repairs) − C(all repaired)` for every sample.
  Costs are cached per coalition, so the price is one simulation per new coalition.

`sensitivity.csv` aggregates counts and summed `ΔC_j` by (condition, delta type, error kind).

## Properties that hold by construction

These are properties of the architecture, not findings. Tests check that the implementation
keeps them:

| Property | Test |
| --- | --- |
| Proposition 1: an executed order that breaks a true constraint traces to a logged delta error | `test_true_violation_traces_to_a_delta_error` |
| Proposition 2(i): identical canonical delta sequences give identical decisions | `test_identical_delta_sequences_give_identical_decisions` |
| The oracle condition has zero interpretation regret and full lineage | `test_oracle_condition_has_no_interpretation_regret` |
| Replaying parsed deltas with no repairs reproduces the run | `test_replaying_proposed_deltas_reproduces_the_run` |

## Run outputs

| File | One row per |
| --- | --- |
| `results.jsonl` | stream × condition |
| `summary.csv` | condition (means over streams) |
| `errors.csv` | delta error, with `ΔC_j` and Shapley value |
| `sensitivity.csv` | condition × delta type × error kind |
| `config.json` | the resolved experiment config |
| `meta.json` | package and schema versions, code commit, whether the tree was dirty, time |
| `skipped.json` | condition that could not run, with the reason |
| `errors.jsonl` | the rows of `errors.csv`, appended as each pair finishes (used by `--resume`) |
| `failed.json` | stream × condition pair that raised, with the error |
| `requests-<model>.jsonl` | model request, tagged `stream/condition`, with the reply, seconds, `done_reason` and token counts (Ollama; off with `log_requests: false`); replies from the disk cache carry `"cached": true` |
| `models.json` | model client: model name, Ollama digest and server version, or Laya checkpoint and revision |

`results.jsonl` and `errors.jsonl` grow as each stream × condition pair finishes. A pair that
raises (a timed-out request after its retries, a server that went away) is listed in
`failed.json` and the run goes on; `cdd run` then exits with status 1 and prints the
`--resume` command, which reruns only the pairs the folder lacks. Resuming requires the same
config. Each pair has its own random generator, seeded from the config seed and the pair's
position, so a resumed run reproduces an uninterrupted one. The summary columns
`model_retries`, `model_capped_replies` (replies cut by `max_tokens`) and `model_truncations`
(Laya inputs it truncated) should be zero; anything else is worth a look before reading the
results.

With `reply_cache` set (the local configs use `runs/cache/replies.sqlite`), every model reply is
stored under a hash of the request body and the model digest, and an identical request returns
the stored reply. A rerun after a change downstream of the readers then takes minutes, and a
model reader or decider is one sample at temperature 0: repeated runs give the same answers.
Measuring decision consistency across samples needs other seeds or temperatures and the cache
switched off.
