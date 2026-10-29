# Architecture

The pipeline runs once per period for each stream, in the order of Figure 1 in the paper:

```text
messages ─▶ L1 interpreter ─▶ L4 span parser ─▶ L2 SOP ─▶ L3 bitemporal state ─▶ L5 solver ─▶ orders
                   │                 │              │              │                  │
                   └─────────────────┴──────────────┴──────────────┴──────────────────┴─▶ L6 audit
```

Only L1, and a model-backed decider if an experiment uses one, can be stochastic. Every other
step is a deterministic function of its inputs, so the interpreted state ŝ is a pure function
of `s0` and the interpreter's outputs. The wiring lives in
[`CompilingPolicy`](../src/cdd/pipeline.py).

## Delta schema (v1)

| Delta type | Entity kind | Attribute | Ops | Value |
| --- | --- | --- | --- | --- |
| `capacity_change` | plant | capacity (units/period) | set, add, scale | number |
| `lead_time_change` | lane | lead time (days) | set, add | number |
| `quality_hold` | lot | hold status | set | `held` |
| `quality_release` | lot | hold status | set | `released` |
| `supplier_status` | supplier | supplier status | set | `active`, `restricted`, `suspended` |
| `demand_signal` | SKU | demand (units/period) | set, add, scale | number |
| `price_change` | offer (supplier × SKU) | price | set, add, scale | number |

A delta is `(id, τ, e, a, op, v, u, w, src, κ, sup)`: type, entity, attribute, operation,
value, unit, validity window in periods (inclusive, `end = null` for open-ended), source span,
optional confidence and the ids of deltas it supersedes. Abstaining is an empty list, not a
type. Delta ids have the form `<message_id>#<suffix>`. Periods are ISO weeks of
`World.calendar_year`.

## L1 Interpreter

`interpret(message, ŝ) → Interpretation(proposals, failures)`. A proposal names the type,
the entity and the supporting span; every other slot is optional. Output the interpreter
cannot turn into a proposal (malformed JSON, evidence that is not in the message, an answer
outside the declared options) is recorded as a failure.

## L4 Span parser

Fills the operation, value, unit and window from the span text and keeps any slot the
interpreter already filled. Failures name the slot: `entity` (the id is not in the world, a
schema failure rather than a mis-scope), `span` (the span does not point into the message),
`value`, `window` or `schema`. A wrong span is therefore a separately attributable
error. When the span holds no magnitude, the parser falls back to a `magnitude` hint, which
the typed-model interpreter supplies.

Windows are read in this order: a week range (`wk42–44`, `weeks 42-44`, `wk42 to wk44`), a
start week (`from wk44`, `effective wk44`), an end week (`through wk44`), an end month
(`through Nov`), a start date (`effective Oct 1`), a single week (`into wk40`). If the span
states no window, the parser uses the interpreter's `window_start` / `window_end` hints (typed
models answer these), and otherwise the window runs open-ended from the message's period. This
is the parser-first rule; `window_rule: model_first` lets an explicit window from the
interpreter win instead.

## L2 SOP admissibility

| Rule | Applies to | Verdict |
| --- | --- | --- |
| `SOP-001` | every delta | reject if the entity is not of the kind the type requires |
| `SOP-002` | `capacity_change` | approve(ℓ) if the relative change exceeds the registered threshold (25% in `sop-v1`) |
| `SOP-003` | `quality_release` | reject unless ŝ holds the lot at the window start |
| `SOP-004` | `supplier_status` = suspended (`sop-v1`); every status (`sop-v2`) | reject unless the sender class is authorised |
| `SOP-005` | `supplier_status` (`sop-v2`) | reject unless the span names the supplier |

The first rejection wins; otherwise the first approval requirement wins; otherwise the delta
is accepted. Thresholds come from a versioned YAML file (`configs/sop/`); the experiments use
`sop-v2`.

Approval requests go to a fixed, deterministic, simulated approver: `always_approve`,
`always_reject` or `magnitude_cap` (approves relative changes up to a cap). Approvers receive
the delta, the rule id and ŝ. They never receive gold deltas.

## L3 Bitemporal state and the update operator

The state keeps an append-only history of four entry kinds: assertions, retractions, merges
and conflict records. A value is computed, not stored: start from the base value in `s0` and
apply, in recording order, every delta that was in force at transaction time `as_of` and
whose window covers the period. Asking with `as_of = t` answers "what did the pipeline
believe at period t".

`s_{t+1} = s_t ⊕ D_t` ([`update.py`](../src/cdd/update.py)):

1. **Canonicalise and sort** the batch by entity, attribute, window, type, op, value and id.
2. **Merge duplicates**, within the batch and against deltas already in force: a delta with
   the same effect is recorded as a merge, and its span is added to the kept delta's sources.
   A repeated "+6 days" notice is therefore not applied twice.
3. **Withhold contradictions**: deltas on the same entity, attribute and window with different
   effects, where none supersedes another, are not applied. They become a conflict record. In
   `sop-v2`, if exactly one of them comes from a sender class that normally issues its type,
   that one is applied; otherwise the approver may choose one or leave the prior state in place.
4. **Write** assertions, then retract every superseded delta, then log the diff.

A release closes a hold because it is a later `set` on the same lot: from its start week on,
the fold returns `released`.

## L5 Solver

A rolling-horizon MILP ([`solver.py`](../src/cdd/solver.py)), solved with HiGHS through
SciPy. Each period it plans `H` periods ahead and executes the first. For every orderable
line (plant `p`, SKU `k`) and period `τ` in the horizon:

```text
minimise   Σ price(p,k,τ)·x[p,k,τ] + Σ holding(k)·I[k,τ] + Σ backlog_cost(k)·B[k,τ]
subject to Σ_k x[p,k,τ] ≤ cap(p,τ)                                   capacity × supplier status
           I[k,τ] − B[k,τ] = I[k,τ−1] − B[k,τ−1] + arrivals(k,τ) + known(k,τ) − d(k,τ)
           moq(p,k)·y ≤ x[p,k,τ] ≤ cap(p,τ)·y,  y ∈ {0,1}                 only if a MOQ is set
           Σ price·x[·,·,τ] ≤ budget                                    only if a budget is set
```

All parameters are read from ŝ: capacity (zero when the supplier is suspended, reduced when
restricted; a status notice from a refused order overrides ŝ's status until a newer status
delta is recorded), lead time (an order placed in `τ` arrives after the lead time in force in `τ`),
demand, prices and lot holds. `known` covers open orders. Each lot has its own stock and draw
variables: a lot can be drawn only in periods where ŝ shows it released, and only up to that
period's demand plus backlog, so a hold freezes whatever is left of it. Duals come from the LP
with integer variables fixed at their MILP values; they are exact local sensitivities only for
right-hand-side changes inside the optimal basis.

## L6 Audit

Every message, failure, admissibility verdict, update (with its diff), conflict resolution
and decision is logged as a JSON record. A decision's lineage is the set of deltas in force
over the planning window; each one links to its source span and the rule that admitted it.
Audit completeness is the share of decisions whose lineage resolves fully.

## Simulator

[`InventorySimulator`](../src/cdd/simulator.py) owns the physics and the true state, built
from every gold delta. Each period: open orders arrive, the policy sees an inventory position,
orders are accepted up to true capacity (nothing from a suspended supplier), each accepted
order ships with the true lead time, and demand is served from released lots first (a later
hold would freeze them) and then from free stock. Orders beyond true capacity or from a suspended supplier, and spend above the
budget at true prices, are logged as true-constraint violations. Cost is purchase plus holding
plus backlog.

## Layer ablations

`CompilingPolicy(sop_gate=False)` admits every parsed delta. `CompilingPolicy(versioning=False)`
replaces the update operator with `apply_overwrite`: each delta retracts every delta in force on
the same entity and attribute, whatever their windows, with no merging, supersession or conflict
detection.

## Simplifications in v0.1

- One distribution centre, several plants, one inbound lane per plant (single echelon).
- The typed-model interpreter asks for at most one delta per message.
- Conflict handling and parsing are rule-based, as in the paper's non-goals.
