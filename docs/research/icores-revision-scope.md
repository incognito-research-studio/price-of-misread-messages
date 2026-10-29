# Research scope and design: replication, challenge set and order promising

**Status (2026-10-26):** the design of the three added studies, written before any of them was
implemented or run. Their results are in [wp1-results.md](wp1-results.md),
[wp2-results.md](wp2-results.md) and [wp3-results.md](wp3-results.md). The planning sections that
followed Section 6 (timeline, length and drafting notes) are omitted here.

## 0. Agreed scope in brief

- One synthetic commerce setting: the existing worlds (suppliers, plants, lanes, SKUs, offers, lots)
  and the existing operational message streams.
- Two downstream decisions, evaluated **separately**, never summed into one enterprise cost:
  - **Replenishment** (existing, the main and deeper experiment): what to order, from which plant, when.
  - **Customer-order promising** (new, required, bounded): which customer orders to accept and which
    delivery week to promise.
- New question (RQ4): how does the downstream decision policy change the magnitude, attribution and
  reader ranking of interpretation loss?
- Three work packages: WP1 fresh-stream replication of the replenishment study; WP2 small unfiltered
  language challenge; WP3 promising study (required; its minimum version is exploratory on the pilot
  streams and recorded reader outputs).
- Out of scope: ticket assignment, financial reconciliation, catalog or image benchmarks, validation
  across industries, joint replenishment-and-promising optimisation, verification allocation, coupled
  or shared-interpretation decision studies (CUE).

## 1. Research questions

RQ1–RQ3 keep the substance of `paper/sections/introduction.tex` (lines 51–63):

- **RQ1** (diagnostic control): given the same information, how much more does it cost to let a
  language model decide orders than to let the planning model decide?
- **RQ2**: under the planning model, what does each reader's misreading cost, and does extraction
  accuracy rank readers the way cost does?
- **RQ3**: where does the cost of misreading sit: in which delta types and error kinds, and in how
  many errors?
- **RQ4 (new)**: how do interpretation losses, error attributions and reader rankings change when the
  same operational message streams support customer-order promising rather than replenishment?

RQ4 tests policy dependence within one setting. It does not establish that compiling is better in
general. A difference between policies is a valid finding; the promising study must not be tuned to
reproduce the replenishment ranking, the concentration result or the decider gap.

## 2. Repository audit

### 2.1 Verified existing capabilities

| Capability | Where | What the code does |
| --- | --- | --- |
| Delta schema | `src/cdd/schema.py`: `DeltaType` (7 types), `DELTA_SPECS`, `Delta`, `Proposal`, `Window`, `Span` | `Proposal` needs type, entity and span; op, value, window and the rest are optional. `Span` is character offsets into one message. |
| World and lots | `src/cdd/world.py`: `World`, `Lot(sku, quantity, hold_status, arrival)` | Lot arrival weeks are master data. Holds apply to lots only. |
| Reader-visible information | `src/cdd/pipeline.py:68-103` (`CompilingPolicy.__call__`) | The interpreter gets `(message, state.view(as_of=t))` only. The JSON prompt renders entities and deltas in force (`src/cdd/interpreters/llm.py:94-97`). Readers never see `InventoryPosition`, orders or decisions. |
| SOP and approval | `src/cdd/sop.py`: `SOPGraph.check(delta, message, view)`, `SOPGraph.resolve`, `MagnitudeCapApprover`; `configs/sop/sop-v2.yaml` | SOP-001 to SOP-005. Approver and conflict resolution read the view, never the position or the plan. |
| Canonical update | `src/cdd/update.py`: `apply_update`, `apply_overwrite`; `src/cdd/state.py`: `BitemporalState`, `StateView` | Merge, supersession, withheld conflicts, `as_of` views. |
| Span parsing | `src/cdd/parser.py:115-140` (`SpanParser.complete`) | Keeps filled slots, fills op, value and window from the span, raises `ParseError(slot)`. It rejects a span that ends beyond the message text. |
| Reader outputs in memory | `CompilingPolicy.proposed` (completed deltas before SOP), `.failures`, `.audit` | Used for alignment and repairs. |
| Reader outputs on disk | `results.jsonl`, `errors.jsonl` (`experiments.py:643-658`), `requests-<model>.jsonl` | **Completed deltas are not persisted.** `errors.jsonl` keeps error kind, type, `predicted_id`, `gold_id`, `delta_c` and `shapley`, not the deltas. Request logs keep raw Ollama request bodies and replies. |
| Reply cache | `src/cdd/adapters/cache.py` (`ReplyCache`, key = sha256 of body and model digest); `src/cdd/adapters/ollama.py:150-206` | Before any cache lookup, the client fetches the model digest from a running Ollama server (`_check_model`, `ollama.py:253-263`), so **cache replay needs Ollama running with the same model pulled**. Cache hits do not increment `requests`. **The Laya adapter has no reply cache.** |
| Closed-loop replenishment | `src/cdd/simulator.py:144-259` (`InventorySimulator.run`) | Per week: receipts, then the policy call (messages read, ŝ updated, plan), then order acceptance at true capacity and status, then demand served from released, arrived lots first and then free stock, then backlog. |
| Open orders and arrivals | `src/cdd/orders.py` (`OutstandingOrder(placed, plant, sku, quantity)`, `lead_periods`, `capacity_available`, `supplier_status`); `src/cdd/solver.py:283-293` (`_expected_arrivals`), `:296` (`_lot_draws`) | An accepted order arrives after the **true lead time in force when it was placed**; later lead-time changes do not move it. A suspension refuses new orders only; **accepted shipments still arrive**. The planner projects an open order's arrival from ŝ's lead time at placement and moves an overdue order to the next week. Free stock (initial stock and receipts) is never held. |
| Ground truth | `pipeline.build_truth`; `experiments.StreamContext` (`:262-276`) | Gold deltas, not SOP-gated. Demand = true demand signal × mean-one lognormal noise, seeded by the stream seed, continuous quantities (`InventorySimulator.sample_demand`). |
| Fixed plans | `pipeline.PlannedPolicy` | Executes precomputed per-week plans. `PeriodRecord.ordered` and `.accepted` hold each week's lines. |
| Repair and replay | `src/cdd/evaluation/regret.py` (`repairs_for`, `apply_repairs`); `experiments.score_compiled.cost_with` (`:551-555`) | Repairs replay **completed deltas** through `ReplayInterpreter` and the full front end (parser, SOP, update) with the same decider. No model calls. |
| Single repair and Shapley | `single_repair_effects`, `shapley_permutation`, `CachedCost`; `AttributionConfig(max_errors=200)`; `configs/experiments/icores-readers-shapley.yaml` (200 permutations) | Shapley values sum to C(no repairs) − C(all repaired) per sample. The paper states that repairing every error reproduces the oracle cost on every stream. |
| Concentration as reported | `scripts/revision_numbers.py:66-80`, `scripts/concentration_figure.py` | Per reader, errors pooled over streams and sorted by attribution. The top max(1, round(0.1·n)) errors' summed value is divided by the sum of positive values. Readers with at least 30 errors are shown. The paper's RQ3 uses Shapley values; `scripts/paper_numbers.py` prints the same statistic on `delta_c`. |
| Run metadata | `experiments._meta` (`:868-898`) → `meta.json`; `models.json`; `config.json`; `--resume` | `git_commit`, `git_dirty`, versions, time; model digests and server version. |
| Reporting | `summarise`, `sensitivity`; `scripts/paper_numbers.py`, `scripts/revision_numbers.py` | 95% bootstrap over streams, 10,000 resamples, seed 0. |
| Splits | `configs/benchmark/*.yaml` | Seeds: pilot 7, train 31, dev 101, out-of-schema 211. Only `out-of-schema.yaml` sets `out_of_schema_rate`. |
| Paraphrase | `src/cdd/benchmark/paraphrase.py:279-282` | Re-anchors gold spans to evidence sentences and keeps a rewrite only if the span parser recovers each gold delta (`supports`). |
| Typed reader limit | `src/cdd/interpreters/typed.py:18` | At most one delta per message. |

### 2.2 What "the same errors" means (verified)

- **The front end does not depend on the decision.** The decider's output never reaches the state, the
  SOP layer or the approver. Refusal notices go only to the solver through `InventoryPosition`
  (`orders.supplier_status`). For a fixed sequence of reader outputs, the following are identical
  under any downstream policy that uses the same front end and SOP file:
  - completed deltas and parse failures
  - admission verdicts, conflicts and ŝ_t
  - the alignment, and so the error set (with error ids `e0000…` in the same order)
- **Identical reader outputs are guaranteed only by replaying recorded outputs.** Re-asking a model
  through the reply cache gives identical outputs only if every request body and digest is
  identical. That is expected, because ŝ is identical, but it must be checked: zero server requests
  (`model_requests`), with all replies counted as `model_cache_hits`.
- **What differs between policies** is the cost of each error, which errors cost anything, and the
  attribution. Reader-proposal errors (alignment) are kept apart from admission outcomes
  (`rejected_correct`, conflicts dropped) and from downstream effects.

### 2.3 New components required

- **N1 Reader-output export.** Persist `policy.proposed`, parse failures and admission verdicts per
  (stream, condition), e.g. `proposals.jsonl` in the run folder, and a loader that feeds them to
  `ReplayInterpreter`.
- **N2 Fixed supply trajectory.** Record the gold-update replenishment run's per-week order lines.
  Derive a supply ledger from the truth: receipts by SKU and week, lot arrivals, true hold status by
  lot and week.
- **N3 Customer-order generator.** Uses its own random generator. Streams and gold deltas are left
  unchanged.
- **N4 Promising policy.** A weekly MILP. It can reuse `solver._Model` and the patterns in
  `_expected_arrivals` and `_lot_draws`.
- **N5 Execution simulator and accounting for commitments.**
- **N6 Promising runner.** Front end, fixed supply, promising, execution, gold-update reference, and
  single-repair and Shapley attribution through the existing `CachedCost` path.
- **N7 Reporting.** Promising tables and cross-policy analysis scripts.
- **N8 Challenge import.** Hand-written texts plus marked evidence go into stream JSON. Gold spans are
  re-anchored; op, value and window are unchanged. Provenance records "hand-written, unfiltered".
- **N9 Parser-isolation interpreter.** Gold type and entity with the marked evidence span, and no op,
  value or window, so the span parser fills them.
- **N10 Replication config and protocol.** A new benchmark config with a fresh seed, plus the protocol
  document.

### 2.4 Design decisions to resolve before implementation

Each has a recommended default; the authors confirm or change them before any promising code is
written.

| Id | Decision | Recommended |
| --- | --- | --- |
| D1 | What the promising projection counts as future supply | Scheduled receipts only (ATP): free stock on hand, lots by arrival week and ŝ hold status, open orders at ŝ-projected arrival. No projected future replenishment (capable-to-promise), which would contradict the fixed physical supply. |
| D2 | Event order within a week | Section 5.3. The fixed replenishment orders of week t are placed before promising, so they count as open orders. |
| D3 | Reservation for forecast demand or service classes | None in the minimum: one customer class, no reservation. |
| D4 | Split delivery | One promise week per order; partial deliveries allowed in execution. An order is on time only if fully delivered by its promised week. |
| D5 | Revising accepted promises | Promises are binding and never revised. |
| D6 | Holding cost in the promising objective | Excluded: supply is fixed, and margins are declared net. |
| D7 | Margin and penalty values, and the sensitivity grid | Declared per SKU and fixed on dev streams by criteria that use no reader outcome; one predetermined sensitivity level per penalty (×0.5, ×2). |
| D8 | Order decomposition | Continuous quantities; orders per SKU and week, the customer lead-time distribution, and the horizon-start rule (5.5: exclude early requested weeks or a reported warm-up) fixed on dev streams. |
| D9 | Maximum lateness offered and treatment at the horizon end | No promise beyond `end_period`; terminal penalty per accepted unit not delivered. |
| D10 | Whether failed deliveries reveal holds to the policy | Main runs: no explicit notice; the policy sees only its physical position and its own commitment ledger. This mirrors replenishment, where only refusals produce notices. A predetermined sensitivity reveals the hold after a failed delivery (5.8). |
| D11 | Promising on the fresh split | Only if the promising protocol is frozen before any promising outcome on that split is computed. |
| D12 | Laya in the promising study | Include only if re-running Laya reproduces its recorded outputs exactly (no reply cache exists). |
| D13 | Shapley permutations and convergence check | 200 permutations, as in the paper; convergence checked against a second independent run of 200 with another seed. |
| D14 | Fresh-split size and model deciders | 24 or 48 streams, chosen by the run-time estimate and the stream-count calculation in Section 3, recorded in the protocol. No deciders. |
| D15 | Challenge streams, writers and verifier | Named in the challenge protocol before writing starts. Amended 2026-10-27, before writing: writers and verifier are anonymous and identified only by code (W1, W2, …; V1), with the key held by the authors ([challenge/README.md](challenge/README.md)). |

## 3. WP1: prospectively specified fresh-stream replication

**Purpose.** Test whether the reader-cost findings (RQ2–RQ3) hold on fresh worlds and streams
generated after the protocol is fixed. The replication does not test RQ1, because LLM deciders are
excluded; RQ1 rests on the pilot alone. The existing pilot study stays exploratory; the replication
does not make it confirmatory after the fact.

**Sequencing.** N1 is built first. The pilot export gate (5.9) runs right after N1 is built and before
the freeze. The gate is N1's real test, and N1 goes into the frozen tag only after it passes.

**Freeze before generating the final split:**
- prompts, examples, constrained schemas, typed questions and option labels
- regex rules, the span parser, the SOP file (`sop-v2`) and the approver
- update logic, the planner and its safety factor (0.3, from `local-readers.yaml`), the horizon
- alignment and error taxonomy, the repair and attribution procedure and Shapley settings
- metrics and analysis scripts
- the number of streams, the conditions, and the exclusion and failure rules below

Debug only on the dev split (seed 101, `data/v2/dev`, `data/v2/dev-para`). Any change to the protocol
after final outcomes have been seen makes the affected analysis exploratory, and it is labelled so.

**Provenance.**
- Freeze code, configs and the protocol document in one clean commit and give it an annotated tag
  (e.g. `icores-replication-freeze`). The protocol names the tag, not its own hash.
- Run every final run from that tag's checkout (a detached checkout or worktree), so `meta.json` shows
  that commit and `git_dirty: false`.
- Later documentation-only commits record the hash and the run times; they never change the frozen
  files.
- Name it **"a prospectively specified fresh-stream replication"**. A git commit is private if the
  repository is private and can be rewritten, so it is not an independent registration. Use the word
  "preregistration" only if the protocol is uploaded, before generation, to an independent timestamped
  registry (e.g. OSF); then state where, when, and whether it is public or embargoed.

**Split.** A new benchmark config identical to `configs/benchmark/pilot.yaml` except `name` and a seed
not in {7, 31, 101, 211}. The seed is written in the protocol. Generate once and never regenerate.
Paraphrase with the same layer and model (Phi-4). The paraphrased text keeps the parser-acceptance
filter; WP2 addresses that.

**Run-time estimate.** Estimate wall-clock time per stream from the existing `icores-final-*` run
folders, including solver, single-repair and Shapley time, not only `model_seconds` and request counts:
- The `seconds` field in `results.jsonl` covers the main run only; repairs and Shapley are not in it.
- Use each step's start and end times in `runs/<name>.log` (written by `scripts/icores-final.sh`) or the
  file times of `results.jsonl`.
- Use the `-r3-shapley` run for the cost of 200 permutations.
- Add Phi-4 paraphrase time from the paraphrase logs.

**Stream count (24 or 48).** Take the pilot's per-stream free-minus-constrained cost differences
(paraphrased reader run). For n = 24 and n = 48:
1. Resample n streams with replacement from the pilot differences, treating the pilot effect as real.
2. Compute the 95% bootstrap lower bound as in the primary analysis.
3. Repeat at least 2,000 times and report the share of repetitions whose lower bound is above 0.

Choose the smaller n whose share is at least 0.8, if the run-time estimate fits the run schedule. If neither reaches 0.8, take 48 if it fits, otherwise 24, and state the low chance in the
protocol. The pilot estimate is optimistic (it was selected after inspection), so this chance is an
upper bound. Record the calculation, its inputs and the result in the protocol before generating the
split.

**Run budget:**
- **Minimum:** 24 streams on paraphrased text. Conditions: oracle, no update, regex, Qwen free JSON,
  Qwen constrained JSON, Llama constrained JSON, Qwen typed questions, Laya typed questions.
  Single repairs and 200 Shapley permutations.
- **Expanded:** 48 streams (per the calculation above), and/or template text as well.
- LLM deciders are excluded (D14), so RQ1 is out of the replication's scope.

**Analysis.**
- The replication unit is the stream/world, not the message.
- **Primary comparison:** Qwen free-form JSON against Qwen constrained JSON. The quantity is the mean
  per-stream paired difference in cost (equal to the difference in R_interp, since both share the
  same oracle reference), free minus constrained. The pilot direction is free > constrained. Report a
  two-sided 95% percentile bootstrap interval over streams (10,000 resamples, fixed seed). The pilot
  direction counts as replicated if the lower bound is above 0. Also report the median and the number
  of streams where free costs more. With one primary comparison, no multiplicity adjustment is needed.
- **Secondary, descriptive** (estimates with unadjusted 95% intervals, no confirmatory claims):
  - each reader's R_interp and its difference from the no-update baseline; there is no requirement
    that every reader beats no update
  - F1 against cost rank agreement (Kendall τ with a stream-bootstrap interval)
  - rank intervals for readers from the bootstrap; there is no fixed percentage threshold for calling
    a ranking
  - concentration (Section 5.10 definition) for readers with at least 30 errors
  - positive Shapley shares by delta type and error kind
- **Failures and exclusions:**
  - Pairs in `failed.json` are rerun once with `--resume`; pairs still failing are reported as missing.
  - Paired comparisons use streams where both conditions finished, with the count reported.
  - No stream is dropped for its outcome.
  - Capped or malformed replies count as reader output.
  - Paraphrase fallbacks to template text are counted from `Stream.provenance`.
  - `scripts/scan_run.py` runs on every run folder and its findings are reported. A harness defect
    found after outcomes are seen is fixed, and the affected analysis is labelled exploratory.
- **Run-folder export:** N1 must be in the frozen commit, so the replication records completed deltas
  for later offline use.

## 4. WP2: unfiltered language challenge

**Purpose.** Measure reading and compiling on wording the span parser was not built for, keeping
every rewrite.

**Material:**
- 3–4 pilot streams, about 60 messages, chosen and listed before writing so they cover hold, release,
  lead-time, supplier-status, capacity, demand and price messages and no-change messages. Pilot streams
  give paired comparisons with the template and paraphrased results.
- Each writer gets a fact sheet per message: sender, the change in plain words (type, entity name,
  value, window), or "no change", plus any distractor to include.
- The rewrites must cover:
  - negation and cancellation
  - corrections of earlier notices
  - dates written in varied ways
  - relevant and irrelevant numbers in the same message
  - unfamiliar sentence structures
  - no-change messages

**Rules:**
- Writers do not use the parser or reader output as a criterion; the parser is not run on challenge
  text before the texts are frozen.
- A second person checks every rewrite against its fact sheet and marks the evidence sentence(s).
  Disagreements are fixed in the text before any reader runs, and the check is recorded.
- Every finalized rewrite is kept, including those the parser cannot process.
- The writers' independence is recorded honestly: who wrote what (by anonymous code, D15), and
  whether they had read the parser or templates.
- No multi-delta messages in the main set: the typed reader asks for at most one delta per message.
  Any multi-delta messages form a separately labelled capability stress test.
- Gold spans are re-anchored to the marked evidence (N8). Without that, even the oracle can fail the
  span check (`parser.py:119-120`). Gold op, value and window do not change.

**Scoring**, reported separately:
1. **Reader identification:** type and entity correct.
2. **Parser completion:** no parse failure (`policy.failures` by slot), and value and window correct
   for identified deltas.
3. **Applied deltas and state:** correct deltas admitted (exact matches not in `rejected_correct`);
   optionally ŝ against s* cell agreement.
4. **Downstream loss:** R_interp under replenishment, and under promising once WP3 runs.

**Parser-isolation condition (N9):** gold type and entity with the independently marked evidence span
and nothing else, so the parser fills op, value and window. The architecture supports it (`Proposal`
slots are optional; `SpanParser.complete`). It measures what the deterministic layer loses on
unfamiliar wording, separately from reader errors.

## 5. WP3: customer-order promising (required)

### 5.1 Status

- **Minimum version:** pilot streams (paraphrased, as for RQ3) and recorded reader outputs. It is
  exploratory, because those outputs and their replenishment outcomes have already been seen.
- **Optional:** the fresh split, under D11.
- Parameters are fixed on dev streams before any pilot promising run, by criteria that use no reader
  outcome (e.g. the gold-reference acceptance rate and on-time rate fall in sensible ranges).

### 5.2 Fixed physical supply

- **How the trajectory is obtained:** the gold-update replenishment run, i.e. gold deltas through the
  same front end with the configured planner (safety factor 0.3), which is the run behind
  `StreamContext.planner_cost`. Its per-week `PeriodRecord.ordered` lines are replayed through
  `PlannedPolicy`; acceptance and arrivals follow the truth, so receipts are identical for every
  promising condition.
- **What information it uses:** messages up to each week, read correctly; master data; its own
  position; the forecast inflated by the safety factor. It never uses realised future demand.
- **Why it is held fixed:** so that a promising reader's misreading cannot change physical supply.
  Promising cost then isolates the promising decision. This is an experimental control, not an
  end-to-end deployable policy, and the paper says so.
- **Gate:** replaying the recorded lines reproduces the gold-update replenishment cost exactly.
- **Consequences remain:** promising decisions still have consequences through acceptance, binding
  promises and delivery timing.

### 5.3 Event order per week t

1. **Physical receipts:** in-transit shipments whose true arrival is ≤ t enter free stock; lots whose
   arrival is ≤ t are present. This is the same rule as `InventorySimulator.run`.
2. **Messages:** messages of week t go through the front end; ŝ_t is updated.
3. **Fixed orders:** this week's replenishment lines are placed and accepted per the truth. Accepted
   lines join the open orders with their placement week.
4. **New customer orders** arriving in week t are revealed.
5. **Promising decision** for the new orders, using ŝ_t, the observed position (free stock, remaining
   lot quantities, open orders with placement week and quantity, but not true arrival or true hold
   status) and the existing commitment ledger.
6. **True execution** of commitments due by t (5.7).
7. **Accounting.**

Definitions:
- *Order placement:* a replenishment line is sent.
- *Supplier acceptance:* the quantity the true capacity and status admit.
- *In transit:* accepted, not yet arrived.
- *Physical receipt:* arrival at the true lead time in force at placement.
- *Available stock:* free stock plus arrived lots truly released.
- *Held stock:* arrived lot remainder under a true hold.
- *Existing commitments:* accepted customer orders not yet fully delivered.

How observation corrects the projection:
- **Timing:** at the start of each week, the observed free stock and lot remainders replace the
  projection.
- **Overdue open orders:** an order that has not arrived when ŝ said it would is projected for next
  week (`_expected_arrivals`).
- **Holds:** the policy is not told about holds (D10).
- **Consistency:** the same rules apply in every condition, including the gold-update reference.

### 5.4 Which delta types can affect promising (under D1 = ATP)

| Delta type | Future POs | Accepted open POs / in transit | Received stock | Promising projection (ŝ) | True execution | Relevant? |
| --- | --- | --- | --- | --- | --- | --- |
| `quality_hold` | — | — | Held lot remainder unusable | Lot counted as unavailable while ŝ holds it | The true hold freezes the lot | **Yes** |
| `quality_release` | — | — | Lot usable again | Lot counted from the release week | True release | **Yes** (SOP-003 also ties it to the hold read earlier) |
| `lead_time_change` | Fixed by trajectory | True arrival fixed at placement | — | Projected arrival of open orders placed while the change is in force | Receipts per truth (fixed) | **Yes** |
| `capacity_change` | Fixed by trajectory | No effect on accepted shipments | — | Not used under ATP | Supply fixed | **No, by design** |
| `supplier_status` | Fixed by trajectory | Accepted shipments still arrive (the simulator does not cancel) | — | Not used under ATP | Supply fixed | **No, by design** |
| `demand_signal` | Fixed by trajectory | — | — | Not used without reservation (D3) | Shapes realised demand, hence order volumes, identically in every condition | **No, by design** |
| `price_change` | Fixed by trajectory | — | — | Not used | Margins are declared per SKU | **No** |

- **Not a finding:** errors whose predicted and gold deltas both touch only non-relevant attributes
  have zero promising effect by construction. The paper reports this as a design consequence, not as
  an empirical result. No influence is added to make every type matter.
- **Cross-type errors can still matter.** A wrong-type error that writes a spurious lead-time or hold
  delta, or an SOP interaction between a hold and its release, can carry promising cost even when the
  gold type is not relevant.
- **Direction:** large differences in type shares between the policies are expected. Supplier status
  carried 55% of replenishment cost in the pilot. They are explained by this table, and no directional
  hypothesis is stated beyond it.

### 5.5 Customer orders (N3)

- **Aggregate demand preserved:** for each SKU k and requested week r in the horizon, split the
  realised demand D(k, r) (from `sample_demand`, the same path as replenishment) into a fixed number
  of orders using shares from a Dirichlet draw. Quantities stay continuous, so totals equal D(k, r) up
  to floating-point error. No rounding.
- **Arrival week:** a = r − L, with the customer lead time L drawn from a small declared distribution
  (D8). Requested week = r.
- **Horizon start:** clipping with a = max(`start_period`, r − L) would pile orders for early weeks
  onto `start_period`. Decide on dev streams (D8) between:
  - excluding requested weeks below `start_period` + L_max, so those weeks generate no customer
    orders and "aggregate demand preserved" applies to the included weeks only; or
  - a warm-up period whose orders run normally but are reported separately and left out of the main
    metrics.
- **Order fields:** each order has id, arrival week, SKU, quantity, requested week and declared unit
  margin m_k.
- **Separate random generator:** seeded from the stream seed and a fixed label. Messages, gold deltas
  and the demand path are untouched.
- **Information:** an order is visible to the policy only from its arrival week. The policy never sees
  future orders, realised future demand or demand noise.

### 5.6 Promise decision (N4)

The smallest defensible model: a weekly MILP over the new orders of week t, with planning weeks t to
`end_period` (or the solver's horizon if shorter, decided in D9).

- **Decisions:** accept or reject each new order; choose one promise week p in [r_o, min(r_o + L_max,
  `end_period`)].
- **Supply:** projected availability per SKU and week from D1 (free stock, per-lot availability with
  holds from ŝ as in `_lot_draws`, open orders at `_expected_arrivals`). Existing commitments are
  fixed requirements at their promised weeks. Cumulative projected availability must cover all
  commitments due by each week.
- **Existing commitments are binding (D5).** If new information makes them infeasible in projection
  (e.g. a newly read hold), a penalised slack keeps the model solvable. Infeasible commitments are
  logged as at-risk, and no new order may use their supply.
- **Objective:** maximise Σ (m_k − ℓ·(p − r_o))·q_o over accepted orders. Rejection earns nothing.
  Since every margin is positive, rejecting everything is never optimal when supply exists.
- **No on-time week feasible:** accept late within L_max if margin net of lateness is positive,
  otherwise reject. No promise beyond the horizon.
- **Tie-breaking:** a deterministic small penalty by order index.
- **Solver status:** recorded per week. If HiGHS returns no solution within the limit, a deterministic
  greedy rule (earliest feasible week in order of arrival and id) applies in every condition; each use
  is counted and reported.

### 5.7 True execution (N5)

- **Each week, in order:** usable stock = free stock + arrived lots released in the truth. Commitments
  due by t that are not fully delivered are served in order of promised week, then arrival week, then
  order id.
- **Stock draw:** released lots first, by lot id, then free stock (the simulator's rule).
- **Partial deliveries** are allowed. Nothing is delivered before its promised week.
- **Broken promises:** an order not fully delivered by its promised week stays outstanding and is
  delivered late when stock allows. There is no re-promising and no cancellation.
- **What the policy observes:** its position and ledger at the start of each week.
- **Horizon end:** accepted units not delivered by `end_period` incur the terminal penalty and earn
  no margin.
- The same rule applies in every condition.

### 5.8 Objective and metrics

**Realised contribution, one formula for every condition:**

Π = Σ m_k · units delivered by `end_period` − ℓ · Σ unit-weeks late − b · units not delivered by their
promised week − τ · units accepted but undelivered at `end_period`

- **Stacking is intentional:** b prices the broken commitment once, ℓ prices its duration, and τ
  prices the final failure.
- **Interpretation loss:** reported as Π(gold-update reference) − Π(reader), so positive means loss.
  It is the same quantity as a cost difference with cost = −Π.

**Metrics with denominators:**
- orders and units arriving
- accepted and rejected orders and units, out of orders arriving
- on-time orders, out of accepted orders, reported with the accepted volume
- broken promises and late orders, out of accepted orders
- units delivered, out of accepted units
- units undelivered at the horizon end
- solver fallbacks and at-risk commitments

**"Unnecessary rejections" is not reported per order:** rejected orders compete for the same supply.
Instead, report rejected and accepted volume against the gold-update reference run of the same policy,
which is the joint counterfactual.

**Parameters:** margins and penalties are synthetic, declared values, not company costs.

**Predetermined sensitivities.** Neither needs new reader outputs. The main runs keep the main
settings.
1. **Penalties:** b and ℓ at ×0.5 and ×2.
2. **Hold revealed (D10):** when a due commitment of week t is not fully delivered while a truly held,
   arrived lot of that SKU has stock left, the policy is told from week t + 1 that the lot is held.
   It uses that notice in place of ŝ's hold status until ŝ records a hold-status delta for that lot
   after the notice (mirroring `orders.supplier_status`).

If sensitivity 2 is not run, hold-error promising costs are labelled as an upper bound.

**Negative losses.** The weekly MILP is myopic, so the gold-update reference is not optimal and a
reader's interpretation loss can be negative. Report the number of stream–reader pairs with negative
interpretation loss.

### 5.9 Reuse of recorded reader outputs

- **Preferred path:** replay stored completed deltas (N1) through `ReplayInterpreter` and the
  unchanged front end. Verified preconditions: messages unchanged (same stream files), reader-visible
  state unchanged (2.2), readers never receive orders, promising costs or decisions (2.1). Repair
  replay already works on completed deltas (`score_compiled.cost_with`).
- **Completed deltas do not exist on disk for the pilot runs yet:**
  1. Add N1.
  2. On the local machine, re-run the reported pilot reader conditions with the reply cache and Ollama
     running, **without single repairs or Shapley** (attribution off), to export completed deltas.
  3. **Local gate:**
     - zero server requests (`model_requests` = 0, all replies counted as cache hits);
     - every row's cost and F1 identical to the recorded `results.jsonl`;
     - error identity identical to the recorded `errors.jsonl`: kind, delta type, `predicted_id` and
       `gold_id`.

     `delta_c` and Shapley values are rechecked on a second machine from the exported deltas against the
     recorded values.
  4. Laya has no reply cache. Its re-run does not block the WP1 freeze. It runs on the local machine after
     the WP1 runs, if time allows, and Laya is included in WP3 only if its outputs reproduce the
     recorded run exactly. Otherwise Laya is excluded from WP3 (D12).
  5. **Fallback if the cache path fails** (e.g. a changed digest): a replay client that serves replies
     from the run's `requests-<model>.jsonl` by request-body hash, checked by the same gate.
- **Sequencing:** the gate runs right after N1 is built and before the WP1 freeze (Section 3).
- **After the export gate passes:** promising runs, repairs and Shapley attribution run on a second machine
  from the exported files, with no model calls and no Ollama. Until the gate passes, "no new model
  calls" is a goal, not a fact.

### 5.10 Interpretation loss and attribution

- **Loss:** the promising policy with the reader's ŝ against the same policy with the gold-update ŝ,
  both executed on the same truth, supply and orders. Promising and replenishment numbers are never
  compared as one objective.
- **Conditional single-repair effects** price individual interventions. **Shapley values** (200
  permutations) allocate the loss additively. Both use the existing repair definitions.
- **Concentration** keeps the manuscript's definition:
  - errors per reader pooled over streams and sorted by Shapley value
  - top max(1, round(0.1·n)) errors
  - share = their summed Shapley value / summed positive Shapley value
  - reported only for readers with at least 30 errors (below 30: list counts)
  - zero positive attributed cost: report "no costly errors" and no share
  - negative values stay in the Shapley sums (which equal the loss) and are excluded from the
    positive denominator
- **Negative losses:** report the number of stream–reader pairs with negative interpretation loss
  (5.8). Negative Shapley values are kept in the sums.
- **Checks:**
  - every stream–reader pair has at most `max_errors` (200) errors, otherwise efficiency fails
  - a second Shapley run with another seed, comparing pooled type shares
  - repairing all errors reproduces the gold-update reference exactly for every stream and reader

### 5.11 Cross-policy analysis (RQ4)

Within each policy:
- reader interpretation losses: mean, median, the number of streams with non-zero loss, and
  stream-bootstrap intervals
- positive Shapley shares by delta type and error kind

Across policies:
- **Reader rank association:** Kendall τ between the two policies' reader means, descriptive, with a
  stream-bootstrap interval where feasible (with 6–8 readers the interval is wide)
- **Paired error examples:** the error ids are identical across policies (2.2), so plot each error's
  ΔC under replenishment against its ΔC under promising. Show examples where the same reading error is
  costly under one policy and free under the other.

Not done:
- no pooling of raw monetary values across policies
- no directional hypotheses about type shares beyond the relevance table
- a ranking reversal or concentration is not a required success outcome

## 6. Tests and validation gates

Unit and integration tests for the new code (`make check`; tests never call a network service):

- **Execution correctness:** no over-allocation of true stock in any week; held stock never used;
  identical inputs give identical decisions and execution (determinism).
- **Information:** no access to future customer orders (the policy input built at week t contains only
  orders with arrival ≤ t); existing commitments persist across weeks and are never revised or
  dropped; horizon-end obligations are counted.
- **Controls:** every reader condition in a stream gets byte-identical supply ledgers; replaying the
  fixed trajectory reproduces the gold-update replenishment cost.
- **Gold-repair gate:** repairing all errors reproduces the gold-update promising reference exactly.
- **Hand-worked cases with known correct promises:** excess supply (accept all on time), scarce supply
  (best-margin subset), delayed receipt (late promise), hold then release, and a misleading message
  (a spurious hold causing a rejection that the gold reference accepts). An author works out the
  expected promises and execution by hand and writes them into the tests before the implementation
  runs. They are never generated from the implementation.
- **Accounting reconciliation:** delivered + undelivered = accepted units; Π recomputed from the
  per-order ledger equals the reported value.
- **Offline replay:** a client that raises on any request proves zero model calls (N1 path).
- **Reproduction:** the N1 export reproduces the recorded pilot results (5.9).
- **WP2:** gold spans re-anchored inside the rewritten text; the oracle on challenge text reproduces
  the oracle cost.

Passing tests is not evidence of business validity. Before reporting, inspect representative
trajectories by hand: at least one stream per relevant delta type, the gold reference against two
readers, checking the order ledger, projections and execution week by week. Record what was
inspected.
