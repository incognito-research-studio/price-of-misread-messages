# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Changes to the delta schema or to a registered SOP
file are always listed, because they change what results mean.

## [Unreleased]

### Added (2026-10-28)

- The order-promising runs behind Table 7 of the paper, `runs/icores-promising-v1` and
  `runs/icores-promising-v1-surge`, scored with no model call from
  `runs/icores-export-merged-para` (the five-reader export plus Laya's, oracle records kept
  once). They reproduce the paper's values and pass the export and repaired-gap gates.
- `FREEZE.md`, the record of the freeze tags.
- `scripts/error_price_figure.py`: the paper's Fig. 3, each pilot error's single-repair effect
  under replenishment against surge promising. Its legend gives each colour's error count.
- `scripts/shapley_seed_replay.py` and `scripts/shapley_seed_compare.py`, with
  `runs/icores-shapley-seed1-pilot` and `runs/icores-shapley-seed1-fresh`: the pilot and fresh
  reader runs re-attributed with Shapley seed 1 (no model call). Costs and errors are unchanged;
  each reader's costliest-tenth share moves by at most 0.8 percentage points and its costliest
  error stays the same.

### Changed (2026-10-28, Fig. 2)

- `scripts/concentration_figure.py` takes an optional second run; the paper's Fig. 2 now shows the
  fresh streams (dashed) beside the pilot (solid) for regex, free JSON and Laya.

### Fixed (2026-10-28, paper build)

- `make count` in `paper/` counted bytes, not characters, under the C locale (curly quotes,
  dashes and math letters are multibyte), overstating the length by about 1,560. It now counts in
  UTF-8.

### Changed (2026-10-28)

- README, CITATION.cff, pyproject.toml and LICENSE use the paper's title and terms. The README
  maps the code's names to the paper's, lists the runs and scripts behind each table and figure,
  and gives the model-free checks. Prose docs say "fixed before evaluation" where they said
  "pre-registered"; configs, code and frozen files are unchanged.

### Fixed (2026-10-27, after the WP2 challenge run's outcomes were seen)

- `apply_repairs` gives a repaired gold delta a fresh id when the reader already holds a delta
  with that id. Only the gold-span reader, which keeps gold ids, can hit this: on challenge
  stream pilot-000 a repair re-inserted gold delta `pilot-000-m0010#g0` beside the reader's own
  copy, and the state refused the duplicate (the run's only failed pair, also after one
  `--resume`). Readers that mint their own ids are unaffected: `replay_check.py` reproduces
  `icores-final-readers-para-r3-shapley` (144 pairs, 112 errors with Shapley values, no problem).
  The rerun gold-span pair on pilot-000 is labelled exploratory.

### Changed

- WP2 writers and verifier are anonymous (D15, amended before writing started): every record
  identifies them by code (W1, W2, ...; V1); `docs/research/challenge/README.md` states the rule
  and how the texts are collected.
- WP3 results recorded (`docs/research/wp3-results.md`): the pilot readers replayed under order
  promising (v1 and surge) with every gate passing (replay check, repaired gap, no fallbacks), a
  second Shapley seed, and the predetermined sensitivities (penalties x0.5 and x2, hold revealed;
  parameter files `configs/promising/promising-v1-surge-*.yaml`). Reader rankings do not carry
  over from replenishment (Kendall's tau -0.20 under surge, wide interval): of 82 errors costly
  under replenishment, 5 are costly under surge promising, because promising holds supply fixed.
- WP1 results recorded (`docs/research/wp1-results.md`): on 48 fresh streams the primary
  free-minus-constrained difference is 7,916.1 per stream, 95% interval [3,863.1, 12,548.8], so
  the pilot direction replicated. The Laya export reproduced its pilot outputs exactly and enters
  WP3 under D12; runbook C4 now states D12's criterion instead of asking for a gate that a reader
  without a reply cache cannot pass.
- WP1 re-frozen before generation (`docs/research/replication-protocol.md`, "Amendment before
  generation"): the fresh split has 48 streams (`configs/benchmark/confirm.yaml`; streams 0-23
  are unchanged), with a time fallback to the first 24 fixed in advance. The run-time table now
  uses the measured reply time (about 2.1 s, not 15.7 s). `scripts/replication_analysis.py`
  counts the paraphrase `fallback` list instead of crashing on it, and takes `--first N`.
  `scripts/wp1-overnight.sh` runs only from the new tag `icores-replication-freeze-v2`; the
  first tag stays on de32e5e as the record of the superseded freeze.
- The `[laya]` extra pins `laya==0.4.1` again (Dependabot had moved it to 0.4.14). The Laya
  adapter loads the checkpoint revision its installed release pins, so the version decides which
  reader runs; every ICORES run, the pilot included, uses 0.4.1. Upgrade only after the revision
  outputs are final.

### Added

- `scripts/buffer_ablation.py` (post hoc, exploratory; 2026-10-27): reprices the recorded reader
  output under smaller replenishment buffers with single-repair attribution and runs the
  model-free check, with no model call; a gate requires the run's own buffer to reproduce every
  recorded cost (passed on `icores-replication-para`). At beta 0 against 0.3, model-free costly
  misses rose for capacity (23% to 40%) and demand (29% to 56%), while supplier-status errors
  still carried 72% of reader error cost.
- `scripts/basestock_baseline.py` (post hoc; 2026-10-27): a periodic-review order-up-to policy
  deciding from the oracle state, its safety factor tuned on the development split (0.3), as a
  non-LLM anchor for RQ1. On the 24 pilot streams it cost 1.20 times the planner; every model
  decider cost more than it on 24 of 24 streams.
- `scripts/snap_ablation.py` (post hoc, exploratory, outside every frozen protocol; 2026-10-27):
  replays the free-JSON reader's own logged replies with the constrained schema's restrictions
  applied after generation (evidence snapped to a message sentence, entity and type to valid
  options, `supersedes` to deltas in force), with no model call. A gate requires the unsnapped
  replay to reproduce the recorded free-JSON cost on every stream (passed on
  `icores-replication-para` and `icores-final-readers-para-r3-shapley`). Snapping only the
  misquoted evidence closed 102% of the free-minus-constrained gap on the 48 fresh streams
  (remaining difference -177, interval -621 to 284) and 99% on the pilot; snapping entities and
  types closed none. Reported in the paper (Section 5.2) as exploratory.
- WP2, the unfiltered language challenge (docs/research/challenge/README.md):
  `cdd.benchmark.challenge` loads the hand-written `rewrites.yaml` and imports it into the pilot
  streams, replacing each message text and re-anchoring each gold delta's span to the verified
  evidence; gold types, entities, values and windows are unchanged, and no message is dropped or
  tested against the parser. The import refuses, listing every problem, unless each message has
  one verified entry with a writer and a verifier and its evidence occurs exactly once (or is
  empty for a message without a change). `scripts/import_challenge.py` writes the streams and a
  separate manifest of their hashes. New interpreter kind `gold_span` (`GoldSpanInterpreter`):
  the gold type, entity and span, with value and window left to the span parser, so its losses are
  the parser's alone. Experiment config `configs/experiments/icores-challenge.yaml` and the
  stage-by-stage analysis `scripts/challenge_analysis.py`.
- WP3 report `scripts/promising_report.py`: reader losses per policy with stream-bootstrap
  intervals and negative-loss counts, volume against the gold-update reference, solver fallbacks
  and at-risk units, the repaired-gap and max_errors checks, concentration and positive Shapley
  shares, Kendall's tau between the policies' reader means, and the paired error table.
- WP1, the fresh-stream replication (`docs/research/replication-protocol.md`): split config
  `configs/benchmark/confirm.yaml` (pilot settings, seed 2027, 24 streams by the protocol's
  rule), experiment config `configs/experiments/icores-replication.yaml` (the paper's readers plus
  the no-update baseline, its own reply cache), the frozen analysis
  `scripts/replication_analysis.py` and the unattended run `scripts/wp1-overnight.sh`.
- Order promising, the second downstream decision (`cdd.promising`;
  docs/research/icores-revision-scope.md, Section 5). Physical supply is fixed to the planner's
  gold-state orders (`reference_supply`, `FixedPlanDecider`); customer orders are split from the
  realised demand path with their own generator (`generate_orders`); a weekly MILP accepts and
  promises orders against the supply ŝ projects (free stock, lots ŝ shows released, open orders
  at ŝ's lead time), with binding commitments; execution serves due commitments from true usable
  stock and values them per order (margin, lateness after the request, broken promises, terminal
  shortfall). `PromisingContext`, `score_promising` and `promising_experiment` score recorded
  readers with single repairs and Shapley values and report the all-repaired gap;
  `scripts/promising_run.py` runs a recorded run, `scripts/promising_dev_check.py` the model-free
  dev check. Parameter files `configs/promising/promising-v1.yaml` and `promising-v1-surge.yaml`
  (orders 1.3 x demand), frozen 2026-10-27 after the dev check
  (docs/research/promising-dev-check.md): under v1 no single missed gold delta changes the
  promising value on the dev split. If the promise solver returns no solution, that week's new
  orders are rejected and the fallback is counted (the scope's greedy rule was not built).
  `_Model.solve` takes an optional `mip_rel_gap`; the replenishment solver keeps the default.
- Recorded reader output (`cdd.recorded`): every compiled (stream, condition) pair appends its
  completed deltas, parse failures and admissibility verdicts to `proposals.jsonl` in the run
  folder. `cdd.experiments.replay_experiment` scores a config's solver-decided conditions from
  those records with no model call, reproducing costs, errors, single repairs and Shapley samples
  of the recorded run. `run_condition` takes an optional `interpreter` and returns the record.
  `configs/experiments/icores-export-readers.yaml` and `icores-export-laya.yaml` export the pilot
  readers' output; `scripts/check_export.py` (export gate on the local machine) and
  `scripts/replay_check.py` (single repairs and Shapley values from the exported deltas) check it
  against the recorded run (docs/research/icores-revision-scope.md, 5.9).
- `interpreter: none`, the no-update baseline: the planner ignores every message and plans on
  the first week's master data, learning of a suspension only from a refused order.
- ICORES revision (2026-10-24): `configs/experiments/icores-no-update.yaml`,
  `icores-readers-shapley.yaml` (the final reader run again with 200 Shapley permutations; replies
  from the cache reproduce every r2 cost) and `icores-buffer-{0.1,0.2,0.4,0.5}.yaml` (buffer
  sensitivity). `scripts/revision_numbers.py` prints their numbers and
  `scripts/concentration_figure.py` writes the paper's concentration figure from a run's
  `errors.jsonl`. `scripts/paper_numbers.py` now runs its report under `main()` so that its
  helpers can be imported; its output is unchanged.
- Ollama adapter (`cdd.adapters.ollama`, standard library only) for local models, implementing
  both the text and the typed-question interface, and `configs/experiments/local-ollama.yaml`.
- `constrained: true` on a condition sends a JSON Schema with each request: condition (c) for
  the JSON interpreter, and constrained orders for the end-to-end decider.
- Tuned benchmark x* (`benchmark: tuned`; the default until the planner benchmark below): the best solver variant on the oracle
  state over `tuned_safety_factors`; the hindsight optimum is reported as the value of perfect
  information. `solver.safety_factor` sets the solver's own buffer.
- Condition (i) for LLMs (`decider: llm_state`) and the hybrid condition (j)
  (`interpreter: hybrid`, `HybridInterpreter`).
- Layer ablations (`sop_gate`, `versioning` with `apply_overwrite`), the window-rule ablation
  (`window_rule`), and option labels for typed questions (`labels`: names, neutral, bare).
- `TypedQuestion.definitions`; the type question now defines every option.
- Metrics: `calibration_ece`, `model_requests`, `model_seconds`, `escalation_share`, and the
  benchmark fields `tuned_cost`, `tuned_safety_factor`, `hindsight_cost`,
  `value_of_perfect_information`.
- Laya adapter (`cdd.adapters.laya`, optional extra `[laya]` pinning `laya==0.4.1`) with
  pinned checkpoint revisions, `cdd laya-export` (fine-tuning rows that match the questions
  asked at inference) and `cdd laya-finetune` (fine-tune and recalibrate, optional held-out
  `--eval`).
- `cdd tune-solver`: chooses the solver's safety factor on development streams.
- Out-of-schema slice: `out_of_schema_rate` adds yield shortfalls that no delta type expresses
  (`Stream.shocks`), applied by the simulator and known to the hindsight benchmark only.
- Benchmark splits: `train`, `dev` and `out-of-schema` generator configs.
- Plug-in API `cdd.evaluate.evaluate` and `report`, `docs/plug-in.md`, and
  `examples/evaluate_your_own_reader.py`.
- `meta.json` in every run folder: package and schema versions, code commit, dirty flag, time.
- Paraphrase layer (`cdd.benchmark.paraphrase`, `cdd paraphrase`): rewrites messages with a
  local model (Phi-4 by default) and keeps a rewrite only if names, numbers, weeks and dates
  survive and the span parser still recovers every gold delta; otherwise the template text
  stays. `Stream.provenance` records the model, prompt version and acceptance counts.
- `cdd run --streams` overrides the config's stream directory.
- Long runs survive failures: results are appended per (stream, condition) pair; a pair that
  raises goes to `failed.json` and the run continues; `cdd run --resume` reruns what is missing,
  and `cdd run` exits with status 1 when pairs failed. The Ollama client retries timeouts,
  dropped connections and HTTP 5xx twice.
- Request logs (`requests-<model>.jsonl`, on by default via `log_requests`) with stream and
  condition tags, `done_reason` and token counts; summary columns `model_retries`,
  `model_capped_replies` and `model_truncations`.
- `max_tokens` for the Ollama client (default 1024, sent as `num_predict`). A free-form Qwen 2.5
  reply that repeated itself for 6,600 tokens hit the request timeout and stopped a run; of
  3,330 other requests, none took longer than 25 seconds.
- Benchmark version b2 (`version` in the generator config, recorded in `Stream.provenance`),
  with calibration settings that default to their b1 values: `capacity_ratio`,
  `min_suppliers_per_sku` (random SKU assignment), `price_spread`, `lot_weeks` and
  `lot_arrival` (`Lot.arrival`), `capacity_weeks`, `cross_week_boundary`, `plain_spikes` and
  `last_message_week`. The b2 configs regenerate into `data/v2/`; b1 streams regenerate
  unchanged. `exclusive_windows` (b2: on) keeps two numeric changes to one entity from
  overlapping unless one revises the other: the gold compounded such pairs (17 in the b1
  pilot), though "down 15% in weeks 41-43" followed by "up 15% in weeks 42-43" reads as a
  correction.
- `cdd calibrate`: a model-free check that removes each gold delta in turn, replays the oracle
  pipeline and reports the cost of each miss by delta type. On the b1 pilot, 2% of capacity
  misses, 4% of hold misses and 0% of release misses cost anything. On the b2 pilot and dev
  splits, with the fixed planner, the share of misses that cost more than 100 is:
  capacity 21% and 9%, demand 20% and 28%, lead time 49% and 51%, holds 54% and 36%,
  releases 73% and 63%, supplier status 42% and 48%, price 29% and 21%. Supplier status
  carries 50% and 40% of the miss cost. The oracle run has no SOP rejection and no conflict
  drop, and fills 96% of demand on time (b1: 96%). The target of 30% costly misses for every
  type is not met for capacity and demand: the tuned 30% buffer absorbs most of those misses.
  Tighter worlds met it only with the oracle filling 70% of demand on time; the realistic
  setting was kept and the shortfall is reported (decision of 2026-10-23).
- SOP file `sop-v2`: SOP-004 checks the sender of every supplier-status change, not only
  suspensions (`sender_checked_statuses`); SOP-005 admits a supplier-status change only if its
  span names the supplier (`grounded_types`); and `type_owners` resolves a conflict in favour of
  the one delta whose sender normally issues its type. SOP-005 closes a path the per-type
  schema opened: a reader that took "QA has suspended Lot 7702" for a status change had to name
  a supplier, and so suspended one the message never mentions. It rejects none of the 71 gold
  status deltas in the b2 pilot and dev splits. Every gold status message comes from
  procurement, so sop-v1 accepted "active" read from an internal "orders on schedule" note.
- Refused orders teach the planner (`InventoryPosition.status_notices`): when a plant refuses
  an order because its supplier is suspended or restricted, later positions carry that status,
  and the solver uses it until a status delta for that supplier is recorded after the refusal.
  Model deciders see the notice in the position. Before, one missed suspension made the
  planner order from a plant that shipped nothing for the rest of the horizon (136,000 extra
  cost for one stream).
- Benchmark `planner` (now the default): x* is the configured solver on the oracle state. Its
  safety factor is tuned once on the dev split and fixed in the configs (on
  2026-10-23: `cdd tune-solver` on `data/v2/dev`, grid 0, 0.05, ..., 0.4, 0.5; best 0.3,
  mean oracle cost 84,967 against 108,745 at 0). An earlier b2 draft with one week of starting
  stock was tuned the same way (best 0.1) and dropped before any model run; the only pilot
  run before the factor was fixed was a model-free check of the runner (oracle and regex). Decision regret is zero for the solver and,
  for a model decider, its extra cost over the solver on the same state; the gap to the
  hindsight optimum is reported as the value of perfect information.
- Persistent reply cache (`reply_cache` in the experiment config, `cdd.adapters.cache`):
  replies stored on disk by request body and model digest, so a repeated request returns the
  first reply. Each model reader or decider is therefore one sample at temperature 0.
- `models.json` in every run folder: each client's model, Ollama digest and server version,
  or Laya checkpoint and revision.
- Metrics `conflicts_dropped`, `invalid_order_lines` and `model_cache_hits`; summary columns
  `median_*`, `n_nonzero_r_total` and `ratio_interp_share` (ΣR_interp / ΣR_total).
- Model deciders are told the objective, the last week of the simulation (orders arriving later
  are paid for and never used), that demand varies about 20% around the forecast, the holding
  and backlog costs and the physical rules;
  the stream deciders also get the master data as of the first week (capacities, demand, lead
  times, prices, supply lines, lots). The state rendering lists each plant's supplier and the SKUs it makes, so a state decider can
  link a plant to its supplier's prices.
- `scripts/`: `scan_run.py` (run-folder checks) and `paper_numbers.py` (the paper's tables, with
  medians, non-zero counts and paired bootstrap intervals).

### Changed

- Each (stream, condition) pair draws from its own random generator, so sampled Shapley values
  no longer depend on run order. Earlier runs used one generator for the whole run; their
  Shapley samples (none in the shipped configs) are not reproduced exactly.
- The span parser reads more window wordings: a connector after the start word ("starting in
  week 38", "effective from wk42"), "week 41 onwards", and ordinal days ("November 5th").
  Template text parses as before.
- The span parser rejects entity ids that are not in the world (slot `entity`), so they count
  as schema failures instead of mis-scoped deltas.
- Alignment compares windows inside the simulated horizon.
- Windows are parser-first: the span parser reads the window from the text and uses typed
  window answers only when the text states none.
- The default benchmark changed from hindsight to tuned, so `r_total` and `r_dec` are not
  comparable with earlier runs. Its default grid now reaches a 50% buffer, because the best
  development-stream factor (30%) sat at the edge of the old grid.
- The simulator's open orders show the accepted quantity; any out-of-schema shortfall appears
  only on arrival.
- The JSON interpreter prompt lists entities as `id = name` and restricts `supersedes` to
  messages that explicitly revise an earlier notice. Prompts changed after inspecting one pilot
  stream; freeze them before any reported run.
- Typed-question readers no longer get window questions for holds, releases and supplier-status
  changes. When such a message states no window, the change holds from the message's week until
  further notice, as for the JSON and regex readers. Asking let typed Qwen answer "wk36 to wk36"
  for "on hold pending CoA": on the 24 pilot template streams, 44 of its 49 errors were such
  one-week windows, and the same errors carried 54% of zero-shot Laya's interpretation cost.
  Typed-reader results before this change are not comparable.
- The type guide's examples (shown to the JSON interpreter and used as typed-question option
  definitions) now name entities that no generated world contains. With the Table 1 names,
  Qwen 2.5 quoted an example as its evidence for 16 of 121 proposals on template text and 28
  of 122 on paraphrased text (type and entity right, evidence not in the message, so the delta
  was dropped). The JSON prompt now says to copy evidence from the message only.
- Condition (c) constrains more than formatting: evidence must be one of the message's
  sentences and `supersedes` may list only deltas in force. Before, constrained and free Qwen
  2.5 replies differed only in trailing text, so (b) and (c) gave identical results.
- The typed-model type question now carries the same type descriptions as the JSON
  interpreter's prompt. Before, it listed bare labels, which made the typed condition misread
  every supplier reinstatement in the six-stream local run.

### Fixed (2026-10-23, from two code audits of the first ICORES run)

- A `supersedes` reference retracts a delta only on the same entity and attribute. A JSON
  reader that linked a lot release to a lane delay deleted the delay (ΔC 17,434 in one pilot
  stream); about a quarter of JSON readers' links pointed at another entity's delta.
- A delta with no `supersedes` reference that restates the same cell and window as an earlier
  delta replaces it instead of compounding with it (0.8 × 0.6 for a revised capacity cut).
- A supersession mismatch counts as a wrong value only when it changes a state cell at or
  after the message's week. Releases that also named their hold were scored as wrong values
  (32-41 per JSON reader on the pilot), and typed readers, which cannot link, never were.
- Repairs of revisions now name the reader's aligned delta, so the single-repair ΔC of a
  revision error can be non-zero; before, the repair's reference named a gold id the
  interpreted state never held.
- The constrained JSON schema lets each delta type name only entities of its kind. Llama 3.1
  gave offer ids for 50 demand signals, which SOP-001 rejected (about 14% of its regret).
- Typed magnitude buckets cover every generated magnitude (2, 3, 4, 6 and 8% and +6 days were
  missing).
- The status words of the span parser: explicit ends of a suspension first, then suspended,
  restricted, and the weak words "lifted", "cleared" and "active". "Suspended until the audit is
  cleared" read as active, and "paused" or "limited to priority orders" read as nothing, so
  status paraphrases fell back to template text (all 6 restrictions and 11 of 15 suspensions).
- The paraphrase layer rejects a rewrite of a no-change message that gains a status or change
  word (`CHECKS_VERSION` c3). "Confirms all open orders are on schedule" became "all active
  orders", which three Qwen readers read as a reinstatement (74,104 each).
- A model decider's order lines for a plant that does not make the SKU are counted
  (`invalid_order_lines`) instead of dropped silently (79 of 1,378 lines in the first run), and
  constrained order schemas allow only real supply lines.
- When the reader is the oracle, a condition's own run is its oracle reference, so a model
  decider's nondeterminism no longer shows as interpretation regret (three streams had
  51,651, 9,648 and 89). Oracle references are cached per decider and client, not per kind.
- The interpretation share is reported as a ratio of sums; the mean of per-stream ratios gave
  0.037 against 0.182 for typed Qwen. `summarise` warns when conditions cover different streams.

- The status words of the span parser do not include "on hold", which is how lot holds are
  worded.
- State deciders see only the refusal notices the solver still uses.
- An unlinked restatement also applies when every `supersedes` reference names another
  entity's delta; restatements are logged only when they retract something.
- The ablation without the SOP gate also drops the sender-authority conflict rule.
- SOP-005 reads the sentences around the span, not the span alone: free-form Qwen quoted
  "Suspension of procurement is in effect until audit completion" from "Supplier C: Suspension
  of procurement...", and the rule rejected a correct suspension (two pilot streams in the
  final run).
- `rejected_correct` counts correct reads the SOP layer did not admit. Alignment scores every
  parsed delta, so such a read counts as exact although it never reached the state.

### Changed (2026-10-23)

- Key-account spikes say "orders for SKU-03 up 20% in wk44" (b2); "pulls 20% forward" implied a
  later decrease the gold delta did not have. The type guide's demand example follows, and the
  regex reader's demand keywords include "key account".
- Every experiment config uses `sop-v2`, the planner benchmark and the fixed safety
  factor, and reads the b2 streams. `local-deciders` adds `qwen-llm-over-oracle-state` (JSON
  orders from the oracle state, condition i) next to the typed variant.

## [0.1.0] - 2026-10-22

### Added

- Delta schema v1: seven delta types over six entity kinds, with `Proposal` (interpreter
  output) and `Delta` (completed by the span parser).
- The six layers: interpreters (oracle, regex baseline, JSON LLM, typed-model atomic
  questions), span parser, SOP admissibility with simulated approvers, bitemporal state with
  the update operator, rolling-horizon MILP solver with duals, and the audit log.
- Model-backed deciders (LLM and typed model, from the raw stream or from ŝ) as rivals to the
  solver.
- Single-echelon, multi-source inventory simulator with common random numbers and
  true-constraint violation tracking; perfect-information hindsight benchmark.
- Delta-first template benchmark generator with hold→release, suspension→reinstatement and
  revision patterns, and a parser round-trip guarantee.
- Evaluation: three-stage delta alignment and the seven error kinds, precision/recall/F1,
  regret decomposition, single-repair ΔC_j and sampled Shapley attribution, sensitivity map.
- `cdd` command (`schema`, `generate`, `run`), pilot configs, SOP file `sop-v1`, CI.
