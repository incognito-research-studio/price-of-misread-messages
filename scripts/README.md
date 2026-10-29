# Scripts

Run every script from the repository root in the project environment, for example
`python scripts/paper_numbers.py ...`. Run folders are named relative to `runs/` where the usage
line says `RUN` without a path. Unless noted, a script reads streams and run records only and
calls no model. The [main README](../README.md) gives the exact command behind each table and
figure.

## Recompute the paper's numbers and figures

- `paper_numbers.py [READERS_PARA] [READERS_TEMPLATE] [DECIDERS]`: the numbers in Tables 4 and 5
  and the pilot cost concentration. The saved output is `runs/icores-final-numbers-r2.txt`.
- `replication_analysis.py RUN_DIR [STREAMS_DIR]`: the analysis frozen in the replication
  protocol (Table 6): the primary free-minus-constrained comparison and the descriptive
  secondary numbers. The saved output is `runs/icores-replication-para.analysis`.
- `revision_numbers.py READERS_SHAPLEY NO_UPDATE DECIDERS [BUFFER_RUN ...]`: Section 5.3 and
  related numbers: the no-update baseline, each reader's worst stream, Shapley values against
  single repairs (with a per-stream check that Shapley values sum to R_interp), pooled shares
  with stream-bootstrap intervals, both decider cost-ratio statistics, and the buffer sensitivity
  runs.
- `challenge_analysis.py CHALLENGE_RUN [PARA_RUN TEMPLATE_RUN]`: Section 5.5, per reader and
  stage (identified, parser-complete, applied, spurious, parse failures by slot, R_interp), with
  the same readers on the same streams in the pilot runs. The saved output is
  `runs/icores-challenge.analysis`.
- `promising_report.py REPLENISHMENT_RUN PROMISING_RUN...`: Table 7 and Section 5.4: losses,
  volume, solver fallbacks, gates, Shapley shares, Kendall's tau and paired errors within and
  across the two decisions.
- `concentration_figure.py RUN FIELD [FRESH_RUN]`: Fig. 2 as pgfplots source, from a run's
  `errors.jsonl`, with `FIELD` `shapley` or `delta_c`; with `FRESH_RUN`, each reader also gets a
  dashed curve from that run.
- `error_price_figure.py READER_RUN PROMISING_RUN`: Fig. 3 as pgfplots source: each error's
  single-repair effect under replenishment and under order promising, on signed log scales.

## Replay recorded reader output

These rebuild results from the readers' recorded deltas (`proposals.jsonl`) instead of reading
stored results.

- `replay_check.py RECORDED_RUN EXPORT_RUN STREAMS_DIR`: re-decides the recorded run's
  conditions from the exported deltas and compares cost, R_interp, F1, single repairs and Shapley
  values with the recorded run. Ends with a list of problems, or "none".
- `promising_run.py RECORDED_RUN EXPORT_RUN STREAMS_DIR PARAMS OUT_DIR [--shapley N] [--seed S]`:
  scores a recorded reader run under order promising, and flags any pair whose all-repaired value
  differs from the gold reference.
- `snap_ablation.py RUN`: the post-hoc quote-snapping replay of Section 5.2. Replays the free-JSON
  reader's logged replies with the constrained schema's restrictions applied after generation;
  gated on reproducing the recorded costs.
- `buffer_ablation.py RUN [--buffers ...]`: post hoc. Reprices recorded reader output under other
  replenishment buffers, with single repairs and the model-free calibration check (Section 7).
- `basestock_baseline.py DECIDERS_RUN`: post hoc. An order-up-to policy from the oracle state,
  tuned on the development split, beside the recorded model deciders (Table 4).
- `shapley_seed_replay.py RECORDED_RUN EXPORT_RUN STREAMS_DIR SEED OUT_DIR` and
  `shapley_seed_compare.py RECORDED_ERRORS REPLAY_ERRORS`: the second-seed check of Section 3.4.
  The shipped outputs and comparisons are in `runs/icores-shapley-seed1-pilot` and
  `runs/icores-shapley-seed1-fresh`.
- `promising_dev_check.py EXPERIMENT_CONFIG STREAMS_DIR PARAMS [OUT_JSON]`: the development-split
  check that fixed the promising parameters (acceptance, on-time and broken shares, the
  miss-each-gold-delta calibration by type, the fixed-supply and repair gates); see
  `docs/research/promising-dev-check.md`.

## Checks run on every recorded run

- `scan_run.py RUN_DIR STREAMS_DIR`: checks a run folder for failed pairs, non-zero oracle regret,
  retries, capped or malformed model replies, evidence not found in the message, unknown entities
  and supersedes links to unknown ids. Ends with a list of problems, or "none".
- `check_export.py EXPORT_RUN RECORDED_RUN`: the export gate. Checks that a re-run that records
  reader output made no model request, wrote `proposals.jsonl` for every compiled pair, and
  reproduced each pair's cost, F1 and error identity.

## How the shipped data and runs were produced

These call the models through a local Ollama server, so they are not needed to check the paper.

- `icores-final.sh`: the b2 pilot conditions (readers on the paraphrased and template pilot,
  model deciders on the paraphrased pilot, baselines on the development split), each step
  scanned and the Ollama version and model list saved. `runs/icores-final-deciders-para` comes
  from this script; the shipped reader runs repeat its reader steps (`-r2`, and `-r3-shapley`
  with Shapley values).
- `paraphrase-v2.sh`: the Phi-4 paraphrases of the development and training splits.
- `wp1-overnight.sh`: the fresh-stream replication from the freeze tag: it checks the tag and a
  clean checkout, generates and paraphrases the fresh split once, runs it, applies the frozen
  analysis, then exports Laya's output and runs its gate.
- `wp1_budget.py READERS_RUN SHAPLEY_RUN PARA_STREAMS_DIR [OLLAMA_URL]`: the calculation recorded
  in the replication protocol for choosing 48 fresh streams: model digests, run time per stream,
  and how often 24 or 48 streams would put the free-minus-constrained lower bound above 0.
- `challenge_fact_sheets.py STREAMS_DIR OUT_DIR STREAM_ID...`: the challenge-set fact sheets,
  built from the gold deltas only (never the generated text), with an empty `rewrites.yaml` and a
  manifest of stream hashes; see `docs/research/challenge/README.md`.
- `import_challenge.py PILOT_STREAMS_DIR REWRITES_YAML OUT_DIR MANIFEST_JSON`: imports the frozen
  challenge texts into stream files, with gold spans re-anchored to the evidence; refuses and
  lists every problem otherwise.
