# WP1 protocol: a prospectively specified fresh-stream replication

Written 2026-10-26, before any stream of the split below was generated. Frozen by the annotated
tag `icores-replication-freeze-v2`: the tag names the commit that contains this file, the code,
the configs and the analysis script. It supersedes the first freeze, tag
`icores-replication-freeze` (tag object 527a9f6, commit de32e5e), which was replaced on
2026-10-26 before the split was generated or any run was made from it (see "Amendment before
generation"). Runs count only if their `meta.json` shows that commit with
`git_dirty: false`. Later commits never change the files listed under "Frozen". This is **not a
preregistration**: the record is a git tag in the project repository, which is not an independent
timestamped registry. It is called a preregistration only if this file is uploaded to such a
registry (e.g. OSF) before the split is generated, and then the registry, date and visibility are
stated here in a later commit.

Context: [icores-revision-scope.md](icores-revision-scope.md), Section 3.

## Question and scope

Do the reader-cost findings of the pilot (RQ2, RQ3) hold on fresh worlds and streams generated
after the protocol was fixed? The replication does **not** test RQ1: model deciders are excluded.
The pilot remains exploratory; this run does not make it confirmatory after the fact.

## Frozen

Everything in the tagged commit, in particular:

- the readers:
  - prompts and examples, constrained schemas, typed questions and option labels
  - regex rules
  - the span parser
- the decision side:
  - the SOP file `configs/sop/sop-v2.yaml` and the `magnitude_cap` approver (cap 0.5)
  - the update logic
  - the planner (horizon 6, time limit 10 s, safety factor 0.3) and the planner benchmark
- the evaluation:
  - alignment and the error taxonomy
  - repairs, single-repair effects, Shapley values (200 permutations, seed rule of `run_experiment`)
  - the metrics and `scripts/replication_analysis.py`
- the environment:
  - `laya==0.4.1` (the pilot's Laya checkpoint)
  - the Ollama models with the digests recorded in the pilot's `models.json` (checked 2026-10-26:
    `qwen2.5:7b-instruct` and `llama3.1:8b` match)
  - Phi-4 for paraphrase, paraphrase checks `c3`
- the split and conditions below, the number of streams, and the failure and exclusion rules.

## Split

- `configs/benchmark/confirm.yaml`: identical to `configs/benchmark/pilot.yaml` (b2) except
  `name: confirm`, `seed: 2027` and `n_streams: 48`. The seed was chosen without generating anything and
  is not one of the existing splits' seeds (7, 31, 101, 211).
- Generated once on the laptop after the tag (`data/v2/confirm`), paraphrased once with Phi-4
  (`data/v2/confirm-para`). Paraphrases keep the parser-acceptance check, so this split shares
  the pilot's paraphrase filter; WP2 addresses that limitation separately.
- Development and debugging used only the dev split (seed 101).

## Number of streams: 48 (D14, amended before generation)

The rule fixed in the scope (Section 3) was to choose the smaller of 24 and 48 whose simulated
chance of a lower bound above 0 is at least 0.8, if its run time fits the laptop schedule.
`scripts/wp1_budget.py` on `icores-final-readers-para-r2` and
`icores-final-readers-para-r3-shapley` gave the following.

**Pilot input:** the per-stream free-minus-constrained cost differences over 24 streams have
mean 6,875, median 0, free costing more on 11 streams, and a 95% interval of 382 to 17,021.

**Simulation:** resample n streams from the pilot differences, compute the 95% bootstrap lower
bound, repeat 2,000 times. The bound is above 0 in **84%** of splits at n = 24 and in **100%** at
n = 48. These are upper bounds, because the pilot estimate was chosen after inspection.

**Run time per stream:**

| Item | Minutes |
| --- | --- |
| qwen, 61 replies at 15.7 s each | 15.9 |
| llama (not priced by the script, because its pilot replies came from the cache; one condition against qwen's three, so about a third of qwen's time) | about 5 |
| Solver, repairs, Shapley and Laya | 0.1 |
| Paraphrase | 0.5 |
| **Total** | **about 22** |

That is about 9 h for 24 streams and 18 h for 48. Both fit before Wed 10-28 12:00.

24 meets the 0.8 rule, so the first freeze set **n = 24**.

### Amendment before generation: n = 48

Made on 2026-10-26, after the first freeze and before any stream of the split was generated or
any reader ran on it. It departs from the rule above as first written, which still gives 24.

**Why the time estimate was wrong.** `wp1_budget.py` priced replies at 15.7 s from the few r2
replies that missed the reply cache. The fresh r1 run (`icores-final-readers-para`, all replies
new) averaged about 2.1 s per reply, and its whole 24-stream reader run took 65 min (10:23 to
11:28 on 10-23, from the laptop's run folder). With paraphrase, single repairs and Shapley values,
WP1 takes about 2 to 2.5 h at n = 24 and about 4 to 5 h at n = 48.

**Why 48.** The 84% at n = 24 is an optimistic upper bound: the pilot difference has median 0 and
rests on a few streams (free costs more on 11 and less on 5). 48 gives 100% in the same
simulation and tighter secondary estimates, for about 2 to 3 more hours. The choice uses no
information from the fresh split.

**Streams 0 to 23 are unchanged.** Each stream is a pure function of (seed, stream index), so the
first 24 streams of the 48-stream split are the 24-stream split. To confirm this, both sizes were
generated once in a scratch folder of the cloud session and compared file by file (identical);
the files were deleted without being opened, and no reader or planner ran on them.

**Time fallback, fixed now.** Streams run in order. If the run has not finished by Wed 10-28
08:00 ET, it is stopped and the analysis uses the first 24 streams
(`replication_analysis.py ... --first 24`). That choice depends on the clock alone, never on
results.

**Also fixed in the second freeze:** `replication_analysis.py` crashed on its last step, the
paraphrase counts, because `fallback` in the stream provenance is a list of message ids. It now
counts the list. The primary and secondary numbers were not affected.

## Pre-run check

Before the second freeze, the author ran the full replication config on 2 cached pilot streams
on the laptop, at the first tag. All 8 conditions finished with single repairs and 200 Shapley
permutations. All 14 comparable reader costs matched the recorded pilot costs exactly, Laya
included, and the analysis script reproduced the pilot's primary numbers (mean 6,874.6,
interval 381.6 to 17,021.1). No fresh stream was involved.

## Conditions

`configs/experiments/icores-replication.yaml`, on `data/v2/confirm-para`, each reader followed
by the planner:
- **Baselines:** oracle, no update (`interpreter: none`), regex.
- **Qwen 2.5 7B:** free JSON, constrained JSON, typed questions.
- **Others:** Llama 3.1 8B constrained JSON, Laya typed questions (zero-shot).

**Model settings:** temperature 0, seed 0. The reply cache is new
(`runs/cache/replies-replication.sqlite`), so every reply is generated for this split.
**Attribution:** single repairs and 200 Shapley permutations. Template-text runs and model
deciders are not part of this protocol.

## Analysis (`scripts/replication_analysis.py RUN_DIR data/v2/confirm-para`)

With the time fallback only, add `--first 24`.

**Primary, the only confirmatory comparison.**
- **Quantity:** per stream, the cost of Qwen free JSON minus the cost of Qwen constrained JSON
  (equal to the difference in R_interp).
- **Reported:** the mean, the median, the streams where free costs more, and the two-sided 95%
  percentile bootstrap interval over streams (10,000 resamples, seed 0).
- **Verdict:** the pilot direction counts as replicated if the lower bound is above 0. Otherwise
  it is reported as not replicated, with the estimate.
- **Multiplicity:** with one primary comparison, no adjustment.
- **Unit:** the stream (world), not the message.

**Secondary, descriptive.** Unadjusted 95% intervals; no confirmatory claim, no ranking threshold.
- each reader's R_interp
- each reader's cost minus the no-update cost (no requirement that every reader beats it)
- Kendall's τ between readers' mean F1 and mean cost, with a stream-bootstrap interval
- cost concentration for readers with at least 30 errors (Shapley values; top
  max(1, round(n/10)) errors over the sum of positive values)
- positive Shapley shares by delta type and error kind, pooled over readers
- paraphrase acceptance and fallback counts

## Failures and exclusions

- **Failed pairs:** pairs in `failed.json` are rerun once with `--resume`
  (`scripts/wp1-overnight.sh` does this). Pairs that still fail are reported as missing.
- **Paired comparisons** use the streams where both conditions finished, with the count reported.
- **No stream is dropped because of its outcome.** Capped or malformed replies count as reader
  output.
- **Paraphrase fallbacks** to template text are counted from `Stream.provenance` and reported.
- **`scripts/scan_run.py`** runs on the run folder and its findings are reported. A harness defect
  found after outcomes are seen is fixed in a later commit, and every analysis it affects is
  labelled exploratory.

## Procedure

On the laptop, after the tag:
```bash
git switch --detach icores-replication-freeze-v2
caffeinate -i -s zsh scripts/wp1-overnight.sh
```
The script, in order:
1. Refuses to run off the tag, with a dirty checkout, or without `laya==0.4.1`.
2. Generates the split.
3. Paraphrases it.
4. Runs the replication (`icores-replication-para`), with one resume if pairs failed.
5. Saves the Ollama version and model list.
6. Scans the run and prints the analysis.
7. Runs the Laya export for WP3 (scope 5.9) and its gate.
