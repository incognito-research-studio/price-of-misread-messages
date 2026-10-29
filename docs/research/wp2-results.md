# WP2 results: the unfiltered language challenge

Run 2026-10-27/28 from the frozen challenge texts (development commit `99806aa`).
Source: `runs/icores-challenge`; `meta.json` records `99806aa` (main run) and
`fc9b04a` (resume of one pair, below), both with `git_dirty: false`. Analysis:
`scripts/challenge_analysis.py runs/icores-challenge runs/icores-final-readers-para-r3-shapley
runs/icores-final-readers-template-r2`; scan: `scripts/scan_run.py runs/icores-challenge
data/v2/challenge`. Every number below comes from those two outputs.

## Material

- 59 messages of four pilot streams (pilot-000, -015, -021, -022), 45 stating a change and 14
  stating none, rewritten from fact sheets by one writer (W1) and checked by one verifier (V1),
  by code only. The writer reported having seen no templates, parser or reader output (59 of 59).
  All 59 texts were verified and kept.
- Imported with `scripts/import_challenge.py` (4 streams, 59 messages, no problem) and frozen in
  `docs/research/challenge/` (rewrites and import manifest with hashes) before any run.
- Two texts name their supplier only by reference or abbreviation; the verifier recorded this in
  the notes and verified both, since each states its change correctly.

## Run and the one failed pair

- 32 pairs (4 streams x 8 conditions). One pair failed in the main run and again after one
  `--resume`: pilot-000 / gold-span, `ValueError: delta 'pilot-000-m0010#g0' is already recorded`.
- Cause: a harness defect in attribution. The gold-span reader keeps gold ids; a repair of another
  error re-inserted that gold delta beside the reader's own copy. Fixed in the harness
  (`apply_repairs` gives the inserted copy a fresh id); `replay_check.py` showed no change to any
  recorded reader result. The pair was then rerun with `--resume` from `fc9b04a`. It was found
  after outcomes were seen, so this pair is exploratory.
- Scan: 0 failed pairs after the resume; reader-output notes: one capped and one malformed Llama
  reply, one unknown entity from free-JSON Qwen (reader output, counted as such).

## Results (4 streams; descriptive)

Identified: gold deltas whose type and entity the reader got right. Completed: of those, value
and window also right. R_interp per stream in cost units.

| Reader | Identified (of 45) | Completed | Parse failures by slot | F1 | R_interp mean / median |
| --- | --- | --- | --- | --- | --- |
| Regex | 30 | 16 (53%) | value 10 | 0.369 | 24,587 / 334 |
| Qwen, free JSON | 29 | 16 (55%) | interpreter 11, value 10, entity 1 | 0.437 | 7,264 / 754 |
| Qwen, constrained JSON | 25 | 15 (60%) | value 25 | 0.387 | 624 / 1,172 |
| Qwen, typed questions | 45 | 26 (58%) | - | 0.597 | 22,238 / 1,041 |
| Llama, constrained JSON | 29 | 21 (72%) | value 20, interpreter 1 | 0.508 | -1,457 / -119 |
| Laya, typed questions | 38 | 17 (45%) | value 3 | 0.265 | 7,507 / 312 |
| Gold span (parser isolation) | 39 | 23 (59%) | value 6 | 0.579 | 20,527 / -2 |

On the same four streams paraphrased (`icores-final-readers-para-r3-shapley`), completion was
88-100% and R_interp regex -381, free JSON 13,627, constrained JSON 0, typed Qwen -908, Llama
-2,934, Laya 1,763.

- The gold-span reader's 22 errors: 16 mis-timed (window misread) and 6 missed (value parse
  failure). Its R_interp per stream: pilot-000 88,460, pilot-015 -238, pilot-021 -6,345,
  pilot-022 233; the costliest single error, a mis-timed supplier suspension on pilot-000, has
  Shapley value 61,323.
- Reading: on unfiltered text the deterministic parser, more than the language model, limits
  what is read correctly. With four streams the costs are descriptive.
