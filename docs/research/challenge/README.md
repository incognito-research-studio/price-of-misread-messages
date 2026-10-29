# WP2 challenge set

## Selection (2026-10-26)

Four pilot streams, 59 messages (45 with one change each, 14 without): `pilot-000`,
`pilot-015`, `pilot-021`, `pilot-022`. Chosen for coverage only, in a separate environment with no
access to any run folder or reader output:

- `pilot-000` and `pilot-021` contain all seven delta types;
- `pilot-015` is the only pilot stream with a revision (a corrected capacity notice);
- `pilot-022` adds two supplier-status changes and three no-change messages within the budget
  of about 60 messages.

No pilot message carries more than one change, so the set has no multi-delta messages.

## Files

- `fact-sheets.md`: what each writer gets, one entry per message. Facts come from the gold
  deltas; the generated text is never shown. Each message has an assigned language feature
  (calendar dates, distractor numbers, unfamiliar structure, negation, informal wording,
  cancellation, correction, and two kinds of no-change message), in a fixed rotation.
- `rewrites.yaml`: one entry per message. The writer fills `writer` and `text`. The verifier
  fills `evidence` (the exact sentence or sentences of `text` that state the change, empty for
  no-change messages), `verified_by`, `verified` and `notes`.
- `manifest.json`: the streams with a hash of their messages and gold deltas, the message ids
  and the assigned features.

Regenerate with
`python scripts/challenge_fact_sheets.py data/v2/pilot OUT pilot-000 pilot-015 pilot-021 pilot-022`.
Regenerate into a temporary folder and compare `manifest.json`: equal hashes mean
your pilot streams match the ones these sheets were made from.

## Writers and verifier (rule amended 2026-10-27, before writing started)

The writers and the verifier are anonymous. Every record, including `rewrites.yaml`, the stream
provenance and the paper, identifies them only by code: W1, W2, … for writers and V1 for the
verifier. The key from codes to people is held by the authors and is not published.
Independence is still recorded per message: whether its writer had seen the span parser, the
templates or generated messages. The verifier is not one of the writers. The number of writers
is reported with the results.

The writers worked in one shared workbook built from these fact sheets (a writer pack with
instructions). The filled workbook was converted mechanically into `rewrites.yaml`, refusing any
writer or verifier entry that is not a code. `rewrites.yaml` is the frozen record.

## Process

1. Writers receive `fact-sheets.md` by Tue 10-27 08:00 ET. They do not look at the generated
   message texts, the span parser, the templates or any reader output, and never test a text
   with them. Record in `notes` whether a writer has read the parser or templates, and never
   record a writer's or the verifier's name.
2. The verifier checks every text against its sheet. A text that adds, drops or alters a fact
   goes back to its writer before the freeze. Every finalized text is kept, whatever the parser
   later does with it.
3. Texts are frozen at Wed 10-28 12:00 ET (commit `rewrites.yaml`). Only then are they imported
   and read by any model or parser. A set not frozen by then is dropped from this
   submission.
