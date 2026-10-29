# Benchmark

## Delta-first generation

The generator ([`generator.py`](../src/cdd/benchmark/generator.py)) writes ground truth
first and text second:

1. **World.** Suppliers, plants (each making a few SKUs, with one inbound lane), SKUs with
   base demand and starting stock, price lines and lots, drawn from the ranges in the config.
2. **Gold delta sequences.** Each week, a Poisson number of events is drawn and each event is
   assigned a type by the configured weights. Patterns:
   - hold → release (a release follows a hold with probability `release_prob`);
   - suspension or restriction → reinstatement (`reinstate_prob`);
   - revision: a capacity cut is revised in a later message that supersedes the first
     (`revise_prob`).
   Two events with the same entity, attribute and window are never generated together, so the
   gold sequence never contains an unreconciled contradiction.
3. **Surface text.** Each delta is rendered through a template ([`templates.py`](../src/cdd/benchmark/templates.py))
   for its channel: portal notice, email, carrier update or quality note. The gold span covers
   the clause that states the delta, not greetings or trailing sentences.
4. **No-change messages** appear in a share of weeks (`no_change_rate`). Some are traps for
   keyword baselines, such as "Plant 2 output unchanged this week."
5. **Distractors**: a share of messages (`distractor_rate`) end with an irrelevant sentence.

Every stream is a pure function of `(seed, stream index)`. Each stream also carries its own
seed for demand noise, so all conditions face the same demand path.

## Calibration (benchmark b2)

A benchmark can only show that misreading a delta type matters if missing that type can cost
something. `cdd calibrate` checks this without any model: it removes each gold delta in turn,
replays the oracle pipeline with the configured planner and records the cost of the miss. On
the b1 settings (capacity 5-10 times demand, small lots on hand from the start, prices 8-20
drawn per offer, SKU-04 made by Supplier B alone), 2% of capacity misses, 4% of hold misses and
0% of release misses cost anything, and supplier status and lead time carried 93% of the miss
cost. "Cost concentrates in a few delta types" was then a property of the settings.

Benchmark b2 (`version: b2`, all splits) changes the world and the events:

| Setting | b2 value | Why |
| --- | --- | --- |
| `capacity_ratio` | 0.9-1.3 | a plant's capacity is that multiple of its share of the demand for its SKUs, so cuts bind |
| `min_suppliers_per_sku` | 2 | SKUs are assigned to plants at random; every SKU has at least two suppliers |
| `price_spread` | 0.05 | a SKU's offers lie within 5% of one base price, so a 2-8% change can reorder them |
| `lots_per_sku`, `lot_weeks`, `lot_arrival` | 1, 2-3 weeks of demand, weeks 39-47 | lots arrive during the horizon; holds go on lots that arrived in the last week or arrive within three |
| `initial_stock_weeks` | 3 | covers the longest lead time, so the first weeks are not short for every policy |
| `capacity_weeks` | 2-4 | a capacity change lasts long enough to bind |
| `cross_week_boundary` | true | a lead-time change always moves the lead time to another whole week |
| `plain_spikes` | true | "Key account orders for SKU-03 up 20% in wk44", a plain one-week increase |
| `last_message_week` | 49 | no message arrives too late to affect an order |
| `exclusive_windows` | true | two numeric changes to one entity never overlap unless one revises the other |

On the b2 pilot and dev splits, with the fixed planner (safety factor 0.3), the share
of misses that cost more than 100 is:

| Type | Pilot | Dev |
| --- | --- | --- |
| capacity change | 21% | 9% |
| demand signal | 20% | 28% |
| lead-time change | 49% | 51% |
| quality hold | 54% | 36% |
| quality release | 73% | 63% |
| supplier status | 42% | 48% |
| price change | 29% | 21% |

Supplier status carries 50% (pilot) and 40% (dev) of the miss cost, holds 33% and 35%. The
oracle run has no SOP rejection and no conflict drop and fills 96% of demand on time, as in b1.
The tuned 30% buffer absorbs most capacity and demand misses; worlds tight enough for 30% of
capacity misses to cost something had the oracle filling only 70% of demand on time, so the
realistic setting was kept. The reports are in `runs/calibration/`.

## Round-trip guarantee

Given only the gold type, entity and span, the span parser must reproduce the gold delta
exactly. [`tests/test_generator.py`](../tests/test_generator.py) checks this for every gold
delta in a dozen dense streams. It is what makes parser failures attributable to the
interpreter's span choice and not to the templates.

## Stream format

One JSON file per stream (`data/<name>/<stream-id>.json`):

```json
{
  "id": "pilot-000",
  "seed": 1234567,
  "world": { "start_period": 36, "end_period": 52, "suppliers": [...], "plants": [...], ... },
  "messages": [
    { "id": "pilot-000-m0000", "period": 36, "sender": "QA", "sender_class": "supplier_quality",
      "channel": "quality", "text": "QA note: Lot 7702 on hold pending CoA." }
  ],
  "gold": [
    { "id": "pilot-000-m0000#g0", "type": "quality_hold", "entity": "LOT-7702",
      "attribute": "hold_status", "op": "set", "value": "held", "unit": null,
      "window": { "start": 36, "end": null },
      "src": { "message_id": "pilot-000-m0000", "start": 9, "end": 38 },
      "confidence": null, "supersedes": [] }
  ]
}
```

## Splits

| Config | Streams | Seed | Use |
| --- | --- | --- | --- |
| `configs/benchmark/train.yaml` | 48 | 31 | fine-tuning typed models (`cdd laya-export`) |
| `configs/benchmark/dev.yaml` | 24 | 101 | tuning the solver (`cdd tune-solver`), choosing the hybrid's escalation set |
| `configs/benchmark/pilot.yaml` | 24 | 7 | the pilot comparison |
| `configs/benchmark/out-of-schema.yaml` | 24 | 211 | the out-of-schema slice |

All four share the generator settings and differ in seed and name, so entity names repeat across
splits. Splits by template family and entity are still to come. The b2 streams are generated
into `data/v2/<split>` (and paraphrased into `data/v2/<split>-para`). The b1 streams of earlier
development runs are not shipped; setting `version: b1` in `configs/benchmark/*.yaml` regenerates
them.

## Paraphrase layer

`cdd paraphrase` rewrites every message of a stream directory with a local model and writes
new stream files; the gold deltas are unchanged except for their spans.

```bash
cdd -v paraphrase --streams data/pilot --out data/pilot-para --model phi4
cdd -v run --config configs/experiments/pilot.yaml --streams data/pilot-para
```

Use a model family that no interpreter under test belongs to. The default, Phi-4 (MIT
licence), is disjoint from the local interpreters (Llama 3.1, Qwen 2.5 and Laya); its training
data includes synthetic text from OpenAI models, so prefer another family when GPT models are
interpreters. The model is told to keep entity names, percentages and day counts verbatim and
may reword weeks and dates. [`paraphrase.py`](../src/cdd/benchmark/paraphrase.py) accepts a
rewrite only when:

1. the names and numbers survive verbatim, the same weeks and dates are named, and no new
   name, number, week or date appears;
2. the span parser, given the gold type and entity and the sentences that now carry the
   delta's names, numbers and dates, recovers the gold operation, value and window (the
   round-trip guarantee, now on the rewritten text); and
3. a hold does not read as a release, a release says it is one, and a revision still says it
   revises an earlier notice. The parser does not read these cues, so they are checked
   separately.

Each message gets one attempt per temperature (0.7, 0.9, 1.1 by default), and a reply that
stops mid-sentence is rejected. Every request is seeded from the message id and the attempt, so
a template sentence that recurs across streams gets a different rewrite each time; with one
fixed seed, the first pilot run gave exactly one rewrite per distinct template sentence (334
for 489 messages). After the last rejection the message keeps its template text, and its id is
listed under `provenance.paraphrase.fallback` in the stream file, with the model, the prompt and
checks versions, and the counts.

The filter is selective. In the first pilot run, 14 of 33 supplier-status messages fell back
(rewrites such as "will only handle priority orders" state no status word the parser reads),
against none of the holds, releases and price changes. A paraphrased split therefore
under-represents wording the span parser cannot read, which favours parser-based conditions.
Report fallback shares by delta type with any result.

Checks version c3 widens the status words the parser reads ("paused", "limited to priority
orders", "the suspension has been lifted") and reads specific statuses before the weak words
"cleared" and "active", so "suspended until the audit is cleared" no longer reads as active.

The checks cannot catch every change of meaning. A rewrite can make a statement vaguer and
still pass, as in "down 20%" becoming "a reduced capacity of 20%", which the parser still reads
as a 20% cut. A no-change message gets check 1 and, from c3, may not gain a status or change
word: in b1, "Supplier B confirms all open orders are on schedule" became "...all active
orders...", which three Qwen readers read as a reinstatement. A rewrite that implies a change
in other words still passes. Read a sample of accepted rewrites before using a paraphrased
split. The parser change that came with this layer (it now reads "starting in week 38",
"week 41 onwards" and "November 5th") applies to every condition.

## Out-of-schema slice

`out_of_schema_rate` adds yield shortfalls: "shipments from Plant 4 will arrive about 20% short,
wk44–46". No delta type can express a partial delivery, so the message has no gold deltas, while
the simulator delivers only that share of every order the plant accepts in the window. The
planner's view of open orders shows the accepted quantity; the shortfall shows only on arrival,
so no compiling condition can learn it except by reading the message. The hindsight benchmark
knows it; the planner benchmark does not. Its cost therefore shows up in the value of perfect
information and in the gap to model deciders that read the raw stream. With the rate at 0 (the
default) the generator draws no extra random numbers, so other benchmarks are unchanged.

## What the pilot can and cannot show

The numbers in this section come from the b1 pilot. On template text the regex baseline is close to the oracle (F1 about 0.998 on the 24 pilot
streams), so the template pilot checks the machinery and the regret accounting but does not
separate interpreters. On the Phi-4 paraphrase of the same streams (447 of 489 messages
rewritten), regex recall falls to 0.873 and F1 to 0.928, and interpretation regret rises from 0
to about 1,400 per stream. The paraphrase checks were developed on these pilot streams; on the
dev split, which they were not developed on (445 of 490 messages rewritten), regex F1 falls
from 1.000 to 0.919 and interpretation regret rises from 0 to about 4,300 per stream. The
remaining difficulty factors below are still to come.

## Planned difficulty factors

- Vague magnitudes and windows, and granularity traps (a month stated where a week is needed)
- Cross-message conflicts and retractions
- Adversarial approval injections from unauthorised senders
- Messages that state more than one delta
- Splits by entity and template family, plus held-out combinations of delta types
