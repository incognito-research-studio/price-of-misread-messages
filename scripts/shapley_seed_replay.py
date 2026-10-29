"""Re-attribute a recorded reader run with another Shapley seed, with no model call.

usage: python scripts/shapley_seed_replay.py RECORDED_RUN EXPORT_RUN STREAMS_DIR SEED OUT_DIR

Replays RECORDED_RUN's config from EXPORT_RUN/proposals.jsonl with the config seed set to SEED. The
seed feeds only the permutation sampler, so costs and errors match the recorded run and only
Shapley values can move. Writes OUT_DIR/errors.jsonl and OUT_DIR/rows.jsonl; compare them with
scripts/shapley_seed_compare.py.
"""

import json
import sys
import time
from pathlib import Path

from cdd.experiments import ExperimentConfig, replay_experiment
from cdd.recorded import read_records

recorded, export, streams = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
seed, out = int(sys.argv[4]), Path(sys.argv[5])

config = ExperimentConfig.model_validate_json((recorded / "config.json").read_text())
config = config.model_copy(update={"seed": seed})
started = time.perf_counter()
replay = replay_experiment(config, read_records(export), streams=streams)

out.mkdir(parents=True, exist_ok=True)
(out / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in replay.errors))
(out / "rows.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in replay.rows))
print(
    f"== {recorded.name} with seed {seed}: {len(replay.rows)} pairs, {len(replay.errors)} errors "
    f"in {time.perf_counter() - started:,.0f} s; not exported: {len(replay.missing)}"
)
