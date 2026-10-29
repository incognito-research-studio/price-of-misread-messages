"""Score a recorded reader run under the order-promising policy (WP3), with no model call.

usage: python scripts/promising_run.py RECORDED_RUN EXPORT_RUN STREAMS_DIR PARAMS OUT_DIR
           [--shapley N] [--seed S] [--no-single]

RECORDED_RUN gives the config (streams order, conditions, layers, seed and attribution);
EXPORT_RUN/proposals.jsonl gives each reader's completed deltas; PARAMS is a promising parameter
file (configs/promising/). OUT_DIR gets results.jsonl and errors.jsonl, appended as each
(stream, condition) pair finishes, plus config.json, meta.json and missing.json. --seed changes
the run seed (the second Shapley run of D13); --shapley overrides the permutation count.
"""

import argparse
import json
from pathlib import Path

from cdd.experiments import AttributionConfig, ExperimentConfig, _meta
from cdd.promising import PromisingParams
from cdd.promising.run import promising_experiment
from cdd.recorded import read_records

parser = argparse.ArgumentParser()
parser.add_argument("recorded", type=Path)
parser.add_argument("export", type=Path)
parser.add_argument("streams")
parser.add_argument("params", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--shapley", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--no-single", action="store_true")
args = parser.parse_args()

config = ExperimentConfig.model_validate_json((args.recorded / "config.json").read_text())
if args.seed is not None:
    config = config.model_copy(update={"seed": args.seed})
attribution = AttributionConfig(
    single_repair=not args.no_single,
    shapley_permutations=(
        config.attribution.shapley_permutations if args.shapley is None else args.shapley
    ),
    max_errors=config.attribution.max_errors,
)
params = PromisingParams.from_yaml(args.params)

args.out.mkdir(parents=True, exist_ok=False)
(args.out / "config.json").write_text(
    json.dumps(
        {
            "recorded_run": str(args.recorded),
            "export_run": str(args.export),
            "streams": args.streams,
            "params_file": str(args.params),
            "params": params.model_dump(mode="json"),
            "seed": config.seed,
            "attribution": attribution.model_dump(mode="json"),
            "experiment": config.model_dump(mode="json"),
        },
        indent=2,
    )
    + "\n"
)
(args.out / "meta.json").write_text(json.dumps(_meta(), indent=2) + "\n")


def append(name: str, rows: list[dict]) -> None:
    with (args.out / name).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def progress(row: dict, errors: list[dict]) -> None:
    append("results.jsonl", [row])
    append("errors.jsonl", errors)
    gap = row["repaired_gap"]
    print(
        f"{row['stream']} {row['condition']}: loss {row['loss']:,.1f}  errors {row['n_errors']}"
        + ("" if gap is None else f"  repaired gap {gap:.2g}"),
        flush=True,
    )


run = promising_experiment(
    config,
    read_records(args.export),
    params,
    streams=args.streams,
    attribution=attribution,
    progress=progress,
)
(args.out / "missing.json").write_text(json.dumps(run.missing, indent=2) + "\n")
gaps = [r for r in run.rows if r["repaired_gap"] is not None and abs(r["repaired_gap"]) > 1e-6]
print(
    f"\n{len(run.rows)} pairs; not exported: {len(run.missing)}; repaired-gap failures: {len(gaps)}"
)
for row in gaps:
    print(f"   GATE FAILED {row['stream']} {row['condition']}: gap {row['repaired_gap']}")
