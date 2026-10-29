"""Command-line interface: ``cdd schema``, ``cdd generate`` and ``cdd run``."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from cdd import __version__
from cdd.benchmark import GeneratorConfig, generate
from cdd.benchmark.stream import load_streams
from cdd.experiments import ExperimentConfig, run_experiment, tune_safety_factor
from cdd.schema import proposal_json_schema


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdd",
        description="Operational compilation: typed state deltas and regret decomposition.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("schema", help="print the JSON Schema of interpreter output")

    gen = commands.add_parser("generate", help="generate benchmark streams")
    gen.add_argument("--config", type=Path, required=True, help="generator YAML")
    gen.add_argument("--out", type=Path, required=True, help="directory for stream files")

    run = commands.add_parser("run", help="run an experiment")
    run.add_argument("--config", type=Path, required=True, help="experiment YAML")
    run.add_argument("--run-name", help="output folder name (default: <name>-<UTC time>)")
    run.add_argument("--streams", type=Path, help="stream directory, overriding the config's")
    run.add_argument("--reply-cache", type=Path, help="reply cache file, overriding the config's")
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue the run named by --run-name, running only the pairs it lacks",
    )

    tune = commands.add_parser(
        "tune-solver", help="pick the solver's safety factor on development streams"
    )
    tune.add_argument("--config", type=Path, required=True, help="experiment YAML")
    tune.add_argument(
        "--streams", type=Path, required=True, help="development streams (not the test set)"
    )
    tune.add_argument("--factors", type=float, nargs="+", help="grid (default: from the config)")

    cal = commands.add_parser(
        "calibrate", help="model-free check: what each missed gold delta would cost"
    )
    cal.add_argument("--config", type=Path, required=True, help="experiment YAML (the planner)")
    cal.add_argument("--streams", type=Path, required=True, help="template streams")
    cal.add_argument("--limit", type=int, help="use only the first N streams")
    cal.add_argument("--workers", type=int, default=1, help="processes (one stream each)")
    cal.add_argument("--out", type=Path, help="write the full report as JSON")

    export = commands.add_parser(
        "laya-export", help="write typed-model fine-tuning rows from training streams"
    )
    export.add_argument("--streams", type=Path, required=True, help="training streams only")
    export.add_argument("--out", type=Path, required=True, help="JSONL file to write")
    export.add_argument("--labels", choices=("names", "neutral", "bare"), default="names")

    ft = commands.add_parser("laya-finetune", help="fine-tune Laya on exported rows (needs [laya])")
    ft.add_argument("--data", type=Path, required=True, help="rows from laya-export")
    ft.add_argument("--out", type=Path, required=True, help="output checkpoint directory")
    ft.add_argument("--epochs", type=int, default=4)
    ft.add_argument("--eval", type=Path, help="held-out rows (e.g. exported dev streams)")

    para = commands.add_parser(
        "paraphrase", help="rewrite stream messages with an LLM, keeping the gold deltas valid"
    )
    para.add_argument("--streams", type=Path, required=True, help="template streams")
    para.add_argument("--out", type=Path, required=True, help="directory for paraphrased streams")
    para.add_argument(
        "--model", default="phi4", help="Ollama model; use a family no interpreter uses"
    )
    para.add_argument("--host", default="http://127.0.0.1:11434")
    para.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.7, 0.9, 1.1],
        help="one attempt per temperature before falling back to the template text",
    )
    para.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    if args.command == "schema":
        print(json.dumps(proposal_json_schema(), indent=2))
    elif args.command == "generate":
        streams = generate(GeneratorConfig.from_yaml(args.config))
        args.out.mkdir(parents=True, exist_ok=True)
        for stream in streams:
            stream.save(args.out / f"{stream.id}.json")
        print(f"Wrote {len(streams)} streams to {args.out}")
    elif args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        if args.streams is not None:
            config = config.model_copy(update={"streams": args.streams})
        if args.reply_cache is not None:
            config = config.model_copy(update={"reply_cache": args.reply_cache})
        if args.resume and not args.run_name:
            raise SystemExit("--resume needs --run-name")
        run_dir = run_experiment(config, run_name=args.run_name, resume=args.resume)
        print(f"Results in {run_dir}")
        failed = json.loads((run_dir / "failed.json").read_text(encoding="utf-8"))
        if failed:
            print(f"{len(failed)} (stream, condition) pairs failed; see {run_dir / 'failed.json'}")
            resume = f"cdd run --config {args.config} --run-name {run_dir.name} --resume"
            print(f"Rerun them with: {resume}")
            return 1
    elif args.command == "tune-solver":
        config = ExperimentConfig.from_yaml(args.config)
        ranking = tune_safety_factor(config, load_streams(args.streams), args.factors)
        print("safety_factor  mean oracle cost")
        for factor, cost in ranking:
            print(f"{factor:>13g}  {cost:,.1f}")
        best = ranking[0][0]
        print(f"\nPre-register in the experiment config:\n  solver:\n    safety_factor: {best:g}")
    elif args.command == "calibrate":
        from cdd.benchmark.calibration import calibration_report

        config = ExperimentConfig.from_yaml(args.config)
        streams = load_streams(args.streams)[: args.limit]
        report = calibration_report(streams, config, workers=args.workers)
        if args.out:
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{report['streams']} streams; oracle: {report['oracle_rejected']} rejected, "
              f"{report['oracle_unapproved']} not admitted, "
              f"{report['oracle_conflicts_dropped']} conflict drops; mean cost "
              f"{report['mean_oracle_cost']:,.0f}, mean VPI "
              f"{report['mean_value_of_perfect_information']:,.0f}")  # fmt: skip
        print(f"{'type':18s} {'misses':>6s} {'costly':>7s} {'free':>6s} {'mean dC':>9s} "
              f"{'cost share':>10s}")  # fmt: skip
        for kind, row in report["by_type"].items():
            print(f"{kind:18s} {row['misses']:6d} {row['costly_share']:7.0%} "
                  f"{row['free_share']:6.0%} {row['mean_delta_c']:9,.0f} "
                  f"{row['share_of_cost']:10.0%}")  # fmt: skip
    elif args.command == "laya-export":
        from cdd.adapters.laya import training_rows

        args.out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with args.out.open("w", encoding="utf-8") as handle:
            for row in training_rows(load_streams(args.streams), labels=args.labels):
                handle.write(json.dumps(row) + "\n")
                count += 1
        print(f"Wrote {count} rows to {args.out}")
    elif args.command == "paraphrase":
        from cdd.adapters.ollama import OllamaClient
        from cdd.benchmark.paraphrase import Paraphraser, provenance_summary

        clients = [
            OllamaClient(
                args.model,
                host=args.host,
                temperature=temperature,
                seed=args.seed + attempt,
                check=attempt == 0,
            )
            for attempt, temperature in enumerate(args.temperatures)
        ]
        paraphraser = Paraphraser(clients, model=args.model)
        args.out.mkdir(parents=True, exist_ok=True)
        done = []
        for stream in load_streams(args.streams):
            rewritten = paraphraser.stream(stream)
            rewritten.save(args.out / f"{rewritten.id}.json")
            done.append(rewritten)
            info = rewritten.provenance["paraphrase"]
            logging.info(
                "%s: %d/%d accepted, %d fallback",
                rewritten.id,
                info["accepted"],
                info["messages"],
                len(info["fallback"]),
            )
        print(json.dumps(provenance_summary(done), indent=2))
    elif args.command == "laya-finetune":
        from cdd.adapters.laya import finetune

        options: dict[str, object] = {"epochs": args.epochs}
        if args.eval:
            options["eval_data"] = str(args.eval)
        summary = finetune(args.data, args.out, **options)
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
