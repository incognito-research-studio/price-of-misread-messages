from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import fakes
from cdd.benchmark import GeneratorConfig, generate
from cdd.cli import main
from cdd.experiments import ConditionConfig, ExperimentConfig, load_factory, run_experiment


@pytest.fixture
def streams_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "streams"
    directory.mkdir()
    for stream in generate(
        GeneratorConfig.model_validate({"name": "t", "n_streams": 2, "seed": 5})
    ):
        stream.save(directory / f"{stream.id}.json")
    return directory


def test_run_experiment_writes_every_artifact(tmp_path: Path, streams_dir: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(tmp_path / "runs"),
            "attribution": {"single_repair": True, "shapley_permutations": 2},
            "conditions": [
                {"name": "oracle"},
                {"name": "regex", "interpreter": "regex"},
                {
                    "name": "typed-over-oracle-state",
                    "interpreter": "oracle",
                    "decider": "typed_state",
                    "client": "fakes:zero_order_typed_client",
                },
                {"name": "llm-json", "interpreter": "llm_json"},
            ],
        }
    )
    run_dir = run_experiment(config, run_name="test")
    for name in ("results.jsonl", "summary.csv", "errors.csv", "sensitivity.csv", "config.json"):
        assert (run_dir / name).exists(), name

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    assert set(by_condition) == {"oracle", "regex", "typed-over-oracle-state"}
    assert all(row["r_interp"] == 0 for row in by_condition["oracle"])
    assert all(row["audit_completeness"] == 1.0 for row in by_condition["regex"])
    # Ordering nothing is worse than the solver, and that gap is decision regret, not
    # interpretation regret, because the typed decider read the oracle state.
    for typed, oracle in zip(
        by_condition["typed-over-oracle-state"], by_condition["oracle"], strict=True
    ):
        assert typed["r_interp"] == 0
        assert typed["cost"] > oracle["cost"]

    assert "llm-json" in json.loads((run_dir / "skipped.json").read_text())
    with (run_dir / "summary.csv").open() as handle:
        assert {row["condition"] for row in csv.DictReader(handle)} == set(by_condition)


def test_condition_shape_is_validated() -> None:
    with pytest.raises(ValidationError, match="raw messages"):
        ConditionConfig(name="x", interpreter="regex", decider="llm_stream")
    with pytest.raises(ValidationError, match="needs an interpreter"):
        ConditionConfig(name="x", interpreter=None, decider="milp")


def test_load_factory() -> None:
    assert callable(load_factory("fakes:zero_order_typed_client"))
    with pytest.raises(ValueError, match=r"package\.module:factory"):
        load_factory("no-colon")


def test_tuned_benchmark_ablations_and_model_conditions(tmp_path: Path, streams_dir: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(tmp_path / "runs"),
            "benchmark": "tuned",
            "tuned_safety_factors": [0.0, 0.2],
            "attribution": {"single_repair": False},
            "conditions": [
                {"name": "oracle"},
                {"name": "no-sop", "interpreter": "regex", "sop_gate": False},
                {"name": "no-versioning", "interpreter": "regex", "versioning": False},
                {
                    "name": "llm-over-state",
                    "interpreter": "oracle",
                    "decider": "llm_state",
                    "client": "fakes:empty_llm_client",
                },
                {
                    "name": "hybrid",
                    "interpreter": "hybrid",
                    "client": "fakes:zero_order_typed_client",
                    "fallback_client": "fakes:empty_llm_client",
                    "threshold": 0.95,
                },
            ],
        }
    )
    run_dir = run_experiment(config, run_name="tuned")
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    assert set(by_condition) == {"oracle", "no-sop", "no-versioning", "llm-over-state", "hybrid"}
    for row in by_condition["oracle"]:
        # x* is the best variant, and the plain solver is one of them, so R_dec >= 0.
        assert row["benchmark"] == "tuned" and row["r_dec"] >= -1e-6
        assert row["tuned_safety_factor"] in (0.0, 0.2)
        assert row["value_of_perfect_information"] >= -1e-6
    assert all(row["escalation_share"] == 1.0 for row in by_condition["hybrid"])
    assert all(row["model_requests"] == 0 for row in by_condition["oracle"])


def test_no_update_baseline_reads_nothing(tmp_path: Path, streams_dir: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(tmp_path / "runs"),
            "conditions": [{"name": "oracle"}, {"name": "no-update", "interpreter": "none"}],
        }
    )
    run_dir = run_experiment(config, run_name="test")
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    oracle = {row["stream"]: row for row in rows if row["condition"] == "oracle"}
    for row in (row for row in rows if row["condition"] == "no-update"):
        assert row["n_proposed"] == 0
        assert row["recall"] == 0.0
        assert row["model_requests"] == 0
        assert row["r_interp"] == pytest.approx(row["cost"] - oracle[row["stream"]]["cost"])


def test_planner_benchmark_is_the_solver_on_the_oracle_state(
    tmp_path: Path, streams_dir: Path
) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(tmp_path / "runs"),
            "solver": {"safety_factor": 0.1},
            "attribution": {"single_repair": False},
            "conditions": [
                {"name": "oracle"},
                {"name": "regex", "interpreter": "regex"},
                {
                    "name": "llm-over-state",
                    "interpreter": "oracle",
                    "decider": "llm_state",
                    "client": "fakes:empty_llm_client",
                },
            ],
        }
    )
    run_dir = run_experiment(config, run_name="planner")
    rows = _rows(run_dir)
    for (_, condition), row in rows.items():
        assert row["benchmark"] == "planner" and row["planner_safety_factor"] == 0.1
        assert row["value_of_perfect_information"] == pytest.approx(
            row["planner_cost"] - row["hindsight_cost"]
        )
        assert row["conflicts_dropped"] == 0 and row["invalid_order_lines"] == 0
        if condition == "oracle":
            assert row["r_total"] == row["r_interp"] == row["r_dec"] == 0.0
        elif condition == "regex":
            assert row["r_dec"] == 0.0 and row["r_interp"] == pytest.approx(row["r_total"])
        else:
            # A model decider on the oracle state: its own run is the reference.
            assert row["r_interp"] == 0.0
            assert row["r_dec"] == pytest.approx(row["cost"] - row["planner_cost"])
    summary = {r["condition"]: r for r in csv.DictReader((run_dir / "summary.csv").open())}
    assert "ratio_interp_share" in summary["regex"] and "median_r_total" in summary["regex"]


def test_hindsight_benchmark_option(tmp_path: Path, streams_dir: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(tmp_path / "runs"),
            "benchmark": "hindsight",
            "attribution": {"single_repair": False},
            "conditions": [{"name": "oracle"}],
        }
    )
    run_dir = run_experiment(config, run_name="hindsight")
    row = json.loads((run_dir / "results.jsonl").read_text().splitlines()[0])
    assert row["benchmark"] == "hindsight"
    assert row["r_dec"] == row["cost"] - row["hindsight_cost"]
    assert "tuned_cost" not in row


def _rows(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    timing = {"seconds", "model_seconds"}
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    return {
        (row["stream"], row["condition"]): {k: v for k, v in row.items() if k not in timing}
        for row in rows
    }


def test_failed_pairs_are_recorded_and_resumed(
    tmp_path: Path, streams_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(tmp_path / "runs"),
            "tuned_safety_factors": [0.0],
            "conditions": [
                {"name": "oracle"},
                {"name": "flaky", "interpreter": "llm_json", "client": "fakes:flaky_llm_client"},
            ],
        }
    )
    monkeypatch.setitem(fakes.SERVER, "down", True)
    run_dir = run_experiment(config, run_name="r")
    failed = json.loads((run_dir / "failed.json").read_text())
    assert {f["condition"] for f in failed} == {"flaky"} and len(failed) == 2
    assert all("TimeoutError" in f["error"] for f in failed)
    assert {condition for _, condition in _rows(run_dir)} == {"oracle"}

    with pytest.raises(ValueError, match="different config"):
        run_experiment(config.model_copy(update={"seed": 1}), run_name="r", resume=True)

    monkeypatch.setitem(fakes.SERVER, "down", False)
    run_experiment(config, run_name="r", resume=True)
    assert json.loads((run_dir / "failed.json").read_text()) == []
    resumed = _rows(run_dir)
    assert len(resumed) == 4
    assert len(json.loads((run_dir / "meta.json").read_text())["resumed"]) == 1

    # The same numbers as a run that never failed.
    assert _rows(run_experiment(config, run_name="fresh")) == resumed

    # Each request is logged with the stream and condition that made it.
    log = (run_dir / "requests-fake-model-1.jsonl").read_text().splitlines()
    assert {json.loads(line)["tag"].split("/")[1] for line in log} == {"flaky"}


def test_cli_reports_failed_pairs(
    tmp_path: Path, streams_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "streams": str(streams_dir),
                "output_dir": str(tmp_path / "runs"),
                "tuned_safety_factors": [0.0],
                "conditions": [
                    {"name": "flaky", "interpreter": "llm_json", "client": "fakes:flaky_llm_client"}
                ],
            }
        )
    )
    monkeypatch.setitem(fakes.SERVER, "down", True)
    assert main(["run", "--config", str(config), "--run-name", "c"]) == 1
    monkeypatch.setitem(fakes.SERVER, "down", False)
    assert main(["run", "--config", str(config), "--run-name", "c", "--resume"]) == 0


def test_correct_reads_the_sop_layer_rejects_are_counted(world: Any) -> None:
    from cdd.evaluation import align
    from cdd.experiments import _rejected_correct
    from cdd.interpreters import ReplayInterpreter
    from cdd.orders import InventoryPosition
    from cdd.pipeline import CompilingPolicy
    from cdd.schema import Span, Window
    from cdd.solver import MILPSolver
    from cdd.sop import SOPConfig, SOPGraph
    from helpers import make_delta, make_message

    # A release with no hold in force is rejected by SOP-003, though it matches the gold.
    message = make_message("Lot 7781 released.", id="r", sender_class="supplier_quality")
    release = make_delta(
        id="r#0", type="quality_release", entity="LOT-7781", op="set", value="released",
        unit=None, window=Window(start=40), src=Span(message_id="r", start=0, end=17),
    )  # fmt: skip
    policy = CompilingPolicy(
        world,
        [message],
        ReplayInterpreter([release]),
        MILPSolver(horizon=2),
        sop=SOPGraph(SOPConfig()),
    )
    policy(InventoryPosition(period=40, on_hand={}, lot_remaining={}, backlog={}))
    gold = release.model_copy(update={"id": "r#g0"})
    assert _rejected_correct(policy, align(policy.proposed, [gold])) == 1
