from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cdd import experiments
from cdd.benchmark import GeneratorConfig, generate
from cdd.experiments import ExperimentConfig, replay_experiment, run_experiment
from cdd.recorded import RECORDS_FILE, ReaderRecord, read_records

ERROR_FIELDS = ("kind", "kinds", "delta_type", "predicted_id", "gold_id", "delta_c", "shapley")
ROW_FIELDS = (
    "cost", "r_total", "r_interp", "r_dec", "precision", "recall", "f1", "n_proposed",
    "n_errors", "approvals_requested", "rejected", "rejected_correct", "conflicts_dropped",
)  # fmt: skip


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> ExperimentConfig:
    root = tmp_path_factory.mktemp("recorded")
    streams_dir = root / "streams"
    streams_dir.mkdir()
    for stream in generate(
        GeneratorConfig.model_validate({"name": "t", "n_streams": 2, "seed": 5})
    ):
        stream.save(streams_dir / f"{stream.id}.json")
    return ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(root / "runs"),
            "attribution": {"single_repair": True, "shapley_permutations": 3},
            "conditions": [
                {"name": "oracle"},
                {"name": "regex", "interpreter": "regex"},
                {
                    "name": "typed",
                    "interpreter": "typed_atomic",
                    "client": "fakes:zero_order_typed_client",
                },
                {
                    "name": "typed-over-oracle-state",
                    "interpreter": "oracle",
                    "decider": "typed_state",
                    "client": "fakes:zero_order_typed_client",
                },
            ],
        }
    )


@pytest.fixture(scope="module")
def run_dir(config: ExperimentConfig) -> Path:
    """One recorded run shared by the tests; none of them changes it."""
    return run_experiment(config, run_name="r")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_compiled_pair_is_recorded(config: ExperimentConfig, run_dir: Path) -> None:
    records = read_records(run_dir)
    rows = _jsonl(run_dir / "results.jsonl")
    assert set(records) == {(row["stream"], row["condition"]) for row in rows}
    for row in rows:
        record = records[(row["stream"], row["condition"])]
        assert record.interpreter == next(
            c.interpreter for c in config.conditions if c.name == row["condition"]
        )
        assert len(record.proposed) == row["n_proposed"]
        assert len(record.failures) == row["parse_failures"]
        assert len(record.admissibility) == len(record.proposed)
        assert sum(not a["admitted"] for a in record.admissibility) >= row["rejected"]
    assert any(record.proposed for record in records.values())


def test_replay_reproduces_scores_and_attribution_without_the_reader(
    config: ExperimentConfig, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = read_records(run_dir)

    def no_client(spec: str) -> Any:
        raise AssertionError(f"replay asked for a client: {spec}")

    monkeypatch.setattr(experiments, "load_factory", no_client)
    replay = replay_experiment(config, records)

    assert replay.missing == []
    recorded = {(r["stream"], r["condition"]): r for r in _jsonl(run_dir / "results.jsonl")}
    # The model decider's condition is not replayed; every solver-decided one is.
    assert {(r["stream"], r["condition"]) for r in replay.rows} == {
        key for key in recorded if key[1] != "typed-over-oracle-state"
    }
    for row in replay.rows:
        expected = recorded[(row["stream"], row["condition"])]
        assert {f: row[f] for f in ROW_FIELDS} == {f: expected[f] for f in ROW_FIELDS}

    def errors(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        return [
            (e["stream"], e["condition"], *(e[f] for f in ERROR_FIELDS))
            for e in rows
            if e["condition"] != "typed-over-oracle-state"
        ]

    recorded_errors = errors(_jsonl(run_dir / "errors.jsonl"))
    assert recorded_errors, "the test needs errors to attribute"
    assert errors(replay.errors) == recorded_errors


def test_missing_records_are_reported(config: ExperimentConfig, run_dir: Path) -> None:
    records = read_records(run_dir)
    dropped = next(key for key in records if key[1] == "regex")
    del records[dropped]
    assert replay_experiment(config, records).missing == [dropped]


def test_records_round_trip_and_reject_duplicates(run_dir: Path, tmp_path: Path) -> None:
    for record in read_records(run_dir).values():
        assert ReaderRecord.from_json(json.loads(json.dumps(record.to_json()))) == record
    text = (run_dir / RECORDS_FILE).read_text()
    path = tmp_path / RECORDS_FILE
    path.write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="two records"):
        read_records(path)
