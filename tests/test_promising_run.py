"""Order promising on generated streams: properties of the runner and its attribution."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

import fakes
from cdd import experiments
from cdd.benchmark import GeneratorConfig, generate
from cdd.experiments import AttributionConfig, ExperimentConfig, StreamContext, run_experiment
from cdd.interpreters import AtomicQuestionInterpreter, OracleInterpreter, RegexInterpreter
from cdd.promising import PromisingParams, PromisingResult, execution
from cdd.promising.run import PromisingContext, promising_experiment, score_promising
from cdd.recorded import read_records

PARAMS = PromisingParams()


@pytest.fixture(scope="module")
def pc() -> PromisingContext:
    stream = generate(GeneratorConfig.model_validate({"name": "t", "n_streams": 1, "seed": 5}))[0]
    config = ExperimentConfig.model_validate(
        {"name": "t", "streams": ".", "conditions": [{"name": "oracle"}]}
    )
    return PromisingContext(StreamContext(stream, config), PARAMS)


def _value(result: PromisingResult, pc: PromisingContext) -> float:
    """Π recomputed from the per-order ledger, independently of the simulator's accounting."""
    end = pc.context.stream.world.end_period
    total = 0.0
    for o in result.outcomes:
        if o.promised is None:
            continue
        m = pc.margin[o.order.sku]
        delivered = sum(q for _, q in o.deliveries)
        missing = max(0.0, o.order.quantity - delivered)
        on_time = sum(q for w, q in o.deliveries if w <= o.promised)
        late = sum(q * max(0, w - o.order.requested) for w, q in o.deliveries)
        late += missing * max(0, end - o.order.requested)
        total += (
            m * delivered
            - PARAMS.lateness_fraction * m * late
            - PARAMS.broken_fraction * m * max(0.0, o.order.quantity - on_time)
            - PARAMS.terminal_fraction * m * missing
        )
    return total


def test_runs_are_deterministic_and_share_physical_supply(pc: PromisingContext) -> None:
    world = pc.context.stream.world
    gold, _ = pc.run(OracleInterpreter(pc.context.stream.gold))
    again, _ = pc.run(OracleInterpreter(pc.context.stream.gold))
    regex, _ = pc.run(RegexInterpreter(world))
    assert [(o.promised, o.deliveries) for o in gold.outcomes] == [
        (o.promised, o.deliveries) for o in again.outcomes
    ]
    assert gold.value == again.value == pc.reference().value
    assert regex.metrics["received_units"] == gold.metrics["received_units"] > 0
    assert gold.metrics["accepted_orders"] > 0 and gold.metrics["fallbacks"] == 0
    for result in (gold, regex):
        assert result.value == pytest.approx(_value(result, pc))
        # Nothing is delivered beyond the stock that ever existed.
        stock = sum(s.initial_stock for s in world.skus) + sum(lot.quantity for lot in world.lots)
        assert result.metrics["delivered_units"] <= stock + result.metrics["received_units"] + 1e-6
        for o in result.outcomes:
            assert sum(q for _, q in o.deliveries) <= o.order.quantity + 1e-6
            assert all(o.promised is not None and w >= o.promised for w, _ in o.deliveries)


def test_the_policy_sees_no_future_orders_and_keeps_its_commitments(
    pc: PromisingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, list[Any], list[Any]]] = []
    original = execution.promise

    def spy(view: Any, position: Any, commitments: Any, new_orders: Any, *rest: Any) -> Any:
        calls.append((position.period, list(new_orders), list(commitments)))
        return original(view, position, commitments, new_orders, *rest)

    monkeypatch.setattr(execution, "promise", spy)
    result, _ = pc.run(RegexInterpreter(pc.context.stream.world))
    outcome = {o.order.id: o for o in result.outcomes}
    promised_at: dict[str, int] = {}
    for t, new_orders, commitments in calls:
        assert all(order.arrival == t for order in new_orders)
        expected = set()
        for oid in promised_at:
            before = sum(q for w, q in outcome[oid].deliveries if w < t)
            if outcome[oid].order.quantity - before > 1e-6:
                expected.add(oid)
        assert {c.order.id for c in commitments} == expected
        for c in commitments:
            assert c.promised == outcome[c.order.id].promised  # never revised
        for order in new_orders:
            if outcome[order.id].promised is not None:
                promised_at[order.id] = t
    assert {t for t, _, _ in calls} == set(pc.context.stream.world.periods)


def test_repairing_every_error_reproduces_the_gold_reference(pc: PromisingContext) -> None:
    # The scripted typed reader always picks the first option, so it makes errors to repair.
    reader = AtomicQuestionInterpreter(fakes.zero_order_typed_client(), pc.context.stream.world)
    attribution = AttributionConfig(single_repair=True, shapley_permutations=3)
    row, errors = score_promising(pc, reader, condition="typed", attribution=attribution)
    assert row["n_errors"] == len(errors) > 0
    assert row["repaired_gap"] == pytest.approx(0.0, abs=1e-6)
    assert row["loss"] == pytest.approx(row["reference_value"] - row["value"])
    # Shapley values split the loss exactly.
    assert sum(e["shapley"] for e in errors) == pytest.approx(row["loss"], abs=1e-6)
    oracle_row, oracle_errors = score_promising(
        pc, OracleInterpreter(pc.context.stream.gold), condition="oracle"
    )
    assert oracle_row["loss"] == 0 and oracle_errors == []


def test_recorded_readers_are_scored_without_any_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    for stream in generate(
        GeneratorConfig.model_validate({"name": "t", "n_streams": 1, "seed": 5})
    ):
        stream.save(streams_dir / f"{stream.id}.json")
    config = ExperimentConfig.model_validate(
        {
            "name": "t",
            "streams": str(streams_dir),
            "output_dir": str(tmp_path / "runs"),
            "attribution": {"single_repair": False},
            "conditions": [
                {"name": "oracle"},
                {"name": "regex", "interpreter": "regex"},
                {
                    "name": "typed",
                    "interpreter": "typed_atomic",
                    "client": "fakes:zero_order_typed_client",
                },
            ],
        }
    )
    records = read_records(run_experiment(config, run_name="r"))

    def no_client(spec: str) -> Any:
        raise AssertionError(f"promising asked for a client: {spec}")

    monkeypatch.setattr(experiments, "load_factory", no_client)
    run = promising_experiment(
        config, records, PARAMS, attribution=AttributionConfig(single_repair=True)
    )
    assert run.missing == []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run.rows:
        by_condition[row["condition"]].append(row)
    assert set(by_condition) == {"oracle", "regex", "typed"}
    assert all(row["loss"] == 0 for row in by_condition["oracle"])
    assert all(row["repaired_gap"] == pytest.approx(0.0, abs=1e-6) for row in run.rows)
    assert {e["condition"] for e in run.errors} <= {"regex", "typed"}
