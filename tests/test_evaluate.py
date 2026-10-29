"""The plug-in API, the example, solver tuning, the out-of-schema slice and Laya export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cdd.adapters.laya import laya_questions, training_rows
from cdd.adapters.ollama import OllamaClient
from cdd.benchmark import GeneratorConfig, generate, generate_stream
from cdd.evaluate import default_config, evaluate, report
from cdd.experiments import tune_safety_factor
from cdd.interpreters import RegexInterpreter
from cdd.interpreters.typed import TypedQuestion, type_question
from cdd.orders import InventoryPosition, OrderLine, Plan
from cdd.pipeline import build_truth
from cdd.simulator import InventorySimulator
from cdd.world import Lane
from helpers import build_world

ROOT = Path(__file__).resolve().parents[1]
SMALL = GeneratorConfig.model_validate({"name": "t", "n_streams": 2, "seed": 5})


def test_evaluate_scores_a_reader_and_sorts_errors_by_cost() -> None:
    stream = generate_stream(SMALL, 0)
    result = evaluate(stream, RegexInterpreter(stream.world))
    assert result.planner == "milp" and result.f1 > 0.9
    assert result.cost == pytest.approx(result.oracle_cost + result.regret.interpretation)
    costs = [e.delta_c or 0.0 for e in result.errors]
    assert costs == sorted(costs, reverse=True)
    assert "stream" in report([result])


def test_the_plug_in_example_runs() -> None:
    spec = importlib.util.spec_from_file_location(
        "example", ROOT / "examples" / "evaluate_your_own_reader.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.main(n_streams=1)
    assert "hold-reader" in text and "order-up-to" in text


def test_tune_safety_factor_ranks_the_grid() -> None:
    ranking = tune_safety_factor(default_config(), generate(SMALL), [0.0, 0.2])
    assert {factor for factor, _ in ranking} == {0.0, 0.2}
    assert ranking[0][1] <= ranking[1][1]
    with pytest.raises(ValueError, match="development"):
        tune_safety_factor(default_config(), [])


def test_out_of_schema_slice() -> None:
    plain = generate_stream(SMALL, 0)
    assert plain.shocks == ()
    explicit_off = GeneratorConfig.model_validate(
        {"name": "t", "n_streams": 2, "seed": 5, "out_of_schema_rate": 0.0}
    )
    assert generate_stream(explicit_off, 0) == plain  # the slice leaves other streams untouched
    on = GeneratorConfig.model_validate(
        {"name": "t", "n_streams": 2, "seed": 5, "out_of_schema_rate": 1.0}
    )
    stream = generate_stream(on, 0)
    assert stream.shocks
    for shock in stream.shocks:
        assert stream.gold_for(shock.message_id) == ()  # no delta type expresses it
        assert 0 < shock.factor < 1
    assert stream.yields()


def test_yields_shrink_deliveries_but_not_the_planners_view() -> None:
    world = build_world(
        lanes=(
            Lane(id="LANE-X", name="Port X", plant="P2", lead_time_days=14.0),
            Lane(id="LANE-Y", name="Port Y", plant="P3", lead_time_days=7.0),
        )
    )
    truth = build_truth(world, [], []).view(None)
    demand = {("SKU-01", p): 0.0 for p in world.periods}
    seen: list[float] = []

    def policy(position: InventoryPosition) -> Plan:
        seen.extend(o.quantity for o in position.outstanding)
        orders = (OrderLine("P2", "SKU-01", 100.0),) if position.period == 40 else ()
        return Plan(period=position.period, orders=orders)

    simulator = InventorySimulator(truth, demand, yields={("P2", 40): 0.5})
    result = simulator.run(policy)
    assert seen[0] == 100.0  # the open order shows what was accepted, not what will arrive
    stock_after = InventorySimulator(truth, demand).run(policy)
    assert result.holding_cost < stock_after.holding_cost  # half the order never arrived


def test_laya_rows_match_the_questions_asked_at_inference() -> None:
    rows = list(training_rows(generate(SMALL)))
    groups = {tuple(sorted(row["questions"])) for row in rows}
    assert ("type",) in groups and ("entity",) in groups
    assert any("span" in group and "window_start" in group for group in groups)
    for row in rows:
        for qid, answer in row["expected"].items():
            assert answer in row["questions"][qid]["criteria"], (qid, answer)
    question, _ = type_question("names")
    rendered = laya_questions([question])["type"]
    assert rendered["type"] == "choice" and len(rendered["criteria"]) == len(question.options)
    json.dumps(rows[:5])  # serialisable


def test_ollama_prompt_shows_option_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []

    def fake_chat(self: OllamaClient, messages: list[dict[str, str]], schema: object) -> str:
        sent.append(messages[1]["content"])
        return json.dumps({"q": "A"})

    monkeypatch.setattr(OllamaClient, "_chat", fake_chat)
    client = OllamaClient("any", check=False)
    client.ask("ctx", [TypedQuestion("q", "Which?", ("A", "B"), ("first thing", "second thing"))])
    assert '"A": first thing' in sent[0]
