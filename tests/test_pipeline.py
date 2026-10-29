"""Integration tests, including the two properties that hold by construction (Section 3.9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cdd.benchmark import GeneratorConfig, Stream, generate_stream
from cdd.deciders import LLMStreamDecider, TypedStateDecider, TypedStreamDecider
from cdd.evaluation import ErrorKind, align
from cdd.experiments import ConditionConfig, ExperimentConfig, StreamContext
from cdd.interpreters import OracleInterpreter, RegexInterpreter, ReplayInterpreter
from cdd.pipeline import CompilingPolicy, DirectPolicy, build_truth
from cdd.schema import Span, Window
from cdd.simulator import InventorySimulator
from cdd.solver import MILPSolver
from cdd.world import World
from fakes import ScriptedLLMClient, ScriptedTypedClient
from helpers import make_delta, make_message


@pytest.fixture(scope="module")
def stream() -> Stream:
    config = GeneratorConfig.model_validate({"name": "t", "seed": 11, "events": {"rate": 1.5}})
    return generate_stream(config, 0)


@pytest.fixture
def context(stream: Stream) -> StreamContext:
    config = ExperimentConfig(
        name="t", streams=Path("."), conditions=(ConditionConfig(name="oracle"),)
    )
    return StreamContext(stream, config)


def test_oracle_condition_has_no_interpretation_regret(
    stream: Stream, context: StreamContext
) -> None:
    policy = context.compiling(OracleInterpreter(stream.gold), MILPSolver())
    result = context.simulator.run(policy)
    assert result.total_cost == context.oracle_cost("milp", MILPSolver())
    assert result.violations == ()
    assert policy.audit.completeness() == 1.0
    assert align(policy.proposed, stream.gold).errors == ()


def test_identical_delta_sequences_give_identical_decisions(
    stream: Stream, context: StreamContext
) -> None:
    """Proposition 2(i): the pipeline after the interpreter is deterministic."""
    runs = [
        context.simulator.run(context.compiling(RegexInterpreter(stream.world), MILPSolver()))
        for _ in range(2)
    ]
    assert runs[0].order_vector() == runs[1].order_vector()


def test_replaying_proposed_deltas_reproduces_the_run(
    stream: Stream, context: StreamContext
) -> None:
    """Attribution relies on this: repairing nothing must leave the cost unchanged."""
    policy = context.compiling(RegexInterpreter(stream.world), MILPSolver())
    cost = context.simulator.run(policy).total_cost
    replay = ReplayInterpreter(policy.proposed)
    assert context.simulator.run(context.compiling(replay, MILPSolver())).total_cost == cost


def test_true_violation_traces_to_a_delta_error(world: World) -> None:
    """Proposition 1: an executed order that breaks a true constraint traces to a delta error."""
    text = "Plant 2 output reduced ~95% wk40–45."
    message = make_message(text, id="m1", period=40)
    cut = make_delta(
        id="m1#g0", value=0.05, window=Window(start=40, end=45),
        src=Span(message_id="m1", start=0, end=len(text)),
    )  # fmt: skip
    truth = build_truth(world, [message], [cut]).view(None)
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    policy = CompilingPolicy(world, [message], ReplayInterpreter([]), MILPSolver())
    result = InventorySimulator(truth, demand).run(policy)
    assert result.violations and {v.entity for v in result.violations} == {"P2"}
    (error,) = align(policy.proposed, [cut]).errors
    assert error.kind is ErrorKind.MISSED and error.gold is not None and error.gold.entity == "P2"


def test_model_backed_deciders(world: World) -> None:
    truth = build_truth(world, [], []).view(None)
    demand = {("SKU-01", p): 100.0 for p in world.periods}
    simulator = InventorySimulator(truth, demand)

    nothing = simulator.run(DirectPolicy([], TypedStreamDecider(ScriptedTypedClient(), world)))
    assert all(record.ordered == () for record in nothing.periods)

    picks = ScriptedTypedClient(lambda q: "100" if q.key.startswith("P2|") else "0")
    typed = CompilingPolicy(world, [], OracleInterpreter([]), TypedStateDecider(picks))
    assert simulator.run(typed).periods[0].ordered[0].quantity == 100.0

    reply = json.dumps({"orders": [{"plant": "P3", "sku": "SKU-01", "quantity": 120}]})
    llm = LLMStreamDecider(ScriptedLLMClient(lambda s, u: reply), world)
    assert simulator.run(DirectPolicy([], llm)).periods[0].ordered[0].plant == "P3"

    broken = LLMStreamDecider(ScriptedLLMClient(lambda s, u: "I would order some."), world)
    assert simulator.run(DirectPolicy([], broken)).periods[0].plan_status == "malformed"
