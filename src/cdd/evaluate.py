"""Evaluate your own reader and planner on benchmark streams, in one call.

You do not need the experiment runner, a config file or any model client. Bring:

- a **reader**: any object with a ``name`` and ``interpret(message, view) -> Interpretation``
  (see :mod:`cdd.interpreters.base`);
- optionally a **planner**: any object with a ``name`` and ``plan(view, position) -> Plan``
  (see :mod:`cdd.deciders`); the default is the MILP solver.

:func:`evaluate` runs them through the compiling pipeline on one stream and returns the cost,
the regret parts, extraction scores, and every delta error with the cost of repairing it, most
expensive first. Operational costs (holding, backlog, prices, capacities) come from the stream's
world, which the generator config controls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cdd.benchmark.stream import Stream
from cdd.deciders import Decider
from cdd.evaluation import RegretDecomposition, prf
from cdd.experiments import (
    AttributionConfig,
    ConditionConfig,
    ExperimentConfig,
    StreamContext,
    score_compiled,
)
from cdd.interpreters.base import Interpreter
from cdd.solver import MILPSolver


@dataclass(frozen=True)
class ErrorCost:
    kind: str
    delta_type: str
    predicted_id: str | None
    gold_id: str | None
    delta_c: float | None

    @property
    def message_id(self) -> str:
        return (self.gold_id or self.predicted_id or "").split("#")[0]


@dataclass(frozen=True)
class Evaluation:
    stream: str
    reader: str
    planner: str
    cost: float
    oracle_cost: float
    benchmark_cost: float
    regret: RegretDecomposition
    precision: float
    recall: float
    f1: float
    violations: int
    audit_completeness: float
    errors: tuple[ErrorCost, ...]


def default_config() -> ExperimentConfig:
    """The pilot's defaults: SOP v1 thresholds, magnitude-cap approver, tuned benchmark."""
    return ExperimentConfig(
        name="evaluate", streams=Path("."), conditions=(ConditionConfig(name="evaluate"),)
    )


def evaluate(
    stream: Stream,
    reader: Interpreter,
    planner: Decider | None = None,
    *,
    config: ExperimentConfig | None = None,
    attribute: bool = True,
) -> Evaluation:
    """Score a reader (and planner) on one stream; see the module docstring."""
    config = config or default_config()
    planner = planner or MILPSolver(
        horizon=config.solver.horizon,
        time_limit=config.solver.time_limit,
        safety_factor=config.solver.safety_factor,
    )
    context = StreamContext(stream, config)
    scored = score_compiled(
        context,
        reader,
        planner,
        decider_key=planner.name,
        attribution=AttributionConfig(single_repair=attribute),
    )
    errors = [
        ErrorCost(
            kind=error.kind.value,
            delta_type=error.delta_type.value,
            predicted_id=None if error.predicted is None else error.predicted.id,
            gold_id=None if error.gold is None else error.gold.id,
            delta_c=scored.single.get(error_id),
        )
        for error_id, error in zip(scored.repairs, scored.alignment.errors, strict=True)
    ]
    errors.sort(key=lambda e: -(e.delta_c or 0.0))
    scores = prf(scored.alignment)
    return Evaluation(
        stream=stream.id,
        reader=reader.name,
        planner=planner.name,
        cost=scored.result.total_cost,
        oracle_cost=scored.result.total_cost - scored.regret.interpretation,
        benchmark_cost=context.benchmark_cost(),
        regret=scored.regret,
        precision=scores.precision,
        recall=scores.recall,
        f1=scores.f1,
        violations=len(scored.result.violations),
        audit_completeness=scored.policy.audit.completeness(),
        errors=tuple(errors),
    )


def report(evaluations: Sequence[Evaluation], top_errors: int = 5) -> str:
    """A plain-text summary: totals per stream and the costliest errors overall."""
    lines = [
        f"{'stream':<14} {'reader':<14} {'planner':<12} {'cost':>10} {'R_interp':>10} "
        f"{'R_dec':>10} {'F1':>5} {'viol':>4}"
    ]
    for e in evaluations:
        lines.append(
            f"{e.stream:<14} {e.reader:<14} {e.planner:<12} {e.cost:>10,.0f} "
            f"{e.regret.interpretation:>+10,.0f} {e.regret.decision:>+10,.0f} {e.f1:>5.2f} "
            f"{e.violations:>4}"
        )
    costly = sorted(
        (error for e in evaluations for error in e.errors if error.delta_c),
        key=lambda error: -(error.delta_c or 0.0),
    )[:top_errors]
    if costly:
        lines.append("\nCostliest errors (cost saved by repairing each one alone):")
        lines.extend(
            f"  {error.delta_c:>+10,.0f}  {error.kind:<12} {error.delta_type:<18} "
            f"message {error.message_id}"
            for error in costly
        )
    return "\n".join(lines)
