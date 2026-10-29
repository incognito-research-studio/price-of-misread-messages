"""Scoring readers under the promising policy (scope 5.9-5.11).

A reader's interpretation loss is Π of the same promising policy given the gold deltas (the
gold-update reference) minus Π given the reader's deltas, on the same truth, supply and customer
orders. Errors come from the same alignment as the replenishment study, so the error ids match
across the two policies. Single repairs and Shapley values replay repaired deltas through the
unchanged front end, as :func:`cdd.experiments.score_compiled` does, with Π as the value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cdd.benchmark.stream import load_streams
from cdd.evaluation import (
    CachedCost,
    align,
    apply_repairs,
    prf,
    repairs_for,
    shapley_permutation,
    single_repair_effects,
)
from cdd.experiments import PLANNER_KEY, AttributionConfig, ExperimentConfig, StreamContext
from cdd.interpreters import Interpreter, OracleInterpreter, ReplayInterpreter
from cdd.pipeline import CompilingPolicy
from cdd.promising.execution import PromisingResult, PromisingSimulator
from cdd.promising.orders import CustomerOrder, generate_orders
from cdd.promising.params import PromisingParams, margins
from cdd.promising.supply import FixedPlanDecider, FixedSupply, reference_supply
from cdd.recorded import ReaderRecord


class PromisingContext:
    """What every promising condition of one stream shares: supply, orders and references."""

    def __init__(self, context: StreamContext, params: PromisingParams) -> None:
        self.context = context
        self.params = params
        world = context.stream.world
        self.supply: FixedSupply = reference_supply(context)
        self.margin = margins(world, params)
        self.orders: tuple[CustomerOrder, ...] = generate_orders(
            world, context.demand, context.stream.seed, params
        )
        self._references: dict[tuple[bool, bool], PromisingResult] = {}

    def run(
        self, interpreter: Interpreter, *, sop_gate: bool = True, versioning: bool = True
    ) -> tuple[PromisingResult, CompilingPolicy]:
        policy = self.context.compiling(
            interpreter,
            FixedPlanDecider(self.supply.plans),
            sop_gate=sop_gate,
            versioning=versioning,
        )
        simulator = PromisingSimulator(
            self.context.truth,
            self.supply,
            self.orders,
            self.params,
            self.margin,
            self.context.yields,
        )
        return simulator.run(policy), policy

    def reference(self, *, sop_gate: bool = True, versioning: bool = True) -> PromisingResult:
        """The gold deltas through the same front end and layers."""
        key = (sop_gate, versioning)
        if key not in self._references:
            gold = OracleInterpreter(self.context.stream.gold)
            self._references[key] = self.run(gold, sop_gate=sop_gate, versioning=versioning)[0]
        return self._references[key]


def score_promising(
    pc: PromisingContext,
    interpreter: Interpreter,
    *,
    condition: str,
    sop_gate: bool = True,
    versioning: bool = True,
    attribution: AttributionConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One reader on one stream: a results row and one row per delta error."""
    stream = pc.context.stream
    layers = {"sop_gate": sop_gate, "versioning": versioning}
    result, policy = pc.run(interpreter, **layers)
    reference = pc.reference(**layers)
    alignment = align(
        policy.proposed,
        stream.gold,
        horizon=(stream.world.start_period, stream.world.end_period),
        periods={message.id: message.period for message in stream.messages},
    )

    attribution = attribution or AttributionConfig()
    repairs = repairs_for(alignment)
    players = list(repairs)[: attribution.max_errors]
    single: dict[str, float] = {}
    shapley: dict[str, float] = {}
    repaired_gap: float | None = None
    if players and (attribution.single_repair or attribution.shapley_permutations > 0):

        def cost_with(repaired: frozenset[str]) -> float:
            deltas = apply_repairs(policy.proposed, [repairs[r] for r in sorted(repaired)])
            replay = ReplayInterpreter(deltas, name=f"{condition}-replay")
            return -pc.run(replay, **layers)[0].value

        cached = CachedCost(cost_with)
        if attribution.single_repair:
            single = single_repair_effects(players, cached)
        if attribution.shapley_permutations > 0:
            shapley = shapley_permutation(
                players, cached, attribution.shapley_permutations, rng or np.random.default_rng(0)
            )
        if len(players) == len(repairs):
            # The gate of scope 5.10: repairing every error gives the gold-update reference.
            repaired_gap = -cached(frozenset(players)) - reference.value
    elif not repairs:
        repaired_gap = result.value - reference.value

    scores = prf(alignment)
    metrics = {k: v for k, v in result.metrics.items() if k != "solver_statuses"}
    row: dict[str, Any] = {
        "stream": stream.id,
        "condition": condition,
        **metrics,
        **{f"value_{name}": amount for name, amount in result.components.items()},
        "reference_value": reference.value,
        "loss": reference.value - result.value,
        "repaired_gap": repaired_gap,
        "precision": scores.precision,
        "recall": scores.recall,
        "f1": scores.f1,
        "n_errors": len(alignment.errors),
        "solver_statuses": result.metrics.get("solver_statuses"),
    }
    errors = []
    for error_id, error in zip(repairs, alignment.errors, strict=True):
        errors.append(
            {
                "stream": stream.id,
                "condition": condition,
                "error_id": error_id,
                "kind": error.kind.value,
                "kinds": ";".join(kind.value for kind in error.kinds),
                "delta_type": error.delta_type.value,
                "predicted_id": None if error.predicted is None else error.predicted.id,
                "gold_id": None if error.gold is None else error.gold.id,
                "delta_c": single.get(error_id),
                "shapley": shapley.get(error_id),
            }
        )
    return row, errors


@dataclass(frozen=True)
class PromisingRun:
    rows: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    # (stream, condition) pairs of solver-decided conditions with no recorded output
    missing: list[tuple[str, str]]


def promising_experiment(
    config: ExperimentConfig,
    records: dict[tuple[str, str], ReaderRecord],
    params: PromisingParams,
    *,
    streams: str | Path | None = None,
    attribution: AttributionConfig | None = None,
    progress: Callable[[dict[str, Any], list[dict[str, Any]]], None] | None = None,
) -> PromisingRun:
    """Score every solver-decided condition of a recorded run under the promising policy.

    Streams, conditions and random generators follow :func:`cdd.experiments.run_experiment`, so
    Shapley samples are drawn as in the replenishment run. Readers are replayed from ``records``;
    no model is asked. ``attribution`` defaults to the recorded run's. ``progress`` gets each
    pair's row and errors as soon as they are scored.
    """
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for stream_index, stream in enumerate(load_streams(streams or config.streams)):
        pc: PromisingContext | None = None
        for condition_index, condition in enumerate(config.conditions):
            if condition.decider != PLANNER_KEY:
                continue
            record = records.get((stream.id, condition.name))
            if record is None:
                missing.append((stream.id, condition.name))
                continue
            pc = pc or PromisingContext(StreamContext(stream, config), params)
            row, condition_errors = score_promising(
                pc,
                record.replay(condition.name),
                condition=condition.name,
                sop_gate=condition.sop_gate,
                versioning=condition.versioning,
                attribution=attribution or config.attribution,
                rng=np.random.default_rng([config.seed, stream_index, condition_index]),
            )
            rows.append(row)
            errors.extend(condition_errors)
            if progress is not None:
                progress(row, condition_errors)
    return PromisingRun(rows, errors, missing)
