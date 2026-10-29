"""Experiment runner: every condition on every stream, scored against the oracle and x*.

For each stream the runner builds the truth from the gold deltas, draws one demand path (common
random numbers), and computes the benchmark x* and the hindsight optimum x^H. With the default
``planner`` benchmark, x* is the configured solver (its safety factor tuned on development
streams and fixed in advance) deciding from the oracle state, so decision regret is zero for the
solver and, for a model decider, is its extra cost over the solver on the same state. The gap
from x* to x^H is reported as the value of perfect information. (``tuned`` picks the best of a
set of safety factors per stream after seeing its cost, and ``hindsight`` uses x^H; both are
kept to reproduce earlier runs.) It runs the oracle condition once per decider and pipeline
variant as the reference for interpretation regret; when the condition's reader is the oracle,
its own run is that reference. Each configured
condition then runs through the same simulator. Model-backed components get their client from
a factory named in the config (``"package.module:factory"``); a condition whose client is
missing is skipped and listed in ``skipped.json``.

Each finished (stream, condition) pair is appended to ``results.jsonl`` and ``errors.jsonl`` as
soon as it completes, and a compiled pair's completed deltas to ``proposals.jsonl``
(:mod:`cdd.recorded`), so later analyses can replay a reader's output without asking it again.
A pair that raises is logged, listed in ``failed.json`` and passed over, so one bad request
cannot stop a long run; ``resume`` reruns only the pairs a run folder lacks.
With ``log_requests``, every model request is logged in the run folder; with ``reply_cache``,
model replies are stored on disk and reused for identical requests (``cdd.adapters.cache``).
``models.json`` records each model client's provenance (name, digest, server version).
"""

from __future__ import annotations

import csv
import importlib
import json
import logging
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cdd.benchmark.stream import Stream, load_streams
from cdd.deciders import (
    Decider,
    LLMStateDecider,
    LLMStreamDecider,
    StreamDecider,
    TypedStateDecider,
    TypedStreamDecider,
)
from cdd.evaluation import (
    Alignment,
    CachedCost,
    RegretDecomposition,
    Repair,
    align,
    apply_repairs,
    decompose,
    expected_calibration_error,
    prf,
    repairs_for,
    shapley_permutation,
    single_repair_effects,
)
from cdd.interpreters import (
    AtomicQuestionInterpreter,
    GoldSpanInterpreter,
    HybridInterpreter,
    Interpreter,
    JsonLLMInterpreter,
    OracleInterpreter,
    RegexInterpreter,
    ReplayInterpreter,
)
from cdd.interpreters.typed import LabelMode, WindowRule
from cdd.pipeline import CompilingPolicy, DirectPolicy, PlannedPolicy, build_truth
from cdd.recorded import RECORDS_FILE, ReaderRecord
from cdd.schema import DeltaType
from cdd.simulator import InventorySimulator, SimulationResult
from cdd.solver import MILPSolver, hindsight_plans
from cdd.sop import SOPConfig, SOPGraph, make_approver

log = logging.getLogger(__name__)

InterpreterKind = Literal[
    "none", "oracle", "gold_span", "regex", "llm_json", "typed_atomic", "hybrid"
]
DeciderKind = Literal["milp", "typed_state", "llm_state", "llm_stream", "typed_stream"]
_STREAM_DECIDERS = ("llm_stream", "typed_stream")


class ConditionConfig(BaseModel):
    """One experimental condition: a state source (interpreter or raw stream) and a decider."""

    model_config = ConfigDict(frozen=True)

    name: str
    interpreter: InterpreterKind | None = "oracle"
    decider: DeciderKind = "milp"
    client: str | None = None
    client_args: dict[str, Any] = Field(default_factory=dict)
    decider_client: str | None = None
    decider_client_args: dict[str, Any] = Field(default_factory=dict)
    context: str = "full"
    constrained: bool = False
    bucket: float = 25.0
    # Typed questions: option labels and window rule (Section 6 ablations).
    labels: LabelMode = "names"
    window_rule: WindowRule = "parser_first"
    # Layer ablations.
    sop_gate: bool = True
    versioning: bool = True
    # Hybrid (j): `client` serves the typed primary, `fallback_client` the LLM interpreter.
    fallback_client: str | None = None
    fallback_client_args: dict[str, Any] = Field(default_factory=dict)
    threshold: float = 0.8
    escalate_types: tuple[DeltaType, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> Self:
        streaming = self.decider in _STREAM_DECIDERS
        if streaming and self.interpreter is not None:
            raise ValueError(
                f"{self.name}: stream deciders read raw messages; set interpreter: null"
            )
        if not streaming and self.interpreter is None:
            raise ValueError(f"{self.name}: decider {self.decider} needs an interpreter")
        return self


class SolverConfig(BaseModel):
    """The solver π. Tune ``safety_factor`` on development streams and pre-register it;
    otherwise decision regret mostly measures an under-buffered point-forecast solver."""

    model_config = ConfigDict(frozen=True)

    horizon: int = 6
    time_limit: float = 10.0
    compute_duals: bool = False
    safety_factor: float = 0.0


class AttributionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    single_repair: bool = True
    shapley_permutations: int = 0
    max_errors: int = 200


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    streams: Path
    output_dir: Path = Path("runs")
    seed: int = 0
    demand_cv: float = 0.2
    solver: SolverConfig = SolverConfig()
    sop: Path | None = None
    approver: Literal["always_approve", "always_reject", "magnitude_cap"] = "magnitude_cap"
    approver_cap: float = 0.5
    hindsight_time_limit: float = 60.0
    benchmark: Literal["planner", "tuned", "hindsight"] = "planner"
    # The grid of `cdd tune-solver`, and of the per-stream `tuned` benchmark.
    tuned_safety_factors: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    attribution: AttributionConfig = AttributionConfig()
    log_requests: bool = True
    reply_cache: Path | None = None
    conditions: tuple[ConditionConfig, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))


class ConditionUnavailableError(RuntimeError):
    """A condition cannot run in this environment, usually because no client is configured."""


def load_factory(spec: str) -> Callable[..., Any]:
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"client must look like 'package.module:factory', got {spec!r}")
    factory: object = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable")
    return factory


_USAGE = {
    "requests": "model_requests",
    "seconds": "model_seconds",
    "truncations": "model_truncations",
    "retried": "model_retries",
    "capped": "model_capped_replies",
    "cache_hits": "model_cache_hits",
}


class _Clients:
    """Builds each configured client once and reuses it across streams.

    With ``log_dir``, a client that has a ``log_path`` attribute (the Ollama client) logs every
    request to ``requests-<model>.jsonl`` there, each record tagged with the stream and
    condition that made it.
    """

    def __init__(self, log_dir: Path | None = None, reply_cache: Path | None = None) -> None:
        self._cache: dict[str, Any] = {}
        self.log_dir = log_dir
        self.reply_cache = reply_cache
        self._tag: str | None = None

    def tag(self, label: str) -> None:
        self._tag = label
        for client in self._cache.values():
            if hasattr(client, "log_tag"):
                client.log_tag = label

    def usage(self) -> dict[str, float]:
        """Counters summed over clients that report them: requests, model seconds, truncated
        inputs (Laya), and retried requests and replies cut by ``max_tokens`` (Ollama)."""
        clients = list(self._cache.values())
        return {key: sum(float(getattr(c, key, 0)) for c in clients) for key in _USAGE}

    def usage_since(self, before: dict[str, float]) -> dict[str, Any]:
        now = self.usage()
        return {
            name: (now[key] - before[key]) if key == "seconds" else int(now[key] - before[key])
            for key, name in _USAGE.items()
        }

    def get(self, spec: str | None, args: dict[str, Any], who: str) -> Any:
        if spec is None:
            raise ConditionUnavailableError(
                f"{who} needs a client; set it in the experiment config"
            )
        key = json.dumps([spec, args], sort_keys=True, default=str)
        if key not in self._cache:
            client = load_factory(spec)(**args)
            if self.log_dir is not None and getattr(client, "log_path", False) is None:
                model = re.sub(r"[^A-Za-z0-9._-]+", "-", str(getattr(client, "model", "client")))
                client.log_path = self.log_dir / f"requests-{model}.jsonl"
            if hasattr(client, "log_tag"):
                client.log_tag = self._tag
            if self.reply_cache is not None and hasattr(client, "cache_path"):
                client.cache_path = self.reply_cache
            self._cache[key] = client
        return self._cache[key]

    def provenance(self) -> list[dict[str, Any]]:
        """What each client reports about itself (model, digest, revision), once per client."""
        out = []
        for client in self._cache.values():
            describe = getattr(client, "provenance", None)
            info = describe() if callable(describe) else {"client": type(client).__name__}
            out.append({"name": getattr(client, "name", None), **info})
        return out


class StreamContext:
    """Everything the conditions of one stream share: truth, demand, x* and oracle references."""

    def __init__(self, stream: Stream, config: ExperimentConfig) -> None:
        self.stream = stream
        self.config = config
        self.truth = build_truth(stream.world, stream.messages, stream.gold).view(as_of=None)
        self.demand = InventorySimulator.sample_demand(self.truth, stream.seed, config.demand_cv)
        self.yields = stream.yields()
        self.simulator = InventorySimulator(self.truth, self.demand, self.yields)
        self.sop = SOPGraph(SOPConfig.from_yaml(config.sop) if config.sop else None)
        self._hindsight: float | None = None
        self._tuned: tuple[float, float] | None = None
        self._oracle: dict[tuple[str, bool, bool], float] = {}
        self._planner = _solver(config)

    def compiling(
        self,
        interpreter: Interpreter,
        decider: Decider,
        *,
        sop_gate: bool = True,
        versioning: bool = True,
    ) -> CompilingPolicy:
        return CompilingPolicy(
            self.stream.world,
            self.stream.messages,
            interpreter,
            decider,
            sop=self.sop,
            approver=make_approver(self.config.approver, self.config.approver_cap),
            sop_gate=sop_gate,
            versioning=versioning,
        )

    def hindsight_cost(self) -> float:
        """C(x^H): one MILP over the whole horizon with true parameters and realised demand."""
        if self._hindsight is None:
            plans = hindsight_plans(
                self.truth,
                self.simulator.initial_position(),
                self.demand,
                time_limit=self.config.hindsight_time_limit,
                yields=self.yields,
            )
            self._hindsight = self.simulator.run(PlannedPolicy(plans)).total_cost
        return self._hindsight

    def tuned(self) -> tuple[float, float]:
        """C(x*) and the safety factor that achieves it: the best solver variant that sees
        only the oracle state, period by period."""
        if self._tuned is None:
            solver = self.config.solver
            costs = []
            for factor in self.config.tuned_safety_factors:
                variant = MILPSolver(
                    horizon=solver.horizon, time_limit=solver.time_limit, safety_factor=factor
                )
                policy = self.compiling(OracleInterpreter(self.stream.gold), variant)
                costs.append((self.simulator.run(policy).total_cost, factor))
            self._tuned = min(costs)
        return self._tuned

    def planner_cost(self) -> float:
        """C(x*) under the ``planner`` benchmark: the configured solver on the oracle state,
        through the full pipeline."""
        return self.oracle_cost(PLANNER_KEY, self._planner)

    def benchmark_cost(self) -> float:
        if self.config.benchmark == "hindsight":
            return self.hindsight_cost()
        if self.config.benchmark == "tuned":
            return self.tuned()[0]
        return self.planner_cost()

    def has_oracle(self, key: str, *, sop_gate: bool = True, versioning: bool = True) -> bool:
        return (key, sop_gate, versioning) in self._oracle

    def set_oracle(
        self, key: str, cost: float, *, sop_gate: bool = True, versioning: bool = True
    ) -> None:
        """Record a run of the oracle reader as the reference for its decider and layers."""
        self._oracle.setdefault((key, sop_gate, versioning), cost)

    def oracle_cost(
        self, key: str, decider: Decider, *, sop_gate: bool = True, versioning: bool = True
    ) -> float:
        """C(x^π(s*)) for a decider and pipeline variant: gold deltas through the same layers."""
        cache_key = (key, sop_gate, versioning)
        if cache_key not in self._oracle:
            policy = self.compiling(
                OracleInterpreter(self.stream.gold),
                decider,
                sop_gate=sop_gate,
                versioning=versioning,
            )
            self._oracle[cache_key] = self.simulator.run(policy).total_cost
        return self._oracle[cache_key]

    def benchmarks(self) -> dict[str, Any]:
        """Benchmark fields reported with every row."""
        fields: dict[str, Any] = {
            "benchmark": self.config.benchmark,
            "hindsight_cost": self.hindsight_cost(),
            "out_of_schema_events": len(self.stream.shocks),
        }
        if self.config.benchmark == "planner":
            fields |= {
                "planner_cost": self.planner_cost(),
                "planner_safety_factor": self.config.solver.safety_factor,
                "value_of_perfect_information": self.planner_cost() - self.hindsight_cost(),
            }
        elif self.config.benchmark == "tuned":
            cost, factor = self.tuned()
            fields |= {
                "tuned_cost": cost,
                "tuned_safety_factor": factor,
                "value_of_perfect_information": cost - self.hindsight_cost(),
            }
        return fields


def _interpreter(condition: ConditionConfig, stream: Stream, clients: _Clients) -> Interpreter:
    kind = condition.interpreter
    if kind == "none":
        # The no-update baseline: the planner ignores every message and plans on the master
        # data, learning of a suspension only when a refused order reports it.
        return ReplayInterpreter((), name=condition.name)
    if kind == "oracle":
        return OracleInterpreter(stream.gold)
    if kind == "gold_span":
        # Parser isolation (WP2): gold type, entity and span; the parser reads the rest.
        return GoldSpanInterpreter(stream.gold)
    if kind == "regex":
        return RegexInterpreter(stream.world)
    client = clients.get(condition.client, condition.client_args, condition.name)
    if kind == "llm_json":
        return JsonLLMInterpreter(client, name=condition.name, constrained=condition.constrained)
    typed = AtomicQuestionInterpreter(
        client,
        stream.world,
        name=condition.name,
        labels=condition.labels,
        window_rule=condition.window_rule,
    )
    if kind == "typed_atomic":
        return typed
    fallback_client = clients.get(
        condition.fallback_client, condition.fallback_client_args, f"{condition.name} fallback"
    )
    fallback = JsonLLMInterpreter(
        fallback_client, name=f"{condition.name}-fallback", constrained=condition.constrained
    )
    return HybridInterpreter(
        typed,
        fallback,
        threshold=condition.threshold,
        escalate_types=condition.escalate_types,
        name=condition.name,
    )


PLANNER_KEY = "milp"


def _decider_key(condition: ConditionConfig) -> str:
    """Names a decider for caching its oracle reference: the solver is one decider whatever
    the condition; a model decider is its kind plus its client and options."""
    if condition.decider == "milp":
        return PLANNER_KEY
    spec = condition.decider_client or condition.client
    args = condition.decider_client_args if condition.decider_client else condition.client_args
    options = {"constrained": condition.constrained, "bucket": condition.bucket}
    return json.dumps([condition.decider, spec, args, options], sort_keys=True, default=str)


def _solver(config: ExperimentConfig) -> MILPSolver:
    return MILPSolver(
        horizon=config.solver.horizon,
        time_limit=config.solver.time_limit,
        compute_duals=config.solver.compute_duals,
        safety_factor=config.solver.safety_factor,
    )


def _decider_client(condition: ConditionConfig, clients: _Clients) -> Any:
    spec = condition.decider_client or condition.client
    args = condition.decider_client_args if condition.decider_client else condition.client_args
    return clients.get(spec, args, condition.name)


def _state_decider(
    condition: ConditionConfig, config: ExperimentConfig, clients: _Clients
) -> Decider:
    if condition.decider == "milp":
        return _solver(config)
    client = _decider_client(condition, clients)
    if condition.decider == "llm_state":
        return LLMStateDecider(
            client, horizon=config.solver.horizon, constrained=condition.constrained
        )
    return TypedStateDecider(client, bucket=condition.bucket, horizon=config.solver.horizon)


def _stream_decider(condition: ConditionConfig, stream: Stream, clients: _Clients) -> StreamDecider:
    client = _decider_client(condition, clients)
    if condition.decider == "llm_stream":
        return LLMStreamDecider(
            client, stream.world, context=condition.context, constrained=condition.constrained
        )
    return TypedStreamDecider(
        client, stream.world, bucket=condition.bucket, context=condition.context
    )


def _outcome(result: SimulationResult) -> dict[str, Any]:
    return {
        "cost": result.total_cost,
        "purchase_cost": result.purchase_cost,
        "holding_cost": result.holding_cost,
        "backlog_cost": result.backlog_cost,
        "fill_rate": result.fill_rate,
        "violations_true": len(result.violations),
        "invalid_order_lines": result.invalid_lines,
    }


@dataclass(frozen=True)
class CompiledScore:
    """One compiled condition on one stream: the run, its regret parts, alignment and
    repair-based attribution (``single`` holds ΔC_j, ``shapley`` the sampled values)."""

    result: SimulationResult
    policy: CompilingPolicy
    regret: RegretDecomposition
    alignment: Alignment
    repairs: dict[str, Repair]
    single: dict[str, float]
    shapley: dict[str, float]
    seconds: float
    model_usage: dict[str, Any]


def score_compiled(
    context: StreamContext,
    interpreter: Interpreter,
    decider: Decider,
    *,
    decider_key: str,
    sop_gate: bool = True,
    versioning: bool = True,
    attribution: AttributionConfig | None = None,
    rng: np.random.Generator | None = None,
    clients: _Clients | None = None,
    reference: bool = False,
) -> CompiledScore:
    """Run a reader and a planner through the compiling pipeline and score them.

    ``decider_key`` names the decider for caching its oracle reference; use a different key for
    every distinct planner. With ``reference`` (the reader is the oracle), this run becomes the
    reference unless one is cached, so a model decider's own nondeterminism cannot show up as
    interpretation regret. Model usage covers the main run only, not references or replays.
    """
    stream = context.stream
    layers = {"sop_gate": sop_gate, "versioning": versioning}
    usage = clients.usage() if clients else {}
    started = time.perf_counter()
    policy = context.compiling(interpreter, decider, **layers)
    result = context.simulator.run(policy)
    seconds = time.perf_counter() - started
    model_usage = clients.usage_since(usage) if clients else {}
    if reference:
        context.set_oracle(decider_key, result.total_cost, **layers)
    regret = decompose(
        result.total_cost,
        context.oracle_cost(decider_key, decider, **layers),
        context.benchmark_cost(),
    )
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
    if players and (attribution.single_repair or attribution.shapley_permutations > 0):

        def cost_with(repaired: frozenset[str]) -> float:
            deltas = apply_repairs(policy.proposed, [repairs[r] for r in sorted(repaired)])
            replay = ReplayInterpreter(deltas, name=f"{interpreter.name}-replay")
            replay_policy = context.compiling(replay, decider, **layers)
            return context.simulator.run(replay_policy).total_cost

        cached = CachedCost(cost_with)
        if attribution.single_repair:
            single = single_repair_effects(players, cached)
        if attribution.shapley_permutations > 0:
            shapley = shapley_permutation(
                players, cached, attribution.shapley_permutations, rng or np.random.default_rng(0)
            )
    return CompiledScore(
        result, policy, regret, alignment, repairs, single, shapley, seconds, model_usage
    )


def run_condition(
    context: StreamContext,
    condition: ConditionConfig,
    clients: _Clients,
    rng: np.random.Generator,
    *,
    interpreter: Interpreter | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], ReaderRecord | None]:
    """Run and score one pair. ``interpreter`` replaces the condition's own reader, e.g. a
    replay of recorded output; the record is None for stream deciders, which compile nothing."""
    config, stream = context.config, context.stream
    base = {
        "stream": stream.id,
        "condition": condition.name,
        "interpreter": condition.interpreter,
        "decider": condition.decider,
    }

    if condition.decider in _STREAM_DECIDERS:
        stream_decider = _stream_decider(condition, stream, clients)
        usage = clients.usage()
        started = time.perf_counter()
        result = context.simulator.run(DirectPolicy(stream.messages, stream_decider))
        seconds = time.perf_counter() - started
        model = clients.usage_since(usage)
        cost = result.total_cost
        row = {
            **base,
            **_outcome(result),
            **context.benchmarks(),
            "r_total": cost - context.benchmark_cost(),
            "regret_vs_oracle": cost - context.planner_cost(),
            "seconds": seconds,
            **model,
        }
        return row, [], None

    interpreter = interpreter or _interpreter(condition, stream, clients)
    state_decider = _state_decider(condition, config, clients)
    scored = score_compiled(
        context,
        interpreter,
        state_decider,
        decider_key=_decider_key(condition),
        sop_gate=condition.sop_gate,
        versioning=condition.versioning,
        attribution=config.attribution,
        rng=rng,
        clients=clients,
        reference=condition.interpreter == "oracle",
    )
    policy, alignment, regret = scored.policy, scored.alignment, scored.regret
    scores = prf(alignment)
    sop_records = [r for r in policy.audit.records if r["kind"] == "sop"]
    row = {
        **base,
        **_outcome(scored.result),
        "r_total": regret.total,
        "r_interp": regret.interpretation,
        "r_dec": regret.decision,
        "interp_share": regret.interpretation_share,
        "precision": scores.precision,
        "recall": scores.recall,
        "f1": scores.f1,
        "n_proposed": scores.n_predicted,
        "n_gold": scores.n_gold,
        "n_errors": len(alignment.errors),
        "parse_failures": len(policy.failures),
        "approvals_requested": sum(r["verdict"] == "approve" for r in sop_records),
        "rejected": sum(r["verdict"] == "reject" for r in sop_records),
        "conflicts_dropped": policy.conflicts_dropped,
        "rejected_correct": _rejected_correct(policy, alignment),
        "audit_completeness": policy.audit.completeness(),
        "calibration_ece": expected_calibration_error(alignment),
        "escalation_share": getattr(interpreter, "escalation_share", None),
        **context.benchmarks(),
        "seconds": scored.seconds,
        **scored.model_usage,
    }
    repairs, single, shapley = scored.repairs, scored.single, scored.shapley

    errors = []
    for error_id, error in zip(repairs, alignment.errors, strict=True):
        errors.append(
            {
                "stream": stream.id,
                "condition": condition.name,
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
    record = ReaderRecord.from_policy(
        stream.id, condition.name, str(condition.interpreter), scored.policy
    )
    return row, errors, record


def _rejected_correct(policy: CompilingPolicy, alignment: Alignment) -> int:
    """Correct reads the SOP layer did not admit. Alignment scores every parsed delta, so these
    count as exact although they never reached the state."""
    blocked = {
        r["delta"]["id"] for r in policy.audit.records if r["kind"] == "sop" and not r["admitted"]
    }
    return sum(pred.id in blocked for pred, _ in alignment.exact)


def run_experiment(
    config: ExperimentConfig, run_name: str | None = None, *, resume: bool = False
) -> Path:
    """Run every condition on every stream and return the run folder.

    With ``resume``, continue the run folder ``run_name``: pairs already in its
    ``results.jsonl`` are kept and the rest (including pairs that failed) are run. The config
    must be the one the run started with. Each pair draws from its own random generator,
    seeded by the config seed and the pair's position, so a resumed run gives the same numbers
    as an uninterrupted one.
    """
    streams = load_streams(config.streams)
    config_json = json.dumps(config.model_dump(mode="json"), indent=2) + "\n"
    if resume:
        if run_name is None:
            raise ValueError("resume needs the name of the run to continue")
        run_dir = config.output_dir / run_name
        if not (run_dir / "config.json").is_file():
            raise FileNotFoundError(f"no run to resume at {run_dir}")
        if (run_dir / "config.json").read_text(encoding="utf-8") != config_json:
            raise ValueError(f"{run_dir} was started with a different config; start a new run")
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        meta.setdefault("resumed", []).append(_meta())
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = config.output_dir / (run_name or f"{config.name}-{stamp}")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(config_json, encoding="utf-8")
        meta = _meta()
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    clients = _Clients(run_dir if config.log_requests else None, config.reply_cache)
    rows = _read_jsonl(run_dir / "results.jsonl")
    errors = _read_jsonl(run_dir / "errors.jsonl")
    done = {(row["stream"], row["condition"]) for row in rows}
    skipped: dict[str, str] = {}
    failed: list[dict[str, str]] = []
    for stream_index, stream in enumerate(streams):
        context: StreamContext | None = None
        for condition_index, condition in enumerate(config.conditions):
            if condition.name in skipped or (stream.id, condition.name) in done:
                continue
            context = context or StreamContext(stream, config)
            rng = np.random.default_rng([config.seed, stream_index, condition_index])
            clients.tag(f"{stream.id}/{condition.name}")
            try:
                row, condition_errors, record = run_condition(context, condition, clients, rng)
            except ConditionUnavailableError as exc:
                log.warning("skipping %s: %s", condition.name, exc)
                skipped[condition.name] = str(exc)
                continue
            except Exception as exc:
                # One failed pair (a dead server, a model bug) must not cost the whole run.
                log.exception("%s %s failed", stream.id, condition.name)
                failed.append(
                    {
                        "stream": stream.id,
                        "condition": condition.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            log.info("%s %s cost=%.1f", stream.id, condition.name, row["cost"])
            rows.append(row)
            errors.extend(condition_errors)
            _append_jsonl(run_dir / "results.jsonl", [row])
            _append_jsonl(run_dir / "errors.jsonl", condition_errors)
            if record is not None:
                _append_jsonl(run_dir / RECORDS_FILE, [record.to_json()])

    _write_csv(run_dir / "errors.csv", errors)
    _write_csv(run_dir / "summary.csv", summarise(rows))
    _write_csv(run_dir / "sensitivity.csv", sensitivity(errors))
    (run_dir / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n", encoding="utf-8")
    (run_dir / "failed.json").write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
    if clients.provenance():
        (run_dir / "models.json").write_text(
            json.dumps(clients.provenance(), indent=2) + "\n", encoding="utf-8"
        )
    if failed:
        log.error(
            "%d (stream, condition) pairs failed; see %s", len(failed), run_dir / "failed.json"
        )
    return run_dir


@dataclass(frozen=True)
class Replay:
    rows: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    # (stream, condition) pairs of compiled conditions with no recorded output
    missing: list[tuple[str, str]]


def replay_experiment(
    config: ExperimentConfig,
    records: dict[tuple[str, str], ReaderRecord],
    *,
    streams: str | Path | None = None,
) -> Replay:
    """Score every compiled condition of ``config`` from recorded reader output.

    Streams, conditions and random generators are taken in the order :func:`run_experiment`
    uses, so with the recorded run's own config the replay reproduces its costs, errors, single
    repairs and Shapley samples, with no model call. Only solver-decided conditions are
    replayed; a model decider would have to be asked again. Nothing is written.
    """
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    clients = _Clients()
    for stream_index, stream in enumerate(load_streams(streams or config.streams)):
        context: StreamContext | None = None
        for condition_index, condition in enumerate(config.conditions):
            if condition.decider != PLANNER_KEY:
                continue
            record = records.get((stream.id, condition.name))
            if record is None:
                missing.append((stream.id, condition.name))
                continue
            context = context or StreamContext(stream, config)
            rng = np.random.default_rng([config.seed, stream_index, condition_index])
            row, condition_errors, _ = run_condition(
                context, condition, clients, rng, interpreter=record.replay(condition.name)
            )
            rows.append(row)
            errors.extend(condition_errors)
    return Replay(rows, errors, missing)


_SUMMARY_FIELDS = (
    "cost", "r_total", "r_interp", "r_dec", "interp_share", "regret_vs_oracle",
    "precision", "recall", "f1", "fill_rate", "violations_true", "invalid_order_lines",
    "parse_failures", "approvals_requested", "rejected", "rejected_correct", "conflicts_dropped",
    "audit_completeness", "calibration_ece", "escalation_share",
    "value_of_perfect_information", "model_requests", "model_seconds", "model_truncations",
    "model_retries", "model_capped_replies", "model_cache_hits", "seconds",
)  # fmt: skip
_MEDIAN_FIELDS = ("cost", "r_total", "r_interp", "r_dec")


def summarise(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per condition: the mean of each metric over the streams where it is defined, medians
    of cost and regret, the number of streams with non-zero total regret, and the
    interpretation share as a ratio of sums (ΣR_interp / ΣR_total; the mean of per-stream
    shares understates it when a few streams carry most of the regret). Conditions that cover
    different streams are not comparable; a warning is logged and ``n_streams`` shows it."""
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    if len({len(group) for group in by_condition.values()}) > 1:
        counts = {condition: len(group) for condition, group in by_condition.items()}
        log.warning("conditions cover different numbers of streams: %s", counts)
    summary = []
    for condition, group in by_condition.items():
        entry: dict[str, Any] = {"condition": condition, "n_streams": len(group)}
        for field in _SUMMARY_FIELDS:
            values = [row[field] for row in group if row.get(field) is not None]
            entry[f"mean_{field}"] = sum(values) / len(values) if values else None
        for field in _MEDIAN_FIELDS:
            values = [row[field] for row in group if row.get(field) is not None]
            entry[f"median_{field}"] = float(np.median(values)) if values else None
        totals = [row["r_total"] for row in group if row.get("r_total") is not None]
        entry["n_nonzero_r_total"] = sum(abs(value) > 1e-6 for value in totals)
        paired = [row for row in group if row.get("r_interp") is not None]
        total = sum(row["r_total"] for row in paired)
        interp = sum(row["r_interp"] for row in paired)
        entry["ratio_interp_share"] = interp / total if paired and total != 0 else None
        summary.append(entry)
    return summary


def sensitivity(errors: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The sensitivity map S(τ, k): error counts and summed ΔC_j by delta type and error kind."""
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for error in errors:
        key = (error["condition"], error["delta_type"], error["kind"])
        cell = cells.setdefault(
            key,
            {
                "condition": key[0],
                "delta_type": key[1],
                "kind": key[2],
                "count": 0,
                "sum_delta_c": 0.0,
            },
        )
        cell["count"] += 1
        cell["sum_delta_c"] += error["delta_c"] or 0.0
    return sorted(cells.values(), key=lambda c: (c["condition"], -c["sum_delta_c"]))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tune_safety_factor(
    config: ExperimentConfig,
    streams: Sequence[Stream],
    factors: Sequence[float] | None = None,
) -> list[tuple[float, float]]:
    """Mean oracle-pipeline cost per solver safety factor, sorted from best to worst.

    Run it on development streams, never on the streams you report: the chosen factor is part
    of the solver π and belongs in the pre-registration.
    """
    if not streams:
        raise ValueError("no development streams")
    grid = tuple(factors) if factors else config.tuned_safety_factors
    totals = dict.fromkeys(grid, 0.0)
    for stream in streams:
        context = StreamContext(stream, config)
        for factor in grid:
            solver = MILPSolver(
                horizon=config.solver.horizon,
                time_limit=config.solver.time_limit,
                safety_factor=factor,
            )
            policy = context.compiling(OracleInterpreter(stream.gold), solver)
            totals[factor] += context.simulator.run(policy).total_cost
    means = [(factor, total / len(streams)) for factor, total in totals.items()]
    return sorted(means, key=lambda pair: pair[1])


def _meta() -> dict[str, Any]:
    """What produced a run: package and schema versions, the code commit, and the time."""
    from cdd import __version__
    from cdd.schema import SCHEMA_VERSION

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "cdd_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit,
        "git_dirty": dirty,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
