"""Policies the simulator can run: the compiling pipeline, direct deciders and fixed plans.

:class:`CompilingPolicy` wires the six layers for one stream, in the order of Figure 1:
interpreter (L1), span parser (L4), SOP admissibility (L2), bitemporal state (L3), decider (L5)
and audit (L6). Only the interpreter, and a model-backed decider if one is used, can be
stochastic; every other step is deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from cdd.audit import AuditLog
from cdd.deciders import Decider, StreamDecider
from cdd.interpreters.base import Interpreter
from cdd.orders import InventoryPosition, Plan
from cdd.parser import ParseError, SpanParser
from cdd.schema import Delta, Message
from cdd.sop import ACCEPT, Approver, MagnitudeCapApprover, SOPGraph, Verdict
from cdd.state import BitemporalState, StateView
from cdd.update import apply_overwrite, apply_update
from cdd.world import World


def _by_period(messages: Iterable[Message]) -> dict[int, list[Message]]:
    inbox: dict[int, list[Message]] = defaultdict(list)
    for message in messages:
        inbox[message.period].append(message)
    return inbox


class CompilingPolicy:
    def __init__(
        self,
        world: World,
        messages: Iterable[Message],
        interpreter: Interpreter,
        decider: Decider,
        *,
        sop: SOPGraph | None = None,
        approver: Approver | None = None,
        audit: AuditLog | None = None,
        sop_gate: bool = True,
        versioning: bool = True,
    ) -> None:
        """``sop_gate=False`` and ``versioning=False`` are the layer ablations of Section 6:
        accept every parsed delta, and overwrite state instead of versioning it."""
        self.world = world
        self.interpreter = interpreter
        self.decider = decider
        self.parser = SpanParser(world.calendar_year, [entity.id for entity in world.entities()])
        self.sop = sop or SOPGraph()
        self.approver = approver or MagnitudeCapApprover()
        self.audit = audit or AuditLog()
        self.sop_gate = sop_gate
        self.update = apply_update if versioning else apply_overwrite
        self.state = BitemporalState(world)
        self.proposed: list[Delta] = []
        self.failures: list[tuple[str, str, str]] = []
        self.conflicts_dropped = 0
        self._inbox = _by_period(messages)

    @property
    def name(self) -> str:
        return f"{self.interpreter.name}+{self.decider.name}"

    def __call__(self, position: InventoryPosition) -> Plan:
        t = position.period
        view = self.state.view(as_of=t)
        admitted: list[Delta] = []
        sender_class: dict[str, str] = {}
        for message in self._inbox.get(t, ()):
            self.audit.message(message)
            interpretation = self.interpreter.interpret(message, view)
            for reason in interpretation.failures:
                self._fail(message.id, "interpreter", reason)
            for index, proposal in enumerate(interpretation.proposals):
                delta_id = proposal.delta_id or f"{message.id}#{index}"
                try:
                    delta = self.parser.complete(proposal, message, delta_id)
                except ParseError as exc:
                    self._fail(message.id, exc.slot, str(exc))
                    continue
                self.proposed.append(delta)
                sender_class[delta.id] = message.sender_class
                if self._admit(delta, message, view):
                    admitted.append(delta)

        result = self.update(self.state, admitted, recorded_at=t)
        self.audit.update(t, result)
        for conflict in result.conflicts:
            by_authority = self.sop.resolve(conflict, sender_class) if self.sop_gate else None
            chosen = by_authority or self.approver.resolve(conflict, view)
            self.audit.resolution(t, conflict, chosen)
            self.conflicts_dropped += len(conflict.deltas) - (chosen is not None)
            if chosen is not None:
                self.audit.update(t, self.update(self.state, [chosen], recorded_at=t))

        view = self.state.view(as_of=t)
        plan = self.decider.plan(view, position)
        self.audit.decision(plan, lineage=self._lineage(view, t))
        return plan

    def _admit(self, delta: Delta, message: Message, view: StateView) -> bool:
        verdict = self.sop.check(delta, message, view) if self.sop_gate else ACCEPT
        if verdict.verdict is Verdict.ACCEPT:
            admitted = True
        elif verdict.verdict is Verdict.APPROVE and verdict.rule_id is not None:
            admitted = self.approver.approve(delta, verdict.rule_id, view)
        else:
            admitted = False
        self.audit.admissibility(delta, verdict, admitted)
        return admitted

    def _fail(self, message_id: str, slot: str, reason: str) -> None:
        self.failures.append((message_id, slot, reason))
        self.audit.failure(message_id, slot, reason)

    @staticmethod
    def _lineage(view: StateView, period: int) -> list[str]:
        """Deltas in force now or later: everything the decision could have relied on."""
        return [
            delta.id
            for delta in view.active()
            if delta.window.end is None or delta.window.end >= period
        ]


class DirectPolicy:
    """A model decides from the raw message history. There is no ŝ and no lineage."""

    def __init__(self, messages: Iterable[Message], decider: StreamDecider) -> None:
        self.decider = decider
        self._inbox = _by_period(messages)
        self._seen: list[Message] = []

    @property
    def name(self) -> str:
        return self.decider.name

    def __call__(self, position: InventoryPosition) -> Plan:
        self._seen.extend(self._inbox.get(position.period, ()))
        return self.decider.decide(tuple(self._seen), position)


class PlannedPolicy:
    """Executes a precomputed plan per period, e.g. the hindsight benchmark."""

    name = "planned"

    def __init__(self, plans: Mapping[int, Plan]) -> None:
        self._plans = plans

    def __call__(self, position: InventoryPosition) -> Plan:
        return self._plans.get(position.period, Plan(period=position.period))


def build_truth(
    world: World, messages: Sequence[Message], gold: Sequence[Delta]
) -> BitemporalState:
    """The physical truth: every gold delta, recorded in the period its message arrived.

    The truth is not SOP-gated. The oracle *condition* (gold deltas through the full pipeline)
    is gated, so a gold delta that the SOP layer blocks shows up as decision regret.
    """
    period_of = {message.id: message.period for message in messages}
    state = BitemporalState(world)
    by_period: dict[int, list[Delta]] = defaultdict(list)
    for delta in gold:
        by_period[period_of[delta.message_id]].append(delta)
    for period in sorted(by_period):
        result = apply_update(state, by_period[period], recorded_at=period)
        if result.conflicts:
            raise ValueError(f"gold deltas conflict in period {period}: {result.conflicts}")
    return state
