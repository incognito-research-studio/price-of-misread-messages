"""Condition (j): a fast typed model reads first, and an LLM takes the hard or risky messages.

A message goes to the fallback interpreter when the primary one fails on it, when any of its
proposals has a confidence below ``threshold``, or when any proposal has a delta type in
``escalate_types``. The escalation set is meant to come from the sensitivity map S(τ, k) of a
development split: the delta types whose errors cost the most. A primary abstention without
failures is accepted as is.
"""

from __future__ import annotations

from collections.abc import Collection

from cdd.interpreters.base import Interpretation, Interpreter
from cdd.schema import DeltaType, Message
from cdd.state import StateView


class HybridInterpreter:
    def __init__(
        self,
        primary: Interpreter,
        fallback: Interpreter,
        *,
        threshold: float = 0.8,
        escalate_types: Collection[DeltaType] = (),
        name: str = "hybrid",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold
        self.escalate_types = frozenset(escalate_types)
        self.name = name
        self.seen = 0
        self.escalated = 0

    @property
    def escalation_share(self) -> float:
        return 0.0 if self.seen == 0 else self.escalated / self.seen

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        self.seen += 1
        first = self.primary.interpret(message, view)
        if not self._escalate(first):
            return first
        self.escalated += 1
        return self.fallback.interpret(message, view)

    def _escalate(self, interpretation: Interpretation) -> bool:
        if interpretation.failures:
            return True
        return any(
            proposal.type in self.escalate_types
            or (proposal.confidence is not None and proposal.confidence < self.threshold)
            for proposal in interpretation.proposals
        )
