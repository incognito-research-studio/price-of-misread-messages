"""The interpreter contract (L1): ``(m, ŝ_t) → schema-conformant proposals with spans``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cdd.schema import Message, Proposal
from cdd.state import StateView


@dataclass(frozen=True)
class Interpretation:
    """Proposals for one message. ``failures`` records output the interpreter could not turn
    into a proposal, such as malformed JSON or evidence that is not in the message."""

    proposals: tuple[Proposal, ...] = ()
    failures: tuple[str, ...] = ()


class Interpreter(Protocol):
    @property
    def name(self) -> str: ...

    def interpret(self, message: Message, view: StateView) -> Interpretation: ...
