"""Interpreters that replay known deltas: the oracle (condition g) and attribution replays."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from cdd.interpreters.base import Interpretation
from cdd.schema import Delta, Message, Proposal
from cdd.state import StateView


class ReplayInterpreter:
    """Returns a fixed list of deltas for each message, keeping their ids.

    Attribution uses it to re-run a stream with an interpreter's output after some of its
    errors have been repaired.
    """

    def __init__(self, deltas: Iterable[Delta], name: str = "replay") -> None:
        self.name = name
        self._by_message: dict[str, list[Delta]] = defaultdict(list)
        for delta in deltas:
            self._by_message[delta.message_id].append(delta)

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        deltas = self._by_message.get(message.id, ())
        return Interpretation(tuple(Proposal.from_delta(delta) for delta in deltas))


class OracleInterpreter(ReplayInterpreter):
    """Condition (g): returns the gold deltas, so the pipeline builds the oracle state s*."""

    def __init__(self, gold: Iterable[Delta]) -> None:
        super().__init__(gold, name="oracle")


class GoldSpanInterpreter(ReplayInterpreter):
    """Parser isolation (WP2): the gold type, entity, revision links and evidence span, with no
    operation, value, unit or window, so the span parser must read those from the text.

    Its errors are the deterministic layer's alone: on wording the parser was not built for,
    they show what parsing loses, apart from any reader's mistakes.
    """

    def __init__(self, gold: Iterable[Delta]) -> None:
        super().__init__(gold, name="gold_span")

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        return Interpretation(
            tuple(
                Proposal(
                    type=delta.type,
                    entity=delta.entity,
                    span=delta.src,
                    supersedes=delta.supersedes,
                    delta_id=delta.id,
                )
                for delta in self._by_message.get(message.id, ())
            )
        )
