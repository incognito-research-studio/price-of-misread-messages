"""Scripted stand-ins for model clients. Tests never call a network service."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cdd.interpreters.typed import TypedAnswer, TypedQuestion

Chooser = Callable[[TypedQuestion], str]


class ScriptedTypedClient:
    """Answers every typed question with ``chooser(question)`` (default: the first option)."""

    def __init__(self, chooser: Chooser | None = None, probability: float = 0.9) -> None:
        self.chooser = chooser or (lambda question: question.options[0])
        self.probability = probability
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def ask(self, context: str, questions: Sequence[TypedQuestion]) -> list[TypedAnswer]:
        self.calls.append((context, tuple(q.key for q in questions)))
        return [TypedAnswer(q.key, self.chooser(q), self.probability) for q in questions]


class ScriptedLLMClient:
    """Returns ``reply(system, user)`` for every completion request and records schemas."""

    def __init__(self, reply: Callable[[str, str], str]) -> None:
        self.reply = reply
        self.calls = 0
        self.schemas: list[dict[str, Any] | None] = []
        self.last_user = ""

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        self.calls += 1
        self.schemas.append(schema)
        self.last_user = user
        return self.reply(system, user)


def zero_order_typed_client() -> ScriptedTypedClient:
    """Factory used by experiment-config tests: always picks the first option (order 0)."""
    return ScriptedTypedClient()


def empty_llm_client() -> ScriptedLLMClient:
    """Factory used by experiment-config tests: returns no deltas and no orders."""
    return ScriptedLLMClient(lambda system, user: '{"deltas": [], "orders": []}')


# Switch for flaky_llm_client: while "down" is true, every request times out.
SERVER = {"down": False}


class FlakyLLMClient(ScriptedLLMClient):
    """Times out while ``SERVER["down"]``; otherwise returns no deltas. It has the logging
    attributes of the Ollama client, so the runner gives it a request log."""

    def __init__(self) -> None:
        super().__init__(lambda system, user: '{"deltas": []}')
        self.model = "fake/model:1"
        self.log_path: Path | None = None
        self.log_tag: str | None = None

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        if SERVER["down"]:
            raise TimeoutError("timed out")
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"tag": self.log_tag}) + "\n")
        return super().complete(system, user, schema)


def flaky_llm_client() -> FlakyLLMClient:
    return FlakyLLMClient()
