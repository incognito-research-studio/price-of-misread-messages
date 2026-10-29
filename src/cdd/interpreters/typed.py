"""Conditions (c)-(e): typed models answer atomic questions about each message.

A typed model returns one choice, with a probability, from at most 255 declared options. Each
message becomes up to six questions: delta type, entity, supporting sentence, magnitude bucket,
window start and window end. Option sets larger than 255 are asked in two stages. Holds,
releases and supplier-status changes get no window questions: when their message states no
window, the change holds from the message's week until further notice, which is also what
every other reader gets. Asking anyway let a model turn "on hold pending CoA" into a one-week
hold.

Options can carry written definitions. Typed models can follow an option's name over its
definition (Sun et al., 2026), so the type question defines every option, and the ``labels``
ablation swaps the names for neutral letters (or drops the definitions, as in the pilot's bare
labels). Under the default ``parser_first`` window rule the span parser reads the value and
window from the chosen sentence and uses the model's magnitude and window answers only when the
sentence states none; ``model_first`` lets the model's window answer win.

v0.1 asks for at most one delta per message.
"""

from __future__ import annotations

import math
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from cdd.interpreters.base import Interpretation
from cdd.prompts import TYPE_GUIDE, render_message
from cdd.schema import DELTA_SPECS, DeltaType, Message, Proposal, Span, Window
from cdd.state import StateView
from cdd.text import Gazetteer, sentence_spans
from cdd.world import World

MAX_OPTIONS = 255
NONE = "none"
NOT_STATED = "not stated"
OPEN_ENDED = "open-ended"

LabelMode = Literal["names", "neutral", "bare"]
WindowRule = Literal["parser_first", "model_first"]

# Every magnitude the generator writes has a bucket (tests/test_generator.py checks this).
_PERCENTS = (50, 40, 30, 25, 20, 15, 10, 8, 6, 5, 4, 3, 2)
PERCENT_BUCKETS = tuple(f"-{p}%" for p in _PERCENTS) + tuple(f"+{p}%" for p in reversed(_PERCENTS))
DAY_BUCKETS = (
    "-14 days", "-7 days", "-5 days", "-3 days", "-1 day",
    "+1 day", "+3 days", "+5 days", "+6 days", "+7 days", "+10 days", "+14 days", "+21 days",
)  # fmt: skip
MAGNITUDES: dict[DeltaType, tuple[str, ...]] = {
    DeltaType.CAPACITY_CHANGE: PERCENT_BUCKETS,
    DeltaType.DEMAND_SIGNAL: PERCENT_BUCKETS,
    DeltaType.PRICE_CHANGE: PERCENT_BUCKETS,
    DeltaType.LEAD_TIME_CHANGE: DAY_BUCKETS,
}

# The same type descriptions the JSON interpreter receives, so the two conditions differ only
# in output format, not in what the model is told about the schema.
TYPE_DEFINITIONS: tuple[tuple[str, str], ...] = (
    *((kind.value, TYPE_GUIDE[kind]) for kind in DeltaType),
    (NONE, "the message changes nothing in the planning state"),
)


@dataclass(frozen=True)
class TypedQuestion:
    key: str
    prompt: str
    options: tuple[str, ...]
    definitions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"{self.key}: {len(self.options)} options (1-{MAX_OPTIONS} allowed)")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"{self.key}: duplicate options")
        if self.definitions and len(self.definitions) != len(self.options):
            raise ValueError(f"{self.key}: one definition per option, or none")


@dataclass(frozen=True)
class TypedAnswer:
    key: str
    choice: str
    probability: float


class TypedModelClient(Protocol):
    def ask(self, context: str, questions: Sequence[TypedQuestion]) -> Sequence[TypedAnswer]: ...


def ask_all(
    client: TypedModelClient, context: str, questions: Sequence[TypedQuestion]
) -> dict[str, TypedAnswer]:
    """Ask independent questions in one call and check every answer against its options."""
    answers = {answer.key: answer for answer in client.ask(context, questions)}
    for question in questions:
        answer = answers.get(question.key)
        if answer is None or answer.choice not in question.options:
            raise ValueError(f"no valid answer for {question.key!r}")
    return answers


def choose(
    client: TypedModelClient, context: str, key: str, prompt: str, options: Sequence[str]
) -> TypedAnswer:
    """Ask one question, in two stages when it has more than 255 options."""
    if len(options) <= MAX_OPTIONS:
        return ask_all(client, context, [TypedQuestion(key, prompt, tuple(options))])[key]
    size = math.ceil(len(options) / MAX_OPTIONS)
    groups = [tuple(options[i : i + size]) for i in range(0, len(options), size)]
    labels = tuple(f"group {i + 1}: {g[0]} to {g[-1]}" for i, g in enumerate(groups))
    first = ask_all(
        client, context, [TypedQuestion(f"{key}:group", f"{prompt} (choose a group)", labels)]
    )[f"{key}:group"]
    group = groups[labels.index(first.choice)]
    second = ask_all(client, context, [TypedQuestion(key, prompt, group)])[key]
    return TypedAnswer(key, second.choice, first.probability * second.probability)


def type_question(labels: LabelMode = "names") -> tuple[TypedQuestion, dict[str, str]]:
    """The delta-type question and a map from option label back to delta type (or ``none``)."""
    values = tuple(value for value, _ in TYPE_DEFINITIONS)
    definitions = tuple(text for _, text in TYPE_DEFINITIONS)
    prompt = "Which state change does this message report?"
    if labels == "neutral":
        letters = tuple(string.ascii_uppercase[: len(values)])
        question = TypedQuestion("type", prompt, letters, definitions)
        return question, dict(zip(letters, values, strict=True))
    if labels == "bare":
        return TypedQuestion("type", prompt, values), {v: v for v in values}
    return TypedQuestion("type", prompt, values, definitions), {v: v for v in values}


class AtomicQuestionInterpreter:
    def __init__(
        self,
        client: TypedModelClient,
        world: World,
        name: str = "typed-atomic",
        *,
        labels: LabelMode = "names",
        window_rule: WindowRule = "parser_first",
    ) -> None:
        self.client = client
        self.world = world
        self.name = name
        self.labels = labels
        self.window_rule = window_rule
        self.gazetteer = Gazetteer(world)

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        context = render_message(message)
        try:
            return self._interpret(message, context)
        except ValueError as exc:
            return Interpretation(failures=(str(exc),))

    # The three question groups are asked in this order and are also the unit of a training
    # example (cdd.adapters.laya.training_rows), so training matches inference.

    def entity_question(self, kind: DeltaType) -> tuple[TypedQuestion, dict[str, str]]:
        """The entity question for a delta type, and its map from entity name to id."""
        entity_kind = DELTA_SPECS[kind].entity_kind
        forms = self.gazetteer.surface_forms(entity_kind)
        prompt = f"Which {entity_kind.value} does it concern?"
        return TypedQuestion("entity", prompt, tuple(forms)), forms

    def detail_questions(
        self, message: Message, kind: DeltaType
    ) -> tuple[list[TypedQuestion], list[tuple[int, int]]]:
        """Span, window and magnitude questions, and the sentence spans behind the span options."""
        spans = sentence_spans(message.text)
        labels = tuple(f"[{i}] {message.text[s:e]}" for i, (s, e) in enumerate(spans))
        weeks = tuple(f"wk{p}" for p in self.world.periods)
        questions = [TypedQuestion("span", "Which sentence states the change?", labels)]
        if not DELTA_SPECS[kind].categorical:
            questions += [
                TypedQuestion(
                    "window_start", "When does the change take effect?", (*weeks, NOT_STATED)
                ),
                TypedQuestion("window_end", "When does it stop?", (*weeks, OPEN_ENDED, NOT_STATED)),
            ]
        if kind in MAGNITUDES:
            questions.append(
                TypedQuestion(
                    "magnitude", "How large is the change?", (*MAGNITUDES[kind], NOT_STATED)
                )
            )
        return questions, spans

    def _interpret(self, message: Message, context: str) -> Interpretation:
        question, decode = type_question(self.labels)
        kind_answer = ask_all(self.client, context, [question])["type"]
        if decode[kind_answer.choice] == NONE:
            return Interpretation()
        kind = DeltaType(decode[kind_answer.choice])
        entity, forms = self.entity_question(kind)
        entity_answer = choose(self.client, context, "entity", entity.prompt, entity.options)

        questions, spans = self.detail_questions(message, kind)
        answers = ask_all(self.client, context, questions)

        start, end = spans[questions[0].options.index(answers["span"].choice)]
        hints = {
            key: answers[key].choice
            for key in ("magnitude", "window_start", "window_end")
            if key in answers and answers[key].choice != NOT_STATED
        }
        window = None
        if self.window_rule == "model_first" and "window_start" in answers:
            window = _window(answers["window_start"].choice, answers["window_end"].choice)
        confidence = (
            kind_answer.probability * entity_answer.probability * answers["span"].probability
        )
        proposal = Proposal(
            type=kind,
            entity=forms[entity_answer.choice],
            span=Span(message_id=message.id, start=start, end=end),
            window=window,
            confidence=min(1.0, max(0.0, confidence)),
            hints=hints,
        )
        return Interpretation((proposal,))


def _window(start: str, end: str) -> Window | None:
    """A window from typed answers, or None to let the span parser read it."""
    if start == NOT_STATED:
        return None
    first = int(start.removeprefix("wk"))
    if end in (OPEN_ENDED, NOT_STATED):
        return Window(start=first)
    last = int(end.removeprefix("wk"))
    return Window(start=first, end=last) if last >= first else None
