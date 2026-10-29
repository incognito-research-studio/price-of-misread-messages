from __future__ import annotations

import json
import re

import pytest

from cdd.benchmark import GeneratorConfig, generate
from cdd.interpreters import AtomicQuestionInterpreter, JsonLLMInterpreter, RegexInterpreter
from cdd.interpreters.typed import MAX_OPTIONS, TypedQuestion, choose, type_question
from cdd.prompts import render_types
from cdd.schema import DeltaType, Window
from cdd.state import BitemporalState
from cdd.world import World
from fakes import ScriptedLLMClient, ScriptedTypedClient
from helpers import make_delta, make_message


def test_regex_finds_type_entity_and_sentence(world: World, state: BitemporalState) -> None:
    message = make_message("Hi team. Plant 2 output reduced ~15% wk42-44. Thanks.")
    (proposal,) = RegexInterpreter(world).interpret(message, state.view(40)).proposals
    assert (proposal.type, proposal.entity) == (DeltaType.CAPACITY_CHANGE, "P2")
    assert message.excerpt(proposal.span) == "Plant 2 output reduced ~15% wk42-44."


def test_regex_resolves_price_lines(world: World, state: BitemporalState) -> None:
    message = make_message("Supplier D: surcharge 4% on SKU-01 effective wk42.")
    (proposal,) = RegexInterpreter(world).interpret(message, state.view(40)).proposals
    assert proposal.entity == "SUP-D/SKU-01"


def test_json_llm_interpreter_maps_evidence_to_spans(world: World, state: BitemporalState) -> None:
    text = "Lot 7781 on hold pending CoA. Please confirm."
    reply = {
        "deltas": [
            {
                "type": "quality_hold",
                "entity": "LOT-7781",
                "evidence": "Lot 7781 on hold pending CoA.",
            }
        ]
    }
    client = ScriptedLLMClient(lambda system, user: f"```json\n{json.dumps(reply)}\n```")
    result = JsonLLMInterpreter(client).interpret(make_message(text), state.view(40))
    assert result.failures == ()
    assert (result.proposals[0].span.start, result.proposals[0].span.end) == (0, 29)


def test_json_llm_interpreter_records_failures(world: World, state: BitemporalState) -> None:
    view, message = state.view(40), make_message("Lot 7781 released.")
    garbage = JsonLLMInterpreter(ScriptedLLMClient(lambda s, u: "no json here"))
    assert garbage.interpret(message, view).failures
    invented = {
        "deltas": [{"type": "quality_release", "entity": "LOT-7781", "evidence": "not in text"}]
    }
    liar = JsonLLMInterpreter(ScriptedLLMClient(lambda s, u: json.dumps(invented)))
    result = liar.interpret(message, view)
    assert result.proposals == () and len(result.failures) == 1


def test_constrained_schema_limits_evidence_and_supersedes(
    world: World, state: BitemporalState
) -> None:
    message = make_message("Plant 2 output reduced 15% wk42-44. Please confirm.")
    client = ScriptedLLMClient(lambda s, u: '{"deltas": []}')
    JsonLLMInterpreter(client, constrained=True).interpret(message, state.view(40))
    options = client.schemas[0]["properties"]["deltas"]["items"]["anyOf"]  # type: ignore[index]
    by_type = {o["properties"]["type"]["const"]: o["properties"] for o in options}
    item = by_type["capacity_change"]
    assert item["evidence"]["enum"] == ["Plant 2 output reduced 15% wk42-44.", "Please confirm."]
    assert item["supersedes"] == {"type": "array", "maxItems": 0}
    # Each type names only entities of its kind.
    assert item["entity"]["enum"] == ["P2", "P3"]
    assert by_type["demand_signal"]["entity"]["enum"] == ["SKU-01"]
    assert by_type["quality_hold"]["entity"]["enum"] == ["LOT-7781"]

    state.assert_delta(make_delta(), recorded_at=40)
    JsonLLMInterpreter(client, constrained=True).interpret(message, state.view(41))
    options = client.schemas[1]["properties"]["deltas"]["items"]["anyOf"]  # type: ignore[index]
    assert options[0]["properties"]["supersedes"]["items"]["enum"] == ["m1#0"]

    JsonLLMInterpreter(client).interpret(message, state.view(41))
    assert client.schemas[2] is None


def test_prompt_examples_name_no_generated_entity() -> None:
    worlds = [stream.world for stream in generate(GeneratorConfig.model_validate({"n_streams": 3}))]
    names = {form for world in worlds for e in world.entities() for form in e.surface_forms}
    guide = render_types()
    assert not [name for name in names if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", guide)]


def test_atomic_question_interpreter(world: World, state: BitemporalState) -> None:
    answers = {
        "type": "capacity_change",
        "entity": "Plant 2",
        "span": "[1] Plant 2 output reduced ~15% wk42-44.",
        "window_start": "wk42",
        "window_end": "wk44",
        "magnitude": "-15%",
    }
    client = ScriptedTypedClient(lambda question: answers[question.key])
    message = make_message("Hello. Plant 2 output reduced ~15% wk42-44.")
    (proposal,) = (
        AtomicQuestionInterpreter(client, world).interpret(message, state.view(40)).proposals
    )
    # Parser first: the model's window answer travels as a hint, not as the window.
    assert (proposal.entity, proposal.window) == ("P2", None)
    assert proposal.hints == {"magnitude": "-15%", "window_start": "wk42", "window_end": "wk44"}
    assert proposal.confidence is not None and 0 < proposal.confidence < 1

    model_first = AtomicQuestionInterpreter(client, world, window_rule="model_first")
    (proposal,) = model_first.interpret(message, state.view(40)).proposals
    assert proposal.window == Window(start=42, end=44)


def test_holds_are_open_ended_unless_the_message_states_a_window(
    world: World, state: BitemporalState
) -> None:
    asked: list[str] = []

    def answer(question: TypedQuestion) -> str:
        asked.append(question.key)
        # A model that would read a one-week hold if it were asked about the window.
        return {
            "type": "quality_hold",
            "entity": "Lot 7781",
            "span": "[0] Lot 7781 on hold pending CoA.",
            "window_start": "wk40",
            "window_end": "wk40",
        }[question.key]

    client = ScriptedTypedClient(answer)
    message = make_message("Lot 7781 on hold pending CoA.", period=40)
    for rule in ("parser_first", "model_first"):
        interpreter = AtomicQuestionInterpreter(client, world, window_rule=rule)  # type: ignore[arg-type]
        (proposal,) = interpreter.interpret(message, state.view(40)).proposals
        assert proposal.window is None and "window_end" not in proposal.hints
    assert "window_start" not in asked and "window_end" not in asked


def test_type_question_label_modes() -> None:
    named, decode = type_question("names")
    assert named.options[0] == "capacity_change" and len(named.definitions) == len(named.options)
    neutral, decode = type_question("neutral")
    assert neutral.options[:3] == ("A", "B", "C") and decode["A"] == "capacity_change"
    assert neutral.definitions == named.definitions
    bare, _ = type_question("bare")
    assert bare.definitions == ()


def test_neutral_labels_decode_to_delta_types(world: World, state: BitemporalState) -> None:
    def chooser(question: TypedQuestion) -> str:
        if question.key == "type":
            return next(
                option
                for option, text in zip(question.options, question.definitions, strict=True)
                if "transit time" in text
            )
        return question.options[0]

    interpreter = AtomicQuestionInterpreter(ScriptedTypedClient(chooser), world, labels="neutral")
    message = make_message("Transit via Port X +6 days from wk41.")
    (proposal,) = interpreter.interpret(message, state.view(40)).proposals
    assert proposal.type is DeltaType.LEAD_TIME_CHANGE


def test_atomic_question_interpreter_can_abstain(world: World, state: BitemporalState) -> None:
    client = ScriptedTypedClient(lambda question: "none")
    result = AtomicQuestionInterpreter(client, world).interpret(
        make_message("FYI."), state.view(40)
    )
    assert result.proposals == () and result.failures == ()


def test_large_option_sets_are_asked_in_two_stages() -> None:
    options = [f"option {i}" for i in range(600)]
    client = ScriptedTypedClient(lambda question: question.options[-1])
    answer = choose(client, "context", "key", "Which?", options)
    assert answer.choice == "option 599"
    assert len(client.calls) == 2
    assert answer.probability == 0.9 * 0.9


def test_typed_questions_enforce_the_option_limit() -> None:
    with pytest.raises(ValueError, match="options"):
        TypedQuestion("k", "p", tuple(str(i) for i in range(MAX_OPTIONS + 1)))
