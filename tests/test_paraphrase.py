from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from cdd.benchmark import GeneratorConfig, Stream, generate
from cdd.benchmark.paraphrase import (
    Paraphraser,
    clean_reply,
    invariants,
    provenance_summary,
    time_refs,
)
from cdd.schema import DeltaType, EntityKind, Message
from fakes import ScriptedLLMClient

CONFIG = GeneratorConfig.model_validate(
    {"name": "test", "n_streams": 3, "seed": 3, "events": {"rate": 2.0, "revise_prob": 0.5}}
)


@pytest.fixture(scope="module")
def streams() -> list[Stream]:
    return generate(CONFIG)


def rewriter(rewrite: Callable[[str], str]) -> ScriptedLLMClient:
    return ScriptedLLMClient(lambda system, user: rewrite(user.split("MESSAGE:\n", 1)[1]))


def single(streams: list[Stream], kind: DeltaType, test: Callable[[Message], bool]) -> Stream:
    """A one-message stream whose message states a delta of ``kind`` and passes ``test``."""
    for stream in streams:
        for message in stream.messages:
            gold = stream.gold_for(message.id)
            if gold and gold[0].type is kind and test(message):
                return stream.model_copy(update={"messages": (message,), "gold": gold})
    raise LookupError(kind)


def outcome(stream: Stream, *rewrites: Callable[[str], str]) -> dict[str, object]:
    result = Paraphraser([rewriter(r) for r in rewrites], model="scripted").stream(stream)
    return result.provenance["paraphrase"]  # type: ignore[no-any-return]


def test_unchanged_text_passes_every_check(streams: list[Stream]) -> None:
    for stream in streams:
        result = Paraphraser([rewriter(lambda text: text)]).stream(stream)
        assert result.provenance["paraphrase"]["fallback"] == []
        assert [m.text for m in result.messages] == [m.text for m in stream.messages]
        for old, new in zip(stream.gold, result.gold, strict=True):
            assert new.model_copy(update={"src": old.src}) == old
            assert new.src.overlaps(old.src)


def test_reworded_weeks_are_accepted_and_the_span_moves(streams: list[Stream]) -> None:
    stream = single(streams, DeltaType.CAPACITY_CHANGE, lambda m: "–" in m.text)

    def reword(text: str) -> str:
        sentence = re.sub(r"wk(\d+)–(\d+)", r"during weeks \1 to \2", text)
        return f"Planning note. {sentence}"

    result = Paraphraser([rewriter(reword)]).stream(stream)
    assert result.provenance["paraphrase"]["accepted"] == 1
    message, gold = result.messages[0], result.gold[0]
    assert "during weeks" in message.text
    assert message.excerpt(gold.src).startswith(stream.messages[0].text.split(" ")[0])
    assert gold.window == stream.gold[0].window


@pytest.mark.parametrize(
    ("kind", "test", "rewrite"),
    [
        # The percentage is dropped.
        (DeltaType.CAPACITY_CHANGE, lambda m: "%" in m.text, lambda t: re.sub(r"~?\d+%", "", t)),
        # The direction flips: the parser reads the opposite sign.
        (
            DeltaType.CAPACITY_CHANGE,
            lambda m: "reduc" in m.text,
            lambda t: t.replace("reduced", "increased").replace("down", "up"),
        ),
        # The window moves by a week.
        (
            DeltaType.DEMAND_SIGNAL,
            lambda m: "wk" in m.text,
            lambda t: re.sub(r"wk(\d+)", lambda w: f"wk{int(w[1]) + 1}", t, count=1),
        ),
        # A hold reads as a release.
        (DeltaType.QUALITY_HOLD, lambda m: True, lambda t: t.replace("on hold", "released")),
        # A revision loses its revision cue.
        (
            DeltaType.CAPACITY_CHANGE,
            lambda m: "revised" in m.text,
            lambda t: t.replace("Update to earlier notice: ", "").replace("revised to", "of"),
        ),
    ],
)
def test_meaning_changes_fall_back_to_the_template(
    streams: list[Stream],
    kind: DeltaType,
    test: Callable[[Message], bool],
    rewrite: Callable[[str], str],
) -> None:
    stream = single(streams, kind, test)
    info = outcome(stream, rewrite)
    assert info["fallback"] == [stream.messages[0].id]


def test_new_entities_are_rejected(streams: list[Stream]) -> None:
    stream = single(streams, DeltaType.QUALITY_RELEASE, lambda m: True)
    others = [
        entity.name
        for entity in stream.world.entities(EntityKind.PLANT)
        if entity.name not in stream.messages[0].text
    ]
    info = outcome(stream, lambda text: f"{text} {others[0]} is not affected.")
    assert info["fallback"] == [stream.messages[0].id]


def test_a_later_client_is_tried_after_a_rejection(streams: list[Stream]) -> None:
    stream = single(streams, DeltaType.LEAD_TIME_CHANGE, lambda m: True)
    info = outcome(stream, lambda text: "Nothing to report.", lambda text: f"FYI: {text}")
    assert info["accepted"] == 1
    assert info["attempts"] == 2
    summary = provenance_summary([stream.model_copy(update={"provenance": {"paraphrase": info}})])
    assert summary["accepted"] == 1
    assert summary["fallback_by_type"] == {"lead_time_change": "0/1"}


def test_replies_that_stop_mid_sentence_are_rejected(streams: list[Stream]) -> None:
    stream = single(streams, DeltaType.QUALITY_HOLD, lambda m: True)
    lot = stream.messages[0].text
    info = outcome(stream, lambda text: f"{text} {invariants(lot, stream.world)[0]}")
    assert info["fallback"] == [stream.messages[0].id]


def test_each_message_and_attempt_gets_its_own_seed(streams: list[Stream]) -> None:
    seen: list[int] = []

    class SeededClient(ScriptedLLMClient):
        def __init__(self, reply: Callable[[str, str], str]) -> None:
            super().__init__(reply)
            self.seed = 7

        def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
            seen.append(self.seed)
            return super().complete(system, user, schema)

    reject = SeededClient(lambda system, user: "Nothing to report.")
    echo = SeededClient(lambda system, user: user.split("MESSAGE:\n", 1)[1])
    paraphraser = Paraphraser([reject, echo])
    stream = streams[0].model_copy(update={"messages": streams[0].messages[:3]})
    paraphraser.stream(stream)
    assert len(seen) == 6 and len(set(seen)) == 6
    first = list(seen)
    seen.clear()
    paraphraser.stream(stream)
    assert seen == first


def test_no_change_messages_keep_their_names(streams: list[Stream]) -> None:
    for stream in streams:
        for message in stream.messages:
            if not stream.gold_for(message.id) and invariants(message.text, stream.world):
                one = stream.model_copy(update={"messages": (message,), "gold": ()})
                assert outcome(one, lambda text: "All quiet this week.")["fallback"] == [message.id]
                return
    pytest.fail("no no-change message names an entity")


def test_time_references_ignore_wording() -> None:
    assert time_refs("wk39–41") == time_refs("weeks 39 to 41") == {"wk39", "wk41"}
    assert time_refs("from week 39 through week 41") == {"wk39", "wk41"}
    assert time_refs("effective Nov 5") == time_refs("starting November 5th") == {"m11d5"}
    assert time_refs("May we confirm?") == time_refs("may we confirm?")


def test_clean_reply() -> None:
    assert clean_reply('"Lot 7702 is on hold."') == "Lot 7702 is on hold."
    assert clean_reply("Rewritten message: Lot 7702\n is on hold. ---") == "Lot 7702 is on hold."


def test_a_no_change_rewrite_may_not_gain_a_change_word(streams: list[Stream]) -> None:
    for stream in streams:
        quiet = [m for m in stream.messages if not stream.gold_for(m.id)]
        if any("open orders" in m.text for m in quiet):
            message = next(m for m in quiet if "open orders" in m.text)
            break
    else:
        pytest.skip("no 'open orders' message in the test streams")
    one = stream.model_copy(update={"messages": (message,), "gold": ()})
    trap = rewriter(lambda text: text.replace("open orders", "active orders"))
    harmless = rewriter(lambda text: text.replace("confirms", "has confirmed"))
    assert Paraphraser([trap]).stream(one).messages[0].text == message.text
    assert "has confirmed" in Paraphraser([harmless]).stream(one).messages[0].text
