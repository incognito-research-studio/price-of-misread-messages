from __future__ import annotations

from pathlib import Path

import pytest

from cdd.benchmark import GeneratorConfig, Stream, generate, generate_stream, load_streams
from cdd.parser import SpanParser
from cdd.pipeline import build_truth
from cdd.schema import DeltaType, Proposal

# More events per week than the pilot, so every template and pattern shows up.
CONFIG = GeneratorConfig.model_validate(
    {"name": "test", "n_streams": 12, "seed": 3, "events": {"rate": 2.0, "revise_prob": 0.5}}
)


@pytest.fixture(scope="module")
def streams() -> list[Stream]:
    return generate(CONFIG)


def test_generation_is_deterministic() -> None:
    assert generate_stream(CONFIG, 0) == generate_stream(CONFIG, 0)
    assert generate_stream(CONFIG, 0) != generate_stream(CONFIG, 1)


def test_every_delta_type_appears(streams: list[Stream]) -> None:
    seen = {delta.type for stream in streams for delta in stream.gold}
    assert seen == set(DeltaType)
    assert any(delta.supersedes for stream in streams for delta in stream.gold)


def test_gold_round_trips_through_the_span_parser(streams: list[Stream]) -> None:
    """Given only type, entity and span, the parser must recover the gold value and window."""
    for stream in streams:
        parser = SpanParser(stream.world.calendar_year)
        messages = {message.id: message for message in stream.messages}
        for gold in stream.gold:
            message = messages[gold.message_id]
            proposal = Proposal(
                type=gold.type, entity=gold.entity, span=gold.src, supersedes=gold.supersedes
            )
            parsed = parser.complete(proposal, message, gold.id)
            assert parsed == gold, message.text


def test_truth_builds_without_conflicts(streams: list[Stream]) -> None:
    for stream in streams:
        build_truth(stream.world, stream.messages, stream.gold)


def test_streams_save_and_load(tmp_path: Path, streams: list[Stream]) -> None:
    streams[0].save(tmp_path / f"{streams[0].id}.json")
    assert load_streams(tmp_path) == [streams[0]]


B2 = GeneratorConfig.from_yaml(Path(__file__).parents[1] / "configs/benchmark/pilot.yaml")


@pytest.fixture(scope="module")
def b2_streams() -> list[Stream]:
    return generate(B2.model_copy(update={"n_streams": 8}))


def test_b2_worlds_are_calibrated(b2_streams: list[Stream]) -> None:
    import math

    for stream in b2_streams:
        world = stream.world
        assert stream.provenance["benchmark_version"] == "b2"
        for sku in world.skus:
            suppliers = {p.supplier for p in world.plants if sku.id in p.skus}
            assert len(suppliers) >= 2
            prices = [o.price for o in world.offers if o.sku == sku.id]
            assert max(prices) / min(prices) <= 1.05 / 0.95 + 1e-9
        assert all(lot.arrival is not None and 39 <= lot.arrival <= 47 for lot in world.lots)
        assert all(message.period <= 49 for message in stream.messages)
        lanes = {lane.id: lane for lane in world.lanes}
        for delta in stream.gold:
            if delta.type is DeltaType.LEAD_TIME_CHANGE:
                base = lanes[delta.entity].lead_time_days
                assert isinstance(delta.value, float)
                assert math.ceil((base + delta.value) / 7) != math.ceil(base / 7)
        assert not any("forward" in message.text for message in stream.messages)


def test_every_generated_magnitude_has_a_typed_bucket(b2_streams: list[Stream]) -> None:
    from cdd.adapters.laya import _nearest_bucket
    from cdd.interpreters.typed import MAGNITUDES

    for stream in [*b2_streams, *generate(CONFIG)]:
        for delta in stream.gold:
            if delta.type in MAGNITUDES:
                bucket = _nearest_bucket(delta, MAGNITUDES[delta.type])
                assert isinstance(delta.value, float)
                expected = (
                    f"{(delta.value - 1) * 100:+.0f}%"
                    if delta.type is not DeltaType.LEAD_TIME_CHANGE
                    else f"{delta.value:+.0f} days"
                )
                assert bucket == expected, (delta, bucket)


def test_b2_numeric_changes_to_one_entity_do_not_overlap(b2_streams: list[Stream]) -> None:
    for stream in b2_streams:
        numeric = [d for d in stream.gold if isinstance(d.value, float)]
        for i, a in enumerate(numeric):
            for b in numeric[i + 1 :]:
                if (a.entity, a.attribute) == (b.entity, b.attribute) and a.window.overlaps(
                    b.window
                ):
                    assert a.id in b.supersedes or b.id in a.supersedes
