"""The WP2 challenge import and the parser-isolation reader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cdd.benchmark import GeneratorConfig, Stream, generate
from cdd.benchmark.challenge import ChallengeError, Rewrite, import_challenge, load_rewrites
from cdd.evaluation import align
from cdd.experiments import ExperimentConfig, StreamContext
from cdd.interpreters import GoldSpanInterpreter, Interpreter, OracleInterpreter
from cdd.pipeline import CompilingPolicy
from cdd.schema import DeltaType
from cdd.solver import MILPSolver

CONFIG = ExperimentConfig.model_validate({"name": "t", "streams": ".", "conditions": []})


@pytest.fixture(scope="module")
def stream() -> Stream:
    return generate(GeneratorConfig.model_validate({"name": "t", "n_streams": 1, "seed": 5}))[0]


def _rewrites(stream: Stream) -> list[Rewrite]:
    """A rewrite per message: a new opening sentence, the gold evidence kept verbatim."""
    out = []
    for message in stream.messages:
        gold = stream.gold_for(message.id)
        evidence = message.excerpt(gold[0].src) if gold else ""
        text = f"Short note from the {message.sender} desk. {evidence or 'Nothing changes.'}"
        out.append(Rewrite(message.id, "w1", text, evidence, "v1", True))
    return out


def _run(stream: Stream, interpreter: Interpreter) -> tuple[float, CompilingPolicy]:
    context = StreamContext(stream, CONFIG)
    policy = context.compiling(interpreter, MILPSolver())
    return context.simulator.run(policy).total_cost, policy


def test_import_reanchors_gold_spans_and_keeps_every_fact(stream: Stream) -> None:
    rewrites = _rewrites(stream)
    [imported] = import_challenge([stream], rewrites, "abc")
    texts = {r.id: r for r in rewrites}
    for message in imported.messages:
        assert message.text == texts[message.id].text
    for before, after in zip(stream.gold, imported.gold, strict=True):
        message = next(m for m in imported.messages if m.id == after.message_id)
        assert message.excerpt(after.src) == texts[message.id].evidence
        assert before.model_dump(exclude={"src"}) == after.model_dump(exclude={"src"})
    assert imported.provenance["challenge"] == {
        "rewrites_sha256": "abc",
        "writers": ["w1"],
        "verifiers": ["v1"],
        "unfiltered": True,
    }
    # The oracle reads completed gold deltas, so re-anchored spans reproduce its cost exactly.
    assert _run(imported, OracleInterpreter(imported.gold))[0] == pytest.approx(
        _run(stream, OracleInterpreter(stream.gold))[0]
    )


def test_import_refuses_and_lists_every_problem(stream: Stream) -> None:
    rewrites = _rewrites(stream)
    with_change = next(i for i, r in enumerate(rewrites) if r.evidence)
    without = next(i for i, r in enumerate(rewrites) if not r.evidence)
    bad = list(rewrites)
    bad[with_change] = Rewrite(bad[with_change].id, "w1", "Changed wording.", "absent", "v1", True)
    bad[without] = Rewrite(bad[without].id, "", bad[without].text, "note", "v1", False)
    bad.append(Rewrite("nope-m0000", "w1", "x", "", "v1", True))
    dropped = bad.pop(-2)
    with pytest.raises(ChallengeError) as caught:
        import_challenge([stream], bad)
    problems = "\n".join(caught.value.problems)
    assert f"{rewrites[with_change].id}: evidence occurs 0 times" in problems
    assert f"{rewrites[without].id}: not verified" in problems
    assert f"{rewrites[without].id}: writer or verifier missing" in problems
    assert f"{rewrites[without].id}: no change, so evidence must be empty" in problems
    assert "nope-m0000: no such message" in problems
    assert f"{dropped.id}: no rewrite" in problems


def test_load_rewrites_reads_the_yaml_form(tmp_path: Path) -> None:
    path = tmp_path / "rewrites.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {"id": "a", "writer": "w", "text": "T. ", "evidence": "T.", "verified_by": "v",
                 "verified": True, "notes": ""},
                {"id": "b", "writer": "", "text": "", "evidence": "", "verified_by": "",
                 "verified": False, "notes": None},
            ]
        )
    )  # fmt: skip
    rewrites, sha = load_rewrites(path)
    assert [r.id for r in rewrites] == ["a", "b"]
    assert rewrites[0].text == "T." and rewrites[0].verified and not rewrites[1].verified
    assert len(sha) == 64


def test_gold_span_reproduces_the_oracle_on_template_text(stream: Stream) -> None:
    oracle_cost, _ = _run(stream, OracleInterpreter(stream.gold))
    cost, policy = _run(stream, GoldSpanInterpreter(stream.gold))
    assert policy.failures == []
    alignment = align(
        policy.proposed,
        stream.gold,
        horizon=(stream.world.start_period, stream.world.end_period),
        periods={m.id: m.period for m in stream.messages},
    )
    assert alignment.errors == ()
    assert cost == pytest.approx(oracle_cost)


def test_gold_span_records_what_the_parser_cannot_read(stream: Stream) -> None:
    rewrites = _rewrites(stream)
    index = next(
        i
        for i, r in enumerate(rewrites)
        if stream.gold_for(r.id) and stream.gold_for(r.id)[0].type is DeltaType.CAPACITY_CHANGE
    )
    vague = "Output there will be trimmed somewhat for a while."
    rewrites[index] = Rewrite(rewrites[index].id, "w1", vague, vague, "v1", True)
    [imported] = import_challenge([stream], rewrites)
    _, policy = _run(imported, GoldSpanInterpreter(imported.gold))
    assert [(m, slot) for m, slot, _ in policy.failures] == [(rewrites[index].id, "value")]
