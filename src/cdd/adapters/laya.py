"""Laya typed decision models (ConvAI Innovations, Apache-2.0) as a typed-question client.

Each :class:`~cdd.interpreters.typed.TypedQuestion` becomes a Laya ``choice`` question whose
``criteria`` map every option label to its definition (or to the label itself when the
question has none). Laya scores all questions in one forward pass and returns a calibrated
confidence per answer, which becomes the answer's probability and hence the delta's κ.

Needs the optional extra: ``pip install -e ".[laya]"``. The checkpoint is downloaded from the
Hugging Face Hub on first use and pinned to the revision the installed ``laya`` release lists.
Laya's model card reports near-chance zero-shot accuracy, so use a checkpoint fine-tuned on the
training split (``cdd laya-export`` then ``laya.train.finetune``) for reported runs.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from cdd.benchmark.stream import Stream
from cdd.interpreters.typed import (
    MAGNITUDES,
    NONE,
    NOT_STATED,
    OPEN_ENDED,
    AtomicQuestionInterpreter,
    LabelMode,
    TypedAnswer,
    TypedQuestion,
    type_question,
)
from cdd.prompts import render_message
from cdd.schema import Delta, Op

DEFAULT_REPO = "convaiinnovations/laya"
DEFAULT_SUBFOLDER = "typed-decisions"


class LayaClient:
    """Typed-question client backed by a local Laya checkpoint.

    ``model`` is a Hugging Face repo id or a local directory (for example a fine-tuned
    checkpoint). ``truncations`` counts calls where Laya reported that the state or a
    question's options did not fit its token budget.
    """

    def __init__(
        self,
        model: str = DEFAULT_REPO,
        *,
        subfolder: str | None = DEFAULT_SUBFOLDER,
        revision: str | None = None,
        device: str | None = None,
        agent: Any = None,
    ) -> None:
        self.model = model
        self.subfolder = subfolder
        self.revision = revision if agent is not None else _pinned(model, revision)
        self.requests = 0
        self.seconds = 0.0
        self.truncations = 0
        self._agent = agent if agent is not None else _load(model, subfolder, revision, device)

    @property
    def name(self) -> str:
        return f"laya:{self.model}" + (f"/{self.subfolder}" if self.subfolder else "")

    def provenance(self) -> dict[str, Any]:
        """The checkpoint behind the answers: repo or directory, subfolder and revision."""
        try:
            from importlib.metadata import version

            package = version("laya")
        except Exception:  # pragma: no cover - depends on the optional extra
            package = None
        return {
            "client": "laya",
            "model": self.model,
            "subfolder": self.subfolder,
            "revision": self.revision,
            "laya_version": package,
        }

    def ask(self, context: str, questions: Sequence[TypedQuestion]) -> list[TypedAnswer]:
        started = time.perf_counter()
        result = self._agent.predict(context, laya_questions(questions))
        self.seconds += time.perf_counter() - started
        self.requests += 1
        usage = result.get("usage", {})
        if usage.get("truncated") or usage.get("options"):
            self.truncations += 1
        answers = result.get("answers", {})
        out = []
        for question in questions:
            answer = answers.get(question.key, {})
            out.append(
                TypedAnswer(
                    question.key,
                    str(answer.get("choice", "")),
                    float(answer.get("confidence", 0.0)),
                )
            )
        return out


def laya_questions(questions: Sequence[TypedQuestion]) -> dict[str, dict[str, Any]]:
    """Laya ``choice`` questions; option labels map to their definitions."""
    rendered: dict[str, dict[str, Any]] = {}
    for question in questions:
        definitions = question.definitions or question.options
        rendered[question.key] = {
            "type": "choice",
            "instructions": question.prompt,
            "criteria": dict(zip(question.options, definitions, strict=True)),
        }
    return rendered


def _pinned(model: str, revision: str | None) -> str | None:
    """The revision a load would use: the one given, else the one the laya release pins."""
    if revision is not None:
        return revision
    try:
        import laya
    except ImportError:  # pragma: no cover - depends on the optional extra
        return None
    pinned = laya.PINNED_REVISIONS.get(model)
    return None if pinned is None else str(pinned)


def _load(model: str, subfolder: str | None, revision: str | None, device: str | None) -> Any:
    try:
        import laya
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError('Laya is not installed; run: pip install -e ".[laya]"') from exc
    if revision is None:
        revision = laya.PINNED_REVISIONS.get(model)
    return laya.load(model, subfolder=subfolder, revision=revision, device=device)


def training_rows(
    streams: Iterable[Stream], labels: LabelMode = "names"
) -> Iterator[dict[str, Any]]:
    """Fine-tuning rows ``{state, questions, expected}`` from gold deltas, in Laya's format.

    Each message gives up to three rows, one per question group the interpreter asks (type;
    entity; span, window and magnitude), with the same context and option wording as at
    inference. A message without gold deltas gives one row whose answer is ``none``. Use
    training streams only; never the development or test split.
    """
    for stream in streams:
        builder = AtomicQuestionInterpreter(_NoClient(), stream.world, labels=labels)
        world = stream.world
        for message in stream.messages:
            context = render_message(message)
            gold = stream.gold_for(message.id)
            question, decode = type_question(labels)
            encode = {value: label for label, value in decode.items()}
            kind_value = gold[0].type.value if gold else NONE
            yield _row(context, [question], {"type": encode[kind_value]})
            if not gold:
                continue
            delta = gold[0]
            entity, forms = builder.entity_question(delta.type)
            name = next(n for n, entity_id in forms.items() if entity_id == delta.entity)
            yield _row(context, [entity], {"entity": name})
            questions, spans = builder.detail_questions(message, delta.type)
            expected = {"span": questions[0].options[_sentence_of(spans, delta.src.start)]}
            if any(question.key == "window_start" for question in questions):
                expected["window_start"] = f"wk{min(delta.window.start, world.end_period)}"
                expected["window_end"] = (
                    OPEN_ENDED
                    if delta.window.end is None
                    else f"wk{min(delta.window.end, world.end_period)}"
                )
            if delta.type in MAGNITUDES:
                expected["magnitude"] = _nearest_bucket(delta, MAGNITUDES[delta.type])
            yield _row(context, questions, expected)


def pinned_checkpoint_dir(subfolder: str = DEFAULT_SUBFOLDER, repo: str = DEFAULT_REPO) -> str:
    """Local directory of the checkpoint at the revision the installed laya release pins."""
    import laya
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    path = snapshot_download(
        repo,
        revision=laya.PINNED_REVISIONS.get(repo),
        allow_patterns=[
            prefix + name
            for name in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
        ],
    )
    return str(Path(path) / subfolder) if subfolder else str(path)


def finetune(data: str | Path, output_dir: str | Path, **config: Any) -> dict[str, Any]:
    """Fine-tune the pinned typed-decisions checkpoint on exported rows and recalibrate it.

    ``config`` goes to ``laya.train.TrainConfig`` (for example ``epochs=2``). The result loads
    with ``make_client(str(output_dir), subfolder=None)``.
    """
    from laya.train import TrainConfig
    from laya.train import finetune as laya_finetune

    summary: dict[str, Any] = laya_finetune(
        str(data), pinned_checkpoint_dir(), str(output_dir), TrainConfig(**config)
    )
    return summary


class _NoClient:
    """Stands in for a model when only the question builders are needed."""

    def ask(self, context: str, questions: Sequence[TypedQuestion]) -> list[TypedAnswer]:
        raise RuntimeError("question builder only")


def _row(context: str, questions: list[TypedQuestion], expected: dict[str, str]) -> dict[str, Any]:
    return {"state": context, "questions": laya_questions(questions), "expected": expected}


def _sentence_of(spans: Sequence[tuple[int, int]], offset: int) -> int:
    for index, (start, end) in enumerate(spans):
        if start <= offset < end:
            return index
    return 0


def _bucket_value(label: str) -> float:
    return float(label.split()[0].rstrip("%"))


def _nearest_bucket(delta: Delta, buckets: Sequence[str]) -> str:
    """The magnitude bucket closest to a numeric delta: percent for scales, days for adds."""
    if isinstance(delta.value, str):
        return NOT_STATED
    amount = (delta.value - 1) * 100 if delta.op is Op.SCALE else delta.value
    return min(buckets, key=lambda label: abs(_bucket_value(label) - amount))


def make_client(model: str = DEFAULT_REPO, **options: Any) -> LayaClient:
    """Factory for experiment configs: ``client: cdd.adapters.laya:make_client``."""
    return LayaClient(model, **options)
