"""A benchmark stream: a world, its messages and the gold deltas behind them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cdd.schema import Delta, Message, Window
from cdd.world import World


class Shock(BaseModel):
    """A cost-relevant event that no delta type can express (the out-of-schema slice).

    A ``yield_shortfall`` delivers only ``factor`` of each order a plant accepts in ``window``.
    The simulator applies it; the gold deltas of its message are empty.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["yield_shortfall"]
    message_id: str
    plant: str
    window: Window
    factor: float


class Stream(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    seed: int
    world: World
    messages: tuple[Message, ...]
    gold: tuple[Delta, ...]
    shocks: tuple[Shock, ...] = ()
    # How the stream was produced beyond the generator, e.g. the paraphrase model and its
    # acceptance counts.
    provenance: dict[str, Any] = Field(default_factory=dict)

    def yields(self) -> dict[tuple[str, int], float]:
        """Delivered share of accepted orders by (plant, period), for every shock."""
        out: dict[tuple[str, int], float] = {}
        for shock in self.shocks:
            for period in self.world.periods:
                if shock.window.contains(period):
                    out[(shock.plant, period)] = out.get((shock.plant, period), 1.0) * shock.factor
        return out

    def gold_for(self, message_id: str) -> tuple[Delta, ...]:
        return tuple(delta for delta in self.gold if delta.message_id == message_id)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Stream:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_streams(directory: str | Path) -> list[Stream]:
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no stream files in {directory}")
    return [Stream.load(path) for path in paths]
