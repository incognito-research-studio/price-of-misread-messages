"""Evaluate your own reader and planner with this package (`cdd`).

Run from the repository root:

    python examples/evaluate_your_own_reader.py

The reader below is deliberately small: it understands quality holds and releases and nothing
else. The planner orders up to a few weeks of forecast demand from the cheapest open plant.
Replace either with your own; the state, SOP gate, simulator and scoring stay the same.
"""

from __future__ import annotations

from cdd.benchmark import GeneratorConfig, generate
from cdd.evaluate import evaluate, report
from cdd.interpreters import RegexInterpreter
from cdd.interpreters.base import Interpretation
from cdd.orders import InventoryPosition, OrderLine, Plan, capacity_available
from cdd.schema import Attribute, DeltaType, EntityKind, Message, Proposal, Span
from cdd.state import StateView
from cdd.text import Gazetteer, sentence_spans
from cdd.world import World


class HoldReader:
    """A reader: turns each message into typed delta proposals (here, holds and releases only).

    A reader proposes the delta type, the entity and the sentence that supports it. The span
    parser fills the value and window, so the reader never writes numbers it did not read.
    """

    name = "hold-reader"

    def __init__(self, world: World) -> None:
        self.gazetteer = Gazetteer(world)

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        proposals = []
        for start, end in sentence_spans(message.text):
            sentence = message.text[start:end].lower()
            lot = self.gazetteer.find(sentence, EntityKind.LOT)
            if lot is None:
                continue
            if "release" in sentence:
                kind = DeltaType.QUALITY_RELEASE
            elif "hold" in sentence:
                kind = DeltaType.QUALITY_HOLD
            else:
                continue
            span = Span(message_id=message.id, start=start, end=end)
            proposals.append(Proposal(type=kind, entity=lot, span=span))
        return Interpretation(tuple(proposals))


class OrderUpTo:
    """A planner: reads the interpreted state and returns this week's orders.

    It orders up to ``weeks`` of forecast demand, net of stock, open orders and backlog, from the
    cheapest plants that are open and have capacity.
    """

    name = "order-up-to"

    def __init__(self, weeks: float = 3.0) -> None:
        self.weeks = weeks

    def plan(self, view: StateView, position: InventoryPosition) -> Plan:
        world, t = view.world, position.period
        used: dict[str, float] = {}
        orders = []
        for sku in world.skus:
            target = self.weeks * view.number(sku.id, Attribute.DEMAND, t)
            open_orders = sum(o.quantity for o in position.outstanding if o.sku == sku.id)
            have = (
                position.on_hand.get(sku.id, 0.0) + open_orders - position.backlog.get(sku.id, 0.0)
            )
            need = max(0.0, target - have)
            options = []
            for plant, line_sku in world.lines():
                offer = world.offer_for(plant, line_sku)
                if line_sku == sku.id and offer is not None:
                    options.append((view.number(offer.id, Attribute.PRICE, t), plant))
            for _, plant in sorted(options):
                quantity = min(need, capacity_available(view, plant, t) - used.get(plant, 0.0))
                if quantity > 0:
                    orders.append(OrderLine(plant, sku.id, quantity))
                    used[plant] = used.get(plant, 0.0) + quantity
                    need -= quantity
                if need <= 0:
                    break
        return Plan(period=t, orders=tuple(orders), status="heuristic")


def main(n_streams: int = 2) -> str:
    # Operational costs live in the world: set them to match your business.
    config = GeneratorConfig.model_validate(
        {
            "name": "example",
            "n_streams": n_streams,
            "seed": 5,
            "world": {"holding_cost": 0.5, "backlog_cost": 25.0},
        }
    )
    results = []
    for stream in generate(config):
        # Your reader with the built-in solver as planner.
        results.append(evaluate(stream, HoldReader(stream.world)))
        # A built-in reader with your planner.
        results.append(evaluate(stream, RegexInterpreter(stream.world), OrderUpTo()))
    text = report(results)
    print(text)
    return text


if __name__ == "__main__":
    main()
