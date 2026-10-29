"""Customer orders for the promising study (D8), split from the replenishment demand path.

For each SKU and requested week from ``start_period + first_requested_offset`` on, the realised
demand of that week (the same common-random-numbers path the replenishment study uses), times
``demand_scale``, is split into ``orders_per_week`` orders with Dirichlet shares. Quantities stay
continuous and the last order takes the remainder, so each week's orders sum exactly to that
total. Each order arrives a customer lead time before its requested week. The draws come from
their own generator, so streams, gold deltas and demand are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cdd.promising.params import PromisingParams
from cdd.world import World

# Mixed into the stream seed so the order draws never share a sequence with the demand draws.
ORDERS_STREAM = 0x0D3E5


@dataclass(frozen=True)
class CustomerOrder:
    id: str
    sku: str
    quantity: float
    arrival: int
    requested: int


def generate_orders(
    world: World,
    demand: Mapping[tuple[str, int], float],
    seed: int,
    params: PromisingParams,
) -> tuple[CustomerOrder, ...]:
    """Orders in arrival order (then id). ``demand`` maps (SKU, week) to realised demand."""
    rng = np.random.default_rng([seed, ORDERS_STREAM])
    leads = np.asarray(params.customer_lead_weeks)
    orders: list[CustomerOrder] = []
    first = world.start_period + params.first_requested_offset
    for week in range(first, world.end_period + 1):
        for sku in world.skus:
            total = max(0.0, demand[(sku.id, week)]) * params.demand_scale
            shares = rng.dirichlet([params.dirichlet_alpha] * params.orders_per_week)
            lead = rng.choice(leads, size=params.orders_per_week)
            quantities = [total * float(share) for share in shares[:-1]]
            quantities.append(max(0.0, total - sum(quantities)))
            for index, (quantity, weeks) in enumerate(zip(quantities, lead, strict=True)):
                orders.append(
                    CustomerOrder(
                        id=f"{sku.id}-w{week}-{index}",
                        sku=sku.id,
                        quantity=quantity,
                        arrival=week - int(weeks),
                        requested=week,
                    )
                )
    return tuple(sorted(orders, key=lambda order: (order.arrival, order.id)))
