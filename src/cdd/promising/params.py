"""Parameters of the order-promising study (docs/research/icores-revision-scope.md, D7–D10).

Margins and penalties are declared synthetic values, not measured costs. Each penalty is a
fraction of the SKU's unit margin, and the margin is a fraction of the SKU's mean offer price in
the world, so one file serves every world.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cdd.world import World


class PromisingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Value: m_k = margin_fraction x mean offer price of SKU k.
    margin_fraction: float = Field(0.5, gt=0)
    # Per unit and week a delivery comes after the requested week.
    lateness_fraction: float = Field(0.2, ge=0)
    # Per unit not delivered by its promised week (once).
    broken_fraction: float = Field(0.5, ge=0)
    # Per accepted unit still undelivered after the last week (once).
    terminal_fraction: float = Field(0.5, ge=0)

    # Customer orders (D8). demand_scale > 1 is the supply-constrained (surge) setting: orders
    # total that multiple of the realised demand the planner's supply was built for.
    demand_scale: float = Field(1.0, gt=0)
    orders_per_week: int = Field(3, ge=1)
    dirichlet_alpha: float = Field(1.0, gt=0)
    customer_lead_weeks: tuple[int, ...] = (1, 2, 3)
    # Requested weeks before start_period + first_requested_offset get no orders.
    first_requested_offset: int = Field(3, ge=0)

    # The promise decision (D9) and the hold-notice sensitivity (D10).
    max_late_weeks: int = Field(2, ge=0)
    time_limit: float = Field(10.0, gt=0)
    reveal_holds: bool = False

    @model_validator(mode="after")
    def _orders_fit_the_horizon(self) -> Self:
        if not self.customer_lead_weeks or min(self.customer_lead_weeks) < 0:
            raise ValueError("customer_lead_weeks must be non-negative and not empty")
        if self.first_requested_offset < max(self.customer_lead_weeks):
            raise ValueError(
                "first_requested_offset must be at least the longest customer lead time, "
                "or early orders would arrive before the first week"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PromisingParams:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))


def margins(world: World, params: PromisingParams) -> dict[str, float]:
    """Unit margin per SKU: a fraction of the SKU's mean offer price across suppliers."""
    prices: dict[str, list[float]] = {sku.id: [] for sku in world.skus}
    for offer in world.offers:
        prices[offer.sku].append(offer.price)
    return {
        sku: params.margin_fraction * sum(values) / len(values)
        for sku, values in prices.items()
        if values
    }


def unit_rates(margin: Mapping[str, float], params: PromisingParams) -> dict[str, dict[str, float]]:
    """Per SKU: margin, lateness per unit-week, broken-promise and terminal penalty per unit."""
    return {
        sku: {
            "margin": m,
            "lateness": params.lateness_fraction * m,
            "broken": params.broken_fraction * m,
            "terminal": params.terminal_fraction * m,
        }
        for sku, m in margin.items()
    }
