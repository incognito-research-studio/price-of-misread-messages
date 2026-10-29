from __future__ import annotations

import pytest

from cdd.state import BitemporalState
from cdd.world import World
from helpers import build_world


@pytest.fixture
def world() -> World:
    return build_world()


@pytest.fixture
def state(world: World) -> BitemporalState:
    return BitemporalState(world)
