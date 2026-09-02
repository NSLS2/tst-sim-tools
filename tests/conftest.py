import asyncio
import os
from collections import defaultdict
from collections.abc import Generator

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pytest
from bluesky import RunEngine


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None]:
    """Close every Matplotlib figure after each test."""
    yield
    plt.close("all")


@pytest.fixture
def run_engine() -> Generator[RunEngine]:
    """Provide a RunEngine with an isolated, deterministically closed loop."""
    loop = asyncio.new_event_loop()
    engine = RunEngine({}, call_returns_result=True, loop=loop)
    try:
        yield engine
    finally:
        if engine.state not in ("idle", "panicked"):
            try:
                engine.halt()
            except RuntimeError:
                pass
        loop.call_soon_threadsafe(loop.stop)
        engine._th.join()
        loop.close()


@pytest.fixture
def documents():
    """Collect emitted Bluesky documents by name."""
    collected = defaultdict(list)

    def collect(name, document):
        collected[name].append(document)

    return collected, collect
