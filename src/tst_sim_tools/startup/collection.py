"""Script to initialize the namespace for running Bluesky."""

import os
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks.zmq import Publisher
from bluesky.plan_stubs import mv, rd
from bluesky.plans import count, grid_scan, list_scan, rel_scan, scan
from bluesky_tiled_plugins import TiledWriter
import ophyd_async.epics.core._aioca as _ophyd_aioca  # noqa: PLC2701

from tst_sim_tools.plans.bmm import (
    acquire_with_energy_scan,
    change_energy_stub,
    scan_energy,
)

# FIXME:
# RunEngine imports pyepics. Reusing pyepics' CA context makes
# aioca/ophyd-async signal connections time out against caproto IOCs.
_ophyd_aioca._use_pyepics_context_if_imported = lambda: None  # noqa: SLF001

RE = RunEngine({})


# --- Tiled setup ---
class TiledChoice(IntEnum):
    """Choice of tiled server to connect to."""

    SIMPLE = 0
    STAGING = 1
    PROD = 2


tiled_choice = int(os.getenv("TILED", "0"))
if tiled_choice != TiledChoice.SIMPLE:
    from tiled.client import from_uri

    if tiled_choice == TiledChoice.PROD:
        print("Will connect to tiled...")
        client = from_uri("https://tiled.nsls2.bnl.gov")["tst/migration"]
    else:
        print("Will connect to tiled-staging...")
        client = from_uri("https://tiled-staging.nsls2.bnl.gov")["tst/raw"]
else:
    from tiled.client import simple

    print("Will connect to local tiled...")
    client = simple(directory="/tmp/tst_testing", readable_storage="/nsls2/data/tst/legacy")

tw = TiledWriter(client)
RE.subscribe(tw)

# publisher = Publisher("127.0.0.1:35091")
# RE.subscribe(publisher)

from .devices import *
