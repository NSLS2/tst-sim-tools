"""Script to initialize the namespace for running Bluesky."""

import os
from enum import IntEnum
from pathlib import PurePath

import ophyd_async.epics.core._aioca as _ophyd_aioca
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plan_stubs import mv, rd
from bluesky.plans import count, grid_scan, scan, rel_scan
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import UUIDFilenameProvider, YMDPathProvider, init_devices

from tst_sim_tools.devices.detectors import XRTScreenDetector
from tst_sim_tools.devices.mirrors import XRTToroidMirror

# FIXME:
# RunEngine imports pyepics. Reusing pyepics' CA context makes
# aioca/ophyd-async signal connections time out against caproto IOCs.
_ophyd_aioca._use_pyepics_context_if_imported = lambda: None

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
    client = simple(directory="/tmp/tst_testing", readable_storage="/tmp/tst_testing")

tw = TiledWriter(client)
RE.subscribe(tw)

# --- Callback setup ---
bec = BestEffortCallback()
RE.subscribe(bec)


path = PurePath("/tmp/tst_testing")
path_provider = YMDPathProvider(UUIDFilenameProvider(), path)

# --- BMM XRT Sim ---
with init_devices():
    m2 = XRTToroidMirror(
        "XF:31ID1-XRT{BMM:01}M2_TFM:",
        name="m2",
    )
    xrd_det = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}XRD_SAMPLE:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="xrd_det",
    )
    xas_det = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}XAS_SAMPLE:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="xas_det",
    )
    nano_bpm = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}NANO_BPM:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="nano_bpm",
    )
    diag1 = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}Diag1:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="diag1",
    )
    diag2 = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}Diag2:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="diag2",
    )
