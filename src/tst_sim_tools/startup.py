"""Script to initialize the namespace for running Bluesky."""

import os
from pathlib import PurePath

import ophyd_async.epics.core._aioca as _ophyd_aioca
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import UUIDFilenameProvider, YMDPathProvider, init_devices
from tsttools.sim.devices.detectors import XRTScreenDetector

from tst_sim_tools.devices.mirrors import XRTToroidMirror

# FIXME:
# RunEngine imports pyepics. Reusing pyepics' CA context makes
# aioca/ophyd-async signal connections time out against caproto IOCs.
_ophyd_aioca._use_pyepics_context_if_imported = lambda: None

RE = RunEngine({})

# --- Tiled setup ---
if os.getenv("TILED") == 1:
    from tiled.client import simple

    client = simple()
else:
    from tiled.client import from_uri

    if os.getenv("TILED") == 2:
        client = from_uri("https://tiled.nsls2.bnl.gov")["tst/migration"]
    else:
        client = from_uri("https://tiled-staging.nsls2.bnl.gov")["tst/raw"]
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
        "XF:31ID1-XRT{BMM}M2_TFM:",
        name="m2",
    )
    xrd_det = XRTScreenDetector(
        "XF:31ID1-XRT{BMM}XRD_SAMPLE:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="xrd_det",
    )
    xas_det = XRTScreenDetector(
        "XF:31ID1-XRT{BMM}XAS_SAMPLE:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="xas_det",
    )
    nano_bpm = XRTScreenDetector(
        "XF:31ID1-XRT{BMM}NANO_BPM:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="nano_bpm",
    )
    diag1 = XRTScreenDetector(
        "XF:31ID1-XRT{BMM}Diag1:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="diag1",
    )
    diag2 = XRTScreenDetector(
        "XF:31ID1-XRT{BMM}Diag2:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="diag2",
    )
