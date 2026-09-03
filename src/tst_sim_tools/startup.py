"""Script to initialize the namespace for running Bluesky."""

import os
from enum import IntEnum
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import ophyd_async.epics.core._aioca as _ophyd_aioca  # noqa: PLC2701
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks.zmq import Publisher
from bluesky.plan_stubs import mv, rd
from bluesky.plans import count, grid_scan, list_scan, rel_scan, scan
from bluesky_tiled_plugins import TiledWriter
from nslsii.ophyd_async.providers import NSLS2PathProvider
from ophyd_async.core import UUIDFilenameProvider, YMDPathProvider, init_devices

from tst_sim_tools.devices.detectors import XRTScreenDetector
from tst_sim_tools.devices.materials import XRTCrystalSi
from tst_sim_tools.devices.mirrors import XRTOpticalElement, XRTParabolicalMirror, XRTToroidMirror
from tst_sim_tools.devices.slits import XRTRectangularAperature
from tst_sim_tools.devices.sources import XRTWiggler
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
    path = PurePath("/nsls2/data/tst/legacy/")
    # TODO: Use NSLS2PathProvider from nslsii package
    path_provider = YMDPathProvider(UUIDFilenameProvider(), path)
else:
    from tiled.client import simple

    print("Will connect to local tiled...")
    client = simple(directory="/tmp/tst_testing", readable_storage="/nsls2/data/tst/legacy")
    path = PurePath("/nsls2/data/tst/legacy/")
    path_provider = YMDPathProvider(UUIDFilenameProvider(), path)

tw = TiledWriter(client)
RE.subscribe(tw)

# publisher = Publisher("127.0.0.1:35091")
# RE.subscribe(publisher)

# --- BMM XRT Sim ---
with init_devices():
    si111 = XRTCrystalSi(
        "XF:31ID1-XRT{BMM:01}Si111:",
        name="si111",
    )
    si311 = XRTCrystalSi(
        "XF:31ID1-XRT{BMM:01}Si311:",
        name="si311",
    )
    tpw = XRTWiggler(
        "XF:31ID1-XRT{BMM:01}TPW:",
        name="tpw",
    )
    fe_mask = XRTRectangularAperature(
        "XF:31ID1-XRT{BMM:01}FE_MASK:",
        name="fe_mask",
    )
    m1 = XRTParabolicalMirror(
        "XF:31ID1-XRT{BMM:01}M1_VCM:",
        name="m1",
    )
    diag1 = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}Diag1:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="diag1",
    )
    dcm_c1 = XRTOpticalElement(
        "XF:31ID1-XRT{BMM:01}DCM_C1:",
        name="dcm_c1",
    )
    dcm_c2 = XRTOpticalElement(
        "XF:31ID1-XRT{BMM:01}DCM_C2:",
        name="dcm_c2",
    )
    pink_beam_stop = XRTRectangularAperature(
        "XF:31ID1-XRT{BMM:01}PinkBeamStop:",
        name="pink_beam_stop",
    )
    dm2_slits = XRTRectangularAperature(
        "XF:31ID1-XRT{BMM:01}DM2_Slits:",
        name="dm2_slits",
    )
    diag2 = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}Diag2:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="diag2",
    )
    m2 = XRTToroidMirror(
        "XF:31ID1-XRT{BMM:01}M2_TFM:",
        name="m2",
    )
    m3 = XRTOpticalElement(
        "XF:31ID1-XRT{BMM:01}M3_HRM:",
        name="m3",
    )
    nano_bpm = XRTScreenDetector(
        "XF:31ID1-XRT{BMM:01}NANO_BPM:",
        datakey_suffix="_image",
        path_provider=path_provider,
        name="nano_bpm",
    )
    beam_shutter = XRTRectangularAperature(
        "XF:31ID1-XRT{BMM:01}BeamShutter:",
        name="beam_shutter",
    )
    dm3_slits = XRTRectangularAperature(
        "XF:31ID1-XRT{BMM:01}DM3_Slits:",
        name="dm3_slits",
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
