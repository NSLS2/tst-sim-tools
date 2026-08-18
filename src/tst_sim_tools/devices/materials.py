"""Material device abstractions."""

from typing import Annotated as Ann

from ophyd_async.core import SignalR
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class XRTCrystalSi(EpicsDevice):
    """Read-only EPICS metadata for an XRT crystal silicon material."""

    # --- Crystal lattice ---
    lattice_spacing: Ann[SignalR[float], PvSuffix("d")]
