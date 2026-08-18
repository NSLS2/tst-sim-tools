"""Mirror device abstractions."""

from typing import Annotated as Ann

from ophyd_async.core import (
    SignalR,
    SignalRW,
    StandardReadable,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class XRTOpticalElement(StandardReadable, EpicsDevice):
    """Optical element controls."""

    fixed_pitch: Ann[SignalR[float], PvSuffix("pitch"), Format.CONFIG_SIGNAL]
    fixed_roll: Ann[SignalR[float], PvSuffix("roll"), Format.CONFIG_SIGNAL]
    fixed_yaw: Ann[SignalR[float], PvSuffix("yaw"), Format.CONFIG_SIGNAL]

    pitch: Ann[SignalRW[float], PvSuffix("extraPitch"), Format.HINTED_UNCACHED_SIGNAL]
    roll: Ann[SignalRW[float], PvSuffix("extraRoll"), Format.HINTED_UNCACHED_SIGNAL]
    yaw: Ann[SignalRW[float], PvSuffix("extraYaw"), Format.HINTED_UNCACHED_SIGNAL]

    center_x: Ann[SignalRW[float], PvSuffix("center:x"), Format.CONFIG_SIGNAL]
    center_y: Ann[SignalRW[float], PvSuffix("center:y"), Format.CONFIG_SIGNAL]
    center_z: Ann[SignalRW[float], PvSuffix("center:z"), Format.CONFIG_SIGNAL]

    # --- Metadata ---
    material: Ann[SignalR[str], PvSuffix("material"), Format.CONFIG_SIGNAL]


class XRTToroidMirror(XRTOpticalElement):
    """Toroid mirror controls."""

    meridional_radius: Ann[SignalRW[float], PvSuffix("R"), Format.CONFIG_SIGNAL]


class XRTParabolicalMirror(XRTOpticalElement):
    """Parabolical mirror controls."""

    ...
