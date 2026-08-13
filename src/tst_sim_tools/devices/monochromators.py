"""Monochromator device abstractions."""

from typing import Annotated as Ann

from ophyd_async.core import SignalR, SignalRW, StandardReadable
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class XRTDCMCrystal(StandardReadable, EpicsDevice):
    """One physical crystal optic in the split XRT DCM model."""

    # --- Physical pose ---
    center_x: Ann[SignalRW[float], PvSuffix("center:x"), Format.CONFIG_SIGNAL]
    center_y: Ann[SignalRW[float], PvSuffix("center:y"), Format.UNCACHED_SIGNAL]
    center_z: Ann[SignalRW[float], PvSuffix("center:z"), Format.UNCACHED_SIGNAL]
    pitch: Ann[SignalRW[float], PvSuffix("pitch"), Format.UNCACHED_SIGNAL]
    roll: Ann[SignalRW[float], PvSuffix("roll"), Format.HINTED_UNCACHED_SIGNAL]
    yaw: Ann[SignalRW[float], PvSuffix("yaw"), Format.UNCACHED_SIGNAL]
    position_roll: Ann[SignalRW[float], PvSuffix("positionRoll"), Format.CONFIG_SIGNAL]
    extra_pitch: Ann[SignalRW[float], PvSuffix("extraPitch"), Format.CONFIG_SIGNAL]
    extra_roll: Ann[SignalRW[float], PvSuffix("extraRoll"), Format.CONFIG_SIGNAL]
    extra_yaw: Ann[SignalRW[float], PvSuffix("extraYaw"), Format.CONFIG_SIGNAL]

    # --- Metadata ---
    material: Ann[SignalR[str], PvSuffix("material"), Format.CONFIG_SIGNAL]
    rotation_sequence: Ann[SignalR[str], PvSuffix("rotationSequence"), Format.CONFIG_SIGNAL]
    extra_rotation_sequence: Ann[SignalR[str], PvSuffix("extraRotationSequence"), Format.CONFIG_SIGNAL]
    shape: Ann[SignalR[str], PvSuffix("shape"), Format.CONFIG_SIGNAL]


class BMMSplitXRTDCM(StandardReadable):
    """BMM DCM as two independently posed XRT crystal optical elements."""

    def __init__(self, prefix: str, name: str = "") -> None:
        with self.add_children_as_readables():
            self.crystal_1 = XRTDCMCrystal(f"{prefix}DCM_C1:")
            self.crystal_2 = XRTDCMCrystal(f"{prefix}DCM_C2:")
        super().__init__(name=name)
