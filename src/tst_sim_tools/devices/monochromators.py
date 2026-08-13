"""Monochromator device abstractions."""

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


class XRTDCM(StandardReadable, EpicsDevice):
    """EPICS controls and metadata for an XRT double-crystal monochromator."""

    # --- Bragg and crystal alignment ---
    bragg: Ann[SignalRW[float], PvSuffix("bragg"), Format.HINTED_UNCACHED_SIGNAL]
    bragg_offset: Ann[SignalRW[float], PvSuffix("braggOffset"), Format.CONFIG_SIGNAL]
    crystal_1_roll: Ann[SignalRW[float], PvSuffix("cryst1roll"), Format.UNCACHED_SIGNAL]
    crystal_2_roll: Ann[SignalRW[float], PvSuffix("cryst2roll"), Format.UNCACHED_SIGNAL]
    crystal_2_pitch: Ann[SignalRW[float], PvSuffix("cryst2pitch"), Format.UNCACHED_SIGNAL]
    crystal_2_fine_pitch: Ann[SignalRW[float], PvSuffix("cryst2finePitch"), Format.UNCACHED_SIGNAL]
    crystal_2_perp_translation: Ann[SignalRW[float], PvSuffix("cryst2perpTransl"), Format.UNCACHED_SIGNAL]
    crystal_2_long_translation: Ann[SignalRW[float], PvSuffix("cryst2longTransl"), Format.UNCACHED_SIGNAL]
    fixed_offset: Ann[SignalRW[str], PvSuffix("fixedOffset"), Format.CONFIG_SIGNAL]

    # --- Optical element pose ---
    center_x: Ann[SignalRW[float], PvSuffix("center:x"), Format.CONFIG_SIGNAL]
    center_y: Ann[SignalRW[float], PvSuffix("center:y"), Format.CONFIG_SIGNAL]
    center_z: Ann[SignalRW[float], PvSuffix("center:z"), Format.CONFIG_SIGNAL]
    pitch: Ann[SignalRW[float], PvSuffix("pitch"), Format.CONFIG_SIGNAL]
    roll: Ann[SignalRW[float], PvSuffix("roll"), Format.CONFIG_SIGNAL]
    yaw: Ann[SignalRW[float], PvSuffix("yaw"), Format.CONFIG_SIGNAL]
    position_roll: Ann[SignalRW[float], PvSuffix("positionRoll"), Format.CONFIG_SIGNAL]
    extra_pitch: Ann[SignalRW[float], PvSuffix("extraPitch"), Format.CONFIG_SIGNAL]
    extra_roll: Ann[SignalRW[float], PvSuffix("extraRoll"), Format.CONFIG_SIGNAL]
    extra_yaw: Ann[SignalRW[float], PvSuffix("extraYaw"), Format.CONFIG_SIGNAL]

    # --- Physical limits ---
    physical_limit_x_min: Ann[SignalRW[float], PvSuffix("limPhysX:lmin"), Format.CONFIG_SIGNAL]
    physical_limit_x_max: Ann[SignalRW[float], PvSuffix("limPhysX:lmax"), Format.CONFIG_SIGNAL]
    physical_limit_y_min: Ann[SignalRW[float], PvSuffix("limPhysY:lmin"), Format.CONFIG_SIGNAL]
    physical_limit_y_max: Ann[SignalRW[float], PvSuffix("limPhysY:lmax"), Format.CONFIG_SIGNAL]
    crystal_2_physical_limit_x_min: Ann[SignalRW[float], PvSuffix("limPhysX2:lmin"), Format.CONFIG_SIGNAL]
    crystal_2_physical_limit_x_max: Ann[SignalRW[float], PvSuffix("limPhysX2:lmax"), Format.CONFIG_SIGNAL]
    crystal_2_physical_limit_y_min: Ann[SignalRW[float], PvSuffix("limPhysY2:lmin"), Format.CONFIG_SIGNAL]
    crystal_2_physical_limit_y_max: Ann[SignalRW[float], PvSuffix("limPhysY2:lmax"), Format.CONFIG_SIGNAL]

    # --- Metadata ---
    crystal_1_material: Ann[SignalR[str], PvSuffix("material"), Format.CONFIG_SIGNAL]
    crystal_2_material: Ann[SignalR[str], PvSuffix("material2"), Format.CONFIG_SIGNAL]
    rotation_sequence: Ann[SignalR[str], PvSuffix("rotationSequence"), Format.CONFIG_SIGNAL]
    extra_rotation_sequence: Ann[SignalR[str], PvSuffix("extraRotationSequence"), Format.CONFIG_SIGNAL]
    shape: Ann[SignalR[str], PvSuffix("shape"), Format.CONFIG_SIGNAL]
