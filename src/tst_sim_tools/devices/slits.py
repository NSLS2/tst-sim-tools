"""Device abstractions for slits."""

from typing import Annotated as Ann

from ophyd_async.core import (
    SignalRW,
    StandardReadable,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class XRTRectangularAperature(StandardReadable, EpicsDevice):
    """EPICS controls for an XRT rectangular aperature."""

    center_x: Ann[SignalRW[float], PvSuffix("center:x"), Format.CONFIG_SIGNAL]
    center_y: Ann[SignalRW[float], PvSuffix("center:y"), Format.CONFIG_SIGNAL]
    center_z: Ann[SignalRW[float], PvSuffix("center:z"), Format.CONFIG_SIGNAL]

    blades_left: Ann[SignalRW[float], PvSuffix("blades:left"), Format.CONFIG_SIGNAL]
    blades_right: Ann[SignalRW[float], PvSuffix("blades:right"), Format.CONFIG_SIGNAL]
    blades_bottom: Ann[SignalRW[float], PvSuffix("blades:bottom"), Format.CONFIG_SIGNAL]
    blades_top: Ann[SignalRW[float], PvSuffix("blades:top"), Format.CONFIG_SIGNAL]
