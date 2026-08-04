"""Mirror device abstractions."""

from typing import Annotated as Ann

from ophyd_async.core import (
    SignalRW,
    StandardReadable,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class XRTToroidMirror(StandardReadable, EpicsDevice):
    """Toroid mirror controls."""

    pitch: Ann[SignalRW[float], PvSuffix("pitch"), Format.HINTED_UNCACHED_SIGNAL]
    roll: Ann[SignalRW[float], PvSuffix("roll"), Format.HINTED_UNCACHED_SIGNAL]
    yaw: Ann[SignalRW[float], PvSuffix("yaw"), Format.HINTED_UNCACHED_SIGNAL]
