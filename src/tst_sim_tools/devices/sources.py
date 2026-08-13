"""Device abstractions for XRT X-ray sources."""

from typing import Annotated as Ann

from ophyd_async.core import (
    SignalRW,
    StandardReadable,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class XRTWiggler(StandardReadable, EpicsDevice):
    """EPICS controls for an XRT wiggler source."""

    # --- Source geometry ---
    center_x: Ann[SignalRW[float], PvSuffix("center:x"), Format.CONFIG_SIGNAL]
    center_y: Ann[SignalRW[float], PvSuffix("center:y"), Format.CONFIG_SIGNAL]
    center_z: Ann[SignalRW[float], PvSuffix("center:z"), Format.CONFIG_SIGNAL]
    pitch: Ann[SignalRW[float], PvSuffix("pitch"), Format.CONFIG_SIGNAL]
    yaw: Ann[SignalRW[float], PvSuffix("yaw"), Format.CONFIG_SIGNAL]

    # --- Ray generation ---
    nrays: Ann[SignalRW[int], PvSuffix("nrays"), Format.CONFIG_SIGNAL]
    filament_beam: Ann[SignalRW[bool], PvSuffix("filamentBeam"), Format.CONFIG_SIGNAL]
    uniform_ray_density: Ann[SignalRW[bool], PvSuffix("uniformRayDensity"), Format.CONFIG_SIGNAL]

    # --- Electron beam ---
    electron_energy: Ann[SignalRW[float], PvSuffix("eE"), Format.CONFIG_SIGNAL]
    electron_current: Ann[SignalRW[float], PvSuffix("eI"), Format.CONFIG_SIGNAL]
    electron_energy_spread: Ann[SignalRW[float], PvSuffix("eEspread"), Format.CONFIG_SIGNAL]
    electron_beam_size_x: Ann[SignalRW[float], PvSuffix("eSigmaX"), Format.CONFIG_SIGNAL]
    electron_beam_size_z: Ann[SignalRW[float], PvSuffix("eSigmaZ"), Format.CONFIG_SIGNAL]
    electron_emittance_x: Ann[SignalRW[float], PvSuffix("eEpsilonX"), Format.CONFIG_SIGNAL]
    electron_emittance_z: Ann[SignalRW[float], PvSuffix("eEpsilonZ"), Format.CONFIG_SIGNAL]
    beta_x: Ann[SignalRW[float], PvSuffix("betaX"), Format.CONFIG_SIGNAL]
    beta_z: Ann[SignalRW[float], PvSuffix("betaZ"), Format.CONFIG_SIGNAL]

    # --- Photon sampling ---
    x_prime_max: Ann[SignalRW[float], PvSuffix("xPrimeMax"), Format.CONFIG_SIGNAL]
    z_prime_max: Ann[SignalRW[float], PvSuffix("zPrimeMax"), Format.CONFIG_SIGNAL]
    energy_distribution: Ann[SignalRW[str], PvSuffix("distE"), Format.CONFIG_SIGNAL]
    min_energy: Ann[SignalRW[float], PvSuffix("eMin"), Format.CONFIG_SIGNAL]
    max_energy: Ann[SignalRW[float], PvSuffix("eMax"), Format.CONFIG_SIGNAL]
    near_field_distance: Ann[SignalRW[str], PvSuffix("R0"), Format.CONFIG_SIGNAL]

    # --- Wiggler magnet ---
    deflection_parameter: Ann[SignalRW[float], PvSuffix("K"), Format.CONFIG_SIGNAL]
    period: Ann[SignalRW[float], PvSuffix("period"), Format.CONFIG_SIGNAL]
    num_periods: Ann[SignalRW[int], PvSuffix("n"), Format.CONFIG_SIGNAL]
