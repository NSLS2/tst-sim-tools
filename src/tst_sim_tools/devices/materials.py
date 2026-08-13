"""Material device abstractions."""

from typing import Annotated as Ann

from ophyd_async.core import SignalR, StandardReadable
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class XRTCrystalMaterial(StandardReadable, EpicsDevice):
    """Read-only EPICS metadata for an XRT crystal material."""

    # --- Crystal lattice ---
    temperature: Ann[SignalR[float], PvSuffix("tK"), Format.CONFIG_SIGNAL]
    lattice_constant: Ann[SignalR[float], PvSuffix("a"), Format.CONFIG_SIGNAL]
    miller_indices: Ann[SignalR[str], PvSuffix("hkl"), Format.CONFIG_SIGNAL]
    lattice_spacing: Ann[SignalR[float], PvSuffix("d"), Format.CONFIG_SIGNAL]
    unit_cell_volume: Ann[SignalR[float], PvSuffix("V"), Format.CONFIG_SIGNAL]

    # --- Material composition and scattering ---
    quantities: Ann[SignalR[str], PvSuffix("quantities"), Format.CONFIG_SIGNAL]
    density: Ann[SignalR[float], PvSuffix("rho"), Format.CONFIG_SIGNAL]
    thickness: Ann[SignalR[str], PvSuffix("t"), Format.CONFIG_SIGNAL]
    debye_waller_factor: Ann[SignalR[float], PvSuffix("factDW"), Format.CONFIG_SIGNAL]
    geometry: Ann[SignalR[str], PvSuffix("geom"), Format.CONFIG_SIGNAL]
    scattering_table: Ann[SignalR[str], PvSuffix("table"), Format.CONFIG_SIGNAL]
    material_name: Ann[SignalR[str], PvSuffix("name"), Format.CONFIG_SIGNAL]
    refractive_index: Ann[SignalR[str], PvSuffix("refractiveIndex"), Format.CONFIG_SIGNAL]

    # --- Diffraction model ---
    volumetric_diffraction: Ann[SignalR[bool], PvSuffix("volumetricDiffraction"), Format.CONFIG_SIGNAL]
    use_takagi_taupin: Ann[SignalR[bool], PvSuffix("useTT"), Format.CONFIG_SIGNAL]
    poisson_ratio: Ann[SignalR[str], PvSuffix("nu"), Format.CONFIG_SIGNAL]
    mosaicity: Ann[SignalR[float], PvSuffix("mosaicity"), Format.CONFIG_SIGNAL]
    efficiency: Ann[SignalR[str], PvSuffix("efficiency"), Format.CONFIG_SIGNAL]
    efficiency_file: Ann[SignalR[str], PvSuffix("efficiencyFile"), Format.CONFIG_SIGNAL]
