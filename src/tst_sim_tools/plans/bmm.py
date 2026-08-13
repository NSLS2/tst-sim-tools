"""Bluesky plans specific to BMM's XRT model."""

import math

import numpy as np
from bluesky import plan_stubs as bps
from bluesky.utils import MsgGenerator, plan

from ..devices.materials import XRTCrystalMaterial
from ..devices.monochromators import XRTDCM
from ..devices.sources import XRTWiggler

HC_EV_ANGSTROM = 12398.419297617678
BMM_BEAM_INCLINATION = 0.00700
BMM_DCM_FIXED_EXIT = 30.0

# Generated from bmm_improved.xml with xrt==2.0.0b1 by loading the XML BeamLine
# and sampling -DCM.material.get_dtheta(energy). Values are radians.
BMM_DCM_REFRACTION_CORRECTION: tuple[tuple[float, float], ...] = (
    (4500.0, 6.2087424660813571e-05),
    (5000.0, 5.4590825791400793e-05),
    (5500.0, 4.8771736568662980e-05),
    (6000.0, 4.4113486912216536e-05),
    (6500.0, 4.0292573454486581e-05),
    (7000.0, 3.7098728818993075e-05),
    (7112.0, 3.6453770798447311e-05),
    (7500.0, 3.4386838059121665e-05),
    (8000.0, 3.2053872597668038e-05),
    (8500.0, 3.0024076328705684e-05),
    (9000.0, 2.8240969686021513e-05),
    (9500.0, 2.6658325042177745e-05),
    (10000.0, 2.5245646449980029e-05),
    (10500.0, 2.3976765827902510e-05),
    (11000.0, 2.2830647791167065e-05),
    (11500.0, 2.1790880020932953e-05),
    (12000.0, 2.0842442095492882e-05),
    (12500.0, 1.9973977040901038e-05),
    (13000.0, 1.9176078875263845e-05),
    (13500.0, 1.8439696875173806e-05),
    (14000.0, 1.7758765963334434e-05),
    (14500.0, 1.7126388172950063e-05),
    (15000.0, 1.6538227869862143e-05),
    (15500.0, 1.5989175530627924e-05),
    (16000.0, 1.5476014088627467e-05),
    (16500.0, 1.4994708363544268e-05),
    (17000.0, 1.4543007956821211e-05),
    (17500.0, 1.4117632929602250e-05),
    (18000.0, 1.3716844203275052e-05),
    (18500.0, 1.3338228120114604e-05),
    (19000.0, 1.2980049746120307e-05),
    (19500.0, 1.2640820777504789e-05),
    (20000.0, 1.2318810178065759e-05),
    (20500.0, 1.2013062683636537e-05),
    (21000.0, 1.1722167099464218e-05),
    (21500.0, 1.1444978452618921e-05),
    (22000.0, 1.1180901129944218e-05),
    (22500.0, 1.0928700419381407e-05),
    (23000.0, 1.0687598381840295e-05),
    (23500.0, 1.0457109964388793e-05),
    (24000.0, 1.0236325578799482e-05),
    (24500.0, 1.0024643088085614e-05),
    (25000.0, 9.8217097340962868e-06),
)
BMM_DCM_REFRACTION_CORRECTION_ENERGIES = np.array([point[0] for point in BMM_DCM_REFRACTION_CORRECTION])
BMM_DCM_REFRACTION_CORRECTIONS = np.array([point[1] for point in BMM_DCM_REFRACTION_CORRECTION])


def bragg_angle(energy_ev: float, lattice_spacing: float) -> float:
    """Compute first-order Bragg angle in radians from photon energy and d-spacing."""
    sine_argument = HC_EV_ANGSTROM / (2.0 * lattice_spacing * energy_ev)
    if not 0.0 < sine_argument <= 1.0:
        raise ValueError(f"Energy {energy_ev} eV is outside the Bragg range for d={lattice_spacing} Å")
    return math.asin(sine_argument)


def dcm_refraction_correction(energy_ev: float) -> float:
    """Return the tabulated BMM DCM refraction correction in radians."""
    if energy_ev < BMM_DCM_REFRACTION_CORRECTION_ENERGIES[0]:
        raise ValueError(f"Energy {energy_ev} eV is below the DCM calibration table")
    if energy_ev > BMM_DCM_REFRACTION_CORRECTION_ENERGIES[-1]:
        raise ValueError(f"Energy {energy_ev} eV is above the DCM calibration table")
    return float(np.interp(energy_ev, BMM_DCM_REFRACTION_CORRECTION_ENERGIES, BMM_DCM_REFRACTION_CORRECTIONS))


@plan
def change_energy_stub(
    tpw: XRTWiggler,
    dcm: XRTDCM,
    crystal: XRTCrystalMaterial,
    energy: float,
    band: float = 20.0,
) -> MsgGenerator[float]:
    """Change source energy band and retune the fixed-exit DCM."""
    if not 6000 <= energy <= 8000:
        raise ValueError(
            "Current XRT model represents BMM Mode C, valid for 6-8 keV focused at XAS. This plan assumes only Mode C."
        )

    yield from bps.mv(
        tpw.min_energy,
        energy - band / 2.0,
        tpw.max_energy,
        energy + band / 2.0,
    )

    lattice_spacing = float((yield from bps.rd(crystal.lattice_spacing)))
    bragg_offset = float((yield from bps.rd(dcm.bragg_offset)))
    bragg = bragg_angle(energy, lattice_spacing)
    bragg += dcm_refraction_correction(energy) + BMM_BEAM_INCLINATION + bragg_offset

    yield from bps.mv(
        dcm.bragg,
        bragg,
        dcm.crystal_2_perp_translation,
        BMM_DCM_FIXED_EXIT / (2.0 * math.cos(bragg)),
    )
    return bragg
