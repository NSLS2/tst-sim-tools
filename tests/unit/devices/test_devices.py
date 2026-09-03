import pytest
from ophyd_async.core import init_devices, set_mock_value

from tst_sim_tools.devices.materials import XRTCrystalSi
from tst_sim_tools.devices.mirrors import XRTOpticalElement, XRTParabolicalMirror, XRTToroidMirror
from tst_sim_tools.devices.slits import XRTRectangularAperature
from tst_sim_tools.devices.sources import XRTWiggler

pytestmark = pytest.mark.asyncio


OPTICAL_SUFFIXES = {
    "fixed_pitch": "pitch",
    "fixed_roll": "roll",
    "fixed_yaw": "yaw",
    "pitch": "extraPitch",
    "roll": "extraRoll",
    "yaw": "extraYaw",
    "center_x": "center:x",
    "center_y": "center:y",
    "center_z": "center:z",
    "material": "material",
}

APERTURE_SUFFIXES = {
    "center_x": "center:x",
    "center_y": "center:y",
    "center_z": "center:z",
    "blades_left": "blades:left",
    "blades_right": "blades:right",
    "blades_bottom": "blades:bottom",
    "blades_top": "blades:top",
}

WIGGLER_SUFFIXES = [
    ("center_x", "center:x"),
    ("center_y", "center:y"),
    ("center_z", "center:z"),
    ("nrays", "nrays"),
    ("filament_beam", "filamentBeam"),
    ("uniform_ray_density", "uniformRayDensity"),
    ("electron_energy", "eE"),
    ("electron_current", "eI"),
    ("electron_energy_spread", "eEspread"),
    ("electron_beam_size_x", "eSigmaX"),
    ("electron_beam_size_z", "eSigmaZ"),
    ("electron_emittance_x", "eEpsilonX"),
    ("electron_emittance_z", "eEpsilonZ"),
    ("beta_x", "betaX"),
    ("beta_z", "betaZ"),
    ("x_prime_max", "xPrimeMax"),
    ("z_prime_max", "zPrimeMax"),
    ("energy_distribution", "distE"),
    ("min_energy", "eMin"),
    ("max_energy", "eMax"),
    ("near_field_distance", "R0"),
    ("deflection_parameter", "K"),
    ("period", "period"),
    ("num_periods", "n"),
]


async def test_crystal_lattice_spacing_is_read_only() -> None:
    async with init_devices(mock=True):
        crystal = XRTCrystalSi("ca://UNIT:", name="crystal")

    set_mock_value(crystal.lattice_spacing, 3.1356)

    assert crystal.lattice_spacing.source == "mock+ca://UNIT:d"
    assert await crystal.lattice_spacing.get_value() == 3.1356
    assert not hasattr(crystal.lattice_spacing, "set")


@pytest.mark.parametrize(("attribute", "suffix"), OPTICAL_SUFFIXES.items())
async def test_optical_element_signal_sources(attribute: str, suffix: str) -> None:
    async with init_devices(mock=True):
        optic = XRTOpticalElement("ca://UNIT:", name="optic")

    assert getattr(optic, attribute).source == f"mock+ca://UNIT:{suffix}"


@pytest.mark.parametrize("device_type", [XRTOpticalElement, XRTParabolicalMirror])
async def test_optical_element_read_and_configuration_contract(device_type) -> None:
    async with init_devices(mock=True):
        optic = device_type("ca://UNIT:", name="optic")

    set_mock_value(optic.fixed_pitch, 0.1)
    set_mock_value(optic.fixed_roll, 0.2)
    set_mock_value(optic.fixed_yaw, 0.3)
    set_mock_value(optic.material, "Si")
    await optic.pitch.set(1.1)
    await optic.roll.set(1.2)
    await optic.yaw.set(1.3)
    await optic.center_x.set(2.1)
    await optic.center_y.set(2.2)
    await optic.center_z.set(2.3)

    reading = await optic.read()
    configuration = await optic.read_configuration()

    assert {name: value["value"] for name, value in reading.items()} == {
        "optic-pitch": 1.1,
        "optic-roll": 1.2,
        "optic-yaw": 1.3,
    }
    assert optic.hints == {"fields": ["optic-pitch", "optic-roll", "optic-yaw"]}
    assert {name: value["value"] for name, value in configuration.items()} == {
        "optic-fixed_pitch": 0.1,
        "optic-fixed_roll": 0.2,
        "optic-fixed_yaw": 0.3,
        "optic-center_x": 2.1,
        "optic-center_y": 2.2,
        "optic-center_z": 2.3,
        "optic-material": "Si",
    }


async def test_toroid_meridional_radius_is_writable() -> None:
    async with init_devices(mock=True):
        toroid = XRTToroidMirror("ca://UNIT:", name="toroid")

    await toroid.meridional_radius.set(4.5)

    assert toroid.meridional_radius.source == "mock+ca://UNIT:R"
    assert await toroid.meridional_radius.get_value() == 4.5


async def test_rectangular_aperture_signal_sources_and_writes() -> None:
    async with init_devices(mock=True):
        aperture = XRTRectangularAperature("ca://UNIT:", name="aperture")

    for attribute, suffix in APERTURE_SUFFIXES.items():
        assert getattr(aperture, attribute).source == f"mock+ca://UNIT:{suffix}"

    await aperture.center_x.set(1.25)
    await aperture.blades_left.set(-2.5)

    assert await aperture.center_x.get_value() == 1.25
    assert await aperture.blades_left.get_value() == -2.5


@pytest.mark.parametrize(("attribute", "suffix"), WIGGLER_SUFFIXES)
async def test_wiggler_signal_sources(attribute: str, suffix: str) -> None:
    async with init_devices(mock=True):
        wiggler = XRTWiggler("ca://UNIT:", name="wiggler")

    assert getattr(wiggler, attribute).source == f"mock+ca://UNIT:{suffix}"


async def test_wiggler_values_are_configuration_only() -> None:
    async with init_devices(mock=True):
        wiggler = XRTWiggler("ca://UNIT:", name="wiggler")

    await wiggler.center_x.set(1.25)
    await wiggler.nrays.set(1000)
    await wiggler.filament_beam.set(True)
    await wiggler.energy_distribution.set("lines")

    assert await wiggler.center_x.get_value() == 1.25
    assert await wiggler.nrays.get_value() == 1000
    assert await wiggler.filament_beam.get_value() is True
    assert await wiggler.energy_distribution.get_value() == "lines"
    assert await wiggler.read() == {}
    assert set(await wiggler.read_configuration()) == {f"wiggler-{attribute}" for attribute, _ in WIGGLER_SUFFIXES}
