import math
from typing import Any

import pytest
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.run_engine import call_in_bluesky_event_loop
from ophyd_async.core import init_devices, set_mock_attr, set_mock_value, soft_signal_rw

from tst_sim_tools.devices.materials import XRTCrystalSi
from tst_sim_tools.devices.mirrors import XRTOpticalElement
from tst_sim_tools.devices.sources import XRTWiggler
from tst_sim_tools.plans import bmm
from tst_sim_tools.plans.bmm import (
    BMM_BEAM_INCLINATION,
    BMM_DCM_FIXED_EXIT,
    BMM_DCM_REFRACTION_CORRECTION,
    HC_EV_ANGSTROM,
    acquire_with_energy_scan,
    bragg_angle,
    change_energy_stub,
    dcm_refraction_correction,
    energy_scan_take_reading,
    scan_energy,
    vendored_default_acquire,
)

MODE_C_ERROR = "Current XRT model represents BMM Mode C, valid for 6-8 keV focused at XAS. This plan assumes only Mode C."


@pytest.fixture
def bmm_devices(run_engine):
    with init_devices(mock=True):
        tpw = XRTWiggler("ca://UNIT:TPW:", name="tpw")
        dcm_c1 = XRTOpticalElement("ca://UNIT:C1:", name="dcm_c1")
        dcm_c2 = XRTOpticalElement("ca://UNIT:C2:", name="dcm_c2")
        crystal = XRTCrystalSi("ca://UNIT:SI:", name="crystal")
        m2 = XRTOpticalElement("ca://UNIT:M2:", name="m2")

    set_mock_value(crystal.lattice_spacing, HC_EV_ANGSTROM / 7000.0)
    set_mock_value(dcm_c1.fixed_pitch, 0.01)
    set_mock_value(dcm_c2.fixed_pitch, -0.02)
    set_mock_value(dcm_c1.center_y, 100.0)
    set_mock_value(dcm_c1.center_z, 200.0)
    return tpw, dcm_c1, dcm_c2, crystal, m2


def run_in_open_run(plan):
    yield from bps.open_run()
    result = yield from plan
    yield from bps.close_run()
    return result


def signal_value(signal):
    return call_in_bluesky_event_loop(signal.get_value())


def expected_dcm_targets(energy: float, lattice_spacing: float) -> tuple[float, float, float, float, float]:
    angle = bragg_angle(energy, lattice_spacing) + dcm_refraction_correction(energy) + BMM_BEAM_INCLINATION
    translation = BMM_DCM_FIXED_EXIT / (2.0 * math.cos(angle))
    return (
        angle,
        angle - 0.01,
        -angle + 0.02,
        100.0 - translation * math.sin(angle),
        200.0 + translation * math.cos(angle),
    )


def test_bragg_angle_known_value_and_inclusive_upper_boundary() -> None:
    assert bragg_angle(7000.0, HC_EV_ANGSTROM / 7000.0) == pytest.approx(math.pi / 6.0)
    assert bragg_angle(7000.0, HC_EV_ANGSTROM / (2.0 * 7000.0)) == pytest.approx(math.pi / 2.0)


@pytest.mark.parametrize(
    ("energy", "spacing"),
    [
        (-7000.0, HC_EV_ANGSTROM / 7000.0),
        (7000.0, HC_EV_ANGSTROM / (4.0 * 7000.0)),
    ],
)
def test_bragg_angle_rejects_invalid_sine_argument(energy: float, spacing: float) -> None:
    with pytest.raises(ValueError) as caught:
        bragg_angle(energy, spacing)

    assert str(caught.value) == f"Energy {energy} eV is outside the Bragg range for d={spacing} Å"


def test_dcm_refraction_correction_interpolates_table() -> None:
    first_energy, first_correction = BMM_DCM_REFRACTION_CORRECTION[0]
    next_energy, next_correction = BMM_DCM_REFRACTION_CORRECTION[1]
    last_energy, last_correction = BMM_DCM_REFRACTION_CORRECTION[-1]

    assert dcm_refraction_correction(first_energy) == first_correction
    assert dcm_refraction_correction(last_energy) == last_correction
    assert dcm_refraction_correction((first_energy + next_energy) / 2.0) == pytest.approx(
        (first_correction + next_correction) / 2.0
    )


@pytest.mark.parametrize(
    ("energy", "message"),
    [
        (4499.0, "Energy 4499.0 eV is below the DCM calibration table"),
        (25001.0, "Energy 25001.0 eV is above the DCM calibration table"),
    ],
)
def test_dcm_refraction_correction_rejects_out_of_table_energy(energy: float, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        dcm_refraction_correction(energy)

    assert str(caught.value) == message


def test_change_energy_moves_source_and_fixed_exit_dcm(run_engine, bmm_devices) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    lattice_spacing = HC_EV_ANGSTROM / 7000.0
    expected_angle, c1_pitch, c2_pitch, c2_y, c2_z = expected_dcm_targets(7000.0, lattice_spacing)

    result = run_engine(change_energy_stub(tpw, dcm_c1, dcm_c2, crystal, 7000.0, band=2.0))

    assert result.plan_result == pytest.approx(expected_angle)
    assert signal_value(tpw.min_energy) == 6999.0
    assert signal_value(tpw.max_energy) == 7001.0
    assert signal_value(dcm_c1.pitch) == pytest.approx(c1_pitch)
    assert signal_value(dcm_c2.pitch) == pytest.approx(c2_pitch)
    assert signal_value(dcm_c2.center_y) == pytest.approx(c2_y)
    assert signal_value(dcm_c2.center_z) == pytest.approx(c2_z)


@pytest.mark.parametrize("energy", [6000.0, 8000.0])
def test_change_energy_accepts_mode_c_boundaries(run_engine, bmm_devices, energy: float) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices

    run_engine(change_energy_stub(tpw, dcm_c1, dcm_c2, crystal, energy))

    assert signal_value(tpw.min_energy) == energy - 0.5
    assert signal_value(tpw.max_energy) == energy + 0.5


@pytest.mark.parametrize("energy", [5999.0, 8001.0])
def test_change_energy_rejects_out_of_range_without_writes(run_engine, bmm_devices, energy: float) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    run_engine(
        bps.mv(
            tpw.min_energy,
            1.0,
            tpw.max_energy,
            2.0,
            dcm_c1.pitch,
            3.0,
            dcm_c2.pitch,
            4.0,
            dcm_c2.center_y,
            5.0,
            dcm_c2.center_z,
            6.0,
        )
    )

    with pytest.raises(ValueError, match=MODE_C_ERROR):
        run_engine(change_energy_stub(tpw, dcm_c1, dcm_c2, crystal, energy))

    assert [
        signal_value(tpw.min_energy),
        signal_value(tpw.max_energy),
        signal_value(dcm_c1.pitch),
        signal_value(dcm_c2.pitch),
        signal_value(dcm_c2.center_y),
        signal_value(dcm_c2.center_z),
    ] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_change_energy_invalid_lattice_fails_after_source_move(run_engine, bmm_devices) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    set_mock_value(crystal.lattice_spacing, 0.1)
    run_engine(bps.mv(dcm_c1.pitch, 3.0, dcm_c2.pitch, 4.0, dcm_c2.center_y, 5.0, dcm_c2.center_z, 6.0))

    with pytest.raises(ValueError, match="outside the Bragg range"):
        run_engine(change_energy_stub(tpw, dcm_c1, dcm_c2, crystal, 7000.0, band=2.0))

    assert signal_value(tpw.min_energy) == 6999.0
    assert signal_value(tpw.max_energy) == 7001.0
    assert [
        signal_value(dcm_c1.pitch),
        signal_value(dcm_c2.pitch),
        signal_value(dcm_c2.center_y),
        signal_value(dcm_c2.center_z),
    ] == [3.0, 4.0, 5.0, 6.0]


def test_energy_scan_take_reading_rejects_empty_energies_before_detector_activity(
    run_engine,
    bmm_devices,
    mocker,
) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    original_read = tpw.max_energy.read
    read = set_mock_attr(tpw.max_energy, "read", mocker.AsyncMock(side_effect=original_read))
    plan = bpp.run_wrapper(
        energy_scan_take_reading(
            [tpw.max_energy],
            tpw=tpw,
            dcm_c1=dcm_c1,
            dcm_c2=dcm_c2,
            crystal=crystal,
            energies=[],
            band=1.0,
        )
    )

    with pytest.raises(ValueError, match="Expected at least one energy"):
        run_engine(plan)

    read.assert_not_called()


def test_energy_scan_take_reading_emits_ordered_events_and_returns_final_reading(
    run_engine,
    documents,
    bmm_devices,
) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    collected, collect = documents
    run_engine.subscribe(collect)
    plan = run_in_open_run(
        energy_scan_take_reading(
            [tpw.max_energy],
            tpw=tpw,
            dcm_c1=dcm_c1,
            dcm_c2=dcm_c2,
            crystal=crystal,
            energies=[7000.0, 7100.0],
            band=1.0,
        )
    )

    result = run_engine(plan)

    assert [event["data"]["tpw-max_energy"] for event in collected["event"]] == [7000.5, 7100.5]
    assert set(result.plan_result) == {"tpw-max_energy"}
    assert result.plan_result["tpw-max_energy"]["value"] == 7100.5


def test_vendored_default_acquire_moves_suggestions_and_preserves_metadata(
    run_engine,
    documents,
) -> None:
    actuator = soft_signal_rw(float, initial_value=0.0, name="actuator")
    sensor = soft_signal_rw(float, initial_value=5.0, name="sensor")
    ignored_sensor: Any = object()
    suggestions = [{"_id": "trial-0", "actuator": 1.0}, {"_id": "trial-1", "actuator": 2.0}]
    metadata = {"purpose": "unit", "run_key": "caller"}
    collected, collect = documents
    run_engine.subscribe(collect)

    result = run_engine(
        vendored_default_acquire(
            suggestions,
            [actuator],
            [sensor, ignored_sensor],
            md=metadata,
        )
    )

    assert result.plan_result == collected["start"][0]["uid"]
    assert signal_value(actuator) == 2.0
    assert collected["start"][0]["blop_suggestions"] == suggestions
    assert collected["start"][0]["purpose"] == "unit"
    assert collected["start"][0]["run_key"] == "default_acquire"
    assert [event["data"]["actuator"] for event in collected["event"]] == [1.0, 2.0]
    assert [event["data"]["sensor"] for event in collected["event"]] == [5.0, 5.0]
    assert all(set(event["data"]) == {"actuator", "sensor"} for event in collected["event"])


def test_vendored_default_acquire_accepts_no_sensors(run_engine, documents) -> None:
    actuator = soft_signal_rw(float, initial_value=0.0, name="actuator")
    collected, collect = documents
    run_engine.subscribe(collect)

    run_engine(vendored_default_acquire([{"_id": "trial-0", "actuator": 1.0}], [actuator], sensors=None))

    assert set(collected["event"][0]["data"]) == {"actuator"}


def test_vendored_default_acquire_missing_actuator_key_fails_before_run_start(run_engine, documents) -> None:
    actuator = soft_signal_rw(float, initial_value=0.0, name="actuator")
    collected, collect = documents
    run_engine.subscribe(collect)

    with pytest.raises(KeyError) as caught:
        run_engine(vendored_default_acquire([{"wrong": 1.0}], [actuator]))

    assert caught.value.args == ("actuator",)
    assert collected["start"] == []


def test_scan_energy_emits_single_run_and_leaves_final_state(run_engine, documents, bmm_devices) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    detector = tpw.max_energy
    collected, collect = documents
    run_engine.subscribe(collect)

    result = run_engine(scan_energy([detector], tpw, dcm_c1, dcm_c2, crystal, [7000.0, 7100.0]))

    assert result.plan_result == collected["start"][0]["uid"]
    assert len(collected["start"]) == 1
    assert len(collected["descriptor"]) == 1
    assert len(collected["stop"]) == 1
    assert collected["stop"][0]["exit_status"] == "success"
    assert [event["data"]["tpw-max_energy"] for event in collected["event"]] == [7000.5, 7100.5]
    _, c1_pitch, c2_pitch, c2_y, c2_z = expected_dcm_targets(7100.0, HC_EV_ANGSTROM / 7000.0)
    assert signal_value(tpw.max_energy) == 7100.5
    assert signal_value(dcm_c1.pitch) == pytest.approx(c1_pitch)
    assert signal_value(dcm_c2.pitch) == pytest.approx(c2_pitch)
    assert signal_value(dcm_c2.center_y) == pytest.approx(c2_y)
    assert signal_value(dcm_c2.center_z) == pytest.approx(c2_z)
    assert run_engine.state == "idle"
    assert getattr(detector, "_cache", None) is None


def test_scan_energy_empty_sequence_is_successful_empty_run(run_engine, documents, bmm_devices) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    detector = tpw.max_energy
    collected, collect = documents
    run_engine.subscribe(collect)

    result = run_engine(scan_energy([detector], tpw, dcm_c1, dcm_c2, crystal, []))

    assert result.plan_result == collected["start"][0]["uid"]
    assert len(collected["event"]) == 0
    assert collected["stop"][0]["exit_status"] == "success"
    assert run_engine.state == "idle"
    assert getattr(detector, "_cache", None) is None


def test_scan_energy_failure_retains_event_and_unstages(run_engine, documents, bmm_devices) -> None:
    tpw, dcm_c1, dcm_c2, crystal, _ = bmm_devices
    detector = tpw.max_energy
    collected, collect = documents
    run_engine.subscribe(collect)

    with pytest.raises(ValueError, match=MODE_C_ERROR):
        run_engine(scan_energy([detector], tpw, dcm_c1, dcm_c2, crystal, [7000.0, 9000.0]))

    assert [event["data"]["tpw-max_energy"] for event in collected["event"]] == [7000.5]
    assert collected["stop"][0]["exit_status"] == "fail"
    assert MODE_C_ERROR in collected["stop"][0]["reason"]
    assert run_engine.state == "idle"
    assert getattr(detector, "_cache", None) is None


def test_acquire_with_energy_scan_orders_suggestions_then_energies(
    run_engine,
    documents,
    bmm_devices,
    mocker,
) -> None:
    tpw, dcm_c1, dcm_c2, crystal, m2 = bmm_devices
    suggestions = [
        {"_id": "trial-0", "dcm_c2-roll": 0.1, "m2-yaw": 0.2, "m2-center_x": 0.3},
        {"_id": "trial-1", "dcm_c2-roll": 1.1, "m2-yaw": 1.2, "m2-center_x": 1.3},
    ]
    energies = [7000.0, 7100.0]
    take_reading = mocker.spy(bmm, "energy_scan_take_reading")
    collected, collect = documents
    run_engine.subscribe(collect)

    result = run_engine(
        acquire_with_energy_scan(
            suggestions,
            [dcm_c2.roll, m2.yaw, m2.center_x],
            [tpw.max_energy],
            md={"blop_correlation_uid": "corr-1"},
            tpw=tpw,
            dcm_c1=dcm_c1,
            dcm_c2=dcm_c2,
            crystal=crystal,
            energies=energies,
        )
    )

    assert result.plan_result == collected["start"][0]["uid"]
    assert len(collected["event"]) == 4
    assert [event["data"]["tpw-max_energy"] for event in collected["event"]] == [7000.5, 7100.5, 7000.5, 7100.5]
    assert [event["data"]["dcm_c2-roll"] for event in collected["event"]] == [0.1, 0.1, 1.1, 1.1]
    assert [event["data"]["m2-yaw"] for event in collected["event"]] == [0.2, 0.2, 1.2, 1.2]
    assert [event["data"]["m2-center_x"] for event in collected["event"]] == [0.3, 0.3, 1.3, 1.3]
    assert collected["start"][0]["blop_correlation_uid"] == "corr-1"
    assert all(call.kwargs["energies"] == (7000.0, 7100.0) for call in take_reading.call_args_list)
    assert signal_value(tpw.max_energy) == 7100.5
    assert signal_value(dcm_c2.roll) == 1.1
    assert signal_value(m2.yaw) == 1.2
    assert signal_value(m2.center_x) == 1.3


def test_acquire_with_energy_scan_empty_energies_fails_and_unstages(run_engine, bmm_devices) -> None:
    tpw, dcm_c1, dcm_c2, crystal, m2 = bmm_devices
    sensor = soft_signal_rw(float, initial_value=5.0, name="sensor")
    suggestions = [{"_id": "trial-0", "dcm_c2-roll": 0.1, "m2-yaw": 0.2, "m2-center_x": 0.3}]

    with pytest.raises(ValueError, match="Expected at least one energy"):
        run_engine(
            acquire_with_energy_scan(
                suggestions,
                [dcm_c2.roll, m2.yaw, m2.center_x],
                [sensor],
                tpw=tpw,
                dcm_c1=dcm_c1,
                dcm_c2=dcm_c2,
                crystal=crystal,
                energies=[],
            )
        )

    assert run_engine.state == "idle"
    assert getattr(sensor, "_cache", None) is None
