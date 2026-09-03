from functools import partial
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from bluesky_queueserver_api.http import REManagerAPI
from ophyd_async.core import init_devices, soft_signal_rw

from tst_sim_tools.agents.energy_alignment import (
    ALIGNMENT_SCORE,
    BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS,
    BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
    BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS,
    BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS,
    BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS,
    CENTROID_RMS_ERROR,
    CENTROID_SPAN,
    FWHM,
    LATERAL_POSITION_ERROR,
    MAX_CENTROID_ERROR,
    MAX_FWHM,
    MIN_INTENSITY,
    VERTICAL_POSITION_ERROR,
    EnergyAlignmentEvalutation,
    build_qs_agent,
    build_re_agent,
)
from tst_sim_tools.devices.materials import XRTCrystalSi
from tst_sim_tools.devices.mirrors import XRTOpticalElement, XRTToroidMirror
from tst_sim_tools.devices.sources import XRTWiggler
from tst_sim_tools.plans.bmm import acquire_with_energy_scan

OUTCOME_KEYS = (
    ALIGNMENT_SCORE,
    CENTROID_RMS_ERROR,
    MAX_CENTROID_ERROR,
    CENTROID_SPAN,
    LATERAL_POSITION_ERROR,
    VERTICAL_POSITION_ERROR,
    FWHM,
    MAX_FWHM,
    MIN_INTENSITY,
    "_id",
)


class FakeField:
    def __init__(self, images: np.ndarray) -> None:
        self.images = images

    def read(self) -> np.ndarray:
        return self.images


class FakeRun:
    def __init__(self, images: np.ndarray, suggestion_ids: list[str]) -> None:
        self.metadata = {
            "start": {
                "blop_suggestions": [{"_id": suggestion_id} for suggestion_id in suggestion_ids],
            }
        }
        self.primary = {"screen_image": FakeField(images)}

    def __getitem__(self, key: str):
        if key != "primary":
            raise KeyError(key)
        return self.primary


class EventuallyConsistentClient:
    def __init__(self, run: FakeRun) -> None:
        self.run = run
        self.calls = 0

    def __getitem__(self, uid: str) -> FakeRun:
        self.calls += 1
        if self.calls == 1:
            raise KeyError(uid)
        return self.run


class FailingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __getitem__(self, uid: str):
        raise self.error


def alignment_images() -> np.ndarray:
    images = np.zeros((4, 1, 5, 5), dtype=float)
    images[0, 0, 2, 2] = 20_000.0
    images[1, 0, 2, 3] = 20_000.0
    images[2, 0, 1, 2] = 18_000.0
    images[3, 0, 3, 2] = 18_000.0
    return images


def make_evaluator(client, *, image_key: str = "screen_image") -> EnergyAlignmentEvalutation:
    return EnergyAlignmentEvalutation(
        client,
        image_key=image_key,
        target_centroid=(2.0, 2.0),
        energies=(7000.0, 7100.0),
        threshold=0.0,
        blur=0.0,
    )


def test_energy_alignment_evaluation_rejects_empty_energies() -> None:
    with pytest.raises(ValueError, match="Expected at least one energy for energy alignment evaluation"):
        EnergyAlignmentEvalutation({}, "screen_image", (2.0, 2.0), [])


def test_energy_alignment_evaluation_reports_each_suggestion() -> None:
    run = FakeRun(alignment_images(), ["trial-0", "trial-1"])
    evaluator = make_evaluator({"run-uid": run})

    outcomes = evaluator("run-uid", suggestions=[])

    assert [tuple(outcome) for outcome in outcomes] == [OUTCOME_KEYS, OUTCOME_KEYS]
    assert outcomes[0]["_id"] == "trial-0"
    assert outcomes[0][CENTROID_RMS_ERROR] == pytest.approx(np.sqrt(0.5))
    assert outcomes[0][MAX_CENTROID_ERROR] == 1.0
    assert outcomes[0][CENTROID_SPAN] == 1.0
    assert outcomes[0][LATERAL_POSITION_ERROR] == pytest.approx(np.sqrt(0.5))
    assert outcomes[0][VERTICAL_POSITION_ERROR] == 0.0
    assert outcomes[0][FWHM] == 1.0
    assert outcomes[0][MAX_FWHM] == 1.0
    assert outcomes[0][MIN_INTENSITY] == 20_000.0

    assert outcomes[1]["_id"] == "trial-1"
    assert outcomes[1][CENTROID_RMS_ERROR] == 1.0
    assert outcomes[1][MAX_CENTROID_ERROR] == 1.0
    assert outcomes[1][CENTROID_SPAN] == 2.0
    assert outcomes[1][LATERAL_POSITION_ERROR] == 0.0
    assert outcomes[1][VERTICAL_POSITION_ERROR] == 1.0
    assert outcomes[1][FWHM] == 1.0
    assert outcomes[1][MAX_FWHM] == 1.0
    assert outcomes[1][MIN_INTENSITY] == 18_000.0


def test_energy_alignment_evaluation_applies_weighted_score() -> None:
    evaluator = make_evaluator({"run-uid": FakeRun(alignment_images(), ["trial-0", "trial-1"])})

    outcomes = evaluator("run-uid", suggestions=[])

    assert outcomes[0][ALIGNMENT_SCORE] == pytest.approx(np.sqrt(0.5) + 0.25 + 0.1 + 0.005 + 0.0025)
    assert outcomes[1][ALIGNMENT_SCORE] == pytest.approx(1.0 + 0.25 + 0.2 + 0.005 + 0.0025)


def test_energy_alignment_evaluation_retries_key_error_once(mocker) -> None:
    client = EventuallyConsistentClient(FakeRun(alignment_images()[:2], ["trial-0"]))
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")

    outcomes = make_evaluator(client)("run-uid", suggestions=[])

    sleep.assert_called_once_with(0.1)
    assert client.calls == 2
    assert outcomes[0]["_id"] == "trial-0"
    assert tuple(outcomes[0]) == OUTCOME_KEYS


@pytest.mark.parametrize(
    ("run", "image_key", "missing_key"),
    [
        ({}, "screen_image", "primary"),
        ({"primary": {}}, "missing_image", "missing_image"),
    ],
)
def test_energy_alignment_evaluation_does_not_retry_missing_run_data(
    run: dict,
    image_key: str,
    missing_key: str,
    mocker,
) -> None:
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")
    evaluator = make_evaluator({"run-uid": run}, image_key=image_key)

    with pytest.raises(KeyError) as caught:
        evaluator("run-uid", suggestions=[])

    assert caught.value.args == (missing_key,)
    sleep.assert_not_called()


def test_energy_alignment_evaluation_times_out_waiting_for_run(mocker) -> None:
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")
    monotonic = mocker.patch("tst_sim_tools.agents.energy_alignment.time.monotonic", side_effect=[0.0, 5.0, 10.0])
    evaluator = make_evaluator({})

    with pytest.raises(TimeoutError) as caught:
        evaluator("run-uid", suggestions=[])

    assert str(caught.value) == "Run 'run-uid' did not appear in Tiled within 10 seconds"
    assert isinstance(caught.value.__cause__, KeyError)
    sleep.assert_called_once_with(0.1)
    assert monotonic.call_count == 3


def test_energy_alignment_evaluation_propagates_non_key_error(mocker) -> None:
    error = RuntimeError("catalog unavailable")
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")
    evaluator = make_evaluator(FailingClient(error))

    with pytest.raises(RuntimeError) as caught:
        evaluator("run-uid", suggestions=[])

    assert caught.value is error
    sleep.assert_not_called()


def test_energy_alignment_evaluation_validates_image_count() -> None:
    images = alignment_images()[:3]
    evaluator = make_evaluator({"run-uid": FakeRun(images, ["trial-0", "trial-1"])})

    with pytest.raises(ValueError) as caught:
        evaluator("run-uid", suggestions=[])

    assert str(caught.value) == "Expected 4 image(s) from Tiled for 2 suggestion(s) and 2 energy point(s), but got 3"


def fake_agent(mocker):
    agent = SimpleNamespace(ax_client=mocker.Mock())
    return agent


def test_build_re_agent_wires_dofs_objective_evaluator_and_acquisition(run_engine, mocker) -> None:
    sensor = soft_signal_rw(float, initial_value=0.0, name="screen")
    with init_devices(mock=True):
        tpw = XRTWiggler("ca://UNIT:TPW:", name="tpw")
        dcm_c1 = XRTOpticalElement("ca://UNIT:C1:", name="dcm_c1")
        dcm_c2 = XRTOpticalElement("ca://UNIT:C2:", name="dcm_c2")
        crystal = XRTCrystalSi("ca://UNIT:SI:", name="crystal")
        m2 = XRTToroidMirror("ca://UNIT:M2:", name="m2")
    dcm_roll = dcm_c2.roll
    m2_yaw = m2.yaw
    m2_center_x = m2.center_x
    tiled_client = object()
    agent = fake_agent(mocker)
    agent_type = mocker.patch("tst_sim_tools.agents.energy_alignment.Agent", return_value=agent)

    result = build_re_agent(
        [sensor],
        tpw,
        dcm_c1,
        dcm_c2,
        crystal,
        m2,
        tiled_client,
        "screen_image",
        (2.0, 2.0),
        [7000.0, 7100.0],
        band=2.0,
        threshold=0.1,
        blur=0.5,
        checkpoint_path="/tmp/re-agent.json",
    )

    assert result is agent
    kwargs = agent_type.call_args.kwargs
    assert kwargs["sensors"] == [sensor]
    assert [dof.actuator for dof in kwargs["dofs"]] == [dcm_roll, m2_yaw, m2_center_x]
    assert [dof.bounds for dof in kwargs["dofs"]] == [
        BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS,
        BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS,
        BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS,
    ]
    assert [dof.parameter_type for dof in kwargs["dofs"]] == ["float", "float", "float"]
    assert len(kwargs["objectives"]) == 1
    assert kwargs["objectives"][0].name == ALIGNMENT_SCORE
    assert kwargs["objectives"][0].minimize is True
    assert kwargs["outcome_constraints"] is BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS
    evaluator = kwargs["evaluation_function"]
    assert isinstance(evaluator, EnergyAlignmentEvalutation)
    assert evaluator._client is tiled_client
    assert evaluator._image_key == "screen_image"
    assert evaluator._target_centroid == (2.0, 2.0)
    np.testing.assert_array_equal(evaluator._energies, [7000.0, 7100.0])
    assert evaluator._threshold == 0.1
    assert evaluator._blur == 0.5
    acquisition_plan = kwargs["acquisition_plan"]
    assert isinstance(acquisition_plan, partial)
    assert acquisition_plan.func is acquire_with_energy_scan
    assert acquisition_plan.keywords == {
        "tpw": tpw,
        "dcm_c1": dcm_c1,
        "dcm_c2": dcm_c2,
        "crystal": crystal,
        "energies": (7000.0, 7100.0),
        "band": 2.0,
    }
    assert kwargs["checkpoint_path"] == "/tmp/re-agent.json"
    agent.ax_client.configure_generation_strategy.assert_called_once_with(
        method="fast",
        initialization_budget=BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
        initialize_with_center=False,
    )


def test_build_qs_agent_wires_remote_dispatcher_and_queue_plan(mocker) -> None:
    re_manager = cast(REManagerAPI, object())
    tiled_client = object()
    dispatcher = object()
    remote_dispatcher = mocker.patch("tst_sim_tools.agents.energy_alignment.RemoteDispatcher", return_value=dispatcher)
    agent = fake_agent(mocker)
    agent_type = mocker.patch("tst_sim_tools.agents.energy_alignment.QueueserverAgent", return_value=agent)

    result = build_qs_agent(
        re_manager,
        ["screen"],
        "si111",
        tiled_client,
        "screen_image",
        (2.0, 2.0),
        [7000.0, 7100.0],
        band=2.0,
        threshold=0.1,
        blur=0.5,
        checkpoint_path="/tmp/qs-agent.json",
    )

    assert result is agent
    remote_dispatcher.assert_called_once_with("127.0.0.1:45105")
    args = agent_type.call_args.args
    kwargs = agent_type.call_args.kwargs
    assert args == (re_manager, dispatcher)
    assert kwargs["sensors"] == ["screen"]
    assert [(dof.actuator, dof.name, dof.bounds) for dof in kwargs["dofs"]] == [
        ("dcm_c2.roll", "dcm_c2-roll", BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS),
        ("m2.yaw", "m2-yaw", BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS),
        ("m2.center_x", "m2-center_x", BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS),
    ]
    assert all(dof.parameter_type == "float" for dof in kwargs["dofs"])
    assert len(kwargs["objectives"]) == 1
    assert kwargs["objectives"][0].name == ALIGNMENT_SCORE
    assert kwargs["objectives"][0].minimize is True
    assert kwargs["acquisition_plan"] == "acquire_with_energy_scan"
    assert kwargs["acquisition_plan_kwargs"] == {
        "tpw": "tpw",
        "dcm_c1": "dcm_c1",
        "dcm_c2": "dcm_c2",
        "crystal": "si111",
        "energies": (7000.0, 7100.0),
        "band": 2.0,
    }
    evaluator = kwargs["evaluation_function"]
    assert isinstance(evaluator, EnergyAlignmentEvalutation)
    assert evaluator._client is tiled_client
    assert evaluator._image_key == "screen_image"
    assert evaluator._target_centroid == (2.0, 2.0)
    np.testing.assert_array_equal(evaluator._energies, [7000.0, 7100.0])
    assert evaluator._threshold == 0.1
    assert evaluator._blur == 0.5
    assert kwargs["outcome_constraints"] is BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS
    assert kwargs["checkpoint_path"] == "/tmp/qs-agent.json"
    agent.ax_client.configure_generation_strategy.assert_called_once_with(
        method="fast",
        initialization_budget=BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
        initialize_with_center=False,
    )
