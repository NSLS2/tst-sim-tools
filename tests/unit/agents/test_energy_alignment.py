import queue
import threading
from collections.abc import Mapping
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

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
    TiledWebsocketEnergyAlignmentEvaluation,
    build_qs_agent,
    build_re_agent,
)
from tst_sim_tools.agents.tiled_queueserver import CORRELATION_UID_KEY, TiledQueueAcquisition
from tst_sim_tools.analysis.image import analyze_image
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
        self.metadata: dict[str, Any] = {
            "start": {
                "blop_suggestions": [{"_id": suggestion_id} for suggestion_id in suggestion_ids],
            }
        }
        self.primary: Any = {"screen_image": FakeField(images)}

    def __getitem__(self, key: str):
        if key != "primary":
            raise KeyError(key)
        return self.primary


class EventuallyConsistentPrimaryRun(FakeRun):
    def __init__(self, images: np.ndarray, suggestion_ids: list[str]) -> None:
        super().__init__(images, suggestion_ids)
        self.calls = 0

    def __getitem__(self, key: str):
        self.calls += 1
        if self.calls == 1:
            raise KeyError(key)
        return super().__getitem__(key)


class EventuallyConsistentStream:
    def __init__(self, field: FakeField) -> None:
        self.field = field
        self.calls = 0

    def __getitem__(self, key: str) -> FakeField:
        self.calls += 1
        if self.calls == 1 or key != "screen_image":
            raise KeyError(key)
        return self.field


class EventuallyConsistentField(FakeField):
    def __init__(self, images: np.ndarray) -> None:
        super().__init__(images)
        self.calls = 0

    def read(self) -> np.ndarray:
        self.calls += 1
        if self.calls == 1:
            raise KeyError("external asset")
        return super().read()


class MissingField:
    def read(self) -> np.ndarray:
        raise KeyError("external asset")


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


class FakeStreamUpdate:
    def __init__(
        self,
        run_uid: str,
        images: np.ndarray,
        start_document: Mapping[str, Any],
        sequence: int = 1,
    ) -> None:
        self.run_uid = run_uid
        self.images = images
        self.start_document = start_document
        self.sequence = sequence

    def data(self) -> np.ndarray:
        return self.images


class FakeStreamSubscription:
    def __init__(self) -> None:
        self.disconnected = False

    def disconnect(self) -> None:
        self.disconnected = True


class SearchableClient:
    def __init__(self, runs: dict[str, FakeRun]) -> None:
        self.runs = runs

    def __getitem__(self, uid: str) -> FakeRun:
        return self.runs[uid]


def make_tiled_evaluator(client, *, re_manager=None, queue_poll_period: float = 0.0):
    return TiledWebsocketEnergyAlignmentEvaluation(
        client,
        image_key="screen_image",
        target_centroid=(2.0, 2.0),
        energies=(7000.0, 7100.0),
        re_manager=re_manager,
        threshold=0.0,
        blur=0.0,
        queue_poll_period=queue_poll_period,
        image_poll_period=0.0,
    )


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


def test_energy_alignment_evaluation_retries_missing_primary_stream(mocker) -> None:
    run = EventuallyConsistentPrimaryRun(alignment_images()[:2], ["trial-0"])
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")

    outcomes = make_evaluator({"run-uid": run})("run-uid", suggestions=[])

    sleep.assert_called_once_with(0.1)
    assert run.calls == 2
    assert outcomes[0]["_id"] == "trial-0"


def test_energy_alignment_evaluation_retries_missing_image_key(mocker) -> None:
    run = FakeRun(alignment_images()[:2], ["trial-0"])
    stream = EventuallyConsistentStream(run.primary["screen_image"])
    run.primary = stream
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")

    outcomes = make_evaluator({"run-uid": run})("run-uid", suggestions=[])

    sleep.assert_called_once_with(0.1)
    assert stream.calls == 2
    assert outcomes[0]["_id"] == "trial-0"


def test_energy_alignment_evaluation_retries_unreadable_image_data(mocker) -> None:
    run = FakeRun(alignment_images()[:2], ["trial-0"])
    field = EventuallyConsistentField(alignment_images()[:2])
    run.primary = {"screen_image": field}
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")

    outcomes = make_evaluator({"run-uid": run})("run-uid", suggestions=[])

    sleep.assert_called_once_with(0.1)
    assert field.calls == 2
    assert outcomes[0]["_id"] == "trial-0"


@pytest.mark.parametrize(
    ("client", "image_key", "waiting_for"),
    [
        ({}, "screen_image", "run 'run-uid' in Tiled"),
        ({"run-uid": {}}, "screen_image", "primary stream for run 'run-uid'"),
        (
            {"run-uid": {"primary": {}}},
            "missing_image",
            "image key 'missing_image' in the primary stream for run 'run-uid'",
        ),
        (
            {"run-uid": {"primary": {"screen_image": MissingField()}}},
            "screen_image",
            "readable data for image key 'screen_image' in run 'run-uid'",
        ),
    ],
    ids=["run", "primary-stream", "image-key", "image-data"],
)
def test_energy_alignment_evaluation_times_out_waiting_for_tiled_data(
    client: dict,
    image_key: str,
    waiting_for: str,
    mocker,
) -> None:
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")
    monotonic = mocker.patch("tst_sim_tools.agents.energy_alignment.time.monotonic", side_effect=[0.0, 5.0, 10.0])
    evaluator = make_evaluator(client, image_key=image_key)

    with pytest.raises(TimeoutError) as caught:
        evaluator("run-uid", suggestions=[])

    assert str(caught.value) == f"Timed out after 10 seconds waiting for {waiting_for}"
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


def test_tiled_evaluation_consumes_retained_stream_update_without_waiting(mocker) -> None:
    run = FakeRun(alignment_images()[:2], ["trial-0"])
    run.metadata["start"][CORRELATION_UID_KEY] = "correlation-uid"
    client = SearchableClient({"run-uid": run})
    subscription = FakeStreamSubscription()

    def subscribe(catalog, stream_name, data_keys, callback, *, start, max_size, metadata_filter):
        _ = catalog, stream_name, data_keys, start, max_size
        assert metadata_filter(run.metadata["start"]) is True
        assert metadata_filter({}) is False
        callback(FakeStreamUpdate("run-uid", alignment_images()[:2], run.metadata["start"]))
        return subscription

    subscribe_to_stream = mocker.patch(
        "tst_sim_tools.agents.energy_alignment.subscribe_to_stream_by_metadata", side_effect=subscribe
    )
    evaluator = make_tiled_evaluator(client)
    wait = mocker.patch.object(
        evaluator._condition,
        "wait",
        side_effect=AssertionError("must not wait when an update is already buffered"),
    )
    acquisition = TiledQueueAcquisition(correlation_uid="correlation-uid", item_uid="item-uid")

    outcomes = evaluator(acquisition, suggestions=[])

    assert outcomes[0]["_id"] == "trial-0"
    assert acquisition.run_uid == "run-uid"
    assert subscribe_to_stream.call_count == 1
    wait.assert_not_called()
    assert subscription.disconnected is False
    evaluator.close()
    evaluator.close()
    assert subscription.disconnected is True


def test_tiled_evaluation_reuses_persistent_subscription(mocker) -> None:
    first_run = FakeRun(alignment_images()[:2], ["trial-0"])
    first_run.metadata["start"][CORRELATION_UID_KEY] = "correlation-1"
    second_run = FakeRun(alignment_images()[2:], ["trial-1"])
    second_run.metadata["start"][CORRELATION_UID_KEY] = "correlation-2"
    client = SearchableClient({"run-1": first_run, "run-2": second_run})
    subscription = FakeStreamSubscription()
    callbacks = []

    def subscribe(catalog, stream_name, data_keys, callback, *, start, max_size, metadata_filter):
        assert catalog is client
        assert stream_name == "primary"
        assert data_keys == "screen_image"
        assert start == 0
        assert max_size == 10_000_000
        assert metadata_filter is not None
        callbacks.append(callback)
        return subscription

    subscribe_to_stream = mocker.patch(
        "tst_sim_tools.agents.energy_alignment.subscribe_to_stream_by_metadata", side_effect=subscribe
    )
    evaluator = make_tiled_evaluator(client)

    callbacks[0](FakeStreamUpdate("run-1", alignment_images()[:2], first_run.metadata["start"]))
    first_outcomes = evaluator(TiledQueueAcquisition("correlation-1"), suggestions=[])
    callbacks[0](FakeStreamUpdate("run-2", alignment_images()[2:], second_run.metadata["start"]))
    second_outcomes = evaluator(TiledQueueAcquisition("correlation-2"), suggestions=[])

    assert first_outcomes[0]["_id"] == "trial-0"
    assert second_outcomes[0]["_id"] == "trial-1"
    assert subscribe_to_stream.call_count == 1
    assert subscription.disconnected is False
    evaluator.close()
    assert subscription.disconnected is True


def test_tiled_evaluation_analyzes_images_as_updates_arrive(mocker) -> None:
    run = FakeRun(alignment_images()[:2], ["trial-0"])
    run.metadata["start"][CORRELATION_UID_KEY] = "correlation-uid"
    subscription = FakeStreamSubscription()
    client = SearchableClient({"run-uid": run})
    callbacks = []

    def subscribe(catalog, stream_name, data_keys, callback, *, start, max_size, metadata_filter):
        _ = catalog, stream_name, data_keys, start, max_size, metadata_filter
        callbacks.append(callback)
        return subscription

    mocker.patch("tst_sim_tools.agents.energy_alignment.subscribe_to_stream_by_metadata", side_effect=subscribe)
    first_image_analyzed = threading.Event()
    analyzed_images = 0

    def analyze(image, *, threshold, blur):
        nonlocal analyzed_images
        metrics = analyze_image(image, threshold=threshold, blur=blur)
        analyzed_images += 1
        if analyzed_images == 1:
            first_image_analyzed.set()
        return metrics

    mocker.patch("tst_sim_tools.agents.energy_alignment.analyze_image", side_effect=analyze)
    evaluator = make_tiled_evaluator(client, queue_poll_period=10.0)
    acquisition = TiledQueueAcquisition(correlation_uid="correlation-uid", item_uid="item-uid")
    results: queue.Queue[list[dict[str, Any]]] = queue.Queue()
    thread = threading.Thread(target=lambda: results.put(evaluator(acquisition, suggestions=[])), daemon=True)
    thread.start()

    callbacks[0](FakeStreamUpdate("run-uid", alignment_images()[:1], run.metadata["start"], sequence=1))
    assert first_image_analyzed.wait(timeout=1)
    assert results.empty()

    callbacks[0](FakeStreamUpdate("run-uid", alignment_images()[1:2], run.metadata["start"], sequence=2))
    outcomes = results.get(timeout=1)
    thread.join(timeout=1)

    assert outcomes[0]["_id"] == "trial-0"
    assert analyzed_images == 2
    assert acquisition.run_uid == "run-uid"
    assert subscription.disconnected is False
    evaluator.close()
    assert subscription.disconnected is True


def test_tiled_evaluation_reports_stream_sequence_gap(mocker) -> None:
    run = FakeRun(alignment_images()[:2], ["trial-0"])
    run.metadata["start"][CORRELATION_UID_KEY] = "correlation-uid"
    client = SearchableClient({"run-uid": run})
    subscription = FakeStreamSubscription()

    def subscribe(catalog, stream_name, data_keys, callback, *, start, max_size, metadata_filter):
        _ = catalog, stream_name, data_keys, start, max_size, metadata_filter
        callback(FakeStreamUpdate("run-uid", alignment_images()[:1], run.metadata["start"], sequence=1))
        callback(FakeStreamUpdate("run-uid", alignment_images()[1:2], run.metadata["start"], sequence=3))
        return subscription

    mocker.patch("tst_sim_tools.agents.energy_alignment.subscribe_to_stream_by_metadata", side_effect=subscribe)
    evaluator = make_tiled_evaluator(client)

    with pytest.raises(RuntimeError, match="expected sequence 2, got 3"):
        evaluator(TiledQueueAcquisition("correlation-uid"), suggestions=[])

    evaluator.close()
    assert subscription.disconnected is True


def test_tiled_evaluation_propagates_queue_failure_before_run(mocker) -> None:
    subscription = FakeStreamSubscription()
    client = SearchableClient({})
    re_manager = mocker.Mock()
    re_manager.queue_get.return_value = {"running_item": {}, "items": []}
    re_manager.history_get.return_value = {
        "items": [
            {
                "item_uid": "item-uid",
                "result": {"exit_status": "failed", "reason": "plan preparation failed"},
            }
        ]
    }
    mocker.patch("tst_sim_tools.agents.energy_alignment.subscribe_to_stream_by_metadata", return_value=subscription)
    evaluator = make_tiled_evaluator(client, re_manager=re_manager)

    with pytest.raises(RuntimeError, match="plan preparation failed"):
        evaluator(TiledQueueAcquisition("correlation-uid", item_uid="item-uid"), suggestions=[])

    assert subscription.disconnected is False
    evaluator.close()
    assert subscription.disconnected is True


def test_closing_tiled_evaluation_wakes_waiter(mocker) -> None:
    subscription = FakeStreamSubscription()
    mocker.patch("tst_sim_tools.agents.energy_alignment.subscribe_to_stream_by_metadata", return_value=subscription)
    evaluator = make_tiled_evaluator(SearchableClient({}), queue_poll_period=10.0)
    started = threading.Event()
    errors: queue.Queue[BaseException] = queue.Queue()

    def evaluate() -> None:
        started.set()
        try:
            evaluator(TiledQueueAcquisition("correlation-uid"), suggestions=[])
        except BaseException as error:
            errors.put(error)

    thread = threading.Thread(target=evaluate, daemon=True)
    thread.start()
    assert started.wait(timeout=1)

    evaluator.close()
    thread.join(timeout=1)

    error = errors.get(timeout=1)
    assert isinstance(error, RuntimeError)
    assert str(error) == "Tiled stream subscription is closed"
    assert subscription.disconnected is True


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


def test_build_qs_agent_wires_tiled_websocket_evaluation_and_queue_plan(mocker) -> None:
    re_manager = cast(REManagerAPI, object())
    tiled_client = object()
    agent = fake_agent(mocker)
    agent_type = mocker.patch("tst_sim_tools.agents.energy_alignment.TiledQueueserverAgent", return_value=agent)
    subscription = FakeStreamSubscription()
    subscribe_to_stream = mocker.patch(
        "tst_sim_tools.agents.energy_alignment.subscribe_to_stream_by_metadata", return_value=subscription
    )

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
        queue_poll_period=0.25,
        websocket_start=7,
        websocket_max_size=2048,
        stream_buffer_size=17,
        autostart=False,
    )

    assert result is agent
    assert agent_type.call_args.args == (re_manager,)
    kwargs = agent_type.call_args.kwargs
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
    assert isinstance(evaluator, TiledWebsocketEnergyAlignmentEvaluation)
    assert evaluator._client is tiled_client
    assert evaluator._re_manager is re_manager
    assert evaluator._image_key == "screen_image"
    assert evaluator._target_centroid == (2.0, 2.0)
    np.testing.assert_array_equal(evaluator._energies, [7000.0, 7100.0])
    assert evaluator._threshold == 0.1
    assert evaluator._blur == 0.5
    assert evaluator._queue_poll_period == 0.25
    subscribe_to_stream.assert_called_once_with(
        tiled_client,
        "primary",
        "screen_image",
        evaluator._record_update,
        metadata_filter=mocker.ANY,
        start=7,
        max_size=2048,
    )
    metadata_filter = subscribe_to_stream.call_args.kwargs["metadata_filter"]
    assert metadata_filter({CORRELATION_UID_KEY: "correlation", "blop_suggestions": []}) is True
    assert metadata_filter({}) is False
    assert evaluator._updates.maxlen == 17
    assert kwargs["outcome_constraints"] is BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS
    assert kwargs["checkpoint_path"] == "/tmp/qs-agent.json"
    assert kwargs["autostart"] is False
    agent.ax_client.configure_generation_strategy.assert_called_once_with(
        method="fast",
        initialization_budget=BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
        initialize_with_center=False,
    )
    evaluator.close()
    assert subscription.disconnected is True
