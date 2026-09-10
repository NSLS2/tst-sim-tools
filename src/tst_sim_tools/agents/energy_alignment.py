"""Blop agent for energy dependent alignment."""

import threading
import time
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from functools import partial
from typing import Any, cast

import numpy as np
from ax.api.protocols import IMetric
from blop.ax import Agent, Objective, OutcomeConstraint, RangeDOF
from blop.protocols import AcquisitionPlan, EvaluationFunction
from bluesky.protocols import Readable
from bluesky_queueserver_api.http import REManagerAPI
from bluesky_tiled_plugins import BlueskyStreamUpdate, subscribe_to_stream_by_metadata

from ..analysis.image import analyze_image, image_series
from ..devices.materials import XRTCrystalSi
from ..devices.mirrors import XRTOpticalElement, XRTToroidMirror
from ..devices.sources import XRTWiggler
from ..plans.bmm import acquire_with_energy_scan
from .tiled_queueserver import CORRELATION_UID_KEY, TiledQueueAcquisition, TiledQueueserverAgent

ALIGNMENT_SCORE = "alignment_score"
CENTROID_RMS_ERROR = "centroid_rms_error"
MAX_CENTROID_ERROR = "max_centroid_error"
CENTROID_SPAN = "centroid_span"
LATERAL_POSITION_ERROR = "lateral_position_error"
VERTICAL_POSITION_ERROR = "vertical_position_error"
FWHM = "fwhm"
MAX_FWHM = "max_fwhm"
MIN_INTENSITY = "min_intensity"
DEFAULT_IMAGE_THRESHOLD = 0.02
DEFAULT_IMAGE_BLUR = 1.0
CATALOG_POLL_INTERVAL = 0.1
CATALOG_POLL_TIMEOUT = 10.0
BMM_ENERGY_ALIGNMENT_MAX_CENTROID_ERROR_WEIGHT = 0.25
BMM_ENERGY_ALIGNMENT_CENTROID_SPAN_WEIGHT = 0.1
BMM_ENERGY_ALIGNMENT_FWHM_WEIGHT = 0.005
BMM_ENERGY_ALIGNMENT_MAX_FWHM_WEIGHT = 0.0025
BMM_ENERGY_ALIGNMENT_MIN_INTENSITY_FLOOR = 15000.0
BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS = (
    OutcomeConstraint(
        f"intensity >= {BMM_ENERGY_ALIGNMENT_MIN_INTENSITY_FLOOR}",
        intensity=IMetric(name=MIN_INTENSITY),
    ),
)
BMM_ENERGY_ALIGNMENT_REFERENCE_ENERGY = 7112.0
BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL = 5e-5
BMM_ENERGY_ALIGNMENT_TFM_LATERAL = 0.33
BMM_ENERGY_ALIGNMENT_TFM_YAW = 0.0
BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS = (-2e-4, 1e-4)
BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS = (-1e-4, 1e-4)
BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS = (-0.5, 0.65)
BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET = 12


class EnergyAlignmentEvalutation(EvaluationFunction):
    """Evaluate Bluesky runs for energy alignment from detector-screen images."""

    def __init__(
        self,
        tiled_client: Any,
        image_key: str,
        target_centroid: tuple[float, float],
        energies: Sequence[float],
        threshold: float = DEFAULT_IMAGE_THRESHOLD,
        blur: float = DEFAULT_IMAGE_BLUR,
        image_poll_period: float = CATALOG_POLL_INTERVAL,
    ) -> None:
        self._client = tiled_client
        self._image_key = image_key
        self._target_centroid = target_centroid
        self._energies = np.asarray(tuple(float(energy) for energy in energies), dtype=float)
        if self._energies.ndim != 1 or self._energies.size == 0:
            raise ValueError("Expected at least one energy for energy alignment evaluation")
        self._threshold = threshold
        self._blur = blur
        self._image_poll_period = image_poll_period

    def _before_image_retry(self, uid: Hashable) -> None:
        return

    def _poll_for_run_images(self, uid: Hashable) -> tuple[Any, np.ndarray]:
        deadline = time.monotonic() + CATALOG_POLL_TIMEOUT
        while True:
            waiting_for = f"run {uid!r} in Tiled"
            try:
                run = self._client[uid]
                waiting_for = f"primary stream for run {uid!r}"
                stream = run["primary"]
                waiting_for = f"image key {self._image_key!r} in the primary stream for run {uid!r}"
                image = stream[self._image_key]
                waiting_for = f"readable data for image key {self._image_key!r} in run {uid!r}"
                return run, image.read()
            except KeyError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {CATALOG_POLL_TIMEOUT:g} seconds waiting for {waiting_for}"
                    ) from error
                self._before_image_retry(uid)
                time.sleep(self._image_poll_period)

    def _evaluate_run(self, uid: Hashable, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        run, images = self._poll_for_run_images(uid)
        return self._evaluate_images(run.metadata["start"], images)

    def _evaluate_images(self, start_document: Mapping[str, Any], images: Any) -> list[dict[str, Any]]:
        return self._evaluate_image_metrics(start_document, self._analyze_images(images))

    def _analyze_images(self, images: Any) -> list[dict[str, float]]:
        return [analyze_image(frame, threshold=self._threshold, blur=self._blur) for frame in image_series(images)]

    def _evaluate_image_metrics(
        self,
        start_document: Mapping[str, Any],
        image_metrics: Sequence[Mapping[str, float]],
    ) -> list[dict[str, Any]]:
        suggestion_ids = self._suggestion_ids(start_document)
        n_energies = self._energies.size
        expected_images = len(suggestion_ids) * n_energies
        if len(image_metrics) != expected_images:
            raise ValueError(
                f"Expected {expected_images} image(s) from Tiled for {len(suggestion_ids)} suggestion(s) "
                f"and {n_energies} energy point(s), but got {len(image_metrics)}"
            )

        outcomes = []
        target_x, target_y = self._target_centroid
        for idx, sid in enumerate(suggestion_ids):
            start = idx * n_energies
            stop = start + n_energies
            suggestion_metrics = image_metrics[start:stop]
            x_centroid = np.array([metrics["x_centroid"] for metrics in suggestion_metrics], dtype=float)
            y_centroid = np.array([metrics["y_centroid"] for metrics in suggestion_metrics], dtype=float)
            fwhm_values = np.array([metrics["fwhm_px"] for metrics in suggestion_metrics], dtype=float)
            intensity = np.array([metrics["total"] for metrics in suggestion_metrics], dtype=float)
            x_error = x_centroid - target_x
            y_error = y_centroid - target_y
            centroid_error = np.hypot(x_error, y_error)
            centroid_rms_error = float(np.sqrt(np.mean(centroid_error**2)))
            max_centroid_error = float(np.max(centroid_error))
            fwhm = float(np.mean(fwhm_values))
            max_fwhm = float(np.max(fwhm_values))
            min_intensity = float(np.min(intensity))
            centroid_span = float(np.hypot(np.ptp(x_centroid), np.ptp(y_centroid)))
            alignment_score = (
                centroid_rms_error
                + BMM_ENERGY_ALIGNMENT_MAX_CENTROID_ERROR_WEIGHT * max_centroid_error
                + BMM_ENERGY_ALIGNMENT_CENTROID_SPAN_WEIGHT * centroid_span
                + BMM_ENERGY_ALIGNMENT_FWHM_WEIGHT * fwhm
                + BMM_ENERGY_ALIGNMENT_MAX_FWHM_WEIGHT * max_fwhm
            )
            outcomes.append(
                {
                    ALIGNMENT_SCORE: alignment_score,
                    CENTROID_RMS_ERROR: centroid_rms_error,
                    MAX_CENTROID_ERROR: max_centroid_error,
                    CENTROID_SPAN: centroid_span,
                    LATERAL_POSITION_ERROR: float(np.sqrt(np.mean(x_error**2))),
                    VERTICAL_POSITION_ERROR: float(np.sqrt(np.mean(y_error**2))),
                    FWHM: fwhm,
                    MAX_FWHM: max_fwhm,
                    MIN_INTENSITY: min_intensity,
                    "_id": sid,
                }
            )
        return outcomes

    def _suggestion_ids(self, start_document: Mapping[str, Any]) -> list[Any]:
        return [suggestion["_id"] for suggestion in start_document["blop_suggestions"]]

    def __call__(self, uid: Hashable, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return self._evaluate_run(uid, suggestions)


class TiledWebsocketEnergyAlignmentEvaluation(EnergyAlignmentEvalutation):
    """Wait for correlated Queue Server runs in Tiled websockets before evaluating images."""

    def __init__(
        self,
        tiled_client: Any,
        image_key: str,
        target_centroid: tuple[float, float],
        energies: Sequence[float],
        *,
        re_manager: REManagerAPI | None = None,
        threshold: float = DEFAULT_IMAGE_THRESHOLD,
        blur: float = DEFAULT_IMAGE_BLUR,
        queue_poll_period: float = 5.0,
        websocket_start: int | None = 0,
        websocket_max_size: int = 10_000_000,
        stream_buffer_size: int = 1024,
        image_poll_period: float = 0.5,
    ) -> None:
        super().__init__(
            tiled_client,
            image_key=image_key,
            target_centroid=target_centroid,
            energies=energies,
            threshold=threshold,
            blur=blur,
            image_poll_period=image_poll_period,
        )
        if stream_buffer_size < 1:
            raise ValueError("stream_buffer_size must be positive")
        self._re_manager = re_manager
        self._queue_poll_period = queue_poll_period
        self._condition = threading.Condition()
        self._updates: deque[BlueskyStreamUpdate] = deque(maxlen=stream_buffer_size)
        self._evaluation_lock = threading.Lock()
        self._closed = False
        self._subscription = subscribe_to_stream_by_metadata(
            self._client,
            "primary",
            self._image_key,
            self._record_update,
            # TODO: Create some sort of agent UID so multiple agents don't collide
            metadata_filter=lambda start_document: (
                CORRELATION_UID_KEY in start_document and "blop_suggestions" in start_document
            ),
            start=websocket_start,
            max_size=websocket_max_size,
        )

    def __call__(self, uid: Any, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not self._evaluation_lock.acquire(blocking=False):
            raise RuntimeError("Tiled energy-alignment evaluation is already running")
        try:
            acquisition = self._coerce_acquisition(uid)
            run_uid, start_document, image_metrics = self._wait_for_correlated_metrics(acquisition)
            acquisition.run_uid = run_uid
            return self._evaluate_image_metrics(start_document, image_metrics)
        finally:
            self._evaluation_lock.release()

    def _coerce_acquisition(self, uid: Any) -> TiledQueueAcquisition:
        if isinstance(uid, TiledQueueAcquisition):
            return uid
        return TiledQueueAcquisition(correlation_uid=str(uid))

    def _record_update(self, update: BlueskyStreamUpdate) -> None:
        with self._condition:
            if self._closed:
                return
            self._updates.append(update)
            self._condition.notify_all()

    def close(self) -> None:
        """Disconnect the persistent Tiled stream subscription."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._updates.clear()
            self._condition.notify_all()
        self._subscription.disconnect()

    def _wait_for_correlated_metrics(
        self, acquisition: TiledQueueAcquisition
    ) -> tuple[str, Mapping[str, Any], list[dict[str, float]]]:
        metrics_by_run: dict[str, list[dict[str, float]]] = {}
        last_sequences: dict[str, int] = {}
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or bool(self._updates),
                    timeout=self._queue_poll_period,
                )
                if self._closed:
                    raise RuntimeError("Tiled stream subscription is closed")
                pending = tuple(self._updates)
                self._updates.clear()
            for update in pending:
                if update.start_document.get(CORRELATION_UID_KEY) != acquisition.correlation_uid:
                    continue
                previous_sequence = last_sequences.get(update.run_uid)
                if previous_sequence is not None:
                    if update.sequence <= previous_sequence:
                        continue
                    if update.sequence != previous_sequence + 1:
                        raise RuntimeError(
                            f"Gap in Tiled stream updates for run {update.run_uid!r}: "
                            f"expected sequence {previous_sequence + 1}, got {update.sequence}"
                        )
                metrics = self._analyze_images(update.data())
                last_sequences[update.run_uid] = update.sequence
                run_metrics = metrics_by_run.setdefault(update.run_uid, [])
                run_metrics.extend(metrics)
                expected_images = len(self._suggestion_ids(update.start_document)) * self._energies.size
                if len(run_metrics) >= expected_images:
                    return update.run_uid, update.start_document, run_metrics
            self._check_queue_state(acquisition)

    def _check_queue_state(self, acquisition: TiledQueueAcquisition) -> None:
        if self._re_manager is None or acquisition.item_uid is None:
            return
        item_uid = acquisition.item_uid
        queue = self._re_manager.queue_get(reload=True)
        running = queue.get("running_item") or {}
        queued = queue.get("items", [])
        if running.get("item_uid") == item_uid or any(item.get("item_uid") == item_uid for item in queued):
            return
        history_item = self._history_item(item_uid)
        if history_item is None:
            raise RuntimeError(f"Queue item {item_uid!r} is no longer queued or running and has no history entry")
        incomplete_reason = self._history_incomplete_reason(history_item)
        if incomplete_reason is not None:
            raise RuntimeError(f"Queue item {item_uid!r} ended before Tiled produced enough data: {incomplete_reason}")

    def _history_item(self, item_uid: str) -> Mapping[str, Any] | None:
        if self._re_manager is None:
            return None
        history = self._re_manager.history_get(reload=True)
        for item in history.get("items", []):
            if item.get("item_uid") == item_uid:
                return item
        return None

    def _history_incomplete_reason(self, history_item: Mapping[str, Any]) -> str | None:
        result = history_item.get("result")
        if not isinstance(result, Mapping):
            return "history entry has no result"
        exit_status = result.get("exit_status")
        if exit_status == "completed":
            return None
        if exit_status is None:
            return "history result has no exit_status"
        message = result.get("msg")
        return f"{exit_status}: {message}" if message else str(exit_status)


def build_re_agent(
    dets: Sequence[Readable],
    tpw: XRTWiggler,
    dcm_c1: XRTOpticalElement,
    dcm_c2: XRTOpticalElement,
    crystal: XRTCrystalSi,
    m2: XRTToroidMirror,
    tiled_client: Any,
    image_key: str,
    target_centroid: tuple[float, float],
    energies: Sequence[float],
    band: float = 1.0,
    threshold: float = DEFAULT_IMAGE_THRESHOLD,
    blur: float = DEFAULT_IMAGE_BLUR,
    checkpoint_path: str = "/tmp/blop/energy-alignment.json",
) -> Agent:
    """Build an agent that optimizes detector-screen statistics across an energy range.

    Parameters
    ----------
    dets
        Detector readables acquired at each energy for each optimization point.
    tpw
        Wiggler source with min/max energy controls.
    dcm_c1
        DCM crystal-1 optical element used for energy retuning.
    dcm_c2
        DCM crystal-2 optical element; its roll is the optimized DCM DOF.
    crystal
        Silicon crystal material supplying the lattice spacing.
    m2
        TFM of the BMM beamline.
    tiled_client
        Tiled client used to read back acquired detector images by run UID.
    image_key
        Tiled primary-stream image data key.
    target_centroid
        ``(x, y)`` pixel coordinate for the aligned beam.
    energies
        Energy points acquired for each optimizer suggestion.
    band
        Source energy band width around each energy point.
    threshold
        Fraction of peak intensity to retain before blurring.
    blur
        Gaussian denoising blur sigma in pixels.
    checkpoint_path
        Ax optimizer checkpoint path.

    Returns
    -------
    Agent
        Configured Blop agent for BMM simulator alignment.
    """
    dofs = [
        RangeDOF(actuator=dcm_c2.roll, bounds=BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS, parameter_type="float"),
        RangeDOF(actuator=m2.yaw, bounds=BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS, parameter_type="float"),
        RangeDOF(actuator=m2.center_x, bounds=BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS, parameter_type="float"),
    ]

    objectives = [
        Objective(name=ALIGNMENT_SCORE, minimize=True),
    ]

    outcome_constraints = BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS

    acquisition_plan = cast(
        AcquisitionPlan,
        partial(
            acquire_with_energy_scan,
            tpw=tpw,
            dcm_c1=dcm_c1,
            dcm_c2=dcm_c2,
            crystal=crystal,
            energies=tuple(float(e) for e in energies),
            band=band,
        ),
    )

    agent = Agent(
        sensors=dets,
        dofs=dofs,
        objectives=objectives,
        evaluation_function=EnergyAlignmentEvalutation(
            tiled_client,
            image_key=image_key,
            target_centroid=target_centroid,
            energies=energies,
            threshold=threshold,
            blur=blur,
        ),
        acquisition_plan=acquisition_plan,
        outcome_constraints=outcome_constraints,
        checkpoint_path=checkpoint_path,
    )
    agent.ax_client.configure_generation_strategy(
        method="fast",
        initialization_budget=BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
        initialize_with_center=False,
    )

    return agent


def build_qs_agent(
    re_manager: REManagerAPI,
    dets: Sequence[str],
    crystal_name: str,
    tiled_client: Any,
    image_key: str,
    target_centroid: tuple[float, float],
    energies: Sequence[float],
    band: float = 1.0,
    threshold: float = DEFAULT_IMAGE_THRESHOLD,
    blur: float = DEFAULT_IMAGE_BLUR,
    checkpoint_path: str = "/tmp/blop/energy-alignment.json",
    queue_poll_period: float = 5.0,
    websocket_start: int | None = 0,
    websocket_max_size: int = 10_000_000,
    stream_buffer_size: int = 1024,
    autostart: bool = True,
) -> TiledQueueserverAgent:
    """Build a Queue Server agent that waits for energy-alignment data through Tiled websockets.

    Parameters
    ----------
    dets
        Detector readables acquired at each energy for each optimization point.
    crystal_name
        Device name for silicon crystal material supplying the lattice spacing.
    tiled_client
        Tiled catalog client whose websocket stream receives Bluesky run containers.
    image_key
        Tiled primary-stream image data key.
    target_centroid
        ``(x, y)`` pixel coordinate for the aligned beam.
    energies
        Energy points acquired for each optimizer suggestion.
    band
        Source energy band width around each energy point.
    threshold
        Fraction of peak intensity to retain before blurring.
    blur
        Gaussian denoising blur sigma in pixels.
    checkpoint_path
        Ax optimizer checkpoint path.
    queue_poll_period
        Seconds between Queue Server health checks while waiting for Tiled websocket updates.
    websocket_start
        Tiled stream sequence to replay from. ``0`` asks Tiled for all retained cached events.
    websocket_max_size
        Maximum websocket message size in bytes.
    stream_buffer_size
        Maximum number of pending physical Tiled updates retained by the persistent subscription.
    autostart
        Whether Queue Server should autostart queued plans.

    Returns
    -------
    TiledQueueserverAgent
        Configured Blop-style agent for BMM simulator alignment without a ZMQ RemoteDispatcher.
    """
    dofs = [
        RangeDOF(
            actuator="dcm_c2.roll",
            name="dcm_c2-roll",
            bounds=BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS,
            parameter_type="float",
        ),
        RangeDOF(
            actuator="m2.yaw",
            name="m2-yaw",
            bounds=BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS,
            parameter_type="float",
        ),
        RangeDOF(
            actuator="m2.center_x",
            name="m2-center_x",
            bounds=BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS,
            parameter_type="float",
        ),
    ]

    objectives = [
        Objective(name=ALIGNMENT_SCORE, minimize=True),
    ]

    outcome_constraints = BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS

    acquisition_plan_name = "acquire_with_energy_scan"
    acquisition_plan_kwargs = {
        "tpw": "tpw",
        "dcm_c1": "dcm_c1",
        "dcm_c2": "dcm_c2",
        "crystal": crystal_name,
        "energies": tuple(float(e) for e in energies),
        "band": band,
    }

    agent = TiledQueueserverAgent(
        re_manager,
        sensors=dets,
        dofs=dofs,
        objectives=objectives,
        evaluation_function=TiledWebsocketEnergyAlignmentEvaluation(
            tiled_client,
            image_key=image_key,
            target_centroid=target_centroid,
            energies=energies,
            re_manager=re_manager,
            threshold=threshold,
            blur=blur,
            queue_poll_period=queue_poll_period,
            websocket_start=websocket_start,
            websocket_max_size=websocket_max_size,
            stream_buffer_size=stream_buffer_size,
        ),
        acquisition_plan=acquisition_plan_name,
        acquisition_plan_kwargs=acquisition_plan_kwargs,
        outcome_constraints=outcome_constraints,
        checkpoint_path=checkpoint_path,
        autostart=autostart,
    )
    agent.ax_client.configure_generation_strategy(
        method="fast",
        initialization_budget=BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
        initialize_with_center=False,
    )

    return agent
