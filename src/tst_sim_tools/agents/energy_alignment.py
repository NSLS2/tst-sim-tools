"""Blop agent for energy dependent alignment."""

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
from ax.api.protocols import IMetric
from blop.ax import Agent, Objective, OutcomeConstraint, RangeDOF
from blop.protocols import AcquisitionPlan, EvaluationFunction
from bluesky.protocols import Readable
from bluesky_queueserver_api.http import REManagerAPI

from ..analysis.image import analyze_energy_scan, image_series
from ..devices.materials import XRTCrystalSi
from ..devices.mirrors import XRTOpticalElement, XRTToroidMirror
from ..devices.sources import XRTWiggler
from ..plans.bmm import acquire_with_energy_scan
from .tiled_queueserver import CORRELATION_UID_KEY, TiledQueueAcquisition, TiledQueueserverAgent

logger = logging.getLogger(__name__)

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
BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS = (-7e-5, 9e-5)
BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS = (-4e-3, 1e-3)
BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS = (-0.5, 0.65)
BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET = 12


@dataclass
class _TiledRunWaitState:
    uid: str | None = None
    stop: Mapping[str, Any] | None = None
    error: Exception | None = None


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
        image_poll_period: float = 0.1,
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

    def _before_image_retry(self, uid: str) -> None:
        return

    def _poll_for_images(self, uid: str) -> np.ndarray:
        while True:
            try:
                run = self._client[uid]
                stream = run["primary"]
                return stream[self._image_key].read()
            except KeyError:
                self._before_image_retry(uid)
                time.sleep(self._image_poll_period)

    def _suggestion_ids(self, uid: str) -> list[Any]:
        return [suggestion["_id"] for suggestion in self._client[uid].metadata["start"]["blop_suggestions"]]

    def _evaluate_run(self, uid: str, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        images = self._poll_for_images(uid)
        image_stack = image_series(images)
        suggestion_ids = self._suggestion_ids(uid)
        n_energies = self._energies.size
        expected_images = len(suggestion_ids) * n_energies
        if image_stack.shape[0] != expected_images:
            raise ValueError(
                f"Expected {expected_images} image(s) from Tiled for {len(suggestion_ids)} suggestion(s) "
                f"and {n_energies} energy point(s), but got {image_stack.shape[0]}"
            )

        outcomes = []
        target_x, target_y = self._target_centroid
        for idx, sid in enumerate(suggestion_ids):
            start = idx * n_energies
            stop = start + n_energies
            suggestion_stack = image_stack[start:stop]
            summary = analyze_energy_scan(
                suggestion_stack,
                energies=self._energies,
                plot=False,
                threshold=self._threshold,
                blur=self._blur,
            )
            x_error = summary["x_centroid"] - target_x
            y_error = summary["y_centroid"] - target_y
            centroid_error = np.hypot(x_error, y_error)
            centroid_rms_error = float(np.sqrt(np.mean(centroid_error**2)))
            max_centroid_error = float(np.max(centroid_error))
            fwhm = float(np.mean(summary["fwhm_px"]))
            max_fwhm = float(np.max(summary["fwhm_px"]))
            min_intensity = float(np.min(summary["total"]))
            centroid_span = float(np.hypot(np.ptp(summary["x_centroid"]), np.ptp(summary["y_centroid"])))
            alignment_score = (
                centroid_rms_error**2
                + BMM_ENERGY_ALIGNMENT_MAX_CENTROID_ERROR_WEIGHT * max_centroid_error**2
                + BMM_ENERGY_ALIGNMENT_CENTROID_SPAN_WEIGHT * centroid_span**2
                + BMM_ENERGY_ALIGNMENT_FWHM_WEIGHT * fwhm**2
                + BMM_ENERGY_ALIGNMENT_MAX_FWHM_WEIGHT * max_fwhm**2
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

    def __call__(self, uid: Any, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return self._evaluate_run(str(uid), suggestions)


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
        self._re_manager = re_manager
        self._queue_poll_period = queue_poll_period
        self._websocket_start = websocket_start
        self._websocket_max_size = websocket_max_size
        self._active_acquisition: TiledQueueAcquisition | None = None

    def __call__(self, uid: Any, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        acquisition = self._coerce_acquisition(uid)
        run_uid = self._wait_for_correlated_run(acquisition)
        acquisition.run_uid = run_uid
        self._active_acquisition = acquisition
        try:
            return self._evaluate_run(run_uid, suggestions)
        finally:
            self._active_acquisition = None

    def _coerce_acquisition(self, uid: Any) -> TiledQueueAcquisition:
        if isinstance(uid, TiledQueueAcquisition):
            return uid
        return TiledQueueAcquisition(correlation_uid=str(uid))

    def _before_image_retry(self, uid: str) -> None:
        if self._active_acquisition is not None:
            self._check_queue_state(self._active_acquisition)

    def _wait_for_correlated_run(self, acquisition: TiledQueueAcquisition) -> str:
        condition = threading.Condition()
        state = _TiledRunWaitState()

        def record(uid: str, metadata: Mapping[str, Any]) -> None:
            try:
                if not self._metadata_matches(metadata, acquisition.correlation_uid):
                    return
                stop = self._stop_document(metadata)
                if stop is not None:
                    self._raise_for_stop(uid, stop)
                with condition:
                    state.uid = uid
                    state.stop = stop
                    condition.notify_all()
            except Exception as exc:
                with condition:
                    state.error = exc
                    condition.notify_all()

        existing = self._find_existing_run(acquisition)
        if existing is not None:
            record(*existing)
        with condition:
            if state.error is not None:
                raise state.error
            if state.uid is not None and state.stop is not None:
                return state.uid

        subscription = self._client.subscribe()

        def on_child_created(update: Any) -> None:
            metadata = getattr(update, "metadata", {})
            if isinstance(metadata, Mapping):
                record(str(update.key), metadata)

        def on_child_metadata_updated(update: Any) -> None:
            metadata = self._read_run_metadata(str(update.key))
            if metadata is not None:
                record(str(update.key), metadata)

        callbacks = [on_child_created, on_child_metadata_updated]
        subscription.child_created.add_callback(on_child_created)
        subscription.child_metadata_updated.add_callback(on_child_metadata_updated)

        def run_subscription() -> None:
            try:
                subscription.start(start=self._websocket_start, max_size=self._websocket_max_size)
            except Exception as exc:
                with condition:
                    state.error = exc
                    condition.notify_all()

        subscription_thread = threading.Thread(
            target=run_subscription,
            name=f"tiled-energy-alignment-{acquisition.correlation_uid}",
            daemon=True,
        )
        subscription_thread.start()
        try:
            existing = self._find_existing_run(acquisition)
            if existing is not None:
                record(*existing)
            while True:
                with condition:
                    if state.error is not None:
                        raise state.error
                    if state.uid is not None and state.stop is not None:
                        return state.uid
                    condition.wait(timeout=self._queue_poll_period)
                self._check_queue_state(acquisition)
                existing = self._find_existing_run(acquisition)
                if existing is not None:
                    record(*existing)
                _ = callbacks
        finally:
            try:
                subscription.disconnect()
            except Exception as exc:
                logger.debug("Failed to disconnect Tiled subscription: %s", exc)
            subscription_thread.join(timeout=1.0)

    def _metadata_matches(self, metadata: Mapping[str, Any], correlation_uid: str) -> bool:
        start = metadata.get("start", {})
        return isinstance(start, Mapping) and start.get(CORRELATION_UID_KEY) == correlation_uid

    def _stop_document(self, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        stop = metadata.get("stop")
        return stop if isinstance(stop, Mapping) else None

    def _raise_for_stop(self, uid: str, stop: Mapping[str, Any]) -> None:
        exit_status = stop.get("exit_status")
        if exit_status != "success":
            reason = stop.get("reason") or "(no reason given)"
            raise RuntimeError(f"Acquisition run {uid!r} ended with status {exit_status!r}: {reason}")

    def _find_existing_run(self, acquisition: TiledQueueAcquisition) -> tuple[str, Mapping[str, Any]] | None:
        for uid in self._search_runs_by_correlation(acquisition.correlation_uid):
            metadata = self._read_run_metadata(uid)
            if metadata is not None and self._metadata_matches(metadata, acquisition.correlation_uid):
                return uid, metadata
        for uid in self._history_run_uids(acquisition):
            metadata = self._read_run_metadata(uid)
            if metadata is not None and self._metadata_matches(metadata, acquisition.correlation_uid):
                return uid, metadata
        return None

    def _search_runs_by_correlation(self, correlation_uid: str) -> list[str]:
        try:
            from tiled.queries import Eq

            results = self._client.search(Eq(f"start.{CORRELATION_UID_KEY}", correlation_uid))
            return [str(key) for key, _run in results.items()]
        except Exception as exc:
            logger.debug("Tiled correlation search did not return runs: %s", exc)
            return []

    def _read_run_metadata(self, uid: str) -> Mapping[str, Any] | None:
        try:
            metadata = self._client[uid].metadata
        except Exception as exc:
            logger.debug("Unable to read Tiled metadata for %s: %s", uid, exc)
            return None
        return metadata if isinstance(metadata, Mapping) else None

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
        failure = self._history_failure_reason(history_item)
        if failure is not None:
            raise RuntimeError(f"Queue item {item_uid!r} failed before Tiled produced a matching run: {failure}")

    def _history_item(self, item_uid: str) -> Mapping[str, Any] | None:
        if self._re_manager is None:
            return None
        history = self._re_manager.history_get(reload=True)
        for item in history.get("items", []):
            if item.get("item_uid") == item_uid:
                return item
        return None

    def _history_failure_reason(self, history_item: Mapping[str, Any]) -> str | None:
        result = history_item.get("result")
        if isinstance(result, Mapping):
            exit_status = result.get("exit_status")
            if exit_status not in (None, "success"):
                return str(result.get("reason") or exit_status)
            if result.get("success") is False:
                return str(result.get("msg") or result)
        status = history_item.get("status")
        if status in {"failed", "aborted", "stopped"}:
            return str(status)
        return None

    def _history_run_uids(self, acquisition: TiledQueueAcquisition) -> list[str]:
        if acquisition.item_uid is None:
            return []
        history_item = self._history_item(acquisition.item_uid)
        if history_item is None:
            return []
        return sorted(
            {
                value
                for value in self._walk_values(history_item)
                if isinstance(value, str) and len(value) == 36 and value.count("-") == 4
            }
        )

    def _walk_values(self, value: Any) -> list[Any]:
        values = [value]
        if isinstance(value, Mapping):
            for child in value.values():
                values.extend(self._walk_values(child))
        elif isinstance(value, list | tuple):
            for child in value:
                values.extend(self._walk_values(child))
        return values


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
