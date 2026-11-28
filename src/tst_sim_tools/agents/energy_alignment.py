"""Bluesky Adaptive agents for energy dependent alignment."""

import threading
import time
from collections.abc import Hashable, Sequence
from functools import partial
from typing import Any, cast

import numpy as np
from ax.api.protocols import IMetric
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.protocols import Readable
from bluesky_adaptive.ax import Agent, Objective, OutcomeConstraint, RangeDOF
from bluesky_adaptive.ax.queueserver_agent import QueueserverAgent
from bluesky_adaptive.evaluation import per_run, per_suggestion
from bluesky_adaptive.evaluation.queueserver import QueueItemRuns
from bluesky_adaptive.plans import ACQUISITION_ORDER_KEY
from bluesky_adaptive.protocols import AcquisitionPlan, EvaluationFunction
from bluesky_queueserver_api.http import REManagerAPI
from tiled.client.container import Container

from ..analysis.image import analyze_energy_scan, image_series
from ..devices.materials import XRTCrystalSi
from ..devices.mirrors import XRTOpticalElement, XRTToroidMirror
from ..devices.sources import XRTWiggler
from ..plans.bmm import acquire_with_energy_scan

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
DEFAULT_CHECKPOINT_PATH = "/tmp/bluesky-adaptive/energy-alignment.json"
CATALOG_POLL_INTERVAL = 0.1
CATALOG_POLL_TIMEOUT = 10.0
# Output of the 0MQ proxy that relays the Queue Server worker's documents (pixi run -e qs zmq-proxy-local)
QUEUE_SERVER_DOCUMENT_ADDRESS = "127.0.0.1:45105"
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


def score_energy_scan(
    images: np.ndarray,
    *,
    target_centroid: tuple[float, float],
    energies: Sequence[float],
    threshold: float = DEFAULT_IMAGE_THRESHOLD,
    blur: float = DEFAULT_IMAGE_BLUR,
) -> dict[str, float]:
    """Score how well one suggestion keeps the beam on target across an energy scan.

    Parameters
    ----------
    images
        Detector images of the suggestion, one per energy.
    target_centroid
        ``(x, y)`` pixel coordinate for the aligned beam.
    energies
        Energy points of the images.
    threshold
        Fraction of peak intensity to retain before blurring.
    blur
        Gaussian denoising blur sigma in pixels.

    Returns
    -------
    dict[str, float]
        Alignment outcomes keyed by outcome name.
    """
    summary = analyze_energy_scan(images, energies=energies, plot=False, threshold=threshold, blur=blur)
    target_x, target_y = target_centroid
    x_error = summary["x_centroid"] - target_x
    y_error = summary["y_centroid"] - target_y
    centroid_error = np.hypot(x_error, y_error)
    centroid_rms_error = float(np.sqrt(np.mean(centroid_error**2)))
    max_centroid_error = float(np.max(centroid_error))
    fwhm = float(np.mean(summary["fwhm_px"]))
    max_fwhm = float(np.max(summary["fwhm_px"]))
    centroid_span = float(np.hypot(np.ptp(summary["x_centroid"]), np.ptp(summary["y_centroid"])))
    alignment_score = (
        centroid_rms_error
        + BMM_ENERGY_ALIGNMENT_MAX_CENTROID_ERROR_WEIGHT * max_centroid_error
        + BMM_ENERGY_ALIGNMENT_CENTROID_SPAN_WEIGHT * centroid_span
        + BMM_ENERGY_ALIGNMENT_FWHM_WEIGHT * fwhm
        + BMM_ENERGY_ALIGNMENT_MAX_FWHM_WEIGHT * max_fwhm
    )
    return {
        ALIGNMENT_SCORE: alignment_score,
        CENTROID_RMS_ERROR: centroid_rms_error,
        MAX_CENTROID_ERROR: max_centroid_error,
        CENTROID_SPAN: centroid_span,
        LATERAL_POSITION_ERROR: float(np.sqrt(np.mean(x_error**2))),
        VERTICAL_POSITION_ERROR: float(np.sqrt(np.mean(y_error**2))),
        FWHM: fwhm,
        MAX_FWHM: max_fwhm,
        MIN_INTENSITY: float(np.min(summary["total"])),
    }


class TiledEnergyScanImages:
    """Wait for an energy-scan run in Tiled and return its detector images by suggestion ID.

    Pass an instance to :func:`bluesky_adaptive.evaluation.per_suggestion`. Like
    :class:`bluesky_adaptive.evaluation.tiled.TiledRunPoller`, it waits for the run's stop document and pairs the
    ``"primary"`` rows with ``start["bluesky-adaptive"]["acquisition_order"]``. Unlike it, each suggestion owns
    ``n_energies`` consecutive rows, one per energy, as :func:`~tst_sim_tools.plans.bmm.acquire_with_energy_scan`
    acquires them.

    Parameters
    ----------
    client
        Tiled container keyed by run UID that a ``TiledWriter`` writes to.
    image_key
        ``"primary"`` stream data key of the detector images.
    n_energies
        Number of energy points acquired for each suggestion.
    poll_interval
        Seconds between polls.
    timeout
        Seconds to wait for the run's stop document before raising :class:`TimeoutError`.
    """

    def __init__(
        self,
        client: Container,
        image_key: str,
        n_energies: int,
        *,
        poll_interval: float = CATALOG_POLL_INTERVAL,
        timeout: float = CATALOG_POLL_TIMEOUT,
    ) -> None:
        self._client = client
        self._image_key = image_key
        self._n_energies = n_energies
        self._poll_interval = poll_interval
        self._timeout = timeout

    def __call__(self, uid: str) -> dict[Hashable, np.ndarray]:
        """Return each suggestion's ``(n_energies, height, width)`` image stack, keyed by suggestion ID."""
        run = self._wait_for_stop(uid)
        acquisition_order = run.metadata["start"]["bluesky-adaptive"][ACQUISITION_ORDER_KEY]
        images = image_series(run["primary"][self._image_key].read())
        expected_images = len(acquisition_order) * self._n_energies
        if images.shape[0] != expected_images:
            raise ValueError(
                f"Expected {expected_images} image(s) from Tiled for {len(acquisition_order)} suggestion(s) "
                f"and {self._n_energies} energy point(s), but got {images.shape[0]}"
            )
        stacks = images.reshape(len(acquisition_order), self._n_energies, *images.shape[1:])
        return dict(zip(acquisition_order, stacks, strict=True))

    def _wait_for_stop(self, uid: str) -> Any:
        """Poll until the run is in Tiled with a stop document and return its client."""
        deadline = time.monotonic() + self._timeout
        while True:
            # Container.get returns None until the run's start document reaches Tiled. TiledWriter writes the stop
            # document after all of the run's data, so a run with a stop document is complete.
            run = self._client.get(uid)
            if run is not None and run.metadata.get("stop") is not None:
                return run
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Run {uid!r} has no stop document in Tiled after {self._timeout:g} seconds")
            time.sleep(self._poll_interval)


def energy_alignment_evaluation(
    tiled_client: Container,
    image_key: str,
    target_centroid: tuple[float, float],
    energies: Sequence[float],
    threshold: float = DEFAULT_IMAGE_THRESHOLD,
    blur: float = DEFAULT_IMAGE_BLUR,
) -> EvaluationFunction[str]:
    """Build an evaluation function that scores each suggestion's energy-scan images read from Tiled.

    Parameters
    ----------
    tiled_client
        Tiled container keyed by run UID that a ``TiledWriter`` writes to.
    image_key
        ``"primary"`` stream data key of the detector images.
    target_centroid
        ``(x, y)`` pixel coordinate for the aligned beam.
    energies
        Energy points acquired for each suggestion.
    threshold
        Fraction of peak intensity to retain before blurring.
    blur
        Gaussian denoising blur sigma in pixels.

    Returns
    -------
    EvaluationFunction[str]
        Evaluates the run UIDs that :func:`~tst_sim_tools.plans.bmm.acquire_with_energy_scan` returns.

    Raises
    ------
    ValueError
        If ``energies`` is empty.
    """
    energies = tuple(float(energy) for energy in energies)
    if not energies:
        raise ValueError("Expected at least one energy for energy alignment evaluation")
    return per_suggestion(
        partial(score_energy_scan, target_centroid=target_centroid, energies=energies, threshold=threshold, blur=blur),
        TiledEnergyScanImages(tiled_client, image_key, len(energies)),
    )


def build_re_agent(
    dets: Sequence[Readable],
    tpw: XRTWiggler,
    dcm_c1: XRTOpticalElement,
    dcm_c2: XRTOpticalElement,
    crystal: XRTCrystalSi,
    m2: XRTToroidMirror,
    tiled_client: Container,
    image_key: str,
    target_centroid: tuple[float, float],
    energies: Sequence[float],
    band: float = 1.0,
    threshold: float = DEFAULT_IMAGE_THRESHOLD,
    blur: float = DEFAULT_IMAGE_BLUR,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
) -> Agent[str]:
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
        Tiled container that the RunEngine's ``TiledWriter`` writes to.
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
    Agent[str]
        Configured Bluesky Adaptive agent for BMM simulator alignment.
    """
    energies = tuple(float(energy) for energy in energies)
    dofs = [
        RangeDOF(actuator=dcm_c2.roll, bounds=BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS, parameter_type="float"),
        RangeDOF(actuator=m2.yaw, bounds=BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS, parameter_type="float"),
        RangeDOF(actuator=m2.center_x, bounds=BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS, parameter_type="float"),
    ]
    acquisition_plan = cast(
        AcquisitionPlan[str],
        partial(
            acquire_with_energy_scan,
            tpw=tpw,
            dcm_c1=dcm_c1,
            dcm_c2=dcm_c2,
            crystal=crystal,
            energies=energies,
            band=band,
        ),
    )

    agent = Agent(
        sensors=dets,
        dofs=dofs,
        objectives=[Objective(name=ALIGNMENT_SCORE, minimize=True)],
        evaluation_function=energy_alignment_evaluation(
            tiled_client, image_key, target_centroid, energies, threshold=threshold, blur=blur
        ),
        acquisition_plan=acquisition_plan,
        outcome_constraints=BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS,
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
    tiled_client: Container,
    image_key: str,
    target_centroid: tuple[float, float],
    energies: Sequence[float],
    band: float = 1.0,
    threshold: float = DEFAULT_IMAGE_THRESHOLD,
    blur: float = DEFAULT_IMAGE_BLUR,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    document_address: str = QUEUE_SERVER_DOCUMENT_ADDRESS,
) -> QueueserverAgent:
    """Build a Queue Server agent that optimizes detector-screen statistics across an energy range.

    The agent evaluates each queue item once the run it opens has stopped, matching items to runs from the Queue
    Server worker's documents. This starts a daemon thread that receives them from ``document_address``.

    Parameters
    ----------
    re_manager
        RE Manager that runs the acquisition plans.
    dets
        Names of the detectors acquired at each energy for each optimization point.
    crystal_name
        Device name for silicon crystal material supplying the lattice spacing.
    tiled_client
        Tiled container that the Queue Server's ``TiledWriter`` writes to.
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
    document_address
        Address of the 0MQ proxy output that relays the Queue Server worker's documents.

    Returns
    -------
    QueueserverAgent
        Configured Bluesky Adaptive agent for BMM simulator alignment.
    """
    energies = tuple(float(energy) for energy in energies)
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
    runs = QueueItemRuns(re_manager)

    agent = QueueserverAgent(
        re_manager,
        sensors=dets,
        dofs=dofs,
        objectives=[Objective(name=ALIGNMENT_SCORE, minimize=True)],
        evaluation_function=per_run(
            energy_alignment_evaluation(tiled_client, image_key, target_centroid, energies, threshold=threshold, blur=blur),
            runs.resolve,
        ),
        acquisition_plan="acquire_with_energy_scan",
        acquisition_plan_kwargs={
            "tpw": "tpw",
            "dcm_c1": "dcm_c1",
            "dcm_c2": "dcm_c2",
            "crystal": crystal_name,
            "energies": energies,
            "band": band,
        },
        outcome_constraints=BMM_ENERGY_ALIGNMENT_OUTCOME_CONSTRAINTS,
        checkpoint_path=checkpoint_path,
    )
    agent.ax_client.configure_generation_strategy(
        method="fast",
        initialization_budget=BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
        initialize_with_center=False,
    )

    # QueueItemRuns must see the start document of every plan the agent submits, so it listens from now on.
    dispatcher = RemoteDispatcher(document_address)
    dispatcher.subscribe(runs)
    threading.Thread(target=dispatcher.start, name="queue-item-runs", daemon=True).start()
    return agent
