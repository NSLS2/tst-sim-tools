"""Blop agent for energy dependent alignment."""

import time
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import numpy as np
from ax.api.protocols import IMetric
from blop.ax import Agent, Objective, OutcomeConstraint, RangeDOF
from blop.ax.queueserver_agent import QueueserverAgent
from blop.protocols import AcquisitionPlan, EvaluationFunction
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.protocols import Readable
from bluesky_queueserver_api.http import REManagerAPI

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
    ) -> None:
        self._client = tiled_client
        self._image_key = image_key
        self._target_centroid = target_centroid
        self._energies = np.asarray(tuple(float(energy) for energy in energies), dtype=float)
        if self._energies.ndim != 1 or self._energies.size == 0:
            raise ValueError("Expected at least one energy for energy alignment evaluation")
        self._threshold = threshold
        self._blur = blur

    def _poll_for_images(self, uid: str) -> np.ndarray:
        while True:
            try:
                run = self._client[uid]
                stream = run["primary"]
                return stream[self._image_key].read()
            except KeyError:
                time.sleep(0.1)

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        images = self._poll_for_images(uid)
        image_stack = image_series(images)
        suggestion_ids = [suggestion["_id"] for suggestion in self._client[uid].metadata["start"]["blop_suggestions"]]
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
) -> QueueserverAgent:
    """Build a queueserver agent that optimizes detector-screen statistics across an energy range.

    Parameters
    ----------
    dets
        Detector readables acquired at each energy for each optimization point.
    crystal_name
        Device name for silicon crystal material supplying the lattice spacing.
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
    QueueserverAgent
        Configured Blop agent for BMM simulator alignment.
    """
    dofs = [
        RangeDOF(name="dcm_c2-roll", bounds=BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL_BOUNDS, parameter_type="float"),
        RangeDOF(name="m2-yaw", bounds=BMM_ENERGY_ALIGNMENT_TFM_YAW_BOUNDS, parameter_type="float"),
        RangeDOF(name="m2-center_x", bounds=BMM_ENERGY_ALIGNMENT_TFM_LATERAL_BOUNDS, parameter_type="float"),
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

    dispatcher = RemoteDispatcher("127.0.0.1:40851")
    agent = QueueserverAgent(
        re_manager,
        dispatcher,
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
        acquisition_plan=acquisition_plan_name,
        acquisition_plan_kwargs=acquisition_plan_kwargs,
        outcome_constraints=outcome_constraints,
        checkpoint_path=checkpoint_path,
    )
    agent.ax_client.configure_generation_strategy(
        method="fast",
        initialization_budget=BMM_ENERGY_ALIGNMENT_INITIALIZATION_BUDGET,
        initialize_with_center=False,
    )

    return agent
