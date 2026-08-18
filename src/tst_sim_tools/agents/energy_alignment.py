"""Blop agent for energy dependent alignment."""

import asyncio
import time
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import numpy as np
from blop.ax import Agent, Objective, RangeDOF
from blop.protocols import AcquisitionPlan, EvaluationFunction
from bluesky.protocols import Readable
from ophyd_async.core import SignalRW

from ..analysis.image import analyze_energy_scan, image_series
from ..devices.materials import XRTCrystalSi
from ..devices.mirrors import XRTOpticalElement
from ..devices.sources import XRTWiggler
from ..plans.bmm import acquire_with_energy_scan

LATERAL_POSITION_ERROR = "lateral_position_error"
FWHM = "fwhm"
LATERAL_POSITION_RANGE = "lateral_position_range"
VERTICAL_POSITION_ERROR = "vertical_position_error"
MAX_FWHM = "max_fwhm"
MEAN_INTENSITY = "mean_intensity"
MIN_INTENSITY = "min_intensity"
DEFAULT_IMAGE_THRESHOLD = 0.02
DEFAULT_IMAGE_BLUR = 1.0
BMM_ENERGY_ALIGNMENT_REFERENCE_ENERGY = 7112.0
BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL = 5e-5
BMM_ENERGY_ALIGNMENT_TFM_LATERAL = 0.33
BMM_ENERGY_ALIGNMENT_TFM_YAW = 0.0


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
            outcomes.append(
                {
                    LATERAL_POSITION_ERROR: float(np.sqrt(np.mean(x_error**2))),
                    LATERAL_POSITION_RANGE: float(np.ptp(summary["x_centroid"])),
                    VERTICAL_POSITION_ERROR: float(np.sqrt(np.mean(y_error**2))),
                    FWHM: float(np.mean(summary["fwhm_px"])),
                    MAX_FWHM: float(np.max(summary["fwhm_px"])),
                    MEAN_INTENSITY: float(np.mean(summary["total"])),
                    MIN_INTENSITY: float(np.min(summary["total"])),
                    "_id": sid,
                }
            )
        return outcomes


async def build_agent(
    dets: Sequence[Readable],
    tpw: XRTWiggler,
    dcm_c1: XRTOpticalElement,
    dcm_c2: XRTOpticalElement,
    crystal: XRTCrystalSi,
    tfm_yaw: SignalRW,
    tfm_lateral: SignalRW,
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
    tfm_yaw
        Toroid mirror yaw actuator.
    tfm_lateral
        Toroid mirror lateral actuator.
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
    await asyncio.gather(
        dcm_c2.roll.set(BMM_ENERGY_ALIGNMENT_DCM_C2_ROLL),
        tfm_lateral.set(BMM_ENERGY_ALIGNMENT_TFM_LATERAL),
        tfm_yaw.set(BMM_ENERGY_ALIGNMENT_TFM_YAW),
    )
    init_roll, init_yaw, init_lat = await asyncio.gather(
        dcm_c2.roll.get_value(),
        tfm_yaw.get_value(),
        tfm_lateral.get_value(),
    )

    dofs = [
        RangeDOF(actuator=dcm_c2.roll, bounds=(init_roll - 1e-4, init_roll + 1e-4), parameter_type="float"),
        RangeDOF(actuator=tfm_yaw, bounds=(init_yaw - 2.5e-4, init_yaw + 2.5e-4), parameter_type="float"),
        RangeDOF(actuator=tfm_lateral, bounds=(init_lat - 0.5, init_lat + 0.5), parameter_type="float"),
    ]

    objectives = [
        Objective(name=LATERAL_POSITION_ERROR, minimize=True),
        Objective(name=FWHM, minimize=True),
    ]

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
        checkpoint_path=checkpoint_path,
    )
    agent.ax_client.configure_generation_strategy(
        method="fast",
        initialize_with_center=False,
    )

    return agent
