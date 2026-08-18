"""Blop agent for energy dependent alignment."""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from blop.ax import Agent, Objective, RangeDOF
from blop.protocols import EvaluationFunction
from bluesky.protocols import Readable
from ophyd_async.core import SignalRW

from ..analysis.image import analyze_image, image_series

LATERAL_POSITION_ERROR = "lateral_position_error"
FWHM = "fwhm"
DEFAULT_IMAGE_THRESHOLD = 0.02
DEFAULT_IMAGE_BLUR = 1.0


class EnergyAlignmentEvalutation(EvaluationFunction):
    """Evaluate Bluesky runs for energy alignment from detector-screen images."""

    def __init__(
        self,
        tiled_client: Any,
        image_key: str,
        target_centroid: tuple[float, float],
        threshold: float = DEFAULT_IMAGE_THRESHOLD,
        blur: float = DEFAULT_IMAGE_BLUR,
    ) -> None:
        self._client = tiled_client
        self._image_key = image_key
        self._target_centroid = target_centroid
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
        if image_stack.shape[0] != len(suggestion_ids):
            raise ValueError(f"Expected {len(suggestion_ids)} image(s) from Tiled, but got {images.shape[0]}")

        outcomes = []
        for idx, sid in enumerate(suggestion_ids):
            metrics = analyze_image(image_stack[idx], threshold=self._threshold, blur=self._blur)
            target_x, target_y = self._target_centroid
            outcomes.append(
                {
                    LATERAL_POSITION_ERROR: float(
                        np.hypot(metrics["x_centroid"] - target_x, metrics["y_centroid"] - target_y)
                    ),
                    FWHM: metrics["fwhm_px"],
                    "_id": sid,
                }
            )
        return outcomes


async def build_agent(
    dets: Sequence[Readable],
    dcm_c2_roll: SignalRW,
    tfm_yaw: SignalRW,
    tfm_lateral: SignalRW,
    tiled_client: Any,
    image_key: str,
    target_centroid: tuple[float, float],
    threshold: float = DEFAULT_IMAGE_THRESHOLD,
    blur: float = DEFAULT_IMAGE_BLUR,
    checkpoint_path: str = "/tmp/blop/energy-alignment.json",
) -> Agent:
    """Build an agent that optimizes detector-screen centroid error and beam FWHM.

    Parameters
    ----------
    dets
        Detector readables acquired for each optimization point.
    dcm_c2_roll
        DCM crystal-2 roll actuator.
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
    init_roll, init_yaw, init_lat = await asyncio.gather(
        dcm_c2_roll.get_value(),
        tfm_yaw.get_value(),
        tfm_lateral.get_value(),
    )

    dofs = [
        RangeDOF(actuator=dcm_c2_roll, bounds=(init_roll - 1e-4, init_roll + 1e-4), parameter_type="float"),
        RangeDOF(actuator=tfm_yaw, bounds=(init_yaw - 2.5e-4, init_yaw + 2.5e-4), parameter_type="float"),
        RangeDOF(actuator=tfm_lateral, bounds=(init_lat - 0.5, init_lat + 0.5), parameter_type="float"),
    ]

    objectives = [
        Objective(name=LATERAL_POSITION_ERROR, minimize=True),
        Objective(name=FWHM, minimize=True),
    ]

    agent = Agent(
        sensors=dets,
        dofs=dofs,
        objectives=objectives,
        evaluation_function=EnergyAlignmentEvalutation(
            tiled_client,
            image_key=image_key,
            target_centroid=target_centroid,
            threshold=threshold,
            blur=blur,
        ),
        checkpoint_path=checkpoint_path,
    )
    agent.ax_client.configure_generation_strategy(
        method="fast",
        initialize_with_center=False,
    )

    return agent
