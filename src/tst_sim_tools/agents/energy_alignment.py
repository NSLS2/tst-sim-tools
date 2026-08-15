"""Blop agent for energy dependent alignment."""

from collections.abc import Sequence

from bluesky.protocols import Readable
from blop.ax import Agent, RangeDOF, Objective
from blop.protocols import EvaluationFunction
from ophyd_async.core import SignalRW


LATERAL_POSITION_ERROR = "lateral_position_error"
FWHM = "fwhm"


class EnergyAlignmentEvalutation(EvaluationFunction):
    """Evaluate Bluesky runs for energy alignment."""

    def __init__(self, tiled_client) -> None:
        self._client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:

        return [
            {
                LATERAL_POSITION_ERROR: 0.0,
                FWHM: 0.0,
                "_id": suggestion.get("_id"),
            }
            for suggestion in suggestions
        ]


def build_agent(
    dets: Sequence[Readable],
    dcm_c2_roll: SignalRW,
    tfm_yaw: SignalRW,
    tfm_lateral: SignalRW,
    tiled_client,
    checkpoint_path: str = "/tmp/blop/energy-alignment.json",
) -> Agent:
    """Build an agent that optimizes lateral position error and fwhm of the beam."""
    dofs = [
        RangeDOF(actuator=dcm_c2_roll, bounds=(-1e-4, 1e-4), parameter_type="float"),
        RangeDOF(actuator=tfm_yaw, bounds=(-2.5e-4, 2.5e-4), parameter_type="float"),
        RangeDOF(actuator=tfm_lateral, bounds=(-0.5, 0.5), parameter_type="float"),
    ]

    objectives = [
        Objective(name=LATERAL_POSITION_ERROR, minimize=True),
        Objective(name=FWHM, minimize=False),
    ]

    agent = Agent(
        sensors=dets,
        dofs=dofs,
        objectives=objectives,
        evaluation_function=EnergyAlignmentEvalutation(tiled_client),
        checkpoint_path=checkpoint_path,
    )
    agent.ax_client.configure_generation_strategy(
        method="fast",
        initialize_with_center=False,
    )

    return agent
