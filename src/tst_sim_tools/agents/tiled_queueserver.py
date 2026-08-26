"""Queue Server optimization runner that delegates completion waits to evaluation functions."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from blop.ax.dof import DOF, DOFConstraint
from blop.ax.objective import Objective, OutcomeConstraint, to_ax_objective_str
from blop.ax.optimizer import AxOptimizer
from blop.protocols import (
    ID_KEY,
    CanRegisterSuggestions,
    Checkpointable,
    EvaluationFunction,
    QueueserverOptimizationProblem,
    TrialFaultAware,
)
from bluesky_queueserver_api import BPlan

logger = logging.getLogger(__name__)

DEFAULT_ACQUIRE_PLAN_NAME = "default_acquire"
CORRELATION_UID_KEY = "blop_correlation_uid"


@dataclass(slots=True)
class TiledQueueAcquisition:
    """Identifier for one Queue Server item whose data will appear in Tiled."""

    correlation_uid: str
    item_uid: str | None = None
    plan_name: str | None = None
    run_uid: str | None = None

    def __hash__(self) -> int:
        """Hash by immutable submission identifiers."""
        return hash((self.correlation_uid, self.item_uid, self.plan_name))


@dataclass(frozen=True)
class OptimizationResult:
    """Summary returned when a Tiled-backed Queue Server optimization finishes."""

    iterations_completed: int
    num_points: int
    uids: tuple[str, ...]


@dataclass
class _OptimizationState:
    max_iterations: int = 1
    num_points: int = 1
    checkpoint_interval: int | None = None
    current_iteration: int = 0
    current_suggestions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    current_acquisition: TiledQueueAcquisition | None = None
    uids: list[str] = field(default_factory=list)

    def build_result(self) -> OptimizationResult:
        return OptimizationResult(
            iterations_completed=len(self.uids),
            num_points=self.num_points,
            uids=tuple(self.uids),
        )


class TiledQueueserverClient:
    """Thin Queue Server HTTP/ZMQ client wrapper with no document listener."""

    def __init__(self, re_manager_api: Any, *, autostart: bool = True) -> None:
        self._rm = re_manager_api
        response = self._rm.queue_autostart(autostart)
        logger.debug("Set queue autostart to %s. Response: %s", autostart, response)

    @property
    def re_manager(self) -> Any:
        """Underlying Queue Server API object."""
        return self._rm

    def check_environment(self) -> None:
        """Verify that the Queue Server worker environment is open."""
        status = self._rm.status()
        if status is None or not status.get("worker_environment_exists", False):
            raise RuntimeError("The queueserver environment is not open")

    def submit_plan(self, plan: BPlan) -> str | None:
        """Submit a plan and return the Queue Server item UID when available."""
        response = self._rm.item_add(plan)
        logger.debug("Submitted plan to queue. Response: %s", response)
        item = response.get("item", {}) if isinstance(response, Mapping) else {}
        item_uid = item.get("item_uid")
        return str(item_uid) if item_uid is not None else None


class TiledQueueserverOptimizationRunner:
    """Run an optimizer loop through Queue Server while Tiled supplies completion semantics."""

    def __init__(
        self,
        optimization_problem: QueueserverOptimizationProblem,
        queueserver_client: TiledQueueserverClient,
    ) -> None:
        self._problem = optimization_problem
        self._client = queueserver_client
        self._plan_name = optimization_problem.acquisition_plan or DEFAULT_ACQUIRE_PLAN_NAME
        self._state: _OptimizationState | None = None
        self._continuous = True
        self._state_lock = threading.RLock()
        self._current_future: Future[OptimizationResult] | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def optimization_problem(self) -> QueueserverOptimizationProblem:
        """Immutable optimization problem definition."""
        return self._problem

    @property
    def current_iteration(self) -> int:
        """Current one-based iteration number, or zero when idle."""
        with self._state_lock:
            return self._state.current_iteration if self._state else 0

    @property
    def re_manager(self) -> Any:
        """Underlying Queue Server API object."""
        return self._client.re_manager

    def run(
        self, iterations: int = 1, num_points: int = 1, checkpoint_interval: int | None = None
    ) -> Future[OptimizationResult]:
        """Start a suggest/acquire/evaluate/ingest loop in a worker thread."""
        return self._start_worker(
            max_iterations=iterations,
            num_points=num_points,
            checkpoint_interval=checkpoint_interval,
            initial_suggestions=None,
            continuous=True,
        )

    def submit_suggestions(self, suggestions: Sequence[Mapping[str, Any]]) -> Future[OptimizationResult]:
        """Submit explicit suggestions and ingest exactly one evaluated acquisition."""
        suggestions = list(suggestions)
        if not isinstance(self._problem.optimizer, CanRegisterSuggestions) and any(
            ID_KEY not in suggestion for suggestion in suggestions
        ):
            raise ValueError(
                f"All suggestions must contain an '{ID_KEY}' key to later match with the outcomes or your optimizer must "
                "implement the `blop.protocols.CanRegisterSuggestions` protocol. Please review your optimizer "
                f"implementation. Got suggestions: {suggestions}"
            )
        if isinstance(self._problem.optimizer, CanRegisterSuggestions):
            suggestions = list(self._problem.optimizer.register_suggestions(suggestions))

        return self._start_worker(
            max_iterations=1,
            num_points=len(suggestions),
            checkpoint_interval=None,
            initial_suggestions=suggestions,
            continuous=False,
        )

    def stop(self) -> None:
        """Stop after the in-flight evaluation returns and resolve the current future with partial results."""
        with self._state_lock:
            self._stop_event.set()
            result = (
                self._state.build_result()
                if self._state is not None
                else OptimizationResult(iterations_completed=0, num_points=0, uids=())
            )
            self._resolve_future(result)
        logger.info("Tiled-backed Queue Server optimization stopped")

    def _start_worker(
        self,
        *,
        max_iterations: int,
        num_points: int,
        checkpoint_interval: int | None,
        initial_suggestions: Sequence[Mapping[str, Any]] | None,
        continuous: bool,
    ) -> Future[OptimizationResult]:
        with self._state_lock:
            self._validate()
            self._state = _OptimizationState(
                max_iterations=max_iterations,
                num_points=num_points,
                checkpoint_interval=checkpoint_interval,
            )
            self._continuous = continuous
            self._stop_event.clear()
            future: Future[OptimizationResult] = Future()
            self._current_future = future
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(initial_suggestions,),
                name="tiled-qserver-optimizer",
                daemon=True,
            )
            self._worker_thread.start()
            return future

    def _validate(self) -> None:
        if self._current_future is not None and not self._current_future.done():
            raise RuntimeError("Optimization loop is already running.")
        self._client.check_environment()

    def _resolve_future(self, result: OptimizationResult) -> None:
        if self._current_future is not None and not self._current_future.done():
            self._current_future.set_result(result)

    def _fail_future(self, exc: Exception) -> None:
        if self._current_future is not None and not self._current_future.done():
            self._current_future.set_exception(exc)

    def _try_register_failures(self, suggestions: Sequence[Mapping[str, Any]]) -> None:
        if suggestions and isinstance(self._problem.optimizer, TrialFaultAware):
            try:
                self._problem.optimizer.register_failures(suggestions)
            except Exception:
                logger.exception("Failed to register trial failures with the optimizer")

    def _build_plan(self, suggestions: Sequence[Mapping[str, Any]]) -> tuple[BPlan, TiledQueueAcquisition]:
        if self._state is None:
            raise RuntimeError("_build_plan() called before run() or submit_suggestions()")

        self._state.current_iteration += 1
        self._state.current_suggestions = suggestions
        acquisition = TiledQueueAcquisition(correlation_uid=str(uuid.uuid4()), plan_name=self._plan_name)
        self._state.current_acquisition = acquisition

        md: dict[str, Any] = {
            CORRELATION_UID_KEY: acquisition.correlation_uid,
            "blop_suggestions": suggestions,
        }
        plan = BPlan(
            self._plan_name,
            suggestions,
            list(self._problem.actuators),
            list(self._problem.sensors),
            md=md,
            **(self._problem.acquisition_plan_kwargs or {}),
        )
        return plan, acquisition

    def _worker_loop(self, initial_suggestions: Sequence[Mapping[str, Any]] | None) -> None:
        suggestions = initial_suggestions
        try:
            while True:
                with self._state_lock:
                    if self._state is None:
                        raise RuntimeError("Optimization worker started without state")
                    if self._stop_event.is_set():
                        self._resolve_future(self._state.build_result())
                        return
                    num_points = self._state.num_points

                if suggestions is None:
                    suggestions = self._problem.optimizer.suggest(num_points)

                with self._state_lock:
                    plan, acquisition = self._build_plan(suggestions)
                    iteration = self._state.current_iteration if self._state is not None else 0
                    max_iterations = self._state.max_iterations if self._state is not None else 0
                    logger.info(
                        "Submitting iteration %s/%s with Tiled correlation uid: %s",
                        iteration,
                        max_iterations,
                        acquisition.correlation_uid,
                    )

                acquisition.item_uid = self._client.submit_plan(plan)
                outcomes = self._problem.evaluation_function(acquisition, suggestions)
                logger.info("Evaluated %s outcomes", len(outcomes))
                self._problem.optimizer.ingest(outcomes)

                with self._state_lock:
                    if self._state is None:
                        raise RuntimeError("Optimization state disappeared during evaluation")
                    self._state.uids.append(acquisition.run_uid or acquisition.correlation_uid)
                    self._maybe_checkpoint(iteration=self._state.current_iteration)
                    if (
                        self._stop_event.is_set()
                        or not self._continuous
                        or self._state.current_iteration >= self._state.max_iterations
                    ):
                        logger.info("Optimization complete after %s iterations", self._state.current_iteration)
                        self._resolve_future(self._state.build_result())
                        return

                suggestions = None
        except Exception as exc:
            logger.exception(
                "Unhandled exception in Tiled-backed optimization worker. Optimization has been stopped."
            )
            with self._state_lock:
                failed_suggestions = self._state.current_suggestions if self._state is not None else ()
                self._fail_future(exc)
            self._try_register_failures(failed_suggestions)

    def _maybe_checkpoint(self, *, iteration: int) -> None:
        if self._state is None or self._state.checkpoint_interval is None:
            return
        if iteration % self._state.checkpoint_interval != 0:
            return
        if not isinstance(self._problem.optimizer, Checkpointable):
            raise ValueError(
                "The optimizer is not checkpointable. Please review your optimizer configuration or implementation."
            )
        self._problem.optimizer.checkpoint()


class TiledQueueserverAgent:
    """Ax-backed Blop-style agent that uses Tiled websocket-aware evaluation functions."""

    def __init__(
        self,
        re_manager_api: Any,
        sensors: Sequence[str],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        evaluation_function: EvaluationFunction,
        acquisition_plan: str | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
        acquisition_plan_kwargs: Mapping[str, Any] | None = None,
        *,
        autostart: bool = True,
        **kwargs: Any,
    ) -> None:
        self._sensors = sensors
        self._actuators: list[str] = []
        for dof in dofs:
            actuator = dof.actuator
            if actuator is None:
                continue
            self._actuators.append(actuator if isinstance(actuator, str) else actuator.name)
        self._evaluation_function = evaluation_function
        self._acquisition_plan = acquisition_plan
        self._acquisition_plan_kwargs = acquisition_plan_kwargs or {}
        self._optimizer = AxOptimizer(
            parameters=[dof.to_ax_parameter_config() for dof in dofs],
            objective=to_ax_objective_str(objectives),
            parameter_constraints=[constraint.ax_constraint for constraint in dof_constraints] if dof_constraints else None,
            outcome_constraints=[constraint.ax_constraint for constraint in outcome_constraints]
            if outcome_constraints
            else None,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self._runner = TiledQueueserverOptimizationRunner(
            self.to_optimization_problem(),
            TiledQueueserverClient(re_manager_api, autostart=autostart),
        )

    @property
    def ax_client(self) -> Any:
        """Underlying Ax client."""
        return self._optimizer.ax_client

    @property
    def checkpoint_path(self) -> str | None:
        """Optimizer checkpoint path."""
        return self._optimizer.checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str) -> None:
        self._optimizer.checkpoint_path = value

    @property
    def evaluation_function(self) -> EvaluationFunction:
        """Evaluation function."""
        return self._evaluation_function

    @property
    def actuators(self) -> Sequence[str]:
        """Actuator names used in submitted plans."""
        return self._actuators

    @property
    def sensors(self) -> Sequence[str]:
        """Sensor names used in submitted plans."""
        return self._sensors

    @property
    def acquisition_plan(self) -> str | None:
        """Acquisition plan name."""
        return self._acquisition_plan

    @property
    def current_iteration(self) -> int:
        """Current runner iteration."""
        return self._runner.current_iteration

    def suggest(self, num_points: int = 1) -> Sequence[Mapping[str, Any]]:
        """Ask the optimizer for candidate parameterizations."""
        return self._optimizer.suggest(num_points)

    def ingest(self, points: Sequence[Mapping[str, Any]]) -> None:
        """Ingest evaluated outcome mappings into the optimizer."""
        self._optimizer.ingest(points)

    def checkpoint(self) -> None:
        """Write an optimizer checkpoint."""
        self._optimizer.checkpoint()

    def stop(self) -> None:
        """Stop the queueserver runner after the in-flight evaluation."""
        self._runner.stop()

    def to_optimization_problem(self) -> QueueserverOptimizationProblem:
        """Convert this agent to Blop's Queue Server optimization problem dataclass."""
        return QueueserverOptimizationProblem(
            optimizer=self._optimizer,
            actuators=self._actuators,
            sensors=self._sensors,
            evaluation_function=self._evaluation_function,
            acquisition_plan=self._acquisition_plan,
            acquisition_plan_kwargs=self._acquisition_plan_kwargs,
        )

    def run(
        self, iterations: int = 1, n_points: int = 1, checkpoint_interval: int | None = None
    ) -> Future[OptimizationResult]:
        """Run the optimization loop asynchronously."""
        return self._runner.run(iterations=iterations, num_points=n_points, checkpoint_interval=checkpoint_interval)

    def submit_suggestions(self, suggestions: Sequence[Mapping[str, Any]]) -> Future[OptimizationResult]:
        """Evaluate explicit suggestions asynchronously."""
        return self._runner.submit_suggestions(suggestions)
