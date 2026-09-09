from collections.abc import Mapping, Sequence
from threading import Event
from typing import Any, cast
from uuid import UUID

import pytest
from blop.protocols import QueueserverOptimizationProblem
from bluesky_queueserver_api import BPlan

from tst_sim_tools.agents.tiled_queueserver import (
    CORRELATION_UID_KEY,
    OptimizationResult,
    TiledQueueAcquisition,
    TiledQueueserverAgent,
    TiledQueueserverClient,
    TiledQueueserverOptimizationRunner,
)


class FakeOptimizer:
    def __init__(self) -> None:
        self.suggest_calls: list[int] = []
        self.suggestion_batches: list[list[dict[str, Any]]] = []
        self.ingested: list[Sequence[Mapping[str, Any]]] = []

    def suggest(self, num_points: int) -> list[dict[str, Any]]:
        iteration = len(self.suggest_calls) + 1
        self.suggest_calls.append(num_points)
        suggestions = [{"x": float(index), "_id": f"trial-{iteration}-{index}"} for index in range(num_points)]
        self.suggestion_batches.append(suggestions)
        return suggestions

    def ingest(self, outcomes: Sequence[Mapping[str, Any]]) -> None:
        self.ingested.append(outcomes)


class FailureAwareOptimizer(FakeOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.failed: list[Sequence[Mapping[str, Any]]] = []
        self.failure_registered = Event()

    def register_failures(self, suggestions: Sequence[Mapping[str, Any]]) -> None:
        self.failed.append(suggestions)
        self.failure_registered.set()


class FakeQueueClient:
    def __init__(self) -> None:
        self.environment_checks = 0
        self.plans: list[BPlan] = []

    def check_environment(self) -> None:
        self.environment_checks += 1

    def submit_plan(self, plan: BPlan) -> str:
        self.plans.append(plan)
        return f"item-{len(self.plans)}"


def make_problem(optimizer: Any, evaluation_function: Any) -> QueueserverOptimizationProblem:
    return QueueserverOptimizationProblem(
        optimizer=optimizer,
        actuators=("motor",),
        sensors=("detector",),
        evaluation_function=evaluation_function,
        acquisition_plan="acquire",
        acquisition_plan_kwargs={"exposure": 0.1},
    )


def make_runner(optimizer: Any, evaluation_function: Any) -> tuple[TiledQueueserverOptimizationRunner, FakeQueueClient]:
    queue_client = FakeQueueClient()
    runner = TiledQueueserverOptimizationRunner(
        make_problem(optimizer, evaluation_function),
        cast(TiledQueueserverClient, queue_client),
    )
    return runner, queue_client


def test_client_configures_autostart_checks_environment_and_submits_plan(mocker) -> None:
    re_manager = mocker.Mock()
    re_manager.queue_autostart.return_value = {"success": True}
    re_manager.status.return_value = {"worker_environment_exists": True}
    re_manager.item_add.return_value = {"success": True, "item": {"item_uid": "item-uid"}}
    plan = BPlan("count", ["detector"])

    client = TiledQueueserverClient(re_manager, autostart=False)
    client.check_environment()
    item_uid = client.submit_plan(plan)

    assert client.re_manager is re_manager
    assert item_uid == "item-uid"
    re_manager.queue_autostart.assert_called_once_with(False)
    re_manager.status.assert_called_once_with()
    re_manager.item_add.assert_called_once_with(plan)


def test_client_rejects_closed_worker_environment(mocker) -> None:
    re_manager = mocker.Mock()
    re_manager.queue_autostart.return_value = {"success": True}
    re_manager.status.return_value = {"worker_environment_exists": False}
    client = TiledQueueserverClient(re_manager)

    with pytest.raises(RuntimeError, match="queueserver environment is not open"):
        client.check_environment()


def test_agent_closes_evaluation_resources(mocker) -> None:
    evaluation = mocker.Mock()
    agent = object.__new__(TiledQueueserverAgent)
    agent._evaluation_function = evaluation

    agent.close()

    evaluation.close.assert_called_once_with()


def test_runner_completes_suggest_submit_evaluate_ingest_loop() -> None:
    optimizer = FakeOptimizer()
    acquisitions: list[TiledQueueAcquisition] = []

    def evaluate(acquisition: TiledQueueAcquisition, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        acquisitions.append(acquisition)
        acquisition.run_uid = f"run-{len(acquisitions)}"
        return [{"_id": suggestion["_id"], "score": float(index)} for index, suggestion in enumerate(suggestions)]

    runner, queue_client = make_runner(optimizer, evaluate)

    result = runner.run(iterations=2, num_points=2).result(timeout=2)

    assert result == OptimizationResult(iterations_completed=2, num_points=2, uids=("run-1", "run-2"))
    assert runner.current_iteration == 2
    assert queue_client.environment_checks == 1
    assert optimizer.suggest_calls == [2, 2]
    assert len(optimizer.ingested) == 2
    assert len(queue_client.plans) == 2
    for index, (plan, acquisition, suggestions) in enumerate(
        zip(queue_client.plans, acquisitions, optimizer.suggestion_batches, strict=True), start=1
    ):
        payload = plan.to_dict()
        metadata = payload["kwargs"]["md"]
        assert payload["name"] == "acquire"
        assert payload["args"] == [suggestions, ["motor"], ["detector"]]
        assert payload["kwargs"]["exposure"] == 0.1
        assert metadata["blop_suggestions"] == suggestions
        assert metadata[CORRELATION_UID_KEY] == acquisition.correlation_uid
        assert acquisition.item_uid == f"item-{index}"
        assert UUID(acquisition.correlation_uid).version == 4


def test_runner_rejects_overlap_and_stops_after_inflight_evaluation() -> None:
    optimizer = FakeOptimizer()
    evaluation_started = Event()
    release_evaluation = Event()

    def evaluate(acquisition: TiledQueueAcquisition, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        acquisition.run_uid = "run-1"
        evaluation_started.set()
        if not release_evaluation.wait(timeout=2):
            raise TimeoutError("test did not release evaluation")
        return [{"_id": suggestion["_id"], "score": 1.0} for suggestion in suggestions]

    runner, _queue_client = make_runner(optimizer, evaluate)
    future = runner.run(iterations=3)
    assert evaluation_started.wait(timeout=2)

    with pytest.raises(RuntimeError, match="already running"):
        runner.run()

    runner.stop()
    assert future.done() is False
    release_evaluation.set()

    assert future.result(timeout=2) == OptimizationResult(iterations_completed=1, num_points=1, uids=("run-1",))
    assert optimizer.suggest_calls == [1]


def test_runner_propagates_evaluation_failure_and_registers_suggestions() -> None:
    optimizer = FailureAwareOptimizer()
    error = RuntimeError("evaluation failed")

    def evaluate(acquisition: TiledQueueAcquisition, suggestions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        raise error

    runner, _queue_client = make_runner(optimizer, evaluate)

    with pytest.raises(RuntimeError) as caught:
        runner.run().result(timeout=2)

    assert caught.value is error
    assert optimizer.failure_registered.wait(timeout=2)
    assert optimizer.failed == optimizer.suggestion_batches
