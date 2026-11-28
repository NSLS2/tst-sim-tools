from typing import Any, cast

import numpy as np
import pytest
from tiled.client.container import Container

from tst_sim_tools.agents.energy_alignment import (
    ALIGNMENT_SCORE,
    CENTROID_RMS_ERROR,
    CENTROID_SPAN,
    FWHM,
    LATERAL_POSITION_ERROR,
    MAX_CENTROID_ERROR,
    MAX_FWHM,
    MIN_INTENSITY,
    VERTICAL_POSITION_ERROR,
    TiledEnergyScanImages,
    energy_alignment_evaluation,
)

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
SUGGESTIONS = [{"_id": "trial-0"}, {"_id": "trial-1"}]


class FakeField:
    def __init__(self, images: np.ndarray) -> None:
        self.images = images

    def read(self) -> np.ndarray:
        return self.images


class FakeRun:
    def __init__(self, images: np.ndarray, acquisition_order: list[str], *, stopped: bool = True) -> None:
        self.metadata: dict[str, Any] = {
            "start": {"bluesky-adaptive": {"acquisition_order": acquisition_order}},
            "stop": {"exit_status": "success"} if stopped else None,
        }
        self.primary = {"screen_image": FakeField(images)}

    def __getitem__(self, key: str) -> dict[str, FakeField]:
        if key != "primary":
            raise KeyError(key)
        return self.primary


class EventuallyStoppedClient:
    """Return each of ``states`` for the run in turn, then keep returning the last one."""

    def __init__(self, *states: FakeRun | None) -> None:
        self.states = list(states)
        self.calls = 0

    def get(self, uid: str) -> FakeRun | None:
        state = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return state


def alignment_images() -> np.ndarray:
    images = np.zeros((4, 1, 5, 5), dtype=float)
    images[0, 0, 2, 2] = 20_000.0
    images[1, 0, 2, 3] = 20_000.0
    images[2, 0, 1, 2] = 18_000.0
    images[3, 0, 3, 2] = 18_000.0
    return images


def make_evaluation(run: FakeRun):
    return energy_alignment_evaluation(
        cast(Container, EventuallyStoppedClient(run)),
        "screen_image",
        target_centroid=(2.0, 2.0),
        energies=(7000.0, 7100.0),
        threshold=0.0,
        blur=0.0,
    )


def test_energy_alignment_evaluation_rejects_empty_energies() -> None:
    with pytest.raises(ValueError, match="Expected at least one energy for energy alignment evaluation"):
        energy_alignment_evaluation(cast(Container, EventuallyStoppedClient(None)), "screen_image", (2.0, 2.0), [])


def test_energy_alignment_evaluation_scores_suggestions_in_acquisition_order() -> None:
    # The acquisition plan may reorder suggestions, so the first two images belong to trial-1
    evaluation = make_evaluation(FakeRun(alignment_images(), ["trial-1", "trial-0"]))

    outcomes = evaluation("run-uid", SUGGESTIONS)

    assert [tuple(outcome) for outcome in outcomes] == [OUTCOME_KEYS, OUTCOME_KEYS]
    assert outcomes[0]["_id"] == "trial-1"
    assert outcomes[0][CENTROID_RMS_ERROR] == pytest.approx(np.sqrt(0.5))
    assert outcomes[0][MAX_CENTROID_ERROR] == 1.0
    assert outcomes[0][CENTROID_SPAN] == 1.0
    assert outcomes[0][LATERAL_POSITION_ERROR] == pytest.approx(np.sqrt(0.5))
    assert outcomes[0][VERTICAL_POSITION_ERROR] == 0.0
    assert outcomes[0][FWHM] == 1.0
    assert outcomes[0][MAX_FWHM] == 1.0
    assert outcomes[0][MIN_INTENSITY] == 20_000.0

    assert outcomes[1]["_id"] == "trial-0"
    assert outcomes[1][CENTROID_RMS_ERROR] == 1.0
    assert outcomes[1][MAX_CENTROID_ERROR] == 1.0
    assert outcomes[1][CENTROID_SPAN] == 2.0
    assert outcomes[1][LATERAL_POSITION_ERROR] == 0.0
    assert outcomes[1][VERTICAL_POSITION_ERROR] == 1.0
    assert outcomes[1][FWHM] == 1.0
    assert outcomes[1][MAX_FWHM] == 1.0
    assert outcomes[1][MIN_INTENSITY] == 18_000.0


def test_energy_alignment_evaluation_applies_weighted_score() -> None:
    evaluation = make_evaluation(FakeRun(alignment_images(), ["trial-0", "trial-1"]))

    outcomes = evaluation("run-uid", SUGGESTIONS)

    assert outcomes[0][ALIGNMENT_SCORE] == pytest.approx(np.sqrt(0.5) + 0.25 + 0.1 + 0.005 + 0.0025)
    assert outcomes[1][ALIGNMENT_SCORE] == pytest.approx(1.0 + 0.25 + 0.2 + 0.005 + 0.0025)


def test_energy_alignment_evaluation_validates_image_count() -> None:
    evaluation = make_evaluation(FakeRun(alignment_images()[:3], ["trial-0", "trial-1"]))

    with pytest.raises(ValueError) as caught:
        evaluation("run-uid", SUGGESTIONS)

    assert str(caught.value) == "Expected 4 image(s) from Tiled for 2 suggestion(s) and 2 energy point(s), but got 3"


def test_tiled_energy_scan_images_waits_for_the_stop_document(mocker) -> None:
    images = alignment_images()[:2]
    client = EventuallyStoppedClient(
        None,
        FakeRun(images, ["trial-0"], stopped=False),
        FakeRun(images, ["trial-0"]),
    )
    sleep = mocker.patch("tst_sim_tools.agents.energy_alignment.time.sleep")

    stacks = TiledEnergyScanImages(cast(Container, client), "screen_image", 2)("run-uid")

    assert client.calls == 3
    assert sleep.call_count == 2
    assert list(stacks) == ["trial-0"]
    np.testing.assert_array_equal(stacks["trial-0"], images[:, 0])


def test_tiled_energy_scan_images_times_out_without_a_stop_document() -> None:
    client = EventuallyStoppedClient(FakeRun(alignment_images()[:2], ["trial-0"], stopped=False))
    fetch = TiledEnergyScanImages(cast(Container, client), "screen_image", 2, timeout=0.0)

    with pytest.raises(TimeoutError, match="Run 'run-uid' has no stop document in Tiled after 0 seconds"):
        fetch("run-uid")
