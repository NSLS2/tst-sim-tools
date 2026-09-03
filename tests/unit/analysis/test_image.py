import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from tst_sim_tools.analysis.image import (
    analyze_energy_scan,
    analyze_image,
    fwhm,
    gaussian_blur,
    image_series,
    locate_centroid,
    preprocess,
    scroll_images,
    threshold_image,
)

METRIC_KEYS = (
    "total",
    "peak",
    "x_centroid",
    "y_centroid",
    "centroid_radius_px",
    "x_sigma_px",
    "y_sigma_px",
    "x_fwhm_px",
    "y_fwhm_px",
    "fwhm_px",
)


def test_image_series_converts_two_dimensional_image_to_one_frame() -> None:
    image = np.arange(6).reshape(2, 3)

    result = image_series(image)

    assert result.shape == (1, 2, 3)
    assert result.dtype == float
    np.testing.assert_array_equal(result[0], image)


def test_image_series_preserves_three_dimensional_stack() -> None:
    stack = np.arange(12).reshape(2, 2, 3)

    result = image_series(stack)

    np.testing.assert_array_equal(result, stack)


def test_image_series_reduces_external_image_planes() -> None:
    singleton_planes = np.arange(12).reshape(2, 1, 2, 3)
    multiple_planes = np.stack((singleton_planes[:, 0], singleton_planes[:, 0] + 10), axis=1)

    np.testing.assert_array_equal(image_series(singleton_planes), singleton_planes[:, 0])
    np.testing.assert_array_equal(image_series(multiple_planes), multiple_planes.sum(axis=1))


def test_image_series_returns_owned_finite_float_data() -> None:
    stack = np.array([[[np.nan, np.inf], [-np.inf, 2.0]]])

    result = image_series(stack)

    np.testing.assert_array_equal(result, [[[0.0, 0.0], [0.0, 2.0]]])
    assert result.dtype == float
    assert not np.shares_memory(result, stack)


def test_image_series_rejects_empty_stack() -> None:
    with pytest.raises(ValueError, match="Expected at least one image"):
        image_series(np.empty((0, 2, 3)))


@pytest.mark.parametrize("shape", [(2,), (1, 1, 1, 1, 1)])
def test_image_series_rejects_invalid_rank(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=f"got {len(shape)} dimensions"):
        image_series(np.zeros(shape))


def test_fwhm_measures_peak_width() -> None:
    assert fwhm([0, 1, 2, 1, 0]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "profile",
    [
        [-2, 0, -1],
        [2, 2, 2],
        [1, 2, 1],
        [0, 1, 2],
    ],
)
def test_fwhm_returns_finite_penalty_for_unmeasurable_profiles(profile: list[float]) -> None:
    assert fwhm(profile) == float(len(profile))


def test_fwhm_selects_highest_peak() -> None:
    assert fwhm([0, 1, 1, 1, 0, 0, 4, 0]) == pytest.approx(1.0)


def test_threshold_image_applies_fraction_without_mutating_input() -> None:
    image = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, -1.0]])
    original_bytes = image.tobytes()

    result = threshold_image(image, threshold=0.5)

    np.testing.assert_array_equal(result, [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    assert image.tobytes() == original_bytes
    assert not np.shares_memory(result, image)


def test_threshold_image_handles_boundary_and_nonpositive_images() -> None:
    np.testing.assert_array_equal(threshold_image([[1.0, 2.0]], threshold=1.0), [[0.0, 0.0]])
    np.testing.assert_array_equal(threshold_image([[-2.0, 0.0]], threshold=0.0), [[0.0, 0.0]])


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_image_rejects_out_of_range_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"Expected threshold in \[0, 1\]"):
        threshold_image([[1.0]], threshold=threshold)


def test_gaussian_blur_zero_returns_owned_finite_copy() -> None:
    image = np.array([[np.nan, np.inf], [-np.inf, 2.0]])

    result = gaussian_blur(image, sigma=0.0)

    np.testing.assert_array_equal(result, [[0.0, 0.0], [0.0, 2.0]])
    assert not np.shares_memory(result, image)


def test_gaussian_blur_preserves_constant_image() -> None:
    image = np.full((3, 3), 7.0)

    result = gaussian_blur(image, sigma=1.0, truncate=2.0)

    np.testing.assert_allclose(result, image)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize(
    ("sigma", "truncate", "message"),
    [
        (-0.01, 4.0, "Expected non-negative sigma"),
        (1.0, 0.0, "Expected positive truncate"),
        (1.0, -1.0, "Expected positive truncate"),
    ],
)
def test_gaussian_blur_rejects_invalid_parameters(sigma: float, truncate: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gaussian_blur([[1.0]], sigma=sigma, truncate=truncate)


def test_preprocess_uses_threshold_then_blur() -> None:
    image = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, -1.0]])

    result = preprocess(image, threshold=0.5, blur=0.0)

    np.testing.assert_array_equal(result, [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    np.testing.assert_array_equal(image, [[0.0, 1.0, 2.0], [3.0, 4.0, -1.0]])


def test_scroll_images_displays_raw_and_processed_frames_and_wraps() -> None:
    stack = np.array(
        [
            [[0.0, 1.0], [3.0, 4.0]],
            [[0.0, 0.0], [2.0, 8.0]],
        ]
    )
    original = stack.copy()

    figure = scroll_images(stack, threshold=0.5, show=False)
    raw_axis, processed_axis = figure.axes
    raw_artist = raw_axis.images[0]
    processed_artist = processed_axis.images[0]

    np.testing.assert_array_equal(raw_artist.get_array(), stack[0])
    np.testing.assert_array_equal(processed_artist.get_array(), [[0.0, 0.0], [3.0, 4.0]])
    assert raw_axis.get_title() == "Raw"
    assert processed_axis.get_title() == "Processed (threshold=0.5, blur=0)"
    assert raw_artist.get_clim() == (0.0, 4.0)
    assert processed_artist.get_clim() == (0.0, 4.0)

    scroll_up = MouseEvent("scroll_event", figure.canvas, 1, 1, button="up", step=1)
    figure.canvas.callbacks.process("scroll_event", scroll_up)
    np.testing.assert_array_equal(raw_artist.get_array(), stack[1])
    np.testing.assert_array_equal(processed_artist.get_array(), [[0.0, 0.0], [0.0, 8.0]])
    assert len(figure.texts) == 1
    assert figure.texts[0].get_text() == "Image 2 / 2; scroll to change frame"

    figure.canvas.callbacks.process("scroll_event", scroll_up)
    np.testing.assert_array_equal(raw_artist.get_array(), stack[0])
    assert figure.texts[0].get_text() == "Image 1 / 2; scroll to change frame"
    np.testing.assert_array_equal(stack, original)


def test_scroll_images_show_calls_matplotlib_once(mocker) -> None:
    show = mocker.patch("tst_sim_tools.analysis.image.plt.show")

    scroll_images([[1.0]], show=True)

    show.assert_called_once_with()


def test_locate_centroid_returns_weighted_xy_coordinates() -> None:
    assert locate_centroid([[0.0, 0.0], [1.0, 3.0]]) == pytest.approx((0.75, 1.0))


def test_locate_centroid_rejects_image_without_positive_signal() -> None:
    with pytest.raises(ValueError, match="no positive signal"):
        locate_centroid(np.zeros((2, 2)))


def test_analyze_image_reports_complete_centered_impulse_metrics() -> None:
    image = np.zeros((5, 5))
    image[2, 2] = 10.0

    metrics = analyze_image(image)

    assert tuple(metrics) == METRIC_KEYS
    assert metrics == {
        "total": 10.0,
        "peak": 10.0,
        "x_centroid": 2.0,
        "y_centroid": 2.0,
        "centroid_radius_px": 0.0,
        "x_sigma_px": 0.0,
        "y_sigma_px": 0.0,
        "x_fwhm_px": 1.0,
        "y_fwhm_px": 1.0,
        "fwhm_px": 1.0,
    }


def test_analyze_image_rejects_all_zero_image() -> None:
    with pytest.raises(RuntimeError, match="No beam captured by detector screen"):
        analyze_image(np.zeros((5, 5)))


def test_analyze_energy_scan_aggregates_translated_impulses() -> None:
    images = np.zeros((2, 5, 5))
    images[0, 2, 1] = 10.0
    images[1, 3, 2] = 20.0

    default_energy = analyze_energy_scan(images, plot=False)
    custom_energy = analyze_energy_scan(images, energies=[7000.0, 7100.0], plot=False)

    assert tuple(custom_energy) == ("energy", *METRIC_KEYS)
    np.testing.assert_array_equal(default_energy["energy"], [6000.0, 8000.0])
    expected = {
        "energy": [7000.0, 7100.0],
        "total": [10.0, 20.0],
        "peak": [10.0, 20.0],
        "x_centroid": [1.0, 2.0],
        "y_centroid": [2.0, 3.0],
        "centroid_radius_px": [1.0, 1.0],
        "x_sigma_px": [0.0, 0.0],
        "y_sigma_px": [0.0, 0.0],
        "x_fwhm_px": [1.0, 1.0],
        "y_fwhm_px": [1.0, 1.0],
        "fwhm_px": [1.0, 1.0],
    }
    for key, values in expected.items():
        np.testing.assert_array_equal(custom_energy[key], values)


def test_analyze_energy_scan_validates_energy_count() -> None:
    with pytest.raises(ValueError, match=r"Expected 2 energies, but got shape \(1,\)"):
        analyze_energy_scan(np.ones((2, 3, 3)), energies=[7000.0], plot=False)


def test_analyze_energy_scan_plots_four_axes(mocker) -> None:
    images = np.zeros((2, 5, 5))
    images[0, 2, 1] = 10.0
    images[1, 2, 3] = 10.0
    show = mocker.patch("tst_sim_tools.analysis.image.plt.show")

    analyze_energy_scan(images, energies=[7000.0, 7100.0], plot=True)

    show.assert_called_once_with()
    assert len(plt.gcf().axes) == 4
