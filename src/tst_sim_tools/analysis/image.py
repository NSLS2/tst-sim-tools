"""Utilities for analyzing simulated detector image series."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy import ndimage, signal


def image_series(images: ArrayLike) -> np.ndarray:
    """Return detector data as an ``(N, height, width)`` floating-point image stack.

    Parameters
    ----------
    images
        A single 2D image, a 3D image stack, or a Tiled/Bluesky external-image stack
        with shape ``(N, M, height, width)``.

    Returns
    -------
    numpy.ndarray
        Floating-point image stack with non-finite values replaced by zero.
    """
    stack = np.asarray(images, dtype=float)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    elif stack.ndim == 4:
        stack = stack[:, 0, :, :] if stack.shape[1] == 1 else stack.sum(axis=1)
    elif stack.ndim != 3:
        raise ValueError(f"Expected 2D image, 3D stack, or 4D external stack, but got {stack.ndim} dimensions")

    if stack.shape[0] == 0:
        raise ValueError("Expected at least one image in the stack")

    return stack


def fwhm(profile: ArrayLike) -> float:
    """Compute FWHM from a 1D marginal profile.

    Parameters
    ----------
    profile
        One-dimensional intensity profile.

    Returns
    -------
    float
        Full width at half maximum in pixels. Profiles with no positive signal, or
        profiles that fill the full detector width, return the profile length as a
        finite penalty for optimization.
    """
    values = np.asarray(profile, dtype=float)
    peak = float(values.max())
    if peak <= 0.0:
        return float(values.size)
    if np.all(values >= peak / 2.0):
        return float(values.size)

    peaks, _ = signal.find_peaks(values)
    if peaks.size == 0:
        return float(values.size)

    peak_index = int(peaks[np.argmax(values[peaks])])
    width = float(signal.peak_widths(values, [peak_index], rel_height=0.5)[0][0])
    return width if width > 0.0 else float(values.size)


def threshold_image(image: ArrayLike, threshold: float = 0.0) -> np.ndarray:
    """Zero pixels below a fractional peak-intensity threshold.

    Parameters
    ----------
    image
        Two-dimensional detector image.
    threshold
        Fraction of the image peak to use as the cutoff. Values must be in the
        closed interval [0, 1].

    Returns
    -------
    numpy.ndarray
        Thresholded image with the original shape.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Expected threshold in [0, 1], but got {threshold}")

    thresholded = np.asarray(image, dtype="float")
    peak = float(thresholded.max())
    if peak <= 0.0:
        thresholded.fill(0.0)
        return thresholded

    thresholded[thresholded <= peak * threshold] = 0.0
    return thresholded


def gaussian_blur(image: ArrayLike, sigma: float = 0.0, truncate: float = 4.0) -> np.ndarray:
    """Apply a separable Gaussian blur to a detector image.

    Parameters
    ----------
    image
        Two-dimensional detector image.
    sigma
        Gaussian standard deviation in pixels. A zero value returns a finite copy of
        ``image``.
    truncate
        Kernel half-width in units of ``sigma``.

    Returns
    -------
    numpy.ndarray
        Blurred image with the original shape.
    """
    if sigma < 0.0:
        raise ValueError(f"Expected non-negative sigma, but got {sigma}")
    if truncate <= 0.0:
        raise ValueError(f"Expected positive truncate, but got {truncate}")

    blurred = np.asarray(image, dtype="float")
    if sigma == 0.0:
        return blurred

    return ndimage.gaussian_filter(blurred, sigma=sigma, mode="nearest", truncate=truncate)


def preprocess(
    image: ArrayLike,
    threshold: float = 0.0,
    blur: float = 0.0,
) -> np.ndarray:
    """Preprocess a detector image before centroid and FWHM extraction.

    Parameters
    ----------
    image
        Two-dimensional detector image.
    threshold
        Fraction of peak intensity to retain before blurring.
    blur
        Gaussian denoising blur sigma in pixels.

    Returns
    -------
    numpy.ndarray
        Preprocessed image with the original shape.
    """
    processed = threshold_image(image, threshold=threshold)
    return gaussian_blur(processed, sigma=blur)


def scroll_images(
    images: ArrayLike,
    *,
    threshold: float = 0.0,
    blur: float = 0.0,
    cmap: str = "viridis",
    show: bool = True,
) -> Figure:
    """Scroll through raw and processed detector images side-by-side.

    Parameters
    ----------
    images
        Detector image stack accepted by :func:`image_series`.
    threshold
        Fraction of peak intensity to retain in the processed image.
    blur
        Gaussian denoising blur sigma in pixels for the processed image.
    cmap
        Matplotlib colormap name for both images.
    show
        Whether to display the interactive figure before returning it.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with a scroll-wheel callback that advances through the image stack.
    """
    stack = image_series(images)
    frame_count = stack.shape[0]
    state = {"index": 0}

    def frame_pair(index: int) -> tuple[np.ndarray, np.ndarray]:
        raw = stack[index]
        return raw, preprocess(raw, threshold=threshold, blur=blur)

    def image_limits(raw: np.ndarray, processed: np.ndarray) -> tuple[float, float]:
        minimum = min(float(raw.min()), float(processed.min()))
        maximum = max(float(raw.max()), float(processed.max()))
        return (minimum, maximum) if minimum < maximum else (minimum, minimum + 1.0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    raw_ax, processed_ax = axs
    for ax in axs:
        ax.set_axis_off()

    raw, processed = frame_pair(state["index"])
    vmin, vmax = image_limits(raw, processed)
    raw_artist = raw_ax.imshow(raw, cmap=cmap, vmin=vmin, vmax=vmax)
    processed_artist = processed_ax.imshow(processed, cmap=cmap, vmin=vmin, vmax=vmax)
    raw_ax.set_title("Raw")
    processed_ax.set_title(f"Processed (threshold={threshold:g}, blur={blur:g})")

    def update(index: int) -> None:
        state["index"] = index % frame_count
        raw, processed = frame_pair(state["index"])
        vmin, vmax = image_limits(raw, processed)
        raw_artist.set_data(raw)
        processed_artist.set_data(processed)
        raw_artist.set_clim(vmin, vmax)
        processed_artist.set_clim(vmin, vmax)
        fig.suptitle(f"Image {state['index'] + 1} / {frame_count}; scroll to change frame")
        fig.canvas.draw_idle()

    def on_scroll(event: object) -> None:
        step = getattr(event, "step", 0)
        if step == 0:
            button = getattr(event, "button", None)
            step = 1 if button == "up" else -1 if button == "down" else 0
        if step != 0:
            update(state["index"] + (1 if step > 0 else -1))

    update(state["index"])
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    if show:
        plt.show()

    return fig


def locate_centroid(image: ArrayLike) -> tuple[float, float]:
    """Find the intensity-weighted centroid of a beam image.

    Parameters
    ----------
    image
        Two-dimensional detector image.

    Returns
    -------
    tuple[float, float]
        ``(x, y)`` centroid in pixel coordinates.
    """
    source = np.asarray(image, dtype="float")
    if float(source.sum()) <= 0.0:
        raise ValueError("Cannot locate centroid of an image with no positive signal")

    centroid = np.asarray(ndimage.center_of_mass(source), dtype=float)
    return float(centroid[1]), float(centroid[0])


def analyze_image(
    image: ArrayLike,
    threshold: float = 0.0,
    blur: float = 0.0,
) -> dict[str, float]:
    """Compute beam intensity, centroid, and spot-size metrics for one image.

    Parameters
    ----------
    image
        Two-dimensional detector image.
    threshold
        Fraction of peak intensity to retain before blurring.
    blur
        Gaussian denoising blur sigma in pixels.

    Returns
    -------
    dict[str, float]
        Scalar metrics in pixel units. ``fwhm_px`` is the mean of the horizontal and
        vertical marginal-profile FWHM values.
    """
    processed = preprocess(image, threshold=threshold, blur=blur)
    height, width = processed.shape
    center_x = width
    center_y = height
    total = float(processed.sum())
    peak = float(processed.max())
    if total <= 0.0:
        raise RuntimeError("No beam captured by detector screen")

    x_centroid, y_centroid = locate_centroid(processed)
    yy, xx = np.indices(processed.shape)
    x_sigma = float(np.sqrt((processed * (xx - x_centroid) ** 2).sum() / total))
    y_sigma = float(np.sqrt((processed * (yy - y_centroid) ** 2).sum() / total))
    x_fwhm = fwhm(processed.sum(axis=0))
    y_fwhm = fwhm(processed.sum(axis=1))

    return {
        "total": total,
        "peak": peak,
        "x_centroid": x_centroid,
        "y_centroid": y_centroid,
        "centroid_radius_px": float(np.hypot(x_centroid - center_x, y_centroid - center_y)),
        "x_sigma_px": x_sigma,
        "y_sigma_px": y_sigma,
        "x_fwhm_px": x_fwhm,
        "y_fwhm_px": y_fwhm,
        "fwhm_px": 0.5 * (x_fwhm + y_fwhm),
    }


def analyze_energy_scan(
    images: ArrayLike,
    energies: ArrayLike | None = None,
    plot: bool = True,
    *,
    threshold: float = 0.0,
    blur: float = 0.0,
) -> dict[str, np.ndarray]:
    """Summarize intensity, centroid, and spot-size trends for an energy scan.

    Parameters
    ----------
    images
        Detector image stack.
    energies
        Optional photon energies matching the image stack length.
    plot
        Whether to display a summary plot.
    threshold
        Fraction of peak intensity to retain before blurring.
    blur
        Gaussian denoising blur sigma in pixels.

    Returns
    -------
    dict[str, numpy.ndarray]
        Per-image metrics keyed by metric name.
    """
    stack = image_series(images)
    energy_values = np.linspace(6000.0, 8000.0, stack.shape[0]) if energies is None else np.asarray(energies, dtype=float)
    if energy_values.shape != (stack.shape[0],):
        raise ValueError(f"Expected {stack.shape[0]} energies, but got shape {energy_values.shape}")

    metrics = [analyze_image(frame, threshold=threshold, blur=blur) for frame in stack]
    out = {"energy": energy_values}
    for key in metrics[0]:
        out[key] = np.array([image_metrics[key] for image_metrics in metrics], dtype=float)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

        axs[0, 0].plot(energy_values, out["total"], "o-")
        axs[0, 0].set_title("Total intensity")

        axs[0, 1].plot(energy_values, out["peak"], "o-")
        axs[0, 1].set_title("Peak intensity")

        axs[1, 0].plot(energy_values, out["x_centroid"] - out["x_centroid"][0], "o-", label="x")
        axs[1, 0].plot(energy_values, out["y_centroid"] - out["y_centroid"][0], "o-", label="y")
        axs[1, 0].set_title("Centroid shift from first image")
        axs[1, 0].set_ylabel("pixels")
        axs[1, 0].legend()

        axs[1, 1].plot(energy_values, out["x_fwhm_px"], "o-", label="x FWHM")
        axs[1, 1].plot(energy_values, out["y_fwhm_px"], "o-", label="y FWHM")
        axs[1, 1].set_title("Spot size")
        axs[1, 1].set_ylabel("pixels")
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.set_xlabel("Energy [eV]")
            ax.grid(True, alpha=0.3)

        plt.show()

    return out
