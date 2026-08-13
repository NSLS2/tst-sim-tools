"""Utilities for analyzing simulated detector image series."""

import matplotlib.pyplot as plt
import numpy as np


def analyze_energy_scan(images, energies=None, plot=True):
    """Summarize intensity, centroid, and spot-size trends for an energy scan."""
    img = np.asarray(images)
    if img.ndim == 4:
        img = img[:, 0]  # (N, 1, H, W) -> (N, H, W)

    if energies is None:
        energies = np.linspace(6000, 8000, img.shape[0])

    yy, xx = np.indices(img.shape[1:])
    total = img.sum(axis=(1, 2))
    peak = img.max(axis=(1, 2))

    xcen = (img * xx).sum(axis=(1, 2)) / total
    ycen = (img * yy).sum(axis=(1, 2)) / total

    xsig = np.sqrt((img * (xx - xcen[:, None, None]) ** 2).sum(axis=(1, 2)) / total)
    ysig = np.sqrt((img * (yy - ycen[:, None, None]) ** 2).sum(axis=(1, 2)) / total)

    out = {
        "energy": energies,
        "total": total,
        "peak": peak,
        "x_centroid": xcen,
        "y_centroid": ycen,
        "x_sigma_px": xsig,
        "y_sigma_px": ysig,
        "x_fwhm_px": 2.355 * xsig,
        "y_fwhm_px": 2.355 * ysig,
    }

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

        axs[0, 0].plot(energies, total, "o-")
        axs[0, 0].set_title("Total intensity")

        axs[0, 1].plot(energies, peak, "o-")
        axs[0, 1].set_title("Peak intensity")

        axs[1, 0].plot(energies, xcen - xcen[0], "o-", label="x")
        axs[1, 0].plot(energies, ycen - ycen[0], "o-", label="y")
        axs[1, 0].set_title("Centroid shift from first image")
        axs[1, 0].set_ylabel("pixels")
        axs[1, 0].legend()

        axs[1, 1].plot(energies, 2.355 * xsig, "o-", label="x FWHM")
        axs[1, 1].plot(energies, 2.355 * ysig, "o-", label="y FWHM")
        axs[1, 1].set_title("Spot size")
        axs[1, 1].set_ylabel("pixels")
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.set_xlabel("Energy [eV]")
            ax.grid(True, alpha=0.3)

        plt.show()

    return out
