from __future__ import annotations

import numpy as np


def global_rgb_histogram(image_rgb: np.ndarray, bins_per_channel: int = 4) -> np.ndarray:
    """
    Supplied baseline descriptor: a global 3-D RGB colour histogram.

    The histogram is L1-normalised so its entries sum to one.
    Students are expected to investigate and justify improvements beyond this baseline.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape H x W x 3.")
    if bins_per_channel < 2:
        raise ValueError("bins_per_channel must be at least 2.")

    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    hist, _ = np.histogramdd(
        pixels,
        bins=(bins_per_channel, bins_per_channel, bins_per_channel),
        range=((0, 256), (0, 256), (0, 256)),
    )
    feature = hist.astype(np.float32).ravel()
    total = feature.sum()
    if total > 0:
        feature /= total
    return feature
