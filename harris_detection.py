"""Harris Corner Detector and Minimum Eigenvalue (λ−) feature detection.

This module provides two feature-detection methods that replace SIFT-based
keypoint detection while remaining compatible with the existing matching
pipeline in ``feature_pipeline.py``.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale (float64)."""
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY).astype(np.float64)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def _get_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Generates a 2D Gaussian kernel matrix from scratch."""
    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_harris_features(
    image: np.ndarray,
    method: str = "harris",
    k: float = 0.04,
    threshold_ratio: float = 0.01,
    nms_window: int = 5,
    gaussian_ksize: int = 5,
    gaussian_sigma: float = 1.5,
    max_keypoints: int = 500,
    keypoint_size: float = 10.0,
) -> Tuple[List[cv2.KeyPoint], float]:
    """Detect corner features using Harris or Minimum-Eigenvalue (λ−).

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR, BGRA, or grayscale).
    method : str
        ``"harris"`` for the Harris corner response, or
        ``"lambda"`` for the minimum-eigenvalue response.
    k : float
        Harris sensitivity constant (only used when *method* is ``"harris"``).
    threshold_ratio : float
        Fraction of the peak response used as the acceptance threshold.
    nms_window : int
        Side length of the square window for non-maximum suppression.
    max_keypoints : int
        Maximum number of keypoints to return. The strongest responses are
        kept.  Set to ``0`` or a negative value to disable the cap.
    keypoint_size : float
        Diameter passed to ``cv2.KeyPoint`` — controls circle size when drawn
        with ``DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS``.
    gaussian_ksize : int
        Kernel size for Gaussian smoothing of the structure-tensor components.
    gaussian_sigma : float
        Standard deviation of the Gaussian smoothing kernel.

    Returns
    -------
    keypoints : List[cv2.KeyPoint]
        Detected feature locations wrapped as OpenCV KeyPoints.
    elapsed : float
        Wall-clock detection time in seconds.
    """
    if method not in ("harris", "lambda"):
        raise ValueError(f"Unknown method '{method}'. Use 'harris' or 'lambda'.")

    start = time.perf_counter()

    # 1. Convert to grayscale (float64 for numerical stability)
    gray = _to_grayscale(image)

    # 2. Compute image gradients using Sobel operators (from scratch)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64)

    # Use scipy's 2D convolution instead of OpenCV's filter2D
    Ix = convolve2d(gray, sobel_x, mode='same', boundary='symm')
    Iy = convolve2d(gray, sobel_y, mode='same', boundary='symm')

    # 3. Products of gradients (element-wise)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # 4. Gaussian smoothing of the structure-tensor components (from scratch)
    gauss_kernel = _get_gaussian_kernel(gaussian_ksize, gaussian_sigma)
    Sxx = convolve2d(Ix2, gauss_kernel, mode='same', boundary='symm')
    Syy = convolve2d(Iy2, gauss_kernel, mode='same', boundary='symm')
    Sxy = convolve2d(IxIy, gauss_kernel, mode='same', boundary='symm')

    # 5. Compute corner response map
    if method == "harris":
        # R = det(H) - k * trace(H)^2
        det_H = Sxx * Syy - Sxy * Sxy
        trace_H = Sxx + Syy
        response = det_H - k * (trace_H ** 2)
    else:  # method == "lambda"
        # λ_min = (trace - sqrt((Sxx - Syy)^2 + 4*Sxy^2)) / 2
        trace_H = Sxx + Syy
        discriminant = np.sqrt((Sxx - Syy) ** 2 + 4.0 * (Sxy ** 2))
        response = (trace_H - discriminant) / 2.0

    # 6. Threshold: keep only strong responses (> ratio * max)
    max_response = response.max()
    if max_response <= 0:
        elapsed = time.perf_counter() - start
        return [], elapsed

    threshold = threshold_ratio * max_response
    response_threshed = np.where(response > threshold, response, 0.0)

    # 7. Non-Maximum Suppression using a local-max filter
    local_max = maximum_filter(response_threshed, size=nms_window)
    nms_mask = (response_threshed == local_max) & (response_threshed > 0)

    # 8. Extract coordinates and their response strengths
    ys, xs = np.nonzero(nms_mask)
    strengths = response[ys, xs]

    # 9. Sort by descending response strength (vectorized argsort)
    order = np.argsort(-strengths)
    xs = xs[order]
    ys = ys[order]
    strengths = strengths[order]

    # 10. Keep only the top-N strongest keypoints
    if max_keypoints > 0 and len(xs) > max_keypoints:
        xs = xs[:max_keypoints]
        ys = ys[:max_keypoints]
        strengths = strengths[:max_keypoints]

    # 11. Wrap as cv2.KeyPoint with increased size for clear visualisation
    keypoints: List[cv2.KeyPoint] = [
        cv2.KeyPoint(x=float(x), y=float(y), size=keypoint_size)
        for x, y in zip(xs, ys)
    ]

    elapsed = time.perf_counter() - start
    return keypoints, elapsed
