"""Smoke-tests for harris_detection.detect_harris_features.

Generates a synthetic image with four known corners (white square on black
background) and verifies that both Harris and λ− detectors find keypoints
near those corners.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from harris_detection import detect_harris_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_square_image(size: int = 200, margin: int = 50) -> np.ndarray:
    """Return a uint8 grayscale image: white square centred on black."""
    img = np.zeros((size, size), dtype=np.uint8)
    img[margin : size - margin, margin : size - margin] = 255
    return img


SQUARE_IMG = _make_square_image()
# Expected corners of the white square (x, y) — approximate
EXPECTED_CORNERS = [
    (50, 50),
    (149, 50),
    (50, 149),
    (149, 149),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHarrisDetection:
    """Tests for method='harris'."""

    def test_returns_keypoints_and_time(self):
        kps, elapsed = detect_harris_features(SQUARE_IMG, method="harris")
        assert isinstance(kps, list)
        assert all(isinstance(kp, cv2.KeyPoint) for kp in kps)
        assert isinstance(elapsed, float) and elapsed > 0

    def test_detects_near_expected_corners(self):
        kps, _ = detect_harris_features(SQUARE_IMG, method="harris")
        detected = {(round(kp.pt[0]), round(kp.pt[1])) for kp in kps}
        tolerance = 10
        for cx, cy in EXPECTED_CORNERS:
            near = any(abs(x - cx) <= tolerance and abs(y - cy) <= tolerance for x, y in detected)
            assert near, f"No Harris keypoint within {tolerance}px of expected corner ({cx},{cy})"

    def test_nonzero_keypoints(self):
        kps, _ = detect_harris_features(SQUARE_IMG, method="harris")
        assert len(kps) > 0


class TestLambdaDetection:
    """Tests for method='lambda' (minimum eigenvalue)."""

    def test_returns_keypoints_and_time(self):
        kps, elapsed = detect_harris_features(SQUARE_IMG, method="lambda")
        assert isinstance(kps, list)
        assert all(isinstance(kp, cv2.KeyPoint) for kp in kps)
        assert isinstance(elapsed, float) and elapsed > 0

    def test_detects_near_expected_corners(self):
        kps, _ = detect_harris_features(SQUARE_IMG, method="lambda")
        detected = {(round(kp.pt[0]), round(kp.pt[1])) for kp in kps}
        tolerance = 10
        for cx, cy in EXPECTED_CORNERS:
            near = any(abs(x - cx) <= tolerance and abs(y - cy) <= tolerance for x, y in detected)
            assert near, f"No λ− keypoint within {tolerance}px of expected corner ({cx},{cy})"

    def test_nonzero_keypoints(self):
        kps, _ = detect_harris_features(SQUARE_IMG, method="lambda")
        assert len(kps) > 0


class TestInvalidMethod:
    def test_raises_on_unknown_method(self):
        with pytest.raises(ValueError):
            detect_harris_features(SQUARE_IMG, method="unknown")


class TestColorInput:
    """Ensure the function handles BGR input without crashing."""

    def test_bgr_input(self):
        bgr = cv2.cvtColor(SQUARE_IMG, cv2.COLOR_GRAY2BGR)
        kps, elapsed = detect_harris_features(bgr, method="harris")
        assert len(kps) > 0
        assert elapsed > 0
