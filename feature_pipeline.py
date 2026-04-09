from __future__ import annotations

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from harris_detection import detect_harris_features

NCC_INVALID_VALUE = float("-inf")


@dataclass
class MatchResult:
    matches: List[cv2.DMatch]
    scores: np.ndarray
    elapsed_seconds: float


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def generate_sift_descriptors(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    gray = _to_grayscale(image)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    return keypoints, descriptors


def _empty_match_result() -> MatchResult:
    return MatchResult(matches=[], scores=np.array([], dtype=np.float32), elapsed_seconds=0.0)


def match_descriptors_ssd(descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> MatchResult:
    if descriptors_1.size == 0 or descriptors_2.size == 0:
        return _empty_match_result()

    start = time.perf_counter()
    diff = descriptors_1[:, None, :] - descriptors_2[None, :, :]
    ssd_matrix = np.sum(diff * diff, axis=2)
    best_indices = np.argmin(ssd_matrix, axis=1)
    best_scores = ssd_matrix[np.arange(ssd_matrix.shape[0]), best_indices]
    elapsed = time.perf_counter() - start

    matches = [
        cv2.DMatch(_queryIdx=int(i), _trainIdx=int(best_indices[i]), _imgIdx=0, _distance=float(best_scores[i]))
        for i in range(descriptors_1.shape[0])
    ]
    matches.sort(key=lambda m: m.distance)
    return MatchResult(matches=matches, scores=best_scores.astype(np.float32), elapsed_seconds=elapsed)


def match_descriptors_ncc(descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> MatchResult:
    if descriptors_1.size == 0 or descriptors_2.size == 0:
        return _empty_match_result()

    start = time.perf_counter()
    d1 = descriptors_1.astype(np.float32)
    d2 = descriptors_2.astype(np.float32)

    d1_centered = d1 - d1.mean(axis=1, keepdims=True)
    d2_centered = d2 - d2.mean(axis=1, keepdims=True)

    d1_norms = np.linalg.norm(d1_centered, axis=1, keepdims=True)
    d2_norms = np.linalg.norm(d2_centered, axis=1, keepdims=True)
    denominator = d1_norms @ d2_norms.T

    numerator = d1_centered @ d2_centered.T
    ncc_matrix = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, fill_value=NCC_INVALID_VALUE, dtype=np.float32),
        where=denominator > 0,
    )

    best_indices = np.argmax(ncc_matrix, axis=1)
    best_scores = ncc_matrix[np.arange(ncc_matrix.shape[0]), best_indices]
    elapsed = time.perf_counter() - start

    valid_rows = np.where(np.isfinite(best_scores))[0]
    matches = [
        cv2.DMatch(
            _queryIdx=int(i),
            _trainIdx=int(best_indices[i]),
            _imgIdx=0,
            _distance=float(1.0 - best_scores[i]),
        )
        for i in valid_rows
    ]
    matches.sort(key=lambda m: m.distance)
    return MatchResult(matches=matches, scores=best_scores.astype(np.float32), elapsed_seconds=elapsed)


def draw_matches(
    image_1: np.ndarray,
    keypoints_1: Sequence[cv2.KeyPoint],
    image_2: np.ndarray,
    keypoints_2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    top_k: int = 50,
) -> np.ndarray:
    selected = list(matches[:top_k])
    return cv2.drawMatches(
        image_1,
        list(keypoints_1),
        image_2,
        list(keypoints_2),
        selected,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def _detect_keypoints(
    image: np.ndarray,
    detector: str,
) -> Tuple[List[cv2.KeyPoint], np.ndarray, float]:
    """Detect keypoints and compute SIFT descriptors.

    When *detector* is ``"sift"`` the full SIFT pipeline is used.
    For ``"harris"`` or ``"lambda"`` the keypoints come from
    :func:`detect_harris_features` and descriptors are computed
    with ``SIFT.compute``.
    """
    if detector == "sift":
        start = time.perf_counter()
        keypoints, descriptors = generate_sift_descriptors(image)
        elapsed = time.perf_counter() - start
        return keypoints, descriptors, elapsed

    # Harris or Lambda detection
    keypoints, detect_time = detect_harris_features(image, method=detector)
    # Compute SIFT descriptors on detected keypoints
    gray = _to_grayscale(image)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    return list(keypoints), descriptors, detect_time


def _save_keypoint_visualization(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    output_path: str,
) -> None:
    """Draw detected keypoints on the image and save to disk."""
    vis = cv2.drawKeypoints(image, keypoints, None,
                            color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(output_path, vis)


def process_image_pair(
    image_path_1: str,
    image_path_2: str,
    output_dir: str,
    top_k: int = 50,
    detector: str = "sift",
) -> None:
    image_1 = _load_image(image_path_1)
    image_2 = _load_image(image_path_2)
    os.makedirs(output_dir, exist_ok=True)
    stem_1 = os.path.splitext(os.path.basename(image_path_1))[0]
    stem_2 = os.path.splitext(os.path.basename(image_path_2))[0]

    # --- Keypoint detection & descriptor computation ---
    keypoints_1, descriptors_1, detect_time_1 = _detect_keypoints(image_1, detector)
    keypoints_2, descriptors_2, detect_time_2 = _detect_keypoints(image_2, detector)
    total_detect_time = detect_time_1 + detect_time_2

    # --- Save keypoint visualizations for Harris / Lambda ---
    if detector in ("harris", "lambda"):
        tag = "harris" if detector == "harris" else "lambda"
        _save_keypoint_visualization(
            image_1, keypoints_1,
            os.path.join(output_dir, f"{stem_1}_{tag}_points.jpg"),
        )
        _save_keypoint_visualization(
            image_2, keypoints_2,
            os.path.join(output_dir, f"{stem_2}_{tag}_points.jpg"),
        )

    # --- Matching ---
    ssd_result = match_descriptors_ssd(descriptors_1, descriptors_2)
    ncc_result = match_descriptors_ncc(descriptors_1, descriptors_2)

    # --- Print summary ---
    print(f"Image Pair: {os.path.basename(image_path_1)} <-> {os.path.basename(image_path_2)}")
    print(f"Detector: {detector.upper()}")
    print(f"  Keypoints (img1): {len(keypoints_1)}")
    print(f"  Keypoints (img2): {len(keypoints_2)}")
    print(f"  Detection_Time: [{total_detect_time:.4f}] seconds")
    print(f"  Matching_SSD_Time: [{ssd_result.elapsed_seconds:.4f}] seconds")
    print(f"  Matching_NCC_Time: [{ncc_result.elapsed_seconds:.4f}] seconds")

    # --- Save match visualizations ---
    ssd_vis = draw_matches(image_1, keypoints_1, image_2, keypoints_2, ssd_result.matches, top_k=top_k)
    ncc_vis = draw_matches(image_1, keypoints_1, image_2, keypoints_2, ncc_result.matches, top_k=top_k)

    cv2.imwrite(os.path.join(output_dir, f"{stem_1}_{stem_2}_ssd_matches.jpg"), ssd_vis)
    cv2.imwrite(os.path.join(output_dir, f"{stem_1}_{stem_2}_ncc_matches.jpg"), ncc_vis)


def _iter_image_pairs(image_paths: Iterable[str]) -> Iterable[Tuple[str, str]]:
    return itertools.combinations(image_paths, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature extraction and SSD/NCC matching pipeline")
    parser.add_argument("images", nargs="+", help="Input image paths. All unique pairs are processed.")
    parser.add_argument("--output-dir", default="output_matches", help="Directory for match visualizations")
    parser.add_argument("--top-k", type=int, default=50, help="Number of matches to visualize")
    parser.add_argument(
        "--detector",
        choices=["sift", "harris", "lambda"],
        default="sift",
        help="Keypoint detector to use (default: sift)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.images) < 2:
        raise ValueError("Provide at least two images to process matching.")
    for image_path_1, image_path_2 in _iter_image_pairs(args.images):
        process_image_pair(
            image_path_1, image_path_2,
            args.output_dir,
            top_k=args.top_k,
            detector=args.detector,
        )


if __name__ == "__main__":
    main()
