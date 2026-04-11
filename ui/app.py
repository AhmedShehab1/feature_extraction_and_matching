"""Streamlit UI for Harris & λ− Feature Detection Visualization.

Run with:
    streamlit run ui/app.py

This module is fully self-contained and does NOT modify any project files.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path handling — allow importing project modules from the parent directory
# ---------------------------------------------------------------------------
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import cv2
import numpy as np
import streamlit as st

from harris_detection import detect_harris_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARRIS_COLOR = (0, 255, 0)   # Green (BGR)
LAMBDA_COLOR = (255, 0, 0)   # Blue  (BGR)
OVERLAP_HARRIS_COLOR = (0, 255, 0)
OVERLAP_LAMBDA_COLOR = (0, 0, 255)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_to_cv2(uploaded_file) -> np.ndarray:
    """Convert a Streamlit UploadedFile to a BGR OpenCV image."""
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def _draw_keypoints(
    image: np.ndarray,
    keypoints: list,
    color: tuple,
) -> np.ndarray:
    """Draw keypoints on a copy of the image and return RGB for display."""
    vis = cv2.drawKeypoints(
        image, keypoints, None,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Page configuration ----
    st.set_page_config(
        page_title="Harris & λ− Feature Detection",
        layout="wide",
    )

    st.title("🔍 Harris & λ− Feature Detection")
    st.markdown(
        "Upload an image, tune the parameters, and compare **Harris** vs "
        "**λ−** (minimum eigenvalue) corner detection side-by-side."
    )

    # ---- Sidebar controls ----
    st.sidebar.header("⚙️ Detection Parameters")

    # ── Harris-specific parameters ──
    with st.sidebar.container(border=True):
        st.markdown("🟢 **Harris Parameters**")
        threshold_harris = st.slider(
            "Harris Threshold",
            min_value=0.001,
            max_value=0.100,
            value=0.010,
            step=0.001,
            format="%.3f",
        )
        k = st.slider(
            "Harris k factor",
            min_value=0.01,
            max_value=0.10,
            value=0.04,
            step=0.01,
            format="%.2f",
        )

    # ── λ−-specific parameters ──
    with st.sidebar.container(border=True):
        st.markdown("🔵 **λ− Parameters**")
        threshold_lambda = st.slider(
            "λ− Threshold",
            min_value=0.001,
            max_value=0.100,
            value=0.010,
            step=0.001,
            format="%.3f",
        )

    # ── Shared parameters (apply to both methods) ──
    with st.sidebar.container(border=True):
        st.markdown("🔗 **Shared Parameters**")
        gaussian_ksize = st.slider(
            "Gaussian Kernel Size",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
        )
        gaussian_sigma = st.slider(
            "Gaussian Sigma",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            format="%.1f",
        )
        nms_window = st.slider(
            "NMS window size",
            min_value=3,
            max_value=9,
            value=5,
            step=2,
        )

    # ── Display options ──
    st.sidebar.header("🎛️ Display Options")
    with st.sidebar.container(border=True):
        keypoint_size = st.slider(
            "Keypoint Size",
            min_value=1,
            max_value=20,
            value=10,
        )
        show_harris = st.checkbox("Show Harris keypoints", value=True)
        show_lambda = st.checkbox("Show λ− keypoints", value=True)
        show_overlay = st.checkbox("Overlay both on one image", value=False)

    # ---- Image upload ----
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image to get started.")
        return

    # ---- Decode image ----
    image_bgr = _bytes_to_cv2(uploaded_file)
    image_rgb = _bgr_to_rgb(image_bgr)

    # ---- Run detection ----
    harris_kps, harris_time = detect_harris_features(
        image_bgr,
        method="harris",
        threshold_ratio=threshold_harris,
        nms_window=nms_window,
        k=k,
        gaussian_ksize=gaussian_ksize,
        gaussian_sigma=gaussian_sigma,
        keypoint_size=keypoint_size,
    )
    lambda_kps, lambda_time = detect_harris_features(
        image_bgr,
        method="lambda",
        threshold_ratio=threshold_lambda,
        nms_window=nms_window,
        gaussian_ksize=gaussian_ksize,
        gaussian_sigma=gaussian_sigma,
        keypoint_size=keypoint_size,
    )

    # ---- Layout: three columns ----
    col_orig, col_harris, col_lambda = st.columns(3)

    with col_orig:
        st.subheader("Original")
        st.image(image_rgb, width="stretch")
        st.caption(f"{image_bgr.shape[1]}×{image_bgr.shape[0]} px")

    with col_harris:
        if show_harris:
            harris_vis = _draw_keypoints(image_bgr, harris_kps, HARRIS_COLOR)
            st.subheader(f"Harris ({len(harris_kps)} pts | {harris_time:.4f}s)")
            st.image(harris_vis, width="stretch")
        else:
            st.subheader("Harris (hidden)")

    with col_lambda:
        if show_lambda:
            lambda_vis = _draw_keypoints(image_bgr, lambda_kps, LAMBDA_COLOR)
            st.subheader(f"λ− ({len(lambda_kps)} pts | {lambda_time:.4f}s)")
            st.image(lambda_vis, width="stretch")
        else:
            st.subheader("λ− (hidden)")

    # ---- Overlay view (bonus) ----
    if show_overlay:
        st.markdown("---")
        st.subheader("🔀 Overlay — Harris (green) + λ− (red)")
        overlay = image_bgr.copy()
        overlay = cv2.drawKeypoints(
            overlay, harris_kps, None,
            color=OVERLAP_HARRIS_COLOR,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        overlay = cv2.drawKeypoints(
            overlay, lambda_kps, None,
            color=OVERLAP_LAMBDA_COLOR,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        st.image(_bgr_to_rgb(overlay), width="stretch")

    # ---- Performance comparison ----
    st.markdown("---")
    st.subheader("📊 Performance Comparison")

    m1, m2, m3 = st.columns(3)
    m1.metric("Harris keypoints", len(harris_kps))
    m2.metric("λ− keypoints", len(lambda_kps))

    faster = "Harris" if harris_time < lambda_time else "λ−"
    m3.metric("Faster method", faster)

    t1, t2 = st.columns(2)
    t1.metric("Harris time", f"{harris_time:.4f} s")
    t2.metric("λ− time", f"{lambda_time:.4f} s")


if __name__ == "__main__":
    main()
