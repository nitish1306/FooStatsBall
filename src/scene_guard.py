"""
Scene-cut detection / Continuous clip validation module.
Detects hard broadcast cuts, camera transitions, or zooms using HSV histogram correlation.
Ensures static planar homography is applied only on continuous shots.
"""

import cv2
import numpy as np


def detect_scene_cuts(
    video_path: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    threshold: float = 0.45,
    sample_stride: int = 2
) -> list[int]:
    """
    Detect hard scene cuts using HSV color histogram correlation difference.

    Args:
        video_path: Video file path.
        start_frame: First frame index.
        end_frame: Last frame index.
        threshold: Histogram correlation drop threshold (< threshold indicates a cut).
        sample_stride: Step size between sampled frames to optimize speed.

    Returns:
        List of frame indices where scene cuts were detected.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame >= total_frames:
        end_frame = total_frames - 1

    cuts = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_hist = None
    prev_frame_idx = start_frame

    curr_frame_idx = start_frame
    while curr_frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and compute 2D H-S histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if prev_hist is not None:
            # Correlation metric: 1.0 = identical, 0.0 or negative = completely different
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < threshold:
                cuts.append(curr_frame_idx)

        prev_hist = hist
        prev_frame_idx = curr_frame_idx
        curr_frame_idx += sample_stride

    cap.release()
    return cuts

