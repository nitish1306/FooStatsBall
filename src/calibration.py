"""
Pitch calibration module.
Computes a static planar homography from user-selected pitch landmarks
to a standard football pitch coordinate system (in meters).
"""

import json
import numpy as np
import cv2
from pathlib import Path

# Standard 5-a-side / 7-a-side pitch dimensions (meters)
# Adjust these to match your actual pitch
PITCH_LENGTH = 90.0  # touchline-to-touchline (x-axis)
PITCH_WIDTH = 45.0   # goal-line-to-goal-line (y-axis)

# Standard pitch landmark coordinates in meters (top-down, origin at top-left corner)
# These are the most common landmarks visible from a fixed camera
STANDARD_LANDMARKS = {
    "top_left_corner": (0.0, 0.0),
    "top_right_corner": (PITCH_LENGTH, 0.0),
    "bottom_left_corner": (0.0, PITCH_WIDTH),
    "bottom_right_corner": (PITCH_LENGTH, PITCH_WIDTH),
    "center_spot": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
    "left_penalty_spot": (11.0, PITCH_WIDTH / 2),
    "right_penalty_spot": (PITCH_LENGTH - 11.0, PITCH_WIDTH / 2),
    "top_left_penalty_box": (0.0, (PITCH_WIDTH - 40.32) / 2),
    "bottom_left_penalty_box": (0.0, (PITCH_WIDTH + 40.32) / 2),
    "top_right_penalty_box_front": (16.5, (PITCH_WIDTH - 40.32) / 2),
    "bottom_right_penalty_box_front": (16.5, (PITCH_WIDTH + 40.32) / 2),
    "top_right_penalty_box": (PITCH_LENGTH, (PITCH_WIDTH - 40.32) / 2),
    "bottom_right_penalty_box": (PITCH_LENGTH, (PITCH_WIDTH + 40.32) / 2),
    "top_left_penalty_box_front": (PITCH_LENGTH - 16.5, (PITCH_WIDTH - 40.32) / 2),
    "bottom_left_penalty_box_front": (PITCH_LENGTH - 16.5, (PITCH_WIDTH + 40.32) / 2),
    "halfway_top": (PITCH_LENGTH / 2, 0.0),
    "halfway_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH),
}

# Smaller pitch presets
PITCH_PRESETS = {
    "5v5": {"length": 40.0, "width": 25.0, "players": 5},
    "6v6": {"length": 50.0, "width": 30.0, "players": 6},
    "7v7": {"length": 60.0, "width": 40.0, "players": 7},
    "8v8": {"length": 70.0, "width": 45.0, "players": 8},
    "9v9": {"length": 75.0, "width": 50.0, "players": 9},
    "10v10": {"length": 90.0, "width": 60.0, "players": 10},
    "11v11": {"length": 105.0, "width": 68.0, "players": 11},
    "futsal": {"length": 40.0, "width": 20.0, "players": 5},
}


class PitchCalibrator:
    """Handles homography computation and coordinate transforms."""

    def __init__(self):
        self.homography_matrix = None
        self.inverse_homography = None
        self.pitch_length = PITCH_LENGTH
        self.pitch_width = PITCH_WIDTH
        self.format = None
        self.players_per_team = None
        self.calibration_points = []  # list of (pixel_xy, pitch_xy) pairs

    def set_pitch_dimensions(self, length: float, width: float):
        """Set custom pitch dimensions."""
        self.pitch_length = length
        self.pitch_width = width

    def set_pitch_preset(self, preset: str):
        """Set pitch dimensions from a preset."""
        if preset in PITCH_PRESETS:
            dims = PITCH_PRESETS[preset]
            self.set_pitch_dimensions(dims["length"], dims["width"])
            self.format, self.players_per_team = preset, dims["players"]
        else:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PITCH_PRESETS.keys())}")

    def add_calibration_point(self, pixel_x: float, pixel_y: float,
                              pitch_x: float, pitch_y: float):
        """Add a pixel-to-pitch coordinate mapping."""
        self.calibration_points.append(
            ((pixel_x, pixel_y), (pitch_x, pitch_y))
        )

    def clear_calibration_points(self):
        """Clear all calibration points."""
        self.calibration_points = []

    def compute_homography(self) -> bool:
        """
        Compute the homography matrix from collected calibration points.
        Requires at least 4 point pairs.
        Returns True if successful.
        """
        if len(self.calibration_points) < 4:
            raise ValueError(
                f"Need at least 4 calibration points, got {len(self.calibration_points)}"
            )

        src_pts = np.array(
            [p[0] for p in self.calibration_points], dtype=np.float32
        )
        dst_pts = np.array(
            [p[1] for p in self.calibration_points], dtype=np.float32
        )

        self.homography_matrix, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 5.0
        )

        if self.homography_matrix is not None:
            self.inverse_homography = np.linalg.inv(self.homography_matrix)
            return True
        return False

    def pixel_to_pitch(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Transform a pixel coordinate to pitch coordinates (meters)."""
        if self.homography_matrix is None:
            raise RuntimeError("Homography not computed. Call compute_homography() first.")

        pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        pitch_x, pitch_y = transformed[0][0]
        return float(pitch_x), float(pitch_y)

    def pitch_to_pixel(self, pitch_x: float, pitch_y: float) -> tuple[float, float]:
        """Transform pitch coordinates (meters) back to pixel coordinates."""
        if self.inverse_homography is None:
            raise RuntimeError("Homography not computed.")

        pt = np.array([[[pitch_x, pitch_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_homography)
        px, py = transformed[0][0]
        return float(px), float(py)

    def pixel_to_pitch_batch(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Transform a batch of pixel coordinates to pitch coordinates.
        pixel_coords: shape (N, 2)
        Returns: shape (N, 2) in pitch meters
        """
        if self.homography_matrix is None:
            raise RuntimeError("Homography not computed.")

        pts = pixel_coords.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
        return transformed.reshape(-1, 2)

    def get_reprojection_error(self) -> float:
        """Compute mean reprojection error in pixels for calibration quality check."""
        if self.homography_matrix is None or not self.calibration_points:
            return float('inf')

        errors = []
        for (px, py), (pitch_x, pitch_y) in self.calibration_points:
            # Project pitch point back to pixel
            reproj_px, reproj_py = self.pitch_to_pixel(pitch_x, pitch_y)
            err = np.sqrt((px - reproj_px) ** 2 + (py - reproj_py) ** 2)
            errors.append(err)

        return float(np.mean(errors))

    def save(self, path: str):
        """Save calibration to JSON file."""
        data = {
            "pitch_length": self.pitch_length,
            "pitch_width": self.pitch_width,
            "format": self.format,
            "players_per_team": self.players_per_team,
            "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            "calibration_points": [
                {"pixel": list(p), "pitch": list(w)}
                for p, w in self.calibration_points
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str):
        """Load calibration from JSON file."""
        data = json.loads(Path(path).read_text())
        self.pitch_length = data["pitch_length"]
        self.pitch_width = data["pitch_width"]
        self.format = data.get("format")
        self.players_per_team = data.get("players_per_team")
        self.reference_frame = int(data.get("reference_frame", 0))
        self.calibration_points = [
            (tuple(p["pixel"]), tuple(p["pitch"]))
            for p in data["calibration_points"]
        ]
        if data["homography_matrix"] is not None:
            self.homography_matrix = np.array(data["homography_matrix"], dtype=np.float64)
            self.inverse_homography = np.linalg.inv(self.homography_matrix)


def extract_frame(video_path: str, frame_number: int = 0) -> np.ndarray:
    """Extract a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_number} from {video_path}")
    return frame


def get_video_info(video_path: str) -> dict:
    """Get basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
    }
    cap.release()
    return info

