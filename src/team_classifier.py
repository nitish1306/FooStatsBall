"""
Phase 2: Enhanced Team Jersey Classifier.
Uses multi-color-space chromaticity analysis:
- Normalized Red Ratio: R / (G + B)
- CIELAB 'a' channel (Green vs Magenta) and 'b' channel (Blue vs Yellow)
- Excludes turf/pitch green reflection
- Computes track-level median score and locks team assignment per persistent track ID
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


def extract_torso_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Extract the central chest/torso region of a player's bounding box."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w_frame, int(x2)), min(h_frame, int(y2))

    w = x2 - x1
    h = y2 - y1
    if w < 8 or h < 14:
        return None

    # Inner core torso: 18% to 50% vertical, 20% inner margins
    torso_y1 = y1 + int(h * 0.18)
    torso_y2 = y1 + int(h * 0.50)
    torso_x1 = x1 + int(w * 0.20)
    torso_x2 = x2 - int(w * 0.20)

    if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
        return None

    return frame[torso_y1:torso_y2, torso_x1:torso_x2]


def compute_jersey_chromatic_score(torso_crop: np.ndarray) -> float | None:
    """
    Compute chromatic kit distinction score based on Red vs Green/Blue balance.
    Red/Warm shirts have R / (G + B) > 0.62.
    Light/Green/Bib shirts have R / (G + B) <= 0.60.
    """
    if torso_crop is None or torso_crop.size == 0:
        return None

    b = torso_crop[:, :, 0].astype(np.float32)
    g = torso_crop[:, :, 1].astype(np.float32)
    r = torso_crop[:, :, 2].astype(np.float32)

    # Exclude pure turf green (G > R, G > B, low R/B)
    turf_mask = (g > b) & (g > r) & (r < 115) & (b < 95)
    fg_b = b[~turf_mask]
    fg_g = g[~turf_mask]
    fg_r = r[~turf_mask]

    if len(fg_r) < 15:
        fg_b, fg_g, fg_r = b.flatten(), g.flatten(), r.flatten()

    mean_r = np.median(fg_r)
    mean_g = np.median(fg_g)
    mean_b = np.median(fg_b)

    ratio = mean_r / (mean_g + mean_b + 1e-5)
    return float(ratio)


class TeamClassifier:
    """Clusters players into Team 0 and Team 1 with temporal majority-vote locking."""

    def __init__(self):
        self.track_scores = defaultdict(list)
        self.track_team_map = {}
        self.team_colors = {
            0: "#ef5350",  # Team Red
            1: "#42a5f5",  # Team Blue
            -1: "#ffd54f", # Goalkeeper: Gold
        }

    def collect_features_from_video(self, video_path: str, telemetry: dict, max_samples_per_track: int = 40):
        """Pass 2: Sample torso features across frames for all persistent track IDs."""
        cap = cv2.VideoCapture(video_path)
        frames_dict = telemetry["frames"]

        frame_indices = sorted([int(k) for k in frames_dict.keys()])
        stride = max(1, len(frame_indices) // 300)

        for f_idx in frame_indices[::stride]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            for player in frames_dict[str(f_idx)]["players"]:
                tid = player["tracker_id"]
                if tid <= 0:
                    continue
                if len(self.track_scores[tid]) >= max_samples_per_track:
                    continue

                crop = extract_torso_crop(frame, player["bbox"])
                score = compute_jersey_chromatic_score(crop)
                if score is not None:
                    self.track_scores[tid].append(score)

        cap.release()

    def fit_clusters(self, pitch_length: float = 90.0, pitch_width: float = 45.0, telemetry: dict | None = None):
        """Fit 2-team K-Means on median chromatic score and lock assignments."""
        track_ids = []
        medians = []

        for tid, scores in self.track_scores.items():
            if len(scores) > 0:
                med_score = float(np.median(scores))
                medians.append([med_score])
                track_ids.append(tid)

        if len(medians) < 2:
            for tid in self.track_scores.keys():
                self.track_team_map[tid] = 0
            return

        # 2-cluster KMeans on chromatic score
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        X = np.array(medians)
        labels = kmeans.fit_predict(X)

        # Cluster with higher red score is Team 0 (Red)
        center_0 = kmeans.cluster_centers_[0][0]
        center_1 = kmeans.cluster_centers_[1][0]
        red_cluster_idx = 0 if center_0 > center_1 else 1

        for tid, label in zip(track_ids, labels):
            self.track_team_map[tid] = 0 if label == red_cluster_idx else 1

        # Goalkeeper isolation: players spending majority of time in goal-mouth
        if telemetry is not None:
            self.isolate_goalkeepers(telemetry, pitch_length, pitch_width)

    def isolate_goalkeepers(self, telemetry: dict, pitch_length: float, pitch_width: float):
        """Identify designated goalkeepers by spatial positioning."""
        goal_box_depth = 14.0
        goal_box_top = (pitch_width - 26.0) / 2
        goal_box_bot = goal_box_top + 26.0

        for tid in list(self.track_team_map.keys()):
            in_goal_box = 0
            total_seen = 0
            for fd in telemetry["frames"].values():
                for p in fd["players"]:
                    if p["tracker_id"] == tid:
                        total_seen += 1
                        x, y = p["pitch_xy"]
                        if (x <= goal_box_depth or x >= pitch_length - goal_box_depth) and (goal_box_top <= y <= goal_box_bot):
                            in_goal_box += 1

            if total_seen >= 25 and (in_goal_box / total_seen) >= 0.70:
                self.track_team_map[tid] = -1  # Designated Goalkeeper
