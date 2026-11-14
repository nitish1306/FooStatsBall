"""
Kit-agnostic team classifier.

- Pitch colour is learned from each frame (no hard-coded "green" thresholds).
- Each torso crop -> dominant non-pitch colour in CIELAB (k-means inside the crop).
- Each track -> robust median kit colour over many frames.
- Tracks -> 2 teams via weighted k-means on (a, b, L*0.5); tracks far from both centres
  (goalkeepers, referees) -> -1 "Other". Similarity score per track is stored.
- Team names/colours come from the cluster centres, so any kit pair works.
"""

from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

NAMED_COLORS = {
    "Red": (200, 30, 40), "Maroon": (120, 20, 35), "Orange": (240, 130, 30), "Yellow": (235, 215, 40),
    "Green": (40, 150, 60), "Sky Blue": (110, 180, 235), "Blue": (40, 80, 200), "Navy": (25, 35, 80),
    "Purple": (120, 50, 150), "Pink": (235, 120, 170), "White": (240, 240, 240), "Grey": (130, 130, 130),
    "Black": (25, 25, 25),
}
L_WEIGHT = 0.5  # lighting changes L more than chroma


def _lab(rgb_or_bgr: np.ndarray, bgr=True) -> np.ndarray:
    arr = np.asarray(rgb_or_bgr, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB if bgr else cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def _named(rgb) -> str:
    """Nearest named colour, lightness down-weighted (floodlights wash kits out)."""
    w = np.array([0.35, 1.0, 1.0])
    lab = _lab([rgb], bgr=False)[0]
    names = list(NAMED_COLORS)
    labs = _lab(list(NAMED_COLORS.values()), bgr=False)
    chroma = np.hypot(lab[1] - 128, lab[2] - 128)
    hue = np.degrees(np.arctan2(lab[2] - 128, lab[1] - 128)) % 360
    greenish = 100 <= hue <= 200  # grass bleeding into a light kit
    if lab[0] > 150 and (chroma < 20 or (greenish and chroma < 24)):
        return "White"
    if chroma <= 18:
        return "Black" if lab[0] < 60 else ("White" if lab[0] > 170 else "Grey")
    # dark reds/browns read as maroon (brown kits are rare; washed maroon is common)
    if lab[0] < 110 and (hue <= 75 or hue >= 340):
        return "Maroon"
    keep = [i for i, n in enumerate(names) if n not in ("White", "Grey", "Black")]
    names, labs = [names[i] for i in keep], labs[keep]
    hues = np.arctan2(labs[:, 2] - 128, labs[:, 1] - 128)
    dh = np.abs(np.angle(np.exp(1j * (hues - np.radians(hue)))))  # hue angle diff, robust to washout
    return names[int(np.argmin(dh + 0.004 * np.abs(labs[:, 0] - lab[0])))]


def _display_hex(lab, chroma_boost=1.5) -> str:
    lab = np.array(lab, np.float32).copy()
    lab[1:] = 128 + (lab[1:] - 128) * chroma_boost
    rgb = cv2.cvtColor(np.uint8([[np.clip(lab, 0, 255)]]), cv2.COLOR_LAB2RGB)[0, 0]
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def pitch_lab(frame: np.ndarray) -> np.ndarray:
    """Dominant colour of the lower 2/3 of the frame = the pitch."""
    h = frame.shape[0]
    small = cv2.resize(frame[h // 3:], (64, 32), interpolation=cv2.INTER_AREA)
    return np.median(_lab(small), axis=0)


def torso_kit_lab(frame, bbox, pitch) -> np.ndarray | None:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    crop = frame[y1 + int(h * 0.15): y1 + int(h * 0.5), x1 + int(w * 0.2): x2 - int(w * 0.2)]
    if crop.size == 0:
        return None
    lab = _lab(crop)
    # drop pitch-coloured pixels (Delta-E vs learned pitch colour)
    lab = lab[np.linalg.norm(lab - pitch, axis=1) > 18]
    if len(lab) < 12:
        return None
    k = 2 if len(lab) >= 24 else 1
    km = KMeans(n_clusters=k, n_init=2, random_state=0).fit(lab)
    dominant = np.bincount(km.labels_).argmax()
    return km.cluster_centers_[dominant]


class TeamClassifier:
    def __init__(self, two_kits_only: bool = False):
        self.two_kits_only = two_kits_only  # no ref / distinct GK kit -> everyone is on a team
        self.track_samples = defaultdict(list)
        self.track_team_map: dict[int, int] = {}
        self.track_similarity: dict[int, float] = {}
        self.team_colors = {0: "#ef5350", 1: "#42a5f5", -1: "#bdbdbd"}
        self.team_names = {0: "Team A", 1: "Team B", -1: "Other / GK / Ref"}
        self.goalkeepers: dict[int, int] = {}

    def collect_features_from_video(self, video_path: str, telemetry: dict, max_samples_per_track: int = 40):
        cap = cv2.VideoCapture(video_path)
        frames = telemetry["frames"]
        idxs = sorted(int(k) for k in frames)
        stride = max(1, len(idxs) // 300)
        for f in idxs[::stride]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            pitch = pitch_lab(frame)
            for p in frames[str(f)]["players"]:
                tid = p["tracker_id"]
                if tid <= 0 or len(self.track_samples[tid]) >= max_samples_per_track:
                    continue
                v = torso_kit_lab(frame, p["bbox"], pitch)
                if v is not None:
                    self.track_samples[tid].append(v)
        cap.release()

    @staticmethod
    def _feat(lab):
        lab = np.atleast_2d(lab)
        return np.column_stack([lab[:, 0] * L_WEIGHT, lab[:, 1], lab[:, 2]])

    def fit_clusters(self, pitch_length=90.0, pitch_width=45.0, telemetry=None):
        tids = [t for t, s in self.track_samples.items() if len(s) >= 2]
        if len(tids) < 2:
            self.track_team_map = {t: 0 for t in self.track_samples}
            return
        kit = np.array([np.median(self.track_samples[t], axis=0) for t in tids])
        weight = np.array([len(self.track_samples[t]) for t in tids], float)
        X = self._feat(kit)

        # Over-cluster (k=2..4), keep the two heaviest clusters as teams; extra clusters are
        # GKs/refs unless they sit close to a team (then they're just a lighting variant of it).
        best = None
        for k in range(2, (2 if self.two_kits_only else min(4, len(tids))) + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X, sample_weight=weight)
            mass = np.bincount(km.labels_, weights=weight, minlength=k)
            order = np.argsort(-mass)
            teams = km.cluster_centers_[order[:2]]
            sep = np.linalg.norm(teams[0] - teams[1])
            # prefer the k whose top-2 hold most mass while staying well separated
            score = sep * mass[order[:2]].sum() / mass.sum()
            if best is None or score > best[0] * 1.05:
                best = (score, km, order, teams, sep)
        _, km, order, teams, sep = best

        d = np.linalg.norm(X[:, None, :] - teams[None], axis=2)  # (n, 2) distance to each team centre
        own = d.argmin(1)
        d_own = d[np.arange(len(tids)), own]
        d_other = d[np.arange(len(tids)), 1 - own]
        in_team_cluster = np.isin(km.labels_, order[:2])
        spread = {c: np.median(d_own[(own == c) & in_team_cluster]) if np.any((own == c) & in_team_cluster) else 1.0
                  for c in (0, 1)}
        for i, t in enumerate(tids):
            sim = float(d_other[i] / (d_own[i] + d_other[i] + 1e-6))  # 0.5 ambiguous .. 1 certain
            far = d_own[i] > max(3.0 * spread[own[i]], 0.4 * sep)
            outlier = (not in_team_cluster[i] and d_own[i] > 0.35 * sep) or far
            self.track_team_map[t] = -1 if (outlier and not self.two_kits_only) else int(own[i])
            self.track_similarity[t] = round(sim, 3)
        for t in self.track_samples:
            self.track_team_map.setdefault(t, -1)

        for c in (0, 1):
            m = [i for i, t in enumerate(tids) if self.track_team_map[t] == c]
            lab = np.average(kit[m], axis=0, weights=weight[m]) if m else teams[c] / [L_WEIGHT, 1, 1]
            self.team_colors[c] = _display_hex(lab)
            rgb = cv2.cvtColor(np.uint8([[np.clip(lab, 0, 255)]]), cv2.COLOR_LAB2RGB)[0, 0]
            self.team_names[c] = _named(tuple(int(v) for v in rgb))
        if self.team_names[0] == self.team_names[1]:
            self.team_names = {**self.team_names, 0: f"{self.team_names[0]} (dark)", 1: f"{self.team_names[1]} (light)"}

        if telemetry is not None and not self.two_kits_only:
            self.assign_goalkeepers(telemetry, pitch_length)

    def assign_goalkeepers(self, telemetry, pitch_length):
        """'Other' tracks that live near one goal = that half's team keeper (team = majority of outfielders there)."""
        pos = defaultdict(list)
        for fd in telemetry["frames"].values():
            for p in fd["players"]:
                pos[p["tracker_id"]].append(p["pitch_xy"][0])
        team_x = {c: np.mean([np.mean(pos[t]) for t, tm in self.track_team_map.items() if tm == c and pos[t]] or [pitch_length / 2])
                  for c in (0, 1)}
        self.goalkeepers = {}
        for t, tm in self.track_team_map.items():
            if tm != -1 or len(pos[t]) < 25:
                continue
            x = np.median(pos[t])
            if x < pitch_length * 0.2 or x > pitch_length * 0.8:
                # keeper defends the end his team sits closer to on average
                near_left = x < pitch_length / 2
                c = min((0, 1), key=lambda c: team_x[c] if near_left else -team_x[c])
                self.goalkeepers[t] = c
