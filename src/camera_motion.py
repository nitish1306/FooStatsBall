"""
Camera motion compensation for panning/zooming cameras (e.g. Veo).

The pitch homography is calibrated on one reference frame. For every later frame we
estimate the frame-to-frame image homography from background features (players masked
out), chain it, and map pixels back into the reference frame before projecting to pitch.
"""

import cv2
import numpy as np

_SCALE = 0.5  # flow runs on a half-res grayscale frame


class CameraMotionEstimator:
    def __init__(self, min_shift_px: float = 0.0):
        self.prev_gray = None
        self.to_ref = np.eye(3)  # current-frame pixels -> reference-frame pixels
        self.min_shift_px = min_shift_px
        self.lost_frames = 0
        S = np.diag([_SCALE, _SCALE, 1.0])
        self._S, self._S_inv = S, np.linalg.inv(S)

    def _gray(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=_SCALE, fy=_SCALE, interpolation=cv2.INTER_AREA)

    def update(self, frame, boxes=None) -> np.ndarray:
        """Feed next frame (+ optional player bboxes [x1,y1,x2,y2] in full-res px). Returns current->ref homography."""
        gray = self._gray(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.to_ref

        mask = np.full(gray.shape, 255, np.uint8)
        for b in boxes or []:
            x1, y1, x2, y2 = (int(v * _SCALE) for v in b)
            cv2.rectangle(mask, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 0, -1)

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=400, qualityLevel=0.01,
                                           minDistance=8, mask=mask)
        H_step = None
        if prev_pts is not None and len(prev_pts) >= 12:
            cur_pts, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None,
                                                      winSize=(21, 21), maxLevel=3)
            ok = st.reshape(-1) == 1
            if ok.sum() >= 12:
                # maps current (small) -> previous (small)
                H_step, inl = cv2.findHomography(cur_pts[ok], prev_pts[ok], cv2.RANSAC, 2.0)
                if H_step is not None and (inl is None or inl.sum() < 10):
                    H_step = None

        if H_step is None:
            self.lost_frames += 1  # keep last estimate (assume no motion)
        else:
            H_full = self._S_inv @ H_step @ self._S
            self.to_ref = self.to_ref @ H_full
            self.to_ref /= self.to_ref[2, 2]

        self.prev_gray = gray
        return self.to_ref

    def to_reference(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.to_ref).reshape(-1, 2)
