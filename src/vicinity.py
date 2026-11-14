"""Vicinity re-detection: a recently lost track is searched for in a crop around its last box."""
import numpy as np


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class VicinityRedetector:
    def __init__(self, model, device, fps, max_lost_sec=1.5, conf=0.08, pad=1.0, max_checks=10):
        self.model, self.device = model, device
        self.max_lost = int(max_lost_sec * fps)
        self.conf, self.pad, self.max_checks = conf, pad, max_checks
        self.last = {}  # tid -> (frame_idx, bbox)

    def update(self, frame, frame_idx, players):
        """Appends re-found players (flag `redetected`) to `players` in place."""
        seen = {p["tracker_id"] for p in players}
        lost = [(tid, f, b) for tid, (f, b) in self.last.items()
                if tid not in seen and 0 < frame_idx - f <= self.max_lost]
        lost.sort(key=lambda t: frame_idx - t[1])
        lost = lost[: self.max_checks]

        if lost:
            h, w = frame.shape[:2]
            crops, origins = [], []
            for _, _, b in lost:
                bw, bh = b[2] - b[0], b[3] - b[1]
                x1 = int(max(0, b[0] - self.pad * bw)); x2 = int(min(w, b[2] + self.pad * bw))
                y1 = int(max(0, b[1] - self.pad * bh * 0.5)); y2 = int(min(h, b[3] + self.pad * bh * 0.5))
                crops.append(frame[y1:y2, x1:x2]); origins.append((x1, y1))
            results = self.model.predict(crops, conf=self.conf, device=self.device, imgsz=320,
                                         classes=[0], verbose=False)
            taken = [p["bbox"] for p in players]
            for (tid, _, last_b), (ox, oy), r in zip(lost, origins, results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                best, best_score = None, 0.0
                for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    cand = [float(box[0] + ox), float(box[1] + oy), float(box[2] + ox), float(box[3] + oy)]
                    if any(_iou(cand, t) > 0.45 for t in taken):
                        continue  # already someone else's detection
                    score = _iou(cand, last_b) + 0.2 * float(c)
                    if score > best_score:
                        best, best_score, best_c = cand, score, float(c)
                if best is not None and best_score > 0.25:
                    taken.append(best)
                    players.append({
                        "tracker_id": tid, "bbox": best,
                        "foot_pixel": [(best[0] + best[2]) / 2.0, best[3]],
                        "confidence": best_c, "redetected": True,
                    })

        for p in players:
            tid = p["tracker_id"]
            # Re-detections keep the original lost-since frame, so a chain caps at max_lost.
            since = self.last[tid][0] if p.get("redetected") else frame_idx
            self.last[tid] = (since, p["bbox"])
        self.last = {t: v for t, v in self.last.items() if frame_idx - v[0] <= self.max_lost}
        return players
