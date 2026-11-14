"""
Dedicated ball detection + global trajectory selection.

1. Per frame: low-confidence, high-res ball-only pass -> top-K candidates (tracker would drop them).
2. Whole clip: Viterbi over {candidates, miss} in camera-compensated pixels, so the chosen path is
   the most confident *physically continuous* sequence; isolated false positives lose to it.
"""

import numpy as np

BALL_CLASS = 32


class BallDetector:
    def __init__(self, model, device: str, imgsz: int = 1920, conf: float = 0.05, top_k: int = 5):
        self.model, self.device, self.imgsz, self.conf, self.top_k = model, device, imgsz, conf, top_k

    def detect(self, frame, player_bboxes=()) -> list[tuple[float, float, float]]:
        r = self.model.predict(frame, classes=[BALL_CLASS], conf=self.conf, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        H, W = frame.shape[:2]
        out = []
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            w, h = x2 - x1, y2 - y1
            if not (2 <= w <= 0.05 * W and 2 <= h <= 0.05 * W) or max(w, h) > 2.2 * min(w, h):
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # heads / upper bodies are the classic false positive; feet are fine
            if any(bx1 <= cx <= bx2 and by1 <= cy <= by1 + 0.55 * (by2 - by1)
                   for bx1, by1, bx2, by2 in player_bboxes):
                continue
            out.append((float(cx), float(cy), float(c)))
        out.sort(key=lambda t: -t[2])
        return out[: self.top_k]


def select_ball_path(
    candidates: dict[int, list[tuple[float, float, float]]],
    ref_candidates: dict[int, list[tuple[float, float]]],
    pitch_candidates: dict[int, list[tuple[float, float] | None]],
    fps: float,
    player_xy: dict[int, list[tuple[float, float]]] | None = None,
    max_speed_mps: float = 35.0,
    slack_m: float = 0.4,
    max_skip: int = 15,
    skip_cost: float = 0.12,
    reacquire_cost: float = 2.5,
    det_weight: float = 2.0,
    far_from_players_m: float = 10.0,
    far_cost: float = 0.6,
    min_run: int = 6,
) -> dict[int, int | None]:
    """
    Gated tracker over pitch metres. A detection can follow one seen up to `max_skip` frames
    earlier only if the implied speed is <= max_speed_mps (+ homography slack), so the track
    remembers where the ball was through short misses instead of jumping to any white spot.
    Starting a new track from nothing ("lost") costs `reacquire_cost`.
    pitch_candidates[f][j] is None for off-pitch candidates (never chosen).
    Returns frame -> chosen candidate index (or None).
    """
    frames = sorted(candidates)
    if not frames:
        return {}
    pos = {f: i for i, f in enumerate(frames)}
    vmax = max_speed_mps / fps
    static = static_candidate_mask(ref_candidates)

    def unary(f, j):
        if static[f][j] or pitch_candidates[f][j] is None:
            return np.inf
        c = -det_weight * candidates[f][j][2]
        if player_xy and player_xy.get(f):
            x, y = pitch_candidates[f][j]
            if min(np.hypot(x - px, y - py) for px, py in player_xy[f]) > far_from_players_m:
                c += far_cost
        return c

    cost: dict[int, np.ndarray] = {}
    back: dict[int, list] = {}
    lost = 0.0  # best cost of being "lost" at the current frame
    lost_hist = []  # (cost, pointer) of lost state per frame index
    lost_ptr = None
    for i, f in enumerate(frames):
        n = len(candidates[f])
        c = np.full(n, np.inf)
        bp = [None] * n
        for j in range(n):
            u = unary(f, j)
            if not np.isfinite(u):
                continue
            best, arg = lost + reacquire_cost, ("lost", i)
            x, y = pitch_candidates[f][j]
            for ii in range(max(0, i - max_skip), i):
                fp = frames[ii]
                gap = f - fp
                if gap > max_skip:
                    continue
                for k, pk in enumerate(pitch_candidates[fp]):
                    if pk is None or not np.isfinite(cost[fp][k]):
                        continue
                    d = np.hypot(x - pk[0], y - pk[1])
                    if d > vmax * gap + slack_m:
                        continue
                    v = cost[fp][k] + skip_cost * (gap - 1) + 0.5 * d / (vmax * gap + slack_m)
                    if v < best:
                        best, arg = v, (fp, k)
            c[j], bp[j] = best + u, arg
        cost[f], back[f] = c, bp
        lost_hist.append((lost, lost_ptr))
        # a track may end here -> lost from the next frame on
        nxt = (lost, lost_ptr)
        if n and np.isfinite(c).any():
            k = int(np.argmin(c))
            if c[k] < nxt[0]:
                nxt = (float(c[k]), (f, k))
        lost, lost_ptr = nxt

    # end: best of lost or any detection in the last max_skip frames
    end_best, end_arg = lost, ("lost", len(frames) - 1)
    lf = frames[-1]
    for fp in frames[max(0, len(frames) - max_skip):]:
        for k in range(len(cost[fp])):
            v = cost[fp][k] + skip_cost * (lf - fp)
            if v < end_best:
                end_best, end_arg = v, (fp, k)

    path = {f: None for f in frames}
    runs, run = [], []
    node = end_arg
    while node is not None:
        if node[0] == "lost":
            if run:
                runs.append(run)
                run = []
            node = lost_hist[node[1]][1]
            continue
        fp, k = node
        path[fp] = k
        run.append(fp)
        node = back[fp][k]
    if run:
        runs.append(run)
    # drop short tracks: a real ball persists, flickers don't
    for r in runs:
        if len(r) < min_run:
            for g in r:
                path[g] = None
    return path


def static_candidate_mask(ref_candidates: dict, cell_px: float = 8.0, max_presence: float = 0.15) -> dict:
    """
    Field markings, spots, signage get detected as 'ball' in the same (camera-compensated) place
    over and over. A real ball in play doesn't sit still for >15% of the clip -> suppress those cells.
    """
    from collections import Counter
    n = max(1, len(ref_candidates))
    cells = Counter()
    for pts in ref_candidates.values():
        seen = {(int(x // cell_px), int(y // cell_px)) for x, y in pts}
        cells.update(seen)
    def is_static(x, y):
        cx, cy = int(x // cell_px), int(y // cell_px)
        hits = sum(cells.get((cx + dx, cy + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        return hits / n > max_presence
    return {f: [is_static(x, y) for x, y in pts] for f, pts in ref_candidates.items()}
