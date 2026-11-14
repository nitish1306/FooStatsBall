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
    frame_width: int,
    max_speed_px_per_frame: float | None = None,
    miss_cost: float = 0.0,
    reacquire_cost: float = 1.5,
    det_weight: float = 2.0,
    min_run: int = 6,
) -> dict[int, int | None]:
    """
    Viterbi over frames. State = candidate index or None (miss).
    Motion cost uses camera-compensated (reference) pixels.
    Returns frame -> chosen candidate index (or None).
    """
    frames = sorted(candidates)
    if not frames:
        return {}
    vmax = max_speed_px_per_frame or frame_width * 0.06  # ~75px/frame at 1280 ≈ very hard shot near camera

    static = static_candidate_mask(ref_candidates)

    def det_cost(c):
        return -det_weight * c  # reward: confident detections pull the path; miss costs 0

    # cost arrays: last element = miss state
    def cand_cost(f, j):
        return np.inf if static[f][j] else det_cost(candidates[f][j][2])

    prev_cost = np.array([cand_cost(frames[0], j) for j in range(len(candidates[frames[0]]))] + [miss_cost])
    back = []
    for i in range(1, len(frames)):
        f, fp = frames[i], frames[i - 1]
        cands, prev_ref = candidates[f], ref_candidates[fp]
        cur_ref = ref_candidates[f]
        n_prev, n_cur = len(prev_ref), len(cands)
        cost = np.full(n_cur + 1, np.inf)
        ptr = np.zeros(n_cur + 1, dtype=int)
        gap = f - fp
        for j in range(n_cur):
            best, arg = prev_cost[-1] + reacquire_cost, n_prev  # from miss
            for k in range(n_prev):
                d = np.hypot(cur_ref[j][0] - prev_ref[k][0], cur_ref[j][1] - prev_ref[k][1]) / gap
                if d > vmax:
                    continue
                v = prev_cost[k] + 0.5 * d / vmax
                if v < best:
                    best, arg = v, k
            cost[j], ptr[j] = best + cand_cost(f, j), arg
        k = int(np.argmin(prev_cost))
        cost[-1], ptr[-1] = prev_cost[k] + miss_cost, k
        back.append(ptr)
        # normalize to avoid drift
        prev_cost = cost - cost.min()

    path, run_start = {}, {}
    state = int(np.argmin(prev_cost))
    for i in range(len(frames) - 1, -1, -1):
        f = frames[i]
        n = len(candidates[f])
        path[f] = state if state < n else None
        if i > 0:
            prev_state = int(back[i - 1][state])
            # a run starts here if we reached this detection from the miss state
            run_start[f] = path[f] is not None and prev_state >= len(candidates[frames[i - 1]])
            state = prev_state
        else:
            run_start[f] = path[f] is not None
    # drop short isolated runs: continuous tracks of a real ball last; FP flickers don't
    run = []
    for f in frames + [None]:
        if f is None or path[f] is None or run_start[f]:
            if 0 < len(run) < min_run:
                for g in run:
                    path[g] = None
            run = []
        if f is not None and path[f] is not None:
            run.append(f)
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
