"""
Phase 4: Physical Workload & Tactical Analytics Engine.
Calculates:
- Player kinematics: Total Distance Covered (km), Instantaneous Speed (km/h), Top Sprint Speed, Sprint Count (>= 20 km/h)
- Team Possession Percentages
- 2D Gaussian Kernel Density Estimation (KDE) Heatmaps (per-player & per-team)
- Voronoi Space Control Tessellation
- Passing Network Adjacency Matrix
"""

import numpy as np
from scipy.ndimage import gaussian_filter


SPRINT_KMH = 20.0
SPRINT_MIN_S = 1.0
MAX_KMH = 36.0
SPEED_WINDOW_S = 0.5
POSSESSION_RADIUS_M = 3.0
POSSESSION_HOLD_S = 2.0


def _team_votes(frames: dict) -> dict:
    """Majority team (0/1) per tracker id, so GK/unknown frames inherit the player's team."""
    votes = {}
    for fd in frames.values():
        for p in fd.get("players", []):
            t = p.get("possession_team", p.get("team_id"))
            if t in (0, 1):
                v = votes.setdefault(p["tracker_id"], [0, 0])
                v[t] += 1
    return {tid: (0 if v[0] >= v[1] else 1) for tid, v in votes.items()}


def _possession(frames: dict, sorted_frames: list, fps: float, team_of: dict) -> dict:
    counts = {0: 0, 1: 0}
    hold = int(POSSESSION_HOLD_S * fps)
    last_team, last_f = None, -10**9
    ball_frames = 0
    for f in sorted_frames:
        fd = frames[str(f)]
        ball = fd.get("ball")
        if ball and ball.get("pitch_xy") and fd.get("players"):
            ball_frames += 1
            bx, by = ball["pitch_xy"]
            best = min(fd["players"], key=lambda p: np.hypot(p["pitch_xy"][0] - bx, p["pitch_xy"][1] - by))
            d = np.hypot(best["pitch_xy"][0] - bx, best["pitch_xy"][1] - by)
            team = team_of.get(best["tracker_id"], best.get("team_id"))
            if d <= POSSESSION_RADIUS_M and team in counts:
                last_team, last_f = team, f
        if last_team is not None and f - last_f <= hold:
            counts[last_team] += 1
    total = counts[0] + counts[1]
    pct = lambda n: round(n / total * 100, 1) if total else None
    return {
        "team_0": pct(counts[0]),
        "team_1": pct(counts[1]),
        "frames_attributed": total,
        "ball_coverage_pct": round(ball_frames / len(sorted_frames) * 100, 1),
    }


def _player_metrics(pts: np.ndarray, fps: float) -> dict | None:
    """pts: (N,3) of frame, x, y. Speed from displacement over a ~0.5 s window within contiguous segments."""
    f, xy = pts[:, 0], pts[:, 1:]
    breaks = np.where(np.diff(f) > fps * 0.5)[0] + 1
    win = max(1, int(round(SPEED_WINDOW_S * fps)))
    dist = 0.0
    tracked_frames = 0
    speeds, sprints, top = [], 0, 0.0
    for seg in np.split(np.arange(len(f)), breaks):
        if len(seg) < 2:
            continue
        sf, sxy = f[seg], xy[seg]
        tracked_frames += sf[-1] - sf[0]
        if len(seg) <= win:
            continue
        disp = np.hypot(*(sxy[win:] - sxy[:-win]).T)
        dt = (sf[win:] - sf[:-win]) / fps
        v = np.clip(disp / dt * 3.6, 0, MAX_KMH)
        speeds.append(v)
        # distance: sum of window-sampled displacement (suppresses jitter)
        idx = np.arange(0, len(seg), win)
        step = np.hypot(*np.diff(sxy[idx], axis=0).T)
        step_dt = np.diff(sf[idx]) / fps
        dist += float(np.sum(np.minimum(step, MAX_KMH / 3.6 * step_dt)))
        top = max(top, float(np.percentile(v, 98)) if len(v) >= 10 else float(v.max()))
        fast = v >= SPRINT_KMH
        min_len = int(SPRINT_MIN_S * fps)
        run = 0
        for is_fast in np.append(fast, False):
            if is_fast:
                run += 1
            else:
                if run >= min_len:
                    sprints += 1
                run = 0
    if not speeds:
        return None
    allv = np.concatenate(speeds)
    return {
        "distance_m": dist,
        "top_speed_kmh": top,
        "avg_speed_kmh": float(allv.mean()),
        "sprints": sprints,
        "minutes_tracked": tracked_frames / (fps * 60.0),
    }


def compute_physical_metrics(telemetry: dict, fps: float = 30.0) -> dict:
    """
    Physical workload + possession.
    - Speed: displacement over a 0.5 s window (pixel jitter averages out), clamped to 36 km/h.
    - Top speed: 98th percentile of windowed speed (robust to single-window spikes).
    - Sprint: >= 20 km/h sustained >= 1 s.
    - minutes_tracked: sum of contiguous segments (gaps excluded).
    - Possession: nearest player within 3 m, held up to 2 s between ball sightings; GK mapped to majority team.
    """
    frames = telemetry.get("frames", {})
    sorted_frames = sorted(int(k) for k in frames)
    if not sorted_frames:
        return {}
    fps = fps or 30.0
    team_of = _team_votes(frames)

    traj = {}
    for f in sorted_frames:
        for p in frames[str(f)].get("players", []):
            tid = p["tracker_id"]
            if tid <= 0:
                continue
            d = traj.setdefault(tid, {"team_color": p.get("team_color", "#4fc3f7"), "pts": []})
            d["pts"].append((f, p["pitch_xy"][0], p["pitch_xy"][1]))

    metrics = {"possession": _possession(frames, sorted_frames, fps, team_of), "players": {}}
    for tid, d in traj.items():
        if len(d["pts"]) < 5:
            continue
        m = _player_metrics(np.asarray(d["pts"], dtype=float), fps)
        if m is None:
            continue
        metrics["players"][str(tid)] = {
            "tracker_id": tid,
            "team_id": team_of.get(tid, -1),
            "team_color": d["team_color"],
            "distance_km": round(m["distance_m"] / 1000.0, 3),
            "distance_m": round(m["distance_m"], 1),
            "top_speed_kmh": round(m["top_speed_kmh"], 1),
            "avg_speed_kmh": round(m["avg_speed_kmh"], 1),
            "sprints": m["sprints"],
            "minutes_tracked": round(m["minutes_tracked"], 2),
        }
    return metrics


def generate_heatmap_grid(telemetry: dict, track_id: int | None = None, team_id: int | None = None, grid_size: tuple[int, int] = (90, 45)) -> list[list[float]]:
    """
    Generate a 2D Gaussian smoothed KDE heatmap grid across pitch coordinates.
    grid_size: (columns, rows)
    """
    cols, rows = grid_size
    grid = np.zeros((rows, cols), dtype=np.float32)

    L = telemetry.get("pitch", {}).get("length", 90.0)
    W = telemetry.get("pitch", {}).get("width", 45.0)

    for fd in telemetry.get("frames", {}).values():
        for p in fd.get("players", []):
            if track_id is not None and p["tracker_id"] != track_id:
                continue
            if team_id is not None and p.get("team_id") != team_id:
                continue

            x, y = p["pitch_xy"]
            if 0 <= x <= L and 0 <= y <= W:
                gx = int((x / L) * (cols - 1))
                gy = int((y / W) * (rows - 1))
                grid[gy, gx] += 1.0

    # Gaussian blur
    smoothed = gaussian_filter(grid, sigma=2.5)
    max_val = smoothed.max()
    if max_val > 0:
        smoothed = smoothed / max_val

    return smoothed.tolist()

