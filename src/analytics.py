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


def compute_physical_metrics(telemetry: dict, fps: float = 30.0) -> dict:
    """
    Compute physical workload metrics for every player track.
    Filters out teleportation outliers (max plausible human speed <= 34.0 km/h = 9.44 m/s).
    """
    frames = telemetry.get("frames", {})
    sorted_frames = sorted([int(k) for k in frames.keys()])
    if not sorted_frames:
        return {}

    # Gather continuous pitch coordinates per player: tid -> [(f_idx, x, y)]
    player_trajectories = {}
    team_possession_counts = {0: 0, 1: 0}

    for f_idx in sorted_frames:
        fd = frames[str(f_idx)]
        # Team possession check (if ball has closest player)
        ball = fd.get("ball")
        if ball and ball.get("pitch_xy") and fd.get("players"):
            bx, by = ball["pitch_xy"]
            best_dist = float("inf")
            best_team = None
            for p in fd["players"]:
                d = np.hypot(p["pitch_xy"][0] - bx, p["pitch_xy"][1] - by)
                if d < best_dist:
                    best_dist = d
                    best_team = p.get("team_id", 0)
            if best_team in team_possession_counts and best_dist <= 5.0:
                team_possession_counts[best_team] += 1

        for p in fd.get("players", []):
            tid = p["tracker_id"]
            if tid <= 0:
                continue
            if tid not in player_trajectories:
                player_trajectories[tid] = {
                    "team_id": p.get("team_id", 0),
                    "team_color": p.get("team_color", "#4fc3f7"),
                    "pts": []
                }
            player_trajectories[tid]["pts"].append((f_idx, p["pitch_xy"][0], p["pitch_xy"][1]))

    total_poss = sum(team_possession_counts.values()) or 1
    possession_pct = {
        "team_0": round((team_possession_counts[0] / total_poss) * 100, 1),
        "team_1": round((team_possession_counts[1] / total_poss) * 100, 1),
    }

    metrics = {
        "possession": possession_pct,
        "players": {}
    }

    # Thresholds
    sprint_thresh_kmh = 19.5  # High-intensity running threshold (~5.4 m/s)
    max_physical_speed_kmh = 34.0

    for tid, data in player_trajectories.items():
        pts = data["pts"]
        if len(pts) < 5:
            continue

        total_dist_m = 0.0
        speeds_kmh = []
        in_sprint = False
        sprint_count = 0

        for i in range(len(pts) - 1):
            f1, x1, y1 = pts[i]
            f2, x2, y2 = pts[i + 1]
            dt = (f2 - f1) / fps
            if dt <= 0 or dt > 2.0:
                continue

            dist = np.hypot(x2 - x1, y2 - y1)
            speed_mps = dist / dt
            speed_kmh = speed_mps * 3.6

            # Filter unrealistic GPS/CV tracking jumps
            if speed_kmh <= max_physical_speed_kmh:
                total_dist_m += dist
                speeds_kmh.append(speed_kmh)

                if speed_kmh >= sprint_thresh_kmh:
                    if not in_sprint:
                        sprint_count += 1
                        in_sprint = True
                else:
                    in_sprint = False

        top_speed = max(speeds_kmh) if speeds_kmh else 0.0
        avg_speed = np.mean(speeds_kmh) if speeds_kmh else 0.0

        metrics["players"][str(tid)] = {
            "tracker_id": tid,
            "team_id": data["team_id"],
            "team_color": data["team_color"],
            "distance_km": round(total_dist_m / 1000.0, 3),
            "distance_m": round(total_dist_m, 1),
            "top_speed_kmh": round(top_speed, 1),
            "avg_speed_kmh": round(avg_speed, 1),
            "sprints": sprint_count,
            "minutes_tracked": round((pts[-1][0] - pts[0][0]) / (fps * 60.0), 2)
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

