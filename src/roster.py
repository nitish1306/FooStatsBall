"""Roster prior: suppress phantom tracks that spawn mid-pitch when a team is already full.

A new track appearing away from the touchlines (subs walk on from a line) is first re-attached to a
recently lost team-mate. If none fits and its team already has `players_per_team` active (soft cap),
it is dropped as a duplicate box, or dropped if short; otherwise kept.
"""
import numpy as np


def _centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def apply_roster(
    frames: dict,
    players_per_team: int,
    fps: float,
    pitch_length: float,
    pitch_width: float,
    edge_m: float = 3.0,
    lost_sec: float = 3.0,
    min_sec: float = 1.0,
    max_speed_mps: float = 9.0,
    max_body_heights_per_sec: float = 6.0,
    dup_ratio: float = 0.5,
) -> dict:
    """Returns frames with phantom tracks removed / re-labelled. Mutates and returns `frames`."""
    if not players_per_team:
        return frames
    keys = sorted(frames, key=int)
    tracks: dict[int, dict[int, dict]] = {}
    for k in keys:
        for p in frames[k]["players"]:
            tracks.setdefault(p["tracker_id"], {})[int(k)] = p

    def team(tid):
        ids = [p.get("team_id", -1) for p in tracks[tid].values()]
        return max(set(ids), key=ids.count)

    def near_edge(xy):
        x, y = xy
        return x <= edge_m or y <= edge_m or x >= pitch_length - edge_m or y >= pitch_width - edge_m

    meta = {t: {"first": min(o), "last": max(o), "team": team(t)} for t, o in tracks.items()}
    alias, drop = {}, set()
    first_frame = int(keys[0]) if keys else 0

    for tid in sorted(tracks, key=lambda t: meta[t]["first"]):
        m = meta[tid]
        f0, T = m["first"], m["team"]
        if T < 0 or f0 == first_frame:
            continue
        occ = tracks[tid]
        p0 = occ[f0]
        live = [t for t in tracks if t != tid and t not in drop and t not in alias
                and meta[t]["team"] == T and meta[t]["first"] < f0 <= meta[t]["last"]]
        if near_edge(p0["pitch_xy"]):
            continue
        full = len(live) >= players_per_team

        # 1. re-attach to a team-mate lost recently, within motion reach
        best, best_d = None, float("inf")
        for t in tracks:
            if t == tid or t in drop or t in alias or meta[t]["team"] != T:
                continue
            gap = f0 - meta[t]["last"]
            if not (0 < gap <= lost_sec * fps):
                continue
            q = tracks[t][meta[t]["last"]]
            dt = gap / fps
            d = float(np.hypot(p0["pitch_xy"][0] - q["pitch_xy"][0], p0["pitch_xy"][1] - q["pitch_xy"][1]))
            ok = d <= max_speed_mps * dt
            if not ok:
                h = max(q["bbox"][3] - q["bbox"][1], p0["bbox"][3] - p0["bbox"][1])
                px = np.hypot(*np.subtract(_centre(p0["bbox"]), _centre(q["bbox"])))
                ok = px <= max_body_heights_per_sec * h * max(dt, 0.2)
            if ok and d < best_d:
                best, best_d = t, d
        if best is not None:
            alias[tid] = best
            for f, p in occ.items():
                p["tracker_id"] = best
                tracks[best][f] = p
            meta[best]["last"] = max(meta[best]["last"], m["last"])
            continue
        if not full:
            continue

        # 2. duplicate: centre sits inside another live player's box most of its life
        boxes = {int(k): [p["bbox"] for p in frames[str(k)]["players"]
                          if p["tracker_id"] not in (tid,) and p["tracker_id"] not in drop]
                 for k in occ}
        hits = sum(any(_inside(_centre(occ[f]["bbox"]), b) for b in boxes[f]) for f in occ)
        if hits >= dup_ratio * len(occ):
            drop.add(tid)
            continue

        # 3. short-lived phantom
        if len(occ) < min_sec * fps:
            drop.add(tid)

    for k in keys:
        frames[k]["players"] = [p for p in frames[k]["players"]
                                if p["tracker_id"] not in drop]
    return frames


def drop_spectators(
    frames: dict,
    fps: float,
    pitch_length: float,
    pitch_width: float,
    touchline_m: float = 4.0,
    max_extent_m: float = 6.0,
    min_static_sec: float = 5.0,
    max_aspect: float = 0.8,
) -> dict:
    """Removes tracks that are not players: median position off the pitch, a wide box (seated people,
    groups), or standing still near a touchline (bench, tents). Goal lines are left alone for keepers."""
    tracks: dict[int, list] = {}
    for fd in frames.values():
        for p in fd["players"]:
            tracks.setdefault(p["tracker_id"], []).append(p)
    drop = set()
    for tid, ps in tracks.items():
        xy = np.array([p["pitch_xy"] for p in ps], dtype=float)
        b = np.array([p["bbox"] for p in ps], dtype=float)
        mx, my = np.median(xy, axis=0)
        aspect = np.median((b[:, 2] - b[:, 0]) / np.maximum(b[:, 3] - b[:, 1], 1e-6))
        off = mx < 0 or my < 0 or mx > pitch_length or my > pitch_width
        extent = float(np.hypot(*np.ptp(xy, axis=0)))
        static_at_line = (min(my, pitch_width - my) <= touchline_m and extent < max_extent_m
                          and len(ps) >= min_static_sec * fps)
        if off or aspect > max_aspect or static_at_line:
            drop.add(tid)
    for fd in frames.values():
        fd["players"] = [p for p in fd["players"] if p["tracker_id"] not in drop]
    return frames
