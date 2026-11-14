"""
Post-processing trajectory stitching, gap interpolation, and anti-jitter smoothing.
Enforces physical object permanence across tracking outputs:
- Micro-track drop (< 15 frames)
- Spatio-temporal tracklet stitching (speed-radius constraint: dist <= max_speed * dt)
- Gap interpolation across temporary occlusions
- Savitzky-Golay / One-Euro anti-jitter smoothing on 2D pitch coordinates
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def stitch_and_clean_trajectories(
    frames_dict: dict,
    fps: float = 30.0,
    min_track_len: int = 15,
    max_stitch_dt_sec: float = 3.0,
    max_sprint_speed_mps: float = 9.0,
    max_body_heights_per_sec: float = 6.0,
    pitch_length: float = 90.0,
    pitch_width: float = 45.0,
    touchline_buffer: float = 0.5,
) -> dict:
    """
    Apply post-processing rules to all player tracks across frames.

    Args:
        frames_dict: {frame_idx_str: {"players": [...], "ball": ...}}
        fps: Video framerate.
        min_track_len: Minimum consecutive detections before discarding as micro-track.
        max_stitch_dt_sec: Maximum time gap allowed to stitch broken track segments.
        max_sprint_speed_mps: Maximum physically plausible sprint speed (e.g., 9 m/s = 32.4 km/h).
        pitch_length: Length of pitch in meters.
        pitch_width: Width of pitch in meters.
        touchline_buffer: Margin beyond touchline where track exits are allowed without stitching.

    Returns:
        Updated frames_dict with stitched IDs, gap-interpolated positions, and smoothed coordinates.
    """
    sorted_frame_indices = sorted([int(k) for k in frames_dict.keys()])
    if not sorted_frame_indices:
        return frames_dict

    # 1. Gather per-track history: track_id -> dict of frame_idx -> player_dict
    track_history: dict[int, dict[int, dict]] = {}
    for f_idx in sorted_frame_indices:
        for p in frames_dict[str(f_idx)]["players"]:
            tid = p["tracker_id"]
            if tid <= 0:
                continue
            if tid not in track_history:
                track_history[tid] = {}
            track_history[tid][f_idx] = p

    # 2. Micro-tracks are dropped only AFTER stitching: fragments left by close contact
    #    are usually the same player and must be re-joined, not deleted.

    # Helper: Check if point is near the boundary/exit
    def is_near_exit(x: float, y: float) -> bool:
        return (
            x <= touchline_buffer or x >= (pitch_length - touchline_buffer) or
            y <= touchline_buffer or y >= (pitch_width - touchline_buffer)
        )

    # 3. Spatio-temporal tracklet stitching
    # Sort tracks by their first appearance frame
    track_meta = []
    for tid, occ in track_history.items():
        frames = sorted(occ.keys())
        first_f, last_f = frames[0], frames[-1]
        start_pos = occ[first_f]["pitch_xy"]
        end_pos = occ[last_f]["pitch_xy"]
        track_meta.append({
            "tid": tid,
            "first_f": first_f,
            "last_f": last_f,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "start_box": occ[first_f].get("bbox"),
            "end_box": occ[last_f].get("bbox"),
            "len": len(frames)
        })

    track_meta.sort(key=lambda t: t["first_f"])

    # Iteratively stitch Track B into Track A if Track A ended and Track B spawned within reach
    alias_map: dict[int, int] = {}  # old_tid -> new_tid

    def resolve_alias(tid: int) -> int:
        curr = tid
        while curr in alias_map:
            curr = alias_map[curr]
        return curr

    max_frame_gap = int(max_stitch_dt_sec * fps)

    for i in range(len(track_meta)):
        curr_track = track_meta[i]
        curr_id = resolve_alias(curr_track["tid"])
        curr_end_pos = curr_track["end_pos"]
        curr_end_f = curr_track["last_f"]

        # If current track ended right at the boundary/bench, it likely subbed out
        if is_near_exit(curr_end_pos[0], curr_end_pos[1]):
            continue

        best_match_idx = None
        min_distance = float("inf")

        for j in range(i + 1, len(track_meta)):
            candidate = track_meta[j]
            cand_id = resolve_alias(candidate["tid"])
            if cand_id == curr_id:
                continue

            frame_gap = candidate["first_f"] - curr_end_f
            if frame_gap <= 0:
                continue  # Overlapping tracks cannot be the same single person
            if frame_gap > max_frame_gap:
                break  # Beyond time threshold

            dt_sec = frame_gap / fps
            max_reachable_dist = max_sprint_speed_mps * dt_sec

            dx = candidate["start_pos"][0] - curr_end_pos[0]
            dy = candidate["start_pos"][1] - curr_end_pos[1]
            dist = np.sqrt(dx * dx + dy * dy)

            # Near the camera homography error inflates metres; also accept a pixel gate
            # measured in body heights (a sprinter covers <= ~6 heights/s on screen).
            reachable = dist <= max_reachable_dist
            eb, sb = curr_track.get("end_box"), candidate.get("start_box")
            if not reachable and eb and sb:
                h = max(eb[3] - eb[1], sb[3] - sb[1])
                px = np.hypot((sb[0] + sb[2] - eb[0] - eb[2]) / 2, sb[3] - eb[3])
                reachable = px <= max_body_heights_per_sec * h * max(dt_sec, 0.2)
            if reachable and dist < min_distance:
                min_distance = dist
                best_match_idx = j

        if best_match_idx is not None:
            cand_track = track_meta[best_match_idx]
            cand_old_id = cand_track["tid"]
            target_id = curr_id

            if cand_old_id in track_history:
                # Merge track history
                alias_map[cand_old_id] = target_id
                for f_idx, p_data in track_history[cand_old_id].items():
                    p_copy = dict(p_data)
                    p_copy["tracker_id"] = target_id
                    track_history[target_id][f_idx] = p_copy
                del track_history[cand_old_id]

                # Update meta of target track
                curr_track["last_f"] = cand_track["last_f"]
                curr_track["end_pos"] = cand_track["end_pos"]
                curr_track["end_box"] = cand_track["end_box"]
                curr_track["len"] += cand_track["len"]

    track_history = {tid: occ for tid, occ in track_history.items() if len(occ) >= min_track_len}

    # 4. Trajectory gap interpolation & Savitzky-Golay smoothing for each continuous track
    for tid, occ in track_history.items():
        frames = sorted(occ.keys())
        if len(frames) < 2:
            continue

        first_f, last_f = frames[0], frames[-1]
        all_track_frames = list(range(first_f, last_f + 1))

        # Check for gaps < 2.0s (120 frames at 60fps / 60 frames at 30fps)
        max_interp_gap = int(2.0 * fps)
        
        # Build coordinate arrays for interpolation
        det_frames = np.array(frames, dtype=np.float64)
        xs = np.array([occ[f]["pitch_xy"][0] for f in frames], dtype=np.float64)
        ys = np.array([occ[f]["pitch_xy"][1] for f in frames], dtype=np.float64)
        px_xs = np.array([occ[f]["foot_pixel"][0] for f in frames], dtype=np.float64)
        px_ys = np.array([occ[f]["foot_pixel"][1] for f in frames], dtype=np.float64)

        interp_x = interp1d(det_frames, xs, kind='linear', fill_value='extrapolate')
        interp_y = interp1d(det_frames, ys, kind='linear', fill_value='extrapolate')
        interp_px_x = interp1d(det_frames, px_xs, kind='linear', fill_value='extrapolate')
        interp_px_y = interp1d(det_frames, px_ys, kind='linear', fill_value='extrapolate')

        # Fill internal missing frames if gap <= max_interp_gap
        for k in range(len(frames) - 1):
            gap = frames[k + 1] - frames[k] - 1
            if 0 < gap <= max_interp_gap:
                for f_missing in range(frames[k] + 1, frames[k + 1]):
                    # Interpolated bbox from nearest anchor
                    # Translate the anchor bbox with the interpolated foot so boxes don't freeze in gaps.
                    anchor = occ[frames[k]]
                    interp_foot = [float(interp_px_x(f_missing)), float(interp_px_y(f_missing))]
                    dx = interp_foot[0] - anchor["foot_pixel"][0]
                    dy = interp_foot[1] - anchor["foot_pixel"][1]
                    x1, y1, x2, y2 = anchor["bbox"]
                    ref_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                    occ[f_missing] = {
                        "tracker_id": tid,
                        "bbox": ref_bbox,
                        "foot_pixel": interp_foot,
                        "pitch_xy": [float(interp_x(f_missing)), float(interp_y(f_missing))],
                        "confidence": float(occ[frames[k]]["confidence"]),
                        "interpolated": True
                    }

        # 5. Savitzky-Golay smoothing on the continuous trajectory
        final_frames = sorted(occ.keys())
        if len(final_frames) >= 7:
            x_vals = np.array([occ[f]["pitch_xy"][0] for f in final_frames])
            y_vals = np.array([occ[f]["pitch_xy"][1] for f in final_frames])

            # Window length must be odd, <= series length
            window_len = min(11, len(final_frames))
            if window_len % 2 == 0:
                window_len -= 1

            if window_len >= 5:
                smooth_x = savgol_filter(x_vals, window_length=window_len, polyorder=2)
                smooth_y = savgol_filter(y_vals, window_length=window_len, polyorder=2)
                for idx, f in enumerate(final_frames):
                    occ[f]["pitch_xy"] = [float(smooth_x[idx]), float(smooth_y[idx])]

    # 6. Rebuild frames_dict with cleaned and smoothed player detections
    cleaned_frames = {}
    for f_idx in sorted_frame_indices:
        f_str = str(f_idx)
        ball_data = frames_dict[f_str].get("ball")
        players_in_frame = []
        for tid, occ in track_history.items():
            if f_idx in occ:
                players_in_frame.append(occ[f_idx])

        cleaned_frames[f_str] = {
            "players": players_in_frame,
            "ball": ball_data
        }

    return cleaned_frames

