"""
Phase 3: Structured Football Event Detection Engine.
Translates spatial coordinates and ball trajectories into structured football events:
- Robust possession state machine: requires the ball to STOP, DECELERATE, or CHANGE DIRECTION
  in a player's immediate vicinity (to avoid counting fly-by through-balls as receptions).
- Pass detection: possession release -> ball vector -> confirmed reception/possession by receiver.
- Turnovers: pass release -> confirmed reception by opponent.
- Progressive Carries / Dribbles: sustained control advancing across pitch.
- Shots on goal.
"""

import numpy as np


class EventDetector:
    def __init__(
        self,
        fps: float = 30.0,
        possession_dist_thresh_m: float = 3.6,  # Realistic threshold accounting for homography calibration tolerance
        min_possession_frames: int = 4,         # ~0.13s stability window
        pass_min_dist_m: float = 3.5,           # Minimum pass length
        dribble_min_dist_m: float = 3.5,        # Minimum carry/dribble length
        shot_speed_thresh_mps: float = 10.0,    # >= 36 km/h
        pitch_length: float = 90.0,
        pitch_width: float = 45.0,
    ):
        self.fps = fps
        self.possession_dist_thresh = possession_dist_thresh_m
        self.min_possession_frames = min_possession_frames
        self.pass_min_dist = pass_min_dist_m
        self.dribble_min_dist = dribble_min_dist_m
        self.shot_speed_thresh = shot_speed_thresh_mps
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def detect_events(self, telemetry: dict) -> list[dict]:
        """
        Process frame telemetry to detect passes, turnovers, dribbles, and shots.
        Enforces ball deceleration/direction change before confirming reception.
        """
        frames_dict = telemetry.get("frames", {})
        sorted_frame_indices = sorted([int(k) for k in frames_dict.keys()])
        if len(sorted_frame_indices) < 10:
            return []

        # 1. Compute ball velocity vector (vx, vy) at every frame
        ball_vel = {}
        for idx in range(len(sorted_frame_indices) - 1):
            f_curr = sorted_frame_indices[idx]
            f_next = sorted_frame_indices[idx + 1]
            b1 = frames_dict[str(f_curr)].get("ball")
            b2 = frames_dict[str(f_next)].get("ball")
            if b1 and b2 and b1.get("pitch_xy") and b2.get("pitch_xy"):
                dt = (f_next - f_curr) / self.fps
                vx = (b2["pitch_xy"][0] - b1["pitch_xy"][0]) / dt
                vy = (b2["pitch_xy"][1] - b1["pitch_xy"][1]) / dt
                speed = np.hypot(vx, vy)
                ball_vel[f_curr] = {"vx": vx, "vy": vy, "speed": speed}

        # 2. Determine instantaneous possession for each frame
        # Rule: A player has possession IF:
        #   (a) Ball is within possession_dist_thresh (<= 2.5m), AND
        #   (b) The ball is NOT simply flying past at high speed without decelerating or changing direction.
        possession_timeline = {}

        for i, f_idx in enumerate(sorted_frame_indices):
            fd = frames_dict[str(f_idx)]
            ball = fd.get("ball")
            players = fd.get("players", [])

            if ball is None or not ball.get("pitch_xy") or not players:
                possession_timeline[f_idx] = None
                continue

            bx, by = ball["pitch_xy"]
            closest_p = None
            min_dist = float("inf")

            for p in players:
                px, py = p["pitch_xy"]
                dist = np.hypot(px - bx, py - by)
                if dist < min_dist:
                    min_dist = dist
                    closest_p = p

            if closest_p is not None and min_dist <= self.possession_dist_thresh:
                # Check ball velocity dynamics near this player
                vel_info = ball_vel.get(f_idx)
                b_speed = vel_info["speed"] if vel_info else 0.0

                # If ball speed is fast (> 7.5 m/s = 27 km/h), check if ball decelerates or changes angle
                is_flyby = False
                if b_speed > 7.5 and i >= 2 and i < len(sorted_frame_indices) - 2:
                    prev_f = sorted_frame_indices[i - 2]
                    next_f = sorted_frame_indices[i + 2]
                    v_prev = ball_vel.get(prev_f)
                    v_next = ball_vel.get(next_f)
                    if v_prev and v_next:
                        # Angle dot product
                        dot = (v_prev["vx"] * v_next["vx"] + v_prev["vy"] * v_next["vy"]) / (v_prev["speed"] * v_next["speed"] + 1e-5)
                        # dot > 0.85 means ball continued in virtually a straight line at high speed (fly-by!)
                        speed_ratio = v_next["speed"] / (v_prev["speed"] + 1e-5)
                        if dot > 0.85 and speed_ratio > 0.70:
                            is_flyby = True

                if not is_flyby:
                    possession_timeline[f_idx] = {
                        "player_id": closest_p["tracker_id"],
                        "team_id": closest_p.get("possession_team", closest_p.get("team_id", 0)),
                        "player_xy": closest_p["pitch_xy"],
                        "ball_xy": [bx, by],
                        "dist": min_dist
                    }
                else:
                    possession_timeline[f_idx] = None
            else:
                possession_timeline[f_idx] = None

        # 3. Extract stable possession intervals
        stable_intervals = []
        current_poss = None
        count = 0
        start_f = None
        start_xy = None

        for f_idx in sorted_frame_indices:
            poss = possession_timeline.get(f_idx)
            if poss is not None:
                pid = poss["player_id"]
                if current_poss is None:
                    current_poss = pid
                    count = 1
                    start_f = f_idx
                    start_xy = poss["player_xy"]
                elif current_poss == pid:
                    count += 1
                else:
                    if count >= self.min_possession_frames:
                        end_xy = possession_timeline[f_idx - 1]["player_xy"]
                        team_id = possession_timeline[f_idx - 1]["team_id"]
                        stable_intervals.append({
                            "player_id": current_poss,
                            "team_id": team_id,
                            "start_frame": start_f,
                            "end_frame": f_idx - 1,
                            "start_xy": start_xy,
                            "end_xy": end_xy,
                        })
                    current_poss = pid
                    count = 1
                    start_f = f_idx
                    start_xy = poss["player_xy"]
            else:
                if current_poss is not None and count >= self.min_possession_frames:
                    end_xy = possession_timeline[f_idx - 1]["player_xy"]
                    team_id = possession_timeline[f_idx - 1]["team_id"]
                    stable_intervals.append({
                        "player_id": current_poss,
                        "team_id": team_id,
                        "start_frame": start_f,
                        "end_frame": f_idx - 1,
                        "start_xy": start_xy,
                        "end_xy": end_xy,
                    })
                current_poss = None
                count = 0
                start_f = None
                start_xy = None

        if current_poss is not None and count >= self.min_possession_frames:
            end_xy = possession_timeline[sorted_frame_indices[-1]]["player_xy"]
            team_id = possession_timeline[sorted_frame_indices[-1]]["team_id"]
            stable_intervals.append({
                "player_id": current_poss,
                "team_id": team_id,
                "start_frame": start_f,
                "end_frame": sorted_frame_indices[-1],
                "start_xy": start_xy,
                "end_xy": end_xy,
            })

        events = []
        event_id_counter = 1

        # 4. Dribble / Progressive Carry Detection
        for inv in stable_intervals:
            s_xy = inv["start_xy"]
            e_xy = inv["end_xy"]
            carry_dist = np.hypot(e_xy[0] - s_xy[0], e_xy[1] - s_xy[1])
            duration_sec = (inv["end_frame"] - inv["start_frame"]) / self.fps

            if carry_dist >= self.dribble_min_dist and duration_sec >= 0.7:
                events.append({
                    "id": event_id_counter,
                    "type": "dribble",
                    "label": "Progressive Carry / Dribble",
                    "color": "#42a5f5",  # Blue
                    "player_id": inv["player_id"],
                    "team_id": inv["team_id"],
                    "start_frame": inv["start_frame"],
                    "end_frame": inv["end_frame"],
                    "timestamp": round(inv["start_frame"] / self.fps, 2),
                    "start_xy": s_xy,
                    "end_xy": e_xy,
                    "distance_m": round(carry_dist, 1),
                    "duration_s": round(duration_sec, 2),
                })
                event_id_counter += 1

        # 5. Pass & Turnover Detection
        for i in range(len(stable_intervals) - 1):
            inv_a = stable_intervals[i]
            inv_b = stable_intervals[i + 1]

            flight_frames = inv_b["start_frame"] - inv_a["end_frame"]
            flight_time_sec = flight_frames / self.fps

            # Pass flight window: 0.15s to 3.0s
            if 0 < flight_time_sec <= 3.0:
                pass_dist = np.hypot(
                    inv_b["start_xy"][0] - inv_a["end_xy"][0],
                    inv_b["start_xy"][1] - inv_a["end_xy"][1]
                )

                same_team = inv_a["team_id"] == inv_b["team_id"]
                if pass_dist >= self.pass_min_dist or not same_team:
                    if same_team:
                        event_type, label, color = "pass_completed", "Completed Pass", "#66bb6a"
                    elif pass_dist >= self.pass_min_dist:
                        # ball travelled (attempted pass) and an opponent took it
                        event_type, label, color = "interception", "Interception", "#ef5350"
                    else:
                        # possession changed at close range: duel / tackle
                        event_type, label, color = "tackle", "Tackle / Duel Won", "#ff9800"

                    events.append({
                        "id": event_id_counter,
                        "type": event_type,
                        "label": label,
                        "color": color,
                        "passer_id": inv_a["player_id"],
                        "receiver_id": inv_b["player_id"],
                        "team_id": inv_a["team_id"],
                        "start_frame": inv_a["end_frame"],
                        "end_frame": inv_b["start_frame"],
                        "timestamp": round(inv_a["end_frame"] / self.fps, 2),
                        "start_xy": inv_a["end_xy"],
                        "end_xy": inv_b["start_xy"],
                        "distance_m": round(pass_dist, 1),
                        "duration_s": round(flight_time_sec, 2),
                    })
                    event_id_counter += 1

        # 6. Shot detection
        for f in range(sorted_frame_indices[0], sorted_frame_indices[-1] - 5, 2):
            b_curr = frames_dict[str(f)].get("ball")
            b_next = frames_dict[str(f + 4)].get("ball")
            if b_curr and b_next and b_curr.get("pitch_xy") and b_next.get("pitch_xy"):
                dx = b_next["pitch_xy"][0] - b_curr["pitch_xy"][0]
                dy = b_next["pitch_xy"][1] - b_curr["pitch_xy"][1]
                dist = np.hypot(dx, dy)
                speed = dist / (4.0 / self.fps)

                bx = b_next["pitch_xy"][0]
                by = b_next["pitch_xy"][1]
                in_goal_mouth = (bx <= 14.0 or bx >= self.pitch_length - 14.0) and (12.0 <= by <= self.pitch_width - 12.0)

                if speed >= self.shot_speed_thresh and in_goal_mouth:
                    ts = round(f / self.fps, 2)
                    already = any(e["type"] == "shot" and abs(e["timestamp"] - ts) < 2.0 for e in events)
                    if not already:
                        events.append({
                            "id": event_id_counter,
                            "type": "shot",
                            "label": "Shot on Goal",
                            "color": "#ffd54f",
                            "start_frame": f,
                            "end_frame": f + 15,
                            "timestamp": ts,
                            "start_xy": b_curr["pitch_xy"],
                            "end_xy": b_next["pitch_xy"],
                            "speed_kmh": round(speed * 3.6, 1),
                            "distance_m": round(dist, 1),
                        })
                        event_id_counter += 1

        events.sort(key=lambda e: e["timestamp"])
        return events
