"""
Upgraded video processing pipeline with 7 object permanence upgrades:
1. Sports-tuned BoT-SORT configuration (extended memory, appearance ReID, no GMC)
2. High-resolution inference (imgsz=1280, FP16)
3. 2D pitch polygon boundary filtering (with 0.5m margin)
4. Player trajectory gap interpolation
5. Spatio-temporal tracklet stitcher & micro-track filtering
6. Savitzky-Golay anti-jitter trajectory smoothing
7. Scene-cut guard validation
"""

import json
import time
import numpy as np
import cv2
from pathlib import Path

from src.calibration import PitchCalibrator, get_video_info
from src.detection import DetectionTracker
from src.ball_interpolation import interpolate_ball_trajectory
from src.track_stitcher import stitch_and_clean_trajectories
from src.scene_guard import detect_scene_cuts


def process_video(
    video_path: str,
    calibration_path: str,
    output_path: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    detection_conf: float = 0.25,
    imgsz: int = 1280,
    device: str = "cuda",
    check_scene_cuts: bool = True,
    touchline_buffer: float = 0.5,
    progress_callback=None,
) -> dict:
    """
    Run detection, tracking, homography projection, and trajectory stitching upgrades.
    """
    # 7. Scene-cut guard / continuous clip check
    scene_cuts = []
    if check_scene_cuts:
        if progress_callback:
            progress_callback(0, 100, "Checking scene continuity...")
        scene_cuts = detect_scene_cuts(video_path, start_frame=start_frame, end_frame=end_frame)
        if len(scene_cuts) > 0:
            print(f"Warning: {len(scene_cuts)} potential scene cuts detected at frames: {scene_cuts}")

    # Load pitch calibration
    calibrator = PitchCalibrator()
    calibrator.load(calibration_path)
    pitch_len = calibrator.pitch_length
    pitch_wid = calibrator.pitch_width

    # Video info
    info = get_video_info(video_path)
    total_frames = info["total_frames"]
    fps = info["fps"]

    if end_frame is None:
        end_frame = total_frames - 1
    end_frame = min(end_frame, total_frames - 1)
    frames_to_process = end_frame - start_frame + 1

    # 1 & 2. Sports-tuned tracker with imgsz=1280 and FP16
    tracker = DetectionTracker(device=device, imgsz=imgsz)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_frames = {}
    ball_raw_positions = {}
    t_start = time.time()

    for current_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        players, ball_center = tracker.detect_and_track(frame, conf=detection_conf)

        # 3. 2D pitch polygon boundary filtering
        players_data = []
        if len(players) > 0:
            foot_pixels = np.array([p["foot_pixel"] for p in players], dtype=np.float32)
            pitch_pts = calibrator.pixel_to_pitch_batch(foot_pixels)

            for i, p in enumerate(players):
                mx, my = float(pitch_pts[i][0]), float(pitch_pts[i][1])

                # Discard detections falling outside the calibrated pitch rectangle (+ buffer)
                if (-touchline_buffer <= mx <= pitch_len + touchline_buffer and
                    -touchline_buffer <= my <= pitch_wid + touchline_buffer):
                    players_data.append({
                        "tracker_id": p["tracker_id"],
                        "bbox": p["bbox"],
                        "foot_pixel": p["foot_pixel"],
                        "pitch_xy": [mx, my],
                        "confidence": p["confidence"],
                    })

        # Ball projection & boundary check
        ball_data = None
        if ball_center is not None:
            ball_pitch = calibrator.pixel_to_pitch(ball_center[0], ball_center[1])
            bx, by = float(ball_pitch[0]), float(ball_pitch[1])
            if (-touchline_buffer <= bx <= pitch_len + touchline_buffer and
                -touchline_buffer <= by <= pitch_wid + touchline_buffer):
                ball_data = {
                    "pixel": ball_center,
                    "pitch_xy": [bx, by],
                }
                ball_raw_positions[current_idx] = (ball_center[0], ball_center[1])
            else:
                ball_raw_positions[current_idx] = None
        else:
            ball_raw_positions[current_idx] = None

        raw_frames[str(current_idx)] = {
            "players": players_data,
            "ball": ball_data,
        }

        if progress_callback and (current_idx - start_frame) % 15 == 0:
            progress_callback(
                current_idx - start_frame,
                frames_to_process,
                f"Tracking frame {current_idx}/{end_frame}"
            )

    cap.release()

    # Interpolate ball gaps
    interpolated_ball = interpolate_ball_trajectory(ball_raw_positions, max_gap=45)
    for frame_str, frame_data in raw_frames.items():
        f_idx = int(frame_str)
        if frame_data["ball"] is None and interpolated_ball.get(f_idx) is not None:
            bx, by = interpolated_ball[f_idx]
            bpitch = calibrator.pixel_to_pitch(bx, by)
            frame_data["ball"] = {
                "pixel": [bx, by],
                "pitch_xy": list(bpitch),
                "interpolated": True,
            }

    # 4, 5 & 6. Spatio-temporal stitching, micro-track dropping, player gap interpolation & Savitzky-Golay smoothing
    if progress_callback:
        progress_callback(frames_to_process, frames_to_process, "Stitching & smoothing player trajectories...")

    cleaned_frames = stitch_and_clean_trajectories(
        frames_dict=raw_frames,
        fps=fps,
        min_track_len=15,
        max_stitch_dt_sec=3.0,
        max_sprint_speed_mps=9.0,
        pitch_length=pitch_len,
        pitch_width=pitch_wid,
        touchline_buffer=touchline_buffer,
    )

    # Phase 2: Team separation pass (torso crop clustering + persistent track voting)
    if progress_callback:
        progress_callback(frames_to_process, frames_to_process, "Classifying team jersey colors...")

    from src.team_classifier import TeamClassifier
    classifier = TeamClassifier()
    temp_telemetry = {"frames": cleaned_frames}
    classifier.collect_features_from_video(video_path, temp_telemetry)
    classifier.fit_clusters(pitch_length=pitch_len, pitch_width=pitch_wid, telemetry=temp_telemetry)

    # Attach team_id and color to every player detection across all frames
    for fd in cleaned_frames.values():
        for p in fd["players"]:
            tid = p["tracker_id"]
            team_id = classifier.track_team_map.get(tid, 0)
            p["team_id"] = team_id
            p["team_color"] = classifier.team_colors.get(team_id, "#4fc3f7")

    # Phase 3: Structured football event detection (passes, turnovers, dribbles, shots)
    if progress_callback:
        progress_callback(frames_to_process, frames_to_process, "Detecting passes, carries & match events...")

    from src.event_detector import EventDetector
    detector = EventDetector(fps=fps, pitch_length=pitch_len, pitch_width=pitch_wid)
    detected_events = detector.detect_events({"frames": cleaned_frames})

    # Phase 4: Physical Workload & Tactical Analytics
    if progress_callback:
        progress_callback(frames_to_process, frames_to_process, "Computing physical metrics & heatmaps...")

    from src.analytics import compute_physical_metrics, generate_heatmap_grid
    physical_metrics = compute_physical_metrics({"frames": cleaned_frames}, fps=fps)
    team_heatmaps = {
        "team_0": generate_heatmap_grid({"frames": cleaned_frames, "pitch": {"length": pitch_len, "width": pitch_wid}}, team_id=0),
        "team_1": generate_heatmap_grid({"frames": cleaned_frames, "pitch": {"length": pitch_len, "width": pitch_wid}}, team_id=1),
    }

    # Compute final telemetry summary stats
    all_final_tids = set()
    for fd in cleaned_frames.values():
        for p in fd["players"]:
            if p["tracker_id"] > 0:
                all_final_tids.add(p["tracker_id"])

    elapsed = time.time() - t_start

    telemetry = {
        "video_info": info,
        "pitch": {
            "length": pitch_len,
            "width": pitch_wid,
        },
        "teams": {
            "team_0": {"name": "Team Red", "color": "#ef5350"},
            "team_1": {"name": "Team Blue", "color": "#42a5f5"},
            "goalkeeper": {"name": "Goalkeeper", "color": "#ffd54f"},
            "track_assignments": {str(k): v for k, v in classifier.track_team_map.items()}
        },
        "events": detected_events,
        "analytics": physical_metrics,
        "heatmaps": team_heatmaps,
        "processing": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "detection_conf": detection_conf,
            "imgsz": imgsz,
            "device": device,
            "unique_track_ids": len(all_final_tids),
            "track_id_list": sorted(list(all_final_tids)),
            "scene_cuts_detected": scene_cuts,
            "elapsed_seconds": round(elapsed, 2),
            "fps_processing": round(frames_to_process / elapsed, 1) if elapsed > 0 else 0,
        },
        "frames": cleaned_frames,
    }

    # Save upgraded telemetry JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(telemetry, f, indent=1)

    return {
        "frames_processed": frames_to_process,
        "unique_track_ids": len(all_final_tids),
        "track_ids": sorted(list(all_final_tids)),
        "scene_cuts": scene_cuts,
        "elapsed_seconds": round(elapsed, 2),
        "fps_processing": telemetry["processing"]["fps_processing"],
        "output_path": output_path,
    }
