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
import os
import time
import numpy as np
import cv2
from pathlib import Path

from src.calibration import PitchCalibrator, get_video_info
from src.detection import DetectionTracker
from src.ball_interpolation import interpolate_ball_trajectory
from src.track_stitcher import stitch_and_clean_trajectories
from src.scene_guard import detect_scene_cuts
from src.camera_motion import CameraMotionEstimator
from src.ball_detector import BallDetector, select_ball_path


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
    compensate_camera_motion: bool = True,
    two_kits_only: bool = False,
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
    ball_detector = BallDetector(tracker.model, device=device, imgsz=max(imgsz, 1920))
    ball_cands, ball_cands_ref = {}, {}

    cap = cv2.VideoCapture(video_path)
    # Camera motion: homography was calibrated on the reference frame; pre-roll flow up to start_frame.
    cam = CameraMotionEstimator() if compensate_camera_motion else None
    ref_frame = int(getattr(calibrator, "reference_frame", 0) or 0)
    if cam is not None and ref_frame < start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
        for _ in range(ref_frame, start_frame):
            ok, f = cap.read()
            if not ok:
                break
            cam.update(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def to_ref(pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        return cam.to_reference(pts) if cam is not None else pts

    raw_frames = {}
    ball_raw_positions = {}
    ball_ref_positions = {}
    t_start = time.time()

    for current_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        players, ball_center = tracker.detect_and_track(frame, conf=detection_conf)
        if cam is not None:
            cam.update(frame, [p["bbox"] for p in players])

        # 3. 2D pitch polygon boundary filtering
        players_data = []
        if len(players) > 0:
            foot_pixels = np.array([p["foot_pixel"] for p in players], dtype=np.float32)
            pitch_pts = calibrator.pixel_to_pitch_batch(to_ref(foot_pixels))

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

        # Ball: collect low-conf candidates; the global path is chosen after the loop.
        cands = ball_detector.detect(frame, [p["bbox"] for p in players])
        ball_cands[current_idx] = cands
        ball_cands_ref[current_idx] = [tuple(map(float, r)) for r in to_ref([(c[0], c[1]) for c in cands])] if cands else []
        ball_data = None

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

    # Global ball path selection (Viterbi over candidates in camera-compensated pixels)
    chosen = select_ball_path(ball_cands, ball_cands_ref, frame_width=info["width"])
    for f_idx, ci in chosen.items():
        ball_raw_positions[f_idx] = None
        if ci is None:
            continue
        cx, cy, cconf = ball_cands[f_idx][ci]
        rbx, rby = ball_cands_ref[f_idx][ci]
        bx, by = (float(v) for v in calibrator.pixel_to_pitch(rbx, rby))
        if (-touchline_buffer <= bx <= pitch_len + touchline_buffer and
                -touchline_buffer <= by <= pitch_wid + touchline_buffer):
            raw_frames[str(f_idx)]["ball"] = {"pixel": [cx, cy], "pitch_xy": [bx, by], "confidence": round(cconf, 3)}
            ball_raw_positions[f_idx] = (cx, cy)
            ball_ref_positions[f_idx] = (rbx, rby)

    # Interpolate ball gaps
    interpolated_ball = interpolate_ball_trajectory(ball_raw_positions, max_gap=45)
    # Pitch projection interpolates in reference-frame pixels so camera pans don't bend the path.
    interpolated_ref = interpolate_ball_trajectory(
        {k: ball_ref_positions.get(k) for k in ball_raw_positions}, max_gap=45)
    for frame_str, frame_data in raw_frames.items():
        f_idx = int(frame_str)
        if frame_data["ball"] is None and interpolated_ball.get(f_idx) is not None:
            bx, by = interpolated_ball[f_idx]
            if interpolated_ref.get(f_idx) is None:
                continue
            rx, ry = interpolated_ref[f_idx]
            bpx, bpy = (float(v) for v in calibrator.pixel_to_pitch(rx, ry))
            # Interpolated points get the same pitch-bounds check as detected ones.
            if (-touchline_buffer <= bpx <= pitch_len + touchline_buffer and
                -touchline_buffer <= bpy <= pitch_wid + touchline_buffer):
                frame_data["ball"] = {
                    "pixel": [float(bx), float(by)],
                    "pitch_xy": [bpx, bpy],
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
    classifier = TeamClassifier(two_kits_only=two_kits_only)
    temp_telemetry = {"frames": cleaned_frames}
    classifier.collect_features_from_video(video_path, temp_telemetry)
    classifier.fit_clusters(pitch_length=pitch_len, pitch_width=pitch_wid, telemetry=temp_telemetry)

    # Attach team_id and color to every player detection across all frames
    for fd in cleaned_frames.values():
        for p in fd["players"]:
            tid = p["tracker_id"]
            team_id = classifier.track_team_map.get(tid, -1)
            p["team_id"] = team_id
            p["team_color"] = classifier.team_colors.get(team_id, "#bdbdbd")
            p["kit_similarity"] = classifier.track_similarity.get(tid)
            gk = getattr(classifier, "goalkeepers", {}).get(tid)
            if gk is not None:
                p["role"] = "GK"
            # team credited with possession when this player has the ball (keepers count for their side)
            p["possession_team"] = gk if gk is not None else team_id

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
        "video_info": {**info, "filename": Path(video_path).name},
        "pitch": {
            "length": pitch_len,
            "width": pitch_wid,
        },
        "teams": {
            "team_0": {"name": classifier.team_names[0], "color": classifier.team_colors[0]},
            "team_1": {"name": classifier.team_names[1], "color": classifier.team_colors[1]},
            "goalkeeper": {"name": classifier.team_names[-1], "color": classifier.team_colors[-1]},
            "goalkeepers": {str(k): v for k, v in getattr(classifier, "goalkeepers", {}).items()},
            "kit_similarity": {str(k): v for k, v in classifier.track_similarity.items()},
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
    # Atomic write: readers never see a half-written file.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(telemetry, f, separators=(",", ":"))
    os.replace(tmp_path, output_path)

    return {
        "frames_processed": frames_to_process,
        "unique_track_ids": len(all_final_tids),
        "track_ids": sorted(list(all_final_tids)),
        "scene_cuts": scene_cuts,
        "elapsed_seconds": round(elapsed, 2),
        "fps_processing": telemetry["processing"]["fps_processing"],
        "output_path": output_path,
    }
