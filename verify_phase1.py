import json
from src.pipeline import process_video

print("Starting verification on 45s (1350 frames) clip...")
result = process_video(
    video_path="Fixedcam football game.mp4",
    calibration_path="output/calibration.json",
    output_path="output/telemetry_verification.json",
    start_frame=0,
    end_frame=1349,
    detection_conf=0.25,
    imgsz=1280,
    device="cuda",
    check_scene_cuts=True,
    progress_callback=lambda c, t, msg: print(f"[{c}/{t}] {msg}") if c % 300 == 0 or c == t else None
)

print("\n================ VERIFICATION GATE RESULTS ================")
print(f"Processed Frames:       {result['frames_processed']}")
print(f"Unique Track IDs Count: {result['unique_track_ids']}")
print(f"Track IDs on Pitch:     {result['track_ids']}")
print(f"Scene Cuts Detected:    {result['scene_cuts']}")
print(f"Processing Speed:       {result['fps_processing']} FPS ({result['elapsed_seconds']}s)")
print("===========================================================\n")

