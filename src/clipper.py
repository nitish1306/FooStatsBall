"""
Phase 5: Automated Highlight Clipper & Match Export Engine.
Uses local FFmpeg to extract, slice, pad, and concatenate event clips into:
- All My Touches reel
- All Passes & Carries reel
- Custom selected events reel
"""

import subprocess
import os
from pathlib import Path


def export_highlight_reel(
    video_path: str,
    event_list: list[dict],
    output_path: str,
    pre_roll_sec: float = 2.0,
    post_roll_sec: float = 2.0,
    fps: float = 30.0,
) -> dict:
    """
    Slices and concatenates highlight events using FFmpeg.
    """
    temp_dir = Path("output/temp_clips")
    temp_dir.mkdir(parents=True, exist_ok=True)
    clip_files = []

    for i, ev in enumerate(event_list):
        ts = ev.get("timestamp", ev.get("start_frame", 0) / fps)
        duration = ev.get("duration_s", (ev.get("end_frame", 0) - ev.get("start_frame", 0)) / fps)

        start_time = max(0.0, ts - pre_roll_sec)
        clip_duration = duration + pre_roll_sec + post_roll_sec
        clip_path = temp_dir / f"clip_{i:03d}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.2f}",
            "-i", video_path,
            "-t", f"{clip_duration:.2f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac",
            str(clip_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if clip_path.exists():
            clip_files.append(clip_path)

    if not clip_files:
        return {"success": False, "error": "No clips were generated"}

    # Write concat manifest
    concat_txt = temp_dir / "concat_list.txt"
    with open(concat_txt, "w") as f:
        for c in clip_files:
            f.write(f"file '{c.resolve()}'\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Concat
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(output_path)
    ]
    subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Cleanup temp clips
    for c in clip_files:
        try:
            os.remove(c)
        except OSError:
            pass
    try:
        os.remove(concat_txt)
    except OSError:
        pass

    return {
        "success": Path(output_path).exists(),
        "output_path": output_path,
        "clips_count": len(clip_files),
        "size_mb": round(Path(output_path).stat().st_size / (1024 * 1024), 2) if Path(output_path).exists() else 0
    }

