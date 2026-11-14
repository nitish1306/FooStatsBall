"""
FastAPI web server for FooStatsBall Phase 1.
Serves the calibration tool, runs the processing pipeline,
and provides the synchronized video + radar dashboard.
"""

import json
import os
import shutil
import asyncio
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.calibration import PitchCalibrator, extract_frame, get_video_info
from src.pipeline import process_video
from src.clipper import export_highlight_reel
from src.analytics import generate_heatmap_grid

# Paths
PROJECT_DIR = Path(__file__).parent
ACTIVE_VIDEO_FILENAME = "Veo 7 a-side.mp4"
VIDEO_PATH = PROJECT_DIR / ACTIVE_VIDEO_FILENAME
OUTPUT_DIR = PROJECT_DIR / "output"


def _video_output_dir(video_path: Path) -> Path:
    """Each video gets its own calibration + telemetry so switching videos never mixes data."""
    return OUTPUT_DIR / video_path.stem


def _migrate_legacy_outputs():
    """Move pre-per-video output/{calibration,telemetry}.json into the default video's folder."""
    target = _video_output_dir(PROJECT_DIR / ACTIVE_VIDEO_FILENAME)
    for name in ("calibration.json", "telemetry.json"):
        legacy = OUTPUT_DIR / name
        if legacy.exists() and not (target / name).exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, target / name)


_migrate_legacy_outputs()


def calibration_path() -> Path:
    return _video_output_dir(VIDEO_PATH) / "calibration.json"


def telemetry_path() -> Path:
    return _video_output_dir(VIDEO_PATH) / "telemetry.json"


_telemetry_cache: dict = {"path": None, "mtime": None, "data": None}


def load_telemetry() -> dict:
    """Parse telemetry once per file version instead of on every request."""
    path = telemetry_path()
    if not path.exists():
        raise HTTPException(404, "No telemetry data")
    mtime = path.stat().st_mtime
    if _telemetry_cache["path"] != path or _telemetry_cache["mtime"] != mtime:
        _telemetry_cache.update(path=path, mtime=mtime, data=json.loads(path.read_text()))
    return _telemetry_cache["data"]


def _default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _list_videos() -> list[str]:
    return sorted(f.name for f in PROJECT_DIR.glob("*.mp4"))

app = FastAPI(title="FooStatsBall", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(PROJECT_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))

# ---------- State ----------
processing_status = {"running": False, "progress": 0, "total": 0, "message": ""}


# ---------- API Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main page — routes to calibration or dashboard based on state."""
    has_calibration = calibration_path().exists()
    has_telemetry = telemetry_path().exists()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "has_calibration": has_calibration,
            "has_telemetry": has_telemetry,
        }
    )


@app.get("/calibrate", response_class=HTMLResponse)
async def calibrate_page(request: Request):
    """Serve the calibration tool page."""
    return templates.TemplateResponse(request=request, name="calibrate.html", context={})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the main dashboard."""
    if not telemetry_path().exists():
        return HTMLResponse("<h2>No telemetry data yet. Please run processing first.</h2>", status_code=404)
    return templates.TemplateResponse(request=request, name="dashboard.html", context={})


@app.get("/api/videos")
async def api_get_videos():
    """List available match videos and current active video."""
    return {
        "videos": _list_videos(),
        "active": VIDEO_PATH.name
    }


@app.post("/api/videos/select")
async def api_select_video(request: Request):
    """Switch active video."""
    global VIDEO_PATH
    if processing_status["running"]:
        raise HTTPException(409, "Cannot switch video while processing")
    data = await request.json()
    video_name = data.get("video") if isinstance(data, dict) else None
    # Only allow names from the listing: blocks ../ traversal and absolute paths.
    if not isinstance(video_name, str) or video_name not in _list_videos():
        raise HTTPException(404, f"Video {video_name!r} not found")
    VIDEO_PATH = PROJECT_DIR / video_name
    return {
        "status": "ok",
        "active": VIDEO_PATH.name,
        "has_calibration": calibration_path().exists(),
        "has_telemetry": telemetry_path().exists(),
    }


@app.get("/api/video-info")
async def api_video_info():
    """Get video metadata."""
    if not VIDEO_PATH.exists():
        raise HTTPException(404, "Video file not found")
    info = get_video_info(str(VIDEO_PATH))
    info["filename"] = VIDEO_PATH.name
    return info


@app.get("/api/frame/{frame_number}")
async def api_get_frame(frame_number: int):
    """Get a specific video frame as JPEG."""
    if not VIDEO_PATH.exists():
        raise HTTPException(404, "Video file not found")
    try:
        frame = extract_frame(str(VIDEO_PATH), frame_number)
    except RuntimeError:
        raise HTTPException(404, f"Frame {frame_number} not available")
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@app.post("/api/calibration")
async def api_save_calibration(request: Request):
    """Save calibration points and compute homography."""
    data = await request.json()

    calibrator = PitchCalibrator()

    # Set pitch dimensions
    if "preset" in data:
        calibrator.set_pitch_preset(data["preset"])
    elif "pitch_length" in data and "pitch_width" in data:
        calibrator.set_pitch_dimensions(data["pitch_length"], data["pitch_width"])

    # Add calibration points
    points = data.get("points", [])
    if len(points) < 4:
        return JSONResponse({"error": "Need at least 4 calibration points"}, status_code=400)

    for pt in points:
        calibrator.add_calibration_point(
            pt["pixel_x"], pt["pixel_y"],
            pt["pitch_x"], pt["pitch_y"]
        )

    # Compute homography
    success = calibrator.compute_homography()
    if not success:
        return JSONResponse({"error": "Failed to compute homography"}, status_code=400)

    error = calibrator.get_reprojection_error()

    # Save
    calibration_path().parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(str(calibration_path()))

    return {
        "success": True,
        "reprojection_error_px": round(error, 2),
        "num_points": len(points),
    }


@app.get("/api/calibration")
async def api_get_calibration():
    """Get current calibration data."""
    if not calibration_path().exists():
        return JSONResponse({"exists": False})
    data = json.loads(calibration_path().read_text())
    data["exists"] = True
    return data


@app.post("/api/process")
async def api_start_processing(request: Request):
    """Start video processing pipeline."""
    global processing_status

    if processing_status["running"]:
        return JSONResponse({"error": "Processing already running"}, status_code=409)

    if not calibration_path().exists():
        return JSONResponse({"error": "No calibration found. Calibrate first."}, status_code=400)

    data = await request.json()
    start_frame = data.get("start_frame", 0)
    end_frame = data.get("end_frame", None)
    device = data.get("device") or "auto"
    if device == "auto":
        device = _default_device()

    processing_status = {"running": True, "progress": 0, "total": 0, "message": "Starting..."}

    # Run in background
    two_kits = bool(data.get("two_kits_only", False))
    asyncio.create_task(_run_processing(start_frame, end_frame, device, two_kits))

    return {"status": "started"}


async def _run_processing(start_frame: int, end_frame: Optional[int], device: str, two_kits_only: bool = False):
    """Background task for video processing."""
    global processing_status

    def progress_cb(current, total, message=None):
        processing_status["progress"] = current
        processing_status["total"] = total
        processing_status["message"] = message or f"Processing frame {current}/{total}"

    video, calib, telem = VIDEO_PATH, calibration_path(), telemetry_path()
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: process_video(
                video_path=str(video),
                calibration_path=str(calib),
                output_path=str(telem),
                start_frame=start_frame,
                end_frame=end_frame,
                device=device,
                imgsz=1920,
                progress_callback=progress_cb,
                two_kits_only=two_kits_only,
            )
        )
        processing_status["message"] = (
            f"Done! {result['frames_processed']} frames in {result['elapsed_seconds']}s "
            f"({result['fps_processing']} FPS) | Unique Tracks: {result['unique_track_ids']}"
        )
    except Exception as e:
        processing_status["message"] = f"Error: {str(e)}"
    finally:
        processing_status["running"] = False


@app.get("/api/process/status")
async def api_processing_status():
    """Get current processing status."""
    return processing_status


@app.get("/api/telemetry")
async def api_get_telemetry():
    """Get full telemetry data."""
    return load_telemetry()


@app.get("/api/analytics")
async def api_get_analytics():
    """Get aggregated team & physical player metrics."""
    telemetry = load_telemetry()
    return {
        "teams": telemetry.get("teams", {}),
        "analytics": telemetry.get("analytics", {}),
    }


@app.get("/api/heatmap/{target}")
async def api_get_heatmap(target: str):
    """
    Get 2D KDE heatmap grid for a team or player.
    target: 'team_0', 'team_1', or a track_id like '1', '2'
    """
    telemetry = load_telemetry()

    if target == "team_0":
        grid = telemetry.get("heatmaps", {}).get("team_0") or generate_heatmap_grid(telemetry, team_id=0)
    elif target == "team_1":
        grid = telemetry.get("heatmaps", {}).get("team_1") or generate_heatmap_grid(telemetry, team_id=1)
    else:
        try:
            tid = int(target)
            grid = generate_heatmap_grid(telemetry, track_id=tid)
        except ValueError:
            grid = []
    return {"target": target, "grid": grid}


@app.post("/api/export/highlights")
async def api_export_highlights(request: Request):
    """
    Export an automated highlight reel using FFmpeg.
    Body: {"preset": "all_passes" | "my_touches" | "all_events", "player_id": Optional[int]}
    """
    telemetry = load_telemetry()
    data = await request.json()
    preset = data.get("preset", "all_events")
    player_id = data.get("player_id")

    events = telemetry.get("events", [])
    if preset == "all_passes":
        selected_events = [e for e in events if e.get("type") in ["pass_completed", "turnover"]]
    elif preset == "my_touches" and player_id is not None:
        selected_events = [e for e in events if e.get("player_id") == player_id or e.get("passer_id") == player_id or e.get("receiver_id") == player_id]
    else:
        selected_events = events

    if not selected_events:
        return JSONResponse({"success": False, "error": "No matching events found for export"}, status_code=400)

    out_file = OUTPUT_DIR / f"highlight_reel_{VIDEO_PATH.stem}_{preset}.mp4"
    result = export_highlight_reel(
        video_path=str(VIDEO_PATH),
        event_list=selected_events,
        output_path=str(out_file),
        fps=telemetry.get("video_info", {}).get("fps", 30.0)
    )
    result["download_url"] = f"/download/highlight/{out_file.name}"
    return result


@app.get("/download/highlight/{filename}")
async def api_download_highlight(filename: str):
    """Download exported highlight reel."""
    file_path = (OUTPUT_DIR / filename).resolve()
    if file_path.parent != OUTPUT_DIR.resolve() or file_path.suffix != ".mp4" or not file_path.exists():
        raise HTTPException(404, "Exported file not found")
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)


@app.get("/api/telemetry/frame/{frame_number}")
async def api_get_frame_telemetry(frame_number: int):
    """Get telemetry for a specific frame."""
    telemetry = load_telemetry()
    frame_data = telemetry["frames"].get(str(frame_number))
    if frame_data is None:
        raise HTTPException(404, f"No data for frame {frame_number}")
    return frame_data


@app.get("/api/telemetry/range")
async def api_get_telemetry_range(start: int, end: int):
    """Get telemetry for a range of frames (for preloading)."""
    telemetry = load_telemetry()
    frames = {}
    for f in range(start, end + 1):
        data = telemetry["frames"].get(str(f))
        if data:
            frames[str(f)] = data
    return {
        "pitch": telemetry["pitch"],
        "frames": frames,
    }


@app.get("/video")
async def serve_video(request: Request):
    """Serve the video file with range request support for seeking."""
    if not VIDEO_PATH.exists():
        raise HTTPException(404, "Video not found")

    file_size = VIDEO_PATH.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header
        # Single range only; supports "a-b", "a-" and suffix "-n".
        try:
            unit, spec = range_header.split("=", 1)
            start_str, end_str = spec.split(",")[0].strip().split("-", 1)
            if unit.strip() != "bytes":
                raise ValueError
            if start_str == "":
                start = max(file_size - int(end_str), 0)
                end = file_size - 1
            else:
                start = int(start_str)
                end = min(int(end_str), file_size - 1) if end_str else file_size - 1
        except ValueError:
            raise HTTPException(416, "Invalid range", headers={"Content-Range": f"bytes */{file_size}"})
        if start >= file_size or start > end:
            raise HTTPException(416, "Range not satisfiable", headers={"Content-Range": f"bytes */{file_size}"})
        content_length = end - start + 1

        def iter_range():
            with open(VIDEO_PATH, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            }
        )

    return FileResponse(str(VIDEO_PATH), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

