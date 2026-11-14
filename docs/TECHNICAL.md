# Technical notes

## Stack
- Python 3.12, FastAPI/Uvicorn, Ultralytics YOLO11, OpenCV, SciPy, scikit-learn, FFmpeg
- Frontend: plain JS + Canvas; hls.js 1.5.17 only on the static page
- Sharing: Cloudflare R2 through its S3 API (`boto3`)

## Pipeline (`src/pipeline.py`)
1. **Detect + track:** `yolo11m.pt` at `imgsz=1920`, BoT-SORT (`sports_botsort.yaml`: `track_high_thresh` and `new_track_thresh` both 0.3).
2. **Vicinity re-detect** (`src/vicinity.py`):
   - Applies to a track lost for up to 1.5 s.
   - Crops around its last box and runs a low-confidence detection (conf 0.08, imgsz 320).
   - A match keeps the same ID and is flagged `redetected`.
3. **Stitcher** (`src/track_stitcher.py`): merges fragments by motion and kit similarity, then drops micro-tracks (<15 frames). Stitching runs before the drop so fragments left after collisions survive.
4. **Smoothing:** Savitzky-Golay on trajectories; ball gaps are interpolated.
5. **Homography:** calibration points → RANSAC homography; pixel foot point → pitch metres.
6. **Teams:** torso crop with grass masked → colour clustering; goalkeepers picked by goal-line proximity and colour.
7. **Events** (`src/event_detector.py`): possession state machine → passes, turnovers, carries. A deflection/deceleration filter stops a ball that only brushed a player from counting as a reception.
8. **Analytics** (`src/analytics.py`): distance, top speed, sprints (≥20 km/h), KDE heatmaps (90×45 grid).

Output: `output/<video stem>/telemetry.json`. Frame keys are absolute source frame numbers.

## Rendering
- The overlay canvas is sized in source pixels; boxes come straight from telemetry.
- `interpolated` / `redetected` players are drawn at alpha 0.45 with dashed boxes.
- The radar interpolates positions between frames for smooth playback.
- The overlay locks to the frame the video is showing via `requestVideoFrameCallback`.

## Static share (`tools/publish_match.py`)
| Step | Detail |
|---|---|
| HLS encode | 720p h264 ~900 kbps, AAC 64k mono, 4 s keyframe-aligned fMP4 segments. The whole video is encoded so frame numbers stay absolute. |
| Telemetry split | `data/meta.json` holds teams, events, analytics, team + per-player heatmaps and `static.chunk_frames`. `data/frames/cNNNN.json` holds 60 s chunks: floats rounded to 2 dp, `confidence` and `kit_similarity` dropped, false flags omitted. |
| Page | `templates/dashboard.html` with paths rewritten to `./assets/`, plus `window.FOOSTATS_STATIC = true`. |
| Upload | Text files are gzipped (`Content-Encoding: gzip`, `no-cache`); media gets `immutable` caching. The raw video is uploaded only if it is missing. |

Static mode in `static/js/dashboard.js`:
- **Data loading:** `ensureChunk(frame)` loads frame chunks lazily into `frameCache`. Seeking and playback prefetch the next chunk.
- **Video:** native HLS on Safari, hls.js everywhere else.
- **Heatmaps:** read from `meta.json`. There is no API in static mode.
- **Highlight reel:**
  1. Pick events with the same presets as the server.
  2. Take each event window ±2 s.
  3. Find the HLS segments that cover those windows.
  4. Join `init.mp4` and those `.m4s` segments into a Blob `.mp4`. `rebaseFragment` rewrites each track's `tfdt` so the reel runs from 0 with no gaps, and drops `sidx`.
  5. Download link, plus a Share / Save button (`navigator.share` with a File) where supported, e.g. iOS.

  Cuts snap to 4 s segment edges.
- **Layout:** desktop ≥1100 px fixed-height grid; 768–1099 px page scrolls, video + radar side by side; <768 px single column (video → timeline → radar → events → analytics), toggles in a swipeable row, export as a bottom sheet.

## R2 auth
- S3 endpoint: `https://<account_id>.r2.cloudflarestorage.com`.
- Access key = API token id; secret = `sha256(token value)`.
- The Cloudflare REST objects API isn't used.
- r2.dev has no default index document, so links must name the file. A custom domain with a rewrite rule would fix that.

## Sizes (30 s Fixedcam sample, full 15 min video encoded)
HLS 101 MB · meta 1.1 MB · frames 3.5 MB raw (much smaller gzipped).
