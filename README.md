# FooStatsBall

Turn single-camera football footage into tracked players, events, physical stats and a shareable tactical dashboard. Processing runs locally. Sharing is static files on Cloudflare R2, with no server.

Technical details: [docs/TECHNICAL.md](docs/TECHNICAL.md)

## Features
- Pitch calibration: click 4+ landmarks to map video pixels to metres. Presets: 11v11, 7v7, 5v5, futsal.
- Player and ball tracking: YOLO11 + BoT-SORT, track stitching, re-detection of lost players. Estimated positions are drawn faded.
- Team split by kit colour, plus goalkeeper detection.
- Events: passes, turnovers, carries, possession.
- Stats: distance, top speed, sprints, heatmaps per player and per team.
- Dashboard: video with overlay, live 2D radar, event feed, analytics.
- "ME" mode: click your player, then export your own highlight reel.
- Share a match as a static page anyone can open.

## Run locally
Needs Python 3.10+ and FFmpeg on PATH.
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
```bash
python -m uvicorn app:app --port 8000
```
Open http://127.0.0.1:8000, then:
1. Pick a video (an `.mp4` in the repo root).
2. Calibrate the pitch.
3. Process the video (choose a range, e.g. the first 60 s).
4. Open the dashboard.

## Share a match (Cloudflare R2)
One-time setup: create a bucket, enable its public `r2.dev` URL, and create `.env` (it is git-ignored, never commit it):
```
R2_PUBLIC_URL=https://pub-xxxx.r2.dev
R2_BUCKET=football-full-game-videos
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...        # API token id
CLOUDFLARE_API_TOKEN=...    # API token value (R2 read/write)
```
Publish a processed video into its own folder:
```bash
venv/bin/python tools/publish_match.py "Fixedcam football game.mp4" --folder Sample
```
It prints the link to share:
```
https://pub-xxxx.r2.dev/Sample/index.html
```
Each match folder in the bucket is self-contained:
```
Sample/
  Fixedcam football game.mp4   raw video
  index.html                   dashboard (share this)
  assets/                      js, css, icon
  data/                        meta.json + frames/
  video/                       HLS stream
```
> The link must end in `/index.html`. r2.dev returns 404 for `/` and `/Sample/`.

Add `--no-upload` to build the folder locally in `export/<folder>/` without uploading anything.

## License
MIT
