# Static shareable dashboard (Cloudflare R2)

Goal: process locally, publish one match to R2, anyone opens a URL and gets the full dashboard
(video + overlay + radar + events + analytics + "ME" tagging + own highlight reel). No server.

## Decisions
- Everything on R2 (public `r2.dev`): app shell + per-match data. Same origin -> no CORS, no Pages admin needed.
  Pages/custom domain can front it later without code changes.
- Video: HLS, fMP4 segments, 720p ~900 kbps, 4 s keyframe-aligned segments. Whole source video
  encoded so telemetry frame numbers stay absolute.
- Telemetry split: `meta.json` (teams, events, analytics, per-team + per-player heatmaps, video_info)
  + `frames/cNNNN.json` 60 s chunks, rounded floats, unused fields dropped, gzip-encoded on upload.
- Highlight reel in browser: event windows (±2 s, merged) -> HLS segments -> concat init + m4s -> `.mp4` download.
  Cuts snap to 4 s segment edges.
- Upload via S3 API (boto3); creds in `.env` (token id = access key, sha256(token) = secret).

## Layout on R2
```
index.html                    match list (reads matches.json)
app/dashboard.html, app/*.js, app/*.css
matches.json                  [{slug, title, date, duration}]
matches/<slug>/meta.json
matches/<slug>/frames/c0000.json ...
matches/<slug>/video/index.m3u8, init.mp4, seg_00000.m4s ...
```
Viewer URL: `<R2_PUBLIC_URL>/app/dashboard.html?m=<slug>`

## Phases
- [x] 0. R2 creds in `.env`, public range fetch (206) + S3 list/put/delete verified
- [x] 1. `tools/publish_match.py`: HLS encode, telemetry split/compact, per-player heatmaps
- [x] 2. Dashboard static mode: config from `?m=`, chunked frame loader, hls.js, heatmaps from meta,
         overlay uses source dims (720p video vs 1080p bboxes)
- [x] 3. Client-side highlight reel (segment concat)
- [x] 4. Site shell upload (index.html, app/) + matches.json update
- [x] 5. Publish Fixedcam sample, verify in browser via r2.dev URL, commit

## Results
- Live: `https://pub-7f25dcbabb54468c9b5d1699690cb526.r2.dev/` -> Fixedcam sample
- Upload: 101 MB HLS (232 segs) + meta 1.1 MB + 2 frame chunks (3.5 MB raw, gzip on wire)
- Verified locally against same files: HLS play, overlay aligned (720p video, source-px canvas), radar, events, heatmap, ME reel (blob .mp4 decodes)
- Known: reel keeps source timestamps -> some players show start offset; fix later via mp4 tfdt rewrite if needed

## Unresolved
- Rotate token (was pasted in chat)?
- r2.dev rate limits ok for now, custom domain later?
