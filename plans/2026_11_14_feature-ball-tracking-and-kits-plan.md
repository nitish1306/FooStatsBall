# Ball tracking + team kits + events (Veo 30s clip)

## Findings (120 sampled Veo frames, yolo11n)
| Ball detector setup | Frames with ball |
|---|---|
| Current (in tracker, conf .25, 1280) | 14% |
| Separate predict, conf .25, 1280 | 29% |
| conf .05, 1280 | 80% |
| conf .05, 1920 | 100% (FP rate unknown) |
| 3x2 tiles, conf .05 | 95% |

Team classifier: kits are red vs yellow (+ black GK/ref). Rule `R/(G+B)>0.62` scores yellow as "red" → 9.4k vs 2.6k split.

## Phase 1 – Ball
- [x] Separate ball pass: `predict(classes=[32], conf=0.05, imgsz=1920)`, outside BoT-SORT (tracker drops low-conf)
- [x] Keep top-K candidates/frame; reject inside player torso boxes (heads), outside pitch mask
- [x] Kalman (constant-velocity) + gating: pick candidate nearest prediction; Viterbi/shortest-path over candidates for whole clip
- [x] Interpolate only gaps bounded by confident detections; mark `interpolated`
- [ ] Eval: hand-label ~60 frames → precision/recall script
- [ ] Optional: football-specific ball model (fine-tuned YOLO on public football ball dataset) — needs weights download

## Phase 2 – Team kits
- [x] Per-track median torso color in Lab (turf-masked) → KMeans k=2 on tracks, + outlier class (GK/ref) by distance
- [x] Auto-name teams by cluster color (red/yellow) → dashboard legend uses real kit colors

## Phase 3 – Events (needs P1+P2)
- [ ] Possession chain from per-frame ball owner (≤1.5 m, hysteresis)
- [x] Pass = owner change same team; interception = owner change other team w/o ball-out; shot = ball speed spike toward goal
- [ ] Event list + timeline in dashboard

## Results (Veo, after)
- Ball: 85.8% frames (484 detected + interp), p50 21 km/h; 78 residual >120 km/h steps (far-field homography noise). Viterbi + static-FP suppression + drop runs <6 frames.
- Kits: kit-agnostic over-clustering k=2..4, hue-angle naming → Yellow / Red, per-track similarity 0.52–0.98.
- Possession 54.5/45.5; events: 4 passes, 3 interceptions, 3 shots.
- Not done: hand-labelled eval, football-ball model, event timeline UI polish.

## Unresolved
- Download larger YOLO (11m/11x) or football-ball model weights? ok?
- Ball-pass at 1920 ~2x slower; acceptable?
- Want manual-label eval set for ball?
