# Player visibility: detector + tracking

Goal: every on-pitch player stays visible; no vanishing during close contact.

## Root causes (code)
- `yolo11n` nano @1280, conf 0.25 -> small/occluded players missed
- `new_track_thresh 0.65` -> players detected 0.25–0.65 never spawn a track (invisible)
- Stitcher drops tracks <15 frames *before* stitching -> post-collision fragments deleted
- No coasting: lost track = nothing drawn

## Phases
- [x] 0. Baseline metric script: visible players/frame, dropouts (track missing then back), raw det count
- [x] 1. Detector: yolo11m/x weights, imgsz 1920 option; compare recall on sampled frames
- [x] 2. Tracker cfg: new_track_thresh ~0.4, NMS iou higher, low-thresh 2nd stage
- [x] 3. Stitcher: stitch before micro-drop; appearance (kit) + motion gated
- [ ] 4. Vicinity re-detect: lost track -> crop around Kalman prediction, upscale, low-conf detect; colour fallback
- [ ] 5. Coast up to ~1.5 s, render faded "estimated"
- [ ] 6. Rerun both clips, compare metrics, browser check, commit

## Unresolved
- Fine-tune on football data later?
- Speed budget: 2–4x slower ok?

## Results (Fixedcam 1109-2008)
- Before: players/frame p10/p50/p90 6/12/15, 138 dip frames
- After (yolo11m@1920, new_track 0.3, stitch-before-drop): 21/22/24, 28 dip frames, 46 tracks, 2.7 FPS
