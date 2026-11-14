# Sprinter tracking dropouts

Issue: near-camera sprinters (#27) vanish while sprinting, reappear as new ID.
Cause: #27 ended f1420, continued as #124 f1441; homography near camera → 7.9 m in 0.7 s (11 m/s) > 9 m/s stitch gate.

- [x] Diagnose via telemetry + frames
- [x] Stitcher: extra pixel gate, `max_body_heights_per_sec=6` (box-height scaled, perspective-invariant)
- [x] Re-run sample 1109–2008: #27 now 1130→1998 continuous; track IDs 58 → 50
- [ ] Republish + PR (awaiting OK)

## Unresolved
- False merges in crowds? gate generous; watch for swaps
