# Roster-constrained tracking (NvN format)

Goal: calibration picks match format (5v5…11v11, default 8v8) → pitch size + player count N.
Pipeline uses 2N as a hard prior: fixed identity slots per team, stable IDs, fewer phantoms/fragments.

## Thoughts
- Good uplift. Today IDs are open-ended (sample: 50 IDs for ~14–16 people). Known roster turns
  tracking from "open set" into "assignment into 2N slots" — much easier, more stable.
- Don't force it at detection level (hard top-2N per frame): off-camera players, occlusion, refs,
  subs, bench, coaches would make it wrong. Apply as a **global post-process over tracklets**.
- Gains: stable IDs across whole video (better ME mode, per-player stats), phantom/spectator pruning,
  per-team cap catches team-colour mistakes.
- Risks: subs (local games: rolling subs), refs/coaches on pitch, GK kit differs, two-camera-view
  coverage gaps (player legitimately absent). Need "unassigned/extra" bucket, not forced merges.

## Decisions (user)
- No refs. Subs exist → soft cap: on-camera people may run N±2 per team briefly. No hard limit.
- Real target: **phantom spawns in clusters** (new IDs appearing mid-pitch when players bunch up).

## Spawn rules (core of this feature)
New tracklet starting at frame f, team T, with `active(T) >= N`:
1. **Spawned mid-pitch** (not near a touchline/edge of view) → not a sub; must be an existing player.
2. Try **re-attach** to a lost slot of T (lost ≤ 3 s) within motion gate → same ID.
3. Else if box overlaps an active player (IoU > 0.3 / centre inside box) → **duplicate**, suppress.
4. Else if short (< 1 s) → drop as phantom.
5. Else keep as `extra` (sub coming on), still allowed (soft cap).
Spawns at edges or with `active(T) < N` → normal new slot.

## Design
1. **Format presets** (frontend + `src/calibration.py` share one table). Editable L×W override.
   | Format | L×W m (default) |
   |---|---|
   | 5v5 | 40×25 |
   | 6v6 | 50×30 |
   | 7v7 | 60×40 |
   | 8v8 | 70×45 |
   | 9v9 | 75×50 |
   | 10v10 | 90×60 |
   | 11v11 | 105×68 |
   Saved in calibration.json as `format`, `players_per_team`.
2. **Per-frame filter**: on-pitch detections only (exists); rank by conf; soft cap 2N + ~3 slack (refs).
   Excess → candidates for drop if short-lived.
3. **Tracklet → slot assignment** (new `src/roster.py`, after stitcher, after team classify):
   - Nodes = tracklets with team, start/end pos/box, appearance embedding (mean ReID/kit hist).
   - Per team: assign tracklets to ≤N slots; constraint: tracklets in one slot never overlap in time.
   - Cost = motion gate (existing pitch + body-height gates) + appearance distance + gap length.
   - Solve greedily by time (interval scheduling w/ Hungarian at each new tracklet start) — slots
     "free" when their tracklet ends; new tracklet takes best free slot within gate, else `extra`.
   - Short tracklets that fit nowhere and are outside 2N → dropped (phantoms).
   - GK = slot 1 of each team (existing GK detection).
4. **Output**: `tracker_id` becomes slot id (`A1..A8`, `B1..B8` or ints 1..2N); raw id kept as `raw_id`.
   `extra` tracks (refs, subs) kept but greyed, excluded from team stats.
5. **UI**: calibration select (7 formats, default 8v8) + L×W fields; dashboard shows "16/16 tracked".
6. **Metrics to prove it**: unique IDs, ID switches on sample (hand-label ~5 players across 30 s),
   % frames with ≤2N players, #27-style continuity.

## Phases
- [ ] 0. Hand-label ground truth for sample (5 players × 900 frames) + metric script
- [x] 1. Format presets table + calibration UI + saved `players_per_team`
- [x] 2. `src/roster.py` slot assignment + tests (synthetic tracklets: overlap, gaps, extras)
- [x] 3. Wire into pipeline (`raw_id`/`extra` flag deferred — drops + re-attach only)
- [ ] 4. Dashboard: count badge, extras greyed
- [ ] 5. Re-run sample, compare metrics before/after, republish

## Results (sample, 11v11, frames 1109–2008)
- `src/roster.py` + `tests/test_roster.py` (6 pass). Runs after team classification.
- Unique IDs 43 (was 50); frames with >24 people 29% (was 40%).
- Still median 24 on screen vs 22: long-lived people near far touchline / bench present from frame 1
  (not spawns, so roster rule skips them).
- Added `drop_spectators` (off-pitch / wide box / static at touchline). Dry-run on sample: median 21 on screen, >24 people 0%, keepers 6 + 22 kept. Jittery far-side 14/104 still in.
- Phase 0 (hand-label GT) + 4 (dashboard badge) not done.

## Resolved
- Sizes: table above (7v7 → 60×40; sample re-calibrated with its format). Editable L×W kept.
- Subs enter from any touchline → rule 1 "edge" = within ~3 m of any pitch line.
- Whole pitch always in frame → missing player = occluded/merged, never off-camera.
- Slot labels: ints 1..2N (team from `team_id`); extras get N*2+1…

## Unresolved
- None
