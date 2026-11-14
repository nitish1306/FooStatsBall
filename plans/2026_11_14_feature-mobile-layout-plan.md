# Mobile layout + reel export fix

Goal: dashboard usable on phones (most viewers), clean on tablet/desktop; reel export plays correctly.

## Phases
- [x] 1. Reel export: rewrite fMP4 `tfdt` per track (continuous from 0), drop `sidx`; ffprobe start 0, decode OK
- [x] 2. Reel save on mobile: `navigator.share({files})` button when supported, plus Download link
- [x] 3. Phone layout (<768px): page scrolls; order video -> timeline -> radar -> events -> analytics;
      header toggles in one horizontal scroll row; timeline wraps (scrubber full width, big touch targets);
      radar 2:1 box; events table capped height, Dist column hidden; export modal as bottom sheet
- [x] 4. Tablet (768-1099px): video + radar side by side, bottom panes stacked
- [x] 5. Verified 375/768/1440 on local static build (no h-scroll, 12 events, reel 2.7 MB, bottom sheet); Sample republished

## Unresolved
- Sticky video while scrolling events on phone?
- Tabs (Events / Analytics) instead of stacked?
