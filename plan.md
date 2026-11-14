# Local Football Match Telemetry & Tactical Radar — Project Plan

---

## 1. System Overview & Architecture Principles

- **100% Offline Execution:** Runs entirely on local hardware (Windows laptop with NVIDIA RTX GPU) using open-source Python libraries and locally cached model weights. Requires zero cloud APIs, external servers, or recurring subscriptions.

- **Decoupled Data Pipeline:** Separates heavy GPU video inference from the interactive web dashboard using a structured local match telemetry file. This allows instant UI iteration, manual track corrections, and highlight exports without re-running computer vision models.

- **VRAM-Safe Two-Pass Inference:** Splits GPU processing into two sequential stages — Pass 1 for player/ball detection and tracking, followed by Pass 2 for team jersey clustering — using FP16/TensorRT optimization to stay safely within 6GB–8GB laptop VRAM limits.

---

## 2. Core Repositories, Models & Dependencies

| Component | Source | Purpose |
|---|---|---|
| **Sports CV Toolkit** | `roboflow/sports` | Planar homography transformation, standard pitch geometry, 2D tactical radar rendering, team clustering workflows |
| **Detection & Tracking Wrappers** | `roboflow/supervision` | Multi-object tracking (BoT-SORT, ByteTrack), bounding-box/trace annotators, video frame generators |
| **Object Detection Engine** | `ultralytics/ultralytics` | YOLO11 / YOLOv8 local inference with TensorRT FP16 export support |
| **Pre-Trained Weights & Encoders** | Roboflow Universe, Hugging Face | Football player/goalkeeper/referee detection checkpoints, small-ball detection models, SigLIP vision encoder for jersey embeddings |
| **Media Processing** | FFmpeg | Lossless clip slicing, highlight concatenation, PiP radar video exports |
| **Web App Stack** | Python + FastAPI, HTML5 Canvas/SVG + JS | Local backend server, telemetry API, synchronized video playback, interactive tactical graphics |

---

## 3. Target Dashboard & UI Architecture

The finished local web dashboard uses a **synchronized multi-pane command center** layout:

| Pane | Content |
|---|---|
| **Top-Left** — Match Video Player | Plays match video with toggleable overlays: ground contact ellipses, persistent track IDs, team-colored rings, ball trajectory trails |
| **Top-Right** — Live 2D Tactical Radar | Top-down vector pitch in millisecond sync with video. Scrubbing either pane updates both. Layer toggles for player dots, Voronoi space control, passing lanes, heatmaps |
| **Middle** — Master Timeline Bar | Full match duration with color-coded event pins: 🟢 Completed Pass, 🔴 Turnover, 🔵 Dribble/Progressive Carry, 🟡 Shot/Goal, ⚪ Your Touches. Click any pin to seek 3s before the event. Playlist modes: "Play All My Touches", "Play All Passes" |
| **Bottom-Left** — Event Feed & Quick-Fix | Scrollable chronological event table (timestamp, passer, receiver, type, distance, outcome) with one-click correction tools (Swap Team, Merge Track ID, Delete False Event) |
| **Bottom-Right** — Analytics Panel | Team Possession %, Pass Completion %, Distance Covered (km), Top Sprint Speed (km/h), Sprint Count, Progressive Carries, interactive Pass Network diagram |

---

## 4. Incremental Development Phases

### Phase 1: Pitch Calibration & Synchronized 2D Tactical Map

**Objective:** Convert angled fixed-camera turf footage into an accurate top-down 2D tactical map synchronized with video playback.

#### Backend & CV Features
- Interactive Frame 0 calibration tool to select 4–8 pitch landmarks and compute a static planar homography matrix.
- YOLO11 player and ball detection paired with BoT-SORT multi-object tracking.
- Ball trajectory interpolation (linear + Kalman smoothing) to bridge frames where the ball vanishes due to motion blur.
- Frame-by-frame projection of player foot coordinates and ball position onto a scaled 2D pitch plane (in meters).

#### UI Delivered
- Split-screen local web interface: **Match Video Player** (left) + **Live 2D Tactical Radar** (right).
- Synchronized playback scrubber bar with play/pause, frame-stepping, and playback speed controls (0.5×, 1×, 2×).
- Toggle switches for bounding boxes, track IDs, and ball trails.

#### Verification Criteria
Scrub through a 1–2 minute sample clip and confirm player dots on the 2D radar stay accurately inside pitch touchlines and penalty boxes without drifting.

---

### Phase 2: Team Separation & Personal Player Identity

**Objective:** Automatically separate the two teams and lock onto your specific player identity across the entire match.

#### Backend & CV Features
- Two-pass VRAM-safe team clustering: upper-torso cropping → SigLIP visual embeddings → UMAP dimensionality reduction → K-Means clustering (k=2).
- Lightweight HSV grass-masked color clustering fallback for casual games with solid dark vs. light shirts (no bibs).
- Goalkeeper isolation via goal-box proximity and majority-vote team assignment per persistent track ID.

#### UI Delivered
- Team-colored dots and rings across both the video player and 2D tactical radar.
- Interactive "Click-to-Tag Me" selector on Frame 0 — highlights your player with a distinct gold halo.
- Post-processing Track ID Merger modal to stitch fragmented track IDs from crowded penalty-box occlusions.

#### Verification Criteria
Confirm zero team-color flipping when opposing players cross paths and verify your gold halo tracks you continuously across the test clip.

---

### Phase 3: Event Detection (Passes, Dribbles, Shots) & Master Timeline Bar

**Objective:** Translate raw spatial trajectories into structured football events and power the interactive timeline.

#### Backend & CV Features
- Possession state machine driven by foot-to-ball distance thresholds and consecutive frame stability rules.
- Pass detection engine (possession loss + ball velocity spike + ball arrival at teammate) distinguishing completed passes from interceptions/turnovers.
- Dribble and progressive carry detector (sustained ball possession while advancing toward opponent's goal).
- Shot detection based on high-velocity ball trajectories entering the goal mouth.

#### UI Delivered
- Full Master Interactive Timeline Bar with clickable color-coded event pins and "My Touches Only" filter.
- Animated pass vectors and dribble paths drawn on the 2D tactical radar during active plays.
- Bottom-Left Event Feed with filtering by team, player, or event type, plus one-click correction buttons.

#### Verification Criteria
Compare automated pass and dribble counts against a manual count on the sample video (targeting >85% accuracy). Verify clicking timeline markers jumps cleanly to the play build-up.

---

### Phase 4: Physical Metrics, Heatmaps & Tactical Visuals

**Objective:** Aggregate spatial and event telemetry into physical workload metrics and tactical pitch visualizations.

#### Backend & CV Features
- Kinematics engine: total distance covered (km), smoothed instantaneous speed (km/h), top sprint speed, high-intensity sprint counts, team possession percentages.
- 2D Gaussian kernel density estimation (KDE) for individual and team heatmaps.
- Voronoi tessellation engine for real-time team space control shading and pass network adjacency aggregation.

#### UI Delivered
- Complete Bottom-Right Analytics Panel with player selector dropdown and personal match scorecard.
- Toggleable Heatmap and Voronoi Space Control layers on the 2D tactical radar.
- Interactive Team Pass Network diagram (node size = touches, line thickness = pass volume).

#### Verification Criteria
Confirm distance and speed metrics fall within realistic physical bounds (e.g., top sprint 22–30 km/h, no tracking-jump outliers).

---

### Phase 5: Automated Highlight Clipper & Match Export

**Objective:** Export shareable video highlight reels and tactical overlays directly from the dashboard.

#### Backend & CV Features
- Local FFmpeg clip slicing and concatenation with configurable pre/post-roll padding (default t−3s to t+3s).
- Preset reel generators: "All My Touches", "My Passes & Dribbles", "All Shots & Goals", "Condensed Match (Dead Ball Removed)".
- Optional picture-in-picture 2D tactical radar burn-in for exported clips.

#### UI Delivered
- "Export Highlights" drawer in the Master Timeline Bar with multi-select event checkboxes, clip padding sliders, and export progress feedback.

#### Verification Criteria
Export a 60-second personal highlight reel and confirm smooth clip transitions, intact audio sync, and accurate play framing.

---

## 5. Recording & Capture Guidelines for Best Accuracy

- **Locked Camera Angle:** Mount the phone on a tripod or high on the turf cage behind one goal or near the halfway line so all four pitch corners remain visible in frame.
- **Frame Rate & Resolution:** Record at 1080p or 4K at 60fps to cut ball motion blur in half during fast passes.
- **Kit Contrast:** Have at least one team wear high-visibility training bibs (neon orange, yellow, or red) and avoid green shirts that blend into artificial turf.
