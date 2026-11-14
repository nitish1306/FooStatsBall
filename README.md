# FooStatsBall ⚽📊

**Local Football Match Telemetry & 2D Tactical Radar**

A 100% offline, local computer vision and tactical analytics system for association football matches. FooStatsBall transforms raw single-camera match footage (broadcast or fixed/elevated camera like Veo) into 2D pitch-plane positional telemetry, physical kinematics, tactical radar feeds, and automated event highlight reels.

---

## 🌟 Key Features

1. **Interactive Pitch Calibration & Planar Homography**
   - Interactive 4-point homography calibration directly on match video frames.
   - Presets for **11v11 standard** ($105\text{m} \times 68\text{m}$), **7v7** ($60\text{m} \times 40\text{m}$), **5v5** ($40\text{m} \times 20\text{m}$), and **Futsal**.
   - Sub-meter coordinate projection from camera pixel plane $(u, v)$ to metric pitch plane $(X, Y)$ via RANSAC with reprojection error estimation.

2. **Multi-Object Tracking with Sports-Tuned BoT-SORT**
   - YOLO11 object detection running locally on GPU/CPU.
   - Sports-tuned tracker parameters with high track buffer (60 frames) to maintain identity continuity through occlusions and referee/bench crossings.

3. **Physical Object Permanence & Coordinate Alignment Upgrades**
   - **Tracklet Stitcher**: Drops micro-tracks (< 15 frames) and merges fragmented tracklets using spatio-temporal kinematic gating ($d \le 9\,\text{m/s} \times \Delta t$).
   - **Savitzky-Golay Anti-Jitter Filter**: Removes high-frequency camera and detector jitter for butter-smooth trajectory lines.
   - **Ball Gap Interpolation**: Reconstructs missing ball positions across occluded and fast-moving frames.
   - **Letterbox Aspect-Ratio Correction**: Dynamically accounts for canvas and video container offsets to prevent bounding box drift.

4. **Turf-Invariant Team Kit Segregation**
   - Torso crop extraction with pitch-green pixel masking.
   - Normalized red chromatic ratio $\frac{R}{G + B}$ clustering to reliably differentiate teams (e.g. Orange/Red vs Bib/White) without turf grass contamination.
   - Automated goalkeeper isolation based on proximity to goal line and color divergence.

5. **Football Event Detection & Vector Deflection Guard**
   - **Pass Detection**: Tracks possession changes with a **directional deflection & deceleration vector filter** to ensure balls shot past a defender are not falsely counted as receptions.
   - **Possession & Turnover Tracking**: Continuous state machine monitoring team and player ball possession.
   - **Carries & Progressive Runs**: Detection of continuous ball movement under player control.

6. **Physical Kinematics & Tactical Heatmaps**
   - Physical load metrics: Total distance (m), top speed (km/h), and high-intensity sprint count ($\ge 20\,\text{km/h}$).
   - Team possession percentage breakdown.
   - 2D Gaussian Kernel Density Estimation (KDE) pitch heatmaps for individual players and teams.

7. **Automated Event Clipping & Highlight Export**
   - Local FFmpeg pipeline with pre-roll (3s) and post-roll (2s) event extraction.
   - Lossless concatenation into downloadable match highlight reels.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, Starlette
- **Computer Vision**: PyTorch, Ultralytics YOLO11, OpenCV, SciPy, Scikit-Learn
- **Media Processing**: FFmpeg
- **Frontend**: HTML5 Canvas, Vanilla ES6 JavaScript, Modern CSS3 Dark Mode UI

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- NVIDIA GPU with CUDA support (optional, but recommended for real-time inference)
- FFmpeg installed and available on system PATH

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/nitish1306/foostatsball.git
cd foostatsball

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 📋 Workflow Guide

1. **Select Match Video**: Place your match footage (`.mp4`) in the root or select from the dropdown.
2. **Calibrate Pitch**: Open the Calibration page, select pitch dimensions (e.g. 7v7 or 11v11), and click 4 ground-truth landmarks (corner flags, penalty marks, center spot).
3. **Run Pipeline**: Click **Run Full Pipeline** to extract detections, track identities, classify jerseys, detect events, and compute physical metrics.
4. **Tactical Dashboard**:
   - Inspect the synchronized split-screen video player and 2D tactical radar.
   - Toggle speed vectors, trajectories, team possession, and event overlays.
   - View player leaderboards (distance, top speed, sprints) and generate 2D heatmaps.
   - Export automated highlight reels to MP4.

---

## 📄 License
MIT License. Free for personal and commercial use.
