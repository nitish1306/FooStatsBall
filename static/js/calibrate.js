/**
 * FooStatsBall - Calibration Tool
 * Interactive pitch landmark selection on video Frame 0.
 */

// Pitch presets (must match backend)
const PITCH_PRESETS = {
    '5v5': { length: 40, width: 25, players: 5 },
    '6v6': { length: 50, width: 30, players: 6 },
    '7v7': { length: 60, width: 40, players: 7 },
    '8v8': { length: 70, width: 45, players: 8 },
    '9v9': { length: 75, width: 50, players: 9 },
    '10v10': { length: 90, width: 60, players: 10 },
    '11v11': { length: 105, width: 68, players: 11 },
    'futsal': { length: 40, width: 20, players: 5 },
};

// Standard landmarks (relative to pitch dims)
function getLandmarks(L, W) {
    const pbW = 40.32; // penalty box width (full-size ratio, we'll scale)
    const pbD = 16.5;  // penalty box depth
    const penTop = (W - Math.min(pbW, W * 0.6)) / 2;
    const penBot = W - penTop;
    return {
        'Top-Left Corner':       [0, 0],
        'Top-Right Corner':      [L, 0],
        'Bottom-Left Corner':    [0, W],
        'Bottom-Right Corner':   [L, W],
        'Halfway Top':           [L/2, 0],
        'Halfway Bottom':        [L/2, W],
        'Center Spot':           [L/2, W/2],
        'Left Pen-Box Top':      [0, penTop],
        'Left Pen-Box Bottom':   [0, penBot],
        'Left Pen-Box Front Top':    [Math.min(pbD, L*0.18), penTop],
        'Left Pen-Box Front Bottom': [Math.min(pbD, L*0.18), penBot],
        'Right Pen-Box Top':     [L, penTop],
        'Right Pen-Box Bottom':  [L, penBot],
        'Right Pen-Box Front Top':    [L - Math.min(pbD, L*0.18), penTop],
        'Right Pen-Box Front Bottom': [L - Math.min(pbD, L*0.18), penBot],
    };
}

let currentPreset = '8v8';
let pitchDims = PITCH_PRESETS[currentPreset];
let landmarks = getLandmarks(pitchDims.length, pitchDims.width);
let calibrationPoints = [];
let pendingPixelClick = null;
let frameImage = null;

// DOM elements
const frameCanvas = document.getElementById('frame-canvas');
const frameCtx = frameCanvas.getContext('2d');
const pitchCanvas = document.getElementById('pitch-canvas');
const pitchCtx = pitchCanvas.getContext('2d');
const pointCountEl = document.getElementById('point-count');
const saveBtn = document.getElementById('save-btn');
const undoBtn = document.getElementById('undo-btn');
const videoSelect = document.getElementById('video-select');

// Populate videos
async function loadVideos() {
    try {
        const resp = await fetch('/api/videos');
        const data = await resp.json();
        videoSelect.innerHTML = '';
        data.videos.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v;
            opt.textContent = v;
            if (v === data.active) opt.selected = true;
            videoSelect.appendChild(opt);
        });
    } catch (e) {
        console.error("Failed to load videos:", e);
    }
}

videoSelect.addEventListener('change', async (e) => {
    const selected = e.target.value;
    await fetch('/api/videos/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video: selected })
    });
    clearAllPoints();
    loadFrame();
});

// Pitch preset change
document.getElementById('pitch-preset').addEventListener('change', (e) => {
    currentPreset = e.target.value;
    pitchDims = PITCH_PRESETS[currentPreset];
    landmarks = getLandmarks(pitchDims.length, pitchDims.width);
    drawPitch();
});

// Load Frame 0
async function loadFrame() {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        frameImage = img;
        frameCanvas.width = img.width;
        frameCanvas.height = img.height;
        drawFrame();
    };
    img.src = '/api/frame/0';
}

function drawFrame() {
    if (!frameImage) return;
    frameCtx.drawImage(frameImage, 0, 0);
    // Draw placed points
    for (let i = 0; i < calibrationPoints.length; i++) {
        const pt = calibrationPoints[i];
        frameCtx.beginPath();
        frameCtx.arc(pt.pixel_x, pt.pixel_y, 8, 0, Math.PI * 2);
        frameCtx.fillStyle = '#4fc3f7';
        frameCtx.fill();
        frameCtx.strokeStyle = '#fff';
        frameCtx.lineWidth = 2;
        frameCtx.stroke();
        frameCtx.fillStyle = '#fff';
        frameCtx.font = 'bold 12px sans-serif';
        frameCtx.fillText(String(i + 1), pt.pixel_x + 12, pt.pixel_y + 4);
    }
    // Draw pending click
    if (pendingPixelClick) {
        frameCtx.beginPath();
        frameCtx.arc(pendingPixelClick.x, pendingPixelClick.y, 10, 0, Math.PI * 2);
        frameCtx.strokeStyle = '#ffd54f';
        frameCtx.lineWidth = 3;
        frameCtx.stroke();
        frameCtx.setLineDash([4, 4]);
        frameCtx.beginPath();
        frameCtx.arc(pendingPixelClick.x, pendingPixelClick.y, 15, 0, Math.PI * 2);
        frameCtx.stroke();
        frameCtx.setLineDash([]);
    }
}

// Frame click handler
frameCanvas.addEventListener('click', (e) => {
    const rect = frameCanvas.getBoundingClientRect();
    const scaleX = frameCanvas.width / rect.width;
    const scaleY = frameCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    pendingPixelClick = { x, y };
    drawFrame();
    showLandmarkSelector();
});

// Pitch diagram drawing
function drawPitch() {
    const canvas = pitchCanvas;
    const ctx = pitchCtx;
    const L = pitchDims.length;
    const W = pitchDims.width;
    const padding = 30;
    const scaleX = (canvas.width - 2 * padding) / L;
    const scaleY = (canvas.height - 2 * padding) / W;
    const scale = Math.min(scaleX, scaleY);

    const offX = (canvas.width - L * scale) / 2;
    const offY = (canvas.height - W * scale) / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Pitch background
    ctx.fillStyle = '#2d5a27';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Pitch outline
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.strokeRect(offX, offY, L * scale, W * scale);

    // Halfway line
    ctx.beginPath();
    ctx.moveTo(offX + (L/2) * scale, offY);
    ctx.lineTo(offX + (L/2) * scale, offY + W * scale);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(offX + (L/2) * scale, offY + (W/2) * scale, Math.min(9.15, W * 0.2) * scale, 0, Math.PI * 2);
    ctx.stroke();

    // Penalty boxes
    const pbD = Math.min(16.5, L * 0.18);
    const pbW = Math.min(40.32, W * 0.6);
    const penTop = (W - pbW) / 2;
    // Left
    ctx.strokeRect(offX, offY + penTop * scale, pbD * scale, pbW * scale);
    // Right
    ctx.strokeRect(offX + (L - pbD) * scale, offY + penTop * scale, pbD * scale, pbW * scale);

    // Draw landmark dots
    for (const [name, [lx, ly]] of Object.entries(landmarks)) {
        const cx = offX + lx * scale;
        const cy = offY + ly * scale;

        // Check if already used
        const used = calibrationPoints.some(p => p.landmark === name);

        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = used ? '#66bb6a' : '#ffffff88';
        ctx.fill();
        ctx.strokeStyle = used ? '#66bb6a' : '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    // Store transform for click mapping
    pitchCanvas._transform = { offX, offY, scale, L, W };
}

// Pitch canvas click -> find nearest landmark
pitchCanvas.addEventListener('click', (e) => {
    if (!pendingPixelClick) return;

    const rect = pitchCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const t = pitchCanvas._transform;

    let bestDist = Infinity;
    let bestName = null;

    for (const [name, [lx, ly]] of Object.entries(landmarks)) {
        const sx = t.offX + lx * t.scale;
        const sy = t.offY + ly * t.scale;
        const dist = Math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2);
        if (dist < bestDist) {
            bestDist = dist;
            bestName = name;
        }
    }

    if (bestName && bestDist < 30) {
        addCalibrationPoint(bestName);
    }
});

// Landmark selector buttons (alternative to clicking pitch diagram)
function showLandmarkSelector() {
    const container = document.getElementById('landmark-selector');
    const buttons = document.getElementById('landmark-buttons');
    container.style.display = 'block';
    buttons.innerHTML = '';

    const usedNames = new Set(calibrationPoints.map(p => p.landmark));

    for (const name of Object.keys(landmarks)) {
        if (usedNames.has(name)) continue;
        const btn = document.createElement('button');
        btn.textContent = name;
        btn.onclick = () => addCalibrationPoint(name);
        buttons.appendChild(btn);
    }
}

function addCalibrationPoint(landmarkName) {
    if (!pendingPixelClick) return;

    const [pitchX, pitchY] = landmarks[landmarkName];
    calibrationPoints.push({
        landmark: landmarkName,
        pixel_x: Math.round(pendingPixelClick.x * 10) / 10,
        pixel_y: Math.round(pendingPixelClick.y * 10) / 10,
        pitch_x: pitchX,
        pitch_y: pitchY,
    });

    pendingPixelClick = null;
    document.getElementById('landmark-selector').style.display = 'none';
    updateUI();
}

function undoLastPoint() {
    calibrationPoints.pop();
    pendingPixelClick = null;
    document.getElementById('landmark-selector').style.display = 'none';
    updateUI();
}

function clearAllPoints() {
    calibrationPoints = [];
    pendingPixelClick = null;
    document.getElementById('landmark-selector').style.display = 'none';
    updateUI();
}

function updateUI() {
    const n = calibrationPoints.length;
    pointCountEl.textContent = `Points: ${n} / 4 minimum`;
    saveBtn.disabled = n < 4;
    undoBtn.disabled = n === 0;

    // Update table
    const tbody = document.getElementById('points-tbody');
    tbody.innerHTML = '';
    for (let i = 0; i < n; i++) {
        const pt = calibrationPoints[i];
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${pt.landmark}</td>
            <td>(${pt.pixel_x}, ${pt.pixel_y})</td>
            <td>(${pt.pitch_x}, ${pt.pitch_y})</td>
            <td><button class="btn btn-ghost btn-icon btn-sm" onclick="removePoint(${i})" aria-label="Remove point"><svg class="icon icon-sm" viewBox="0 0 24 24"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg></button></td>
        `;
        tbody.appendChild(row);
    }

    drawFrame();
    drawPitch();
}

function removePoint(index) {
    calibrationPoints.splice(index, 1);
    updateUI();
}

async function saveCalibration() {
    const resultMsg = document.getElementById('result-msg');
    resultMsg.className = 'result-msg';
    resultMsg.textContent = 'Computing homography...';
    resultMsg.classList.add('show');

    try {
        const resp = await fetch('/api/calibration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                preset: currentPreset,
                points: calibrationPoints,
            }),
        });
        const data = await resp.json();

        if (data.success) {
            resultMsg.innerHTML = `
                <strong>Homography saved.</strong>
                Reprojection error <span class="num">${data.reprojection_error_px} px</span>
                (${data.reprojection_error_px < 5 ? 'excellent' : data.reprojection_error_px < 15 ? 'acceptable' : 'consider adding points'}),
                ${data.num_points} points.
                <div style="margin-top:0.75rem"><a href="/" class="btn btn-primary btn-sm">Continue to processing</a></div>
            `;
            
        } else {
            resultMsg.innerHTML = `Error: ${data.error}`;
            resultMsg.classList.add('error');
        }
    } catch (e) {
        resultMsg.innerHTML = `Network error: ${e.message}`;
        resultMsg.classList.add('error');
    }
}

// Init
loadVideos();
loadFrame();
drawPitch();

