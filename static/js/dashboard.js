/**
 * FooStatsBall — Tactical Dashboard
 * Synchronized video player + 2D tactical radar.
 */

// ========== State ==========
let telemetryData = null;
let videoInfo = null;
let isPlaying = false;
let currentFrame = 0;
let animFrameId = null;

// Telemetry frame cache for fast lookup
let frameCache = {};

// Ball trail buffer
const BALL_TRAIL_LENGTH = 60;
let ballTrail = [];

// Personal player identity tag
let myPlayerTrackId = null;

// DOM
const video = document.getElementById('match-video');
const overlayCanvas = document.getElementById('overlay-canvas');
const overlayCtx = overlayCanvas.getContext('2d');
const radarCanvas = document.getElementById('radar-canvas');
const radarCtx = radarCanvas.getContext('2d');
const scrubber = document.getElementById('scrubber');
const frameCounter = document.getElementById('frame-counter');
const currentTimeEl = document.getElementById('current-time');
const totalTimeEl = document.getElementById('total-time');
const radarTimeEl = document.getElementById('radar-time');
const playBtn = document.getElementById('play-btn');

// Toggles
const toggleBbox = document.getElementById('toggle-bbox');
const toggleIds = document.getElementById('toggle-ids');
const toggleBallTrail = document.getElementById('toggle-ball-trail');
const toggleVoronoi = document.getElementById('toggle-voronoi');
const toggleHeatmap = document.getElementById('toggle-heatmap');

// ========== Init ==========
async function init() {
    // Load video info
    const infoResp = await fetch('/api/video-info');
    videoInfo = await infoResp.json();

    // Load telemetry
    const telResp = await fetch('/api/telemetry');
    telemetryData = await telResp.json();

    // Build frame cache
    frameCache = telemetryData.frames;

    // Set scrubber range
    const startFrame = telemetryData.processing.start_frame;
    const endFrame = telemetryData.processing.end_frame;
    scrubber.min = startFrame;
    scrubber.max = endFrame;
    scrubber.value = startFrame;
    currentFrame = startFrame;

    // Set total time
    totalTimeEl.textContent = formatTime(videoInfo.duration_seconds);

    // Setup video
    function onReady() {
        setupOverlay();
        setupRadar();
        seekToFrame(startFrame);
    }

    if (video.readyState >= 1) {
        onReady();
    } else {
        video.addEventListener('loadedmetadata', onReady, { once: true });
    }

    // Also handle direct video play/pause events from native or UI controls
    video.addEventListener('play', () => {
        if (!isPlaying) {
            isPlaying = true;
            playBtn.textContent = '⏸';
            animLoop();
        }
    });

    video.addEventListener('pause', () => {
        if (isPlaying) {
            isPlaying = false;
            playBtn.textContent = '▶';
            if (animFrameId) cancelAnimationFrame(animFrameId);
        }
    });

    // Scrubber input
    scrubber.addEventListener('input', (e) => {
        const frame = parseInt(e.target.value);
        seekToFrame(frame);
    });

    // Video timeupdate for sync
    video.addEventListener('timeupdate', () => {
        if (isPlaying) {
            const frame = Math.floor(video.currentTime * videoInfo.fps);
            currentFrame = frame;
            scrubber.value = frame;
            updateDisplay();
        }
    });

    // Toggle redraws
    toggleBbox.addEventListener('change', () => updateDisplay());
    toggleIds.addEventListener('change', () => updateDisplay());
    toggleBallTrail.addEventListener('change', () => updateDisplay());
    if (toggleVoronoi) toggleVoronoi.addEventListener('change', () => updateDisplay());
    if (toggleHeatmap) toggleHeatmap.addEventListener('change', () => updateDisplay());

    // Phase 3 & 4: Setup Timeline Pins, Event Feed & Analytics Scorecard
    setupTimelinePins();
    renderEventFeed('all');
    setupAnalyticsPanel();
}

    function setupOverlay() {
    // Determine the exact rendered display box of the video under object-fit: contain
    const rect = video.getBoundingClientRect();
    const vw = video.videoWidth || (videoInfo ? videoInfo.width : 1280);
    const vh = video.videoHeight || (videoInfo ? videoInfo.height : 720);

    const videoAspect = vw / vh;
    const containerAspect = rect.width / rect.height;

    let renderW, renderH, offsetX, offsetY;

    if (containerAspect > videoAspect) {
        // Pillarboxed (black bars on left/right)
        renderH = rect.height;
        renderW = renderH * videoAspect;
        offsetX = (rect.width - renderW) / 2;
        offsetY = 0;
    } else {
        // Letterboxed (black bars on top/bottom)
        renderW = rect.width;
        renderH = renderW / videoAspect;
        offsetX = 0;
        offsetY = (rect.height - renderH) / 2;
    }

    // Set canvas dimensions and position directly over the active video content
    overlayCanvas.width = vw;
    overlayCanvas.height = vh;
    overlayCanvas.style.width = `${renderW}px`;
    overlayCanvas.style.height = `${renderH}px`;
    overlayCanvas.style.left = `${offsetX}px`;
    overlayCanvas.style.top = `${offsetY}px`;
    overlayCanvas.style.position = 'absolute';
}

// Click on video to "Tag Me"
overlayCanvas.style.pointerEvents = 'auto';
overlayCanvas.addEventListener('click', (e) => {
        const rect = overlayCanvas.getBoundingClientRect();
        const scaleX = overlayCanvas.width / rect.width;
        const scaleY = overlayCanvas.height / rect.height;
        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        const frameData = frameCache[String(currentFrame)];
        if (!frameData) return;

        let bestTid = null;
        let minD = Infinity;
        for (const p of frameData.players) {
            const [x1, y1, x2, y2] = p.bbox;
            if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
                bestTid = p.tracker_id;
                break;
            }
            const cx = (x1 + x2) / 2;
            const cy = (y1 + y2) / 2;
            const dist = Math.hypot(clickX - cx, clickY - cy);
            if (dist < minD && dist < 50) {
                minD = dist;
                bestTid = p.tracker_id;
            }
        }

        if (bestTid !== null) {
            myPlayerTrackId = (myPlayerTrackId === bestTid) ? null : bestTid;
            updateDisplay();
        }
    });

function setupRadar() {
    // Set radar canvas size (maintain pitch aspect ratio)
    const pitch = telemetryData.pitch || { length: 90, width: 45 };
    const parent = radarCanvas.parentElement;
    const maxW = Math.max(parent.clientWidth - 20, 300);
    const maxH = Math.max(parent.clientHeight - 20, 200);
    const aspect = pitch.length / pitch.width;

    let w, h;
    if (maxW / maxH > aspect) {
        h = maxH;
        w = h * aspect;
    } else {
        w = maxW;
        h = w / aspect;
    }

    radarCanvas.width = Math.floor(w);
    radarCanvas.height = Math.floor(h);
    radarCanvas._pitchScale = { w, h, L: pitch.length, W: pitch.width };
    drawRadar(frameCache[String(currentFrame)]);
}

// ========== Playback ==========
function togglePlay() {
    if (video.paused) {
        video.play().then(() => {
            isPlaying = true;
            playBtn.textContent = '⏸';
            animLoop();
        }).catch(err => {
            console.error("Playback error:", err);
        });
    } else {
        video.pause();
        isPlaying = false;
        playBtn.textContent = '▶';
        if (animFrameId) cancelAnimationFrame(animFrameId);
    }
}

function animLoop() {
    if (!isPlaying) return;
    const frame = Math.floor(video.currentTime * videoInfo.fps);
    if (frame !== currentFrame) {
        currentFrame = frame;
        scrubber.value = frame;
        updateDisplay();
    }
    animFrameId = requestAnimationFrame(animLoop);
}

function seekToFrame(frame) {
    currentFrame = frame;
    video.currentTime = frame / videoInfo.fps;
    scrubber.value = frame;
    updateDisplay();
}

function skipFrames(n) {
    const newFrame = Math.max(
        parseInt(scrubber.min),
        Math.min(parseInt(scrubber.max), currentFrame + n)
    );
    seekToFrame(newFrame);
}

function setSpeed(speed) {
    video.playbackRate = parseFloat(speed);
}

// ========== Rendering ==========
function updateDisplay() {
    setupOverlay();
    const frameData = frameCache[String(currentFrame)];

    // Update frame counter
    frameCounter.textContent = `Frame: ${currentFrame}`;
    const timeSec = currentFrame / videoInfo.fps;
    currentTimeEl.textContent = formatTime(timeSec);
    radarTimeEl.textContent = formatTime(timeSec);

    // Draw overlay
    drawOverlay(frameData);

    // Draw radar
    drawRadar(frameData);
}

function drawOverlay(frameData) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!frameData) return;

    const showBbox = toggleBbox.checked;
    const showIds = toggleIds.checked;

    // Draw players
    for (const player of frameData.players) {
        const [x1, y1, x2, y2] = player.bbox;
        const isMe = (myPlayerTrackId !== null && player.tracker_id === myPlayerTrackId);
        const color = isMe ? '#ffd700' : (player.team_color || '#4fc3f7');

        if (showBbox) {
            overlayCtx.strokeStyle = color;
            overlayCtx.lineWidth = isMe ? 4 : 2;
            overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }

        if (isMe) {
            // Gold halo around tagged player
            overlayCtx.beginPath();
            overlayCtx.ellipse((x1 + x2) / 2, y2, (x2 - x1) / 1.5, 8, 0, 0, Math.PI * 2);
            overlayCtx.strokeStyle = '#ffd700';
            overlayCtx.lineWidth = 3;
            overlayCtx.shadowColor = '#ffd700';
            overlayCtx.shadowBlur = 10;
            overlayCtx.stroke();
            overlayCtx.shadowBlur = 0;
        }

        if (showIds && player.tracker_id >= 0) {
            overlayCtx.fillStyle = color;
            overlayCtx.font = 'bold 14px sans-serif';
            overlayCtx.fillText(isMe ? `★ ME #${player.tracker_id}` : `#${player.tracker_id}`, x1, y1 - 5);
        }

        // Foot dot
        overlayCtx.beginPath();
        overlayCtx.arc(player.foot_pixel[0], player.foot_pixel[1], isMe ? 6 : 4, 0, Math.PI * 2);
        overlayCtx.fillStyle = color;
        overlayCtx.fill();
    }

    // Draw ball
    if (frameData.ball) {
        const [bx, by] = frameData.ball.pixel;
        overlayCtx.beginPath();
        overlayCtx.arc(bx, by, 8, 0, Math.PI * 2);
        overlayCtx.fillStyle = frameData.ball.interpolated ? '#ffd54f88' : '#ffd54f';
        overlayCtx.fill();
        overlayCtx.strokeStyle = '#fff';
        overlayCtx.lineWidth = 2;
        overlayCtx.stroke();

        // Ball trail
        if (toggleBallTrail.checked) {
            ballTrail.push([bx, by]);
            if (ballTrail.length > BALL_TRAIL_LENGTH) ballTrail.shift();

            overlayCtx.beginPath();
            for (let i = 0; i < ballTrail.length; i++) {
                const alpha = i / ballTrail.length;
                overlayCtx.strokeStyle = `rgba(255, 213, 79, ${alpha * 0.6})`;
                overlayCtx.lineWidth = 2;
                if (i === 0) overlayCtx.moveTo(ballTrail[i][0], ballTrail[i][1]);
                else overlayCtx.lineTo(ballTrail[i][0], ballTrail[i][1]);
            }
            overlayCtx.stroke();
        }
    } else {
        if (!isPlaying) ballTrail = [];
    }
}

function drawRadar(frameData) {
    const ctx = radarCtx;
    const canvas = radarCanvas;
    const s = canvas._pitchScale;
    if (!s) return;

    // Clear and draw pitch
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawPitchGraphics(ctx, canvas.width, canvas.height, s.L, s.W);

    if (!frameData) return;

    // Scale helpers
    const px = (x) => (x / s.L) * canvas.width;
    const py = (y) => (y / W_val) * canvas.height;
    const W_val = s.W;

    // Phase 4: Draw Space Control (Voronoi) Layer
    if (toggleVoronoi && toggleVoronoi.checked && frameData.players.length >= 2) {
        drawVoronoiSpace(ctx, canvas.width, canvas.height, frameData.players, s.L, s.W);
    }

    // Phase 4: Draw Heatmap Layer
    if (toggleHeatmap && toggleHeatmap.checked && activeHeatmapGrid) {
        drawHeatmapLayer(ctx, canvas.width, canvas.height, activeHeatmapGrid);
    }

    // Draw players
    for (const player of frameData.players) {
        const [mx, my] = player.pitch_xy;
        const cx = px(mx);
        const cy = py(my);

        const isMe = (myPlayerTrackId !== null && player.tracker_id === myPlayerTrackId);
        const color = isMe ? '#ffd700' : (player.team_color || '#4fc3f7');

        ctx.beginPath();
        ctx.arc(cx, cy, isMe ? 8 : 6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = isMe ? 2.5 : 1.5;
        ctx.stroke();

        if (isMe) {
            // Gold radar halo
            ctx.beginPath();
            ctx.arc(cx, cy, 12, 0, Math.PI * 2);
            ctx.strokeStyle = '#ffd700';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        if (toggleIds.checked && player.tracker_id >= 0) {
            ctx.fillStyle = isMe ? '#ffd700' : '#fff';
            ctx.font = isMe ? 'bold 11px sans-serif' : 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(isMe ? `★${player.tracker_id}` : `${player.tracker_id}`, cx, cy - (isMe ? 12 : 9));
        }
    }

    // Draw ball trail on radar
    if (toggleBallTrail.checked) {
        // Collect ball positions within a window of currentFrame
        const trail = [];
        for (let f = Math.max(currentFrame - BALL_TRAIL_LENGTH, 0); f <= Math.min(currentFrame + 5, parseInt(scrubber.max)); f++) {
            const fd = frameCache[String(f)];
            if (fd && fd.ball && fd.ball.pitch_xy) {
                trail.push({ pt: fd.ball.pitch_xy, frame: f });
            }
        }
        if (trail.length > 1) {
            ctx.beginPath();
            ctx.moveTo(px(trail[0].pt[0]), py(trail[0].pt[1]));
            for (let i = 1; i < trail.length; i++) {
                ctx.lineTo(px(trail[i].pt[0]), py(trail[i].pt[1]));
            }
            ctx.strokeStyle = '#ffd54f';
            ctx.lineWidth = 3;
            ctx.shadowColor = '#ffd54f';
            ctx.shadowBlur = 6;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
    }

    // Phase 3: Draw active play vector arrows on 2D tactical radar
    const events = telemetryData.events || [];
    for (const ev of events) {
        if (currentFrame >= ev.start_frame && currentFrame <= ev.end_frame) {
            if (ev.start_xy && ev.end_xy) {
                const sx = px(ev.start_xy[0]);
                const sy = py(ev.start_xy[1]);
                const ex = px(ev.end_xy[0]);
                const ey = py(ev.end_xy[1]);

                // Draw connecting dashed vector line
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(ex, ey);
                ctx.strokeStyle = ev.color;
                ctx.lineWidth = 2.5;
                ctx.setLineDash([6, 4]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Arrow head at target
                const angle = Math.atan2(ey - sy, ex - sx);
                ctx.beginPath();
                ctx.moveTo(ex, ey);
                ctx.lineTo(ex - 10 * Math.cos(angle - Math.PI / 6), ey - 10 * Math.sin(angle - Math.PI / 6));
                ctx.lineTo(ex - 10 * Math.cos(angle + Math.PI / 6), ey - 10 * Math.sin(angle + Math.PI / 6));
                ctx.fillStyle = ev.color;
                ctx.fill();
            }
        }
    }

    // Draw ball
    if (frameData.ball && frameData.ball.pitch_xy) {
        const [bx, by] = frameData.ball.pitch_xy;
        const cx = px(bx);
        const cy = py(by);

        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ffd54f';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

function drawPitchGraphics(ctx, w, h, L, W) {
    const px = (x) => (x / L) * w;
    const py = (y) => (y / W) * h;

    // Background
    ctx.fillStyle = '#2d5a27';
    ctx.fillRect(0, 0, w, h);

    // Pitch grass stripes (subtle)
    ctx.fillStyle = '#305c29';
    const stripeW = w / 10;
    for (let i = 0; i < 10; i += 2) {
        ctx.fillRect(i * stripeW, 0, stripeW, h);
    }

    // Lines
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1.5;

    // Outline
    ctx.strokeRect(1, 1, w - 2, h - 2);

    // Halfway line
    ctx.beginPath();
    ctx.moveTo(px(L / 2), 0);
    ctx.lineTo(px(L / 2), h);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    const radius = Math.min(9.15, W * 0.2);
    ctx.arc(px(L / 2), py(W / 2), px(radius) - px(0), 0, Math.PI * 2);
    ctx.stroke();

    // Center spot
    ctx.beginPath();
    ctx.arc(px(L / 2), py(W / 2), 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fill();

    // Penalty boxes
    const pbD = Math.min(16.5, L * 0.18);
    const pbW = Math.min(40.32, W * 0.6);
    const penTop = (W - pbW) / 2;

    // Left
    ctx.strokeRect(px(0), py(penTop), px(pbD), py(pbW));
    // Right
    ctx.strokeRect(px(L - pbD), py(penTop), px(pbD), py(pbW));

    // Goal boxes
    const gbD = Math.min(5.5, L * 0.06);
    const gbW = Math.min(18.32, W * 0.3);
    const goalTop = (W - gbW) / 2;
    ctx.strokeRect(px(0), py(goalTop), px(gbD), py(gbW));
    ctx.strokeRect(px(L - gbD), py(goalTop), px(gbD), py(gbW));
}

// ========== Helpers ==========
function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

// ========== Phase 3: Events, Vectors & Timeline Pins ==========
let activeEventFilter = 'all';

function setupTimelinePins() {
    const track = document.getElementById('timeline-pins-track');
    if (!track) return;
    track.innerHTML = '';

    const events = telemetryData.events || [];
    const minF = parseInt(scrubber.min);
    const maxF = parseInt(scrubber.max);
    const span = maxF - minF;
    if (span <= 0) return;

    events.forEach(ev => {
        const pin = document.createElement('div');
        pin.className = 'timeline-pin';
        pin.style.backgroundColor = ev.color;
        const pct = ((ev.start_frame - minF) / span) * 100;
        pin.style.left = `${Math.max(0, Math.min(100, pct))}%`;
        pin.title = `${ev.label} @ ${ev.timestamp}s (Click to jump 2s prior)`;

        pin.onclick = () => {
            const jumpF = Math.max(minF, ev.start_frame - Math.floor(2.0 * videoInfo.fps));
            seekToFrame(jumpF);
        };
        track.appendChild(pin);
    });
}

function renderEventFeed(filterType) {
    activeEventFilter = filterType;
    const tbody = document.getElementById('event-feed-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const events = telemetryData.events || [];
    const filtered = filterType === 'all' ? events : events.filter(e => e.type === filterType);

    filtered.forEach(ev => {
        const row = document.createElement('tr');
        const actor = ev.passer_id ? `#${ev.passer_id}` : (ev.player_id ? `#${ev.player_id}` : '-');
        const target = ev.receiver_id ? `#${ev.receiver_id}` : '-';
        const dist = ev.distance_m ? `${ev.distance_m}m` : '-';

        row.innerHTML = `
            <td><strong>${ev.timestamp}s</strong></td>
            <td><span style="color:${ev.color};font-weight:600;">● ${ev.label}</span></td>
            <td>${actor}</td>
            <td>${target}</td>
            <td>${dist}</td>
            <td><button class="event-jump-btn" onclick="jumpToEvent(${ev.start_frame})">Seek ▶</button></td>
        `;
        tbody.appendChild(row);
    });
}

function jumpToEvent(frame) {
    const jumpF = Math.max(parseInt(scrubber.min), frame - Math.floor(2.0 * videoInfo.fps));
    seekToFrame(jumpF);
}

function filterEvents(type, btnElement) {
    document.querySelectorAll('.filter-chip').forEach(btn => btn.classList.remove('active'));
    if (btnElement) {
        btnElement.classList.add('active');
    } else if (window.event && window.event.target) {
        window.event.target.classList.add('active');
    }
    renderEventFeed(type);
}

// ========== Phase 4 & 5: Analytics, Space Control & Highlights Export ==========
let activeHeatmapGrid = null;

function setupAnalyticsPanel() {
    const sel = document.getElementById('analytics-player-select');
    if (!sel) return;
    sel.innerHTML = '<option value="all">Match Overview</option>';

    const analytics = telemetryData.analytics || {};
    const players = analytics.players || {};

    // Populate player dropdown
    Object.keys(players).sort((a,b) => parseInt(a) - parseInt(b)).forEach(tid => {
        const opt = document.createElement('option');
        opt.value = tid;
        opt.textContent = `Player #${tid} (Team ${players[tid].team_id === 0 ? 'Red' : 'Blue'})`;
        sel.appendChild(opt);
    });

    updatePlayerScorecard('all');
    fetchHeatmap('team_1');
}

function updatePlayerScorecard(val) {
    const analytics = telemetryData.analytics || {};
    const poss = analytics.possession || { team_0: 0, team_1: 100 };
    document.getElementById('metric-possession').textContent = `🔴 ${poss.team_0}% | 🔵 ${poss.team_1}%`;

    if (val === 'all') {
        let totalD = 0;
        let maxSpd = 0;
        let totalSp = 0;
        const pl = analytics.players || {};
        Object.values(pl).forEach(p => {
            totalD += p.distance_m;
            if (p.top_speed_kmh > maxSpd) maxSpd = p.top_speed_kmh;
            totalSp += p.sprints;
        });
        document.getElementById('metric-distance').textContent = `${(totalD / 1000).toFixed(2)} km total`;
        document.getElementById('metric-topspeed').textContent = `${maxSpd.toFixed(1)} km/h`;
        document.getElementById('metric-sprints').textContent = `${totalSp} sprints`;
        fetchHeatmap('team_1');
    } else {
        const p = (analytics.players || {})[val];
        if (p) {
            document.getElementById('metric-distance').textContent = `${p.distance_m} m (${p.distance_km} km)`;
            document.getElementById('metric-topspeed').textContent = `${p.top_speed_kmh} km/h`;
            document.getElementById('metric-sprints').textContent = `${p.sprints} sprints`;
            fetchHeatmap(val);
        }
    }
}

async function fetchHeatmap(target) {
    try {
        const resp = await fetch(`/api/heatmap/${target}`);
        const data = await resp.json();
        activeHeatmapGrid = data.grid;
        if (toggleHeatmap && toggleHeatmap.checked) updateDisplay();
    } catch (e) {
        console.error("Heatmap fetch error:", e);
    }
}

function drawHeatmapLayer(ctx, w, h, grid) {
    if (!grid || !grid.length) return;
    const rows = grid.length;
    const cols = grid[0].length;
    const cw = w / cols;
    const ch = h / rows;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = grid[r][c];
            if (val > 0.12) {
                ctx.fillStyle = `rgba(255, 87, 34, ${val * 0.45})`;
                ctx.fillRect(c * cw, r * ch, cw + 1, ch + 1);
            }
        }
    }
}

function drawVoronoiSpace(ctx, w, h, players, L, W) {
    // Discrete grid Voronoi space allocation for real-time performance
    const step = 8;
    for (let y = 0; y < h; y += step) {
        for (let x = 0; x < w; x += step) {
            const px_m = (x / w) * L;
            const py_m = (y / h) * W;

            let closestP = null;
            let minDist = Infinity;
            for (const p of players) {
                const d = Math.hypot(p.pitch_xy[0] - px_m, p.pitch_xy[1] - py_m);
                if (d < minDist) {
                    minDist = d;
                    closestP = p;
                }
            }

            if (closestP) {
                const color = closestP.team_id === 0 ? 'rgba(239, 83, 80, 0.18)' : 'rgba(66, 165, 245, 0.18)';
                ctx.fillStyle = color;
                ctx.fillRect(x, y, step, step);
            }
        }
    }
}

// ========== Phase 5: Export Modal Handlers ==========
function openExportModal() {
    document.getElementById('export-modal').style.display = 'flex';
    document.getElementById('export-feedback').textContent = '';
}

function closeExportModal() {
    document.getElementById('export-modal').style.display = 'none';
}

async function startExport() {
    const btn = document.getElementById('start-export-btn');
    const feedback = document.getElementById('export-feedback');
    const preset = document.getElementById('export-preset').value;

    btn.disabled = true;
    feedback.textContent = 'Rendering and concatenating video highlights via FFmpeg...';

    try {
        const resp = await fetch('/api/export/highlights', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset, player_id: myPlayerTrackId })
        });
        const data = await resp.json();
        if (data.success) {
            feedback.innerHTML = `
                ✅ Export complete! (${data.clips_count} plays sliced, ${data.size_mb} MB)<br>
                <a href="${data.download_url}" download class="btn btn-primary" style="margin-top:0.5rem;display:inline-block;">📥 Download Highlight Reel</a>
            `;
        } else {
            feedback.textContent = `❌ Export error: ${data.error || 'Failed'}`;
        }
    } catch (e) {
        feedback.textContent = `❌ Network error: ${e.message}`;
    } finally {
        btn.disabled = false;
    }
}

// Handle window resize
window.addEventListener('resize', () => {
    setupOverlay();
    setupRadar();
    updateDisplay();
});

// Start
init();

