/**
 * FooStatsBall - Tactical Dashboard
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

// Ball trail length (frames looked back in the telemetry cache)
const BALL_TRAIL_LENGTH = 60;

// Smooth rendering state
let TEAM_RGB = { 0: [239, 83, 80], 1: [66, 165, 245], [-1]: [189, 189, 189] };
let TEAM_NAMES = { 0: 'Team A', 1: 'Team B', [-1]: 'Other/GK' };
const hexToRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
// Teams/colours are learned from kits per video; pull them from telemetry.
function applyTeamMeta(teams) {
    if (!teams) return;
    [[0, teams.team_0], [1, teams.team_1], [-1, teams.goalkeeper]].forEach(([k, t]) => {
        if (!t) return;
        if (t.color) TEAM_RGB[k] = hexToRgb(t.color);
        if (t.name) TEAM_NAMES[k] = t.name;
    });
    const legend = document.getElementById('team-legend');
    if (legend) {
        const keys = teams.goalkeeper ? [0, 1, -1] : [0, 1];
        legend.innerHTML = keys.map(k => `<span><span class="dot" style="background:rgb(${TEAM_RGB[k].join(',')})"></span>${TEAM_NAMES[k]}</span>`).join('');
    }
}
const esc = v => String(v ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function kitSimilarity(tid) {
    const ks = telemetryData?.teams?.kit_similarity;
    return ks ? ks[tid] ?? ks[String(tid)] : undefined;
}
let pitchLayer = null;      // cached static pitch drawing (offscreen)
let heatmapLayer = null;    // cached heatmap bitmap (offscreen)
let voronoiLayer = null;    // low-res Voronoi buffer, upscaled with smoothing
let overlayFrame = 0;       // frame currently presented by the <video>

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
const ICON_PLAY = '<svg class="icon" viewBox="0 0 24 24"><path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none"/></svg>';
const ICON_PAUSE = '<svg class="icon" viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/></svg>';
function setPlayIcon(playing) {
    playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
    playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    document.getElementById('live-dot')?.classList.toggle('paused', !playing);
}
setPlayIcon(false);

// Toggles
const toggleBbox = document.getElementById('toggle-bbox');
const toggleIds = document.getElementById('toggle-ids');
const toggleBallTrail = document.getElementById('toggle-ball-trail');
const toggleVoronoi = document.getElementById('toggle-voronoi');
const toggleHeatmap = document.getElementById('toggle-heatmap');

// ========== Init ==========
async function init() {
    // Load video info
    const [infoResp, telResp] = await Promise.all([fetch('/api/video-info'), fetch('/api/telemetry')]);
    if (!infoResp.ok || !telResp.ok) {
        document.body.insertAdjacentHTML('afterbegin',
            '<div style="padding:1rem;background:#7f1d1d;color:#fff">Failed to load video info or telemetry for the active video.</div>');
        return;
    }
    videoInfo = await infoResp.json();
    telemetryData = await telResp.json();
    applyTeamMeta(telemetryData.teams);

    // Build frame cache
    frameCache = telemetryData.frames;

    // Set scrubber range
    const processing = telemetryData.processing || {};
    const frameKeys = Object.keys(frameCache).map(Number);
    const startFrame = processing.start_frame ?? Math.min(...frameKeys);
    const endFrame = processing.end_frame ?? Math.max(...frameKeys);
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
            setPlayIcon(true);
            animLoop();
        }
    });

    video.addEventListener('pause', () => {
        if (isPlaying) {
            isPlaying = false;
            setPlayIcon(false);
            if (animFrameId) cancelAnimationFrame(animFrameId);
        }
    });

    // Scrubber input
    scrubber.addEventListener('input', (e) => {
        const frame = parseInt(e.target.value);
        seekToFrame(frame);
    });

    // Redraw once a seek lands (covers native controls + scrubber)
    video.addEventListener('seeked', () => {
        if (!isPlaying) {
            currentFrame = Math.round(video.currentTime * videoInfo.fps);
            overlayFrame = currentFrame;
            updateDisplay();
        }
    });
    startVideoFrameSync();

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
    if (overlayCanvas.width !== vw) overlayCanvas.width = vw;
    if (overlayCanvas.height !== vh) overlayCanvas.height = vh;
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

    w = Math.floor(w);
    h = Math.floor(h);
    const dpr = window.devicePixelRatio || 1;
    radarCanvas.width = Math.round(w * dpr);
    radarCanvas.height = Math.round(h * dpr);
    radarCanvas.style.width = `${w}px`;
    radarCanvas.style.height = `${h}px`;
    radarCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    radarCanvas._pitchScale = { w, h, L: pitch.length, W: pitch.width };

    // Static pitch is drawn once into an offscreen canvas and blitted each frame
    pitchLayer = document.createElement('canvas');
    pitchLayer.width = radarCanvas.width;
    pitchLayer.height = radarCanvas.height;
    const pctx = pitchLayer.getContext('2d');
    pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawPitchGraphics(pctx, w, h, pitch.length, pitch.width);

    drawRadar(interpolatedFrame(currentFrame));
}

// ========== Playback ==========
function togglePlay() {
    if (video.paused) {
        video.play().then(() => {
            isPlaying = true;
            setPlayIcon(true);
            animLoop();
        }).catch(err => {
            console.error("Playback error:", err);
        });
    } else {
        video.pause();
        isPlaying = false;
        setPlayIcon(false);
        if (animFrameId) cancelAnimationFrame(animFrameId);
    }
}

// Radar runs at display refresh rate with sub-frame interpolation.
function animLoop() {
    if (!isPlaying) return;
    const t = video.currentTime * videoInfo.fps;
    const frame = Math.floor(t);
    if (frame !== currentFrame) {
        currentFrame = frame;
        scrubber.value = frame;
        updateHud();
    }
    drawRadar(interpolatedFrame(t));
    if (!hasVideoFrameCallback && overlayFrame !== frame) {
        overlayFrame = frame;
        drawOverlay(frameCache[String(frame)], frame);
    }
    animFrameId = requestAnimationFrame(animLoop);
}

// Overlay is locked to the exact frame the video presents (no drift vs boxes).
const hasVideoFrameCallback = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
function startVideoFrameSync() {
    if (!hasVideoFrameCallback) return;
    const onFrame = (_now, meta) => {
        const f = Math.round(meta.mediaTime * videoInfo.fps);
        if (f !== overlayFrame) {
            overlayFrame = f;
            drawOverlay(frameCache[String(f)], f);
        }
        video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
}

function seekToFrame(frame) {
    currentFrame = frame;
    overlayFrame = frame;
    // Seek to frame centre so the decoder lands on the intended frame
    video.currentTime = (frame + 0.5) / videoInfo.fps;
    scrubber.value = frame;
    updateDisplay();
}

// ----- Sub-frame interpolation -----
const lerp = (a, b, t) => a + (b - a) * t;
const lerpArr = (a, b, t) => a.map((v, i) => lerp(v, b[i], t));

function interpolatedFrame(t) {
    const f0 = Math.floor(t);
    const a = t - f0;
    const A = frameCache[String(f0)];
    const B = frameCache[String(f0 + 1)];
    if (!A) return B || null;
    if (!B || a <= 0) return A;

    const next = new Map(B.players.map(p => [p.tracker_id, p]));
    const players = A.players.map(p => {
        const q = next.get(p.tracker_id);
        if (!q) return p;
        return { ...p, pitch_xy: lerpArr(p.pitch_xy, q.pitch_xy, a) };
    });
    let ball = A.ball;
    if (A.ball && B.ball && A.ball.pitch_xy && B.ball.pitch_xy) {
        ball = { ...A.ball, pitch_xy: lerpArr(A.ball.pitch_xy, B.ball.pitch_xy, a) };
    }
    return { ...A, players, ball };
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
function updateHud() {
    frameCounter.textContent = `Frame ${currentFrame}`;
    const timeSec = currentFrame / videoInfo.fps;
    currentTimeEl.textContent = formatTime(timeSec);
    if (radarTimeEl) radarTimeEl.textContent = formatTime(timeSec);
}

function updateDisplay() {
    if (!videoInfo || !telemetryData) return;
    updateHud();
    const frameData = frameCache[String(currentFrame)];
    drawOverlay(frameData, currentFrame);
    drawRadar(isPlaying ? interpolatedFrame(video.currentTime * videoInfo.fps) : frameData);
}

function drawOverlay(frameData, frameIdx = currentFrame) {
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
            overlayCtx.stroke();
        }

        if (showIds && player.tracker_id >= 0) {
            overlayCtx.fillStyle = color;
            overlayCtx.font = 'bold 14px sans-serif';
            overlayCtx.fillText(isMe ? `ME #${player.tracker_id}` : `#${player.tracker_id}`, x1, y1 - 5);
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

        // Ball trail rebuilt from telemetry (stateless: correct after seeks), fading segments
        if (toggleBallTrail.checked) {
            const pts = [];
            for (let f = Math.max(frameIdx - BALL_TRAIL_LENGTH, 0); f <= frameIdx; f++) {
                const fd = frameCache[String(f)];
                if (fd && fd.ball && fd.ball.pixel) pts.push(fd.ball.pixel);
            }
            overlayCtx.lineWidth = 2;
            overlayCtx.lineCap = 'round';
            for (let i = 1; i < pts.length; i++) {
                overlayCtx.strokeStyle = `rgba(255, 213, 79, ${(i / pts.length) * 0.6})`;
                overlayCtx.beginPath();
                overlayCtx.moveTo(pts[i - 1][0], pts[i - 1][1]);
                overlayCtx.lineTo(pts[i][0], pts[i][1]);
                overlayCtx.stroke();
            }
        }
    }
}

function drawRadar(frameData) {
    const ctx = radarCtx;
    const canvas = radarCanvas;
    const s = canvas._pitchScale;
    if (!s) return;

    const cw = s.w, chh = s.h;  // logical (CSS px) size; ctx is DPR-scaled

    ctx.clearRect(0, 0, cw, chh);
    if (pitchLayer) ctx.drawImage(pitchLayer, 0, 0, cw, chh);
    else drawPitchGraphics(ctx, cw, chh, s.L, s.W);

    if (!frameData) return;

    const px = (x) => (x / s.L) * cw;
    const py = (y) => (y / s.W) * chh;
    const frameNow = currentFrame;

    if (toggleVoronoi && toggleVoronoi.checked && frameData.players.length >= 2) {
        drawVoronoiSpace(ctx, cw, chh, frameData.players, s.L, s.W);
    }

    if (toggleHeatmap && toggleHeatmap.checked && heatmapLayer) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(heatmapLayer, 0, 0, cw, chh);
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
            ctx.fillText(isMe ? `${player.tracker_id}` : `${player.tracker_id}`, cx, cy - (isMe ? 12 : 9));
        }
    }

    // Draw ball trail on radar
    if (toggleBallTrail.checked) {
        // Collect ball positions within a window of currentFrame
        const trail = [];
        for (let f = Math.max(frameNow - BALL_TRAIL_LENGTH, 0); f <= frameNow; f++) {
            const fd = frameCache[String(f)];
            if (fd && fd.ball && fd.ball.pitch_xy) {
                trail.push({ pt: fd.ball.pitch_xy, frame: f });
            }
        }
        if (frameData.ball && frameData.ball.pitch_xy) trail.push({ pt: frameData.ball.pitch_xy });
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        for (let i = 1; i < trail.length; i++) {
            ctx.strokeStyle = `rgba(255, 213, 79, ${0.15 + 0.85 * (i / trail.length)})`;
            ctx.beginPath();
            ctx.moveTo(px(trail[i - 1].pt[0]), py(trail[i - 1].pt[1]));
            ctx.lineTo(px(trail[i].pt[0]), py(trail[i].pt[1]));
            ctx.stroke();
        }
    }

    // Phase 3: Draw active play vector arrows on 2D tactical radar
    const events = telemetryData.events || [];
    for (const ev of events) {
        if (frameNow >= ev.start_frame && frameNow <= ev.end_frame) {
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
        const who = ev.passer_id ? ` #${ev.passer_id}` : (ev.player_id ? ` #${ev.player_id}` : '');
        pin.title = `${ev.label}${who}${ev.receiver_id ? ' to #' + ev.receiver_id : ''} · ${ev.timestamp}s`;

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

    document.querySelectorAll('.filter-chip').forEach(chip => {
        const f = chip.dataset.f;
        const n = f === 'all' ? events.length : events.filter(e => e.type === f).length;
        let c = chip.querySelector('.count');
        if (!c) { c = document.createElement('span'); c.className = 'count'; chip.appendChild(c); }
        c.textContent = n;
    });
    if (!filtered.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty">No events of this type</td></tr>';
        return;
    }
    filtered.forEach(ev => {
        const row = document.createElement('tr');
        const actor = ev.passer_id ? `#${ev.passer_id}` : (ev.player_id ? `#${ev.player_id}` : '-');
        const target = ev.receiver_id ? `#${ev.receiver_id}` : '-';
        const dist = ev.distance_m ? `${ev.distance_m}m` : '-';

        row.innerHTML = `
            <td class="num">${ev.timestamp}s</td>
            <td><span class="event-type"><span class="dot" style="background:${esc(ev.color)}"></span>${esc(ev.label)}</span></td>
            <td>${actor}</td>
            <td>${target}</td>
            <td>${dist}</td>
            <td><button class="btn btn-ghost btn-sm" onclick="jumpToEvent(${ev.start_frame})" aria-label="Seek to event">Seek</button></td>
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
    }
    renderEventFeed(type);
}

// ========== Phase 4 & 5: Analytics, Space Control & Highlights Export ==========
let activeHeatmapGrid = null;

function setupAnalyticsPanel() {
    const sel = document.getElementById('analytics-player-select');
    if (!sel) return;
    sel.innerHTML = '<option value="all">Match overview</option>';

    const analytics = telemetryData.analytics || {};
    const players = analytics.players || {};

    // Populate player dropdown
    Object.keys(players).sort((a,b) => parseInt(a) - parseInt(b)).forEach(tid => {
        const opt = document.createElement('option');
        opt.value = tid;
        const tname = TEAM_NAMES[players[tid].team_id] ?? 'Unassigned';
        const sim = kitSimilarity(tid);
        opt.textContent = `#${tid} · ${tname}${sim != null ? ` · ${Math.round(sim * 100)}% kit match` : ''}`;
        sel.appendChild(opt);
    });

    updatePlayerScorecard('all');
    fetchHeatmap('team_1');
}

function updatePlayerScorecard(val) {
    const analytics = telemetryData.analytics || {};
    const poss = analytics.possession;
    const possEl = document.getElementById('metric-possession');
    if (poss && poss.team_0 != null) {
        const c0 = `rgb(${TEAM_RGB[0].join(',')})`, c1 = `rgb(${TEAM_RGB[1].join(',')})`;
        possEl.innerHTML = `<div class="poss-labels"><span class="num">${esc(TEAM_NAMES[0])} ${poss.team_0}%</span><span class="num">${poss.team_1}% ${esc(TEAM_NAMES[1])}</span></div>
            <div class="poss-bar"><div style="width:${poss.team_0}%;background:${c0}"></div><div style="width:${poss.team_1}%;background:${c1}"></div></div>`;
    } else {
        possEl.innerHTML = `<span class="metric-val">n/a</span> <small class="help">ball seen ${poss?.ball_coverage_pct ?? 0}%</small>`;
    }
    const kitEl = document.getElementById('metric-kit');
    const kitTitle = document.getElementById('metric-kit-title');

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
        const sims = Object.values(telemetryData?.teams?.kit_similarity || {}).filter(v => v != null);
        kitTitle.textContent = 'Avg kit match';
        kitEl.textContent = sims.length ? `${Math.round(100 * sims.reduce((a, b) => a + b, 0) / sims.length)}%` : '-';
        fetchHeatmap('team_1');
    } else {
        const p = (analytics.players || {})[val];
        if (p) {
            document.getElementById('metric-distance').textContent = `${p.distance_m} m (${p.distance_km} km)`;
            document.getElementById('metric-topspeed').textContent = `${p.top_speed_kmh} km/h`;
            document.getElementById('metric-sprints').textContent = `${p.sprints} sprints`;
            const sim = kitSimilarity(val);
            kitTitle.textContent = 'Kit match';
            kitEl.textContent = sim != null ? `${Math.round(sim * 100)}%` : '-';
            fetchHeatmap(val);
        }
    }
}

async function fetchHeatmap(target) {
    try {
        const resp = await fetch(`/api/heatmap/${target}`);
        const data = await resp.json();
        activeHeatmapGrid = data.grid;
        heatmapLayer = buildHeatmapLayer(activeHeatmapGrid);
        if (toggleHeatmap && toggleHeatmap.checked) updateDisplay();
    } catch (e) {
        console.error("Heatmap fetch error:", e);
    }
}

// One pixel per grid cell; drawImage upscales with bilinear smoothing -> soft heatmap.
function buildHeatmapLayer(grid) {
    if (!grid || !grid.length) return null;
    const rows = grid.length, cols = grid[0].length;
    const c = document.createElement('canvas');
    c.width = cols;
    c.height = rows;
    const img = c.getContext('2d').createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
        for (let k = 0; k < cols; k++) {
            const v = grid[r][k];
            const i = (r * cols + k) * 4;
            img.data[i] = 255; img.data[i + 1] = 87; img.data[i + 2] = 34;
            img.data[i + 3] = v > 0.12 ? Math.round(v * 0.45 * 255) : 0;
        }
    }
    c.getContext('2d').putImageData(img, 0, 0);
    return c;
}

// Low-res nearest-player grid, upscaled with smoothing (soft region borders, ~20x cheaper).
function drawVoronoiSpace(ctx, w, h, players, L, W) {
    const gw = 96;
    const gh = Math.max(8, Math.round(gw * (W / L)));
    if (!voronoiLayer || voronoiLayer.width !== gw || voronoiLayer.height !== gh) {
        voronoiLayer = document.createElement('canvas');
        voronoiLayer.width = gw;
        voronoiLayer.height = gh;
    }
    const vctx = voronoiLayer.getContext('2d');
    const img = vctx.createImageData(gw, gh);
    const pts = players.map(p => [p.pitch_xy[0], p.pitch_xy[1], TEAM_RGB[p.team_id] || [200, 200, 200]]);
    for (let y = 0; y < gh; y++) {
        const my = ((y + 0.5) / gh) * W;
        for (let x = 0; x < gw; x++) {
            const mx = ((x + 0.5) / gw) * L;
            let best = null, bestD = Infinity;
            for (const p of pts) {
                const dx = p[0] - mx, dy = p[1] - my;
                const d = dx * dx + dy * dy;
                if (d < bestD) { bestD = d; best = p; }
            }
            const i = (y * gw + x) * 4;
            img.data[i] = best[2][0]; img.data[i + 1] = best[2][1]; img.data[i + 2] = best[2][2];
            img.data[i + 3] = 46;
        }
    }
    vctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(voronoiLayer, 0, 0, w, h);
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
                <span>Export complete: ${data.clips_count} clips, ${data.size_mb} MB</span>
                <a href="${data.download_url}" download class="btn btn-primary btn-sm">Download reel</a>
            `;
        } else {
            feedback.textContent = `Export error: ${data.error || 'Failed'}`;
        }
    } catch (e) {
        feedback.textContent = `Network error: ${e.message}`;
    } finally {
        btn.disabled = false;
    }
}

// Handle window resize
let resizePending = false;
window.addEventListener('resize', () => {
    if (resizePending || !telemetryData) return;
    resizePending = true;
    requestAnimationFrame(() => {
        resizePending = false;
        setupOverlay();
        setupRadar();
        updateDisplay();
    });
});

// Start
init();

