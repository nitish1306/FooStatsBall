"""Publish a processed match as a static, shareable dashboard on Cloudflare R2.

    python tools/publish_match.py "Fixedcam football game.mp4" --folder Sample

Steps: HLS encode (skipped if export/<folder>/video exists) -> telemetry split -> page build -> upload into <folder>/.
Creds come from .env (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, CLOUDFLARE_API_TOKEN, R2_BUCKET, R2_PUBLIC_URL).
"""
import argparse
import gzip
import hashlib
import json
import os
from urllib.parse import quote
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.analytics import generate_heatmap_grid  # noqa: E402

CHUNK_SEC = 60
SEGMENT_SEC = 4
DROP_PLAYER_KEYS = {"confidence", "kit_similarity"}
CONTENT_TYPES = {
    ".json": "application/json", ".m3u8": "application/vnd.apple.mpegurl", ".m4s": "video/iso.segment",
    ".mp4": "video/mp4", ".html": "text/html; charset=utf-8", ".js": "text/javascript",
    ".css": "text/css", ".svg": "image/svg+xml",
}


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def s3_client():
    import boto3
    return boto3.client(
        "s3", region_name="auto",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=hashlib.sha256(os.environ["CLOUDFLARE_API_TOKEN"].encode()).hexdigest(),
    )


def encode_hls(video: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", "scale=-2:'min(720,ih)'", "-c:v", "h264_videotoolbox", "-b:v", "900k",
        "-maxrate", "1200k", "-bufsize", "2400k",
        "-force_key_frames", f"expr:gte(t,n_forced*{SEGMENT_SEC})",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1",
        "-f", "hls", "-hls_time", str(SEGMENT_SEC), "-hls_playlist_type", "vod",
        "-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", str(out / "seg_%05d.m4s"), str(out / "index.m3u8"),
    ], check=True)


def _round(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, list):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v


def _compact_player(p):
    out = {k: _round(v) for k, v in p.items() if k not in DROP_PLAYER_KEYS}
    for flag in ("redetected", "interpolated"):
        if not out.get(flag):
            out.pop(flag, None)
    return out


def split_telemetry(tel: dict, out: Path):
    """meta.json + 60 s frame chunks. Chunk index = frame // chunk_frames."""
    fps = tel["video_info"]["fps"]
    chunk_frames = int(round(CHUNK_SEC * fps))
    chunks: dict[int, dict] = {}
    for key, fr in tel["frames"].items():
        f = int(key)
        chunks.setdefault(f // chunk_frames, {})[key] = {
            "players": [_compact_player(p) for p in fr.get("players", [])],
            "ball": _round(fr.get("ball")),
        }
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for idx, frames in chunks.items():
        (out / "frames" / f"c{idx:04d}.json").write_text(json.dumps(frames, separators=(",", ":")))

    heatmaps = dict(tel.get("heatmaps", {}))
    for tid in tel.get("processing", {}).get("track_id_list", []):
        grid = generate_heatmap_grid(tel, track_id=tid)
        if grid:
            heatmaps[str(tid)] = [[round(v, 2) for v in row] for row in grid]

    meta = {k: v for k, v in tel.items() if k not in ("frames", "heatmaps")}
    meta["heatmaps"] = heatmaps
    meta["static"] = {"chunk_frames": chunk_frames, "chunks": sorted(chunks), "segment_sec": SEGMENT_SEC}
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return meta


def build_page(out: Path):
    """Self-contained dashboard: index.html + assets/ beside data/ and video/."""
    (out / "assets").mkdir(parents=True, exist_ok=True)
    html = (ROOT / "templates" / "dashboard.html").read_text()
    html = html.replace('href="/static/css/', 'href="./assets/').replace('src="/static/js/', 'src="./assets/')
    html = html.replace('"/static/favicon.svg"', '"./assets/favicon.svg"')
    html = html.replace('href="/" class="brand"', 'href="./index.html" class="brand"')
    html = html.replace('src="/video" ', "")
    html = html.replace('<script src="./assets/dashboard.js"></script>',
                        '<script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js"></script>\n'
                        '    <script>window.FOOSTATS_STATIC = true;</script>\n'
                        '    <script src="./assets/dashboard.js"></script>')
    (out / "index.html").write_text(html)
    for rel in ("js/dashboard.js", "css/style.css", "favicon.svg"):
        (out / "assets" / Path(rel).name).write_bytes((ROOT / "static" / rel).read_bytes())


def upload_dir(s3, bucket: str, local: Path, prefix: str):
    files = [p for p in local.rglob("*") if p.is_file()]
    for i, p in enumerate(files):
        key = f"{prefix}{p.relative_to(local).as_posix()}"
        body = p.read_bytes()
        extra = {"ContentType": CONTENT_TYPES.get(p.suffix, "application/octet-stream")}
        if p.suffix in (".json", ".html", ".js", ".css", ".m3u8"):
            body = gzip.compress(body)
            extra["ContentEncoding"] = "gzip"
            extra["CacheControl"] = "no-cache"
        else:
            extra["CacheControl"] = "public, max-age=31536000, immutable"
        s3.put_object(Bucket=bucket, Key=key, Body=body, **extra)
        if i % 50 == 0:
            print(f"  {prefix}: {i + 1}/{len(files)}", flush=True)


def main():
    """Bucket layout, one folder per match (all inside <folder>/):
    <raw video>.mp4, index.html, assets/, data/meta.json, data/frames/cNNNN.json, video/index.m3u8 + segments
    Share: <R2_PUBLIC_URL>/<folder>/index.html
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="local source video (relative to repo root)")
    ap.add_argument("--folder", required=True, help="bucket folder for this match, e.g. Sample")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    load_env()
    bucket = os.environ["R2_BUCKET"]
    video = ROOT / a.video
    folder = a.folder.strip("/")
    out = ROOT / "export" / folder
    if not (out / "video" / "index.m3u8").exists():
        print("Encoding HLS ...", flush=True)
        encode_hls(video, out / "video")
    tel = json.loads((ROOT / "output" / video.stem / "telemetry.json").read_text())
    split_telemetry(tel, out / "data")
    build_page(out)
    if not a.no_upload:
        s3 = s3_client()
        raw_key = f"{folder}/{video.name}"
        try:
            s3.head_object(Bucket=bucket, Key=raw_key)
        except s3.exceptions.ClientError:
            print(f"Uploading raw {video.name} ...", flush=True)
            s3.upload_file(str(video), bucket, raw_key, ExtraArgs={"ContentType": "video/mp4"})
        upload_dir(s3, bucket, out, f"{folder}/")
    print(f"{os.environ['R2_PUBLIC_URL']}/{quote(folder)}/index.html")


if __name__ == "__main__":
    main()
