"""Player visibility metrics from telemetry: per-frame count, dropouts (track gone then back)."""
import json, sys, numpy as np

def metrics(path):
    t = json.load(open(path)); fr = t["frames"]; ks = sorted(fr, key=int)
    counts = np.array([len(fr[k]["players"]) for k in ks])
    pres = {}
    for i, k in enumerate(ks):
        for p in fr[k]["players"]:
            pres.setdefault(p["tracker_id"], []).append(i)
    interp = sum(1 for k in ks for p in fr[k]["players"] if p.get("interpolated"))
    gaps = [b - a - 1 for v in pres.values() for a, b in zip(v, v[1:]) if b - a > 1]
    lens = sorted(len(v) for v in pres.values())
    return {
        "frames": len(ks), "tracks": len(pres),
        "players/frame p10/p50/p90": [int(np.percentile(counts, q)) for q in (10, 50, 90)],
        "frames<median-2": int((counts < np.median(counts) - 2).sum()),
        "missing gaps (count, frames)": [len(gaps), int(sum(gaps))],
        "interp detections": interp,
        "track len p10/p50": [lens[len(lens) // 10], lens[len(lens) // 2]],
    }

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p.split("/")[-2], metrics(p))
