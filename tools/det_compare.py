"""Compare person-detector recall proxies on sampled frames (count of person boxes by conf)."""
import sys, time, cv2, numpy as np
from ultralytics import YOLO
video, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cap = cv2.VideoCapture(video); frames = []
for f in np.linspace(start, end, 15).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read(); frames.append(im)
for w in ["yolo11n.pt", "yolo11m.pt", "yolo11x.pt"]:
    m = YOLO(w)
    for sz in (1280, 1920):
        t = time.time(); c25 = []; c10 = []
        for im in frames:
            r = m.predict(im, imgsz=sz, conf=0.1, classes=[0], device="mps", verbose=False)[0]
            cf = r.boxes.conf.cpu().numpy(); c25.append((cf >= .25).sum()); c10.append((cf >= .1).sum())
        print(f"{w:11s} {sz}: >=.25 {np.mean(c25):5.1f}  >=.10 {np.mean(c10):5.1f}  {1000*(time.time()-t)/len(frames):4.0f} ms/f", flush=True)
